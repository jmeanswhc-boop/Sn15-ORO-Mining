from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
# retry pauses in seconds, capped in total so the 300 s episode keeps room
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
# budget of store calls for the process check ahead of a finish
GATE_ROUNDS = 3
# extra result pages read by code when the pick is an extreme
AUTO_PAGES = 6
# seconds after which the walk stops and the episode closes; the sandbox
# kills the agent at 300 s
CLOCK_WRAP = 230
# cheaper same-kind rows opened before a cheapest checkout
CHEAPER_OPENS = 3

# the LLM client tries once; consult() owns the resend policy
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


WALKER_PROMPT = """\
Purpose: You are Kyo, shopping for one shopper in an online store; its \
tools are your only means of acting. Scoring is all or nothing: \
the episode scores 1 only when it finishes exactly right, else 0. A right \
finish is a purchase holding every requested item at its requested \
quantity, each satisfying every requirement in force when you check out, \
all within the budget; or, only when the store offers nothing \
that satisfies everything, an ending through the no-purchase tool done the \
way the request says.

What you get: tools, the request, Kyo's ledger, tool results.

How to decide:
1. Each tool call spends one step and steps are capped, while thinking is \
free. This store has its own tool names, argument names and result shapes, \
so read every tool's description and schema and pass exactly the argument \
names and permitted values it defines. \
Expect limited searches: a category filter, a keyword mode that may look \
only at the query's opening words, pages of a few rows, a token or an \
offset for the next page. When the answer hinges on seeing every row, compare \
the reported total with the rows seen and keep paging, passing \
the token or offset exactly as the store returns it. Keyword queries do \
best short and led by the value (the wanted value, then the product noun). \
A filtered search returning nothing usually has a wrong filter value \
or the wrong filter parameter: try again with the label spelled exactly as \
result rows print it, or with a different filter parameter.
2. A search row is only a summary. Stores divide a listing's record in \
different ways: one view may give its attributes (properties), another just \
its offer and description, and a side-by-side view may hold either. Once \
Kyo knows which tool gives attributes, the ledger names it. Judge every \
requirement on a property from those attributes, so read them for every \
listing you might buy; one call covering several listings saves steps when \
it shows what you need. Do not reopen a listing when nothing has changed \
since you read it.
3. Kyo's interviewer handles questions to the shopper, so you never message \
the shopper yourself. Its answers show up in the conversation and go into \
the ledger. \
The request may also call for steps along the way: an approval, a \
message, something to check or read first; the ledger lists them. Those due \
right before the order or the ending are done by Kyo itself when you call \
the checkout or the ending tool, so never wait for them and never end \
without a purchase on their account; do any other such step with the \
store's tools (product questions go through Kyo's interviewer).
4. The listing's category label settles what kind of product it is: count \
only the label that matches the requested item as a whole, never a \
neighbouring label. \
A property requirement is met when the listing's attribute for \
that property gives the value. How the key is spelled is irrelevant \
(spaces, underscores, camel case, prefixes), and one element of a comma or \
slash list suffices. Values are compared by their words: differences in \
case, singular versus plural, and plain synonyms are fine, and a \
value that includes the words of the required value counts ("deep teal" \
meets teal, "extra roomy fit" meets roomy). A value in other words does not \
count: an abbreviation or code is not the word behind it, and "teal green" \
is not "sea green". Convert units before comparing numbers; a limit \
includes its edge value, and a range must sit wholly within a limit. \
An attribute whose key names the requirement's property, by that name or \
a plain synonym, is decisive: if it gives something else, the listing \
fails whatever other keys or the title say. A key for another \
property, even a related one (a handle colour is not the body's colour), \
settles nothing about the requirement. After opened listings Kyo appends \
[evidence] lines holding the keys whose names match a requirement. \
If the listing has no attribute for that property, the wanted value \
counts when it appears under another attribute (say, a list of styles or \
features with it) or in the title or description. A number of a \
different kind does not count (a weight is not a capacity), and neither \
does a bare figure whose text does not say what it measures. \
If the category label itself states the value a requirement asks for (a \
season or an audience in the label), the category satisfies that \
requirement.
5. Some requirements name another listing rather than a value ("the same X \
as the listing sold by <shop> with <property>", "within N of the X on the \
page of ..."). Settle these first: locate that listing (search by its \
category, brand, shop or words) and open its full page with the single-item \
or side-by-side tool, as a search row will not do; when that record directs the \
reader to a further listing for the property, locate and open that one likewise, \
and the final page supplies the value. The order counts only if every page \
in that chain was opened. If the wanted value is anything but that \
listing's value, the rule excludes only listings with that exact value, and \
it is never alone a reason to end without a purchase. \
If a listing you are about to buy refers you to another page for more \
about it, check that page before buying. \
When one requested item has to fit another (an accessory for a device), \
pick the pair together from the listings' own words (a list of what it \
fits, a model or series code on both pages). If no accessory \
meeting its own requirements fits the device you chose, move to another \
device that meets its requirements and that such an accessory fits; do not \
end without a purchase before trying that switch. \
Honour exclusions and closed lists. Wishes ("ideally", "would be nice") \
only break ties.
6. Never buy an item the store marks as sold out. The total, price times \
quantity summed over every line, must not exceed the budget; exactly on \
it is fine. \
For a cheapest request (lowest total, smallest price), the order must \
hold the lowest-priced listing among ALL fitting listings. Gather the \
whole fitting set by paging through it (rows often come in no price order, \
and Kyo reads the next pages of a short result itself), test candidates \
from the cheapest up, and take the first that fits fully. If a \
search accepts a maximum price, use it to list the category under your \
current best pick's price, so only cheaper candidates remain to test. \
For the most or the least of some measure, compare all fitting listings \
on that measure and take the extreme one; a listing that does not state the \
measure cannot win. \
Otherwise any listing that meets everything is correct; do \
not spend steps looking for a better one. \
Cart each item as soon as you have opened a listing that meets all its \
requirements (under a cheapest, most or least pick, once the fitting set \
has been compared), then go on to the next. Stores often sell out a line \
soon after it is carted, and carting early leaves steps to replace it. \
A request for several different items needs one cart line per item, \
each at the quantity requested.
7. The shopper or the store may change things while you work: a new \
budget, a requirement added or removed, a price change, a listing selling \
out. These show up inside tool results, often when you try to check out \
(a refusal); a note counts no matter which tool's result carries it, and \
the ledger lists the requirements in force. A refusal does not end the \
episode: apply every change, repair the cart and check out again. \
Shortly before checking out, read each chosen listing's record \
again, as price and stock can change without notice.
8. Make the cart hold exactly the chosen lines (remove anything else), \
view the cart once, then check out in a call by itself. \
End without a purchase only once a real search has found no listing \
that meets everything. Being low on steps proves nothing of that kind: \
then check out the best cart you have. Word that ending as the request \
says, and if a remark is needed, make it brief and name no listing.

Answer format: Every reply must call at least one tool. Keep text \
brief: what you know, then the next move."""


BRIEF_PROMPT = """\
Purpose: Get a shopping task ready for an agent by turning it into ONE JSON \
object.

What you get: the store's tools, the shopper's request, maybe a store note.

How to decide:
1. "items": one entry for each distinct product the shopper wants bought, \
shaped {"name": the product kind in the request's own words, "shelf": the \
store's category label for it, copied exactly, when some tool lists labels, \
otherwise "", "quantity": how many units of that one listing (an integer, 1 \
when the request gives no number; a pair or a set sold as one listing \
counts as 1), "musts": the firm requirements on that product, each written \
"property: constraint" holding the exact value, limit, exclusion or closed \
list}.
2. Call each must's property by the request's own word for it ("meant for a \
lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives \
"blade length: 12 inch"), not by a neighbouring property. A must that reads \
its value from another listing is ONE must carrying that listing's whole \
description ("rim colour: unlike the rim colour of the kettle Ovarno sells \
with a glass lid"): the words describing that other listing (its shop, its \
kind, the "with <property> <value> and <property> <value>" after it) remain \
inside that single must and never turn into musts of this product. \
Price, budget, stock and quantity never go into musts. The story around \
the request (the person who will use the product, where it will stand, what \
went wrong before, the occasion behind it) is context rather than a must, \
unless it outright states a property's value or limit. Words that form part \
of what the product is called (a season, an audience or a use named along \
with the product) belong to "name" and never become musts. \
A product the request names in a single phrase joined by "and" (one "I \
want ..." with one quantity, often the name of one store department) is ONE \
item; split it into separate items only when each product comes with its \
own quantity or its own requirements.
3. "budget": the top total the shopper accepts, a number in the request's \
own unit, or null. "budget_unit": the unit in words (for example "cents"), \
or null. "budget_scope": "total" for a limit on the whole order, "each" for \
a per-item limit, null without a limit. \
"in_stock_only": true unless the request clearly allows unavailable \
items.
4. "pick": "cheapest" if the request wants the lowest price or the smallest \
total, "most" or "least" if it wants the extreme of some measure, otherwise \
"any"; remarks that money is tight do not make a pick rule. "pick_measure": \
the measure pushed up or down for "most"/"least", otherwise "". \
"messages": how many messages the shopper lets the agent send (integer), \
or null when unstated. "steps": how many steps, moves or tool calls the \
request permits (integer), or null.
5. "lookups": the musts (copied exactly) whose value first has to be read \
from another listing the request points at, otherwise []. \
"rules": each instruction the request gives on how to work or how to \
finish, copied word for word from the request. "nothing_fits": the \
request's own sentences on what to do if no listing meets everything, \
copied in full with every condition they carry, or "" when it says \
nothing about that. "wishes": what the shopper would merely \
like. "ask_first": what the request explicitly says to check with the \
shopper before choosing (empty if nothing).
6. "process": the steps the request demands of the agent apart from \
finding and buying the right items (getting the shopper's approval, telling \
or showing the shopper something, checking something with them, reading \
something first, using a particular tool), each {"step": what has to be \
done, in 15 words or fewer, "when": "choose" (before the products are \
chosen), "order" (right before checking out) or "end" (just before ending \
without a purchase), "messages": how many messages to the shopper it takes \
(0 if none)}; [] if the request asks for no such step. These are not steps: \
requirements on the items, the budget, stock, the limits on messages and \
steps, and how to end when nothing fits.
7. "quiet_decline": true if the request forbids suggesting or naming \
alternatives when nothing fits, false if it wants them, null if silent. \
"roles": an object giving the exact tool for each job, with null where \
no tool does it: "search", "open" (shows one listing's attributes or \
properties; if there are two single-listing tools, the one whose \
description promises attributes, properties or specifications), "offer" \
(shows one listing's current offer, such as its price and stock, when that \
tool differs from "open"), "compare" (several listings in full), "ask" \
(sends the shopper a message), "cart" (changes the cart), "cart_view" \
(shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
8. "buy_args": the fixed arguments that make the "buy" tool check out (for \
a tool that takes a mode argument, that mode), otherwise {}. "refuse_args": \
the fixed arguments that make the "refuse" tool end without a purchase, \
otherwise {}.

Answer format: ONE JSON object with the keys above, written compactly (no \
indentation), and nothing besides it.
"""


PLAN_PROMPT = """\
Purpose: Decide which questions a shopping agent puts to a shopper. Every \
shopper holds a few firm preferences left out of the request, each tied to \
one property of one requested product. A question brings a preference to \
light only when it names exactly that property, and the reply covers only \
the first property named, so each question uses up one property.

What you get: the request; the items with what the request already fixes \
for each; and, per item category, a numbered list of attribute keys found \
on opened listings, with how many of them carry each key and sample values. \
A preference left out may also concern a property that no listed key names.

How to decide (properties in this order):
1. First, anything the request itself tells you to check with the shopper \
(under ASK FIRST if present), each named by the property it concerns.
2. Next, hinted properties. The request's story often hints at an unwritten \
preference without its value: a sentence about something the shopper cannot \
judge or leaves to you, a person whose taste decides, the product's looks \
or the setting it belongs in, the person who will use or wear it, or its \
contents. For each such sentence, name the one property it hints at and the \
item it concerns, in the story's order. Use the property's plain name: a \
hint about the wearer points at gender, one about looks at colour or finish \
(whichever the request leaves open).
3. Then the listed keys, by rank: prefer properties that buyers of such a \
product compare first (often its size, capacity or dimensions, its type, \
its material or its colour, whichever such a product is chosen by) and keys \
that most of the item's listings carry; pass over keys found on only one or \
two.
4. Leave out any property the request already fixes for that item, any \
alternative name of a property already chosen, any key that lists purchase \
variants (a mix of sizes, colours and pack counts), identifiers, warranty \
and other logistics.
5. With several items, put every item's hinted properties first, then \
rotate between the items.
6. "label" gives the property's plain name: for a listed key, every word of \
it spelled plainly ("lid colour" for lidColour, "rim finish" for \
rim_finish; never cut to "colour" or "finish"); just the name, without \
example values, brackets or a second property.

Answer format: Choose at most COUNT, best first. Reply with ONE JSON object \
and nothing else: {"ask": [{"item": item number, "label": "...", "hint": \
"the story's words it comes from, 8 at most, or an empty string"}]}"""


HEAR_PROMPT = """\
Purpose: Turn a shopper's replies into needs.

What you get: items, questions, replies.

How to decide:
1. "needs": every firm preference a reply expresses, with its precise value \
or limit and unit ("at least 30 cm", "oval", "not wicker"). Label the need \
with the property the reply itself mentions, maybe not the one asked about; \
replies on the same property merge into one need. Reproduce the value \
verbatim as the reply puts it, nouns and units included ("at least 3 \
shelves", not "at least 3").
2. A reply can supply the value by referring to another listing (a value \
like, unlike or near the one on some shop's product, where a few of that \
product's properties identify it). Record that as ONE need on the property \
the reply mentions, with "points": true, whose value is the reply's own \
wording copied verbatim from the comparison words up to the end of that \
description, together with anything it says about a listing that page may \
name; do not shorten or rephrase it. The other product's describing \
properties are not needs of this item.
3. "free": for each question, the property the shopper said they do not \
mind about.
4. Folksy or indirect wording that names a value still gives that value.

Answer format: ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": \
true when the value comes from another listing, else false}], "drop": \
[{"item": item number, "property": "a requirement the shopper now \
abandons"}], "free": [{"item": item number, "property": "..."}], \
"budget": the new total limit as a number, or null}"""


NOTE_PROMPT = """\
Purpose: Classify a note shown while a shopping agent works (from the \
shopper or the store).

What you get: items, requirements, the note.

How to decide:
1. "kind" is "change" when the note alters the task (budget, requirements, \
a price, availability, a refusal of the order), "error" when it merely \
reports a malformed or rejected call, else "other".
2. A replaced requirement: drop the old one, add the new.
3. List every change the note contains.

Answer format: ONE JSON object and nothing else: {"kind": "change", "error" \
or "other", "budget": new total limit as a number in the request's unit, or \
null, "add": [{"item": item number, "need": "property: constraint"}], \
"drop": [{"item": item number, "need": "the requirement dropped"}], "gone": \
[listing ids called unavailable], "sold_out": true if a chosen listing is \
said to be gone, sold out or unavailable (named or not), "price_moved": \
true if a price is said to have changed, "summary": "one short sentence"}"""


CHECK_PROMPT = """\
Purpose: Give the last review before a shopping agent checks out: judge \
from the records whether this cart is exactly right.

What you get: request, requirements, cart, records.

How to decide:
1. Reading rules: a requirement holds when the listing's attribute for its \
property gives the value (spell keys any way, one item of a list is enough, \
ignore case and singular or plural, accept plain synonyms and a value that \
contains the required value's words, such as "deep teal" for teal; convert \
units, a limit includes its edge, and a range must fall wholly inside it); \
an attribute keyed to the requirement's own property is decisive, and the \
listing fails when it gives something else; an attribute keyed to some \
other property, even a related one (a handle colour is not the body's \
colour), proves nothing; when the listing lacks an attribute for the \
property, the wanted value counts if it appears under another attribute or \
in the title or description (not a number of another kind, as a weight is \
not a capacity, and not a bare figure whose text does not say what it \
measures); a requirement stated by the category label itself is met by that \
category; the category label must match the requested item as a whole; a \
rule to differ from another listing's value holds unless the records show \
that very value.
2. Every requested item needs a separate cart line at the requested \
quantity; the cart may hold nothing else and nothing shown out of stock, \
and price times quantity summed over all lines must stay within budget. A \
listing's newest record counts.
3. When the pick rule is cheapest, most or least, examine the other \
listings shown too: the cart is wrong if one of them clearly meets every \
requirement and beats the cart's choice on that rule.
4. Use ok=false only for a problem the records clearly show, or for a \
requirement no record speaks to.

Answer format: ONE JSON object and nothing else: {"ok": true or false, \
"problems": ["short, concrete problem", ...], "better": [listing ids to put \
in place of a cart line, best first]}"""


READING_RULES = """\
when an attribute's key names the property a requirement is about, that \
attribute decides it (key spelling is irrelevant, one list element \
suffices, synonyms and converted units are accepted, and so is a value that \
includes the required value's words, such as "deep teal" for teal; a limit \
includes its edge); an attribute keyed to another property decides nothing, \
even a related one (a handle colour is not the body's colour); if no \
attribute names the property, the wanted value counts when given under \
another attribute or in the title or description (a bare figure that does \
not say what it measures counts for nothing); the category label satisfies \
a requirement it states itself; the category label has to match the item; a \
rule of differing from another listing's value is met unless the records \
show that very value; a listing without stock, or one that breaks the \
budget, is not a fit."""


AUDIT_PROMPT = (
"""\
Purpose: A shopping agent wants to stop without buying, thinking nothing \
fits; verify it.

What you get: requirements and listing records.

How to decide:
1. Hold the records against the requirements by these reading rules: """
    + READING_RULES
    + """
2. A record marked ROW was seen only as a search result row: its page (the \
attributes) is unread, so on a property the row does not show it can \
neither pass nor fail.

Answer format: ONE JSON object and nothing else: {"fits": {"item number": \
[listing ids that seem to meet every requirement, cheapest first, 4 at \
most]}, "open": [ids of ROW listings of a requested kind, within the \
budget, whose page might show they fit, most promising first, 6 at most], \
"reason": "one sentence"}"""
)


UNMIX_PROMPT = """\
Purpose: Requirements were drawn from a shopper's request for one product. \
Some borrow their value from a different listing the request refers to, and \
the words naming it (its kind, its shop, the properties after it introduced \
by "with ...") can wrongly become requirements on the product bought.

What you get: the request and requirements.

How to decide:
1. By the request's wording, pick out the OTHER REQUIREMENTS below that \
merely describe that other listing.

Answer format: ONE JSON object and nothing else: {"describe_other": [each \
such requirement, copied exactly; empty if all truly apply to the product]}"""


TRAIL_PROMPT = """\
Purpose: The agent is ready to check out. For some requirements the value \
is not written out but borrowed from another listing, and that listing's \
page may in turn refer onward to yet another listing for the property. The \
order is valid only when the agent fully opened each listing on every such \
chain.

What you get: requirements and records (OPENED: a page the agent opened; \
ROW: only a search result row, page unread).

How to decide:
1. Walk each requirement's chain one link after another, starting from the \
requirement's own description, then by what each page says.
2. List a chain's links in sequence, halting at the first link not OPENED \
or not located, since its page's words are still unknown.

Answer format: ONE JSON object and nothing else: {"chains": [{"need": the \
requirement copied exactly, "links": [{"id": the listing id when one of the \
records is that link, else "", "about": the way the requirement or the \
preceding page describes that listing}], "done": true if the final link is \
an OPENED page giving the property's value itself, else false, "value": the \
property's value as given on that final page, or ""}]}"""


GATE_PROMPT = """\
Purpose: The agent will soon STAGE, and its request lists steps to do along \
the way. From the numbered RECORD of its actions, judge for every step owed \
before it may STAGE whether it is done for the cart as it is now.

What you get: steps, RECORD, TOOLS.

How to decide:
1. A step is done only when a RECORD line shows it happening; for a step on \
the cart or the order, that line must also follow the latest cart change, \
with a reply granting the step. Asking about a product's properties never \
does a step.
2. For a missing step, name the single store call to make next: a tool in \
TOOLS with its arguments. Messages to the shopper use the messaging tool \
and hold one brief sentence or question about that step only, worded \
plainly, mentioning no product properties or values that step does not \
need.
3. Return no call if all due steps are done, or if the remaining messages \
or steps cannot do the missing one.

Answer format: ONE JSON object and nothing else: {"steps": [{"step": the \
step, "done": true or false, "lines": [numbers of the RECORD lines proving \
it]}], "call": {"tool": "...", "args": {...}} or null, "hold": the change a \
reply demands before the agent continues, or "", "needs": [{"item": item \
number, "need": "property: constraint"} for every firm requirement that \
reply states]}"""


SWAP_PROMPT = (
"""\
Purpose: A cart listing just sold out; find replacements for that line \
among the other records shown.

What you get: requirements and records.

How to decide:
1. Use these reading rules: """
    + READING_RULES
    + """

Answer format: ONE JSON object and nothing else: {"fits": [{"id": listing \
id, "price": its price as a number, as the record shows it} for each \
listing that seems to meet every requirement of the sold-out line's item, 6 \
at most], "reason": "one sentence"}"""
)


SHAPE_PROMPT = """\
Purpose: Map a store tool's result.

What you get: tool and result.

How to decide:
1. Identify each part by the exact key used in the result or the tool's \
parameters.

Answer format: ONE JSON object and nothing else: {"rows": key of the list \
of listings, "count": key holding how many listings matched overall (not \
how many are shown), or null, "next": key holding the value that asks for \
the next page, or null, "next_arg": tool parameter taking that value, or \
null, "price": key, in a row, of the listing's price, or null, "stock": \
key, in a row, saying if the listing can be bought now, or null}"""


def to_json(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    return text


def clip(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + "..."
    deeper = depth + 1
    if isinstance(value, dict):
        return {key: clip(item, limit, deeper) for key, item in value.items()}
    if isinstance(value, list):
        return [clip(item, limit, deeper) for item in value]
    return value


def _decode_dict(text: str) -> dict[str, Any] | None:
    loaded = json.loads(text)
    if isinstance(loaded, dict):
        return loaded
    return None


def parse_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    fenced = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", body, re.S)
    if fenced:
        body = fenced.group(1)
    try:
        return _decode_dict(body)
    except ValueError:
        pass
    start, end = body.find("{"), body.rfind("}")
    if not 0 <= start < end:
        return None
    try:
        return _decode_dict(body[start:end + 1])
    except ValueError:
        return None


def spaced_words(key: str) -> str:
    split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", split).strip().lower()


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def word_stems(text: str) -> frozenset[str]:
    return frozenset(_singular(word) for word in spaced_words(text).split())


def names_same(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not (prop and key):
        return False
    if prop == key or prop <= key:
        return True
    return len(key) >= 2 and key <= prop


def schema_types(schema: dict[str, Any]) -> list[str]:
    declared = schema.get("type")
    if isinstance(declared, str):
        return [declared]
    if not declared:
        return []
    return [entry for entry in declared if isinstance(entry, str)]


def _plain_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def fits_type(value: Any, kind: str) -> bool:
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "number":
        return _plain_number(value)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    return True


def _fits_any(value: Any, kinds: list[str]) -> bool:
    return any(fits_type(value, kind) for kind in kinds)


def _convert(value: Any, kind: str) -> tuple[bool, Any]:
    if kind in ("integer", "number") and isinstance(value, str):
        if not re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
            return False, value
        number = float(value)
        if kind == "integer" or number.is_integer():
            return True, int(number)
        return True, number
    if kind == "integer" and isinstance(value, float):
        if value.is_integer():
            return True, int(value)
        return False, value
    if kind == "string" and _plain_number(value):
        return True, str(value)
    if kind == "boolean" and isinstance(value, str):
        word = value.strip().lower()
        if word in ("true", "false"):
            return True, word == "true"
        return False, value
    if kind == "array" and value is not None and not isinstance(value, list):
        return True, [value]
    return False, value


def _coerce(value: Any, kinds: list[str]) -> Any:
    for kind in kinds:
        changed, converted = _convert(value, kind)
        if changed:
            return converted
    return value


def _pick_option(value: Any, schema: dict, where: str,
                 faults: list[str]) -> Any:
    options = schema.get("enum")
    if not isinstance(options, list) or value in options:
        return value
    if isinstance(value, str):
        wanted = value.strip().lower()
        for option in options:
            if isinstance(option, str) and option.lower() == wanted:
                value = option
                break
    if value not in options:
        faults.append(f"{where}: allowed values are {options}")
    return value


def _fit_text(value: str, schema: dict, where: str,
              faults: list[str]) -> str:
    longest = schema.get("maxLength")
    if isinstance(longest, int) and len(value) > longest:
        value = value[:longest]
    shortest = schema.get("minLength")
    if isinstance(shortest, int) and len(value) < shortest:
        faults.append(f"{where} is too short")
    return value


def _check_bounds(value: Any, schema: dict, where: str,
                  faults: list[str]) -> None:
    low = schema.get("minimum")
    if isinstance(low, (int, float)) and value < low:
        faults.append(f"{where} is below {low}")
    high = schema.get("maximum")
    if isinstance(high, (int, float)) and value > high:
        faults.append(f"{where} is above {high}")


def _fit_list(value: list, schema: dict, where: str,
              faults: list[str]) -> list:
    most = schema.get("maxItems")
    if isinstance(most, int) and len(value) > most:
        faults.append(f"{where} allows at most {most} entries")
    least = schema.get("minItems")
    if isinstance(least, int) and len(value) < least:
        faults.append(f"{where} needs at least {least} entries")
    inner = schema.get("items")
    if not isinstance(inner, dict):
        return value
    return [
        conform(entry, inner, f"{where}[{index}]", faults)
        for index, entry in enumerate(value)
    ]


def _fit_mapping(value: dict, schema: dict, where: str,
                 faults: list[str]) -> dict:
    props = schema.get("properties")
    if not isinstance(props, dict):
        props = {}
    if schema.get("additionalProperties") is False:
        extra = [name for name in value if name not in props]
        for name in extra:
            faults.append(f"{where}.{name} is not an allowed argument")
    for name in schema.get("required") or []:
        if name not in value:
            faults.append(f"{where}.{name} is required")
    shaped = {}
    for key, item in value.items():
        if key in props:
            item = conform(item, props[key], f"{where}.{key}", faults)
        shaped[key] = item
    return shaped


def conform(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = schema_types(schema)
    if kinds and not _fits_any(value, kinds):
        value = _coerce(value, kinds)
        if not _fits_any(value, kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    value = _pick_option(value, schema, where, faults)
    if isinstance(value, str):
        value = _fit_text(value, schema, where, faults)
    if _plain_number(value):
        _check_bounds(value, schema, where, faults)
    if isinstance(value, list):
        value = _fit_list(value, schema, where, faults)
    if isinstance(value, dict):
        value = _fit_mapping(value, schema, where, faults)
    return value


def as_rows(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []


def as_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value):
        return int(float(value))
    return None


def looks_like_id(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    core = value.strip()
    if not 3 <= len(core) <= 64 or " " in core:
        return False
    return named or bool(re.search(r"\d", value))


def guess_id_key(rows: list[dict[str, Any]]) -> str | None:
    top_score = None
    top_key = None
    for key in rows[0]:
        named = "id" in str(key).lower()
        values = [row.get(key) for row in rows]
        if not all(looks_like_id(value, named) for value in values):
            continue
        cleaned = [value.strip() for value in values]
        if len(rows) > 1 and len(set(cleaned)) < len(cleaned):
            continue
        score = (named, sum(map(len, cleaned)) / len(cleaned))
        if top_score is None or score > top_score:
            top_score, top_key = score, key
    return top_key


def _row_has(entry: Any, ident: str) -> bool:
    if not isinstance(entry, dict):
        return False
    texts = [str(x).strip() for x in entry.values() if isinstance(x, str)]
    return ident in texts


def holds_id(node: Any, ident: str, depth: int = 0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, dict):
        children = list(node.values())
    elif isinstance(node, list):
        if any(_row_has(entry, ident) for entry in node):
            return True
        children = node
    else:
        return False
    return any(holds_id(child, ident, depth + 1) for child in children)


def find_facts(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for block in node.values():
        if not isinstance(block, dict) or len(block) < 2:
            continue
        scalars = {
            key: item for key, item in block.items()
            if isinstance(item, str) or _plain_number(item)
        }
        if len(scalars) < 2:
            continue
        if best is None or len(scalars) > len(best):
            best = scalars
    return best


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name, self.shelf = name, shelf
        self.qty, self.musts = qty, musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Ledger:
    def __init__(self, problem: dict[str, Any]):
        self.started = time.monotonic()
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.view = env["policy_view"]
        self.request = str(self.view.get("query") or "")
        self.tools = [
            spec for spec in self.view.get("tools") or []
            if self._named_tool(spec)
        ]
        self.schemas = {
            spec["function"]["name"]: self._params_of(spec)
            for spec in self.tools
        }
        self.max_steps = self._step_budget(self.view.get("max_steps"))
        self.step = 0
        self.finished = bool(self.view.get("finished"))
        self.broke = False
        self.no_credit = False
        # LLM accounting
        self.waited = 0
        self.misses = 0
        self.cost = 0.0
        self.calls = 0
        self.loop_calls = 0
        self.trace: list[dict[str, Any]] = []
        # what the request asks for
        self.brief: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.items: list[Item] = []
        self.budget: Any = None
        self.msg_limit = 0
        self.msg_used = 0
        self.process: list[dict[str, Any]] = []
        self.reserve = 0
        self.gate_msgs = 0
        self.gate_ok: set[tuple[str, str]] = set()
        self.done_log: list[str] = []
        self.done_steps: list[int] = []
        self.asked = False
        self.qa: list[tuple[str, str]] = []
        # listings seen so far
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.seen_order: list[str] = []
        self.heard: set[str] = set()
        self.notes: list[str] = []
        self.version = 0
        # cart bookkeeping
        self.cart_touched = False
        self.cart_at = -1
        self.cart_seen_at = -1
        self.cart_view: Any = None
        self.cart_lines: dict[str, int] = {}
        self.checked: set[str] = set()
        self.audited = False
        self.offer_seen: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.held = 0
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ""
        self.refused_at = -1
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.pushed = False
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.late_held = False
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed: str | None = None
        self.completed = False
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.queued: list[str] = []
        self.buys = 0
        self.refusals = 0
        self.recent: list[str] = []
        self.chat: list[dict[str, Any]] = []
        # pages opened and values pointed at by other listings
        self.opened: set[str] = set()
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.trails = 0
        self.trail_blocks = 0
        self.counts_held = False
        self.categories_held = False
        self.fallback_held = False
        self.compare_ok = 0
        self.compare_bad = 0
        # result shapes and caches
        self.tool_keys: dict[str, set[str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.pages_read = 0
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.paged_obs: list[Any] = []

    @staticmethod
    def _named_tool(spec: Any) -> bool:
        if not isinstance(spec, dict):
            return False
        fn = spec.get("function")
        return isinstance(fn, dict) and isinstance(fn.get("name"), str)

    @staticmethod
    def _params_of(spec: dict[str, Any]) -> dict[str, Any]:
        params = spec["function"].get("parameters")
        if isinstance(params, dict):
            return params
        return {"type": "object", "properties": {}}

    @staticmethod
    def _step_budget(raw: Any) -> int:
        steps = as_count(raw)
        if steps is None and isinstance(raw, float):
            steps = int(raw)
        if not steps or steps <= 0:
            return 30
        return steps

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def out_of_time(self) -> bool:
        return time.monotonic() - self.started >= CLOCK_WRAP

    def free_messages(self) -> int:
        held_back = max(0, self.reserve - self.gate_msgs)
        return self.msg_limit - self.msg_used - held_back

    def cart_signature(self) -> str:
        return to_json(sorted(self.cart_lines.items()))

    def gate_pending(self, stage: str) -> bool:
        if not any(step["when"] == stage for step in self.process):
            return False
        mark = self.cart_signature() + f"#{self.version}"
        return (stage, mark) not in self.gate_ok

    def finish_reserve(self) -> int:
        talks_at_order = any(
            step["when"] == "order" and step["messages"] > 0
            for step in self.process
        )
        if not talks_at_order or self.msg_used >= self.msg_limit:
            return 1
        return 2 if self.gate_pending("order") else 1


class Service:
    def __init__(self, hub: Episode):
        self.hub = hub
        self.led = hub.led


class LlmService(Service):
    def consult(
        self,
        messages: list[dict[str, Any]],
        models: list[str],
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        for model in models:
            if self.led.calls >= ALL_CALLS:
                return None
            self.led.calls += 1
            message, status = self._with_resends(
                model, messages, max_tokens, tools
            )
            if message is not None:
                return message
            if status == 402:
                self.led.no_credit = True
                return None
        return None

    def _with_resends(self, model, messages, max_tokens, tools):
        attempt = 0
        while True:
            message, status = self.post_once(
                model, messages, max_tokens, tools, self._last_go(attempt)
            )
            if message is not None:
                return message, status
            if not self._may_resend(status, attempt):
                return None, status
            wait = WAITS[attempt]
            pause = wait * (0.8 + 0.4 * random.random())
            self.led.waited += wait
            attempt += 1
            time.sleep(pause)

    def _may_resend(self, status: Any, attempt: int) -> bool:
        transient = (
            status is None
            or status in (402, 408, 425, 429)
            or (isinstance(status, int) and status >= 500)
        )
        if not transient:
            return False
        # a timed-out call already spent its 50 s, so it gets one more go
        allowed = 1 if status in (None, 408) else len(WAITS)
        if attempt >= allowed:
            return False
        return self.led.waited + WAITS[attempt] <= WAIT_TOTAL

    def _last_go(self, attempt: int) -> bool:
        if attempt == 0:
            return False
        return (
            attempt >= len(WAITS)
            or self.led.waited + WAITS[attempt] > WAIT_TOTAL
        )

    @staticmethod
    def _with_cache_marks(model, messages):
        if not model.startswith("anthropic/") or len(messages) <= 1:
            return messages
        marked = [dict(message) for message in messages]
        for index in (0, len(marked) - 1):
            text = marked[index].get("content")
            if isinstance(text, str) and text:
                marked[index]["content"] = [{
                    "type": "text",
                    "text": text,
                    "cache_control": {"type": "ephemeral"},
                }]
        return marked

    def post_once(self, model, messages, max_tokens, tools, last=False):
        body: dict[str, Any] = {
            "model": model,
            "messages": self._with_cache_marks(model, messages),
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if last:
            # no resend can follow this one, so only the reply is needed
            data = _llm.post("/inference/chat/completions", body)
            if isinstance(data, dict) and data.get("choices"):
                return self._read_reply(model, data)
            self.led.trace.append(
                {"kind": "llm_fail", "model": model, "status": None}
            )
            return None, None
        result = _llm.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            return self._read_reply(model, data)
        problem = result.error or {}
        status = problem.get("status")
        inner = data.get("error") if isinstance(data, dict) else None
        if isinstance(inner, dict):
            # provider error wrapped in a 200 reply
            status = as_count(inner.get("code"))
            problem = {"body": inner.get("message")}
        self.led.trace.append({
            "kind": "llm_fail",
            "model": model,
            "status": status,
            "body": str(problem.get("body") or "")[:200],
        })
        return None, status

    def _read_reply(self, model, data):
        usage = data.get("usage") or {}
        if isinstance(usage.get("cost"), (int, float)):
            self.led.cost += float(usage["cost"])
        message = data["choices"][0].get("message") or {}
        if message.get("content") or message.get("tool_calls"):
            return message, 200
        self.led.trace.append(
            {"kind": "llm_fail", "model": model, "status": 0}
        )
        return None, 0

    def ask_for_json(
        self, prompt: str, models: list[str], max_tokens: int = 1800
    ) -> dict[str, Any] | None:
        led = self.led
        for model in models:
            turn = [{"role": "user", "content": prompt}]
            reply = self.consult(turn, [model], max_tokens) or {}
            parsed = parse_object(reply.get("content"))
            if parsed is not None:
                return parsed
            if led.broke or led.no_credit or led.calls >= ALL_CALLS:
                return None
        return None


class RequestReader(Service):
    def read(self) -> None:
        led = self.led
        prompt = self._brief_prompt()
        ask = self.hub.llm.ask_for_json
        brief = ask(prompt, [LIGHT], 4000) or {}
        if not led.no_credit and self._brief_is_thin(brief):
            brief = ask(prompt, [HEAVY, LIGHT], 2500) or brief
        led.brief = brief
        found = brief.get("roles")
        if not isinstance(found, dict):
            found = {}
        led.roles = {
            role: tool for role, tool in found.items()
            if isinstance(tool, str) and tool in led.schemas
        }
        for key in ("buy_args", "refuse_args"):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        self._split_shared_end_tool(brief)
        self._take_items(brief)
        self._take_lookups(brief)
        self.drop_pointer_needs()
        self._take_limits(brief)
        self._take_process(brief)

    def _brief_prompt(self) -> str:
        led = self.led
        listed = []
        for tool in led.tools:
            spec = tool["function"]
            params = to_json(spec.get("parameters") or {})
            listed.append(
                f"- {spec['name']}: {spec.get('description', '')}"
                f" | parameters: {params}"
            )
        prompt = (
            BRIEF_PROMPT + "\nTOOLS:\n" + "\n".join(listed)
            + "\n\nREQUEST:\n" + led.request
        )
        opening = led.view.get("opening")
        if opening is None:
            return prompt
        return (
            prompt + "\n\nOPENING NOTE FROM THE STORE:\n"
            + to_json(clip(opening, 1500))
        )

    @staticmethod
    def _brief_is_thin(brief: dict[str, Any]) -> bool:
        if not as_rows(brief.get("items")):
            return True
        roles = brief.get("roles")
        return not isinstance(roles, dict) or not roles.get("buy")

    def _split_shared_end_tool(self, brief: dict[str, Any]) -> None:
        buy = self.led.roles.get("buy")
        if not buy or buy != self.led.roles.get("refuse"):
            return
        buy_args, refuse_args = brief["buy_args"], brief["refuse_args"]
        if buy_args and refuse_args and buy_args != refuse_args:
            return
        ask = (
            f"Depending on its arguments, the store tool {buy} either "
            "checks out or ends with no purchase. Schema of the tool: "
            + to_json(self.led.schemas.get(buy))
            + "\nReply with ONE JSON object only: {\"buy_args\": the fixed "
            "arguments that make it check out, \"refuse_args\": the fixed "
            "arguments that make it end with no purchase} (include only "
            "the arguments that select the mode)."
        )
        fixed = self.hub.llm.ask_for_json(ask, [LIGHT, HEAVY], 600) or {}
        new_buy, new_refuse = fixed.get("buy_args"), fixed.get("refuse_args")
        if not isinstance(new_buy, dict) or not isinstance(new_refuse, dict):
            return
        if new_buy != new_refuse:
            brief["buy_args"], brief["refuse_args"] = new_buy, new_refuse

    def _take_items(self, brief: dict[str, Any]) -> None:
        led = self.led
        for entry in as_rows(brief.get("items")):
            if not isinstance(entry, dict):
                continue
            wanted = as_count(entry.get("quantity"))
            if not wanted or wanted <= 0:
                wanted = 1
            musts = [
                str(need) for need in as_rows(entry.get("musts")) if need
            ]
            name = str(entry.get("name") or "the product")
            shelf = str(entry.get("shelf") or "")
            led.items.append(Item(name, shelf, wanted, musts))
        if not led.items:
            led.items = [Item("the requested product", "", 1, [])]

    def _take_lookups(self, brief: dict[str, Any]) -> None:
        led = self.led
        for lookup in as_rows(brief.get("lookups")):
            text = self._matching_must(str(lookup).strip())
            number = 1
            for position, item in enumerate(led.items, 1):
                if text in item.musts:
                    number = position
                    break
            if text and (number, text) not in led.refs:
                led.refs.append((number, text))

    def _matching_must(self, text: str) -> str:
        items = self.led.items
        if any(text in item.musts for item in items):
            return text
        # a loose copy: take the must that names the same property
        key = spaced_words(text.split(":")[0])
        for item in items:
            for need in item.musts:
                if spaced_words(need.split(":")[0]) == key:
                    return need
        return text

    def _take_limits(self, brief: dict[str, Any]) -> None:
        led = self.led
        budget = brief.get("budget")
        if isinstance(budget, str) and re.fullmatch(
                r"\s*\d+(\.\d+)?\s*", budget):
            budget = float(budget)
        is_number = (
            isinstance(budget, (int, float)) and not isinstance(budget, bool)
        )
        if is_number and budget > 0:
            led.budget = budget
        limit = as_count(brief.get("messages"))
        if limit is not None and limit >= 0:
            led.msg_limit = limit
        elif led.roles.get("ask"):
            led.msg_limit = 3
        steps = as_count(brief.get("steps"))
        if steps is not None and 10 <= steps < led.max_steps:
            led.max_steps = steps

    def _take_process(self, brief: dict[str, Any]) -> None:
        led = self.led
        for entry in as_rows(brief.get("process")):
            if not isinstance(entry, dict):
                continue
            if not str(entry.get("step") or "").strip():
                continue
            when = str(entry.get("when") or "").strip().lower()
            if when not in ("choose", "order", "end"):
                when = "order"
            led.process.append({
                "step": str(entry["step"]).strip(),
                "when": when,
                "messages": max(0, as_count(entry.get("messages")) or 0),
            })
        if not led.roles.get("ask"):
            return
        talk = [
            entry["messages"] for entry in led.process
            if entry["when"] in ("order", "end") and entry["messages"]
        ]
        led.reserve = min(max(talk), led.msg_limit) if talk else 0

    def drop_pointer_needs(self) -> None:
        for number, item in enumerate(self.led.items, 1):
            pointers = [
                need for at, need in self.led.refs
                if at == number and need in item.musts
            ]
            if not pointers:
                continue
            rest = [need for need in item.musts if need not in pointers]
            if rest:
                self._move_listing_words(number, item, pointers, rest)

    def _move_listing_words(
        self, number: int, item: Item, pointers: list[str], rest: list[str]
    ) -> None:
        led = self.led
        prompt = (
            UNMIX_PROMPT + "\n\nREQUEST:\n" + led.request
            + f"\n\nPRODUCT: {item.name}\n"
            "REQUIREMENTS POINTING AT ANOTHER LISTING:\n"
            + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n"
            + "\n".join(rest)
        )
        got = self.hub.llm.ask_for_json(prompt, [HEAVY, LIGHT], 500) or {}
        named = [str(x).strip() for x in as_rows(got.get("describe_other"))]
        moved = [need for need in rest if need in named]
        if not moved:
            return
        head = pointers[0]
        joined = head + " (that listing: " + "; ".join(moved) + ")"
        kept = [need for need in item.musts if need not in moved]
        item.musts = [joined if need == head else need for need in kept]
        led.refs = [
            (at, joined if (at, need) == (number, head) else need)
            for at, need in led.refs
        ]
        led.trace.append({"kind": "unmix", "item": number, "moved": moved})

    def summary(self) -> str:
        led = self.led
        lines = self._item_lines()
        brief = led.brief
        if led.budget is not None:
            if brief.get("budget_scope") == "each":
                scope = "on every item"
            else:
                scope = "all told"
            unit = brief.get("budget_unit") or ""
            lines.append(
                f"budget: ceiling {led.budget} {unit} {scope}"
                .replace("  ", " ")
            )
        lines.append(self._pick_line(brief))
        if brief.get("in_stock_only", True) is not False:
            lines.append("only in stock")
        lines.extend(
            f"wish (only breaks ties): {wish}"
            for wish in as_rows(brief.get("wishes"))
        )
        lines.extend(
            f"rule given by request: {rule}"
            for rule in as_rows(brief.get("rules"))
        )
        fallback = str(brief.get("nothing_fits") or "").strip()
        if fallback:
            lines.append(f"if no listing fits, the request says: {fallback}")
        if brief.get("quiet_decline") is True:
            lines.append(
                "if no listing fits: finish without naming or recommending "
                "one"
            )
        moments = {
            "choose": "ahead of choice",
            "order": "at checkout (Kyo carries it out as you order)",
            "end": "when it ends with no purchase (Kyo carries it out)",
        }
        for entry in led.process:
            lines.append(
                f"step this request demands, {moments[entry['when']]}: "
                f"{entry['step']}"
            )
        facts = self.hub.catalogue.facts_tool_name()
        if facts and (
                led.facts_from.get(facts) is True or led.roles.get("offer")):
            lines.append(self._facts_line(facts))
        return "\n".join("- " + line for line in lines)

    def _item_lines(self) -> list[str]:
        led = self.led
        numbered = len(led.items) > 1
        lines = []
        for number, item in enumerate(led.items, 1):
            needs = [
                self._with_page_value(need)
                for need in item.musts + item.told
                if need not in item.dropped
            ]
            if numbered:
                text = f"item {number}: {item.name}"
            else:
                text = f"item: {item.name}"
            if item.shelf:
                text += f" (shelf label: {item.shelf})"
            text += f", quantity {item.qty}; must satisfy: "
            if needs:
                text += "; ".join(needs)
            else:
                text += "only the kind of product itself"
            if item.free:
                text += "; shopper is easy about: " + ", ".join(item.free)
            lines.append(text)
        return lines

    def _with_page_value(self, need: str) -> str:
        values = self.led.ref_values
        if need not in values:
            return need
        return need + f" (value its target page gives: {values[need]})"

    @staticmethod
    def _pick_line(brief: dict[str, Any]) -> str:
        pick = brief.get("pick")
        if pick == "cheapest":
            return "pick: the fitting listing of lowest cost (cheapest)"
        if pick not in ("most", "least"):
            return "pick: any listing that meets all needs"
        measure = brief.get("pick_measure") or ""
        return (
            f"pick: the {pick} {measure} across fitting options"
            .replace("  ", " ")
        )

    def _facts_line(self, facts: str) -> str:
        led = self.led
        catalogue = self.hub.catalogue
        offer = led.roles.get("offer") or ""
        bare = sorted(
            tool for tool in catalogue.page_tool_set() | {offer}
            if tool and led.facts_from.get(tool) is False
        )
        pages = sorted(
            tool for tool in catalogue.page_tool_set() if tool != facts
        )
        text = f"listing attributes appear in {facts}"
        if bare:
            text += f"; {', '.join(bare)} give none"
        if pages:
            text += (
                "; a listing's page (what a page chain or a page check "
                f"refers to) is shown by {' or '.join(pages)}"
            )
        return text

    def status_line(self) -> str:
        led = self.led
        left = led.steps_left()
        text = (
            f"[Kyo] Step {led.step} of {led.max_steps} done, "
            f"{left} left."
        )
        text += self._pace_hint(left)
        live = {need for _, need in self.hub.judge.current_needs()}
        open_refs = [
            need for _, need in led.refs
            if need in live and need not in led.ref_values
        ]
        if open_refs and led.buys == 0:
            text += (
                "\n[resolve first: open the listing each of these points "
                "to, then every listing its page points to; the value is "
                "on the end page] " + " | ".join(open_refs[:3])
            )
        if led.leads:
            text += "\n[review candidates] " + ", ".join(led.leads)
        if led.notes:
            text += "\n[changes so far] " + " | ".join(led.notes[-4:])
        return text + "\n[ledger]\n" + self.summary()

    def _pace_hint(self, left: int) -> str:
        led = self.led
        unbought = led.buys == 0
        if left <= led.finish_reserve() + 1:
            return (
                " FINAL STEPS: if the cart has a line for every item, check "
                "out now; end with no purchase only if your search proved "
                "some item has no fitting listing (a shortfall of steps "
                "proves nothing of that kind)."
            )
        if left <= 6 or (unbought and left <= len(led.items) + 5):
            return (
                " Just a few steps remain: cart the chosen lines, then "
                "check out now."
            )
        if unbought and left <= 12:
            return (
                " Steps are getting short: decide on your picks, fill the "
                "cart and check out over the next few steps."
            )
        if (unbought and not led.cart_touched
                and led.step >= led.max_steps // 2):
            return (
                " Half of the steps are gone, the cart empty: "
                "add a fitting listing for each item now (the store can "
                "still sell one out)."
            )
        return ""

    def row_id(
        self, node: Any, in_list: bool = False, depth: int = 0
    ) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        key = self.led.id_key
        if key and looks_like_id(node.get(key), True):
            return node[key].strip()
        nested = any(isinstance(value, dict) for value in node.values())
        if not in_list and depth != 0 and not nested:
            return None
        guess = guess_id_key([node])
        if not guess:
            return None
        if "id" in str(guess).lower() or node[guess].strip() in self.led.rows:
            return node[guess].strip()
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [
            entry for entry in node
            if isinstance(entry, dict) and len(entry) >= 3
        ]
        if not rows or len(rows) < len(node) / 2:
            return False
        key = self.led.id_key
        if key and all(looks_like_id(row.get(key), True) for row in rows):
            return True
        guess = guess_id_key(rows)
        if not guess:
            return False
        self.led.id_key = guess
        return True

    def absorb(self, observation: Any) -> None:
        self._visit(observation, 0)

    def _visit(self, node: Any, depth: int, in_list: bool = False) -> None:
        if depth > 6:
            return
        if isinstance(node, list):
            listed = self.learn_rows(node)
            for value in node:
                self._visit(value, depth + 1, listed)
            return
        if not isinstance(node, dict):
            return
        ident = self.row_id(node, in_list, depth)
        if not ident:
            for value in node.values():
                self._visit(value, depth + 1)
            return
        self._store_row(ident, node)

    def _store_row(self, ident: str, node: dict[str, Any]) -> None:
        led = self.led
        if ident not in led.rows:
            led.seen_order.append(ident)
            led.rows[ident] = {}
        known = led.rows[ident]
        # a flag of the record flipping true -> false marks it unbuyable
        if any(known.get(k) is True and v is False for k, v in node.items()):
            led.unavailable.add(ident)
        elif any(
                known.get(k) is False and v is True
                for k, v in node.items()):
            led.unavailable.discard(ident)
        known.update(node)
        if find_facts(node):
            led.full[ident] = dict(known)
        elif ident in led.full:
            led.full[ident].update(node)


class Catalogue(Service):
    def page_tool_set(self) -> set[str]:
        roles = self.led.roles
        page = roles.get("offer") or roles.get("open")
        tools: set[str] = set()
        for tool in (page, roles.get("compare")):
            if tool:
                tools.add(tool)
        return tools

    def facts_tool_name(self) -> str | None:
        facts_from = self.led.facts_from
        for tool, showed in facts_from.items():
            if showed and self.takes_one(tool):
                return tool
        opener = self.led.roles.get("open")
        if facts_from.get(opener or "") is False:
            return None
        return opener if self.takes_one(opener) else None

    def compare_shows_facts(self) -> bool | None:
        compare = self.led.roles.get("compare")
        return self.led.facts_from.get(compare or "")

    def compare_is_bare(self) -> bool:
        seen = self.compare_shows_facts()
        if seen is not None:
            return seen is False
        reader = self.facts_tool_name()
        if self.led.facts_from.get(reader or "") is True:
            return True
        return bool(self.led.roles.get("offer"))

    def attribute_labels(self) -> set[str]:
        labels: set[str] = set()
        for schema in self.led.schemas.values():
            self._gather_enums(schema, 0, labels)
        return labels

    def _gather_enums(self, schema: Any, depth: int, into: set[str]) -> None:
        if depth > 6 or not isinstance(schema, dict):
            return
        for option in schema.get("enum") or []:
            if isinstance(option, str):
                into.add(option.strip().lower())
        for child in schema.values():
            if isinstance(child, dict):
                self._gather_enums(child, depth + 1, into)

    def category_field(self) -> str | None:
        distinct: dict[str, set[str]] = {}
        carried: dict[str, int] = {}
        for row in self.led.rows.values():
            for key, value in row.items():
                if not self._short_label(value):
                    continue
                distinct.setdefault(key, set()).add(value)
                carried[key] = carried.get(key, 0) + 1
        total = max(1, len(self.led.rows))
        known = self.attribute_labels()
        fields = [
            key for key, values in distinct.items()
            if carried[key] >= 0.8 * total and len(values) <= 40
        ]
        if not fields:
            return None

        def rank(key: str) -> tuple[float, int, int, float]:
            values = distinct[key]
            in_schema = sum(
                1 for v in values if v.strip().lower() in known
            ) / len(values)
            mean_len = sum(len(v) for v in values) / len(values)
            return (-in_schema, len(values), -carried[key], -mean_len)

        return min(fields, key=rank)

    @staticmethod
    def _short_label(value: Any) -> bool:
        return (
            isinstance(value, str)
            and 0 < len(value) <= 40
            and not re.search(r"\d{3}", value)
        )

    def _wanted_terms(self) -> list[tuple[str, str, set[str]]]:
        terms = []
        for item in self.led.items:
            shelf = item.shelf or ""
            stems = word_stems(shelf) | word_stems(item.name)
            long_stems = {w for w in stems if len(w) >= 4}
            terms.append(
                (spaced_words(shelf), spaced_words(item.name), long_stems)
            )
        return terms

    @staticmethod
    def _label_wanted(
        label: str, terms: list[tuple[str, str, set[str]]]
    ) -> bool:
        words = spaced_words(label)
        long_stems = {w for w in word_stems(label) if len(w) >= 4}
        for shelf, name, stems in terms:
            if words in (shelf, name) or long_stems & stems:
                return True
        return False

    def property_table(self) -> list[tuple[str, str, int, list[str]]]:
        groups = self._fact_groups(self.category_field())
        terms = self._wanted_terms()
        hits = {
            label for label in groups if self._label_wanted(label, terms)
        }
        if hits:
            groups = {
                label: group for label, group in groups.items()
                if label in hits
            }
        table: list[tuple[str, str, int, list[str]]] = []
        by_size = sorted(groups.items(), key=lambda pair: -pair[1]["count"])
        for label, group in by_size:
            for name, slot in self._useful_keys(group["keys"]):
                samples = [v[:40] for v in slot["values"][:4]]
                table.append((label, name, slot["count"], samples))
        return table

    def _fact_groups(self, cat_key: str | None) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.led.seen_order:
            record = self.led.full.get(ident)
            if not record:
                continue
            facts = find_facts(record) or {}
            label = "listings"
            if cat_key:
                label = str(record.get(cat_key) or "listings")
            group = groups.setdefault(label, {"count": 0, "keys": {}})
            group["count"] += 1
            self._tally_facts(group["keys"], ident, facts)
        return groups

    @staticmethod
    def _tally_facts(
        slots: dict[str, dict[str, Any]], ident: str, facts: dict[str, Any]
    ) -> None:
        names_done: set[str] = set()
        for key, value in facts.items():
            name = spaced_words(key)
            if not name or name in names_done:
                continue
            names_done.add(name)
            slot = slots.setdefault(
                name, {"count": 0, "values": [], "per": {}}
            )
            slot["count"] += 1
            text = str(value).strip().lower()
            slot["per"][ident] = text
            if text not in slot["values"]:
                slot["values"].append(text)

    def _useful_keys(
        self, slots: dict[str, dict[str, Any]]
    ) -> list[tuple[str, dict[str, Any]]]:
        kept: list[tuple[str, dict[str, Any]]] = []
        for name, slot in sorted(
            slots.items(), key=lambda pair: -pair[1]["count"]
        ):
            if len(kept) == 14:
                break
            if not self._readable_values(list(slot["per"].values())):
                continue
            if any(
                name.startswith(other + " ") and slot["per"] == prev["per"]
                for other, prev in kept
            ):
                continue
            kept.append((name, slot))
        return kept

    @staticmethod
    def _readable_values(values: list[str]) -> bool:
        if not values:
            return False
        mean_len = sum(len(v) for v in values) / len(values)
        if mean_len > 40:
            return False
        if sum(1 for v in values if " / " in v) * 2 >= len(values):
            return False
        all_unique = len(set(values)) == len(values)
        return not (len(values) >= 4 and all_unique and mean_len > 12)

    def item_kinds(self) -> list[list[str]]:
        cat_key = self.category_field()
        groups: dict[str, list[str]] = {}
        for ident in self.led.seen_order:
            label = ""
            if cat_key:
                label = str(self.led.rows[ident].get(cat_key) or "")
            groups.setdefault(label, []).append(ident)
        terms = self._wanted_terms()
        pool = [
            ids for label, ids in groups.items()
            if self._label_wanted(label, terms)
        ]
        if not pool:
            pool = list(groups.values())
        pool = sorted(pool, key=lambda ids: -len(ids))
        return pool[:max(1, len(self.led.items))]

    def takes_one(self, tool: str | None) -> str | None:
        schema = self.led.schemas.get(tool or "") or {}
        params = schema.get("properties") or {}
        free_text = [
            key for key, spec in params.items()
            if isinstance(spec, dict)
            and "string" in schema_types(spec)
            and "enum" not in spec
        ]
        required = schema.get("required") or []
        target = next((key for key in required if key in free_text), None)
        if not target:
            target = free_text[0] if len(free_text) == 1 else None
        if not tool or not target:
            return None
        for key in required:
            if key != target:
                return None
        return target

    def price(self, ident: str) -> float | None:
        price_key = self.led.price_key
        if not price_key:
            return None
        value = (self.led.rows.get(ident) or {}).get(price_key)
        if isinstance(value, dict):
            value = next(
                (v for v in value.values() if self._is_number(v)), None
            )
        if isinstance(value, str) and re.fullmatch(
            r"\s*\d+(\.\d+)?\s*", value
        ):
            value = float(value)
        return float(value) if self._is_number(value) else None

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def in_stock(self, ident: str) -> bool:
        if self.hub.judge.is_sold_out(ident):
            return False
        stock_key = self.led.stock_key
        if not stock_key:
            return True
        flag = (self.led.rows.get(ident) or {}).get(stock_key)
        if flag is False:
            return False
        if self._is_number(flag):
            return not flag <= 0
        return True

    def records(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, list):
            return self._identified(observation)
        if not isinstance(observation, dict):
            return []
        own = self.hub.intake.row_id(observation)
        if own:
            return [(own, observation)]
        found: list[tuple[str, dict[str, Any]]] = []
        for value in observation.values():
            if isinstance(value, list):
                found += self._identified(value)
            elif isinstance(value, dict):
                for inner in value.values():
                    if isinstance(inner, list):
                        found += self._identified(inner)
        return found

    def _identified(
        self, rows: list[Any]
    ) -> list[tuple[str, dict[str, Any]]]:
        found = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            ident = self.hub.intake.row_id(row, True)
            if ident:
                found.append((ident, row))
        return found

    def remember_view(self, name: str, observation: Any) -> None:
        led = self.led
        for ident, record in self.records(observation):
            fields = set(record)
            led.tool_keys.setdefault(name, set()).update(fields)
            stamp, before = led.view_keys.get(ident, (-1, set()))
            carried = before if stamp == led.version else set()
            led.view_keys[ident] = (led.version, carried | fields)

    def already_shown(self, name: str, ident: str) -> bool:
        led = self.led
        stamp, have = led.view_keys.get(ident, (-1, set()))
        fields = led.tool_keys.get(name)
        if not fields or stamp != led.version:
            return False
        return fields <= have and ident not in led.gone

    def learn_shape(
        self, name: str, observation: Any
    ) -> dict[str, Any] | None:
        shapes = self.led.shapes
        if name in shapes:
            return shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [
            key for key, value in observation.items()
            if isinstance(value, list) and len(value) >= 2
            and all(isinstance(row, dict) for row in value)
        ]
        if not lists:
            return None
        shapes[name] = None
        got = self._ask_shape(name, observation, lists) or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        schema = self.led.schemas.get(name) or {}
        params = schema.get("properties") or {}
        count = self._key_in(got.get("count"), observation)
        if count and not self._whole(observation[count]):
            count = None
        shape = {
            "rows": rows,
            "count": count,
            "next": self._key_in(got.get("next"), observation),
            "next_arg": self._key_in(got.get("next_arg"), params),
            "price": self._key_in(got.get("price"), first),
            "stock": self._key_in(got.get("stock"), first),
        }
        shapes[name] = shape
        self.led.price_key = self.led.price_key or shape["price"]
        self.led.stock_key = self.led.stock_key or shape["stock"]
        self.led.trace.append({"kind": "shape", "tool": name, "shape": shape})
        return shape

    def _ask_shape(
        self, name: str, observation: dict[str, Any], lists: list[str]
    ) -> dict[str, Any] | None:
        sample = {
            key: (value[:3] if key in lists else value)
            for key, value in observation.items()
        }
        schema_text = to_json(self.led.schemas.get(name) or {})
        sample_text = to_json(clip(sample, 120))[:4000]
        prompt = (
            SHAPE_PROMPT + "\n\nTool used: " + name
            + "; parameters: " + schema_text
            + "\n\nResult (top rows only):\n" + sample_text
        )
        return self.hub.llm.ask_for_json(prompt, [LIGHT, HEAVY], 1500)

    @staticmethod
    def _key_in(key: Any, where: Any) -> str | None:
        if isinstance(key, str) and key in where:
            return key
        return None

    @staticmethod
    def _whole(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def paging_keys(
        self, name: str, observation: Any
    ) -> tuple[int, int, str, Any] | None:
        shape = self.learn_shape(name, observation)
        if not shape or not isinstance(observation, dict):
            return None
        if not (shape["count"] and shape["next"] and shape["next_arg"]):
            return None
        rows = observation.get(shape["rows"])
        count = observation.get(shape["count"])
        token = observation.get(shape["next"])
        if not isinstance(rows, list) or not rows or not self._whole(count):
            return None
        if token is not None:
            if isinstance(token, bool) or not isinstance(token, (str, int)):
                return None
        return count, len(rows), shape["next_arg"], token

    def sorted_by_price(self, observation: Any) -> bool:
        prices = [self.price(ident) for ident, _ in self.records(observation)]
        if len(prices) < 3 or None in prices:
            return False
        return all(low <= high for low, high in zip(prices, prices[1:]))

    def _agent_pages_itself(self, given: Any) -> bool:
        if isinstance(given, int) and not isinstance(given, bool):
            return given not in (None, 0, 1)
        return bool(str(given or "").strip())

    def _page_plan(self, name: str, args: dict[str, Any],
                   observation: Any, pick: Any) -> Any:
        if pick not in ("cheapest", "most", "least"):
            return None
        plan = self.paging_keys(name, observation)
        if not plan:
            return None
        count, shown, param, value = plan
        if value is None or self._agent_pages_itself(args.get(param)):
            return None
        if count <= shown:
            return None
        if pick == "cheapest" and self.sorted_by_price(observation):
            return None
        return plan

    def _too_many_pages(self, pages: int) -> bool:
        led = self.led
        if pages > 2 or led.pages_read + pages > AUTO_PAGES:
            return True
        return led.steps_left() - led.finish_reserve() - pages < 12

    def read_more_pages(self, name: str, args: dict[str, Any],
                        observation: Any) -> str:
        led = self.led
        led.paged_obs = []
        plan = self._page_plan(name, args, observation, led.brief.get("pick"))
        if plan is None:
            return ""
        count, shown, param, value = plan
        pages = -(-(count - shown) // shown)
        if self._too_many_pages(pages):
            return (
                f"\n[Kyo] This search yielded {count} listings; it showed "
                f"{shown} only, unsorted by price. Each fitting listing "
                "counts for the pick: narrow the search words (the "
                f"requirement's value) or request {param}={to_json(value)}."
            )
        texts = self._walk_pages(name, args, param, value, pages)
        led.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        if not texts:
            return ""
        return ("\n[Kyo paged through the rest of this search, one step "
                "each]\n" + "\n".join(texts))

    def _walk_pages(self, name: str, args: dict[str, Any], param: str,
                    value: Any, pages: int) -> list[str]:
        led = self.led
        texts: list[str] = []
        for _ in range(pages):
            faults: list[str] = []
            call = conform({**args, param: value}, led.schemas.get(name),
                           "args", faults)
            if faults:
                break
            reply = self.hub.store.call(name, call)
            if reply is None or led.finished:
                break
            led.pages_read += 1
            seen = reply.get("observation")
            self.observe(name, seen)
            led.paged_obs.append(seen)
            shown_text = self.hub.store.render(seen)
            led.read_cache[name + to_json(call)] = (
                led.version, led.step, shown_text)
            texts.append(f"{param}={to_json(value)}: {shown_text}")
            following = self.paging_keys(name, seen)
            if not following:
                break
            value = following[3]
            if value is None:
                break
        return texts

    def cheapest_in(self, observations: list[Any]) -> str:
        if self.led.brief.get("pick") != "cheapest":
            return ""
        if not self.led.price_key:
            return ""
        found = {ident for seen in observations
                 for ident, _ in self.records(seen)}
        rows = []
        for ident in found:
            if self.in_stock(ident) and self.price(ident) is not None:
                rows.append((self.price(ident), ident))
        if len(rows) < 2:
            return ""
        listed = ", ".join(f"{ident} ({cost:g})"
                           for cost, ident in sorted(rows)[:6])
        return "\n[Kyo] buyable rows, lowest prices first: " + listed

    def hold_wider(self) -> str:
        led = self.led
        if led.wider_held or led.brief.get("pick") != "cheapest":
            return ""
        # a cheapest order that reaches checkout with steps to spare can
        # still afford one capped search and a few opens
        if led.steps_left() < 8 + 2 * len(led.items):
            return ""
        led.wider_held = True
        spare = led.steps_left() - led.finish_reserve() - 3
        lines = []
        for ident in self.hub.judge.cart_listing_ids():
            if self.price(ident) is not None:
                lines.append(f"{ident} at {self.price(ident):g}")
        cart_part = f" ({'; '.join(lines)})" if lines else ""
        return (
            f"Not run yet (no step used). {led.steps_left()} steps remain "
            "and the pick is the cheapest. Store search reads listing "
            "titles, so searching for a requirement's value (usually present"
            " only in the attributes) finds few fitting listings, and "
            "results are not sorted by price. Before checkout, search a few "
            "more times for each item using other product names (its plain "
            "one-word name, synonyms, words from fitting titles seen) and "
            "consider only buyable rows cheaper than the cart's line"
            + cart_part + "; read the attributes of the cheapest of those "
            "rows, swap in what fits, then check out. If the search takes "
            "a maximum price, start with one search of the item's category "
            "alone (no search words) whose maximum sits just under the "
            "cart line's price, and open the cheapest rows from it that "
            f"are not opened yet. Spend at most {spare} steps on this "
            "before the checkout."
        )

    def fetch_facts(self, ids: list[str], keep: int) -> list[str]:
        led = self.led
        tool = self.facts_tool_name()
        target = self.takes_one(tool)
        if not tool or not target:
            return []
        judge = self.hub.judge
        unread = [ident for ident in ids
                  if ident in led.rows and ident not in led.full
                  and not judge.is_sold_out(ident)]
        texts: list[str] = []
        for ident in unread:
            if led.steps_left() <= keep or led.finished:
                break
            if led.facts_from.get(tool) is False:
                break
            text = self._read_facts(tool, target, ident)
            if text is None:
                break
            texts.append(text)
        if texts:
            led.trace.append({"kind": "fill_facts", "n": len(texts)})
        return texts

    def _read_facts(self, tool: str, target: str, ident: str) -> str | None:
        led = self.led
        faults: list[str] = []
        args = conform({target: ident}, led.schemas.get(tool), "args", faults)
        if faults:
            return None
        reply = self.hub.store.call(tool, args)
        if reply is None:
            return None
        seen = reply.get("observation")
        self.observe(tool, seen)
        self.remember_view(tool, seen)
        if isinstance(seen, dict) and self.hub.intake.row_id(seen):
            shows = bool(find_facts(seen))
            led.facts_from[tool] = led.facts_from.get(tool, False) or shows
        return f"{ident}: " + self.hub.store.render(seen)[:1500]

    def _kind_of(self, ident: str, cat_key: str) -> str:
        record = self.led.rows.get(ident) or {}
        return str(record.get(cat_key) or "").strip().lower()

    def _cheaper_pool(self, line: str, cart: list[str],
                      cat_key: str) -> list[str]:
        price = self.price(line)
        kind = self._kind_of(line, cat_key)
        if price is None or not kind:
            return []
        led = self.led
        pool = []
        for ident in led.seen_order:
            if ident in cart or ident in led.full:
                continue
            if not self.in_stock(ident):
                continue
            if self._kind_of(ident, cat_key) != kind:
                continue
            cost = self.price(ident)
            if (cost if cost is not None else float("inf")) < price:
                pool.append(ident)
        pool.sort(key=lambda ident: self.price(ident) or 0.0)
        return pool

    def cheaper_candidates(self) -> list[str]:
        led = self.led
        if led.brief.get("pick") != "cheapest":
            return []
        mark = led.cart_signature() + f"#{led.version}"
        cat_key = self.category_field()
        if mark in led.cheaper_marks or not cat_key:
            return []
        led.cheaper_marks.add(mark)
        cart = self.hub.judge.cart_listing_ids()
        picks: list[str] = []
        for line in cart:
            pool = self._cheaper_pool(line, cart, cat_key)
            fresh = [ident for ident in pool[:2] if ident not in picks]
            picks.extend(fresh)
        return self.fetch_facts(picks[:CHEAPER_OPENS],
                                led.finish_reserve() + 4)

    def observe(self, name: str, observation: Any) -> str:
        self.hub.intake.absorb(observation)
        lines = self.hub.judge.free_lines(observation)
        if not lines:
            return ""
        meaning = self.hub.judge.absorb_note(lines)
        if meaning:
            self.led.queued.append(f"(read on {name}) {meaning}")
        return meaning

    def _list_param(self, props: dict[str, Any]) -> str | None:
        for key, spec in props.items():
            if isinstance(spec, dict) and "array" in schema_types(spec):
                return key
        return None

    def _compare_usable(self, tool: str | None, target: str | None,
                        schema: dict[str, Any]) -> bool:
        if not tool or not target:
            return False
        if any(key != target for key in schema.get("required") or []):
            return False
        return not self.compare_is_bare()

    def _spread_picks(self, cap: int) -> list[str]:
        ranked = self.item_kinds()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            unopened = [ident for ident in ids if ident not in self.led.full]
            picks.extend(unopened[:share])
        return picks[:cap]

    def widen(self) -> None:
        led = self.led
        tool = led.roles.get("compare")
        schema = led.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = self._list_param(props)
        if not self._compare_usable(tool, target, schema):
            self.widen_one()
            return
        if led.steps_left() <= 12:
            return
        limit = props[target].get("maxItems")
        cap = limit if isinstance(limit, int) else 4
        picks = self._spread_picks(cap)
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = conform({target: picks}, schema, "args", faults)
        if faults:
            return
        before = len(led.full)
        reply = self.hub.store.call(tool, args)
        if reply is None:
            return
        seen = reply.get("observation")
        self.observe(tool, seen)
        self.remember_view(tool, seen)
        rows = self.records(seen)
        if rows:
            shows = any(find_facts(record) for _, record in rows)
            led.facts_from[tool] = led.facts_from.get(tool, False) or shows
        led.trace.append({"kind": "widen", "ids": picks,
                          "opened": len(led.full) - before})
        if rows and led.facts_from.get(tool) is False:
            self.widen_one()
            return
        if len(led.full) != before or len(picks) <= 2:
            return
        if led.steps_left() > 12:
            self._retry_pair(tool, target, schema, picks)

    def _retry_pair(self, tool: str, target: str, schema: dict[str, Any],
                    picks: list[str]) -> None:
        faults: list[str] = []
        args = conform({target: picks[:2]}, schema, "args", faults)
        if faults:
            return
        reply = self.hub.store.call(tool, args)
        if reply is not None:
            self.observe(tool, reply.get("observation"))

    def widen_one(self) -> None:
        tool = self.facts_tool_name()
        target = self.takes_one(tool)
        if not tool or not target:
            return
        led = self.led
        cheapest = led.brief.get("pick") == "cheapest"

        def cost(ident: str) -> float:
            value = self.price(ident)
            return value if value is not None else float("inf")

        lanes: list[list[str]] = []
        for ids in self.item_kinds():
            opened = sum(1 for ident in ids if ident in led.full)
            pool = [ident for ident in ids
                    if ident not in led.full and self.in_stock(ident)]
            if cheapest:
                pool.sort(key=cost)
            lanes.append(pool[:max(0, min(2, 8 - opened))])
        # alternate between item kinds: first pick of each, then the second
        queue = [lane[rank] for rank in range(2) for lane in lanes
                 if rank < len(lane)]
        self.fetch_facts(queue[:5], 22)


class QuestionDesk(Service):
    def should_ask(self) -> bool:
        led = self.led
        if led.asked or not led.roles.get("ask"):
            return False
        if led.free_messages() <= 0 or led.steps_left() <= 8:
            return False
        if len(led.full) >= 3 * len(led.items) + 1:
            return True
        return led.step >= max(8, led.max_steps // 4)

    def message_payload(self, text: str) -> dict[str, Any] | None:
        led = self.led
        schema = led.schemas.get(led.roles.get("ask") or "") or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        free_text: list[str] = []
        for key, spec in props.items():
            if not isinstance(spec, dict):
                continue
            if "string" in schema_types(spec) and "enum" not in spec:
                free_text.append(key)
        target = None
        for key in required:
            if key in free_text:
                target = key
                break
        if not target and free_text:
            target = free_text[0]
        if not target:
            return None
        payload: dict[str, Any] = {target: text}
        for key in required:
            if key in payload:
                continue
            options = (props.get(key) or {}).get("enum")
            if not isinstance(options, list) or not options:
                return None
            payload[key] = options[0]
        faults: list[str] = []
        shaped = conform(payload, schema, "args", faults)
        if faults:
            return None
        return shaped

    def interview(self) -> str:
        led = self.led
        led.asked = True
        count = led.free_messages()
        if count <= 0:
            return ""
        self.hub.catalogue.widen()
        table = self.hub.catalogue.property_table()
        told: list[str] = []
        prompt = self._plan_prompt(count, table)
        plan = self.hub.llm.ask_for_json(prompt, [HEAVY, LIGHT], 1800) or {}
        ordered = self._picks_from(plan)
        led.trace.append({"kind": "plan", "ask": clip(plan.get("ask"), 120)})
        if len(led.items) == 2:
            sent = self._take_turns(ordered, told)
            if told:
                led.version += 1
            return self._interview_report(sent, told) if sent else ""
        # two items with the same name would get the same question twice
        batch: list[tuple[int, str, str]] = []
        for number, label in ordered:
            text = self.question_text(number, label)
            if len(batch) == count or any(
                spaced_words(text) == spaced_words(q) for _, _, q in batch
            ):
                continue
            batch.append((number, label, text))
        sent = self.send(batch)
        if sent:
            self.apply_answers(self.listen(sent), told)
        if not sent:
            return ""
        if told:
            led.version += 1
        return self._interview_report(sent, told)

    def _take_turns(
        self, ordered: list[tuple[int, str]], told: list[str]
    ) -> list[tuple[int, str, str]]:
        # with two products the plan's picks are asked one at a time, the
        # products taking turns; a product that has revealed a preference
        # steps aside while the other has not
        queues: dict[int, list[str]] = {}
        for number, label in ordered:
            queues.setdefault(number, []).append(label)
        sent: list[tuple[int, str, str]] = []
        answered: set[int] = set()
        turn = 0
        while self.led.free_messages() > 0:
            ready = [n for n in queues if queues[n]]
            if not ready:
                break
            pool = [n for n in ready if n not in answered] or ready
            number = pool[turn % len(pool)]
            turn += 1
            label = queues[number].pop(0)
            text = self.question_text(number, label)
            if any(
                spaced_words(text) == spaced_words(q) for _, q, _ in sent
            ):
                continue
            batch = self.send([(number, label, text)])
            if not batch:
                break
            sent += batch
            heard = self.listen(batch, (LIGHT, HEAVY))
            for entry in as_rows(heard.get("needs")):
                if isinstance(entry, dict) and entry.get("need"):
                    answered.add(self._item_number(entry.get("item")))
            self.apply_answers(heard, told)
        return sent

    def _plan_prompt(self, count: int, table: list[Any]) -> str:
        led = self.led
        settled: list[str] = []
        for number, item in enumerate(led.items, 1):
            fixed = "; ".join(item.musts) or "nothing"
            settled.append(
                f"item {number}: {item.name} - already fixed by the "
                f"request: {fixed}"
            )
        listing: list[str] = []
        current = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != current:
                current = label
                listing.append(f"category {label!r}:")
            listing.append(
                f"  {index}. {name} (seen on {carried}): "
                + " | ".join(values)
            )
        prompt = (
            PLAN_PROMPT.replace("COUNT", str(count + 1))
            + "\n\nREQUEST:\n" + led.request
            + "\n\nITEMS:\n" + "\n".join(settled)
        )
        first = [
            str(x) for x in as_rows(led.brief.get("ask_first"))
            if str(x).strip()
        ]
        if first:
            prompt += (
                "\n\nASK FIRST (the request wants the shopper asked "
                "about these):\n" + "\n".join(first)
            )
        keys = "\n".join(listing) or "(no listing opened in full so far)"
        return prompt + "\n\nATTRIBUTE KEYS SEEN:\n" + keys

    def _item_number(self, value: Any) -> int:
        return min(max(1, as_count(value) or 1), len(self.led.items))

    def _picks_from(self, plan: dict[str, Any]) -> list[tuple[int, str]]:
        ordered: list[tuple[int, str]] = []
        for pick in as_rows(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = self._item_number(pick.get("item"))
            # keep only the first property a label names
            head = re.split(
                r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or "")
            )[0]
            label = head.strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            key = spaced_words(label)
            repeated = any(
                other_number == number and spaced_words(other) == key
                for other_number, other in ordered
            )
            if not repeated:
                ordered.append((number, label))
        return self._fresh_labels_first(ordered)

    @staticmethod
    def _fresh_labels_first(
        ordered: list[tuple[int, str]]
    ) -> list[tuple[int, str]]:
        # a property already put to one item rarely brings anything new
        # when put to another, so such repeats wait behind fresh ones
        fresh: list[tuple[int, str]] = []
        later: list[tuple[int, str]] = []
        used: set[str] = set()
        for number, label in ordered:
            key = spaced_words(label)
            (later if key in used else fresh).append((number, label))
            used.add(key)
        return fresh + later

    def _interview_report(
        self, sent: list[tuple[int, str, str]], told: list[str]
    ) -> str:
        led = self.led
        exchanges = "\n".join(f"Q: {q}\nA: {a}" for _, q, a in sent)
        report = (
            "Kyo's interviewer put questions to the shopper "
            "(each cost a step):\n" + exchanges
            + "\nFirm requirements that these answers add: "
            + ("; ".join(told) if told else "none")
            + ". Hold every pick to them, searching again with the new "
            "values when needed."
        )
        pointed = [
            need for _, need in led.refs if any(need in t for t in told)
        ]
        if pointed:
            report += (
                " These get their value from a second listing: "
                + " | ".join(pointed) + ". Locate that listing, open its "
                "full page, then open every further listing that record "
                "directs the reader to; "
                "the final page gives the value, and the order counts only "
                "once all of them are opened."
            )
        if led.queued:
            report += (
                "\n[later notes received] "
                + " | ".join(led.queued)
            )
            led.queued = []
        return report

    def question_text(self, number: int, label: str) -> str:
        name = self.led.items[number - 1].name
        return f"What {label} would you like for the {name}?"

    def send(
        self, questions: list[tuple[int, str, str]], floor: int = 6
    ) -> list[tuple[int, str, str]]:
        led = self.led
        tool = led.roles["ask"]
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if not self._may_send(floor):
                break
            payload = self.message_payload(text)
            if not payload:
                break
            reply = self.hub.store.call(tool, payload)
            if reply is None:
                break
            led.msg_used += 1
            answer = reply.get("observation")
            words = self.hub.judge.shopper_said(answer)
            sent.append((number, text, words))
            self._log_exchange(text, words)
            if isinstance(answer, dict) and answer.get("error"):
                break
        return sent

    def _may_send(self, floor: int) -> bool:
        led = self.led
        if led.finished or led.steps_left() <= floor:
            return False
        return led.msg_used < led.msg_limit

    def _log_exchange(self, text: str, words: str) -> None:
        led = self.led
        led.done_log.append(
            f"step {led.step}: sent the shopper {text!r} -> answer "
            f"{words[:300]!r} (cart at that point: "
            f"{led.cart_signature()})"
        )
        led.done_steps.append(led.step)
        led.trace.append({"kind": "ask", "q": text, "a": words[:300]})

    def apply_answers(self, heard: dict[str, Any], told: list[str]) -> None:
        led = self.led
        many = len(led.items) > 1
        for entry in as_rows(heard.get("needs")):
            self._take_need(entry, told, many)
        for entry in as_rows(heard.get("drop")):
            self._take_drop(entry, told)
        for entry in as_rows(heard.get("free")):
            if not isinstance(entry, dict) or not entry.get("property"):
                continue
            number = self._item_number(entry.get("item"))
            led.items[number - 1].free.append(str(entry["property"]))
        budget = heard.get("budget")
        if not isinstance(budget, (int, float)) or isinstance(budget, bool):
            return
        if budget > 0:
            led.budget = budget

    def _take_need(self, entry: Any, told: list[str], many: bool) -> None:
        if not isinstance(entry, dict):
            return
        need = str(entry.get("need") or "").strip()
        if not need:
            return
        led = self.led
        number = self._item_number(entry.get("item"))
        item = led.items[number - 1]
        if need not in item.told:
            item.told.append(need)
            told.append(f"item {number}: {need}" if many else need)
        if entry.get("points") is True and (number, need) not in led.refs:
            led.refs.append((number, need))

    def _take_drop(self, entry: Any, told: list[str]) -> None:
        if not isinstance(entry, dict):
            return
        if not str(entry.get("property") or "").strip():
            return
        item = self.led.items[self._item_number(entry.get("item")) - 1]
        gone = spaced_words(str(entry["property"]).split(":")[0])
        hits: list[str] = []
        for must in item.musts + item.told:
            if must in item.dropped:
                continue
            if spaced_words(must.split(":")[0]) == gone:
                hits.append(must)
        item.dropped.extend(hits)
        told.extend(f"dropped {must}" for must in hits)

    def listen(
        self,
        sent: list[tuple[int, str, str]],
        models: tuple[str, ...] = (HEAVY, LIGHT),
    ) -> dict[str, Any]:
        items = "\n".join(
            f"item {n}: {it.name}" for n, it in enumerate(self.led.items, 1)
        )
        exchanges = "\n".join(
            f"{i}. (on item {n}) Q: {q}\nA: {a}"
            for i, (n, q, a) in enumerate(sent, 1)
        )
        prompt = (
            HEAR_PROMPT + "\n\nITEMS:\n" + items
            + "\n\nQUESTIONS AND ANSWERS:\n" + exchanges
        )
        return self.hub.llm.ask_for_json(prompt, list(models), 1500) or {}

    def run_gate(self, stage: str) -> tuple[bool, str]:
        led = self.led
        steps = [p for p in led.process if p["when"] == stage]
        if not steps or led.finished or not led.gate_pending(stage):
            return True, ""
        ending = {led.roles.get("buy"), led.roles.get("refuse")} - {None}
        tools = self._gate_tools(ending)
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            # the light LLM leads until a nudge, then the heavy one does
            models = [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY]
            prompt = self._gate_prompt(stage, steps, tools)
            got = self.hub.llm.ask_for_json(prompt, models, 2000)
            if got is None:
                break
            told: list[str] = []
            hold = ""
            if said:
                self.apply_answers({"needs": got.get("needs")}, told)
                hold = str(got.get("hold") or "").strip()
            call = got.get("call")
            if not isinstance(call, dict):
                call = None
            led.trace.append({
                "kind": "gate",
                "stage": stage,
                "call": clip(call, 200),
                "steps": clip(got.get("steps"), 120),
                "hold": hold[:200],
                "told": told,
            })
            if told or hold:
                led.version += 1
                return False, self._task_changed(said, told, hold)
            unproven = self._unproven(got, stage, steps)
            if (not call and unproven and not said and not nudged
                    and self._nudge_room()):
                nudged = True
                self._nudge(unproven)
                continue
            if not call:
                led.gate_ok.add(
                    (stage, led.cart_signature() + f"#{led.version}")
                )
                break
            name = str(call.get("tool") or "")
            if name not in led.schemas or name in ending:
                break
            if led.steps_left() < 2:
                break
            asking = name == led.roles.get("ask")
            if asking and led.msg_used >= led.msg_limit:
                break
            raw = call.get("args")
            faults: list[str] = []
            args = conform(
                raw if isinstance(raw, dict) else {},
                led.schemas[name], "args", faults,
            )
            if faults:
                break
            if asking:
                text = next(
                    (v for v in args.values() if isinstance(v, str)), ""
                )
                batch = self.send([(1, "", text)], 1)
                if not batch:
                    break
                led.gate_msgs += 1
                said.append(f"Q: {text}\nA: {batch[0][2]}")
                continue
            done = self._gate_store_call(name, args)
            if done is None:
                break
            said.append(done)
        return True, self._steps_done(said) if said else ""

    def _gate_tools(self, ending: set[Any]) -> list[str]:
        lines: list[str] = []
        for tool in self.led.tools:
            spec = tool["function"]
            if spec["name"] in ending:
                continue
            lines.append(
                f"- {spec['name']}: {spec.get('description', '')} | "
                f"parameters: {to_json(spec.get('parameters') or {})}"
            )
        return lines

    def _gate_prompt(
        self, stage: str, steps: list[dict[str, Any]], tools: list[str]
    ) -> str:
        led = self.led
        ordering = stage == "order"
        goal = "place the order" if ordering else "end without a purchase"
        due = "\n".join(f"- {p['step']}" for p in steps)
        ledger = self.hub.intake.summary()
        cart = self.hub.judge.cart_summary()
        record = "\n".join(
            f"{i}. {line}" for i, line in enumerate(led.done_log, 1)
        ) or "(empty so far)"
        notes = ""
        if led.notes:
            notes = "\n\nSTORE NOTES SEEN:\n" + "\n".join(led.notes[-4:])
        final = "order" if ordering else "ending"
        budget_line = (
            f"\n\nMESSAGES LEFT: {max(0, led.msg_limit - led.msg_used)}"
            f"   STEPS LEFT: {led.steps_left()} (one of them goes to the "
            f"{final} itself)"
        )
        return (
            GATE_PROMPT.replace("STAGE", goal)
            + "\n\nREQUEST:\n" + led.request
            + "\n\nSTEPS DUE NOW:\n" + due
            + "\n\nLEDGER:\n" + ledger
            + "\n\nCART NOW: " + cart
            + "\n\nRECORD (every message and step-using call so far):\n"
            + record + notes + budget_line
            + "\n\nTOOLS:\n" + "\n".join(tools)
        )

    def _unproven(
        self, got: dict[str, Any], stage: str, steps: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        led = self.led
        # at the order, a step needing a message must be shown by a line
        # written after the cart last changed
        talk = stage == "order" and any(p["messages"] > 0 for p in steps)
        proof: set[int] = set()
        for number in range(1, len(led.done_log) + 1):
            if not talk:
                proof.add(number)
            elif (number <= len(led.done_steps)
                    and led.done_steps[number - 1] > led.cart_at):
                proof.add(number)
        missing: list[dict[str, Any]] = []
        for entry in as_rows(got.get("steps")):
            if not isinstance(entry, dict):
                continue
            proven = entry.get("done") is True and any(
                (as_count(n) or 0) in proof
                for n in as_rows(entry.get("lines"))
            )
            if not proven:
                missing.append(entry)
        return missing

    def _nudge_room(self) -> bool:
        led = self.led
        if not led.roles.get("ask") or led.msg_used >= led.msg_limit:
            return False
        return led.steps_left() >= 2

    def _nudge(self, unproven: list[dict[str, Any]]) -> None:
        led = self.led
        led.trace.append(
            {"kind": "gate_unproven", "steps": clip(unproven, 120)}
        )
        named = "; ".join(str(e.get("step") or "") for e in unproven)
        led.done_log.append(
            "(Kyo: no record line after the cart's last change at step "
            f"{led.cart_at} shows " + named[:200]
            + f" as done, while {led.msg_limit - led.msg_used} message(s) "
            f"and {led.steps_left()} steps remain: give the call that "
            "does it)"
        )
        led.done_steps.append(-1)

    def _gate_store_call(self, name: str, args: Any) -> str | None:
        led = self.led
        reply = self.hub.store.call(name, args)
        if reply is None:
            return None
        seen = reply.get("observation")
        self.hub.intake.absorb(seen)
        led.done_log.append(
            f"step {led.step}: {name} {to_json(args)} -> "
            f"{self.hub.store.render(seen)[:300]} "
            f"(cart at that point: {led.cart_signature()})"
        )
        led.done_steps.append(led.step)
        shown = self.hub.store.render(seen)[:400]
        return f"{name} {to_json(args)} -> {shown}"

    def _steps_done(self, said: list[str]) -> str:
        return (
            "Kyo first completed the steps the request asks for here:\n"
            + "\n".join(said)
        )

    def _task_changed(
        self, said: list[str], told: list[str], hold: str
    ) -> str:
        changes = told + ([hold] if hold else [])
        prefix = self._steps_done(said) + "\n" if said else ""
        return (
            prefix + "The shopper's answer alters the task: "
            + "; ".join(changes)
            + ". Adjust the cart, then check out once more."
        )


class StoreService(Service):
    DECLINE_NOTE = (
        "Sorry, nothing in the store meets all of your requirements, "
        "so I am not buying anything."
    )

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        led = self.led
        payload = dict(led.binding)
        payload["step"] = led.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                return self._accept(name, args, result.data)
            if not self._retry_after(result.error or {}, payload):
                return None
        return None

    def _accept(
        self, name: str, args: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        led = self.led
        step = data.get("step")
        led.step = step if isinstance(step, int) else led.step + 1
        if data.get("finished"):
            led.finished = True
        seen = data.get("observation")
        failed = isinstance(seen, dict) and seen.get("error")
        if name in self.hub.catalogue.page_tool_set() and not failed:
            self._mark_opened(args)
        return data

    def _mark_opened(self, args: dict[str, Any]) -> None:
        for value in args.values():
            for part in value if isinstance(value, list) else [value]:
                if isinstance(part, str):
                    self.led.opened.add(part.strip())

    def _retry_after(
        self, error: dict[str, Any], payload: dict[str, Any]
    ) -> bool:
        status = error.get("status")
        body = str(error.get("body") or "")
        self.led.trace.append(
            {"kind": "env_fail", "status": status, "body": body[:200]}
        )
        if status == 409:
            return self._resync(body, payload)
        if status == 402:
            self.led.broke = True
        return status not in (400, 401, 402, 403, 404, 422)

    def _resync(self, body: str, payload: dict[str, Any]) -> bool:
        led = self.led
        low = body.lower()
        if "finish" in low or "quarantin" in low or "finaliz" in low:
            led.finished = True
            return False
        found = re.search(r"expected (\d+)", body)
        if found:
            led.step = int(found.group(1)) - 1
        elif "reuse" in low or "different action" in low:
            led.step += 1
        else:
            return False
        payload["step"] = led.step + 1
        return True

    def _ends_with(
        self, role: str, brief_key: str, name: str, args: dict[str, Any]
    ) -> bool:
        if name != self.led.roles.get(role):
            return False
        fixed = self.led.brief.get(brief_key)
        if not isinstance(fixed, dict):
            return True
        return all(args.get(k) == v for k, v in fixed.items())

    def is_checkout(self, name: str, args: dict[str, Any]) -> bool:
        return self._ends_with("buy", "buy_args", name, args)

    def is_decline(self, name: str, args: dict[str, Any]) -> bool:
        return self._ends_with("refuse", "refuse_args", name, args)

    def tool_specs(self) -> list[dict[str, Any]]:
        led = self.led
        roles = led.roles
        ends = {roles.get("buy"), roles.get("refuse")} - {None}
        keep = self._kept_tools(led.steps_left(), ends)
        hidden = roles.get("ask")
        tools = [t for t in led.tools if t["function"]["name"] != hidden]
        if keep is None or not ends:
            return tools
        narrowed = [t for t in tools if t["function"]["name"] in keep]
        return narrowed or tools

    def _kept_tools(self, left: int, ends: set[Any]) -> set[Any] | None:
        led = self.led
        roles = led.roles
        reserve = led.finish_reserve()
        if left <= reserve:
            return set(ends)
        cart_side = {roles.get("cart"), roles.get("cart_view")}
        if left <= reserve + 3:
            return ends | cart_side
        if left <= len(led.items) + 5 and led.buys == 0:
            return ends | cart_side | {roles.get("offer")}
        return None

    def quiet_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.led.schemas.get(name) or {}
        required = schema.get("required") or []
        out = dict(args)
        for key, spec in (schema.get("properties") or {}).items():
            if not self._plain_text(spec):
                continue
            value = out.get(key)
            if value is None and key not in required:
                continue
            if self._should_mask(str(value or "")):
                limit = spec.get("maxLength", 200)
                out[key] = self.DECLINE_NOTE[:limit]
        return out

    @staticmethod
    def _plain_text(spec: Any) -> bool:
        return (
            isinstance(spec, dict)
            and "string" in schema_types(spec)
            and "enum" not in spec
        )

    def _should_mask(self, text: str) -> bool:
        if not text.strip():
            return True
        return self.led.brief.get("quiet_decline") is True

    def fix_ids(
        self, name: str, args: dict[str, Any]
    ) -> tuple[dict[str, Any], str]:
        key = self._id_argument(name)
        if key is None:
            return args, ""
        out = dict(args)
        if key not in args:
            return out, ""
        given = args[key]
        many = isinstance(given, list)
        fixed = []
        for value in given if many else [given]:
            value, problem = self._listing_id(value)
            if problem:
                return args, problem
            fixed.append(value)
        out[key] = fixed if many else fixed[0]
        return out, ""

    def _id_argument(self, name: str) -> str | None:
        led = self.led
        roles = led.roles
        id_tools = {roles.get(r) for r in ("cart", "open", "offer", "compare")}
        if name not in id_tools - {None} or not led.id_key:
            return None
        props = (led.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if self._free_slot(v)]
        if len(free) != 1:
            return None
        if name != roles.get("cart") and (name, free[0]) not in led.id_args:
            return None
        return free[0]

    @staticmethod
    def _free_slot(spec: Any) -> bool:
        if not isinstance(spec, dict) or "enum" in spec:
            return False
        kinds = schema_types(spec)
        return "string" in kinds or "array" in kinds

    def _listing_id(self, value: Any) -> tuple[Any, str]:
        led = self.led
        if isinstance(value, str) and value.strip() in led.rows:
            return value.strip(), ""
        if not isinstance(value, str) or looks_like_id(value, True):
            return value, ""
        if len(value.strip()) < 8:
            return value, ""
        low = value.strip().lower()
        hits = [i for i, row in led.rows.items() if self._titled(row, low)]
        if len(hits) == 1:
            return hits[0], ""
        if hits:
            return value, (
                f"several listings match the text {value!r} "
                f"({', '.join(hits[:6])}); name one listing id"
            )
        return value, (
            f"{value!r} is no listing id; use the id shown in the result "
            f"rows (their {led.id_key} field)"
        )

    def _titled(self, row: dict[str, Any], low: str) -> bool:
        for key, text in row.items():
            if key == self.led.id_key or not isinstance(text, str):
                continue
            if text.strip().lower().startswith(low):
                return True
        return False

    def render(self, observation: Any) -> str:
        text = to_json(clip(observation, SHOW_CHARS))
        if len(text) <= OBS_LIMIT:
            return text
        return text[:OBS_LIMIT] + "...(cut)"


class CartDesk(Service):
    def open_listings(self, rows: list[str]) -> bool:
        led = self.led
        plan = self._open_plan(rows)
        if plan is None or led.steps_left() < len(plan) + 5:
            return False
        before = len(led.full)
        for name, args in plan:
            faults: list[str] = []
            args = conform(args, led.schemas.get(name), "args", faults)
            if faults:
                return False
            reply = self.hub.store.call(name, args)
            if reply is None or led.finished:
                break
            self.hub.catalogue.observe(name, reply.get("observation"))
        led.trace.append({
            "kind": "audit_open",
            "rows": rows,
            "opened": len(led.full) - before,
        })
        return len(led.full) > before

    def _open_plan(
        self, rows: list[str]
    ) -> list[tuple[str, dict[str, Any]]] | None:
        catalogue = self.hub.catalogue
        many = self.compare_tool()
        joiner = self.led.roles.get("compare")
        single = catalogue.facts_tool_name()
        target = catalogue.takes_one(single)
        if many and joiner and not catalogue.compare_is_bare():
            return [(joiner, {many: rows[:self.compare_slots()]})]
        if not single or not target:
            return None
        return [(single, {target: ident}) for ident in rows[:2]]

    @staticmethod
    def _text_field(spec: Any) -> bool:
        return (
            isinstance(spec, dict)
            and "string" in schema_types(spec)
            and "enum" not in spec
        )

    @staticmethod
    def _int_field(spec: Any) -> bool:
        return isinstance(spec, dict) and "integer" in schema_types(spec)

    @staticmethod
    def _mode_field(spec: Any) -> bool:
        if not isinstance(spec, dict):
            return False
        options = spec.get("enum")
        return isinstance(options, list) and len(options) >= 2

    @staticmethod
    def _line_fields(spec: Any) -> dict[str, Any] | None:
        if not isinstance(spec, dict) or "array" not in schema_types(spec):
            return None
        shape = spec.get("items")
        if not isinstance(shape, dict):
            return None
        inner = shape.get("properties")
        return inner if isinstance(inner, dict) else None

    def cart_actions(
        self, lines: dict[str, int], drop: list[str]
    ) -> list[dict[str, Any]] | None:
        led = self.led
        name = led.roles.get("cart") or ""
        schema = led.schemas.get(name) or {}
        props = schema.get("properties") or {}
        for key, spec in props.items():
            inner = self._line_fields(spec)
            if inner is not None:
                return self._list_form(key, inner, lines)
        target = self._id_slot(name, schema, props)
        if not target:
            return None
        count = next((k for k, s in props.items() if self._int_field(s)), None)
        mode = next((k for k, s in props.items() if self._mode_field(s)), None)
        actions: list[dict[str, Any]] = []
        for ident in drop:
            action = self._removal(name, props, target, count, mode, ident)
            if action is None:
                return None
            actions.append(action)
        in_cart = self.hub.judge.cart_listing_ids()
        for ident, want in lines.items():
            have = led.cart_lines.get(ident, 0) if ident in in_cart else 0
            if have == want:
                continue
            if have and not self.sets_quantity(name):
                return None
            action = {target: ident}
            if mode:
                action[mode] = props[mode]["enum"][0]
            if count:
                action[count] = want
            actions.append(action)
        return actions

    def _list_form(
        self, key: str, inner: dict[str, Any], lines: dict[str, int]
    ) -> list[dict[str, Any]] | None:
        id_key = next(
            (k for k, s in inner.items() if self._text_field(s)), None
        )
        count_key = next(
            (k for k, s in inner.items() if self._int_field(s)), None
        )
        if not id_key:
            return None
        entries = []
        for ident, number in lines.items():
            entry = {id_key: ident}
            if count_key:
                entry[count_key] = number
            entries.append(entry)
        return [{key: entries}]

    def _id_slot(
        self, name: str, schema: dict[str, Any], props: dict[str, Any]
    ) -> str | None:
        texts = [k for k, s in props.items() if self._text_field(s)]
        seen = next(
            (k for t, k in self.led.id_args if t == name and k in texts),
            None,
        )
        if seen:
            return seen
        required = schema.get("required") or []
        listed = next((k for k in required if k in texts), None)
        if listed:
            return listed
        return texts[0] if len(texts) == 1 else None

    def _removal(
        self,
        name: str,
        props: dict[str, Any],
        target: str,
        count: str | None,
        mode: str | None,
        ident: str,
    ) -> dict[str, Any] | None:
        have = self.led.cart_lines.get(ident, 1)
        if mode:
            action = {target: ident, mode: props[mode]["enum"][1]}
            if count:
                action[count] = have
            return action
        if not count:
            return None
        if self.sets_quantity(name):
            return {target: ident, count: 0}
        floor = props[count].get("minimum")
        if isinstance(floor, int) and floor <= -have:
            return {target: ident, count: -have}
        return None

    def _line_kind(self, ident: str, cat_key: str | None) -> str:
        if not cat_key:
            return ident
        row = self.led.rows.get(ident) or {}
        return str(row.get(cat_key) or "").strip().lower()

    def _lead_plan(self, cat_key: str | None) -> dict[str, int]:
        led = self.led
        judge = self.hub.judge
        plan: dict[str, int] = {}
        for number, ids in led.lead_items.items():
            index = as_count(number)
            pick = next((i for i in ids if not judge.is_sold_out(i)), None)
            if not pick or index is None:
                continue
            if not 1 <= index <= len(led.items):
                continue
            taken = {self._line_kind(p, cat_key) for p in plan}
            if self._line_kind(pick, cat_key) not in taken:
                plan[pick] = led.items[index - 1].qty
        return plan

    def fill_cart(self) -> str:
        led = self.led
        judge = self.hub.judge
        cart, buy = led.roles.get("cart"), led.roles.get("buy")
        if led.completed or not led.confirmed:
            return ""
        if judge.is_sold_out(led.confirmed) or not cart or not buy:
            return ""
        led.completed = True
        cat_key = self.hub.catalogue.category_field()
        plan = self._lead_plan(cat_key)
        if led.confirmed not in plan:
            return ""
        planned = {self._line_kind(p, cat_key) for p in plan}
        keep: dict[str, int] = {}
        drop: list[str] = []
        current = judge.cart_listing_ids()
        for ident in current:
            if ident in plan:
                continue
            if (not judge.is_sold_out(ident)
                    and self._line_kind(ident, cat_key) not in planned):
                keep[ident] = led.cart_lines.get(ident, 1)
        for ident in current:
            if ident not in plan and ident not in keep:
                drop.append(ident)
        kept_kinds = {self._line_kind(i, cat_key) for i in keep}
        if len(plan) + len(kept_kinds) < len(led.items):
            return ""
        lines = {**keep, **plan}
        actions = self.cart_actions(lines, drop)
        if not actions or len(actions) + 1 > led.steps_left():
            return ""
        self.hub.catalogue.fetch_facts(list(lines), len(actions) + 3)
        if not self._cart_passes(lines):
            return ""
        done = self._set_cart(cart, actions)
        if done is None:
            return ""
        faults: list[str] = []
        args = conform(
            dict(led.brief.get("buy_args") or {}),
            led.schemas.get(buy), "args", faults,
        )
        if faults:
            return ""
        replies = self._check_out(buy, args, lines, drop)
        if not replies:
            return ""
        return (
            "Not run (Kyo acted on its own): the independent check passed "
            + led.confirmed + ", so Kyo carted lines ("
            + "; ".join(done) + ") and checked out. Store wrote: "
            + " | then: ".join(replies)
        )

    def _cart_passes(self, lines: dict[str, int]) -> bool:
        judge = self.hub.judge
        judge.review_cart(list(lines), SPEND_HARD)
        return bool(self.led.verdicts.get(judge.verdict_key(list(lines))))

    def _set_cart(
        self, cart: str, actions: list[dict[str, Any]]
    ) -> list[str] | None:
        led = self.led
        done: list[str] = []
        for args in actions:
            faults: list[str] = []
            args = conform(args, led.schemas.get(cart), "args", faults)
            if faults:
                return None
            reply = self.hub.store.call(cart, args)
            if reply is None or led.finished:
                return None
            self.record_cart(cart, args)
            self.hub.catalogue.observe(cart, reply.get("observation"))
            done.append(to_json(args))
        return done

    def _still_fine(self, lines: dict[str, int]) -> bool:
        led = self.led
        self.hub.judge.refresh_offers()
        if not self._cart_passes(lines):
            return False
        if led.steps_left() < 1 or led.finished:
            return False
        return not any(self.hub.judge.is_sold_out(i) for i in lines)

    def _check_out(
        self,
        buy: str,
        args: dict[str, Any],
        lines: dict[str, int],
        drop: list[str],
    ) -> list[str]:
        led = self.led
        store = self.hub.store
        replies: list[str] = []
        for attempt in range(2):
            if attempt and not self._still_fine(lines):
                break
            go, said = self.hub.questions.run_gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = store.call(buy, args)
            led.trace.append({
                "kind": "completed",
                "lines": lines,
                "drop": drop,
                "reply": clip(reply, 200),
            })
            if reply is None:
                break
            led.buys += 1
            if not reply.get("finished"):
                led.refused_at = led.step
            seen = reply.get("observation")
            self.hub.catalogue.observe(buy, seen)
            replies.append(store.render(seen))
            if led.finished or reply.get("finished"):
                break
        return replies

    def sets_quantity(self, name: str) -> bool:
        schema = self.led.schemas.get(name, {})
        specs = [
            s for s in (schema.get("properties") or {}).values()
            if isinstance(s, dict)
        ]
        if any(self._mode_field(s) for s in specs):
            return False
        return any(
            "integer" in schema_types(s) and s.get("minimum") == 0
            for s in specs
        )

    @staticmethod
    def _loose_number(value: Any) -> bool:
        return as_count(value) is not None and not isinstance(value, str)

    def record_cart(self, name: str, args: dict[str, Any]) -> None:
        led = self.led
        led.cart_touched = True
        led.cart_at = led.step
        spec = led.schemas.get(name, {}).get("properties") or {}
        flags = [self._takes_off(spec, k, v) for k, v in args.items()]
        removal = any(flags)
        loose = next(
            (as_count(v) for v in args.values() if self._loose_number(v)),
            None,
        )
        lines, whole = self._named_lines(spec, args, loose)
        if whole:
            led.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.sets_quantity(name)
        for ident, count in lines:
            have = led.cart_lines.get(ident, 0)
            if removal:
                have = have - count if count else 0
            elif sets:
                have = 1 if count is None else count
            else:
                have += 1 if count is None else count
            if have > 0:
                led.cart_lines[ident] = have
            else:
                led.cart_lines.pop(ident, None)

    @staticmethod
    def _takes_off(spec: dict[str, Any], key: str, value: Any) -> bool:
        options = (spec.get(key) or {}).get("enum")
        if not isinstance(options, list) or len(options) < 2:
            return False
        return value in options[1:]

    def _named_lines(
        self, spec: dict[str, Any], args: dict[str, Any], loose: int | None
    ) -> tuple[list[tuple[str, int | None]], bool]:
        rows = self.led.rows
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, str):
                if value in rows:
                    lines.append((value, loose))
                continue
            if not isinstance(value, list):
                continue
            shape = (spec.get(key) or {}).get("items") or {}
            if not value and isinstance(shape, dict):
                if shape.get("type") == "object" or "properties" in shape:
                    whole = True
            for line in value:
                if isinstance(line, dict):
                    whole = True
                    entry = self._dict_line(line)
                    if entry is not None:
                        lines.append(entry)
                elif isinstance(line, str) and line in rows:
                    lines.append((line, loose))
        return lines, whole

    def _dict_line(self, line: dict[str, Any]) -> tuple[str, int] | None:
        rows = self.led.rows
        ident = next(
            (v for v in line.values() if isinstance(v, str) and v in rows),
            None,
        )
        count = next(
            (as_count(v) for v in line.values() if self._loose_number(v)),
            1,
        )
        if not ident:
            return None
        return ident, count

    def _compare_schema(self) -> dict[str, Any]:
        compare = self.led.roles.get("compare") or ""
        return self.led.schemas.get(compare) or {}

    def compare_tool(self) -> str | None:
        schema = self._compare_schema()
        props = schema.get("properties") or {}
        target = None
        for key, spec in props.items():
            if isinstance(spec, dict) and "array" in schema_types(spec):
                target = key
                break
        if not target:
            return None
        for key in schema.get("required") or []:
            if key != target:
                return None
        return target

    def compare_slots(self) -> int:
        led = self.led
        props = self._compare_schema().get("properties") or {}
        spec = props.get(self.compare_tool() or "") or {}
        limit = spec.get("maxItems")
        if isinstance(limit, int):
            return limit
        if not led.compare_bad:
            return max(4, led.compare_ok)
        return max(2, min(led.compare_bad - 1, max(led.compare_ok, 2)))

    def merge_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        led = self.led
        catalogue = self.hub.catalogue
        opener = led.roles.get("open")
        joiner = led.roles.get("compare")
        target = catalogue.takes_one(opener)
        many = self.compare_tool()
        if not (opener and joiner and target and many) or opener == joiner:
            return calls
        if catalogue.compare_is_bare():
            return calls
        picks = self._single_opens(calls, opener, target)
        picks = picks[:self.compare_slots()]
        if len(picks) < 2:
            return calls
        chosen = [ident for _, ident in picks]
        folded = {index for index, _ in picks}
        first = picks[0][0]
        out: list[dict[str, Any]] = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": joiner,
                        "arguments": to_json({many: chosen}),
                    },
                })
            elif index in folded:
                out.append({"id": call["id"], "joined": True})
            else:
                out.append(call)
        led.trace.append({"kind": "joined", "ids": list(chosen)})
        return out

    def _single_opens(
        self, calls: list[dict[str, Any]], opener: str, target: str
    ) -> list[tuple[int, str]]:
        led = self.led
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get("function") or {}
            raw = fn.get("arguments")
            args = parse_object(raw) if isinstance(raw, str) else None
            if fn.get("name") != opener or not args:
                continue
            if set(args) != {target}:
                continue
            ident = str(args.get(target) or "").strip()
            if ident in led.rows and ident not in led.gone:
                picks.append((index, ident))
        return picks


class Judge(Service):
    def shopper_said(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        phrases: list[str] = []
        self._gather_phrases(observation, 0, phrases)
        if not phrases:
            return to_json(observation)[:400]
        return " ".join(phrases)[:800]

    def _gather_phrases(self, node: Any, depth: int,
                        phrases: list[str]) -> None:
        if depth > 4:
            return
        if isinstance(node, str):
            if len(node.split()) >= 2:
                phrases.append(node)
            return
        if isinstance(node, dict):
            children = list(node.values())
        elif isinstance(node, list):
            children = node
        else:
            return
        for child in children:
            self._gather_phrases(child, depth + 1, phrases)

    def free_lines(self, observation: Any) -> list[str]:
        if not isinstance(observation, dict):
            return []
        if self.hub.intake.row_id(observation):
            return []
        lines: list[str] = []
        for value in observation.values():
            lines.extend(self._wordy_parts(value))
        heard = self.led.heard
        return [line for line in lines if line not in heard]

    def _wordy_parts(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if len(value.split()) >= 5 else []
        if isinstance(value, list):
            if not value or not all(isinstance(v, str) for v in value):
                return []
            return [v for v in value if len(v.split()) >= 5]
        if not isinstance(value, dict):
            return []
        if self.hub.intake.row_id(value, False, 1):
            return []
        return [
            inner for inner in value.values()
            if isinstance(inner, str) and len(inner.split()) >= 6
        ]

    def absorb_note(self, lines: list[str]) -> str:
        led = self.led
        led.heard.update(lines)
        prompt = (
            NOTE_PROMPT
            + "\n\nCURRENT ITEMS AND REQUIREMENTS:\n"
            + self.hub.intake.summary()
            + "\n\nNOTE TEXT:\n"
            + "\n".join(lines)
        )
        got = self.hub.llm.ask_for_json(prompt, [LIGHT, SPARE], 1200) or {}
        gone = [
            str(g) for g in as_rows(got.get("gone"))
            if isinstance(g, (str, int))
        ]
        led.unavailable.update(g for g in gone if g in led.rows)
        led.note_sold = bool(gone) or got.get("sold_out") is True
        # a lone sell-out line is kept so the same words are known later
        if led.note_sold and len(lines) == 1:
            led.sold_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed: list[str] = []
        budget = got.get("budget")
        if self._is_new_budget(budget):
            led.budget = budget
            changed.append(f"new budget {budget}")
        drops = self._need_entries(got, "drop")
        adds = self._need_entries(got, "add")
        for entry in drops:
            changed.append(self._drop_need(entry))
        for entry in adds:
            changed.append(self._add_need(entry))
        if gone:
            changed.append("not on sale: " + ", ".join(gone))
        if got.get("price_moved") is True:
            changed.append("one price moved")
        summary = str(got.get("summary") or "").strip()
        if changed or summary:
            led.version += 1
            led.recent = []
            led.notes.append(summary or "; ".join(changed))
        tail = " (" + "; ".join(changed) + ")" if changed else ""
        return (summary + tail).strip()

    def _is_new_budget(self, budget: Any) -> bool:
        if isinstance(budget, bool):
            return False
        if not isinstance(budget, (int, float)):
            return False
        return budget > 0 and budget != self.led.budget

    def _need_entries(self, got: dict[str, Any],
                      key: str) -> list[dict[str, Any]]:
        entries = []
        for entry in as_rows(got.get(key)):
            if not isinstance(entry, dict):
                continue
            if str(entry.get("need") or "").strip():
                entries.append(entry)
        return entries

    def _item_of(self, entry: dict[str, Any]) -> Item:
        items = self.led.items
        number = max(1, as_count(entry.get("item")) or 1)
        return items[min(number, len(items)) - 1]

    def _drop_need(self, entry: dict[str, Any]) -> str:
        item = self._item_of(entry)
        target = str(entry["need"]).strip()
        live = [m for m in item.musts + item.told if m not in item.dropped]
        hit = [m for m in live if m == target]
        if not hit:
            prop = spaced_words(target.split(":")[0])
            for need in live:
                if spaced_words(need.split(":")[0]) == prop:
                    hit = [need]
                    break
        item.dropped.extend(hit or [target])
        return f"removed {target}"

    def _add_need(self, entry: dict[str, Any]) -> str:
        item = self._item_of(entry)
        need = str(entry["need"]).strip()
        if need in item.dropped:
            item.dropped.remove(need)
        if need not in item.musts + item.told:
            item.told.append(need)
        return f"needs {need}"

    def current_needs(self) -> list[tuple[int, str]]:
        needs: list[tuple[int, str]] = []
        for number, item in enumerate(self.led.items, 1):
            needs.extend(
                (number, need) for need in item.musts + item.told
                if need not in item.dropped
            )
        return needs

    def evidence_rows(self, observation: Any) -> str:
        ids: list[str] = []
        self._gather_fact_ids(observation, 0, ids)
        needs = self.current_needs()
        if not ids or not needs:
            return ""
        lines = []
        for ident in ids[:12]:
            line = self._evidence_line(ident, needs[:8])
            if line:
                lines.append(line)
        if not lines:
            return ""
        return (
            "[evidence: keys whose names hold a requirement's property; a"
            " key naming some other property (a different part or aspect)"
            " settles nothing, and a key under another name can still"
            " carry that property] " + " | ".join(lines)
        )

    def _gather_fact_ids(self, node: Any, depth: int,
                         ids: list[str]) -> None:
        if depth > 6:
            return
        if isinstance(node, list):
            for child in node:
                self._gather_fact_ids(child, depth + 1, ids)
            return
        if not isinstance(node, dict):
            return
        ident = self.hub.intake.row_id(node, False, depth)
        if ident and find_facts(node) and ident not in ids:
            ids.append(ident)
            return
        for child in node.values():
            self._gather_fact_ids(child, depth + 1, ids)

    def _evidence_line(self, ident: str,
                       needs: list[tuple[int, str]]) -> str:
        block = find_facts(self.led.full.get(ident) or {}) or {}
        tagged = len(self.led.items) > 1
        parts = []
        for number, need in needs:
            prop = need.split(":", 1)[0].strip()
            wanted = word_stems(prop)
            hits = [
                f"{key}={str(value)[:50]}" for key, value in block.items()
                if names_same(wanted, word_stems(key))
            ]
            if not hits:
                continue
            tag = f"(item {number}) " if tagged else ""
            parts.append(f"{tag}{prop} -> " + ", ".join(hits[:3]))
        if not parts:
            return ""
        return f"{ident}: " + "; ".join(parts)

    def _cart_view_current(self) -> bool:
        led = self.led
        if led.cart_view is None:
            return False
        return led.cart_seen_at >= led.cart_at

    def cart_summary(self) -> str:
        if self._cart_view_current():
            return to_json(clip(self.led.cart_view, 300))[:3000]
        # an older view could still list lines that were removed since
        return (
            "cart lines as Kyo logged its cart calls (listing: units): "
            + to_json(self.led.cart_lines)
        )

    def cart_listing_ids(self) -> list[str]:
        if not self._cart_view_current():
            return list(self.led.cart_lines)
        text = to_json(self.led.cart_view)
        return [ident for ident in self.led.rows if ident in text]

    def record(self, ident: str) -> str:
        found = self.led.full.get(ident) or self.led.rows.get(ident)
        if not found:
            return "(no record held)"
        return to_json(clip(found, 400))[:3000]

    def other_records(self, skip: list[str], limit: int = 30) -> str:
        led = self.led
        order = led.seen_order[::-1]
        if led.brief.get("pick") == "cheapest":
            order.sort(key=self._cheapest_rank)
        lines: list[str] = []
        for ident in order:
            if ident in skip or self.is_sold_out(ident):
                continue
            found = led.full.get(ident) or led.rows.get(ident)
            lines.append(to_json(clip(found, 160))[:900])
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def _cheapest_rank(self, ident: str) -> tuple[bool, float]:
        cost = self.hub.catalogue.price(ident)
        return (
            ident not in self.led.full,
            float("inf") if cost is None else cost,
        )

    def verdict_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.led.version}"

    def review_cart(self, ids: list[str] | None = None,
                    limit: float = SPEND_SOFT) -> str:
        led = self.led
        planned = ids is not None
        if not planned:
            ids = self.cart_listing_ids()
        key = self.verdict_key(ids)
        if key in led.checked or led.cost > limit or not ids:
            return ""
        led.checked.add(key)
        prompt = self._review_prompt(ids, planned)
        got = self.hub.llm.ask_for_json(prompt, [HEAVY, LIGHT], 1500)
        led.trace.append({"kind": "check", "cart": ids, "got": got})
        led.verdicts[key] = bool(got) and got.get("ok") is not False
        rejected = bool(got) and got.get("ok") is False
        if not rejected:
            return ""
        led.leads = [i for i in led.leads if i not in ids]
        return self._review_note(got)

    def _review_prompt(self, ids: list[str], planned: bool) -> str:
        led = self.led
        parts = [
            CHECK_PROMPT,
            "\n\nREQUEST:\n",
            led.request,
            "\n\nREQUIREMENTS THAT APPLY NOW:\n",
            self.hub.intake.summary(),
        ]
        if planned:
            quantities = "; ".join(
                f"item {n}: quantity {item.qty}"
                for n, item in enumerate(led.items, 1)
            )
            parts.append(
                "\n\nCART (not placed yet; treat these listings as already"
                " carted at the requested quantities: " + quantities
                + "):\n" + ", ".join(ids)
            )
        else:
            parts.append("\n\nCART:\n" + self.cart_summary())
        rows = []
        for ident in ids:
            flag = ""
            if self.is_sold_out(ident):
                flag = "SOLD OUT (per the store it can no longer be bought). "
            rows.append(f"{ident}: " + flag + self.record(ident))
        parts.append(
            "\n\nRECORDS FOR THE LISTINGS IN THE CART:\n" + "\n".join(rows)
        )
        if led.brief.get("pick") in ("cheapest", "most", "least"):
            parts.append(
                "\n\nOTHER LISTINGS SHOWN (most recent first):\n"
                + self.other_records(ids, 15)
            )
        return "".join(parts)

    def _review_note(self, got: dict[str, Any]) -> str:
        problems = [str(p) for p in as_rows(got.get("problems"))][:5]
        better = []
        for entry in as_rows(got.get("better")):
            ident = str(entry)
            if ident in self.led.rows and not self.is_sold_out(ident):
                better.append(ident)
        better = better[:4]
        text = (
            "An independent check finds this cart still wrong: "
            + "; ".join(problems or ["see the requirements"])
        )
        if better:
            text += ". Listings that seem better: " + ", ".join(better)
        return text + (
            ". Correct the cart (or re-read the records to show the check"
            " is wrong), then check out."
        )

    def follow_trails(self) -> tuple[str, bool]:
        led = self.led
        live = {need for _, need in self.current_needs()}
        pending = [
            need for _, need in led.refs
            if need in live and need not in led.ref_values
        ]
        # chain links are pages, so use the offer view when it stands apart
        tool = led.roles.get("offer") or led.roles.get("open")
        target = self.hub.catalogue.takes_one(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and led.trails < 4 and led.cost < SPEND_HARD:
            led.trails += 1
            got = self._ask_trail(pending)
            reach, lost = self._walk_chains(got, pending)
            pending = [p for p in pending if p not in led.ref_values]
            if not reach or not tool or not target:
                break
            if led.steps_left() < len(reach) + 4 or led.finished:
                break
            self._open_links(reach, tool, target, seen)
        return self._trail_outcome(seen, lost)

    def _trail_records(self) -> list[str]:
        led = self.led
        records: list[str] = []
        for ident in reversed(led.seen_order):
            found = None
            if ident in led.opened:
                found = led.full.get(ident) or led.rows.get(ident)
            if found:
                records.append("OPENED " + to_json(clip(found, 300))[:1400])
            else:
                row = clip(led.rows.get(ident), 80)
                records.append("ROW " + to_json(row)[:300])
            if len(records) >= 60:
                break
        return records

    def _ask_trail(self, pending: list[str]) -> dict[str, Any]:
        records = self._trail_records()
        prompt = (
            TRAIL_PROMPT
            + "\n\nREQUIREMENTS WHOSE VALUE LIVES ON ANOTHER LISTING:\n"
            + "\n".join(pending)
            + "\n\nRECORDS (most recent first):\n"
            + "\n".join(records)
        )
        got = self.hub.llm.ask_for_json(prompt, [LIGHT, HEAVY], 1800) or {}
        self.led.trace.append(
            {"kind": "trail", "pending": pending, "got": clip(got, 150)}
        )
        return got

    def _chain_need(self, chain: dict[str, Any],
                    pending: list[str]) -> str | None:
        named = str(chain.get("need") or "").strip()
        for need in pending:
            if need == named:
                return need
        return pending[0] if len(pending) == 1 else None

    def _walk_chains(self, got: dict[str, Any],
                     pending: list[str]) -> tuple[list[str], list[str]]:
        led = self.led
        reach: list[str] = []
        lost: list[str] = []
        for chain in as_rows(got.get("chains")):
            if not isinstance(chain, dict):
                continue
            need = self._chain_need(chain, pending)
            if not need:
                continue
            links = [
                link for link in as_rows(chain.get("links"))
                if isinstance(link, dict)
            ]
            ids = [str(link.get("id") or "").strip() for link in links]
            if self._chain_settled(chain, links, ids):
                led.ref_values[need] = str(chain["value"]).strip()
                led.version += 1
                continue
            closed = [i for i in ids if i in led.rows and i not in led.opened]
            if closed:
                reach.extend([i for i in closed if i not in reach])
                continue
            about = ""
            for link, ident in zip(links, ids):
                if ident not in led.rows:
                    about = str(link.get("about") or "")
                    break
            about = about or "the listing it refers to"
            lost.append(f"{need} -> seek {about}")
        return reach, lost

    def _chain_settled(self, chain: dict[str, Any],
                       links: list[dict[str, Any]], ids: list[str]) -> bool:
        if chain.get("done") is not True or not links:
            return False
        if any(ident not in self.led.opened for ident in ids):
            return False
        return bool(str(chain.get("value") or "").strip())

    def _open_links(self, reach: list[str], tool: str, target: str,
                    seen: list[str]) -> None:
        led = self.led
        room = max(0, min(3, 6 - len(seen)))
        for ident in reach[:room]:
            faults: list[str] = []
            args = conform(
                {target: ident}, led.schemas.get(tool), "args", faults
            )
            if faults:
                return
            reply = self.hub.store.call(tool, args)
            if reply is None or led.finished:
                return
            observation = reply.get("observation")
            self.hub.catalogue.observe(tool, observation)
            shown = self.hub.store.render(observation)[:1500]
            seen.append(f"{ident}: " + shown)

    def _trail_outcome(self, seen: list[str],
                       lost: list[str]) -> tuple[str, bool]:
        led = self.led
        text = ""
        if seen:
            text += (
                "[Kyo opened every page that requirements cite]\n"
                + "\n".join(seen) + "\n"
            )
        if led.ref_values:
            pairs = "; ".join(
                f"{need} -> {value}" for need, value in led.ref_values.items()
            )
            text += (
                "[values taken off the pages requirements refer to] "
                + pairs + "\n"
            )
        hold = bool(lost) and led.trail_blocks < 2
        hold = hold and led.steps_left() > 5
        if not hold:
            return text, False
        led.trail_blocks += 1
        text += (
            "The order is valid only once every page a requirement refers"
            " to is opened in full (a search row does not suffice), going"
            " through each page that names a further listing."
            " Not yet found and opened: " + " | ".join(lost[:3])
            + ". Look it up by its shop, brand or wording, open it, use the"
            " value its page provides, then check out."
        )
        return text, True

    def offer_view_tool(self) -> str | None:
        roles = self.led.roles
        return roles.get("offer") or roles.get("open")

    def refresh_offers(self) -> str:
        tool = self.offer_view_tool()
        target = self.hub.catalogue.takes_one(tool)
        if not tool or not target:
            return ""
        led = self.led
        floor = max(led.step - 2, led.refused_at)
        stale = [
            ident for ident in self.cart_listing_ids()
            if led.offer_seen.get(ident, -99) < floor
            and ident not in led.gone
        ]
        if not stale or led.steps_left() < len(stale) + 3:
            return ""
        changes = self._reread_listings(tool, target, stale)
        led.trace.append(
            {"kind": "reread", "ids": stale, "changed": len(changes)}
        )
        if changes:
            led.version += 1
        return "\n".join(changes)

    def _reread_listings(
        self, tool: str, target: str, stale: list[str]
    ) -> list[str]:
        led = self.led
        changes: list[str] = []
        for ident in stale:
            faults: list[str] = []
            args = conform(
                {target: ident}, led.schemas.get(tool), "args", faults
            )
            if faults:
                break
            before = dict(led.rows.get(ident) or {})
            reply = self.hub.store.call(tool, args)
            if reply is None or led.finished:
                break
            seen = reply.get("observation")
            led.offer_seen[ident] = led.step
            self.hub.catalogue.observe(tool, seen)
            shifted = self._shifted_fields(
                before, led.rows.get(ident) or {}
            )
            if not shifted:
                continue
            fields = ", ".join(shifted[:4])
            changes.append(
                f"{ident} differs ({fields}): "
                f"{to_json(clip(seen, 300))[:900]}"
            )
        return changes

    @staticmethod
    def _shifted_fields(
        before: dict[str, Any], after: dict[str, Any]
    ) -> list[str]:
        shifted = []
        for key, value in after.items():
            if isinstance(value, (dict, list)) or key not in before:
                continue
            if before[key] != value:
                shifted.append(key)
        return shifted

    def is_sold_out(self, ident: str) -> bool:
        return ident in self.led.unavailable or ident in self.led.gone

    def swap_advice(self) -> str:
        led = self.led
        lost = [
            ident for ident in self.cart_listing_ids()
            if ident not in led.gone and ident in led.unavailable
        ]
        if not lost:
            return ""
        for ident in lost:
            led.gone[ident] = led.step
        led.leads = [ident for ident in led.leads if ident not in lost]
        led.version += 1
        picks = self.find_replacements(lost)
        opening = (
            f"[Kyo] Out of stock: {', '.join(lost)} (the store marks it as "
            "unbuyable). It stays gone, so there is no use opening it "
            "again."
        )
        return opening + self._swap_steps(picks)

    def _swap_steps(self, picks: list[str]) -> str:
        if not picks:
            return (
                " Take that line out of the cart and put in the best other "
                "fitting listing; if nothing you read fits, search for one."
            )
        first, listed = picks[0], ", ".join(picks)
        if self.led.steps_left() <= 6:
            rest = ", ".join(picks[1:]) or "none"
            return (
                f" Steps are running low: right now put {first} in the "
                "cart at the requested quantity and drop the sold-out line "
                "(two calls), then check out. Other read listings that seem "
                f"to fit: {rest}."
            )
        if self.led.brief.get("pick") == "cheapest":
            return (
                " Listings you have read that seem to suit that item "
                f"instead, cheapest first: {listed}. Since the pick rule is "
                "the cheapest, unless you have already seen all fitting "
                f"listings, run a search below {first}'s price (with the "
                "search's top price limit, if any) before you replace the "
                "sold-out line by the cheapest one that fits."
            )
        return (
            " Listings you have read that seem to suit that item "
            f"instead: {listed}. Right now take the sold-out line out of "
            f"the cart and put in {first} at the quantity requested, then "
            "check out."
        )

    def sold_report(self) -> str:
        led = self.led
        ids = [i for i in self.cart_listing_ids() if i not in led.gone]
        if not ids or any(self.is_sold_out(i) for i in ids):
            return ""
        if len(ids) > 1:
            return self.refresh_offers()
        led.unavailable.add(ids[0])
        return ""

    def find_replacements(self, lost: list[str]) -> list[str]:
        led = self.led
        if led.cost > SPEND_SOFT:
            return []
        records = self._swap_candidates(lost)
        if not records:
            return []
        got = self.hub.llm.ask_for_json(
            self._swap_prompt(lost, records), [HEAVY, LIGHT], 1200
        )
        fits = self._priced_fits(got, lost)
        if led.brief.get("pick") == "cheapest":
            ordered = sorted(fits, key=lambda pair: pair[0])
        else:
            ordered = fits
        led.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return [ident for _, ident in ordered][:4]

    def _swap_candidates(self, lost: list[str]) -> list[str]:
        led = self.led
        records: list[str] = []
        for ident in reversed(led.seen_order):
            skip = ident in lost or ident in led.gone
            if skip or self.is_sold_out(ident):
                continue
            record = led.full.get(ident) or led.rows.get(ident)
            records.append(to_json(clip(record, 200))[:1200])
            if len(records) == 25:
                break
        return records

    def _swap_prompt(self, lost: list[str], records: list[str]) -> str:
        sold = "\n".join(f"{i}: {self.record(i)}" for i in lost)
        return "".join([
            SWAP_PROMPT,
            "\n\nREQUEST:\n", self.led.request,
            "\n\nREQUIREMENTS THAT APPLY:\n", self.hub.intake.summary(),
            "\n\nSOLD-OUT CART LINE:\n", sold,
            "\n\nCURRENT CART:\n", self.cart_summary(),
            "\n\nOTHER LISTING RECORDS (most recent first):\n",
            "\n".join(records),
        ])

    def _priced_fits(
        self, got: dict[str, Any] | None, lost: list[str]
    ) -> list[tuple[float, str]]:
        led = self.led
        in_cart = self.cart_listing_ids()
        fits: list[tuple[float, str]] = []
        for entry in as_rows((got or {}).get("fits")):
            if isinstance(entry, dict):
                ident, price = str(entry.get("id")), entry.get("price")
            else:
                ident, price = str(entry), None
            known = ident in led.rows and ident not in lost
            if not known or ident in led.gone or ident in in_cart:
                continue
            if ident in [taken for _, taken in fits]:
                continue
            numeric = (
                isinstance(price, (int, float))
                and not isinstance(price, bool)
            )
            fits.append((float(price) if numeric else float("inf"), ident))
        return fits

    def audit_decline(self) -> str:
        led = self.led
        if led.audited or led.cost > SPEND_SOFT:
            return ""
        led.audited = True
        got = self.audit_request()
        named = self.audit_matches(got)
        if not named:
            got, named = self._audit_after_opening(got)
            if not named:
                return ""
        led.lead_items = named
        led.leads = [ident for ids in named.values() for ident in ids]
        groups = "; ".join(
            f"item {item}: {', '.join(ids)}" for item, ids in named.items()
        )
        why = f" ({got.get('reason')})" if got.get("reason") else ""
        return (
            "A further look suggests these listings may fit: "
            + groups + why + ". Open them and buy whichever really fits; "
            "finish without buying only if none fits."
        )

    def _audit_after_opening(
        self, got: dict[str, Any] | None
    ) -> tuple[dict[str, Any] | None, dict[str, list[str]]]:
        led = self.led
        unread = []
        for raw in as_rows((got or {}).get("open")):
            ident = str(raw)
            if ident not in led.rows or ident in led.opened:
                continue
            if not self.is_sold_out(ident):
                unread.append(ident)
        if not unread or not self.hub.cart.open_listings(unread):
            return got, {}
        got = self.audit_request()
        return got, self.audit_matches(got)

    def audit_request(self) -> dict[str, Any] | None:
        led = self.led
        records: list[str] = []
        for ident in reversed(led.seen_order):
            if self.is_sold_out(ident):
                continue
            opened = led.full.get(ident)
            if opened:
                records.append(
                    "OPENED " + to_json(clip(opened, 200))[:1200]
                )
            else:
                row = led.rows.get(ident)
                records.append("ROW " + to_json(clip(row, 120))[:500])
            if len(records) == 30:
                break
        if not records:
            return None
        prompt = "".join([
            AUDIT_PROMPT,
            "\n\nREQUEST:\n", led.request,
            "\n\nREQUIREMENTS THAT APPLY:\n", self.hub.intake.summary(),
            "\n\nLISTING RECORDS (most recent first):\n",
            "\n".join(records),
        ])
        got = self.hub.llm.ask_for_json(prompt, [HEAVY, LIGHT], 1500)
        led.trace.append({"kind": "audit", "got": got})
        return got

    def audit_matches(
        self, got: dict[str, Any] | None
    ) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named: dict[str, list[str]] = {}
        for item, listed in fits.items():
            kept = [
                str(i) for i in as_rows(listed)
                if str(i) in self.led.rows and not self.is_sold_out(str(i))
            ]
            named[str(item)] = kept[:4]
        return {item: ids for item, ids in named.items() if ids}


class ToolRunner(Service):
    SKIP = "Skipped, no step spent: "
    HOLD = "On hold, no step was spent. "
    WAIT = "On hold now. "

    class Turn:
        def __init__(self, call: dict[str, Any], crowd: int) -> None:
            self.cid = call["id"]
            self.call = call
            self.crowd = crowd
            self.name: str | None = None
            self.raw: Any = None
            self.args: dict[str, Any] = {}
            self.mark = ""
            self.extra = ""
            self.many: str | None = None
            self.target: str | None = None
            self.ending = False
            self.readers: set[str] = set()
            self.plain: set[str] = set()

    def execute(
        self, calls: list[dict[str, Any]]
    ) -> list[tuple[str, str]]:
        replies: list[tuple[str, str]] = []
        calls = self.hub.cart.merge_opens(calls)
        for call in calls:
            turn = self.Turn(call, len(calls))
            replies.append((turn.cid, self._handle(turn)))
        return replies

    def _handle(self, turn: ToolRunner.Turn) -> str:
        if turn.call.get("joined"):
            return (
                "Kyo merged these opens into one side-by-side call; "
                "the result above already covers all."
            )
        stages = (
            self._screen_tool,
            self._decode_args,
            self._validate_args,
            self._fit_compare,
            self._guard_sold_out,
            self._guard_removal,
            self._guard_crowded_end,
            self._guard_reserve,
            self._guard_repeat,
            self._guard_cache,
            self._guard_shown,
            self._trim_shown,
            self._hold_finish,
            self._hold_late_decline,
            self._push_search,
            self._prepare_decline,
            self._pass_gate,
        )
        for stage in stages:
            refusal = stage(turn)
            if refusal is not None:
                return refusal
        return self._dispatch(turn)

    def _roles_of(self, *jobs: str) -> tuple[str | None, ...]:
        return tuple(self.led.roles.get(job) for job in jobs)

    def _with_cart(self, text: str) -> str:
        return text + "\nCart now: " + self.hub.judge.cart_summary()

    # guards that run before any store call

    def _screen_tool(self, turn: ToolRunner.Turn) -> str | None:
        led = self.led
        fn = turn.call.get("function") or {}
        turn.name = name = fn.get("name")
        turn.raw = fn.get("arguments")
        if led.finished:
            return "Skipped: the episode is done."
        if name == led.roles.get("ask"):
            return (
                self.SKIP + "every shopper message is sent by "
                "Kyo's own interviewer."
            )
        allowed = {
            spec["function"]["name"]
            for spec in self.hub.store.tool_specs()
        }
        if name not in led.schemas:
            return (
                f"{self.SKIP}tool not found. "
                f"Use: {', '.join(sorted(allowed))}."
            )
        if name not in allowed:
            return (
                self.SKIP + "steps left are too few for that; "
                "use: " + ", ".join(sorted(allowed)) + "."
            )
        return None

    def _decode_args(self, turn: ToolRunner.Turn) -> str | None:
        raw = turn.raw
        if isinstance(raw, dict):
            turn.args = raw
            return None
        args = parse_object(raw) if (raw or "").strip() else {}
        if args is None:
            return (
                self.SKIP + "arguments must make up one JSON object."
            )
        turn.args = args
        return None

    def _validate_args(self, turn: ToolRunner.Turn) -> str | None:
        faults: list[str] = []
        args = conform(
            turn.args, self.led.schemas[turn.name], "args", faults
        )
        if faults:
            return self.SKIP + "; ".join(faults[:6])
        args, wrong = self.hub.store.fix_ids(turn.name, args)
        if wrong:
            return self.SKIP + wrong + "."
        turn.args = args
        return None

    def _fit_compare(self, turn: ToolRunner.Turn) -> None:
        led = self.led
        desk = self.hub.cart
        if turn.name == led.roles.get("compare"):
            turn.many = desk.compare_tool()
        many, args = turn.many, turn.args
        too_many = (
            many
            and isinstance(args.get(many), list)
            and len(args[many]) > desk.compare_slots()
            and led.compare_bad
        )
        if not too_many:
            return None
        room = desk.compare_slots()
        unseen = ", ".join(map(str, args[many][room:]))
        turn.extra = (
            f"\n[Kyo] Only the first {room} listings were sent; store "
            f"takes at most {room} at once. Still left unshown: {unseen}."
        )
        turn.args = {**args, many: args[many][:room]}
        return None

    def _guard_sold_out(self, turn: ToolRunner.Turn) -> str | None:
        led = self.led
        target = turn.target = self.hub.catalogue.takes_one(turn.name)
        if not target or turn.name not in self._roles_of("open", "offer"):
            return None
        key = str(turn.args.get(target))
        if key not in led.gone:
            return None
        return (
            f"{self.SKIP}{turn.args[target]} sold out at step "
            f"{led.gone[key]} for good and will never return. Swap its cart "
            "line for a listing that fits."
        )

    def _guard_removal(self, turn: ToolRunner.Turn) -> str | None:
        led = self.led
        args = turn.args
        if turn.name != led.roles.get("cart"):
            return None
        lowers = any(
            (as_count(value) or 0) < 0
            for value in args.values()
            if not isinstance(value, str)
        )
        if not lowers:
            return None
        fresh = led.cart_seen_at >= led.cart_at
        view = to_json(led.cart_view) if fresh else ""
        absent = [
            value for value in args.values()
            if isinstance(value, str)
            and value in led.rows
            and value not in led.cart_lines
            and value not in view
            and value not in led.cart_said
        ]
        if not absent:
            return None
        return (
            f"{self.SKIP}{absent[0]} is not a cart line, so there is "
            "nothing to take out. Cart now: " + self.hub.judge.cart_summary()
        )

    def _guard_crowded_end(self, turn: ToolRunner.Turn) -> str | None:
        store = self.hub.store
        turn.ending = (
            store.is_checkout(turn.name, turn.args)
            or store.is_decline(turn.name, turn.args)
        )
        if turn.ending and turn.crowd > 1:
            return (
                self.SKIP + "read your other results first and then send "
                "the finishing call alone."
            )
        return None

    def _guard_reserve(self, turn: ToolRunner.Turn) -> str | None:
        led = self.led
        if turn.ending or led.steps_left() > led.finish_reserve():
            return None
        if not any(self._roles_of("buy", "refuse")):
            return None
        return (
            self.SKIP + "the remaining steps are reserved for checkout "
            "(after the shopper approves the basket, when due) or "
            "for ending without a purchase."
        )

    def _guard_repeat(self, turn: ToolRunner.Turn) -> str | None:
        led = self.led
        turn.mark = turn.name + to_json(turn.args)
        if turn.ending or turn.name == led.roles.get("cart"):
            return None
        if turn.mark not in led.recent:
            return None
        return (
            self.SKIP + "this same call was just made and nothing "
            "changed; its result is above."
        )

    def _guard_cache(self, turn: ToolRunner.Turn) -> str | None:
        led = self.led
        turn.readers = set(
            self._roles_of("search", "open", "offer", "compare")
        ) - {None}
        cached = led.read_cache.get(turn.mark)
        if turn.name not in turn.readers or not cached:
            return None
        if cached[0] != led.version:
            return None
        return (
            f"{self.SKIP}the call made at step {cached[1]} gave this "
            "result, and nothing new came since:\n" + cached[2][:3000]
        )

    def _guard_shown(self, turn: ToolRunner.Turn) -> str | None:
        target, name = turn.target, turn.name
        if not target or name not in turn.readers:
            return None
        if name == self.led.roles.get("compare"):
            return None
        ident = str(turn.args.get(target))
        if not self.hub.catalogue.already_shown(name, ident):
            return None
        return (
            f"{self.SKIP}{turn.args[target]} has nothing new to show in "
            "this view since the last change. Its record: "
            + self.hub.judge.record(str(turn.args[target]))
        )

    def _trim_shown(self, turn: ToolRunner.Turn) -> str | None:
        many = turn.many
        listed = turn.args.get(many) if many else None
        if not (isinstance(listed, list) and listed):
            return None
        shown = self.hub.catalogue.already_shown
        keep = [i for i in listed if not shown(turn.name, str(i))]
        if not keep:
            record = self.hub.judge.record
            records = "\n".join(f"{i}: {record(str(i))}" for i in listed)
            return (
                self.SKIP + "these listings have nothing new to show in "
                "this view since the last change. Their records:\n"
                + records[:4000]
            )
        if len(keep) < len(listed):
            left_out = ", ".join(str(i) for i in listed if i not in keep)
            turn.extra += (
                f"\n[Kyo] Not sent {left_out}: already shown in full; it is "
                "unchanged since."
            )
            turn.args = {**turn.args, many: keep}
            turn.mark = turn.name + to_json(turn.args)
        return None

    # holds on a checkout or an ending while steps remain

    def _hold_finish(self, turn: ToolRunner.Turn) -> str | None:
        if not turn.ending or self.led.steps_left() <= 2:
            return None
        if self.hub.store.is_checkout(turn.name, turn.args):
            return self._hold_checkout()
        if self.led.steps_left() > 5:
            return self._hold_early_decline()
        return None

    def _hold_checkout(self) -> str | None:
        judge = self.hub.judge
        wider = self.hub.catalogue.hold_wider()
        if wider:
            return self._with_cart(wider)
        reread = judge.refresh_offers()
        change = self._look_at_cart()
        if change is not None:
            return change
        lost = self._hold_lost_lines()
        if lost is not None:
            return lost
        counts = self._hold_counts()
        if counts is not None:
            return counts
        categories = self._hold_categories()
        if categories is not None:
            return categories
        chain, hold = judge.follow_trails()
        if hold:
            return self._with_cart(self.HOLD + chain)
        verdict = self._hold_on_check(chain)
        if verdict is not None:
            return verdict
        if reread and self.led.cost > SPEND_SOFT:
            return (
                self.HOLD + "Kyo just now re-read the cart's listings:\n"
                + reread + "\nRetry the checkout once every line still has "
                "stock and a price within the budget; if not, swap out that "
                "line first."
            )
        return None

    def _hold_counts(self) -> str | None:
        led = self.led
        have = sum(led.cart_lines.values())
        want = sum(item.qty for item in led.items)
        lines, items = len(led.cart_lines), len(led.items)
        if led.counts_held or not led.cart_lines:
            return None
        if have == want and lines == items:
            return None
        led.counts_held = True
        wanted = "; ".join(f"{item.name} x{item.qty}" for item in led.items)
        return self._with_cart(
            self.HOLD + f"The cart holds {have} units across {lines} "
            f"lines, but the request asks for {items} items and {want} "
            f"units in all ({wanted}): one line per requested item, each at "
            "its quantity, and nothing else. Add any missing item, remove "
            "any extra line, fix the counts, then check out again."
        )

    def _hold_categories(self) -> str | None:
        led = self.led
        if led.categories_held or not led.cart_lines:
            return None
        field = self.hub.catalogue.category_field()
        if not field:
            return None
        labels = {
            word_stems(str(row[field])): str(row[field])
            for row in led.rows.values()
            if isinstance(row.get(field), str) and row[field].strip()
        }
        carted = {
            ident: str(led.rows.get(ident, {}).get(field) or "")
            for ident in led.cart_lines
        }
        named = []
        for item in led.items:
            own = labels.get(word_stems(item.name))
            if not own and item.shelf:
                own = labels.get(word_stems(item.shelf))
            if own:
                named.append((item.name, own))
        missing = [
            (name, own) for name, own in named
            if own not in carted.values()
        ]
        owned = {own for _, own in named}
        stray = [
            (ident, label) for ident, label in carted.items()
            if label and label not in owned
        ]
        if not missing or not stray:
            return None
        led.categories_held = True
        want = "; ".join(f"{name} -> '{own}'" for name, own in missing)
        have = "; ".join(f"{ident} is in '{label}'" for ident, label in stray)
        return self._with_cart(
            self.HOLD + "The store has a category named after a requested "
            f"item, yet no cart line comes from it ({want}), while {have}. "
            "Unless the request itself points to that other category, "
            "search the named category, cart a listing from it that meets "
            "the requirements, and check out again."
        )

    def _look_at_cart(self) -> str | None:
        led = self.led
        tool = led.roles.get("cart_view")
        stale = led.cart_touched and led.cart_seen_at < led.cart_at
        if not (stale and tool and led.steps_left() > 3):
            return None
        faults: list[str] = []
        view_args = conform({}, led.schemas.get(tool), "args", faults)
        if faults:
            return None
        view = self.hub.store.call(tool, view_args)
        if view is None:
            return None
        led.cart_view = view.get("observation")
        led.cart_seen_at = led.step
        change = self.hub.catalogue.observe(
            led.roles["cart_view"], led.cart_view
        )
        if not change:
            return None
        led.queued = []
        return (
            self._with_cart(
                self.HOLD + "The cart view revealed a change now: " + change
            )
            + "\nAdopt the change, fix the cart if needed, then check out."
        )

    def _hold_lost_lines(self) -> str | None:
        led = self.led
        judge = self.hub.judge
        swap = judge.swap_advice()
        held: list[str] = []
        if led.held < 2:
            held = [i for i in judge.cart_listing_ids() if i in led.gone]
        if not (swap or held):
            return None
        if not swap:
            led.held += 1
        reason = swap or (
            f"{', '.join(held)} in the cart is sold out: correct that "
            "line first."
        )
        return self._with_cart(self.HOLD + reason)

    def _hold_on_check(self, chain: str) -> str | None:
        led = self.led
        judge = self.hub.judge
        catalogue = self.hub.catalogue
        read = catalogue.fetch_facts(
            judge.cart_listing_ids(), led.finish_reserve() + 3
        ) + catalogue.cheaper_candidates()
        verdict = judge.review_cart()
        if not verdict:
            return None
        seen = ""
        if read:
            seen = (
                "For the check, Kyo read these listings' attributes, a "
                "step apiece:\n" + "\n".join(read) + "\n"
            )
        return self._with_cart(self.WAIT + chain + seen + verdict)

    def _hold_fallback(self) -> str | None:
        led = self.led
        clause = str(led.brief.get("nothing_fits") or "").strip()
        if not clause or led.fallback_held:
            return None
        if led.steps_left() < len(led.items) + 3:
            return None
        led.fallback_held = True
        led.trace.append({"kind": "fallback_held", "clause": clause[:200]})
        return self._with_cart(
            self.HOLD + "Before ending without a purchase, read what the "
            f"request says to do when nothing fits: \"{clause}\". Follow "
            "it exactly: if it asks you to buy or put forward a listing "
            "that falls short, cart the listing it describes at the "
            "requested quantities and check out (or name it in your ending "
            "text, as it says); end without a purchase only if that is "
            "what it asks for."
        )

    def _hold_early_decline(self) -> str | None:
        led = self.led
        review = self.hub.judge.audit_decline()
        if review:
            return self.HOLD + review
        if not led.leads or led.steps_left() <= 4:
            return None
        if led.refuse_blocks == 0:
            return self._point_at_leads()
        if led.refuse_blocks == 1:
            return self._confirm_first_lead()
        return None

    def _point_at_leads(self) -> str:
        led = self.led
        led.refuse_blocks = 1
        led.trace.append({"kind": "refuse_blocked", "leads": list(led.leads)})
        return (
            self.HOLD + "The review turned up listings that might fit: "
            + ", ".join(led.leads) + ". Open them if you need to, cart the "
            "one that fits best (per the pick rule) and check out. Only "
            "if every one of them plainly breaks a requirement may you end "
            "without a purchase; then name which in your text and "
            "send the ending call again."
        )

    def _confirm_first_lead(self) -> str | None:
        led = self.led
        judge = self.hub.judge
        led.refuse_blocks = 2
        first = led.leads[0]
        judge.review_cart([first], SPEND_HARD)
        approved = led.verdicts.get(judge.verdict_key([first]))
        if not (approved and first in led.leads):
            return None
        led.confirmed = first
        led.trace.append({"kind": "refuse_blocked", "confirmed": first})
        return (
            f"{self.HOLD}A separate check run on {first}'s records finds "
            "that it meets every requirement. Cart it at the requested "
            "quantity and check out; end without a purchase only when its "
            "record plainly breaks some requirement."
        )

    def _hold_late_decline(self, turn: ToolRunner.Turn) -> str | None:
        led = self.led
        judge = self.hub.judge
        if not self.hub.store.is_decline(turn.name, turn.args):
            return None
        if led.steps_left() > 5 or led.late_held:
            return None
        lines = [i for i in led.built if not judge.is_sold_out(i)]
        fresh = led.cart_view is not None and led.cart_seen_at >= led.cart_at
        missing = [
            i for i in lines
            if i not in judge.cart_listing_ids()
            or (i in led.dropped and not fresh)
        ]
        if len(lines) < len(led.items):
            return None
        if len(missing) + 1 > led.steps_left():
            return None
        led.late_held = True
        led.trace.append(
            {"kind": "late_decline_held", "lines": lines, "missing": missing}
        )
        if missing:
            redo = (
                f"Place {', '.join(missing)} back into the cart at the "
                "requested quantities, and check out."
            )
        else:
            redo = "Check out this cart, unchanged."
        return (
            self.HOLD + "If the store has a listing that fits, ending "
            "without a purchase scores nothing, and a shortage of steps is "
            "no proof that it lacks one. " + redo + " Only end without a "
            "purchase when your search showed that some requested item has "
            "no listing meeting its requirements; then send the ending call "
            "again."
        )

    def _push_search(self, turn: ToolRunner.Turn) -> str | None:
        led = self.led
        if not self.hub.store.is_decline(turn.name, turn.args):
            return None
        if led.leads or led.pushed:
            return None
        if led.steps_left() <= led.max_steps // 3:
            return None
        led.pushed = True
        return (
            f"{self.HOLD}{led.steps_left()} steps remain, so before ending "
            "try searches you have not used yet: browse each "
            "requested item's category and page through it using the token "
            "or offset the store gives back, query single-word values and "
            "other names for that property, and open those listings that "
            "might fit. Only after that may you end without a purchase."
        )

    def _prepare_decline(self, turn: ToolRunner.Turn) -> str | None:
        store = self.hub.store
        if not store.is_decline(turn.name, turn.args):
            return None
        done = self.hub.cart.fill_cart()
        if done:
            return done + (" [episode finished]" if self.led.finished else "")
        # the request's own words on what to do when nothing fits come last,
        # once the review and the push to search more have had their turn
        fallback = self._hold_fallback()
        if fallback is not None:
            return fallback
        turn.args = store.quiet_args(turn.name, turn.args)
        return None

    def _pass_gate(self, turn: ToolRunner.Turn) -> str | None:
        if not turn.ending or self.led.steps_left() < 2:
            return None
        buying = self.hub.store.is_checkout(turn.name, turn.args)
        go, said = self.hub.questions.run_gate("order" if buying else "end")
        if not go:
            return self._with_cart(self.WAIT + said)
        if said:
            turn.extra = "\n[Kyo] " + said + turn.extra
        return None

    # the store call and what it teaches

    def _dispatch(self, turn: ToolRunner.Turn) -> str:
        led = self.led
        reply = self.hub.store.call(turn.name, turn.args)
        if reply is None:
            return (
                "The call did not go through (network or server). "
                "Retry."
            )
        observation = reply.get("observation")
        self._watch_compare(turn, observation)
        self._learn(turn, observation)
        paged = self._page(turn, observation)
        self._learn_single_view(turn, observation)
        if turn.name == led.roles.get("cart"):
            self._after_cart(turn, observation)
        if turn.name == led.roles.get("cart_view"):
            led.cart_view = observation
            led.cart_seen_at = led.step
        if self.hub.store.is_checkout(turn.name, turn.args):
            led.buys += 1
            if not reply.get("finished"):
                led.refused_at = led.step
        return self._compose(turn, reply, observation, paged)

    def _watch_compare(self, turn: ToolRunner.Turn, observation: Any) -> None:
        led = self.led
        many = turn.many
        listed = turn.args.get(many) if many else None
        if not (isinstance(listed, list) and len(listed) > 1):
            return
        shown = to_json(observation)
        if any(str(i) in shown for i in listed):
            led.compare_ok = max(led.compare_ok, len(listed))
            return
        led.compare_bad = min(led.compare_bad or 999, len(listed))
        turn.extra += (
            "\n[Kyo] The side-by-side view came out empty for "
            f"{len(listed)} listings: pass at most "
            f"{self.hub.cart.compare_slots()} per call."
        )

    def _names_rows(self, value: Any) -> bool:
        values = value if isinstance(value, list) else [value]
        return bool(values) and all(
            isinstance(x, str) and x in self.led.rows for x in values
        )

    def _learn(self, turn: ToolRunner.Turn, observation: Any) -> None:
        led = self.led
        name = turn.name
        led.recent = (led.recent + [turn.mark])[-3:]
        led.id_args.update(
            (name, key)
            for key, value in turn.args.items()
            if self._names_rows(value)
        )
        self.hub.intake.absorb(observation)
        if name == led.roles.get("compare"):
            rows = self.hub.catalogue.records(observation)
            if rows:
                led.facts_from[name] = led.facts_from.get(name, False) or any(
                    find_facts(row) for _, row in rows
                )
        if name in turn.readers - {led.roles.get("search")}:
            self.hub.catalogue.remember_view(name, observation)

    def _page(self, turn: ToolRunner.Turn, observation: Any) -> str:
        turn.plain = {
            tool for job, tool in self.led.roles.items() if job != "search"
        }
        if turn.name in turn.plain or turn.ending:
            return ""
        return self.hub.catalogue.read_more_pages(
            turn.name, turn.args, observation
        )

    def _learn_single_view(
        self, turn: ToolRunner.Turn, observation: Any
    ) -> None:
        led = self.led
        roles = led.roles
        name = turn.name
        if not self.hub.catalogue.takes_one(name):
            return
        if name in self._roles_of("cart", "buy", "refuse", "ask"):
            return
        if not isinstance(observation, dict):
            return
        listing = self.hub.intake.row_id(observation)
        if not listing:
            return
        facts = led.facts_from
        facts[name] = facts.get(name, False) or bool(find_facts(observation))
        if name == roles.get("open") and not facts[name]:
            other = self._fact_viewer(name)
            if other:
                roles["open"] = other
        if (
            not facts[name]
            and name != roles.get("open")
            and not roles.get("offer")
        ):
            roles["offer"] = name
        if name == self.hub.judge.offer_view_tool():
            led.offer_seen[listing] = led.step

    def _fact_viewer(self, name: str) -> str | None:
        led = self.led
        takes_one = self.hub.catalogue.takes_one
        for tool in led.schemas:
            if tool == name or not takes_one(tool):
                continue
            if led.facts_from.get(tool) is False:
                continue
            busy = self._roles_of("cart", "buy", "refuse", "ask", "search")
            if tool not in busy:
                return tool
        return None

    def _after_cart(self, turn: ToolRunner.Turn, observation: Any) -> None:
        led = self.led
        judge = self.hub.judge
        args = turn.args
        self.hub.cart.record_cart(turn.name, args)
        led.cart_said = to_json(observation)
        led.recent = []
        first = next(
            (
                count
                for count in (
                    as_count(value)
                    for value in args.values()
                    if not isinstance(value, str)
                )
                if count is not None
            ),
            None,
        )
        if first is not None:
            for value in args.values():
                if not (isinstance(value, str) and value in led.rows):
                    continue
                if first > 0:
                    led.dropped.discard(value)
                elif not holds_id(observation, value):
                    led.dropped.add(value)
        lines = [
            i for i in judge.cart_listing_ids()
            if not judge.is_sold_out(i) and i not in led.dropped
        ]
        if len(lines) >= len(led.built):
            led.built = lines

    # the text handed back to the walking LLM

    def _compose(
        self,
        turn: ToolRunner.Turn,
        reply: dict[str, Any],
        observation: Any,
        paged: str,
    ) -> str:
        led = self.led
        judge = self.hub.judge
        text = self.hub.store.render(observation) + paged + turn.extra
        if turn.name not in turn.plain and not turn.ending:
            text += self.hub.catalogue.cheapest_in(
                [observation] + led.paged_obs
            )
        plain_text = text
        proof = judge.evidence_rows(observation)
        if proof:
            text += "\n" + proof
        text += self._zero_count_note(turn, observation)
        text += self._note_text(observation)
        refused = (
            self.hub.store.is_checkout(turn.name, turn.args)
            and not reply.get("finished")
            and self._refusal_names_sold(observation)
        )
        if refused:
            seen = judge.sold_report()
            if seen:
                text += (
                    "\n[Kyo read the cart lines anew after the refusal]\n"
                    + seen
                )
        swap = judge.swap_advice()
        if swap:
            text += "\n" + swap
        if led.queued:
            text += "\n[notes Kyo's own calls carried] " + " | ".join(
                led.queued
            )
            led.queued = []
        if turn.name in turn.readers:
            led.read_cache[turn.mark] = (led.version, led.step, plain_text)
        if reply.get("finished"):
            text += " [episode finished]"
        return text

    def _zero_count_note(
        self, turn: ToolRunner.Turn, observation: Any
    ) -> str:
        led = self.led
        if turn.name != led.roles.get("cart"):
            return ""
        values = turn.args.values()
        zeroed = any(
            as_count(value) == 0
            for value in values
            if not isinstance(value, str)
        )
        if not zeroed:
            return ""
        still = any(
            isinstance(value, str)
            and value in led.rows
            and holds_id(observation, value)
            for value in values
        )
        if not still:
            return ""
        return (
            "\n[Kyo] The line remained in the cart after a 0 count: this "
            "tool adds the count onto the line, so negatives take off units."
        )

    def _note_text(self, observation: Any) -> str:
        judge = self.hub.judge
        self.led.note_sold = False
        lines = judge.free_lines(observation)
        if not lines:
            return ""
        meaning = judge.absorb_note(lines)
        if not meaning:
            return ""
        return (
            "\n[note read] " + meaning
            + ". Update your choices for this change, then check out."
        )

    def _refusal_names_sold(self, observation: Any) -> bool:
        led = self.led
        if led.note_sold:
            return True
        said = self.hub.judge.shopper_said
        return any(line in said(observation) for line in led.sold_lines)


class Closer(Service):
    def finish(self, again: bool = False) -> str:
        led = self.led
        if led.finished or led.broke or led.steps_left() <= 0:
            return ""
        buy = led.roles.get("buy")
        refuse = led.roles.get("refuse")
        if buy and self._worth_checkout():
            outcome = self._checkout(buy, again)
            if outcome is not None:
                return outcome
        return self._decline(refuse) if refuse else ""

    def _over(self) -> bool:
        return self.led.finished or self.led.steps_left() <= 0

    def _worth_checkout(self) -> bool:
        led = self.led
        fresh = led.cart_view is not None and led.cart_seen_at >= led.cart_at
        if fresh:
            holding = bool(self.hub.judge.cart_listing_ids())
        else:
            holding = bool(led.cart_lines)
        if holding:
            return True
        return led.cart_touched and not fresh and led.steps_left() >= 2

    def _checkout(self, buy: str, again: bool) -> str | None:
        led = self.led
        faults: list[str] = []
        wanted = dict(led.brief.get("buy_args") or {})
        args = conform(wanted, led.schemas.get(buy), "args", faults)
        if faults:
            return None
        # a forced finish goes ahead even if the gate adds a requirement
        if led.steps_left() >= 2:
            self.hub.questions.run_gate("order")
        if self._over():
            return ""
        reply = self.hub.store.call(buy, args)
        led.trace.append({"kind": "forced_buy", "reply": clip(reply, 200)})
        if self._over():
            return ""
        if reply is None or not again:
            return None
        led.buys += 1
        led.refused_at = led.step
        self.hub.catalogue.observe(buy, reply.get("observation"))
        shown = self.hub.store.render(reply.get("observation"))
        return "checkout got turned down: " + shown

    def _decline(self, refuse: str) -> str:
        led = self.led
        faults: list[str] = []
        wanted = self.hub.store.quiet_args(
            refuse, dict(led.brief.get("refuse_args") or {})
        )
        args = conform(wanted, led.schemas.get(refuse), "args", faults)
        if faults:
            return ""
        if led.steps_left() >= 2:
            self.hub.questions.run_gate("end")
        if self._over():
            return ""
        reply = self.hub.store.call(refuse, args)
        led.trace.append({"kind": "forced_refuse", "reply": clip(reply, 200)})
        if reply is None or led.finished:
            return ""
        self.hub.catalogue.observe(refuse, reply.get("observation"))
        shown = self.hub.store.render(reply.get("observation"))
        return "a no-purchase ending was not accepted: " + shown

    def trim_history(self) -> None:
        chat = self.led.chat
        size = sum(len(to_json(entry.get("content"))) for entry in chat)
        if size <= HISTORY_LIMIT:
            return
        results = [
            index for index, entry in enumerate(chat)
            if entry.get("role") == "tool"
        ]
        for index in results[:max(1, len(results) // 2)]:
            self._shorten(chat[index])

    def _shorten(self, entry: dict[str, Any]) -> None:
        text = entry.get("content") or ""
        if len(text) <= 600:
            return
        found = re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', text)
        known = [
            ident for ident in dict.fromkeys(found) if ident in self.led.rows
        ]
        entry["content"] = (
            text[:400]
            + "...(earlier result cut down; the ids in it: "
            + ", ".join(known[:20]) + ")"
        )


class Phase:
    def __init__(self, hub: Episode):
        self.hub = hub
        self.led = hub.led


class OpenPhase(Phase):
    def handle(self) -> str:
        led = self.led
        self.hub.intake.read()
        led.trace.append({"kind": "brief", "brief": led.brief})
        opening = led.view.get("opening")
        if opening is not None:
            self.hub.intake.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + led.request
        if opening is not None:
            intro += (
                "\n\nWHAT THE STORE SHOWS FIRST:\n"
                + self.hub.store.render(opening)
            )
        intro += (
            "\n\nBegin with a search per item inside its category, then "
            "open several listings in full, so the interviewer can learn "
            "which properties such listings carry.\n\n"
            + self.hub.intake.status_line()
        )
        led.chat = [
            {"role": "system", "content": WALKER_PROMPT},
            {"role": "user", "content": intro},
        ]
        return "walk"


class WalkPhase(Phase):
    def __init__(self, hub: Episode, then: str):
        super().__init__(hub)
        self.then = then

    def limits(self) -> tuple[int, float]:
        return LOOP_CALLS, SPEND_HARD

    def handle(self) -> str:
        loop_cap, spend_cap = self.limits()
        self.walk(loop_cap, spend_cap)
        return self.then

    def _room_to_walk(self, loop_cap: int, spend_cap: float) -> bool:
        led = self.led
        stopped = led.finished or led.broke
        return (
            not stopped
            and not led.out_of_time()
            and led.steps_left() > 0
            and led.loop_calls < loop_cap
            and led.cost < spend_cap
        )

    def _say(self, role: str, content: Any, **extra: Any) -> None:
        self.led.chat.append({"role": role, "content": content, **extra})

    def walk(self, loop_cap: int, spend_cap: float) -> None:
        led = self.led
        idle = 0
        while self._room_to_walk(loop_cap, spend_cap):
            if self.hub.questions.should_ask():
                report = self.hub.questions.interview()
                if report:
                    self._say(
                        "user", report + "\n" + self.hub.intake.status_line()
                    )
                continue
            self.hub.closing.trim_history()
            led.loop_calls += 1
            message = self.hub.llm.consult(
                led.chat, [LIGHT, SPARE], 3000, self.hub.store.tool_specs()
            )
            if message is None:
                led.misses += 1
                if led.no_credit or led.calls >= ALL_CALLS or led.misses > 2:
                    break
                continue
            calls = self._wire_calls(message)
            text = self._text_of(message)
            if calls:
                idle = 0
                self._take_turn(calls, text)
                continue
            idle += 1
            self._say("assistant", text or "(no tool)")
            self._say(
                "user",
                "Go on by calling one of the tools.\n"
                + self.hub.intake.status_line(),
            )
            if idle >= 3:
                break

    def _wire_calls(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        wired = []
        for number, call in enumerate(message.get("tool_calls") or []):
            if isinstance(call, dict):
                wired.append(self._wire(number, call))
        return wired

    def _wire(self, number: int, call: dict[str, Any]) -> dict[str, Any]:
        fn = call.get("function")
        if not isinstance(fn, dict):
            fn = {}
        raw = fn.get("arguments")
        if isinstance(raw, dict):
            args = raw
        elif isinstance(raw, str) and raw.strip():
            args = parse_object(raw)
        else:
            args = {}
        fallback = f"k{self.led.loop_calls}_{number}"
        return {
            "id": str(call.get("id") or fallback),
            "type": "function",
            "function": {
                "name": str(fn.get("name") or ""),
                "arguments": to_json(args if isinstance(args, dict) else {}),
            },
        }

    @staticmethod
    def _text_of(message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        return to_json(content) if content else ""

    def _commits(self, calls: list[dict[str, Any]]) -> bool:
        roles = self.led.roles
        binding = {roles.get(job) for job in ("cart", "buy", "refuse")}
        binding.discard(None)
        return any(call["function"]["name"] in binding for call in calls)

    def _ask_first(self, calls: list[dict[str, Any]]) -> bool:
        led = self.led
        return bool(
            not led.asked
            and led.roles.get("ask")
            and led.free_messages() > 0
            and self._commits(calls)
            and led.steps_left() > 8
        )

    def _take_turn(self, calls: list[dict[str, Any]], text: str) -> None:
        self._say("assistant", text or None, tool_calls=calls)
        if self._ask_first(calls):
            for call in calls:
                self._say(
                    "tool",
                    "Held (no step spent): the shopper gets asked first.",
                    tool_call_id=call["id"],
                )
            report = self.hub.questions.interview()
            self._say(
                "user",
                (report or "The interviewer saw nothing to ask.")
                + "\n" + self.hub.intake.status_line(),
            )
            return
        self.led.trace.append({
            "kind": "turn",
            "text": text[:300],
            "calls": [call["function"] for call in calls],
        })
        replies = self.hub.runner.execute(calls)
        last = len(replies) - 1
        for index, (call_id, reply) in enumerate(replies):
            if index == last:
                reply += "\n" + self.hub.intake.status_line()
            self._say("tool", reply, tool_call_id=call_id)


class RewalkPhase(WalkPhase):
    def limits(self) -> tuple[int, float]:
        return self.led.loop_calls + 10, SPEND_HARD + 0.01


class SettlePhase(Phase):
    def handle(self) -> str | None:
        led = self.led
        if led.finished:
            return None
        turned = self.hub.closing.finish(again=True)
        if not turned or led.finished or led.broke or led.no_credit:
            return "close"
        if led.steps_left() <= 1 or led.out_of_time():
            return "close"
        # the store refused the ending: one short extra round handles it
        led.chat.append({
            "role": "user",
            "content": "[Kyo] Kyo tried to close the task, but "
            + turned
            + "\nDo what the store says, then finish it.\n"
            + self.hub.intake.status_line(),
        })
        return "rewalk"


class ClosePhase(Phase):
    def handle(self) -> None:
        if not self.led.finished:
            self.hub.closing.finish()
        return None


class Dispatcher:
    def __init__(self, hub: Episode):
        self.handlers = {
            "open": OpenPhase(hub),
            "walk": WalkPhase(hub, then="settle"),
            "settle": SettlePhase(hub),
            "rewalk": RewalkPhase(hub, then="close"),
            "close": ClosePhase(hub),
        }

    def run(self, phase: str | None) -> None:
        while phase is not None:
            phase = self.handlers[phase].handle()


class Episode:
    def __init__(self, problem: dict[str, Any]):
        self.led = Ledger(problem)
        self.llm = LlmService(self)
        self.store = StoreService(self)
        self.intake = RequestReader(self)
        self.catalogue = Catalogue(self)
        self.questions = QuestionDesk(self)
        self.judge = Judge(self)
        self.cart = CartDesk(self)
        self.closing = Closer(self)
        self.runner = ToolRunner(self)
        self.dispatcher = Dispatcher(self)

    def run(self) -> list[dict[str, Any]]:
        led = self.led
        if led.finished:
            return led.trace
        self.dispatcher.run("open")
        led.trace.append({
            "kind": "end",
            "finished": led.finished,
            "steps": led.step,
            "cost": round(led.cost, 5),
            "calls": led.calls,
        })
        return led.trace


def _failure_entry(error: BaseException) -> dict[str, str]:
    label = type(error).__name__
    return {"kind": "crash", "error": f"{label}: {error}"[:400]}


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Episode(problem_data)
    except Exception as exc:
        return [_failure_entry(exc)]
    trace = episode.led.trace
    try:
        return episode.run()
    except Exception as exc:
        trace.append(_failure_entry(exc))
        try:
            if not episode.led.finished:
                episode.closing.finish()
        except Exception as again:
            trace.append(_failure_entry(again))
        return trace


__all__ = ["agent_main"]
