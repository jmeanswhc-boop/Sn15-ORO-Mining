from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from threading import Lock
from typing import Any

from src.agent.proxy_client import ProxyClient

READER, SCOUT, SCOUT_BACKUP, READER_BACKUP = "anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "openai/gpt-6-sol"
EPISODE_CAP, CLOSE_RESERVE, FIRST_BILL, STOP_SECONDS = 0.055, 0.012, 0.5, 286
REQUEST_GAP, GAP_FLOOR, GAP_CAP = 0.75, 0.5, 4.0
LABELS, TIERS = ("key", "other", "title", "options", "description", "absent", "conflict"), {"met": 0, "weak": 1, "unknown": 2}
GRADE_STATE = {"key": "met", "other": "met", "title": "weak", "options": "weak", "conflict": "violated"}
NEGATION = re.compile(r"\b(?:not|no|never|without|except|nothing|but)\b")
JUDGE_BATCH, JUDGE_CALLS, COMBOS, LIST_SIZE = 4, 32, 4096, 4
CACHE_SHARE, CACHE_READ = 0.5, 0.08
CLASSES, FORMS = ("listing", "availability", "price", "context"), ("none of", "at least", "at most", "only", "fits")
SEAT_POLICY = {
    "reader": ("sonnet", "low"), "judge": ("sonnet", "low"), "update": ("sonnet", "low"), "refs": ("haiku", "low"),
    "scout": ("haiku", "off"), "toolmap": ("haiku", "off"), "listing_update": ("haiku", "off"),
    "scout_judge": ("haiku", "off"), "planner": ("sonnet", "low"), "searcher": ("haiku", "off"),
}
FILLER = {"the", "a", "an", "of", "with", "and", "for", "from", "to", "in", "on", "or", "as", "its", "that", "this", "same", "shown", "listing", "product", "item", "page",
          "another", "other", "one", "sold", "by", "version", "instead", "via"}
_model_proxy = ProxyClient(timeout=30, max_retries=1)
_store_proxy = ProxyClient(timeout=12, max_retries=1)

FACTS_RULES = """<role>You list the facts of a shopping order: what to buy, how many, the money limit, the objective and the shopper's rules for the process.</role>
<stakes>A wrong item, quantity, limit or rule here makes the task 0, and nothing later can recover it.</stakes>
<inputs><request> is the shopper's message. <ranges>, when present, lists the store's ranges. <review>, when present, lists faults code found in your earlier reading.</inputs>
<steps>
1. ITEM: each distinct product to buy, named in the request's words, with its quantity; a product the request names by one store range is one item, even when the range's name
joins two kinds ("<kind> and <kind>"). The quantity is the number of units the request asks to buy, 1 when it gives none. A later sentence that
names the product again or says who it is for adds no unit unless it asks for another one; numbers in the story are background, and a product named only to describe another listing
is not an item.
2. CAP: the spending limit for the whole order as total; a limit the request ties to one product, with that item's number. Copy the number exactly as written: it is already in the
store's price unit.
3. GOAL: cheapest when the request asks for the lowest price, the lowest total or the least spend; most or least with its measure for another extreme; no record otherwise. Use
order unless the request ties the objective to one item.
4. STOCK yes when the request wants only items available now, however it says so.
5. MESSAGES and MOVES: the message and step limits the request states.
6. DUTY: each message or action toward the shopper that the request asks for besides buying: before choosing (choose), right before buying (order, such as a yes on the basket or
the total), or before ending without a purchase (decline). Its message count is 1 when it needs a message of its own, 0 when the purchase or decline call itself carries it (such as
saying that nothing fits). A condition on which products qualify (a fit, a compatibility, an if-rule, any requirement) is never a duty, and neither is an invitation to ask
questions.
</steps>
<output_format>
Write only the records, one JSON array per line in the shapes below; a|b means one of those strings.
["ITEM", item_number, quantity, "product_words"]
["CAP", "total|item_number", amount]
["GOAL", "order|item_number", "cheapest|most|least", "measure_or_empty"]
["STOCK", "yes"]
["MESSAGES", number]
["MOVES", number]
["DUTY", duty_number, "choose|order|decline", message_count, "instruction_words"]
</output_format>
<example>
["ITEM", 1, 2, "<product words>"]
["CAP", "total", 100]
["STOCK", "yes"]
["DUTY", 1, "order", 1, "<instruction words>"]
</example>"""

MUSTS_RULES = """<role>You list what each item of a shopping order must be, in the shopper's own words.</role>
<stakes>A missed, changed or invented requirement makes the order 0; later stages judge only the requirements you record.</stakes>
<inputs><request> is the shopper's message; <items> gives the numbered products. <review>, if present, identifies faults in an earlier reading.</inputs>
<steps>
1. For each item, record every firm product requirement once, using the request's property words and exact values. Words that restate or soften a recorded requirement add no
record. The product's kind is the item, not a MUST, and so is the store range, category or department the request names; price, budget, quantity and stock are recorded
elsewhere. A value the request attaches to a property word inside the item's description ("<item> in the <value> <property>") is a MUST. If there are no MUST, LOOKUP or RELATE
records, return one NONE record.
2. Use one constraint form per MUST: "<value>" or "<value> | <value>" (one must hold); "none of <value> | <value>" (none may hold); "at least <number> <unit>" or
"at most <number> <unit>" (two MUSTs for a range); "only <value> | <value>"; "fits <model, size or standard>"; otherwise the request's exact constraint words.
Do not split a choice followed by "nothing outside these" into duplicate requirements. A negative word inside a store value remains part of that value. A value is the words the
request gives as the property's value, as a store would write them; "only", an explanation or a restatement around it stays out of the value, so when the request names the
value with the property ("the <value> <property>: <explanation>") and then explains it, the named word is the value, never the explanation.
3. A value taken from another listing is a MUST preserving the whole relation, property and reference, including any instruction to follow where the page points. Add a LOOKUP
with that property and the referred listing's seller, kind and every value the request uses to tell that listing apart (its form, size, pack, skin type or any other stated
value), copied in full. Those identifying values describe the referred listing only: they go into the LOOKUP and into the MUST's reference words, never into MUSTs of their own.
4. A rule joining two bought items is RELATE, not MUST: fit/compatibility, or a condition (including an exception). Put the constrained item second. Preserve the whole rule;
for a condition, also copy the value it asks of that second item.
5. Who the item is for, a use, a date in the shopper's plans or a story is background; an occasion, style or season the request says the item is for or must suit ("<item> for a
<value> occasion") is a MUST on that property. A firm condition remains a MUST even when phrased casually or followed by the shopper saying they will check it themselves. An
opposite phrase that merely restates a wanted value adds nothing.
</steps>
<output_format>
Write only records, one JSON array per line; copy values and numbers exactly. NONE is used alone when no requirement or relation applies.
["MUST", item_number, "property_words", "constraint_words"]
["LOOKUP", item_number, "property_words", "listing_description"]
["RELATE", "relation_id", first_item, second_item, "fit|if", "rule_words", "value_for_second_item"]
["NONE"]
</output_format>
<example>
["MUST", 1, "<property>", "same as the <property> shown on the page of <seller>'s <kind> with <property2> <value2> and <property3> <value3>"]
["LOOKUP", 1, "<property>", "<seller>'s <kind> with <property2> <value2> and <property3> <value3>"]
["RELATE", "r1", 1, 2, "if", "<conditional requirement on item 2>", "<value for item 2>"]
</example>"""

TOOL_RULES = """<role>You map a store's tools to shopping roles from their names, descriptions and schemas.</role>
<stakes>A wrong map sends basket writes, purchases or messages through the wrong tool, and the task scores 0.</stakes>
<inputs><tool_schemas> lists the store's tools and <opening> is the store's first view.</inputs>
<steps>
1. Map each role to the tool that does it: search finds listings; detail opens one listing's full record; page opens a further page or view of a listing; compare reads several
listings side by side; message writes to the shopper; write adds, changes or removes basket lines; basket shows the current basket; purchase places the order; decline ends without
buying. Leave out a role no tool does; one tool may serve two roles.
2. Give the arguments the purchase tool and the decline tool need to end the episode as a JSON object; for a tool that does both, the arguments that select each mode. Give the
arguments the write tool needs to add one unit of a listing, and to remove a listing from the basket, with "<id>" where the listing identifier goes.
3. For each tool that takes a listing identifier, give the slash path to that argument, with * for an array position.
</steps>
<output_format>
Write only the records, one JSON array per line in the shapes below; a|b means one of those strings.
["ROLE", "search|detail|page|compare|message|write|basket|purchase|decline", "tool_name"]
["MODE", "purchase|decline|add|remove", arguments_object]
["IDARG", "tool_name", "slash_path"]
</output_format>
<example>
["ROLE", "search", "<tool name>"]
["MODE", "purchase", {"<argument>": "<value>"}]
["IDARG", "<tool name>", "/<argument>"]
</example>"""

ANSWER_RULES = """<role>You read the newest shopper reply or store response and apply every change it makes to the order sheet.</role>
<stakes>A disclosed, replaced or dropped condition that you miss or invent makes the order 0.</stakes>
<inputs>The request, the order sheet lines, the selected basket and actual cart, your_call (the call that produced the response) and latest, the newest response.</inputs>
<steps>
1. Find every condition the response adds, replaces or drops. A refusal can carry several; apply all of them.
2. For a replaced property, drop the old requirement with its property and constraint copied exactly from the sheet, then add the new one. A requirement is the property the
response itself names, which can differ from the one your call asked about, and its constraint, with values and numbers copied from the response: "<value> | <value>", "none of
<value> | <value>", "at least <number> <unit>", "at most <number> <unit>", "only <value>", "fits <model or size>", or the response's own words. A value read from another listing
keeps the whole reference in the constraint, and that listing's description (its seller, its kind and every value the response states about it) goes in listing_description; leave
it empty otherwise. Never add a constraint the response does not state. The class is listing for any property of the product, which includes every property the shopper names for
it; availability for stock; price for money; context only for the shopper's own situation, which no listing states.
3. Record a changed cap, an explicitly changed quantity, a changed objective, each listing the response itself says is gone (a refusal about a line of the cart names that
line; the words of your own call never make a listing gone), and one sentence for any other new rule or procedure.
4. For a reply to a question, classify the answer: need (it states a requirement), free (it leaves the choice to you), ambiguous (it says it missed the question or answers nothing)
or crowded (mixed topics make it unclear). Silence is never free. A reply that leaves the asked property open drops nothing: a requirement is dropped only when the response names
it and withdraws it.
5. When the response changes nothing, say it is unchanged.
</steps>
<rules>Use only what the response states; a story detail alone never becomes a product property. A notice that a price, a stock state or a page changed withdraws no
requirement and removes no listing: record it as a NOTICE. Copy listing ids exactly as observed.</rules>
<output_format>
Write only the records, one JSON array per line in the shapes below; a|b means one of those strings.
["UPDATE", line, "add|drop", "property_words", "constraint_words", "listing|availability|price|context", "listing_description"]
["BUDGET", "total|each|line_number", amount]
["QUANTITY", line, positive_integer]
["GOAL", line, "any|cheapest|most|least", "measure"]
["GONE", "listing_id"]
["NOTICE", "sentence"]
["ANSWER", "need|free|ambiguous|crowded"]
["UNCHANGED"]
</output_format>
<example>
["UPDATE", 1, "drop", "<old property>", "<old constraint>", "listing", ""]
["UPDATE", 1, "add", "<property>", "<relation> the <property> shown on the page of <seller>'s <kind>", "listing", "<seller>'s <kind> with <property2> <value2>"]
["ANSWER", "need"]
</example>"""

WALK_RULES = """<role>You are MICKEY, the shopper's assistant in a store: you find, compare and buy what the order sheet asks for, using the store's tools.</role>
<stakes>The task scores 1 only if the final order meets every requirement, objective and procedure of the request, including needs the shopper reveals when asked and changes that
arrive later. A wrong listing, a missed hidden need, a wrong quantity or a skipped procedure makes it 0.</stakes>
<inputs>This conversation holds your calls and their results, each with its event number. Each user turn gives new events (code also opens listings and asks the shopper) and the
state: the order sheet with numbered requirements, each a property and its constraint ("a | b" any one, "none of a | b" none, "at least" and "at most" bounds, "only", "fits", or
the request's words; context requirements are never checked against listings), dropped requirements, budget and objectives, procedures, the cart, graded candidates per line with
their chance and price, remaining steps and messages, notices, calls that were not run and why, and judge findings. The store's tools are attached as functions.</inputs>
<steps>
1. Search every requested line before going deeper on one, with short queries: the product noun and one wanted value, two at most; words about stock, price or the
shopper's situation make a search miss. The state's search_notes shows what this store's search did: whether every word must match, which words it matched in titles or in
other fields, the words it never matched, the numbers it matched as written, and the queries tried with their hit counts; a query with no rows has a word to drop, and a query
with more rows than you can page through needs one more wanted value. Read the next page of a result that holds fitting rows, and never repeat a call you already made.
2. A listing is the requested kind of product only when its category label is the line's label in line_categories; a neighbouring label never counts. Compare listings by
their attributes: read each promising listing once with the state's reading_tool, one step per listing, and open its other views only for the listing you mean to buy or when
its attributes leave a requirement open. A requirement is met when the listing's attribute for that property states the wanted value in the same words (any key spelling,
singular or plural); a different value under that property fails it, and only when no attribute names the property can the title or description state it. Availability is
proven by the listing's stock field.
3. A requirement that takes its value from another listing is resolved first: find that listing by its seller and kind, open its page, and when the page sends you to a
further listing for that property, find and open that one too; the last page gives the value, and every page of the chain must be opened. Code asks the shopper about open
preferences once a few listings are opened; replies appear in dialogue and in the requirements. Use messages yourself only for the procedures the request sets.
4. For a cheapest line, find every listing that fits: page through the result, check candidates from the lowest price up, and take the first that meets everything. Treat a
most or least objective the same way for its measure. Otherwise any listing that meets everything is right.
5. Put a line's listing in the cart at its quantity as soon as you have read one that meets every requirement (for a cheapest, most or least line, once you have compared the
fitting set), remove anything else, and check the cart in the next result.
6. Follow every procedure the request sets, such as a confirmation message, once the cart is final.
7. Call the purchase tool when every line is chosen. Code then has an independent judge grade every opened listing against the requirements and orders the best-proven listing per
line; when it names a different listing, put exactly that one in the cart and call the purchase tool again. Call the decline tool only when your searches show that no listing can
fit and the request allows declining; code checks that with the same grades. When a call is not run, fix exactly what it says and continue. A refused checkout means the task
continues: apply the change it states and choose again.
8. When a response carries a notice or a changed condition, write NOTICE with its event number and its words.
</steps>
<rules>
- Use only ids and values that appear in results, copied exactly.
- Make one function call per reply.
- Never stop while steps remain; the episode ends only with an order or a decline.
</rules>
<output_format>
Write one or two sentences of reasoning, then any record below, then make the store call as one function call.
["NOTICE", event_number, "notice_words"]
</output_format>
<example>
["NOTICE", 7, "<the changed condition in its own words>"]
</example>"""

PLAN_RULES = """<role>You order the questions code will send the shopper, from the properties the shopper's request points at and the properties the item's listings carry.</role>
<stakes>The shopper holds a few unwritten preferences, each about one property of one item, and reveals one only when a question names that property. An unrevealed preference
makes the order 0, and every question costs one of a few messages; a question about a stated property, about price, stock or offer details (seller, brand, model or option
codes, warranty), or about a second spelling of an asked property wastes its message and leaves the preference hidden.</stakes>
<inputs><order> holds the request and each item with its musts; <menus> lists, for each item, the attribute keys its listings carry, most carried first, with how many
listings carry each and some values; <asked> lists the properties already asked for each item, never to be asked again; <room> is how many questions can be sent.</inputs>
<steps>
1. First, each thing the request asks to check with the shopper, named by the property it is about, with the request words that ask for it.
2. Then a property a request sentence clearly points at without giving its value (something the shopper leaves to you or to a later answer, what went wrong last time, what
the shopper will judge the item by), with that sentence quoted; a sentence that only tells a story, a use or an occasion points at nothing, and a pointed-at property that few
of the item's listings carry comes after the carried keys.
3. Then every other menu key of the item, ranked by what a buyer of that kind of product decides first: the property that tells one variety of the product from another (its
type, flavour, pattern, style, cut, fit, sleeve or collar, material, colour), then its size or capacity, then the rest; a key with a few named values outranks a key of
measurements, counts or weights, and packaging, storage, pack counts, warranty, condition, seller, brand, model and option codes come last or are left out. Prefer keys most
listings carry, and once five or more listings were read leave out a key seen on only one or two. Ask each property once (keys that mean the same property count as one, but a
key with an extra word is another property: a frame material is not the material, a lid colour is not the colour) and skip a property a must already settles. When the menu is
short, add the properties such a buyer chooses first, by plain name.
4. Name each property with the item's menu key that means it, spelled as the store spells it, or by its plain name when no key means it, and give the item's plain product name.
</steps>
<output_format>
Write only the records, best first, at most eight per item, one JSON array per line in the shape below; request_words is the request sentence that asks for or points at the
property, empty for a menu key. Code sends them in this order, a few per message, and stops when the room runs out.
["ASK", item_number, "key", "product_name", "request_words"]
</output_format>
<example>
["ASK", 1, "<key>", "<product>", "<the request words that ask for it>"]
["ASK", 1, "<key>", "<product>", ""]
</example>"""

JUDGE_RULES = """<role>You grade how each opened listing states each requirement of the order, quoting the listing's own words.</role>
<stakes>Code builds the order only from requirements your grades show met and checks every quote against the listing. A met grade taken from another field, the title or a
near word while the listing's own field says otherwise, a conflict on a listing whose own field holds the wanted value, or a wrong pair verdict sends a wrong order, and the task
scores 0.</stakes>
<inputs><order> holds the lines with their numbered requirements; they replace the original request wherever a later change dropped or replaced something (<changes>).
<relations>, when present, gives rules joining two lines and the listing pairs to judge; <measure>, when present, names the property the order ranks by and the unit to give.
<listings> holds, for each listing to grade, its recorded views and own_fields: for each "line.requirement" number, the listing's fields whose name is that requirement's
property in any spelling, empty when the listing has none.</inputs>
<steps>
1. For each listing, decide which lines, if any, ask for the product it is. When the listing has a category, aisle or department field, that field alone decides: yes when it
names the line's product or the store range the request names, no when it names another kind of product, whatever the title says. Only a listing without such a field is judged
by its title. One listing can be the product of several lines.
2. For each line the listing is, grade every numbered requirement. When own_fields lists fields for it, those fields alone decide, whatever other fields, the title or the
description say; a field whose name only contains the property's name among other words is another field. For "fits <model or size>" the own field is whatever states this
listing's compatibility, a field or a sentence of its description or page; a sentence about another part does not describe this listing.
3. Judge the constraint against the deciding words. "<value> | <value>" and "only <value> | <value>": met when one wanted value appears, its words in the same order, alone or
inside a longer value or list; only the same words count, in any letter case, spelling variant or plural, never an abbreviation, a synonym, a translation or a range that would
include the value; a value that reads like a bound but is not written "at least" or "at most" (such as "up to <number>") is matched by its words. "only" is broken only by another
value and none of the wanted ones. "none of <value> | <value>": met when the deciding words state the property and no excluded value appears in them; broken when one appears. "at
least" or "at most" a number: compare the stated number, converting the unit when the listing uses another. "fits <model>": met when the compatibility statement names the model;
broken when it names models and not this one. A reference (referenced_value and that listing): judge against that value; absent when it is missing. When several own fields
disagree, the requirement is met if any of them meets it.
4. Grade where the listing meets the constraint. key: an own field meets it. other: there is no own field and another field meets it. title: only the title meets it. options: only
a sale option or variant meets it. description: only free description text meets it. absent: nothing states the property. conflict: the own fields break the constraint, or, with
no own field, the title does.
5. For every grade except absent, name the field and quote the shortest exact words that decide it, at most six: the wanted value for a met grade, the breaking value for a
conflict.
6. Give each listing's price and stock as the listing states them. With <measure>, give each listing's value of that property converted to the named unit, quoting its words.
7. For each pair in <relations>: yes when the listings' own words show the rule holds (for a conditional rule, also when its condition does not hold), no when they show it fails,
unknown when they do not show it. Quote the deciding words.
</steps>
<output_format>
Write only the records, one JSON array per line in the shapes below; a|b means one of those strings. Write a KIND row for each line a listing is; a listing with no KIND row is no
line's product.
["KIND", line_number, "listing_id"]
["GRADE", line_number, "listing_id", requirement_number, "key|other|title|options|description|absent|conflict", "field_name", "exact_words"]
["OFFER", "listing_id", price_number_or_null, "in|out|unknown", "price_field_name"]
["MEASURE", "listing_id", number_in_named_unit, "exact_words"]
["PAIR", "relation_id", "first_listing_id", "second_listing_id", "yes|no|unknown", "exact_words"]
</output_format>
<example>
["KIND", 1, "<listing id>"]
["GRADE", 1, "<listing id>", 1, "key", "<field name>", "<value words>"]
["OFFER", "<listing id>", 12.5, "in", "<field name>"]
</example>"""


SEARCH_RULES = """<role>You write the next query for a store's keyword search, from what the search must find and what earlier searches showed about this store.</role>
<stakes>A query the store matches nothing with wastes a step, and a query that matches many pages hides the listing; the steps lost here are missing at checkout, and the task
scores 0.</stakes>
<inputs><goal> says what the search must find: a product kind with the values it must carry, or the description of one listing (its seller, kind and stated values), with any
words suggested for it. <notes> gives the facts code measured about this store's search: whether every word must match, which words of earlier queries the store matched and
where (in titles, in other fields), words it never matched, numbers it matched as written, rows per page, and the queries tried with their hit counts.</inputs>
<steps>
1. Use two to five words: the product noun and the most selective words of the goal (a seller or brand name, a distinctive value), spelled as matched rows wrote them when the
notes show a spelling. When the goal lists values, the query carries at least one of them; the product noun alone is sent only when every value is listed as never matched.
Leave out words the notes list as never matched, words about price, stock or the shopper, and numbers with units unless the notes show the store matched such a number as
written.
2. When every word must match, prefer fewer words; when a query returned more rows than two pages, add one more distinctive word. Never repeat a query already tried.
3. For a listing description, start from its seller or brand and its kind; add one stated value only when the seller and kind alone returned more than a page.
</steps>
<output_format>
Write only one record in the shape below.
["QUERY", "words"]
</output_format>
<example>
["QUERY", "<seller> <product noun>"]
</example>"""

OPEN_RULES = """<role>You pick, from search result rows, the listings worth opening for a goal.</role>
<stakes>Every opened listing costs a step; opening the wrong rows leaves the right listing unread and the order wrong, and the task scores 0.</stakes>
<inputs><goal> says what is wanted: a product kind with its required values, or the description of one listing. <rows> lists the result rows as short cards with their ids.
<room> is how many may be opened.</inputs>
<steps>
1. Rank the rows by how likely their full record meets the goal: the right kind of product first, then the rows whose words carry the goal's values or match the description's
seller and values; a row that contradicts a required value is left out.
2. Give at most <room> ids, best first; give none when no row can be the goal.
</steps>
<output_format>
Write only the records, one JSON array per line in the shape below.
["OPEN", "listing_id"]
</output_format>
<example>
["OPEN", "<listing id>"]
</example>"""

REF_RULES = """<role>You find the listing a requirement refers to and read the value that listing shows.</role>
<stakes>The requirement asks for the same value as that listing, or a different one. A wrong listing or a misread value makes the order 0.</stakes>
<inputs><reference> gives the property and the description of the referred listing; <listings> holds candidate listings, each with its own fields for the property, the
sentences of its page that mention the property and its recorded views.</inputs>
<steps>
1. Pick the listing whose seller, kind and stated values match the description; none when no listing matches. A listing whose own values contradict the description (another
pack type, another size, another form) is not the listing, even when its page points to a further listing.
2. Read the value that listing shows for the property in its own field (own_fields) and quote it. A sentence of the page that names the property and sends you to another
listing for it (property_mentions) is a pointer: give the words that identify that other listing instead, whatever the own field says. A sentence that pairs or maps values
("<a> goes with <b>", a table of pairs) gives the answer for the page's own value: the value paired with it, quoted from that sentence. A value the description uses to identify
the listing is never the answer.
3. When no listing matches, give the search that would find the described listing: the seller's name and one or two values it states, written as the store writes values, with
no property names and no other words.
</steps>
<output_format>
Write only the records, one JSON array per line in the shapes below; a|b means one of those strings.
["REF", "listing_id|none", "value", "exact_words"]
["LINK", "words_that_identify_the_listing_it_points_to"]
["SEARCH", "query_words"]
</output_format>
<example>
["REF", "<listing id>", "<value>", "<exact words>"]
</example>"""


def serial(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def cell(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return value if isinstance(value, str) else "" if value is None else ",".join(map(cell, value)) if isinstance(value, list) else serial(value)


def records(text: str, rules: str) -> list[list[str]]:
    specs = (
        [w.strip(' "') for w in spec.split("]")[0].strip("[ ").split(",")] if spec.lstrip().startswith("[") else spec.split("(")[0].split()
        for spec in rules.split("<output_format>")[-1].split("</output_format>")[0].splitlines()
    )
    shapes = {words[0]: len(words) for words in specs if words and len(words[0]) > 2 and words[0].isalpha() and words[0].isupper()}
    blocks: list[list[list[str]]] = [[]]
    for line in str(text or "").splitlines():
        line, found = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line.strip().strip("`").strip()).rstrip(","), []
        if line.startswith("["):
            try:
                got = json.loads(line)
            except ValueError:
                got = None
            if isinstance(got, list) and got:
                found = [[cell(v) for v in row] for row in (got if all(isinstance(v, list) for v in got) else [got]) if row and isinstance(row[0], str)]
        elif "\t" in line:
            found = [line.split("\t")]
        elif line.split(None, 1)[:1] and line.split(None, 1)[0].upper() in shapes:
            found = [line.split(None, shapes[line.split(None, 1)[0].upper()] - 1)]
            found[0][0] = found[0][0].upper()
        if found:
            blocks[-1] += [row + [""] * (shapes.get(row[0], 0) - len(row)) for row in found]
        elif line and blocks[-1]:
            blocks.append([])
    rows, keys = [], set()
    for block in reversed([b for b in blocks if b]):
        if not keys & {tuple(row[:3]) for row in block}:
            rows, keys = block + rows, keys | {tuple(row[:3]) for row in block}
    return rows


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and not re.fullmatch(r"\s*-?[\d.]+\s*", value):
        found = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        value = found.group(0) if found else value
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError, OverflowError):
        return None


def constraint_form(words: str) -> tuple[str, list[str]]:
    head = next((f for f in FORMS if words.lower().startswith(f + " ")), "")
    return head, [v.strip() for v in words[len(head) :].split("|") if v.strip()]


def pointer(value: Any, path: str) -> Any:
    if path == "?":
        return None
    try:
        for part in path.strip("/").split("/") if path.strip("/") else []:
            key = part.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
        return value
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def leaves(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [v for x in value.values() for v in leaves(x)]
    if isinstance(value, list):
        return [v for x in value for v in leaves(x)]
    return [value]


def plain(value: Any) -> str:
    return " ".join(re.sub(r'["{}\[\],:]', " ", value if isinstance(value, str) else serial(value)).lower().split())


def fields(node: Any, key: str = "") -> list[tuple[str, str]]:
    if isinstance(node, dict):
        return [f for k, v in node.items() for f in fields(v, str(k))]
    if isinstance(node, list) and not all(v is None or isinstance(v, (str, int, float, bool)) for v in node):
        return [f for v in node for f in fields(v, key)]
    return [(key, cell(node))] if key else []


def terms(text: str) -> set[str]:
    return {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def singular(word: str) -> str:
    if len(word) <= 3 or not word.endswith("s") or word.endswith("ss"):
        return word
    return word[:-3] + "y" if word.endswith("ies") else word[:-2] if word.endswith(("sses", "ches", "shes", "xes")) else word[:-1]


def spaced(text: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)


def nouns(text: Any) -> set[str]:
    return {singular(w) for w in re.findall(r"[a-z0-9]+", spaced(str(text)).lower()) if len(w) > 2}


def key_words(text: Any) -> str:
    return " ".join(singular(w) for w in re.findall(r"[a-z0-9]+", spaced(str(text)).lower()))


def stem(word: str) -> str:
    word = singular(word)
    for suffix in ("ing", "ery", "ed", "er", "y", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
    return word


def value_words(text: Any) -> str:
    return " ".join(stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower()))


def row_terms(row: dict[str, Any]) -> set[str]:
    return terms(" ".join(cell(v) for v in row.values()))


def spoken(text: str) -> str:
    text = re.sub(r"([,;])", r" \1 ", str(text).lower())
    return " ".join(w.rstrip(".:") or w for w in re.sub(r"[^a-z0-9%.:,;']+", " ", text).split())


def headline(row: dict[str, Any], ident: str) -> str:
    return max((v for v in row.values() if isinstance(v, str) and v != ident and len(v) < 300), key=len, default="")


def phrase_in(value: str, text: str) -> bool:
    want = [singular(w) for w in re.findall(r"[a-z0-9]+", value.lower())]
    items = [[singular(w) for w in re.findall(r"[a-z0-9]+", part.lower())] for part in re.split(r"[,;/|]", text)]
    return bool(want) and "+" not in text and any(have[i : i + len(want)] == want for have in items for i in range(len(have) - len(want) + 1))


def leaves_lists(node: Any) -> list[list]:
    found = [node] if isinstance(node, list) else []
    for child in node.values() if isinstance(node, dict) else node if isinstance(node, list) else ():
        found += leaves_lists(child)
    return found


def find_record(node: Any, ident: str, path: str = "") -> tuple[str, dict] | None:
    if isinstance(node, dict):
        inner = next((hit for key, child in node.items() if (hit := find_record(child, ident, path + "/" + str(key)))), None)
        return inner or ((path, node) if ident in [v for v in node.values() if isinstance(v, str)] else None)
    if isinstance(node, list):
        return next((hit for i, child in enumerate(node) if (hit := find_record(child, ident, path + "/" + str(i)))), None)
    return None


PART = re.compile(r"^\s*([A-Za-z][\w .\-/()']{0,40}?)\s*[:=]\s*(.*)$", re.S)


def record_of(text: str) -> dict[str, Any] | list[dict[str, Any]] | None:
    lines = [p for p in text.split("\n") if p.strip()]
    rows = [record_of(p) for p in lines] if len(lines) >= 2 else []
    if rows and all(isinstance(r, dict) for r in rows):
        return rows
    for sep in (" | ", "\n", "; ", ", "):
        pairs: list[list[str]] = []
        for part in text.split(sep):
            found = PART.match(part)
            if found and len(found.group(1).split()) <= 4:
                pairs.append([found.group(1).strip(), found.group(2).strip()])
            elif pairs and part.strip():
                pairs[-1][1] += sep + part
            elif part.strip():
                pairs = []
                break
        if len(pairs) >= (2 if sep in (" | ", "\n") else 3) and len({k.lower() for k, _ in pairs}) == len(pairs):
            record: dict[str, Any] = {}
            for key, value in pairs:
                inner = record_of(value) if sep != ", " else None
                if isinstance(inner, dict):
                    record.update({k: v for k, v in inner.items() if k not in record})
                else:
                    record[key] = int(value) if re.fullmatch(r"-?\d+", value) else float(value) if re.fullmatch(r"-?\d+\.\d+", value) else value
            return record
    return None


def structured(value: Any, top: bool = True) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            inner = record_of(v) if isinstance(v, str) and len(v) > 8 and PART.match(v) else None
            if isinstance(inner, dict) and len(inner) >= 3:
                out.update({ik: iv for ik, iv in inner.items() if ik not in out})
            elif isinstance(inner, list) and top:
                out[k] = inner
            else:
                out[k] = structured(v, False) if isinstance(v, (dict, list)) else v
        return out
    if isinstance(value, list):
        return [structured(v, False) if isinstance(v, (dict, list)) else (record_of(v) or v) if isinstance(v, str) and len(v) > 8 else v for v in value]
    return (record_of(value) or value) if isinstance(value, str) and len(value) > 8 else value


def placeholders(node: Any, known: set[str]) -> Any:
    if isinstance(node, dict):
        return {k: placeholders(v, known) for k, v in node.items()}
    if isinstance(node, list):
        lines = [placeholders(v, known) for v in node]
        return lines[:1] if lines and isinstance(lines[0], dict) and "<id>" in serial(lines[0]) else lines
    return "<id>" if isinstance(node, str) and node in known else node


def small_view(value: Any, room: int = 1800) -> str:
    if len(serial(value)) <= room:
        return serial(value)
    if isinstance(value, dict):
        pieces = []
        share = max(60, room // max(1, len(value)))
        for key, child in value.items():
            pieces.append(str(key) + ":" + small_view(child, share))
        return "{" + ",".join(pieces) + "} [view shortened; original retained]"
    if isinstance(value, list):
        share = max(80, room // max(1, min(len(value), 12)))
        return "[" + ",".join(small_view(x, share) for x in value[:12]) + f"] [{len(value)} entries; original retained]"
    return str(value)[:room] + " [shortened]"


def schema_fault(value: Any, schema: dict[str, Any], path: str = "args") -> str:
    options = schema.get("enum")
    if isinstance(options, list) and value not in options:
        return path + " has an invalid enum value"
    types = schema.get("type")
    types = types if isinstance(types, list) else [types] if types else []
    kind = (
        "null"
        if value is None
        else "boolean"
        if isinstance(value, bool)
        else "integer"
        if isinstance(value, int)
        else "number"
        if isinstance(value, float)
        else "string"
        if isinstance(value, str)
        else "array"
        if isinstance(value, list)
        else "object"
        if isinstance(value, dict)
        else "unknown"
    )
    if types and kind not in types and not (kind == "integer" and "number" in types):
        return path + " has the wrong type"
    if "const" in schema and value != schema["const"]:
        return path + " differs from its required constant"
    for branch in ("anyOf", "oneOf", "allOf"):
        options = schema.get(branch)
        if options:
            matches = sum(not schema_fault(value, child, path) for child in options)
            if (branch == "anyOf" and not matches) or (branch == "oneOf" and matches != 1) or (branch == "allOf" and matches != len(options)):
                return path + " does not satisfy " + branch
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        if any(k not in value for k in schema.get("required", [])):
            return path + " lacks a required field"
        if schema.get("additionalProperties") is False and any(k not in props for k in value):
            return path + " has an extra field"
        for key, child in value.items():
            fault = schema_fault(child, props.get(key, {}), path + "/" + key)
            if fault:
                return fault
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            return path + " has the wrong array length"
        for child in value:
            fault = schema_fault(child, schema.get("items", {}), path + "/*")
            if fault:
                return fault
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", float("inf")):
            return path + " has the wrong text length"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value) or value < schema.get("minimum", -float("inf")) or value > schema.get("maximum", float("inf")):
            return path + f" must be between {schema.get('minimum', 'any')} and {schema.get('maximum', 'any')}"
    return ""


def seat_policy(role):
    family, effort = SEAT_POLICY.get(role, ("sonnet", "low"))
    models = (SCOUT, SCOUT_BACKUP) if family == "haiku" else (READER, READER_BACKUP)
    return models, {"enabled": False} if effort == "off" else {"effort": effort}


class InferenceBook:
    def __init__(self):
        self.start = time.monotonic()
        self.spent = 0.0
        self.ratio, self.markup = 0.5, 1.15
        self.calls, self.next_start, self.gap = 0, 0.0, REQUEST_GAP
        self.reserve, self.floor, self.billing = CLOSE_RESERVE, 0.0, {}
        self.last_message: dict[str, Any] | None = None
        self.lock = Lock()
        self.log: list[dict[str, Any]] = []

    def remaining(self) -> float:
        return max(0.0, STOP_SECONDS - (time.monotonic() - self.start))

    def bill_ratio(self, role: str) -> float:
        family = seat_policy(role)[0][0]
        seen = [r for other, r in self.billing.items() if seat_policy(other)[0][0] == family]
        return self.billing.get(role) or (sum(seen) / len(seen) if seen else FIRST_BILL)

    def admits(self, role: str, estimate: float, final: bool) -> bool:
        closing, held = final and role == "judge", self.floor if final else max(self.floor, self.reserve)
        return self.spent + estimate <= EPISODE_CAP and (closing or self.spent + estimate * self.bill_ratio(role) <= EPISODE_CAP - held)

    def body(self, role, prompt, material, tokens, model, tools=None, messages=None, stable=""):
        anthropic = model.startswith("anthropic/")
        parts = [{"type": "text", "text": "<inputs>\n" + stable, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": material + "\n</inputs>"}]
        system = [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}] if anthropic else prompt
        sent = [{"role": "system", "content": system}, {"role": "user", "content": parts if stable and anthropic else "<inputs>\n" + stable + material + "\n</inputs>"}]
        if messages and anthropic:
            sent = [
                dict(m, content=[{"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}])
                if i in (0, len(messages) - 2, len(messages) - 1) and isinstance(m.get("content"), str) and m["content"]
                else m
                for i, m in enumerate(messages)
            ]
        return {
            "model": model,
            "temperature": 0,
            "max_tokens": tokens,
            "reasoning": seat_policy(role)[1],
            "messages": sent if not messages or anthropic else messages,
            **({"tools": tools, "tool_choice": "auto"} if tools else {}),
        }

    def price(self, body):
        text = serial(body)
        bound = sum(self.ratio if ord(c) < 128 else len(c.encode("utf-8")) for c in text)
        incoming, outgoing = (0.125e-6, 0.5e-6) if body["model"] in (SCOUT, SCOUT_BACKUP) else (2.5e-6, 10e-6)
        if body["model"] in (SCOUT, SCOUT_BACKUP) and bound >= (272000 if body["model"] == SCOUT_BACKUP else 100000):
            incoming, outgoing = (0.25e-6, 0.75e-6) if body["model"] == SCOUT_BACKUP else (0.625e-6, 2.5e-6)
        cached = body["model"].startswith("anthropic/") and "cache_control" in text
        return (bound * incoming * (1 - CACHE_SHARE + CACHE_SHARE * CACHE_READ if cached else 1) + body["max_tokens"] * outgoing) * self.markup

    def fits(self, role, prompt, material, tokens, stable="", final=False) -> bool:
        body = self.body(role, prompt, material, tokens, seat_policy(role)[0][0], stable=stable)
        with self.lock:
            return self.admits(role, self.price(body), final)

    def pace(self, margin=0):
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_start - now)
            if self.remaining() <= delay + margin + 2:
                return False
            self.next_start = max(now, self.next_start) + self.gap
        if delay:
            time.sleep(delay)
        return True

    def rate(self, limited: bool, attempt: int = 0) -> float:
        with self.lock:
            self.gap = min(GAP_CAP, self.gap * 2) if limited else max(GAP_FLOOR, self.gap / 2)
        return random.uniform(0, min(4.0, 0.5 * 2**attempt, max(0.0, self.remaining() - 40))) if limited else 0.0

    def complete(self, role: str, prompt: str, material: str, final: bool = False, tokens: int = 1024, proxy=None, tools=None, messages=None, stable: str = "") -> str:
        proxy = _model_proxy if proxy is None else proxy
        margin = 0 if final else 32
        for model in seat_policy(role)[0]:
            for attempt in range(4):
                body = self.body(role, prompt, material, tokens, model, tools, messages, stable)
                with self.lock:
                    estimate, priced_markup = self.price(body), self.markup
                    if not self.admits(role, estimate, final) or self.remaining() <= margin + 2 or (self.calls >= 120 and not final):
                        self.log.append({"kind": "inference_held", "role": role, "estimate": estimate, "spent": self.spent, "final": final})
                        return ""
                    self.spent += estimate
                    self.calls += 1
                if not self.pace(margin):
                    with self.lock:
                        self.spent -= estimate
                        self.calls -= 1
                    return ""
                proxy.timeout = min(30, max(1, self.remaining() - margin - 1))
                started = time.monotonic()
                try:
                    reply = proxy.post_verbose("/inference/chat/completions", body)
                    data, fault = reply.data if isinstance(reply.data, dict) else {}, reply.error or {}
                except Exception as error:
                    data, fault = {}, {"kind": type(error).__name__}
                    self.log.append({"kind": "inference_transport", "role": role, "error": type(error).__name__})
                status = fault.get("status")
                usage = data.get("usage") or {}
                billed = (
                    sum(
                        v
                        for v in (
                            finite_number(usage.get("cost")),
                            finite_number((usage.get("cost_details") or {}).get("upstream_inference_cost")) if usage.get("is_byok") else None,
                        )
                        if v is not None
                    )
                    or None
                )
                status = data["error"].get("code", status) if isinstance(data.get("error"), dict) else status
                code = finite_number(status)
                rejected = billed is None and code is not None and 400 <= code < 500 and code != 408
                with self.lock:
                    self.spent -= estimate if rejected else 0.0
                    if not rejected:
                        self.billing[role] = max(self.billing.get(role, 0.0), billed / estimate if billed is not None and billed >= 0 else 1.0)
                    if billed is not None and billed >= 0:
                        self.spent += billed - estimate
                        self.markup = max(self.markup, billed / max(estimate / priced_markup, 1e-9))
                    used = usage.get("prompt_tokens")
                    if isinstance(used, int):
                        self.ratio = max(self.ratio, used / max(1, len(serial(body))) * 1.1)
                choice = (data.get("choices") or [{}])[0]
                reply = choice.get("message") or {}
                self.last_message = reply
                message = reply.get("content") if isinstance(reply.get("content"), str) else ""
                cut = choice.get("finish_reason") == "length"
                if cut:
                    self.log.append({"kind": "inference_cut", "role": role, "model": model, "tokens": tokens})
                    if tools:
                        return ""
                for call in (reply.get("tool_calls") or [])[:1]:
                    function = call.get("function") or {}
                    try:
                        arguments = json.loads(function.get("arguments") or "{}")
                    except ValueError:
                        arguments = function.get("arguments")
                    message += "\n" + serial(["CALL", function.get("name", ""), arguments])
                if message:
                    self.rate(False)
                    self.log.append(
                        {
                            "kind": "inference",
                            "role": role,
                            "model": model,
                            "spent": self.spent,
                            "charge": billed if billed is not None and billed >= 0 else estimate,
                            "accounting": "reported" if billed is not None and billed >= 0 else "estimated",
                            "seconds": round(time.monotonic() - started, 4),
                            "cut": cut,
                        }
                    )
                    return message
                self.log.append(
                    {
                        "kind": "inference_error",
                        "role": role,
                        "status": status,
                        "source": "proxy" if fault.get("kind") == "upstream" and not str(fault.get("body", "")).lstrip().startswith("{") else fault.get("kind"),
                        "model": model,
                        "charge": 0.0 if rejected else billed if billed is not None and billed >= 0 else estimate,
                        "accounting": "rejected" if rejected else "reported" if billed is not None and billed >= 0 else "estimated",
                    }
                )
                if code == 429 and attempt < 3:
                    time.sleep(self.rate(True, attempt))
                    continue
                if status not in (None, 408, 429, 500, 502, 503, 504) or attempt:
                    break
                time.sleep(min(0.25 + random.random() * 0.4, max(0.0, self.remaining() - margin)))
        return ""

    def complete_parallel(self, jobs):
        if len(jobs) != 2:
            return None
        with self.lock:
            estimate = sum(
                max(
                    self.price(self.body(role, prompt, material, tokens, model))
                    for model in seat_policy(role)[0]
                )
                for role, prompt, material, tokens in jobs
            )
            admitted = self.admits("", estimate, False) and self.remaining() > 66 and self.calls + len(jobs) <= 120
            self.log.append({"kind": "parallel_admission", "admitted": admitted, "jobs": len(jobs), "estimate": estimate})
        if not admitted:
            return None

        def infer(job):
            role, prompt, material, tokens = job
            proxy = ProxyClient(timeout=30, max_retries=1) if isinstance(_model_proxy, ProxyClient) else _model_proxy
            try:
                return self.complete(role, prompt, material, final=seat_policy(role)[0][0] == READER, tokens=tokens, proxy=proxy)
            except Exception as error:
                self.log.append({"kind": "parallel_worker_error", "role": role, "error": type(error).__name__})
                return ""

        with ThreadPoolExecutor(max_workers=2) as workers:
            return list(workers.map(infer, jobs))


class Mickey:
    def __init__(self, problem: dict[str, Any]):
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.request = str(self.policy.get("query") or "")
        self.walk_tools = [t for t in self.policy.get("tools", []) if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.tools = {t["function"]["name"]: t["function"] for t in self.walk_tools}
        self.roles: dict[str, str] = {}
        self.modes: dict[str, dict[str, Any]] = {}
        self.ident_args: dict[str, list[str]] = {}
        self.lines: dict[int, dict[str, Any]] = {}
        self.duties: dict[int, dict[str, Any]] = {}
        self.relations: dict[str, dict[str, Any]] = {}
        self.budget: float | None = None
        self.each_cap: float | None = None
        self.order_goal, self.order_measure = "any", ""
        self.max_messages: int | None = None
        self.used_messages = 0
        self.events: list[dict[str, Any]] = []
        self.actual_cart: dict[str, int] | None = None
        self.cart_event = self.cart_epoch = self.changed_at = self.judgments = self.walker_messages = 0
        self.duty_told = self.searched_out = False
        self.need_kind: dict[str, str] = {}
        self.need_parts: dict[str, tuple[str, str, str]] = {}
        self.faults: list[str] = []
        self.fresh_values: set[int] = set()
        self.dead_words: set[str] = set()
        self.starved = False
        self.dropped: list[str] = []
        self.grades: dict[str, dict[str, Any]] = {}
        self.price_field: str | None = None
        self.cache, self.cache_at = {}, -1
        self.add_template: dict[str, Any] | None = None
        self.discovered = False
        self.pairs: dict[tuple[str, str, str], str] = {}
        self.carted: set[str] = set()
        self.references: dict[str, str | None] = {}
        self.lookups: dict[str, str] = {}
        self.referenced: dict[str, str] = {}
        self.chains: dict[str, dict[str, Any]] = {}
        self.goal_queries: dict[str, int] = {}

        self.basket_shape: tuple[str, str, str] | None = None
        self.exhausted: set[str] = set()
        self.gone: set[str] = set()
        self.last_actions: dict[str, tuple[int, int]] = {}
        self.not_run: list[dict[str, Any]] = []
        self.notices: list[str] = []
        self.refusals = 0
        self.findings: list[str] = []
        self.dialogue: list[list[Any]] = []
        self.moves = int(self.policy.get("max_steps") or 0)
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.book = InferenceBook()
        self.trace: list[dict[str, Any]] = []
        self.walk_chat: list[dict[str, Any]] = []
        self.walk_seen = 0
        self.planned = False
        self.list_size, self.late_asked, self.asked_keys = LIST_SIZE, False, {}
        self.questions: list[list[str]] = []
        self.asked: dict[int, int] = {}
        self.revealed: set[int] = set()
        self.last_answer = ""

    def read_order(self) -> bool:
        opening = "\n<opening>\n" + serial(self.policy.get("opening")) + "\n</opening>"
        ranges = sorted({v for t in self.tools.values() for prop in ((t.get("parameters") or {}).get("properties") or {}).values()
                         if len(prop.get("enum") or []) >= 5 for v in prop["enum"] if isinstance(v, str)})
        asked = "<request>\n" + self.request + "\n</request>" + ("\n<ranges>\n" + serial(ranges) + "\n</ranges>" if ranges else "")
        jobs = [("reader", FACTS_RULES, asked, 512), ("toolmap", TOOL_RULES, "<tool_schemas>\n" + serial(list(self.tools.values())) + "\n</tool_schemas>" + opening, 1024)]
        texts = self.book.complete_parallel(jobs) or ["", ""]
        texts = [text or self.book.complete(role, prompt, material, final=role == "reader", tokens=tokens) for text, (role, prompt, material, tokens) in zip(texts, jobs)]
        tool_rows = [r for r in records(texts[1], TOOL_RULES) if r[0] in ("ROLE", "MODE", "IDARG")]
        if not any(r[0] == "ROLE" and len(r) == 3 and r[1] == "purchase" and r[2] in self.tools for r in tool_rows):
            tool_rows = [
                r for r in records(self.book.complete("reader", TOOL_RULES, jobs[1][2], final=True, tokens=1024), TOOL_RULES) if r[0] in ("ROLE", "MODE", "IDARG")
            ] or tool_rows
        self.apply_tools(tool_rows)
        facts = self.read_part(FACTS_RULES, asked, records(texts[0], FACTS_RULES), 512)
        items = "\n<items>\n" + serial({n: line["name"] for n, line in self.lines.items()}) + "\n</items>"
        self.read_part(MUSTS_RULES, asked + items, records(self.book.complete("reader", MUSTS_RULES, asked + items, final=True, tokens=1024), MUSTS_RULES), 1024, facts)
        self.hold_verdict()
        sheet = {"lines": self.lines, "parts": self.need_parts, "duties": self.duties, "relations": self.relations, "budget": self.budget}
        sheet.update(goal=[self.order_goal, self.order_measure], messages=self.max_messages, moves=self.moves, faults=self.faults, lookups=self.lookups)
        self.trace.append({"kind": "order_sheet", **json.loads(serial(sheet))})
        return bool(self.lines)

    def read_part(self, rules: str, material: str, rows: list[list[str]], tokens: int, facts: list[list[str]] | None = None) -> list[list[str]]:
        self.apply_sheet((facts or []) + rows)
        faults = self.order_faults(rows)
        if faults:
            again = records(self.book.complete("reader", rules, material + "\n<review>\n" + "\n".join(faults) + "\n</review>", final=True, tokens=tokens), rules)
            self.apply_sheet((facts or []) + again)
            rows = again if self.lines and len(self.order_faults(again)) < len(faults) else rows
        self.faults += faults
        self.apply_sheet((facts or []) + rows)
        return rows

    def apply_tools(self, rows: list[list[str]]):
        for r in rows:
            try:
                if r[0] == "ROLE" and len(r) == 3 and r[2] in self.tools:
                    self.roles[r[1]] = r[2]
                elif r[0] == "MODE" and len(r) == 3 and isinstance(value := json.loads(r[2]), dict):
                    self.modes[r[1]] = value
                elif r[0] == "IDARG" and len(r) == 3 and r[1] in self.tools:
                    self.ident_args.setdefault(r[1], []).append(r[2])
            except (ValueError, TypeError):
                continue
        used = {self.roles.get(r) for r in ("search", "message", "write", "basket", "purchase", "decline")}
        for name, tool in self.tools.items() if self.roles.get("message") else ():
            params = tool.get("parameters") or {}
            need = params.get("required") or []
            spec = (params.get("properties") or {}).get(need[0], {}) if len(need) == 1 else {}
            if name not in used and name not in self.ident_args and len(need) == 1 and spec.get("type", "string") == "string" and not {"enum", "maximum"} & set(spec):
                self.ident_args[name] = ["/" + need[0]]

    def apply_sheet(self, rows: list[list[str]]):
        self.lines, self.duties, self.relations, self.need_kind, self.need_parts, self.lookups = {}, {}, {}, {}, {}, {}
        self.budget = self.each_cap = self.max_messages = None
        self.order_goal, self.order_measure, self.moves = "any", "", int(self.policy.get("max_steps") or 0)
        line = lambda x: int(x) if str(x).isdigit() and int(x) in self.lines else None
        for r in rows:
            try:
                if r[0] == "ITEM" and len(r) >= 4 and int(r[1]) > 0:
                    quantity = int(r[2]) if r[2].isdigit() and int(r[2]) > 0 else 1
                    self.lines[int(r[1])] = {"name": " ".join(r[3:]), "quantity": quantity, "needs": [], "goal": "any", "measure": "", "cap": None}
                elif r[0] == "MESSAGES" and len(r) == 2 and int(r[1]) >= 0:
                    self.max_messages = int(r[1])
                elif r[0] == "MOVES" and len(r) == 2 and int(r[1]) > 0:
                    self.moves = min(self.moves, int(r[1]))
                elif r[0] == "DUTY" and len(r) == 5:
                    self.duties[int(r[1])] = {"stage": r[2], "messages": max(1 if r[2] in ("order", "choose") else 0, int(r[3])), "text": r[4]}
            except (ValueError, TypeError):
                continue
        lookups = {(r[1], plain(r[2])): r[3] if len(r) > 3 else "" for r in rows if r[0] == "LOOKUP" and len(r) >= 3}
        for r in rows:
            try:
                if r[0] == "MUST" and len(r) >= 4 and line(r[1]) and r[2].strip() and r[3].strip():
                    target = lookups.get((r[1], plain(r[2])))
                    need = self.add_need(int(r[1]), r[2], r[3], "listing", target is not None)
                    self.lookups.update({plain(need): target or r[3]} if target is not None else {})
                elif r[0] == "CAP" and len(r) >= 3:
                    self.set_budget(r[1], r[2])
                elif r[0] == "GOAL" and len(r) >= 3 and r[2] in ("cheapest", "most", "least"):
                    if r[1] == "order":
                        self.order_goal, self.order_measure = r[2], r[3] if len(r) > 3 else ""
                    elif line(r[1]):
                        self.lines[int(r[1])].update(goal=r[2], measure=r[3] if len(r) > 3 else "")
                elif r[0] == "RELATE" and len(r) >= 6 and r[5].strip() and line(r[2]) and line(r[3]):
                    rule = {"left": int(r[2]), "right": int(r[3]), "rule": f"{r[4]}: {r[5]}", **({"value": r[6].strip()} if len(r) > 6 and r[6].strip() else {})}
                    self.relations[r[1] or f"relation{len(self.relations) + 1}"] = rule
            except (ValueError, TypeError):
                continue
        if any(r[0] == "STOCK" for r in rows):
            for n in self.lines:
                self.add_need(n, "availability", "in stock", "availability")

    def order_faults(self, rows: list[list[str]]) -> list[str]:
        text, faults = spoken(self.request), []
        numbers = set(text.split())
        for r in rows:
            amount = finite_number(r[2]) if r[0] == "CAP" and len(r) >= 3 else None
            if amount is not None and (str(int(amount)) if float(amount).is_integer() else str(amount)) not in numbers:
                faults.append(f"{serial(r)}: the amount is not a number the request writes; copy it exactly")
            elif r[0] == "MESSAGES" and len(r) == 2 and r[1] not in numbers:
                faults.append(f"{serial(r)}: the message limit is not a number the request writes")
            elif r[0] == "MOVES" and len(r) == 2 and r[1] not in numbers and r[1] != str(self.policy.get("max_steps")):
                faults.append(f"{serial(r)}: the move limit is not a number the request writes")
            elif r[0] == "MUST" and len(r) >= 4 and (r[1], plain(r[2])) not in {(x[1], plain(x[2])) for x in rows if x[0] == "LOOKUP" and len(x) >= 3}:
                values = constraint_form(r[3])[1]
                codes = {w for w in spoken(r[3]).split() if any(c.isdigit() for c in w)} - numbers
                faults += [f"{serial(r)}: '{w}' is not a number or code the request writes; copy it exactly" for w in sorted(codes)]
                faults += [f"{serial(r)}: '{v}' mixes forms; write each form as its own MUST" for v in values[1:] if constraint_form(v + " |")[0]]
        for n in self.lines:
            values = [(k, spoken(v)) for k, need in enumerate(self.checked(n)) if not constraint_form(self.part(need, 2))[0] for v in constraint_form(self.part(need, 2))[1]]
            split = [(a, b) for i, a in values for j, b in values if i < j and f"{a} or {b}" in text]
            faults += [f"item {n}: '{a}' and '{b}' are alternatives of one property; write one MUST with |" for a, b in split]
        return faults

    def set_budget(self, scope: str, raw: Any) -> bool:
        amount, before = finite_number(raw), (self.budget, self.each_cap, serial(self.lines))
        if amount is None or amount < 0:
            return False
        if scope == "total":
            self.budget = amount
        elif scope == "each":
            self.each_cap = amount
        elif scope.isdigit() and int(scope) in self.lines:
            self.lines[int(scope)]["cap"] = amount
        return before != (self.budget, self.each_cap, serial(self.lines))

    def add_need(self, line: int, prop: str, constraint: str, kind: str = "listing", lookup: bool = False) -> str:
        prop, constraint = prop.strip(), constraint.strip()
        need = f"{prop}: {constraint}" if prop else constraint
        if constraint and plain(need) not in map(plain, self.lines[line]["needs"]):
            self.lines[line]["needs"].append(need)
            self.need_kind[plain(need)], self.need_parts[plain(need)] = kind if kind in CLASSES else "listing", (prop, "@" if lookup else "=", constraint)
        return need

    def requirement(self, need: str) -> dict[str, str]:
        prop, _, constraint = self.need_parts.get(plain(need), ("", "=", need))
        found = {"property": prop, "constraint": constraint}
        if not self.references.get(plain(need)):
            return found
        listing = self.listing_views(self.referenced[plain(need)], 900)[-1:] if plain(need) in self.referenced else None
        return {**found, "referenced_value": self.references[plain(need)], **({"referenced_listing": listing} if listing else {})}

    def part(self, need: str, index: int) -> str:
        return self.need_parts.get(plain(need), ("", "=", need))[index]

    def facts(self, ident: str) -> list[tuple[str, str]]:
        return self.cached(("facts", ident), lambda: [(k, v) for view in self.records_of(ident) + [self.search_rows().get(ident) or {}] for k, v in fields(view) if v.strip()])

    def own_values(self, ident: str, prop: Any) -> list[str]:
        return [value_words(v) for k, v in self.facts(ident) if key_words(k) == key_words(prop)]

    def store_values(self, need: str) -> list[str]:
        head, options = constraint_form(str(self.part(need, 2)))
        if self.part(need, 1) != "=" or head not in ("", "only", "none of"):
            return []
        store = self.cached(("store", plain(need)), lambda: {v for i in self.opened() for v in self.own_values(i, self.part(need, 0))})
        bare = [value_words(re.sub(r"\bonly\b", " ", NEGATION.split(o.lower())[0])) for o in options]
        return [w for w in bare if w and any(f" {w} " in f" {v} " for v in store)]

    def judge_view(self, ident: str, room: int) -> dict[str, Any]:
        named = lambda prop: {k: v for k, v in self.facts(ident) if key_words(k) == key_words(prop)}
        own = {f"{n}.{k}": named(self.part(need, 0)) for n in self.lines for k, need in enumerate(self.checked(n), 1) if self.part(need, 0)}
        return {"own_fields": own, "views": self.listing_views(ident, room)}

    def revision(self, ident: str) -> int:
        return self.cached(("rev", ident), lambda: hash(serial(self.record_paths(ident))))

    def request_rev(self) -> int:
        return hash(serial({n: [line["name"], self.checked(n)] for n, line in self.lines.items()}))

    def current(self, ident: str) -> bool:
        grade = self.grades.get(ident)
        return bool(grade) and grade["rev"] == self.revision(ident) and grade["req"] == self.request_rev()

    def price_of(self, ident: str) -> float | None:
        grade, row = self.grades.get(ident) or {}, self.search_rows().get(ident) or {}
        return grade["price"] if grade.get("price") is not None else finite_number(row.get(self.price_field or self.row_price_field())) if row else None

    def row_price_field(self) -> str | None:
        def learn() -> str | None:
            rows = list(self.search_rows().values())
            counted = lambda k: sum(isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool) for r in rows)
            keys = [k for k in {k for r in rows for k in r} if counted(k) >= 0.8 * len(rows) and re.search(r"price|cost|amount|cents|fee", k.lower())]
            return min(keys, key=len) if keys else None

        return self.cached("price_field", learn)

    def affordable(self, ident: str, n: int) -> bool:
        price = self.price_of(ident)
        if price is None:
            return True
        return (self.budget is None or price * self.lines[n]["quantity"] <= self.budget) and (self.each_cap is None or price <= self.each_cap)

    def stated_number(self, ident: str, number: float) -> bool:
        return bool(re.search(rf"(?<![\d.]){re.escape(cell(number))}(?![\d])", self.record_text(ident)))

    def bound(self, rule: str, words: str) -> bool | None:
        want, have = re.match(r"(at least|at most)\s+(-?\d+(?:\.\d+)?)\s*(.*)", rule), re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z]*)", str(words).lower())
        if not want or not have or key_words(want.group(3)).split()[:1] != key_words(have.group(2)).split()[:1]:
            return None
        return float(have.group(1)) >= float(want.group(2)) if want.group(1) == "at least" else float(have.group(1)) <= float(want.group(2))

    def need_state(self, ident: str, n: int, need: str) -> str:
        grade = self.grades.get(ident) or {}
        label, (_, words) = grade.get("needs", {}).get((n, plain(need)), "absent"), grade.get("quotes", {}).get((n, plain(need)), ("", ""))
        rule, strong = str(self.part(need, 2)), grade.get("by") == "judge"
        head, options = constraint_form(rule)
        shown = any(f" {key_words(o)} " in f" {key_words(words)} " for o in options if key_words(o))
        if self.part(need, 1) == "@" and not self.references.get(plain(need)):
            return "unknown"
        known, held = self.store_values(need), self.own_values(ident, self.part(need, 0))
        said = lambda texts: any(f" {w} " in f" {t} " for w in known for t in texts)
        text = self.cached(("text", ident), lambda: value_words(self.record_text(ident)))
        if known and (said(held) if head == "none of" else not said(held or [text])):
            return "violated"
        claimed = GRADE_STATE.get(label, "unknown")
        if claimed in ("met", "weak") and known and not held and head != "none of" and not said([value_words(words)]):
            claimed = "unknown"
        if claimed == "unknown":
            return claimed
        if head in ("at least", "at most"):
            met = self.bound(rule.lower(), words)
            return claimed if met is None and strong else "unknown" if met is None else "met" if met else "violated"
        if strong or any(len(o.split()) > 4 for o in options):
            return claimed
        if head == "none of":
            return claimed if shown == (claimed == "violated") else "unknown"
        return claimed if shown != (claimed == "violated") else "unknown"

    def line_state(self, ident: str, n: int) -> str:
        grade, row, field, label = self.grades.get(ident), self.search_rows().get(ident) or {}, self.category_field(), self.line_category(n)
        if not grade or ident in self.gone or grade["stock"] == "out" or (label and field and row.get(field) not in (None, label)):
            return "out"
        if grade["kind"].get(n) != "yes" and not (label and field and row.get(field) == label):
            return "out"
        price, caps = self.price_of(ident), [c for c in (self.lines[n]["cap"], self.each_cap) if c is not None]
        if price is not None and caps and price > min(caps):
            return "out"
        states = {self.need_state(ident, n, need) for need in self.checked(n)} if self.current(ident) else {"unknown"}
        return "out" if "violated" in states else "unknown" if "unknown" in states or (caps and price is None) else "weak" if "weak" in states else "met"

    def checked(self, line: int) -> list[str]:
        return [need for need in self.lines[line]["needs"] if self.need_kind.get(plain(need), "listing") == "listing"]

    def known_values(self) -> list[Any]:
        return self.cached("known", lambda: [v for e in self.events for v in leaves(e["observation"])])

    def observed(self, value: Any) -> bool:
        known = self.cached("known_set", lambda: {v for v in self.known_values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)})
        text = self.cached("seen_text", lambda: serial([e["observation"] for e in self.events]))
        return (isinstance(value, (str, int)) and not isinstance(value, bool) and value in known) or (isinstance(value, (str, int)) and len(str(value)) > 3 and str(value) in text)

    def identifier_fault(self, tool: str, args: dict[str, Any]) -> str:
        def visit(node: Any, parts: list[str]) -> list[Any]:
            if not parts:
                return leaves(node)
            if parts[0] == "*" and isinstance(node, list):
                return [v for child in node for v in visit(child, parts[1:])]
            value = pointer(node, parts[0])
            return visit(value, parts[1:]) if value is not None else []

        return next(
            (
                "identifier argument is absent from observed results"
                for path in self.ident_args.get(tool, [])
                for value in visit(args, path.strip("/").split("/"))
                if not self.observed(value)
            ),
            "",
        )

    def hold(self, reason: str, tool: str | None = None, args: Any = None) -> bool:
        self.trace.append({"kind": "held", "reason": reason})
        self.not_run = (self.not_run + [{"call": [tool, args] if tool is not None else None, "reason": reason, "after_step": self.step}])[-4:]
        return False

    def ending_kind(self, tool: str, args: Any) -> str:
        buy, drop = self.roles.get("purchase"), self.roles.get("decline")
        if tool not in (buy, drop):
            return ""
        if buy == drop:
            mode = self.modes.get("decline") or {}
            return "decline" if mode and isinstance(args, dict) and all(args.get(k) == v for k, v in mode.items()) else "purchase"
        return "purchase" if tool == buy else "decline"

    def call_store(self, tool: str, args: dict[str, Any], ending: bool = False, spare: int = 2) -> bool:
        if self.ending_kind(tool, args) and not ending:
            return self.hold("endings are sent only after the judge", tool, args)
        if tool not in self.tools or tool in self.exhausted or self.finished or self.step >= self.moves:
            return self.hold("tool unknown, out of budget or no steps left", tool, args)
        if not isinstance(args, dict):
            return self.hold("arguments must be a JSON object", tool, args)
        if tool == self.roles.get("message") and self.max_messages is not None and self.used_messages >= self.max_messages:
            return self.hold("message allowance exhausted", tool, args)
        fault = schema_fault(args, self.tools[tool].get("parameters") or {}) or self.identifier_fault(tool, args)
        if fault:
            return self.hold(fault, tool, args)
        mark = serial((tool, args))
        if not ending and self.last_actions.get(mark) == (self.cart_epoch, self.changed_at) and tool != self.roles.get("basket"):
            return self.hold("this exact call was already made and nothing has changed since; use its result", tool, args)
        closing = tool in (self.roles.get("write"), self.roles.get("basket")) or any(i in (self.actual_cart or {}) for i in leaves(args) if isinstance(i, str))
        message = tool == self.roles.get("message")
        if not ending and self.moves - self.step <= (spare if closing else max(spare, self.closing_steps(not message) + (0 if message else self.pending_asks()))):
            return self.hold("the last steps are kept for putting every line in the cart, reading its page and checking out", tool, args)
        payload = {**self.binding, "step": self.step + 1, "action": {"name": tool, "args": args}}
        for attempt in range(4):
            if self.book.remaining() < 2 or not self.book.pace():
                return False
            _store_proxy.timeout = min(12, max(1, self.book.remaining() - 1))
            response = _store_proxy.post_verbose("/environment/call", payload)
            if isinstance(response.data, dict):
                self.book.rate(False)
                result = response.data
                self.step = result.get("step") if isinstance(result.get("step"), int) else self.step + 1
                self.finished = bool(result.get("finished"))
                try:
                    seen = structured(result.get("observation"))
                except (TypeError, ValueError, RecursionError):
                    seen = result.get("observation")
                self.events.append({"tool": tool, "args": dict(args), "observation": seen, "envelope": result, "step": self.step})
                self.not_run = []
                self.last_actions[mark] = (self.cart_epoch, self.changed_at)
                if "budget_exceeded" in serial(seen):
                    self.exhausted.add(tool)
                failed = isinstance(seen, dict) and bool(seen.get("error"))
                plain_tool = tool not in self.readers() and tool != self.roles.get("message") and not ending
                riding = [v for v in seen.values() for v in (v if isinstance(v, list) else [v]) if isinstance(v, str) and len(v.split()) >= 4] if isinstance(seen, dict) else []
                if plain_tool and riding and not failed and not result.get("refusal"):
                    self.apply_update(serial(result), len(self.events), strong=True)
                if failed and tool in self.readers() and "invalid_arguments" in serial(seen):
                    self.ident_args.pop(tool, None)
                if tool == self.roles.get("message"):
                    self.used_messages += 1
                if tool == self.roles.get("write") and not failed:
                    self.cart_epoch += 1
                    self.cart_event = len(self.events)
                    self.actual_cart = None
                if tool in (self.roles.get("write"), self.roles.get("basket")):
                    self.read_basket()
                write = next(
                    (e for e in reversed(self.events) if e["tool"] == self.roles.get("write") and not (isinstance(e["observation"], dict) and e["observation"].get("error"))), None
                )
                if (
                    tool in (self.roles.get("write"), self.roles.get("basket"))
                    and write
                    and any(i in (self.actual_cart or {}) for i in leaves(write["args"]) if isinstance(i, str))
                    and not self.removes(write["args"])
                ):
                    self.add_template = dict(write["args"])
                self.trace.append(
                    {
                        "kind": "store",
                        "event": len(self.events),
                        "status": "error" if failed else "refused" if result.get("refusal") or (ending and not self.finished) else "ok",
                        "tool": tool,
                        "args": args,
                        "step": self.step,
                    }
                )
                return True
            status = (response.error or {}).get("status")
            body = str((response.error or {}).get("body") or "")
            self.trace.append({"kind": "store_error", "status": status, "step": payload["step"]})
            if status == 409:
                expected = re.search(r"expected\D{0,20}(\d+)", body, re.IGNORECASE)
                if expected:
                    payload["step"] = int(expected.group(1))
                    self.step = payload["step"] - 1
                    continue
                return False
            if status not in (None, 408, 429, 500, 502, 503, 504):
                return False
            if attempt < 3:
                time.sleep(self.book.rate(True, attempt) if status == 429 else 0.2 + random.random() * 0.5)
        return False

    def read_basket(self):
        readers = (self.roles.get("basket"),) if self.roles.get("basket") in self.tools else (self.roles.get("write"), self.roles.get("basket"))
        latest = next((i for i in range(len(self.events), max(self.cart_event, 1) - 1, -1) if self.events[i - 1]["tool"] in readers), None)
        if latest is None:
            return
        seen, known = self.events[latest - 1]["observation"], set(self.known_values())
        shapes = [self.basket_shape] if self.basket_shape else []

        def lists(node, path=""):
            if isinstance(node, list) and node and all(isinstance(row, dict) for row in node):
                yield path, node
            for key, child in node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else ():
                yield from lists(child, path + "/" + str(key))

        for path, rows in lists(seen):
            ids = sorted(
                (k for k in rows[0] if all(isinstance(row.get(k), str) and row.get(k) in known and " " not in row[k] for row in rows)),
                key=lambda k: sum(len(row[k]) for row in rows),
            )
            counts = [k for k in rows[0] if all(isinstance(row.get(k), int) and not isinstance(row.get(k), bool) and row.get(k) >= 0 for row in rows)]
            if ids and counts:
                shapes.append((path, ids[0], sorted(counts, key=lambda k: max(row[k] for row in rows))[0]))
        if self.events[latest - 1]["tool"] == self.roles.get("basket") and isinstance(seen, dict) and not any(leaves_lists(seen)):
            shapes.append(("?empty", "", ""))
        for rows_path, id_key, qty_key in shapes:
            rows = [] if rows_path == "?empty" else pointer(seen, rows_path)
            if isinstance(rows, list) and all(
                isinstance(row, dict) and isinstance(row.get(id_key), str) and isinstance(row.get(qty_key), int) and row.get(qty_key) >= 0 for row in rows
            ):
                basket = {}
                for row in rows:
                    if row[qty_key]:
                        basket[row[id_key]] = basket.get(row[id_key], 0) + row[qty_key]
                self.actual_cart, self.basket_shape = basket, (rows_path, id_key, qty_key) if rows_path != "?empty" else self.basket_shape
                self.carted |= set(basket)
                return

    def apply_update(self, raw: str, event: int, strong: bool = False) -> bool:
        self.last_answer = ""
        trigger = self.events[event - 1] if 1 <= event <= len(self.events) else {}
        text = self.book.complete(
            "update" if strong else "listing_update",
            ANSWER_RULES,
            serial({"actual_cart": self.actual_cart, "your_call": [trigger.get("tool"), trigger.get("args")], "latest": raw}),
            tokens=512,
            final=strong,
            stable=serial({"request": self.request, "lines": self.lines}) + "\n",
        )
        changed = False
        updates = records(text, ANSWER_RULES)
        cut = bool(self.book.log and self.book.log[-1].get("cut"))
        replacements = {(r[1], plain(r[3])) for r in updates if r[0] == "UPDATE" and len(r) >= 5 and r[2] == "add" and r[4].strip()}
        for r in updates:
            try:
                if r[0] == "UPDATE" and len(r) >= 5 and int(r[1]) in self.lines and r[3].strip():
                    needs, words = self.lines[int(r[1])]["needs"], plain(f"{r[3].strip()}: {r[4].strip()}")
                    old = next((need for need in needs if plain(need) == words), None) or (
                        next((need for need in needs if plain(self.part(need, 0)) == plain(r[3])), None) if r[2] == "drop" else None
                    )
                    if r[2] == "drop" and old is not None:
                        named = nouns(f"{self.part(old, 0)} {self.part(old, 2)}") & nouns(raw)
                        if (cut or not named) and (r[1], plain(r[3])) not in replacements:
                            self.notices += ["The reply was cut before its condition changes were complete; the prior condition remains active."] * cut
                            continue
                        needs.remove(old)
                        self.dropped.append(old)
                        changed = True
                    elif r[2] == "add" and old is None and r[4].strip():
                        need = self.add_need(int(r[1]), r[3], r[4], r[5] if len(r) > 5 else "listing", bool(len(r) > 6 and r[6].strip()))
                        self.lookups.update({plain(need): r[6].strip()} if len(r) > 6 and r[6].strip() else {})
                        self.fresh_values.add(int(r[1]))
                        changed = True
                elif r[0] == "QUANTITY" and len(r) == 3 and int(r[1]) in self.lines and int(r[2]) > 0 and self.lines[int(r[1])]["quantity"] != int(r[2]):
                    self.lines[int(r[1])]["quantity"] = int(r[2])
                    changed = True
                elif r[0] == "GOAL" and len(r) == 4 and int(r[1]) in self.lines and r[2] in ("any", "cheapest", "most", "least"):
                    changed = (self.lines[int(r[1])]["goal"], self.lines[int(r[1])]["measure"]) != (r[2], r[3]) or changed
                    self.lines[int(r[1])].update(goal=r[2], measure=r[3])
                elif r[0] == "BUDGET" and len(r) == 3:
                    changed = self.set_budget(r[1], r[2]) or changed
                elif r[0] == "GONE" and len(r) == 2 and r[1] not in self.gone:
                    self.gone.add(r[1])
                    changed = True
                elif r[0] == "NOTICE" and len(r) == 2 and r[1] not in self.notices:
                    self.notices.append(r[1])
                    changed = True
                elif r[0] == "ANSWER" and len(r) == 2:
                    self.last_answer = r[1]
            except (ValueError, TypeError):
                continue
        if changed:
            self.changed_at = len(self.events)
            self.hold_verdict()
        self.trace.append({"kind": "update", "event": event, "changed": changed, "strong": strong})
        return changed

    def event_view(self, number: int, room: int = 3000) -> dict[str, Any]:
        event = self.events[number - 1]
        extra = {k: v for k, v in (event.get("envelope") or {}).items() if k not in ("observation", "step", "finished", "replayed")}
        refused = self.ending_kind(event["tool"], event["args"]) and not self.finished
        empty = event["tool"] == self.roles.get("search") and not any(found and all(isinstance(r, dict) for r in found) for found in leaves_lists(event["observation"]))
        return {
            "event": number,
            "tool": event["tool"],
            "args": event["args"],
            "result": small_view(event["observation"], room)[:room],
            **({"envelope": small_view(extra, 800)[:800]} if extra else {}),
            **(
                {
                    "status": "empty: no listing matched every word; search again dropping one word at a time, "
                    "first words about stock, price or the shopper's situation, and keep the product and requirement values"
                }
                if empty
                else {}
            ),
            **({"outcome": "the checkout was refused: no order was placed and the task continues"} if refused else {}),
        }

    def walk_state(self) -> dict[str, Any]:
        duty_messages = sum(d["messages"] for d in self.duties.values())
        tag = lambda need: need if self.need_kind.get(plain(need), "listing") == "listing" else f"{need} [{self.need_kind[plain(need)]}]"
        return {
            "lines": {n: {**line, "needs": [f"{i}. {tag(need)}" for i, need in enumerate(line["needs"], 1)]} for n, line in self.lines.items()} or self.request,
            "dropped": self.dropped[-4:],
            "budget": self.budget,
            "each_cap": self.each_cap,
            "order_goal": [self.order_goal, self.order_measure],
            "relations": self.relations,
            "procedures": self.duties,
            "cart": self.actual_cart,
            "candidates": self.board(),
            "search_notes": self.search_notes(),
            "line_categories": {n: self.line_category(n) for n in self.lines},
            "steps_kept_for_questions_and_closing": self.reserve(),
            "reading_tool": self.first_reader(),
            "gone": sorted(self.gone),
            "steps_left": self.moves - self.step,
            "seconds_left": round(self.book.remaining()),
            "messages_left": None if self.max_messages is None else self.max_messages - self.used_messages,
            "messages_needed_for_procedures": duty_messages,
            "dialogue": self.dialogue[-4:],
            "notices": self.notices[-4:],
            "judge_findings": self.findings[-4:],
            "not_run": self.not_run,
            "order_placed": False,
        }

    def board(self) -> dict[int, list[list[Any]]]:
        listed = lambda n: [[i, s, self.price_of(i)] for i in self.grades if (s := self.line_state(i, n)) != "out"]
        return {n: sorted(listed(n), key=lambda c: (TIERS[c[1]], c[2] is None, c[2] or 0))[:6] for n in self.lines if self.grades}

    def opened(self) -> list[str]:
        skip = (self.roles.get("search"), self.roles.get("write"), self.roles.get("basket"), self.roles.get("message"))
        found = lambda: dict.fromkeys(v for e in self.events if e["tool"] not in skip for v in leaves(e["args"]) if isinstance(v, str) and self.observed(v))
        return self.cached("opened", lambda: list(found()))

    def listing_views(self, ident: str, room: int = 2000) -> list[dict[str, Any]]:
        views = []
        for number, event in enumerate(self.events, 1):
            if ident in leaves(event["args"]) and event["tool"] not in (self.roles.get("write"), self.roles.get("basket")):
                shared = sum(isinstance(v, str) and self.observed(v) for v in leaves(event["args"])) > 1 and find_record(event["observation"], ident)
                views.append({"event": number, "tool": event["tool"], "record": small_view(shared[1] if shared else event["observation"], room)[:room]})
        searched = [(n, e) for n, e in enumerate(self.events, 1) if e["tool"] == self.roles.get("search")]
        rows = [{"event": n, "row": small_view(hit[1], 800)[:800]} for n, e in searched if (hit := find_record(e["observation"], ident))]
        return rows[-1:] + views[-2:] if views else rows[-1:]

    def records_of(self, ident: str) -> list[Any]:
        skip = (self.roles.get("write"), self.roles.get("basket"), self.roles.get("message"))
        views = lambda: [
            e["observation"] if sum(isinstance(v, str) and self.observed(v) for v in leaves(e["args"])) < 2 else (find_record(e["observation"], ident) or ("", {}))[1]
            for e in self.events
            if e["tool"] not in skip and ident in leaves(e["args"])
        ]
        return self.cached(("records", ident), views)

    def record_text(self, ident: str) -> str:
        rows = lambda: [hit[1] for e in self.events if e["tool"] == self.roles.get("search") and (hit := find_record(e["observation"], ident))]
        return self.cached(("record_text", ident), lambda: plain(self.records_of(ident) + rows()))

    def readers(self) -> list[str]:
        skip = {self.roles.get(r) for r in ("write", "basket", "purchase", "decline", "message", "search")}
        return [
            t
            for t, paths in self.ident_args.items()
            if t in self.tools
            and t not in skip
            and paths
            and "/" not in paths[0].strip("/")
            and not set((self.tools[t].get("parameters") or {}).get("required") or []) - {paths[0].strip("/")}
        ]

    def views(self, ident: str) -> int:
        return len({e["tool"] for e in self.events if ident in leaves(e["args"]) and e["tool"] in set(self.readers()) | {self.roles.get("compare")}})

    def unread(self, ident: str) -> list[str]:
        return [tool for tool in self.readers() if not any(e["tool"] == tool and ident in leaves(e["args"]) for e in self.events)]

    def first_reader(self) -> str | None:
        detail, seen = self.roles.get("detail"), {e["tool"]: e["observation"] for e in self.events}
        return next((t for t in self.readers() if t != detail and (t not in seen or len(fields(seen[t])) >= 3)), detail)

    def read_fully(self, ident: str, final: bool = True):
        for tool in self.unread(ident) if final else [self.first_reader()] * (not self.views(ident)):
            if self.moves - self.step > 8:
                self.open_listing(ident, tool)

    def record_paths(self, ident: str) -> list[tuple[str, str]]:
        def walk(node: Any, path: str) -> list[tuple[str, str]]:
            if isinstance(node, dict):
                return [p for k, v in node.items() for p in walk(v, f"{path}.{k}" if path else str(k))]
            if isinstance(node, list) and not all(v is None or isinstance(v, (str, int, float, bool)) for v in node):
                return [p for v in node for p in walk(v, path)]
            return [(path.lower(), cell(node).lower())]

        return [p for view in self.records_of(ident) + [self.search_rows().get(ident) or {}] for p in walk(view, "")]

    def verified(self, ident: str, grade: str, field: str, words: str) -> str:
        name, parts = field.lower().strip(), [plain(p) for p in re.split(r"\.\.\.|…", words) if plain(p)]
        held = [plain(value) for path, value in self.record_paths(ident) if name and (path == name or path.endswith("." + name))]
        if grade == "absent" or (held and parts and any(all(p in value for p in parts) for value in held)):
            return grade
        return grade if not held and self.stated(ident, words, field) else "absent"

    def stated(self, ident: str, words: str, field: str = "") -> bool:
        text = self.record_text(ident)
        parts = [plain(p) for p in re.split(r"\.\.\.|…", words) if plain(p)]
        return bool(parts) and all(p in text or p.removeprefix(plain(field) + " ") in text for p in parts)

    def cached(self, key: Any, make):
        if self.cache_at != len(self.events):
            self.cache, self.cache_at = {}, len(self.events)
        return self.cache[key] if key in self.cache else self.cache.setdefault(key, make())

    def search_rows(self) -> dict[str, dict[str, Any]]:
        return self.cached("rows", self.collect_rows)

    def collect_rows(self) -> dict[str, dict[str, Any]]:
        searches = [e["observation"] for e in self.events if e["tool"] in (self.roles.get("search"), self.roles.get("page"))]
        keys = {k for seen in searches for ident in self.opened() if (hit := find_record(seen, ident)) for k, v in hit[1].items() if v == ident}
        if not keys:
            found = max((f for seen in searches for f in leaves_lists(seen) if f and all(isinstance(r, dict) for r in f)), key=len, default=[])
            unique = [k for k in (found[0] if found else {}) if all(isinstance(r.get(k), str) and " " not in r[k] for r in found) and len({r[k] for r in found}) == len(found)]
            keys = {max(unique, key=lambda k: sum(any(c.isdigit() for c in r[k]) for r in found))} if unique else set()
        rows: dict[str, dict[str, Any]] = {}
        for row in (row for seen in searches for found in leaves_lists(seen) for row in found if isinstance(row, dict)):
            ident = next((row[k] for k in keys if isinstance(row.get(k), str)), None)
            if ident:
                rows.setdefault(ident, row)
        return rows

    def open_listing(self, ident: str, tool: str | None = None) -> bool:
        tool = tool or self.roles.get("detail")
        path = (self.ident_args.get(tool) or [""])[0].strip("/")
        if not path or "/" in path or set(((self.tools.get(tool) or {}).get("parameters") or {}).get("required") or []) - {path}:
            return False
        return self.call_store(tool, {path: ident})

    def search_template(self) -> tuple[dict[str, Any], str] | None:
        tool, known = self.roles.get("search"), self.known_values()
        schema = (self.tools.get(tool) or {}).get("parameters") or {}
        props, required = schema.get("properties") or {}, set(schema.get("required") or [])
        for e in self.events:
            texts = (
                [k for k, v in e["args"].items() if isinstance(v, str) and re.search(r"[A-Za-z]{2}", v) and v not in known and "enum" not in (props.get(k) or {})]
                if e["tool"] == tool and isinstance(e["args"], dict)
                else []
            )
            texts = [k for k in texts if not re.search(r"label|category|department|section|range|sort|order", k.lower())] or texts
            if texts:
                named = [k for k in texts if re.search(r"query|search|keyword|text|term|word", (k + " " + str((props.get(k) or {}).get("description", ""))).lower())]
                key = max(named or texts, key=lambda k: len(e["args"][k]))
                return {k: v for k, v in e["args"].items() if k == key or k in required or "enum" in (props.get(k) or {})}, key
        free = [k for k, p in props.items() if p.get("type", "string") == "string" and "enum" not in p]
        key = next((k for k in free if k in required), None) or next(
            (k for k in free if re.search(r"query|search|keyword|text|term|word", (k + " " + str(props[k].get("description", ""))).lower())), None
        )
        if not tool or not key:
            return None
        return {**{k: 1 if props[k].get("type") == "integer" else "" for k in required if k != key and "enum" not in props[k]}, key: ""}, key

    def result(self, event: dict[str, Any]) -> tuple[list[dict[str, Any]], int, list[float]]:
        seen = event["observation"]
        rows = max((found for found in leaves_lists(seen) if found and all(isinstance(r, dict) for r in found)), key=len, default=[])
        hits = max((v for v in (seen.values() if isinstance(seen, dict) else []) if isinstance(v, int) and not isinstance(v, bool) and v >= len(rows)), default=len(rows))
        return rows, hits, [p for r in rows if self.price_field and (p := finite_number(r.get(self.price_field))) is not None]

    def searches(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e["tool"] == self.roles.get("search") and isinstance(e["args"], dict)]

    def search_notes(self) -> dict[str, Any]:
        key = (self.search_template() or ({}, ""))[1]
        tokens = lambda text: re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text).lower())
        log, in_title, in_field, unmatched, ordered, widths = [], set(), set(), set(), [], []
        for e in self.searches():
            rows, hits, prices = self.result(e)
            words, log = tokens(e["args"].get(key, "")), log + [[e["args"].get(key), hits]]
            ordered += [prices == sorted(prices)] if len(prices) > 2 else []
            widths += [len(rows)] if rows else []
            titles = {w for r in rows for w in tokens(headline(r, ""))}
            texts = {w for r in rows for w in tokens(" ".join(cell(v) for v in r.values()))}
            in_title |= {w for w in words if w in titles}
            in_field |= {w for w in words if w in texts and w not in titles}
            unmatched |= {w for w in words if rows and w not in texts}
        empty = [set(tokens(q)) for q, h in log if not h and len(tokens(q)) > 1]
        strict = any(t > set(tokens(q)) and h for t in empty for q, h in log) or (None if not empty else False)
        return {
            "every_word_must_match": strict,
            "rows_per_page": max(widths, default=None),
            "price_order": all(ordered) if ordered else None,
            "words_matched_in_titles": sorted(in_title)[:20],
            "words_matched_in_other_fields": sorted(in_field)[:20],
            "numbers_matched_as_written": sorted(w for w in in_title | in_field if any(c.isdigit() for c in w))[:8],
            "words_never_matched": sorted((self.dead_words | unmatched) - in_title - in_field)[:20],
            "queries_tried": log[-8:],
            "searches_fully_paged": [e["args"].get(key) for e in self.searches() if self.next_page(e) is None and self.result(e)[1] > len(self.result(e)[0])][-6:],
        }

    def page_rule(self) -> tuple[str, str, int] | None:
        made = self.search_template()
        searches = [e for e in self.events if e["tool"] == self.roles.get("search") and isinstance(e["args"], dict)]
        for before, after in zip(searches, searches[1:]):
            for k, v in after["args"].items():
                same = made and before["args"].get(made[1]) == after["args"].get(made[1])
                if same and k not in made[0] and before["args"].get(k) != v and self.result(before)[1] == self.result(after)[1]:
                    seen = before["observation"] if isinstance(before["observation"], dict) else {}
                    field = next((f for f, x in seen.items() if isinstance(v, str) and x == v), None)
                    if field or (isinstance(v, int) and not isinstance(v, bool) and v >= 2):
                        first = before["args"].get(k) or ((self.tools.get(after["tool"]) or {}).get("parameters") or {}).get("properties", {}).get(k, {}).get("minimum", 0)
                        return k, field or "", 0 if field else v - int(first)
        return None

    def next_page(self, event: dict[str, Any]) -> dict[str, Any] | None:
        rule, seen = self.page_rule(), event["observation"] if isinstance(event["observation"], dict) else {}
        if not rule:
            schema, made = (self.tools.get(event["tool"]) or {}).get("parameters") or {}, self.search_template()
            props = schema.get("properties") or {}
            key = next((k for k, v in props.items() if v.get("type") == "integer" and "enum" not in v and (not made or k != made[1])), None)
            rows, hits, _ = self.result(event)
            if not key or not rows:
                return None
            numbered = re.search(r"page", f"{key} {props[key].get('description', '')} {self.tools[event['tool']].get('description', '')}".lower())
            step = 1 if numbered and not re.search(r"offset|skip|start|row", key.lower()) else len(rows)
            start = int(event["args"].get(key) or (1 if step == 1 else 0))
            nxt, top = start + step, props[key].get("maximum")
            more = hits > start * len(rows) if step == 1 else hits > nxt
            return {**event["args"], key: nxt} if more and (top is None or nxt <= top) else None
        key, field, stride = rule
        token = seen.get(field) if field else int(event["args"].get(key) or 0) + stride
        return {**event["args"], key: token} if token not in (None, "") and (field or token < self.result(event)[1]) else None

    def closing_steps(self, repair_round: bool = True) -> int:
        cart = self.actual_cart or {}
        held = {n for n in self.lines if any((self.grades.get(i) or {}).get("kind", {}).get(n) == "yes" for i in cart)}
        return 2 + 2 * len(set(self.lines) - held) + sum(len(self.unread(i)) or i not in self.opened() for i in cart)

    def pending_asks(self) -> int:
        if self.roles.get("message") not in self.tools:
            return 0
        return self.question_room() if not self.planned else min(len(self.questions), self.question_room())

    def reserve(self) -> int:
        return len(self.lines) + 3 + self.question_room()

    def likely_lines(self, ident: str) -> set[int]:
        def lines() -> set[int]:
            words = terms(self.record_text(ident))
            overlap = {n: len(words & terms(line["name"])) for n, line in self.lines.items()}
            return {n for n, k in overlap.items() if k == max(overlap.values())}

        return self.cached(("likely", ident), lines)

    def proven(self, line: int) -> bool:
        return self.lines[line]["goal"] == self.order_goal == "any" and any(self.line_state(i, line) == "met" and self.affordable(i, line) for i in self.grades)

    def line_kind(self, n: int) -> str:
        name = re.split(r",|\b(?:where|with|for|that|whose|which|of the|in)\b", self.lines[n]["name"].lower())[0].strip()
        first = name.split(" & ")[0].strip()
        name = re.sub(r"^(?:a |an |the |pairs? of |sets? of |packs? of )", "", first if len(first.split()) >= 2 else name.replace(" & ", " "))
        return " ".join(singular(w) for w in name.split())

    def search_values(self, n: int) -> list[str]:
        needs = [need for need in self.checked(n) if self.part(need, 1) == "=" and self.need_kind.get(plain(need)) != "availability"]
        forms = [(constraint_form(self.part(need, 2)), str(self.part(need, 0))) for need in needs]
        values = [found[0] for (head, found), _ in forms if head in ("", "only", "fits") and found and len(found[0].split()) <= 4]
        values += [rule["value"] for rule in self.relations.values() if rule["right"] == n and rule.get("value") and len(rule["value"].split()) <= 4]
        values += [prop for (head, _), prop in forms if head not in ("", "only", "fits")]
        values = [v for need in self.checked(n) if self.part(need, 1) == "@" and (v := self.references.get(plain(need))) and len(v.split()) <= 4] + values
        return [v for v in dict.fromkeys(values) if v and not set(plain(v).split()) & self.dead_words]

    def ask_query(self, goal: dict[str, Any], tried: list[str]) -> str | None:
        material = "<goal>\n" + serial(goal) + "\n</goal>\n<notes>\n" + serial(self.search_notes()) + "\n</notes>"
        rows = records(self.book.complete("searcher", SEARCH_RULES, material, tokens=80), SEARCH_RULES)
        query = next((r[1].strip() for r in rows if r[0] == "QUERY" and len(r) >= 2 and r[1].strip()), None)
        return query if query and plain(query) not in {plain(q) for q in tried} else None

    def search_for(self, goal: dict[str, Any], tries: int = 3, extra: dict[str, Any] | None = None) -> bool:
        made = self.search_template()
        if not made or self.moves - self.step <= self.reserve():
            return False
        (base, key), ranged = made, self.range_args(" ".join(str(v) for v in goal.values()))
        required = set(((self.tools.get(self.roles.get("search")) or {}).get("parameters") or {}).get("required") or [])
        base = {**{k: v for k, v in base.items() if k == key or k in required}, **ranged, **(extra or {})}
        tried, mark = [str(e["args"].get(key)) for e in self.searches()], plain(serial(goal) + serial(extra or {})) + f"@{self.changed_at}:{len(self.gone)}"
        if self.goal_queries.get(mark, 0) >= 3:
            return False
        for _ in range(tries):
            self.goal_queries[mark] = self.goal_queries.get(mark, 0) + 1
            query = self.ask_query(goal, tried)
            if not query or self.moves - self.step <= self.reserve() or not self.call_store(self.roles["search"], {**base, key: query}):
                break
            tried.append(query)
            if self.result(self.events[-1])[0]:
                return True
            self.learn_dead(query)
        words = [w for w in re.findall(r"[A-Za-z][\w'-]{2,}", str(goal.get("description") or "")) if w.lower() not in FILLER]
        fallback = next((w.rstrip("'s") for w in words[1:] if w[0].isupper()), None) or str(goal.get("product") or "")
        if not fallback or plain(fallback) in {plain(q) for q in tried} or self.moves - self.step <= self.reserve():
            return False
        return self.call_store(self.roles["search"], {**base, key: fallback}) and bool(self.result(self.events[-1])[0])

    def pick_rows(self, goal: dict[str, Any], idents: list[str], room: int) -> list[str]:
        rows = self.search_rows()
        fresh = [i for i in idents if i in rows and i not in self.opened() and i not in self.gone]
        cards = {i: {k: (v[:120] if isinstance(v, str) else v) for k, v in rows[i].items() if isinstance(v, (str, int, float)) and (len(str(v)) < 80 or v == headline(rows[i], i))}
                 for i in fresh}
        if not cards or room <= 0:
            return []
        material = "<goal>\n" + serial(goal) + "\n</goal>\n<rows>\n" + serial(dict(list(cards.items())[:16])) + "\n</rows>\n<room>" + str(room) + "</room>"
        chosen = records(self.book.complete("searcher", OPEN_RULES, material, tokens=300), OPEN_RULES)
        return list(dict.fromkeys(r[1] for r in chosen if r[0] == "OPEN" and len(r) >= 2 and r[1] in cards))[:room]

    def batch_reader(self) -> tuple[str, str, int] | None:
        skip = {self.roles.get(r) for r in ("write", "basket", "purchase", "decline", "message", "search")}
        for tool, paths in self.ident_args.items():
            parts = paths[0].strip("/").split("/") if paths else []
            if tool in self.tools and tool not in skip and len(parts) == 2 and parts[1] == "*":
                spec = ((self.tools[tool].get("parameters") or {}).get("properties") or {}).get(parts[0]) or {}
                return tool, parts[0], int(spec.get("maxItems") or 4)
        return None

    def open_listings(self, idents: list[str]) -> int:
        wanted, before = [i for i in dict.fromkeys(idents) if i not in self.opened() and i not in self.gone], len(self.opened())
        batch = self.batch_reader()
        for k in range(0, len(wanted), batch[2] if batch else 1):
            if self.moves - self.step <= self.reserve():
                break
            if not (self.call_store(batch[0], {batch[1]: wanted[k : k + batch[2]]}) if batch else self.open_listing(wanted[k], self.first_reader())):
                break
        return len(self.opened()) - before

    def learn_dead(self, query: str):
        words, key = set(plain(query).split()), (self.search_template() or ({}, ""))[1]
        alive = [w for e in self.searches() if self.result(e)[1] and (w := set(plain(str(e["args"].get(key))).split())) < words]
        extra = [words - w for w in alive if len(words - w) == 1]
        self.dead_words |= set().union(*extra) if extra else set()

    def category_field(self) -> str | None:
        return self.cached("field", self.learn_category_field)

    def learn_category_field(self) -> str | None:
        rows, kinds = list(self.search_rows().values()), set().union(*(nouns(self.line_kind(n)) for n in self.lines)) if self.lines else set()
        keys = {k for r in rows for k, v in r.items() if isinstance(v, str) and 0 < len(v) < 60}
        fields = {k for k in keys if len({r.get(k) for r in rows}) <= max(2, len(rows) // 2)}
        share = lambda k: sum(bool(nouns(r.get(k)) & kinds) for r in rows) - len({r.get(k) for r in rows}) / max(1, len(rows))
        best = max(fields, key=share, default=None)
        return best if best and share(best) > len(rows) / 3 else None

    def line_category(self, n: int) -> str | None:
        return self.cached(("label", n), lambda: self.learn_line_category(n))

    def learn_line_category(self, n: int) -> str | None:
        field, kind = self.category_field(), nouns(self.line_kind(n))
        labels = {r[field] for r in self.search_rows().values() if field and isinstance(r.get(field), str)}
        rank = lambda c: (len(kind & nouns(c)), -len(nouns(c) - kind))
        ordered = sorted(labels, key=rank, reverse=True)
        return ordered[0] if ordered and rank(ordered[0])[0] and (len(ordered) == 1 or rank(ordered[0]) > rank(ordered[1])) else None

    def errand(self, n: int, first: bool = False):
        if self.proven(n) or self.moves - self.step <= self.reserve():
            return
        goal = {"product": self.line_kind(n), "values": self.search_values(n), **({"cheapest": True} if self.cheapest(n) else {})}
        cap = self.cap_rule()
        best = min((p for i in self.grades if self.line_state(i, n) == "met" and (p := self.price_of(i)) is not None), default=None)
        limits = [c for c in (self.lines[n]["cap"], self.each_cap, self.budget / self.lines[n]["quantity"] if self.budget is not None else None) if c is not None]
        limits += [best - 1] if self.cheapest(n) and best is not None and best > 1 else []
        extra = {cap: int(min(limits)) if float(min(limits)).is_integer() else min(limits)} if cap and limits else {}
        if not self.search_for(goal, tries=2 if first else 3, extra=extra):
            self.screen(n, 2 if first else 4)
            return
        rows, hits, _ = self.result(self.events[-1])
        asking = not self.planned and self.question_room() and self.roles.get("message") in self.tools
        ranked = self.cheapest(n) or self.ranked_by_measure(n)
        depth = 0 if asking else 6 if ranked and not self.search_notes()["price_order"] else 1 if hits <= 3 * max(1, len(rows)) else 0
        page, pages = self.next_page(self.events[-1]) if hits > len(rows) else None, 0
        while page and pages < depth and self.moves - self.step > self.reserve() and self.call_store(self.roles["search"], page):
            pages, page = pages + 1, self.next_page(self.events[-1])
        self.screen(n, 2 if first else 4)

    def screen(self, n: int, count: int):
        cat, field, done = self.line_category(n), self.category_field(), set(self.opened())
        price, matches = lambda row: finite_number(row.get(self.price_field)) if self.price_field else None, self.row_matches(n)
        order = lambda r: (cat is not None and r[1].get(field) != cat, -matches.get(r[0], 0), (price(r[1]) or 0) if self.cheapest(n) or self.budget is not None else 0)
        limit = min(c for c in (self.lines[n]["cap"], self.each_cap, self.budget / self.lines[n]["quantity"] if self.budget is not None else None, math.inf) if c is not None)
        within = lambda row: price(row) is None or price(row) <= limit
        ranked = [i for i, _ in sorted(((i, r) for i, r in self.search_rows().items() if i not in done and i not in self.gone and within(r)), key=order)]
        goal = {"product": self.line_kind(n), "values": self.search_values(n)}
        self.open_listings(ranked[:count] if self.cheapest(n) else self.pick_rows(goal, ranked[:16], count) or ranked[:count])

    def unopened_rows(self, n: int) -> list[str]:
        field, label, done, matches = self.category_field(), self.line_category(n), set(self.opened()), self.row_matches(n)
        limit = min(c for c in (self.lines[n]["cap"], self.each_cap, self.budget / self.lines[n]["quantity"] if self.budget is not None else None, math.inf) if c is not None)
        price = lambda row: finite_number(row.get(self.price_field)) if self.price_field else None
        kin = lambda i, row: matches.get(i) or (label is not None and field is not None and row.get(field) == label)
        return [i for i, row in self.search_rows().items() if i not in done and i not in self.gone and kin(i, row) and (price(row) is None or price(row) <= limit)]

    def cheaper_unseen(self, n: int) -> bool:
        best = min((p for i in self.grades if self.line_state(i, n) == "met" and (p := self.price_of(i)) is not None), default=None)
        field, label, done = self.category_field(), self.line_category(n), set(self.opened())
        kin = lambda i, row: self.row_matches(n).get(i) or (label is not None and field is not None and row.get(field) == label)
        cheaper = lambda row: self.price_field and (p := finite_number(row.get(self.price_field))) is not None and best is not None and p < best
        return any(i not in done and i not in self.gone and kin(i, row) and cheaper(row) for i, row in self.search_rows().items())

    def proven_basket(self) -> dict[str, Any]:
        self.resolve_references()
        self.searched_out = False
        for _ in range(3):
            self.judge(self.grading_order())
            unmet = [n for n in self.lines if not any(self.line_state(i, n) == "met" and self.affordable(i, n) for i in self.grades)]
            cheap = [n for n in self.lines if n not in unmet and self.cheapest(n) and self.cheaper_unseen(n)]
            if not (unmet or cheap) or self.moves - self.step <= self.reserve() or self.book.remaining() < 60:
                break
            before = len(self.events)
            for n in unmet + cheap:
                self.errand(n)
            if len(self.events) == before:
                self.searched_out = not any(self.unopened_rows(n) for n in unmet)
                break
        return self.settle()

    def row_matches(self, n: int) -> dict[str, int]:
        def count() -> dict[str, int]:
            values, key = self.search_values(n), (self.search_template() or ({}, ""))[1]
            asked = [serial(e["observation"]) for e in self.searches() if any(set(plain(v).split()) <= set(plain(str(e["args"].get(key, ""))).split()) for v in values)]
            return {i: sum(nouns(v) <= row_terms(row) for v in values) + sum(i in text for text in asked) for i, row in self.search_rows().items()}

        return self.cached(("matches", n, serial(self.lines[n]["needs"])), count)

    def ungraded(self, ident: str) -> bool:
        grade = self.grades.get(ident)
        return (
            not grade
            or not self.current(ident)
            or any(n not in grade["kind"] or (grade["kind"][n] == "yes" and any((n, plain(need)) not in grade["needs"] for need in self.checked(n))) for n in self.lines)
        )

    def rule_grade(self, ident: str, n: int, need: str) -> tuple[str, str, str] | None:
        prop, link, rule = self.need_parts.get(plain(need), ("", "=", need))
        head, options = constraint_form(rule)
        wanted = [self.references.get(plain(need)) or ""] if link == "@" else options
        if not prop or not all(wanted) or head in ("fits", "only"):
            return None
        own = [(k, v) for k, v in self.facts(ident) if key_words(k) == key_words(prop) and v.strip()]
        if head in ("at least", "at most"):
            bounds = [(k, v, self.bound(rule.lower(), v)) for k, v in own if len(re.findall(r"\d+(?:\.\d+)?", v)) == 1]
            return next((("key" if met else "conflict", k, v) for k, v, met in bounds if met is not None), None)
        hit = next(((k, v, o) for k, v in own for o in wanted if phrase_in(o, v)), None)
        if head == "none of":
            return ("conflict", hit[0], hit[2]) if hit else None
        return ("key", hit[0], hit[2]) if hit else None

    def stock_by_rule(self, ident: str) -> str:
        for k, v in self.facts(ident):
            if nouns(k) & {"stock", "availability", "available"}:
                said = plain(v)
                if said in ("true", "yes", "in stock", "available", "in", "1"):
                    return "in"
                if said in ("false", "no", "out of stock", "unavailable", "out", "sold out", "0"):
                    return "out"
        return "unknown"

    def rule_settle(self, ident: str) -> bool:
        field, row, labels = self.category_field(), self.search_rows().get(ident) or {}, {n: self.line_category(n) for n in self.lines}
        if not field or row.get(field) is None or None in labels.values() or self.price_of(ident) is None or any(self.ranked_by_measure(n) for n in self.lines):
            return False
        kinds = {n: "yes" if row.get(field) == label else "no" for n, label in labels.items()}
        grades = {(n, plain(need)): self.rule_grade(ident, n, need) for n in self.lines if kinds[n] == "yes" for need in self.checked(n)}
        stock = self.stock_by_rule(ident)
        if None in grades.values() or stock == "unknown":
            return False
        self.grades[ident] = {
            "by": "judge", "rule": True, "kind": kinds, "needs": {k: g[0] for k, g in grades.items()}, "quotes": {k: (g[1], g[2]) for k, g in grades.items()},
            "price": self.price_of(ident), "stock": stock, "measure": None, "rev": self.revision(ident), "req": self.request_rev(),
        }
        graded = {ident: {"kind": kinds, "needs": {f"{n}:{k}": g[0] for (n, k), g in grades.items()}, "price": self.price_of(ident), "stock": stock}}
        self.trace.append({"kind": "judge", "by": "rule", "graded": graded, "pairs": {}})
        return True

    def off_kind(self, ident: str) -> bool:
        field, labels = self.category_field(), {self.line_category(n) for n in self.lines} - {None}
        return bool(field and labels) and (self.search_rows().get(ident) or {}).get(field) not in (None, *labels)

    def grading_order(self) -> list[str]:
        opened = self.opened()[::-1]
        cost = lambda i: self.price_of(i) if self.price_of(i) is not None else math.inf
        return sorted(opened, key=cost) if any(self.cheapest(n) for n in self.lines) else opened

    def judge(self, idents: list[str], final: bool = False, pairs: tuple = ()) -> int:
        tries, widest = {}, JUDGE_BATCH
        role, sent = "judge" if final else "scout_judge", 0
        todo = [p for p in pairs if self.pairs.get(p, "unknown") == "unknown"]
        while (final or self.judgments < JUDGE_CALLS) and self.book.remaining() > (20 if final else 45):
            settled = lambda i: not final and i not in self.grades and i not in (self.actual_cart or {}) and all(self.proven(n) for n in self.likely_lines(i))
            strong = lambda i: self.current(i) and self.grades[i]["by"] == "judge"
            open_to = lambda i: i not in self.gone and not self.off_kind(i) and tries.get(i, 0) < 2
            wanted = [i for i in dict.fromkeys(idents) if open_to(i) and (not strong(i) if final else self.ungraded(i) and not settled(i))]
            for ident in wanted[:JUDGE_BATCH]:
                self.read_fully(ident, final)
            wanted = [i for i in wanted if not self.rule_settle(i)]
            if not wanted and not todo:
                return sent
            order, changes, rows = self.verdict_frame()
            for size, room in ((JUDGE_BATCH, self.verdict_room() if final else 2500), (2, self.verdict_room() if final else 2500), (1, self.verdict_room() if final else 2000)):
                if size > widest:
                    continue
                batch = wanted[:size]
                shown = list(dict.fromkeys(batch + [i for p in todo for i in p[1:]]))
                material = self.verdict_extra(todo) + changes + serial({i: self.judge_view(i, room) for i in shown}) + "\n</listings>"
                tokens = self.verdict_tokens(rows, len(shown), len(todo)) if role == "judge" else 4096
                if self.book.fits(role, JUDGE_RULES, material, tokens, order, final):
                    break
            tries.update({i: tries.get(i, 0) + 1 for i in batch})
            text = self.book.complete(role, JUDGE_RULES, material, final=final, tokens=tokens, stable=order)
            if not text:
                return sent
            sent, self.judgments = sent + 1, self.judgments + (0 if final else 1)
            widest = max(1, len(batch) // 2) if self.book.log and self.book.log[-1].get("cut") else widest
            role = role if self.read_grades(text, shown, todo, "scout" if role == "scout_judge" else "judge") else "judge"
            todo = []
            self.hold_verdict()

    def read_grades(self, text: str, shown: list[str], todo: list[tuple[str, str, str]], by: str) -> int:
        fresh = lambda i: {
            "by": by, "kind": dict.fromkeys(self.lines, "no"), "needs": {}, "quotes": {}, "price": None, "stock": "unknown", "measure": None,
            "rev": self.revision(i), "req": self.request_rev(),
        }
        graded, offered = {i: fresh(i) for i in shown}, set()
        for r in records(text, JUDGE_RULES):
            try:
                if r[0] == "KIND" and r[2] in graded and int(r[1]) in self.lines:
                    graded[r[2]]["kind"][int(r[1])] = "yes"
                elif r[0] == "GRADE" and r[2] in graded and int(r[1]) in self.lines and r[4] in LABELS:
                    need = dict(enumerate(self.checked(int(r[1])), 1)).get(int(r[3]))
                    if need:
                        key = (int(r[1]), plain(need))
                        graded[r[2]]["needs"][key], graded[r[2]]["quotes"][key] = self.verified(r[2], r[4], r[5], r[6]), (r[5], r[6])
                elif r[0] == "OFFER" and r[1] in graded:
                    price = finite_number(r[2])
                    graded[r[1]].update(price=price if price is not None and self.stated_number(r[1], price) else None, stock=r[3] if r[3] in ("in", "out") else "unknown")
                    self.price_field = r[4] if price is not None and r[4] else self.price_field
                    offered.add(r[1])
                elif r[0] == "MEASURE" and r[1] in graded and self.stated(r[1], r[3]):
                    graded[r[1]]["measure"] = finite_number(r[2])
                elif r[0] == "PAIR" and tuple(r[1:4]) in todo:
                    quoted = not r[5] or self.stated(r[2], r[5]) or self.stated(r[3], r[5])
                    self.pairs[tuple(r[1:4])] = r[4] if r[4] in ("yes", "no") and quoted else "unknown"
            except (ValueError, TypeError):
                continue
        kept = {i: g for i, g in graded.items() if (i in offered or g["needs"]) and not (by == "scout" and (self.grades.get(i) or {}).get("by") == "judge" and self.current(i))}
        self.grades.update(kept)
        self.trace.append(
            {
                "kind": "judge",
                "by": by,
                "graded": {i: {"kind": g["kind"], "needs": {f"{n}:{k}": v for (n, k), v in g["needs"].items()}, "price": g["price"], "stock": g["stock"]} for i, g in kept.items()},
                "pairs": {"|".join(p): self.pairs.get(p, "unknown") for p in todo},
            }
        )
        return len(kept)

    def verdict_frame(self) -> tuple[str, str, int]:
        sheet = {
            n: {"product": line["name"], "quantity": line["quantity"], "requirements": {k: self.requirement(need) for k, need in enumerate(self.checked(n), 1)}}
            for n, line in self.lines.items()
        }
        order = "<order>\n" + serial({"original_request": self.request, "lines": sheet}) + "\n</order>\n"
        changes = "<changes>\n" + serial({"dropped": self.dropped, "notices": self.notices}) + "\n</changes>\n<listings>\n"
        return order, changes, 2 + max([len(self.checked(n)) for n in self.lines] or [0])

    def verdict_extra(self, pairs: list[tuple[str, str, str]]) -> str:
        frame, named = self.measure_frame(), {rid: [[a, b] for r, a, b in pairs if r == rid] for rid in dict.fromkeys(p[0] for p in pairs) if rid in self.relations}
        relations = {rid: {**self.relations[rid], "pairs": todo} for rid, todo in named.items()}
        return ("<relations>\n" + serial(relations) + "\n</relations>\n" if relations else "") + ("<measure>\n" + serial(frame) + "\n</measure>\n" if frame else "")

    def verdict_room(self) -> int:
        return 1800 if len(self.lines) <= 2 else 1200 if len(self.lines) == 3 else 900

    def verdict_tokens(self, rows: int, count: int, pairs: int) -> int:
        return min(1024, 1 << (256 + 30 * rows * count + 40 * pairs + 30 * count * bool(self.measure_frame())).bit_length())

    def hold_verdict(self):
        if self.lines and self.book.reserve:
            order, changes, rows = self.verdict_frame()
            result = self.basket() if self.grades else {"best": None, "fallback": None, "missing": [], "better": []}
            listings, open_pairs = self.doubts(result)
            chosen = result["best"] or result["fallback"]
            count = min(JUDGE_BATCH, max(1, len(listings) if chosen else len(self.lines)))
            pairs = len(open_pairs) if chosen else 2 * len(self.relations)
            known = [i for i in listings if i in self.grades][:count] if chosen else []
            views = {**{i: self.judge_view(i, self.verdict_room()) for i in known}, **{f"x{k}": "x" * (1200 + 2 * self.verdict_room()) for k in range(count - len(known))}}
            material = "x" * 300 * pairs + changes + serial(views) + "\n</listings>"
            self.book.floor = self.book.price(self.book.body("judge", JUDGE_RULES, material, self.verdict_tokens(rows, count, pairs), READER, stable=order))

    def resolve_references(self):
        told = lambda text: {singular(w) for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if (len(w) > 2 or w[0].isdigit()) and w not in FILLER}
        for need in [need for n in self.lines for need in self.checked(n) if self.part(need, 1) == "@" and not self.references.get(plain(need))]:
            self.references[plain(need)] = None
            prop = self.part(need, 0)
            chain = self.chains.setdefault(plain(need), {"target": self.lookups.get(plain(need)) or self.part(need, 2), "hint": "", "asked": set(), "paged": set(), "queries": 0,
                                                        "path": []})
            for _ in range(20):
                if self.book.remaining() < 40:
                    break
                target, goal = chain["target"], {"description": chain["target"], **({"suggested_words": chain["hint"]} if chain["hint"] else {})}
                selective = told(target) - set().union(*(nouns(self.line_kind(n)) for n in self.lines))
                found = [i for i in self.rare_matches(target, count=len(self.search_rows())) if i not in chain["path"] and (not selective or selective & told(self.record_text(i)))]
                carries = [i for i in found if i in self.opened() and told(target) <= told(self.record_text(i))]
                fresh = [i for i in found if i not in self.opened()]
                matched = lambda e: sum(find_record(e["observation"], i) is not None for i in found) if selective & told(serial(e["args"])) else 0
                best = max(self.searches(), key=matched, default=None)
                page = self.next_page(best) if best and matched(best) and plain(serial(best["args"])) not in chain["paged"] else None
                exhausted = not fresh and not page and chain["queries"] >= 3
                coverage = lambda i: len(told(target) & told(self.record_text(i)))
                shown = (carries or (exhausted and sorted((i for i in found if i in self.opened()), key=coverage, reverse=True)) or [])[:4]
                if shown and (target, tuple(shown)) not in chain["asked"]:
                    chain["asked"].add((target, tuple(shown)))
                    mentions = lambda i: [t for t in re.split(r"(?<=[.;!?])\s+", " ".join(cell(v) for _, v in self.facts(i))) if nouns(prop) & nouns(t)][:4]
                    listings = serial({i: {"own_fields": {k: v for k, v in self.facts(i) if key_words(k) == key_words(prop)}, "property_mentions": mentions(i),
                                          "views": self.listing_views(i, 1500)} for i in shown})
                    material = "<reference>\n" + serial({"property": prop, "description": target}) + "\n</reference>\n<listings>\n" + listings + "\n</listings>"
                    rows = records(self.book.complete("refs", REF_RULES, material, final=True, tokens=1024), REF_RULES)
                    usable = lambda r: r[0] == "REF" and len(r) >= 4 and r[1] in shown and plain(r[2]) not in ("", "none", "null", "unknown", "n a")
                    link = next((r[1] for r in rows if r[0] == "LINK" and len(r) >= 2 and r[1].strip()), None)
                    ref = None if link else next((r for r in rows if usable(r) and any(phrase_in(r[2], v) for v in self.own_values(r[1], prop))), None)
                    if ref:
                        self.references[plain(need)], self.referenced[plain(need)] = ref[2], ref[1]
                        self.fresh_values.update(n for n in self.lines if need in self.lines[n]["needs"])
                        break
                    if link:
                        chain.update(target=link, hint="", queries=0, path=chain["path"] + shown)
                    else:
                        chain["hint"] = next((r[1] for r in rows if r[0] == "SEARCH" and len(r) >= 2 and r[1].strip()), chain["hint"])
                    continue
                if self.moves - self.step <= self.reserve():
                    break
                if fresh and self.open_listings(self.pick_rows(goal, fresh, 4) or fresh[:3]):
                    continue
                if page:
                    chain["paged"].add(plain(serial(best["args"])))
                    if self.call_store(self.roles["search"], page):
                        continue
                if chain["queries"] < 3:
                    chain["queries"] += 1
                    if self.search_for(goal):
                        continue
                if not fresh and not page:
                    break
            self.trace.append({"kind": "reference", "need": need, "value": self.references[plain(need)], "target": chain["target"]})

    def match_weight(self, ident: str, text: str) -> tuple[int, float]:
        rows, wanted = self.search_rows(), terms(text) - FILLER
        hit = wanted & row_terms(rows.get(ident) or {})
        return len(hit), sum(1 / n for w in hit if (n := sum(w in row_terms(row) for row in rows.values())))

    def rare_matches(self, text: str, count: int = 4) -> list[str]:
        wanted = terms(text) - FILLER
        rows = [i for i, row in self.search_rows().items() if i not in self.gone and len(wanted & row_terms(row)) >= 2]
        return sorted(rows, key=lambda i: (self.match_weight(i, text), i), reverse=True)[:count]

    def range_args(self, text: str) -> dict[str, str]:
        props = ((self.tools.get(self.roles.get("search")) or {}).get("parameters") or {}).get("properties") or {}
        wanted, found = nouns(text), {}
        for key, prop in props.items():
            labels = [v for v in prop.get("enum") or [] if isinstance(v, str) and nouns(v) and nouns(v) <= wanted]
            if labels:
                found[key] = max(labels, key=lambda v: len(nouns(v)))
        return found

    def cheapest(self, n: int) -> bool:
        priced = lambda goal, measure: goal == "cheapest" or (goal == "least" and bool(terms(measure) & {"price", "cost", "total", "spend", "spending"}))
        return priced(self.lines[n]["goal"], self.lines[n]["measure"]) or priced(self.order_goal, self.order_measure)

    def objective(self, n: int) -> tuple[str, str]:
        line = self.lines[n]
        return (line["goal"], line["measure"]) if line["goal"] in ("most", "least") else (self.order_goal, self.order_measure)

    def ranked_by_measure(self, n: int) -> bool:
        return self.objective(n)[0] in ("most", "least") and not self.cheapest(n)

    def measure_frame(self) -> dict[str, str] | None:
        measures = sorted({self.objective(n)[1] for n in self.lines if self.ranked_by_measure(n)})
        if not measures:
            return None
        wanted = set(key_words(measures[0]).split())
        stated = [str(v).lower() for i in self.grades for k, v in self.facts(i) if wanted and wanted <= set(key_words(k).split())]
        units = Counter(m.group(1) for v in stated if (m := re.search(r"\d\s*([a-z]+)", v)))
        return {"property": measures[0], "unit": units.most_common(1)[0][0] if units else "the unit the request uses"}

    def basket(self) -> dict[str, Any]:
        lines = list(self.lines)
        state = {n: {i: s for i in self.grades if (s := self.line_state(i, n)) != "out"} for n in lines}

        def order(n: int, i: str) -> tuple:
            value = (self.grades[i].get("measure") if self.ranked_by_measure(n) else 0) or None
            score = -value if value is not None and self.objective(n)[0] == "most" else value
            return (TIERS[state[n][i]], self.ranked_by_measure(n) and score is None, score or 0, self.price_of(i) is None, self.price_of(i) or 0)

        domains = {n: sorted(state[n], key=lambda i: order(n, i)) for n in lines}
        empty = [n for n in lines if not domains[n]]
        if empty or not lines:
            return {"status": "none", "best": None, "fallback": None, "better": [], "missing": [], "empty": empty, "incomplete": False}
        incomplete = math.prod(len(d) for d in domains.values()) > COMBOS
        if incomplete:
            domains = {n: d[: max(2, int(COMBOS ** (1 / len(lines))))] for n, d in domains.items()}
        found = []
        for combo in product(*domains.values()):
            chosen = dict(zip(lines, combo))
            if len(set(combo)) < len(combo):
                continue
            verdicts = {rid: self.pairs.get((rid, chosen[r["left"]], chosen[r["right"]]), "unknown") for rid, r in self.relations.items() if {r["left"], r["right"]} <= set(chosen)}
            total = sum((self.price_of(i) or 0) * self.lines[n]["quantity"] for n, i in chosen.items())
            if "no" in verdicts.values() or (self.budget is not None and total > self.budget):
                continue
            missing = [("line", n, i) for n, i in chosen.items() if state[n][i] != "met"]
            missing += [("pair", rid, (chosen[self.relations[rid]["left"]], chosen[self.relations[rid]["right"]])) for rid, v in verdicts.items() if v != "yes"]
            missing += [("price", n, i) for n, i in chosen.items() if self.price_of(i) is None and self.budget is not None]
            found.append((chosen, missing, total))
        rank = lambda c: (sum(order(n, i)[1] for n, i in c[0].items()), sum(order(n, i)[2] for n, i in c[0].items()), c[2])
        complete = [c for c in found if not c[1]]
        best = min(complete, key=rank) if complete else None
        better = sorted((c for c in found if c[1] and (best is None or rank(c) < rank(best))), key=rank)[:3]
        fallback = min(found, key=lambda c: (sum(TIERS[state[n][i]] for n, i in c[0].items()), len(c[1]), rank(c))) if found else None
        return {
            "status": "complete" if best else "evidence" if found else "none",
            "best": best[0] if best else None,
            "fallback": fallback[0] if fallback else None,
            "better": better,
            "missing": fallback[1] if fallback and not best else [],
            "empty": [],
            "incomplete": incomplete,
        }

    def doubts(self, result: dict[str, Any]) -> tuple[list[str], list[tuple[str, str, str]]]:
        weak = lambda i: not (self.current(i) and self.grades[i]["by"] == "judge")
        listings = [i for i in (result["best"] or result["fallback"] or {}).values() if weak(i)]
        listings += [i for _, missing, _ in result["better"] for kind, _, i in missing if kind == "line" and weak(i)]
        pairs = [(rid, *pair) for missing in [result["missing"]] + [c[1] for c in result["better"]] for kind, rid, pair in missing if kind == "pair"]
        return list(dict.fromkeys(listings))[: 2 * JUDGE_BATCH], list(dict.fromkeys(pairs))[:6]

    def settle(self) -> dict[str, Any]:
        for _ in range(2):
            result = self.basket()
            listings, pairs = self.doubts(result)
            if not (listings or pairs) or not self.judge(listings, final=True, pairs=tuple(pairs)):
                return result
        return self.basket()

    def missing_text(self, result: dict[str, Any]) -> str:
        if result["empty"]:
            lines = ", ".join(f"{n} ({self.lines[n]['name']})" for n in result["empty"])
            return f"no opened listing can be line {lines}; search with other words and open listings that could be it, then call the purchase tool again"
        if result["status"] == "none":
            return "every known combination breaks a rule joining the lines or the budget; search and open other listings, then call the purchase tool again"
        said = {
            "line": lambda n, i: f"line {n}: {i} does not show "
            + ", ".join(str(self.part(need, 0)) for need in self.checked(n) if self.need_state(i, n, need) != "met")
            + " (open its page or table, or other listings)",
            "pair": lambda rid, pair: f"rule {rid}: no listing words show whether {pair[0]} and {pair[1]} go together",
            "price": lambda n, i: f"line {n}: the price of {i} is not shown",
        }
        return "no basket is proven yet: " + "; ".join(said[kind](a, b) for kind, a, b in result["missing"][:6]) + "; settle this, then call the purchase tool again"

    def target(self, chosen: dict[int, str | None]) -> dict[str, int]:
        basket: dict[str, int] = {}
        for n, ident in chosen.items():
            if ident:
                basket[ident] = basket.get(ident, 0) + self.lines[n]["quantity"]
        return basket

    def cap_rule(self) -> str | None:
        made, page = self.search_template(), (self.page_rule() or ("",))[0]
        for e in self.searches():
            rows, hits, prices = self.result(e)
            for k, v in e["args"].items():
                if not made or k in (made[1], page) or isinstance(v, bool) or not isinstance(v, (int, float)) or not prices or max(prices) > v:
                    continue
                twins = [self.result(x)[1] for x in self.searches() if x["args"].get(made[1]) == e["args"].get(made[1]) and k not in x["args"]]
                if not twins or min(twins) > hits:
                    return k
        props = ((self.tools.get(self.roles.get("search")) or {}).get("parameters") or {}).get("properties") or {}
        numeric = [k for k, p in props.items() if p.get("type") in ("integer", "number") and "enum" not in p and k != page]
        return next((k for k in numeric if re.search(r"price|cost|cap|budget|spend|limit|max|under|ceiling", (k + " " + str(props[k].get("description", ""))).lower())), None)

    def add_to_cart(self, basket: dict[str, int], spare: int = 2) -> bool:
        tool, path = self.roles.get("write"), (self.ident_args.get(self.roles.get("write")) or [""])[0].strip("/")
        added = lambda args: isinstance(args, dict) and any(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in args.values())
        template = self.add_template or next((e["args"] for e in reversed(self.events) if e["tool"] == tool and added(e["args"]) and not self.removes(e["args"])), None)
        template = template or self.modes.get("add")
        if template and "<id>" not in serial(template):
            known = set(self.search_rows()) | set(self.grades) | set(self.actual_cart or {})
            template = placeholders(template, known) if known else template
        if template and "<id>" not in serial(template) and path and "/" not in path:
            template = {**template, path: "<id>"}
        if not template or "<id>" not in serial(template):
            self.trace.append({"kind": "add_failed", "template": bool(template), "path": path})
            return False
        for args in self.cart_args(template, basket):
            self.call_store(tool, args, spare=spare)
        viewer = self.roles.get("basket")
        if self.actual_cart is None and viewer in self.tools and not (self.tools[viewer].get("parameters") or {}).get("required"):
            self.call_store(viewer, {}, spare=spare)
        return bool(self.actual_cart)

    def cart_args(self, template: Any, basket: dict[str, int]) -> list[dict[str, Any]]:
        def fill(node: Any, ident: str, quantity: int) -> Any:
            if isinstance(node, dict):
                return {k: fill(v, ident, quantity) for k, v in node.items()}
            if isinstance(node, list):
                return [fill(v, ident, quantity) for v in node]
            return ident if isinstance(node, str) and "<id>" in node else quantity if isinstance(node, int) and not isinstance(node, bool) else node

        line_list = next((found for found in leaves_lists(template) if found and isinstance(found[0], dict) and "<id>" in serial(found[0])), None)
        if line_list is None:
            return [fill(template, ident, quantity) for ident, quantity in basket.items()]
        lines = [fill(line_list[0], ident, quantity) for ident, quantity in basket.items()]
        def swap(node: Any) -> Any:
            return lines if node is line_list else {k: swap(v) for k, v in node.items()} if isinstance(node, dict) else [swap(v) for v in node] if isinstance(node, list) else node

        return [swap(template)]

    def sets_whole_cart(self) -> bool:
        return any(found and isinstance(found[0], dict) and "<id>" in serial(found[0]) for found in leaves_lists(self.modes.get("add") or {}))

    def guess(self, n: int) -> str | None:
        words, rows, field, label = terms(self.line_kind(n)), self.search_rows(), self.category_field(), self.line_category(n)
        kin = lambda i: (label is not None and field is not None and rows[i].get(field) == label, len(words & row_terms(rows[i])))
        return max((i for i in rows if i not in self.gone and (kin(i)[0] or kin(i)[1])), key=kin, default=None)

    def explain(self, chosen: dict[int, str | None]) -> str:
        return "; ".join(f"line {n}: {i} x {self.lines[n]['quantity']} ({self.line_state(i, n)}, price {self.price_of(i)})" for n, i in chosen.items() if i)

    def pointers(self, ident: str) -> list[str]:
        own, asked = self.record_text(ident), plain(self.request + " " + serial(self.lines))
        title = plain((self.search_rows().get(ident) or {}).get("title") or "")
        codes = [c for c in dict.fromkeys(re.findall(r"\b[a-z]{1,4}\d{4,}\b", own)) if c not in title and c not in asked]
        rows = self.search_rows()
        return [i for c in codes for i, row in rows.items() if i != ident and c in plain(row.get("title") or "")]

    def follow_pointers(self, basket: dict[str, int]):
        for ident in basket:
            for target in self.pointers(ident):
                if target not in self.opened() and self.moves - self.step > 2:
                    self.open_listing(target)

    def removes(self, args: Any) -> bool:
        path = (self.ident_args.get(self.roles.get("write")) or [""])[0].strip("/")
        flags = {k: v for k, v in (self.modes.get("remove") or {}).items() if isinstance(v, str) and k != path}
        return isinstance(args, dict) and bool(flags) and all(args.get(k) == v for k, v in flags.items())

    def remove_line(self, ident: str, units: int = 1) -> bool:
        tool, path, add = self.roles.get("write"), (self.ident_args.get(self.roles.get("write")) or [""])[0].strip("/"), self.add_template or self.modes.get("add") or {}
        counted = lambda v: isinstance(v, int) and not isinstance(v, bool)
        mode = self.modes.get("remove") or {k: 0 if counted(v) else v for k, v in add.items()}
        if not mode or not path or "/" in path:
            self.trace.append({"kind": "remove_failed", "id": ident})
            return False
        units_args = {k: units for k, v in add.items() if counted(v) and k not in mode}
        return self.call_store(tool, {**{k: v if not counted(v) or v == 0 else units for k, v in mode.items()}, **units_args, path: ident}, spare=1)

    def reconcile(self, basket: dict[str, int]):
        viewer = self.roles.get("basket")
        viewable = viewer in self.tools and not (self.tools[viewer].get("parameters") or {}).get("required")
        if self.actual_cart is None and viewable and self.moves - self.step > 2:
            self.call_store(viewer, {}, spare=1)
        cart = dict(self.actual_cart or {})
        if self.sets_whole_cart():
            if cart != basket:
                self.add_to_cart(basket, spare=1)
        else:
            for ident, count in cart.items():
                if basket.get(ident) != count:
                    self.remove_line(ident, count)
            extra = {i: q for i, q in basket.items() if cart.get(i) != q}
            if extra:
                self.add_to_cart(extra, spare=1)
        if viewable and self.actual_cart is None and self.moves - self.step > 1:
            self.call_store(viewer, {}, spare=1)

    def close(self, tool: str, args: dict[str, Any], kind: str, forced: bool = False) -> str:
        viewer = self.roles.get("basket")
        viewable = viewer in self.tools and not (self.tools[viewer].get("parameters") or {}).get("required")
        if self.actual_cart is None and viewable and self.moves - self.step > 2:
            self.call_store(viewer, {}, spare=1)
        if not self.planned and not forced:
            self.prepare_questions()
        if not forced and self.questions:
            sent = False
            while self.questions and self.ask_next():
                sent = True
            if sent:
                return "not run: code opened more listings and asked the shopper about open preferences first; read the replies and the requirements, then finish again"
        covered = len(self.searches()) >= 3 and len(self.opened()) >= 3
        if kind == "decline" and not forced and not covered:
            return "not run: search more widely and open more full records before declining (at least three searches and three opened listings)"
        result = self.proven_basket()
        self.book.reserve = self.book.floor = 0.0
        unseen = any(self.unopened_rows(n) for n in self.lines)
        proof = result["status"] == "none" and covered and not unseen and (self.searched_out or self.moves - self.step <= self.reserve())
        if kind == "decline" and not forced and not proof:
            chosen = result["best"] or result["fallback"]
            known = "this basket fits what is known: " + self.explain(chosen) if chosen else self.missing_text(result)
            return self.finding("not run: not every line is ruled out by a stated requirement; " + known)
        kind = "decline" if proof and kind == "decline" else "purchase"
        spare_message = self.question_room(True) > self.question_room() and self.moves - self.step > self.closing_steps(False) + 4
        if kind == "purchase" and not forced and result["best"] and not self.late_asked and spare_message:
            self.late_asked = True
            if not self.questions:
                self.plan_questions(True)
            if self.questions and self.ask_next(True):
                return self.finding("not run: a message the order does not need was spent on one more question; read the reply and the requirements, then finish again")
        for _ in range(2):
            chosen = result["best"] or result["fallback"] or {}
            chosen = {n: chosen.get(n) or (self.guess(n) if forced and self.guess(n) not in chosen.values() else None) for n in self.lines}
            basket = self.target(chosen)
            if kind != "purchase":
                break
            for ident in basket:
                for reader in self.unread(ident):
                    if self.moves - self.step > 2:
                        self.open_listing(ident, reader)
            self.follow_pointers(basket)
            if all(self.current(i) for i in basket):
                break
            self.judge(sorted(basket), final=True)
            result = self.settle()
        wanted = basket or {i: q for i, q in (self.actual_cart or {}).items() if i not in self.gone}
        if kind == "purchase" and self.actual_cart != wanted and (result["best"] or forced or not basket):
            self.reconcile(wanted)
        approved = bool(result["best"]) and self.actual_cart == basket and all(self.current(i) for i in basket)
        self.trace.append(
            {"kind": "judgment", "mode": kind, "forced": forced, "status": result["status"], "best": basket, "cart": self.actual_cart, "approved": approved,
             "missing": result["missing"], "unopened": {n: len(self.unopened_rows(n)) for n in self.lines}, "searched_out": self.searched_out}
        )
        if kind == "purchase" and not approved and not forced and self.moves - self.step > self.closing_steps(False):
            changed = "the cart or a chosen listing changed while it was checked; read the cart, then call the purchase tool again"
            return self.finding("not run: " + (changed if result["best"] else self.missing_text(result)))
        if kind == "purchase" and not self.actual_cart:
            kind = "decline"
        messaged = any(e["tool"] == self.roles.get("message") for e in self.events[self.cart_event :])
        pending = [d["text"] for d in self.duties.values() if d["stage"] == "order" and d["messages"] and not messaged]
        if kind == "purchase" and pending and not self.duty_told:
            self.duty_told = True
            return self.finding("not run: the cart is final; do this procedure first, then call the purchase tool: " + " | ".join(pending))
        tool = self.roles.get(kind) if self.roles.get(kind) in self.tools else tool
        sent = args if self.ending_kind(tool, args) == kind and not schema_fault(args, self.tools[tool].get("parameters") or {}) else self.modes.get(kind) or {}
        if kind == "decline":
            named = lambda v: isinstance(v, str) and any(isinstance(x, str) and len(x) > 3 and x in v for x in self.search_rows())
            sent = {k: "Nothing in the store meets every requirement of the order." if named(v) else v for k, v in sent.items()}
        if not self.call_store(tool, sent, ending=True):
            return "not run: " + (self.not_run[-1]["reason"] if self.not_run else "the store call failed")
        if self.finished:
            return "the order was placed" if kind == "purchase" else "the decline was sent"
        self.apply_update(serial(self.events[-1]["envelope"]), len(self.events), strong=True)
        self.findings, self.duty_told = [], False
        for ident in sorted(set(basket) - self.gone):
            self.open_listing(ident)
        self.judge(sorted(set(basket) - self.gone), final=True)
        if self.basket()["status"] == "complete" and self.refusals < 2 and self.moves - self.step > 2:
            self.refusals += 1
            return self.close(tool, args, "purchase")
        self.fresh_values.update(self.lines)
        return "the checkout was refused: no order was placed; read the response, apply the change and choose again"

    def finding(self, text: str) -> str:
        self.findings = (self.findings + [text])[-4:]
        return text

    def act(self, tool: str, args: Any) -> tuple[bool, Any]:
        kind = self.ending_kind(tool, args)
        if kind:
            before = len(self.events)
            outcome = self.close(tool, args if isinstance(args, dict) else {}, kind)
            return len(self.events) > before, {"outcome": outcome, **({"event": self.event_view(len(self.events))} if len(self.events) > before else {})}
        if tool == self.roles.get("message") and not self.planned:
            before = len(self.events)
            if self.start_questions():
                return True, {
                    "outcome": "code opened more listings and asked the shopper its planned questions first; read the replies in dialogue and the requirements before messaging",
                    "results": [self.event_view(i) for i in range(before + 1, len(self.events) + 1)],
                }
        if tool == self.roles.get("message") and not (
            self.planned and not self.pending_asks() and any(d["messages"] and d["stage"] in ("choose", "order", "decline") for d in self.duties.values())
        ):
            return False, {"not_run": "code asks the shopper about open preferences first; send a message yourself only for a procedure the request sets, after code's questions"}
        if not isinstance(args, dict) or not self.call_store(tool, args):
            return False, {"not_run": self.not_run[-1]["reason"] if self.not_run else "the store call failed"}
        number = len(self.events)
        envelope = self.events[-1]["envelope"]
        if tool == self.roles.get("message"):
            self.walker_messages += 1
            self.dialogue.append([args, self.events[-1]["observation"]])
            self.apply_update(serial(envelope), number, strong=True)
        elif envelope.get("refusal"):
            self.apply_update(serial(envelope), number, strong=True)
        viewer = self.roles.get("basket")
        if tool == self.roles.get("write") and viewer in self.tools and not (self.tools[viewer].get("parameters") or {}).get("required"):
            self.call_store(viewer, {})
        return True, {"results": [self.event_view(i) for i in range(number, len(self.events) + 1)], "cart": self.actual_cart}

    def question_room(self, closing: bool = False) -> int:
        stages = ("order",) if closing else ("order", "decline")
        reserve = max(0, sum(min(1, d["messages"]) for d in self.duties.values() if d["stage"] in stages) - self.walker_messages)
        return max(0, (3 if self.max_messages is None else self.max_messages) - self.used_messages - reserve)

    def key_table(self, n: int) -> dict[str, list[Any]]:
        named = lambda key: " ".join(re.sub(r"[_\-]+", " ", spaced(str(key))).lower().split())
        rows, idents, flags = list(self.search_rows().values()), set(self.search_rows()), {"true", "false", "yes", "no"}
        free_text = lambda v: cell(v).lower() in idents | flags or len(cell(v)) >= 60 or len(cell(v).split()) >= 5
        schema = {named(k) for row in rows for k, v in row.items() if not isinstance(v, (dict, list))} | {named(k) for k in (self.category_field(), self.price_field) if k}
        offer = re.compile(r"\b(?:comments?|options?|variants?|models?|sku|codes?|ids?|units?|warrant\w*|shipping|sellers?|merchants?|brands?|stock|price|url|links?|images?)\b")
        mine = self.line_listings(n)
        fitting = [i for i in mine if self.line_state(i, n) != "out"]
        mine = fitting if len(fitting) >= 2 else mine
        carry, values = Counter(), {}
        for ident in mine:
            found = {named(k): v for k, v in self.facts(ident)}
            for key, value in found.items():
                if key not in schema and not offer.search(key) and not self.settled_key(n, key) and value and not free_text(value):
                    carry[key] += 1
                    values.setdefault(key, set()).add(value)
        name_like = lambda k: carry[k] >= 4 and len(values[k]) >= 0.6 * carry[k] and not any(c.isdigit() for v in values[k] for c in v)
        keys = [k for k in sorted(carry, key=lambda k: (-carry[k], len(k))) if not name_like(k)]
        twins = [k for i, k in enumerate(keys) if not any(values[k] == values[o] and nouns(k) & nouns(o) for o in keys[:i])]
        ranked = sorted(twins, key=lambda k: (-carry[k], -min(len(values[k]), 6), len(k)))[:16]
        return {k: [carry[k], sorted(values[k])[:4]] for k in ranked}

    def settled_key(self, n: int, key: str) -> bool:
        words = nouns(key)
        return not words or bool(words & {"price", "cost", "stock", "availability"}) or any(words <= nouns(self.part(need, 0)) for need in self.lines[n]["needs"])

    def line_listings(self, n: int) -> list[str]:
        field, label, rows = self.category_field(), self.line_category(n), self.search_rows()
        return [i for i in self.opened() if (self.grades.get(i) or {}).get("kind", {}).get(n) == "yes" or (label and field and (rows.get(i) or {}).get(field) == label)]

    def discover(self):
        self.discovered = True
        self.resolve_references()
        for line in self.lines:
            self.errand(line, first=True)
        self.judge(self.opened()[::-1])
        for n in self.lines:
            if len(self.line_listings(n)) < 5 and self.moves - self.step > self.reserve() + 3:
                self.screen(n, 5 - len(self.line_listings(n)))
        self.plan_questions()
        while self.questions and self.ask_next():
            pass
        self.trace.append({"kind": "discovery_done", "request_sha256": hashlib.sha256(self.request.encode()).hexdigest(), "steps": self.step,
                           "calls": [[e["tool"], e["args"]] for e in self.events], "lines": self.lines, "parts": self.need_parts, "kinds": self.need_kind,
                           "lookups": self.lookups, "messages_used": self.used_messages})

    def line_priority(self, n: int) -> int:
        median = lambda m: (lambda ps: ps[len(ps) // 2] if ps else 0.0)(sorted(p for i in self.line_listings(m) if (p := self.price_of(i)) is not None))
        return 0 if n == min(self.lines) else 1 if n == max(self.lines, key=median) else 2

    def plan_questions(self, closing: bool = False):
        room = self.question_room(closing)
        self.planned = not room or self.roles.get("message") not in self.tools
        if self.planned:
            return
        for n in self.lines:
            for ident in [i for i in (self.line_listings(n) or self.opened()) if self.unread(i)][:3] if len(self.key_table(n)) < 3 else []:
                if self.moves - self.step > self.reserve() + 2:
                    self.open_listing(ident, self.unread(ident)[0])
        self.judge(self.opened()[::-1])
        order = {n: {"product": line["name"], "musts": line["needs"]} for n, line in self.lines.items()}
        menus = {n: self.key_table(n) for n in self.lines}
        asked = {n: sorted(self.asked_keys.get(n, ())) for n in self.lines if self.asked_keys.get(n)}
        material = ("<order>\n" + serial({"request": self.request, "items": order}) + "\n</order>\n<menus>\n" + serial(menus) + "\n</menus>\n<asked>\n" + serial(asked)
                    + "\n</asked>\n<room>" + str(room) + "</room>")
        text = self.book.complete("planner", PLAN_RULES, material, final=True, tokens=1024)
        rows = records(text, PLAN_RULES)
        named, said = lambda r: r[3].strip() or self.lines[int(r[1])]["name"], lambda key: " ".join(re.sub(r"[_\-]+", " ", key).split())
        request = set(plain(self.request).split())
        quoted = lambda r: bool(r[4].strip()) and (words := set(plain(r[4]).split())) and len(words & request) >= max(2, round(0.6 * len(words)))
        usable = lambda r: r[0] == "ASK" and len(r) >= 5 and r[1].isdigit() and int(r[1]) in self.lines and r[2].strip() and (quoted(r) or not menus[int(r[1])])
        rows = [r for r in rows if usable(r) and not self.settled_key(int(r[1]), r[2])]
        asks = [["ASK", r[1], said(r[2]), f"What {said(r[2])} would you like for the {named(r)}?", named(r)] for r in rows]
        labels = room * self.list_size
        cap = labels if len(self.lines) == 1 else max(3, labels // len(self.lines))
        asks = [r for k, r in enumerate(asks) if sum(q[1] == r[1] for q in asks[:k]) < cap]
        for n, key in [(n, keys[k]) for k in range(25) for n, keys in ((n, list(menus[n])) for n in self.lines) if k < len(keys)]:
            if len(asks) < labels and not any(q[1] == str(n) and plain(q[2]) == plain(key) for q in asks):
                asks.append(["ASK", str(n), key, f"What {key} would you like for the {self.lines[n]['name']}?", self.lines[n]["name"]])
        self.questions = asks
        self.trace.append({"kind": "question_plan", "questions": list(self.questions), "answered": bool(text)})
        self.planned = bool(text) or sum(t["kind"] == "question_plan" for t in self.trace) >= 2

    def prepare_questions(self):
        for line in self.lines:
            self.errand(line)
        self.plan_questions()

    def start_questions(self) -> bool:
        before = len(self.events)
        self.prepare_questions()
        while self.questions and self.ask_next():
            pass
        return len(self.events) > before

    def list_text(self, batch: list[list[str]]) -> str:
        return f"What {', '.join(q[2] for q in batch[:-1])} or {batch[-1][2]} would you like for the {batch[0][4]}?" if len(batch) > 1 else batch[0][3]

    def ask_next(self, closing: bool = False) -> bool:
        tool = self.roles.get("message")
        props = (self.tools.get(tool, {}).get("parameters") or {}).get("properties", {})
        text_fields = [k for k, v in props.items() if v.get("type") == "string" and "enum" not in v]
        required = set((self.tools.get(tool, {}).get("parameters") or {}).get("required") or [])
        self.questions = [
            q for q in self.questions if not self.settled_key(int(q[1]), q[2]) and (q[3].startswith("Is there") or plain(q[2]) not in self.asked_keys.get(int(q[1]), set()))
        ]
        if not self.questions or len(text_fields) != 1 or not required <= set(text_fields):
            self.questions = []
            return False
        if not self.question_room(closing):
            return False
        pending = [q for q in self.questions if int(q[1]) not in self.revealed] or self.questions
        first = min(pending, key=lambda q: (self.asked.get(int(q[1]), 0), self.line_priority(int(q[1])), self.questions.index(q)))
        single = self.list_size <= 1 or first[3].startswith("Is there")
        batch = [first] if single else [q for q in self.questions if q[1] == first[1] and not q[3].startswith("Is there")][: self.list_size]
        while len(batch) > 1 and len(self.list_text(batch)) > (props[text_fields[0]].get("maxLength") or 10**6):
            batch.pop()
        text = self.list_text(batch)
        if not self.call_store(tool, {text_fields[0]: text}):
            self.questions = []
            return False
        for q in batch:
            self.questions.remove(q)
        line = int(first[1])
        self.asked[line] = self.asked.get(line, 0) + 1
        self.dialogue.append([text, self.events[-1]["observation"]])
        before = list(self.lines[line]["needs"])
        self.apply_update(serial(self.events[-1]["envelope"]), len(self.events), strong=True)
        new = [need for need in self.lines[line]["needs"] if need not in before]
        if self.last_answer == "need":
            self.revealed.add(line)
            self.resolve_references()
            self.errand(line, first=True)
        self.settle_labels(line, batch, new)
        self.trace.append({"kind": "asked", "line": line, "properties": [q[2] for q in batch], "answer": self.last_answer, "disclosed": new})
        if self.last_answer not in ("ambiguous", "crowded") and self.question_room(closing):
            self.plan_questions(closing)
        return True

    def settle_labels(self, line: int, batch: list[list[str]], new: list[str]):
        asked, done = batch[0], batch[:1]
        follow = lambda example="": ["ASK", asked[1], asked[2], f"Is there a {asked[2]} you need for the {asked[4]}{example}?", asked[4]]
        if len(batch) == 1:
            if self.last_answer in ("ambiguous", "crowded") and not asked[3].startswith("Is there") and self.question_room():
                seen = (self.key_table(line).get(asked[2]) or [0, []])[1][:2]
                self.questions.insert(0, follow(f", for example {' or '.join(seen)}" if len(seen) == 2 else ""))
        else:
            rest, others = [q for q in self.questions if q[1] == asked[1]], [q for q in self.questions if q[1] != asked[1]]
            if self.last_answer in ("crowded", "ambiguous") and not new:
                self.list_size = 1 if self.last_answer == "crowded" else self.list_size
                self.questions = ([asked] if self.last_answer == "crowded" else [follow()]) + batch[1:] + rest + others
                self.trace.append({"kind": "lists_off" if self.last_answer == "crowded" else "list_unclear", "line": line})
                return
            if not new:
                self.list_size = 1
                self.trace.append({"kind": "lists_off", "line": line, "reason": "a bundle drew nothing"})
            hits = []
            for need in new:
                scores = [len(nouns(self.part(need, 0)) & nouns(q[2])) for q in batch]
                if not any(scores):
                    scores = [len(nouns(self.part(need, 2)) & nouns(" ".join((self.key_table(line).get(q[2]) or [0, []])[1]))) for q in batch]
                hits += [scores.index(max(scores))] if any(scores) else []
            hit = min(hits) if hits else -1
            done, back = batch[: max(hit, 0) + 1], batch[max(hit, 0) + 1:]
            self.questions = back + rest + others
            self.trace.append({"kind": "list_settled", "line": line, "done": [q[2] for q in done], "next": [q[2] for q in self.questions[:4]], "list_size": self.list_size})
        self.asked_keys.setdefault(line, set()).update(plain(q[2]) for q in done)

    def trim_chat(self):
        for message in self.walk_chat[1:-6]:
            content = message.get("content")
            if message.get("role") in ("tool", "user") and isinstance(content, str) and len(content) > 600:
                ids = [i for i in dict.fromkeys(re.findall(r'"([A-Za-z0-9][\w\-:.]{5,40})"', content)) if i in self.grades or i in self.search_rows()][:20]
                message["content"] = content[:400] + "...(older result shortened; listings in it: " + ", ".join(ids) + ")"

    def walk(self) -> bool:
        if not self.discovered:
            self.discover()
        for line in sorted(self.fresh_values):
            self.fresh_values.discard(line)
            self.errand(line)
        if self.questions and self.ask_next():
            return True
        if len(serial(self.walk_chat)) > 120000:
            self.trim_chat()
        fresh = list(range(self.walk_seen + 1, len(self.events) + 1))
        turn = {"new_events": [self.event_view(i, 3000 if i > len(self.events) - 3 else 200) for i in fresh], "state": self.walk_state()}
        chat = (self.walk_chat or [{"role": "system", "content": WALK_RULES}]) + [{"role": "user", "content": serial(turn)}]
        self.book.last_message = None
        text = self.book.complete("scout", WALK_RULES, "", tokens=1024, tools=self.walk_tools, messages=chat)
        if not text:
            self.starved = (self.book.log[-1:] or [{}])[0].get("kind") == "inference_held"
            return False
        tool_call = next(iter((self.book.last_message or {}).get("tool_calls") or []), None)
        self.walk_chat = chat + [{"role": "assistant", "content": (self.book.last_message or {}).get("content") or text, **({"tool_calls": [tool_call]} if tool_call else {})}]
        self.walk_seen = len(self.events)
        parts = records(text, WALK_RULES)
        for r in parts:
            if r[0] == "NOTICE" and len(r) >= 3 and r[1].isdigit() and 1 <= int(r[1]) <= len(self.events):
                self.apply_update(serial(self.events[int(r[1]) - 1]["envelope"]), int(r[1]))
        call = next((r for r in parts if r[0] == "CALL" and len(r) == 3), None)
        if not call:
            started = not self.planned and self.start_questions()
            self.hold("no call was made; continue with the next useful store call")
            return started
        try:
            args = json.loads(call[2])
        except ValueError:
            args = call[2]
        progressed, outcome = self.act(call[1], args)
        if not progressed and len(self.not_run) >= 3 and len({serial(h["reason"]) for h in self.not_run[-3:]}) == 1:
            self.fresh_values.update(self.lines)
            outcome = {**outcome, "not_run": outcome.get("not_run", "") + "; code will search and open listings for every line next turn instead"}
        self.walk_chat.append(
            {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": serial(outcome)}
            if tool_call
            else {"role": "user", "content": serial({"result_of_your_call": outcome})}
        )
        self.walk_seen = len(self.events)
        return progressed

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return []
        time.sleep(random.random() * 1.5)
        self.read_order()
        ending = self.roles.get("purchase") if self.roles.get("purchase") in self.tools else self.roles.get("decline")
        for _ in range(2):
            idle = 0
            while not self.finished and self.step < self.moves and self.book.remaining() > 70 and idle < 6 and not self.starved:
                idle = 0 if self.walk() else idle + 1
            if self.finished or ending not in self.tools or self.step >= self.moves or self.book.remaining() <= 6:
                break
            outcome = self.close(ending, {}, "purchase", forced=True)
            self.trace.append({"kind": "closing", "outcome": outcome})
            if self.finished or not outcome.startswith("the checkout was refused"):
                break
        self.trace.append(
            {
                "kind": "episode_end",
                "agent": "MICKEY",
                "finished": self.finished,
                "steps": self.step,
                "spent": round(self.book.spent, 6),
                "judgments": self.judgments,
                "reason": "finished" if self.finished else "no finish within the remaining resources",
            }
        )
        return self.trace + self.book.log


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode = None
    try:
        episode = Mickey(problem_data)
        return episode.run()
    except Exception as error:
        record = {"kind": "agent_error", "error": type(error).__name__ + ": " + str(error)[:180]}
        return (episode.trace + episode.book.log if episode is not None else []) + [record]
