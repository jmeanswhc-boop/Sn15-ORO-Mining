"""Shopper-111: one procedure for ORO store worlds.

Stores change their tools, argument names and result layouts from race to race, and shoppers word
everything their own way, so code assumes none of it: models read every format once and code keeps
to what they named. Code holds only the hard limits: time, spend and steps, schema arguments, one line
per item and the budget total.

1. Brief   - models read the request and the tools.
2. Explore - a fast model calls the store's tools; results are read into a ledger of listings.
3. Ask     - when it has read enough, a model chooses store attribute names to ask about and others
             write the questions and read the replies.
4. Select  - one judgement picks the listing to buy or names further store calls and rows to read,
             repeated after every change; code sets the cart and orders.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------
# models and limits

_MODELS = {
    "openrouter": {
        "actor": "anthropic/claude-haiku-5.5",      # walks the store, words the questions, re-plans, selects (the king agents' choice; Sonnet behind it)
        "heavy": "anthropic/claude-sonnet-5.5",     # the brief, the must labels and the first question plan of an episode; fallback behind the selection
        "reader": "openai/gpt-6-luna",              # notes, terms, reply reading, layouts, label splits, references
        "alt": "mistralai/mistral-small-2603",      # last resort
    },
    "chutes": {
        "actor": "deepseek-ai/DeepSeek-V3.2-TEE",
        "heavy": "deepseek-ai/DeepSeek-V3.2-TEE",
        "reader": "deepseek-ai/DeepSeek-V3.2-TEE",
        "alt": "moonshotai/Kimi-K2.6-TEE",
    },
}

TASK_SECONDS = 272.0       # the sandbox allows 300 s per task
WRAP_SECONDS = 55.0        # below this, no new checks: head for the finish
FORCE_SECONDS = 22.0       # below this, the code finishes without the model
MAX_TURNS = 60
_NO_JSON_MODE: set[str] = set()  # models that refuse JSON mode (learned once per process)
DEFAULT_MESSAGES = 4       # messages to plan for when the request states no limit
OBS_CHARS = 4000
HISTORY_CHARS = 60000      # about 18k tokens; the unchanged part is a prompt-cache read, so a long history stays cheap
SPEND_SOFT = 0.035         # $ an episode may spend before Sonnet is no longer called
SPEND_HARD = 0.06          # $ after which the actor stops and the episode finishes with what it has
ALL_CALLS = 110            # model calls an episode may make (the dollar cap is the real limit; race episodes run ~1.9x the practice steps)


def _models() -> dict[str, str]:
    provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    return dict(_MODELS.get(provider, _MODELS["openrouter"]))   # a copy: an episode's table is its own


def _now() -> float:
    return time.monotonic()


# ---------------------------------------------------------------------------
# prompts

GUIDE = """You shop for one shopper in an online store, acting only through the store's tools. The episode scores 1 only when the finish is exactly right: an order holding one line per requested item, at the requested quantity, each listing meeting every requirement in force, in stock, with the total inside the budget; or, when the store truly has no listing that meets everything for some item, an ending without a purchase made the way the request asks.

The store
- Every tool call is one step and steps are limited; thinking is free. Use each tool's exact argument names and allowed values from its schema.
- Search results may be short and in no useful order. Compare the reported number of matches with the rows you have seen and fetch further pages with the exact page value the store gives when the answer depends on seeing all of them. Keyword searches work best short: the wanted value and the product noun; search that way first, since a category filter alone lists the whole category in store order and its pages are worth reading only under a price ceiling. Some details (what other products an item suits, codes it carries) may appear only in a listing's description, which a keyword search may not read: browse the item's category and read the rows' descriptions. A filtered search that returns nothing usually means the filter value is spelled differently from the store's labels: retry with a label exactly as result rows show it.
- A listing's attributes (its property table) are shown only by the view that opens it in full. Judge requirements on properties from those attributes, so open every listing you may buy. Never open a listing again when nothing has changed since you read it.
- The harness plans and sends the questions to the shopper once a listing of every item has been read. Their answers and any changes are added to the ledger in the status card.

What fits
- The kind of product is the listing's category label, nothing else: every listing under the label the status card names for an item is that item, whatever its title or description calls it, and a listing under any other label is not.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter, one entry of a list is enough, another word form or a plain synonym of the value counts, a value containing all the words of the wanted value counts; a merely related or broader value does not. Units are converted before numbers are compared; a limit includes its edge.
- An attribute whose key names the requirement's property decides; a key for another property, even a related one, decides nothing. When no attribute names the property, the wanted value stated in another attribute, the title or the description counts. A field listing the purchase options or variants a buyer can choose states no value of the listing itself.
- A requirement whose value comes from another listing ("the same as" or "unlike" a listing described by its seller and properties) is resolved by finding that listing and opening it in full, and any listing its page refers you to; the last page gives the value.
- Respect exclusions and closed lists. Wishes only break ties. Never buy a listing shown as sold out. The total of price times quantity over all lines may not exceed the budget.

Choosing
- Cheapest: the order must hold the lowest-priced listing among all fitting listings. Find the fitting set, check candidates in ascending price, use a maximum-price filter when the search has one to list only cheaper candidates.
- Most or least of a measure: compare every fitting listing on it; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right.
- Your job is to find and open the candidates of each item. The harness selects the listing to buy from the listings you opened, puts it in the cart and orders it: once the shopper has been asked, and whenever you call the order tool. Call the order tool as soon as you have opened listings that may fit; when the harness reports that no opened listing fits an item, search and open more of it.

Changes
- The shopper or the store can change things during the task: a new budget, a requirement added or dropped, a listing sold out, a price change. They arrive inside tool results, often as a refusal of the order. The episode goes on after a refusal: apply the change, fix the cart and order again.

Finishing
- You may cart a listing you judge right. At the order the harness judges the cart and carries out the steps the request asks for.
- End without a purchase only when your searches showed that some requested item has no fitting listing. Running short of steps shows nothing of the kind: then order the best cart you have.
- Every reply must call a tool. Keep your text short."""

BRIEF = """You prepare a shopping task for an agent. Read the shopper's request and return ONE JSON object (no prose) with these keys:
"items": one entry per distinct product: {"name": the kind of product in the request's words (a store range or department the request names is the name; a stated property value is a must, never part of the name), "quantity": units of that one listing (integer, 1 when unstated; a pair or set sold as one listing is 1), "musts": the firm requirements, each "property: constraint" with the exact value, limit, exclusion or closed list, the property named with the request's own noun, copied, never a neighbouring property's, "search": for each must naming a wanted value (not a limit, range or exclusion), a short phrase for a word search: the product's plain name and the value's own words; [] when none}. A requirement read from another listing is ONE must holding that listing's whole description; the words describing it are never musts of this product. Never put price, budget, stock, quantity, delivery, shipping or arrival timing into musts (listings state none of these; they are rules). Story (who uses it, where, why) is not a must unless it states a value outright; a value the shopper says they will check or verify themselves is a must whatever it is about ("they're for a wedding (I'll check)" gives "occasion: wedding"; the same words without the remark are story). Words that are part of what the product is called (a season, audience, style or use named with it) belong to "name", not musts. One phrase with one quantity is one item; separate items only when each has its own quantity or requirements.
"budget": the most the shopper will pay as a number in the request's unit, or null. "budget_unit": that unit in words, or null. "budget_scope": "total", "each" or null.
"in_stock_only": true unless the request clearly accepts unavailable items.
"pick": "cheapest" when the request asks for the lowest price, "most" or "least" for the extreme of a measure, else "any". "pick_measure": the measure, else "".
"messages": how many messages the shopper allows (integer), or null when not stated. "steps": how many steps or moves the request allows (integer), or null.
"lookups": the musts (copied exactly) whose value is read from another listing the request points at; a must that one requested item suit or fit another item of this order is no lookup: keep it as a must ("<property>: <relation> the <other item> in this order"). [] when none.
"rules": every instruction about how to work or how to end, each in at most 12 words.
"wishes": things the shopper would merely like.
"ask_first": things the request says to check with the shopper before choosing, each named by the one property it is about; [] when none.
"hints": story sentences pointing at a property the shopper cares about without giving its value (who will use it, how it should look, what it must go with): [{"item": item number, "property": plain name}]; [] when none. Who will wear or use it points at gender (or age group); its looks give the property "looks".
"process": steps the request requires besides finding and buying (an approval, telling or showing the shopper something, checking with them): [{"step": at most 15 words, "when": "choose", "order" (right before the order) or "end" (right before ending without a purchase), "messages": messages to the shopper it takes}]; [] when none. Requirements, the budget, limits and how to end when nothing fits are not steps; nor is the shopper's remark that they will check something themselves: it takes no message and marks the value before it as a must.
"quiet_decline": true when the request says not to name or suggest alternatives when nothing fits, else false."""

BRIEF_TOOLS = """You prepare a shopping task for an agent. Read the store's tools (their descriptions and parameter schemas) and return ONE JSON object (no prose) with these keys:
"shelves": when a tool lists the store's category labels, {item number: the label (copied exactly) for that requested item}; {} when no tool lists labels.
"roles": the exact tool name for each job, or null: "search" (finds listings by words or filters), "open" (shows one listing in full with its attributes; when two single-listing tools exist, the one promising attributes, properties or specifications), "offer" (shows one listing's current price and stock when that is a different tool from "open"), "compare" (shows several listings side by side), "message" (sends a message to the shopper), "cart_add" (puts a listing in the cart), "cart_remove" (takes a listing out; may be the same tool as cart_add), "cart_view" (shows the cart), "order" (places the order), "decline" (ends without a purchase; may be the same tool as order).
"query_arg": the name of the search tool's parameter that takes free words to search for, or null.
"cart_shape": how the cart tool takes its lines: {"lines_arg": its parameter holding a list of lines, or null when it takes one listing per call, "id_field": inside a line, the field naming the listing, or null when a line is just the id, "count_field": the parameter, or inside a line the field, for how many, or null, "whole": true when what a call names replaces the entire cart, false when it changes only the lines named}.
"value_args": {tool name: its parameter taking the listing id, or the list of listing ids, or for the messaging tool the message text}, for every tool with such a parameter.
"fixed_args": for each role whose tool needs a fixed argument value to do that job (a mode or kind argument), {role: {argument: value}}, values copied exactly from the schema's allowed values; for "search", the way of searching that takes free words; {} when none.
"filter_args": how the search tool lists a category under a price limit without words: {"shelf": its parameter taking a category label, or null, "cap": its parameter taking the highest price to show, or null, "fixed": {argument: value} for the mode that makes those parameters work (when a mode argument chooses between word search and filter search), {} when none}."""

CONSTRAINT = """"must": first work out, in one plain sentence, what a listing has to satisfy for this shopper, resolving double negatives and turns of phrase (refusing everything other than a value means that value is required; accepting anything other than a value, or saying never that value, means that value is ruled out). "constraint": that requirement in plain words: the comparison ("not", "at least", "at most", "no lower than", "within 2 inches of", "compatible with", "the same as", "different from", or nothing when the listing simply has the value) followed by the value with its unit, adding no limit the reply does not state. "points": true when the value is given by pointing at another listing instead of being stated; then, after the comparison, copy the reply's description of that listing up to where it ends, including any instruction to follow the page on to a further listing, and leave out the shopper's chatter after it. The properties and values that describe the pointed-at listing are part of its description, not requirements."""

MUSTS = """A shopping agent's brief lists the firm requirements ("musts") it took from the shopper's request below. Each must becomes a hard check: a listing must show every one, so one value the request states once but the brief lists twice, under two property names, asks every listing to state it twice and can make the task fail. Label every must:
"same": it repeats the value of another must of the same item under another property name, where the request states that value only once; give "same_as": the number of the must it repeats, the one worded closest to the request;
"kind": it states what the product is rather than a property's value: the range, department, season, audience or use the request names the product with ("fall/winter clothing", "men's shoes"), which the store's category label settles;
"own": anything else (a value the request states for its own property).
Give "quote": the request's words the must comes from, copied exactly.
Return ONE JSON object and nothing else: {"musts": [{"number": must number, "kind": "own", "same" or "kind", "same_as": must number or null, "quote": "..."}]}"""

TERMS = """A store records a product's properties under its own attribute names, often several names for one property, and a shopper's requirement is checked against them. For each requirement below, say how this store records it, from the item's attributes (each name with the values the listings read so far carry):
"keys": every attribute that records the requirement's property for this product, copied exactly: those whose name names that property (in any spelling, with or without the product's word or a prefix, so "shape" and "clock shape" record one property) and whose values are values of it; an attribute naming a different property records another one even when its values look alike ("style" is not "type"), and one whose values are of another kind records another property whatever its name; [] when none does, or when the requirement is about the category label or a pointer to another listing.
"meets": the values under those keys, copied exactly, that satisfy the requirement; "fails": their other values. A value meets it when it or one of its entries states the wanted value: the same words in any form or order, a value holding all the words of the wanted value or held whole inside it, or a plain synonym naming the same thing; a value naming a different thing, or only a related or looser quality, fails. For a limit, compare the numbers in the value's unit: values inside the limit (the edge included) meet it; for an exclusion, values other than the excluded meet it.
Return ONE JSON object and nothing else: {"terms": [{"number": requirement number, "keys": [], "meets": [], "fails": []}]}"""

NOTE = """While a shopping agent works, the store showed the lines below (they may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when they change the task (budget, requirements, a price, availability, a refusal of the order), "error" when they only report that a call was malformed or not accepted, "other" otherwise,
 "budget": the new total limit as {"amount": number, "unit": the unit as the lines write it, or ""}, or null,
 "add": [{"item": item number, "need": "property: constraint", "points": true when its value is to be read from another listing}], "drop": [{"item": item number, "need": a current requirement copied exactly that is no longer wanted}],
 "quantity": [{"item": item number, "quantity": integer}],
 "gone": [listing ids said to be unavailable], "sold_out": true when a chosen listing is said to be gone without its id, "price_moved": true when a price is said to have changed,
 "wait": true when the lines ask for something before the order goes ahead (an answer, a figure, a look at something) rather than agreeing or changing the task, else false,
 "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. A new budget replaces the old one whether higher or lower. Shoppers write in their own voice, so judge from the meaning of the lines, not their wording. In a need, the constraint follows these rules: """ + CONSTRAINT

PLAN = """You choose the questions a shopping agent asks a shopper to find the preferences the request does not state. The shopper holds a few firm preferences they did not write down, each about one property of one requested item, and such a property goes by the name of one of the store's attributes. A question naming that exact attribute name reveals the preference. A synonym, a near word, or a broader or narrower name does not: the shopper shrugs or does not follow. One message can carry up to three names of one item, and there are very few messages, so which names you choose decides the result.
Each item below lists its attributes: the name, on how many of the listings read so far it appears, and some values. Label every listed attribute of every item, copying each name exactly as written: "kind" is "property" for the name of a property a person would say and understand at once, "settled" for one a requirement in force already decides, "identifier" for a code, an abbreviation, a stray prefix or a name whose value differs on every listing, "options" for a field listing purchase variants, "selling" for brand, seller, condition, warranty or delivery, "duplicate" for a second name of a property already given. The code asks what the request says to check (ASK FIRST) and what its story hints at (STORY HINTS) first, then the properties most carried first, so leave none out.
When an item's list may miss much of what the store records for that product, say so under "read" with how many more of its listings to read first, a few at a time; you choose again after they are read, and see which new names they added; when fewer unread listings of it are on the shelf than you ask for, the agent searches for more first. Listings of one product repeat each other's attributes, so stop when reads add little. Each one spends a free step, as STEPS below shows, and the free steps must also pay for finding listings with the values the shopper then states. Give "ask" from what the list shows now.
Return ONE JSON object and nothing else: {"read": [{"item": item number, "count": how many}], "ask": [{"item": item number, "label": "the attribute name copied exactly", "kind": "property"}]}"""

READ = """A shopping agent sent a shopper a message. Read the shopper's reply for what it means. Shoppers write in their own voice: slang, jokes, side stories and sayings, worded differently every time, so nothing about the wording can be assumed. Decide from the meaning of the whole reply, as a person reading it would, and read past the voice and any story to what is being communicated.
"meaning" is one of:
"value": the shopper states what they want: a value, a limit, an exclusion, or a value given by pointing at another listing. A reply pointing at another listing states a value even though finding that listing is left to the agent.
"no_preference": the shopper took in the question and leaves the choice to the agent, because they do not care, have no view or have never thought about it. Being unsure of the answer and leaving it to the agent belongs here.
"not_understood": the shopper did not take in the message: they ask for it again, or in other or simpler words, or say they could not follow it or were distracted. It says nothing about the property.
"request": the shopper wants something else from the agent first (one thing at a time, an answer, a figure, a look at something); put it in "wants".
"other": anything else.
For "value", list in "needs" every requirement stated: "item": the requested item it is about, as its number, read from the reply's own words when they name a product, else the item the message was about; "property": the property as the reply itself words it, without the product's name; "constraint": the value exactly as stated, with its unit and limit words such as at least or at most, without the connecting words around it. """ + CONSTRAINT + """
"crowded": true when the reply asks for fewer things at once, one question at a time, or says the message was too much; else false. "drop": current requirements the shopper gives up, each {"item": item number, "need": the requirement copied exactly}. A requirement counts as given up only when the reply itself says so about it. A reply that has no preference, or does not follow the message, gives nothing up: the requirements in force stay, even when the message named the same or a similar property. "budget": a new spending limit the reply states, or null.
"search": when a firm value is stated, the short words to search the store for it (the value and the product noun), else "".
Return ONE JSON object and nothing else: {"quote": "the words of the reply that carry its meaning", "meaning": "...", "wants": "", "needs": [{"item": 1, "property": "...", "must": "...", "constraint": "...", "points": false}], "crowded": false, "drop": [], "budget": null, "search": ""}"""

TALK = """A shopping agent is talking with a shopper to learn the firm preferences the request does not state. Each preference is about one property of a requested item, and goes by the name of one of the store's attributes. Only a message naming that exact name reveals it. A reply not understood or shrugged at almost always means none of the names asked is one of the shopper's properties, whatever the wording: rewording rarely helps, so go on to the next names.
Write the agent's next message: one clean question a person would understand at once, naming exactly the names listed under NAMES FOR THIS MESSAGE, in that order, for example "What pattern, sleeves or collar type would you like for the jumpsuit?" (one name: "What pattern would you like for the jumpsuit?"). When the names belong to two items, one message covers both, for example "What color or size would you like for the floor lamp, and what type for the bulbs?". Use each name word for word as listed; do not swap in a synonym, drop one or explain it. Refer to each product by a short plain name. Add no example values, choices, greeting, apology or second request, and keep to one sentence. When the shopper wants something, give it and go on. Send nothing ("next": null) only when no name is listed.
Return ONE JSON object and nothing else: {"next": {"asks": [{"item": item number, "name": "a name asked, as listed"}], "message": "..."} or null}"""

READING = """Reading rules: a requirement on a property is met when the listing's attribute for that property states the value (any key spelling; one entry of a list or of a comma-separated value is enough; case, singular/plural and plain synonyms count; a value containing all the words of the wanted value counts; units converted; limits include the edge; a range must lie wholly inside a limit). An attribute whose key names the requirement's property decides, and when it states something else the listing fails; when a key is spelled exactly like the property it alone decides (a longer key holding its words, such as "clothing material" beside "material", does not overrule it); only when no key is spelled exactly like it, the listing meets it when any key naming it states the value. A key for a different property decides nothing. When no attribute names the property, the value stated in another attribute, the title or the description counts. A field listing the purchase options or variants a buyer can choose states no value of the listing itself. The store also writes in its own sentences (remarks, notes): such a sentence can state a property's value, say that a property is to be taken from another listing (named by its kind, its seller and its values) rather than from this one, or give a pairing table (which value goes with which); apply them as written, the way the store words them. A requirement that the listing be available, in stock or on hand is decided by the record's in_stock field alone, true meeting it. A requirement the category label itself states is met by that label. A listing's kind is its category label: each record says which requested item its label holds, and a title or description calling it something else does not change its kind. A requirement defined through another listing is met only when that listing's record is shown and states a matching value, or, for an "unlike" rule, unless the records show the same value. Pairs: when the request or a requirement says one item must work with, fit or suit another requested item (or a model the shopper names), the two listings' own words must connect them (a code or name both records carry, or a statement on one record that covers the other); a pair whose records show no such link is wrong. Sold-out listings and listings over the budget do not fit. STORE TERMS, when shown, say which attribute records a requirement in this store and which of its values meet it: they overrule the rules above."""

LAYOUT = """A store tool returned the text below. Code must turn such text into listing records with named fields, so say how it is laid out. Copy every separator exactly, spaces included.
Return ONE JSON object and nothing else: {"kind": "separators" when fixed separators split its fields, "free" when it describes listings in a layout no separators capture, "not_records" when it is not about listings' fields (a message, a note, an error),
 "record_separator": what separates listings when one text holds several, or null, "field_separator": what separates fields, or null, "pair_separator": what separates a field's name from its value, or null,
 "table_field": the name of a field whose value holds further name/value pairs (a listing's attributes), or null, "table_entry_separator": what separates those pairs, or null, "table_pair_separator": what separates a name from its value inside them, or null}"""

EXTRACT = """A store tool returned the text below, describing listings in free form. Write it as data, leaving nothing out and changing no value. Use the same field names for the same things in every record, and put a listing's attributes in an object under "attributes".
Return ONE JSON object and nothing else: {"records": [{field name: value}]}"""

FIELDS = """Below are records a store's tools returned. Name the job of their fields, copying each field name exactly, or null when no field does that job or these are not product listings.
Return ONE JSON object and nothing else: {"id": the field holding the value tools take to name a listing, distinct for each listing, "title": the listing's name, "price": what one unit costs now, "stock": whether it can be bought now, "category": the store's category label, "text": the free text describing it, "attributes": the field holding the listing's table of attributes, "quantity": how many of it a cart line holds}"""

STOCK = """A store describes whether a listing is available with the text below. Can it be bought now? Return ONE JSON object and nothing else: {"buyable": true, false, or null when the text does not say}"""

FINISH = """A shopping agent has run out of steps or time before its cart holds a line for every requested item. The episode scores only an exact finish: an order holding one line per item, each meeting every requirement, or, when the store truly has no fitting listing for some item, an ending without a purchase. From what the agent found, say which ending is more likely right. Return ONE JSON object and nothing else: {"end": "order" or "decline", "reason": "one sentence"}"""

RESULT = """A store's search returned a result with listing rows and the other fields below, one per line as name = value. Say which field NAME holds how many listings matched in all, and which field NAME holds what to pass to get the next page. Answer with the name on the left of the = sign, never the value. Judge a field by its name as well as its value: on a last page the next-page field is null or empty, and it is still that field. Answer null only when no field does that job.
Return ONE JSON object and nothing else: {"count": "field name", "next": "field name"}"""

PROCESS = """A shopping agent must carry out a step the shopper's request asks for, by sending the shopper one message. Write that message: one or two short plain sentences that do exactly this step from the facts shown (name the cart's items, quantities and total when the step is about the basket or the order; name the listings read so far, by what they are and cost, when the step is about what the shelf offers), and nothing else. State only what the records show. Return ONE JSON object and nothing else: {"message": "..."}"""

SELECT = """You choose what a shopping agent buys. For each requested item, pick from its OPENED listings the one to buy: one that meets every requirement in force, in stock and inside the budget, and under the pick rule the best of those that fit (cheapest: the lowest price among all fitting listings; most or least: the extreme of the measure, and a listing that does not state the measure cannot win; any: any listing that fits, preferring one whose attributes state the required values plainly). A wish only breaks ties between fitting listings: never reject a listing for a wish it does not show. When a listing is proposed for an item (PROPOSED BY THE AGENT: the agent's own pick, or a line already in the cart), choose it if it fits every requirement in force, unless the pick rule prefers another fitting listing; a requirement the shopper dropped never unfits it.
""" + READING + """
For every requirement of the item give the evidence on the chosen listing: the attribute key that decides it (copied exactly), or "title", "description" or "category" only when no key of that listing's own record names the property, and the value written there.
For a chosen listing also give "backups": the other OPENED listings that meet every requirement in force, best first; [] when there is none. When no opened listing fits an item, or under a cheapest, most or least pick a ROW could beat your choice (cheaper, or possibly more or less of the measure) and could fit, name those ROWS under "open", most promising first.
You also decide whether to look further in the store. Under "calls" name store calls to run now, each {"tool": a tool listed under STORE TOOLS, "args": its arguments}: a further page using the store's next page value, a price limit or a category listing when the tool has them, or a search with other words. Compare what each search matched with the rows seen, and repeat no search of SEARCHES SO FAR. Look further only when the answer could change: when nothing seen fits the item, or under a cheapest, most or least pick while listings not yet seen could beat your choice. Every call costs a step. The first order is often refused because a listing sold out, and a replacement then takes more steps, so the first order must go out with steps to spare: name no call when a fitting choice is good enough to order now.
Return ONE JSON object and nothing else: {"items": [{"item": item number, "choice": listing id or null, "evidence": [{"requirement": copied from the list, "source": key or "title" / "description" / "category", "value": copied, "met": true or false}], "backups": [OPENED ids], "open": [ROW ids], "calls": [{"tool": "...", "args": {}}], "reason": "one sentence"}]}"""

LABELS = """A store files every listing under a category label, and files each requested item under one label. For each label below, say what it is to a requested item: "names" when the label names the item's kind of product as the request names it (the same kind under another wording, or a group label that lists the item's kind by name, such as "A & B" for an item asked for as A or B), "narrower" when it is one range or sub-kind of the item (an item asked for as "A or B" is not named by a label "B": that label is narrower), "wider" when it is a range that contains the item among other kinds, and "other" when its listings are another kind of product, or the same product for another style, audience or use. Return ONE JSON object and nothing else: {"labels": {"<label copied exactly>": {"item": item number or null, "kind": "names" | "narrower" | "wider" | "other"}}}"""

TRAIL = """A requirement on a product takes its value from another listing instead of stating it. The described listing's page can itself send the reader on to a further listing for that property. Follow the chain from the requirement's description, link by link, using the records below (OPENED: the page was read in full; ROW: seen only as a search row).
Return ONE JSON object and nothing else: {"links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes it}], "done": true when the last link is an OPENED page that states the property's value itself, "value": the value the requirement asks for, in the pages' own words: the last page's value, or, when the requirement asks for what pairs with, suits or goes with that value and a page gives the pairing, the paired value; or "", "look_for": when the chain stops at a listing not found yet, a search phrase for it as the store's listings word it: its seller, its category label copied exactly as the records write it, and its values, without property names, and none already tried under SEARCHES SO FAR (after a search that matched nothing, or found only listings that are not it, use fewer words or other words of its description); "" only when it is found}
List the links in order and stop at the first one that is not OPENED, since its words are still unknown. A link is the listing whose kind, seller and values match the description (units converted, in the record's attributes or in its own sentences); an OPENED page that shows another value for one of them is not the link, so it never ends the chain. When ROWS of that kind and seller could be it but their rows do not show the values, give the likeliest one not opened yet as the next link: opening it shows its values."""

# ---------------------------------------------------------------------------
# small helpers

def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return str(value)


def _clip(value: Any, limit: int = 400, depth: int = 0) -> Any:
    """Shorten long strings and long lists inside a result, keeping its structure."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if depth > 6:
        return "..."
    if isinstance(value, dict):
        return {k: _clip(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        out = [_clip(v, limit, depth + 1) for v in value[:40]]
        if len(value) > 40:
            out.append(f"... {len(value) - 40} more")
        return out
    return value


def _parse_json(text: Any) -> dict[str, Any] | None:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:index + 1])
                    except Exception:
                        break
                    return value if isinstance(value, dict) else None
        start = text.find("{", start + 1)
    return None


def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        found = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if found:
            return float(found.group())
    if isinstance(value, dict):
        for inner in value.values():
            number = _number(inner)
            if number is not None:
                return number
    return None


def _plain(key: Any) -> str:
    """A key in plain lower-case words: camel case, underscores and dashes split."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return " ".join(re.split(r"[\s_\-.]+", text.lower())).strip()


def _kinds(schema: Any) -> list[str]:
    if not isinstance(schema, dict):
        return []
    kind = schema.get("type")
    if isinstance(kind, list):
        return [str(k) for k in kind]
    if isinstance(kind, str):
        return [kind]
    out: list[str] = []
    for key in ("anyOf", "oneOf"):
        for option in schema.get(key) or []:
            out += _kinds(option)
    return out


def _fit_value(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Coerce one argument value to its schema where that is unambiguous; report what cannot fit."""
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        if value in enum:
            return value
        match = [e for e in enum if isinstance(e, str) and isinstance(value, str) and e.lower() == value.strip().lower()]
        if match:
            return match[0]
        faults.append(f"{where} must be one of {enum}")
        return value
    if "integer" in kinds and not isinstance(value, bool):
        number = _int(value)
        if number is not None:
            value = number
        elif "string" not in kinds and "null" not in kinds:
            faults.append(f"{where} must be a whole number")
            return value
    elif "number" in kinds and isinstance(value, str) and "string" not in kinds:
        number = _number(value)
        if number is None:
            faults.append(f"{where} must be a number")
            return value
        value = number
    if "string" in kinds and isinstance(value, (int, float)) and not isinstance(value, bool) and not ({"integer", "number"} & set(kinds)):
        value = str(value)
    if "array" in kinds and not isinstance(value, list) and value is not None:
        value = [value]
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        value = [_fit_value(v, schema["items"], f"{where}[{i}]", faults) for i, v in enumerate(value)]
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            faults.append(f"{where} takes at most {schema['maxItems']} entries")
    if isinstance(value, dict) and isinstance(schema.get("properties"), dict):
        value = _fit_args(value, schema, where, faults)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]:
            faults.append(f"{where} must be at least {schema['minimum']}")
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]:
            faults.append(f"{where} must be at most {schema['maximum']}")
    if isinstance(value, str) and isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
        value = value[: schema["maxLength"]]
    return value


def _fit_args(args: Any, schema: Any, where: str = "arguments", faults: list[str] | None = None) -> dict[str, Any]:
    faults = faults if faults is not None else []
    if not isinstance(args, dict):
        faults.append(f"{where} must be one JSON object")
        return {}
    props = (schema or {}).get("properties") if isinstance(schema, dict) else None
    props = props if isinstance(props, dict) else {}
    out: dict[str, Any] = {}
    for key, value in args.items():
        if props and key not in props:
            # a near spelling of a parameter is that parameter; anything else is left out
            near = [p for p in props if _plain(p) == _plain(key)]
            if not near:
                faults.append(f"{key} is not a parameter of this tool (its parameters: {', '.join(props)})")   # dropping it would run a different call than the one asked
                continue
            key = near[0]
        out[key] = _fit_value(value, props.get(key), key, faults)
    for key in (schema or {}).get("required") or []:
        if key not in out:
            faults.append(f"missing required argument {key}")
    return out


def _strings(node: Any, depth: int = 0) -> list[str]:
    if depth > 5:
        return []
    if isinstance(node, str):
        return [node] if node.strip() else []
    if isinstance(node, dict):
        return [s for v in node.values() for s in _strings(v, depth + 1)]
    if isinstance(node, list):
        return [s for v in node for s in _strings(v, depth + 1)]
    return []


# ---------------------------------------------------------------------------
# reply shapes: every structured model reply is checked against the shape its call declares

def _shape_check(value: Any, spec: Any, path: str, errors: list[str], ctx: dict[str, Any]) -> Any:
    """Check one value against a spec and return it cleaned (entries of a list that do not fit are left out).
    Specs: "str", "text" (non-empty string), "int", "num", "bool", "any", "id" (a known listing id), "item" (a requested
    item number), ("opt", spec), ("enum", values), ("list", spec), ("obj", {key: spec}, required keys), ("map", spec)."""
    if isinstance(spec, tuple) and spec[0] == "opt":
        return None if value is None else _shape_check(value, spec[1], path, errors, ctx)
    if spec == "any":
        return value
    if spec in ("str", "text"):
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, (int, float)):
            value = str(value)
        if not isinstance(value, str) or (spec == "text" and not value.strip()):
            errors.append(f"{path} must be a {'non-empty ' if spec == 'text' else ''}string")
            return None
        return value
    if spec == "int":
        number = _int(value)
        if number is None:
            errors.append(f"{path} must be a whole number")
        return number
    if spec == "num":
        number = value if isinstance(value, (int, float)) and not isinstance(value, bool) else (_number(value) if isinstance(value, str) else None)
        if number is None:
            errors.append(f"{path} must be a number")
        return number
    if spec == "bool":
        if not isinstance(value, bool):
            errors.append(f"{path} must be true or false")
            return None
        return value
    if spec == "id":
        if not isinstance(value, (str, int)) or str(value) not in ctx.get("ids", ()):
            errors.append(f"{path} {_dump(value)[:40]} is not an id of the records given")
            return None
        return str(value)
    if spec == "item":
        number = _int(value)
        if number is None or not 1 <= number <= ctx.get("items", 1):
            errors.append(f"{path} must be an item number from 1 to {ctx.get('items', 1)}")
            return None
        return number
    kind = spec[0] if isinstance(spec, tuple) else None
    if kind == "enum":
        if value not in spec[1]:
            errors.append(f"{path} must be one of {_dump(list(spec[1]))[:200]}")
            return None
        return value
    if kind == "list":
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        out = []
        for index, entry in enumerate(value):
            local: list[str] = []
            cleaned = _shape_check(entry, spec[1], f"{path}[{index}]", local, ctx)
            errors += local
            if not local:
                out.append(cleaned)
        return out
    if kind == "map":
        if not isinstance(value, dict):
            errors.append(f"{path} must be an object")
            return {}
        out = {}
        for key, entry in value.items():
            local = []
            cleaned = _shape_check(entry, spec[1], f"{path}.{key}", local, ctx)
            errors += local
            if not local:
                out[key] = cleaned
        return out
    if kind == "obj":
        if not isinstance(value, dict):
            errors.append(f"{path or 'the reply'} must be a JSON object")
            return {}
        out = dict(value)
        for key, sub in spec[1].items():
            if key in value:
                local = []
                out[key] = _shape_check(value[key], sub, f"{path}.{key}" if path else key, local, ctx)
                errors += local
            elif key in (spec[2] if len(spec) > 2 else ()):
                errors.append(f"{path + '.' if path else ''}{key} is missing")
        return out
    return value


def _pairs(text: str, field: str, pair: str) -> dict[str, str]:
    """A text's name/value pairs under the separators a store uses. A piece naming no field belongs to the value before it."""
    out: dict[str, str] = {}
    last = None
    for piece in text.split(field):
        name, sep, value = piece.partition(pair)
        if sep and name.strip() and len(name) <= 40:
            last = name.strip()
            out[last] = value.strip()
        elif last:
            out[last] += field + piece
    return out


def _typed(value: str) -> Any:
    """Text as the value it writes: nothing, true/false, a number; anything else stays text."""
    try:
        got = json.loads({"none": "null", "true": "true", "false": "false"}.get(value.strip().lower(), value.strip()))
    except ValueError:
        return value
    return got if got is None or isinstance(got, (bool, int, float)) else value


NEED_SPEC = ("obj", {"item": "item", "need": "text", "points": ("opt", "bool")}, ("need",))
BUDGET_SPEC = ("opt", ("obj", {"amount": "num", "unit": ("opt", "str")}, ("amount",)))
SPECS: dict[str, Any] = {
    "note": ("obj", {"kind": ("enum", ("change", "error", "other")), "budget": BUDGET_SPEC, "add": ("list", NEED_SPEC), "drop": ("list", NEED_SPEC),
                     "quantity": ("list", ("obj", {"item": "item", "quantity": "int"}, ("quantity",))), "gone": ("list", "id"), "sold_out": ("opt", "bool"),
                     "price_moved": ("opt", "bool"), "wait": ("opt", "bool"), "summary": ("opt", "str")}, ("kind",)),
    "plan": ("obj", {"read": ("list", ("obj", {"item": "item", "count": "int"}, ("count",))), "ask": ("list", ("obj", {"item": "item", "label": "text", "kind": ("opt", "str")}, ("label",)))}, ("ask",)),
    "read": ("obj", {"quote": ("opt", "str"), "meaning": ("enum", ("value", "no_preference", "not_understood", "request", "other")), "wants": ("opt", "str"), "search": ("opt", "str"),
                     "needs": ("list", ("obj", {"item": "item", "must": ("opt", "str"), "property": "text", "constraint": "text", "points": ("opt", "bool")}, ("property", "constraint"))),
                     "crowded": ("opt", "bool"), "drop": ("list", NEED_SPEC), "budget": BUDGET_SPEC}, ("meaning",)),
    "talk": ("obj", {"next": ("opt", ("obj", {"asks": ("list", ("obj", {"item": "item", "name": "text"}, ("name",))), "message": "text"}, ("asks", "message")))}),
    "trail": ("obj", {"links": ("list", ("obj", {"id": ("opt", "str"), "about": ("opt", "str")})), "done": "bool", "value": ("opt", "str"), "look_for": ("opt", "str")}, ("links", "done")),
    "process": ("obj", {"message": "text"}, ("message",)),
    "terms": ("obj", {"terms": ("list", ("obj", {"number": "int", "keys": ("list", "str"), "meets": ("list", "str"), "fails": ("list", "str")}, ("number",)))}, ("terms",)),
    "select": ("obj", {"items": ("list", ("obj", {"item": "item", "choice": ("opt", "str"), "evidence": ("list", ("obj", {"requirement": "text", "source": "str", "value": ("opt", "str"), "met": "bool"}, ("requirement", "source", "met"))),
                                              "backups": ("list", "str"), "open": ("list", "str"), "calls": ("list", ("obj", {"tool": "text", "args": ("obj", {})}, ("tool",))), "reason": ("opt", "str")}, ("item",)))}, ("items",)),
}


# ---------------------------------------------------------------------------
# the episode

class Item:
    def __init__(self, name: str, shelf: str, quantity: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.quantity = quantity
        self.musts = musts            # from the request
        self.told: list[str] = []     # from the shopper's answers and notes
        self.dropped: list[str] = []
        self.free: list[str] = []     # properties the shopper does not mind about

    def needs(self) -> list[str]:
        return [n for n in self.musts + self.told if n not in self.dropped]


class Episode:
    def __init__(self, problem: dict[str, Any]):
        self.started = _now()
        environment = problem["environment"]
        self.binding = dict(environment.get("binding") or {})
        view = environment.get("policy_view") or {}
        self.request = str(view.get("query") or "")
        self.opening = view.get("opening")
        self.tools = [t for t in view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict) and t["function"].get("name")]
        self.schemas = {t["function"]["name"]: (t["function"].get("parameters") if isinstance(t["function"].get("parameters"), dict) else {"type": "object", "properties": {}}) for t in self.tools}
        steps = _int(view.get("max_steps"))
        self.max_steps = steps if steps and steps > 0 else 30
        self.step = 0
        self.finished = bool(view.get("finished"))
        self.models = _models()
        self.llm = ProxyClient(timeout=60, max_retries=1, retry_delay=1)
        self.env = ProxyClient(timeout=90, max_retries=2, retry_delay=1)
        self.trace: list[dict[str, Any]] = []
        self.cost = 0.0
        self.calls = 0                # model calls made (ALL_CALLS bounds them)
        self.dead = False             # the inference key is out of credit
        self.invalid_replies: dict[str, int] = {}   # role -> replies that failed their shape or grounding

        # brief
        self.brief: dict[str, Any] = {}
        self.items: list[Item] = []
        self.roles: dict[str, str] = {}
        self.fixed: dict[str, dict[str, Any]] = {}
        self.budget: float | None = None          # as the shopper states it, compared with the store's prices as plain numbers
        self.budget_unit = ""
        self.pick = "any"
        self.msg_limit = 0
        self.msg_used = 0
        self.process: list[dict[str, Any]] = []
        self.version = 0              # bumps whenever the requirements, budget or quantities change

        # store reading
        self.meanings: list[str] = []                  # what each shopper reply meant, as the reader read it
        self.talked = False                            # the questions have been sent
        self.layouts: dict[str, dict[str, Any]] = {}   # for each tool, how it writes records as text, when it does
        self.extracted: dict[str, list[dict[str, Any]]] = {}
        self.shape_tries: dict[frozenset[str], int] = {}
        self.shapes: list[frozenset[str]] = []         # record layouts whose field jobs a model has named
        self.cart_shape: dict[str, Any] = {}
        self.keys: dict[str, str | None] = {"id": None, "price": None, "stock": None, "title": None, "text": None, "category": None, "attributes": None, "quantity": None}
        self.lines_seen: set[str] = set()              # note lines already read
        self.listings: dict[str, dict[str, Any]] = {}
        self.seen_order: list[str] = []
        self.labels: dict[str, int] = {}     # category label (lower case) -> item number, 0 for none
        self.notes: list[str] = []           # changes not yet shown to the actor
        self.list_ok: dict[str, int] = {}       # tool -> most listings one call has shown
        self.list_failed: dict[str, int] = {}   # tool -> fewest listings a call was turned down for
        self.read_step: dict[str, int] = {}     # listing -> step its page was last read
        self.change_step = -1                    # step of the last change to prices, stock or the order's terms
        self.refs: list[dict[str, Any]] = []     # requirements whose value is read from another listing
        self.replaced: set[str] = set()          # dead cart lines already handled

        # cart
        self.cart: dict[str, int] = {}
        self.cart_version = 0
        self.cart_viewed_version = -1
        self.gone: set[str] = set()

        # questions
        self.asked = False
        self.qa: list[tuple[int, str, str]] = []

        # finishing
        self.passed: set[str] = set()        # check keys already run (a hold happens at most once per state)
        self.process_done: set[str] = set()
        self.refusals = 0
        self.selection: dict[int, dict[str, Any]] = {}
        self.selected_key = ""
        self.filter: dict[str, Any] = {}   # the search tool's filter mode: a category label under a price cap (shelf, cap, fixed)
        self.lists = True                  # whether this shopper takes a message naming several properties (until a reply says otherwise)
        self.short: dict[int, int] = {}    # item -> rows seen when the question planner wanted more of its listings than the shelf showed
        self.waited: set[int] = set()      # items the plan has waited for once: the next plan reads what the shelf shows
        self.deferred: set[tuple[int, str]] = set()   # names asked behind a name the shopper shrugged at: asked again only after the fresh ones
        self.glanced: dict[tuple[int, str], int] = {}   # (item, fitting choice) -> cheaper rows read while it held
        self.capped: set[tuple[int, str, float]] = set()
        self.why = ""          # the reason the deciding component gives for the next store step, logged with it
        self.actor_why = ""    # the actor's own words for its latest turn
        self.terms: dict[tuple[int, str], dict[str, Any]] = {}   # (item, requirement) -> the store attribute recording it, its values meeting and failing it
        self.proposed: dict[int, str] = {}   # item -> the listing the actor last asked to cart for it
        self.seed: list[tuple[int, str]] = []   # keyword phrases for the request's stated values, searched before the actor's first turn
        self.searches: list[tuple[str, dict[str, Any], list[str], int | None, Any]] = []   # (tool, arguments, listings shown, matched, next-page value) of every search
        self.result_fields: dict[str, dict[str, Any]] = {}   # tool -> the fields of its result holding the match count and next-page value
        self.stock_words: dict[str, bool | None] = {}
        self.called: set[str] = set()             # store calls the selector asked for and the harness ran
        self.settle_due = False
        self.history: list[dict[str, Any]] = []

    # -- bookkeeping -------------------------------------------------------

    def note(self, kind: str, **data: Any) -> None:
        entry = {"kind": kind, "t": round(_now() - self.started, 1), "step": self.step}
        entry.update(data)
        self.trace.append(entry)

    def time_left(self) -> float:
        return TASK_SECONDS - (_now() - self.started)

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def msgs_free(self) -> int:
        """Messages left for questions: the ones the request's own steps at the order or the ending take stay back."""
        # only the steps due at the order are kept back: an ending without a purchase is the rare finish, and a message
        # kept for it would cost a question in every episode
        return max(0, self.msg_limit - self.msg_used - self.order_messages())

    def order_messages(self) -> int:
        """Messages the request's steps at the order still take for the current cart (each step's own count, all of them)."""
        key = _dump(sorted(self.cart.items()))
        return sum(p["messages"] for p in self.process if p["when"] == "order" and p["step"] + "|" + key not in self.process_done)

    def tail(self) -> int:
        """Steps kept for the finish: the order, the replacement a refused first order needs (taking the line out and putting
        another in, one call when a call states the whole cart, then ordering again), the look at the cart before ordering
        when the store has one, and one for each message the steps at the order still take."""
        return (3 if self.cart_shape.get("whole") else 4) + bool(self.roles.get("cart_view")) + min(self.order_messages(), max(0, self.msg_limit - self.msg_used))

    def needed(self, questions: bool = True) -> int:
        """Steps the rest of the episode still needs: the finish, the cart calls for items without a line (one when a call
        states the whole cart), and the questions not yet sent. Reading and searching may spend only what is above this."""
        lines = sum(1 for n in range(1, len(self.items) + 1) if not self.valid_line(n))
        return self.tail() + (min(lines, 1) if self.cart_shape.get("whole") else lines) + (self.msgs_free() if questions and not self.talked and self.roles.get("message") else 0)

    def steps_text(self) -> str:
        left, need = self.steps_left(), self.needed()
        return f"STEPS: {left} left; {need} needed to finish (questions not yet sent, cart, order, and a replacement if the first order is refused); {max(0, left - need)} free for reading and searching, one each."

    # -- inference ---------------------------------------------------------

    def infer(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None, role: str = "", json_mode: bool = False) -> dict[str, Any] | None:
        """One model reply (the first model that answers). The reply carries "_finish" (why the model stopped) and "_model"."""
        for model in models:
            left = self.time_left()
            if self.dead or left < FORCE_SECONDS - 8 or self.calls >= ALL_CALLS:
                return None
            if model == self.models["heavy"] and self.cost > SPEND_SOFT:
                continue
            self.calls += 1
            self.llm.timeout = int(max(8, min(55, left - FORCE_SECONDS + 6)))
            sent = messages
            if model.startswith("anthropic/") and len(messages) >= 2:
                # prompt cache on the system text and on the last message before the status card (the card is always last
                # and changes every turn; everything before it stays as it was)
                sent = [dict(m) for m in messages]
                last = len(sent) - 2 if str(sent[-1].get("content") or "").startswith("[status] ") else len(sent) - 1
                for index in {0, max(1, last)}:
                    content = sent[index].get("content")
                    if isinstance(content, str) and content:
                        sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
            body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": max_tokens}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            if json_mode and model not in _NO_JSON_MODE and not model.startswith("anthropic/"):
                body["response_format"] = {"type": "json_object"}
            result = self.llm.post_verbose("/inference/chat/completions", body)
            if "response_format" in body and result.data is None and (result.error or {}).get("status") in (400, 422):
                # this model does not take JSON mode: remember it and ask again without
                _NO_JSON_MODE.add(model)
                body.pop("response_format")
                result = self.llm.post_verbose("/inference/chat/completions", body)
            data = result.data
            if isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                cost = usage.get("cost")
                if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                    self.cost += float(cost)
                self.note("llm", role=role, model=model, cost=cost, prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
                choice = data["choices"][0] or {}
                message = dict(choice.get("message") or {})
                message["_finish"] = choice.get("finish_reason")
                message["_model"] = model
                if message.get("content") or message.get("tool_calls"):
                    return message
                self.note("llm_empty", role=role, model=model, finish=choice.get("finish_reason"))
            error = result.error or {}
            self.note("llm_failure", role=role, model=model, status=error.get("status"), body=str(error.get("body") or "")[:200])
            if error.get("status") == 402:
                self.dead = True
                return None
        return None

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1500, role: str = "", spec: Any = None,
                 ground: Any = None, ctx: dict[str, Any] | None = None, strict: bool = False, escalate: bool = False) -> dict[str, Any] | None:
        """A structured reply: parsed, checked against the role's shape, then against its sources (`ground` returns the
        claims the sources do not bear out). A failing reply goes back to the same model once with the exact errors, and a
        reply cut off by its length limit comes again with room; then the next model. Every failure is recorded. When no
        reply passes, the last one that parsed is returned cleaned (entries that broke the shape left out)."""
        spec = spec if spec is not None else SPECS.get(role)
        ctx = dict(ctx or {})
        ctx.setdefault("items", len(self.items) or 1)
        ctx.setdefault("ids", self.listings)
        fallback: dict[str, Any] | None = None
        for model in models:
            tokens, feedback = max_tokens, ""
            for attempt in range(2):
                if attempt and self.time_left() < WRAP_SECONDS:
                    break
                message = self.infer([{"role": "user", "content": prompt + feedback}], [model], tokens, role=role, json_mode=True)
                if message is None:
                    break
                text = message.get("content") if isinstance(message.get("content"), str) else ""
                parsed = _parse_json(text)
                errors: list[str] = []
                cleaned: dict[str, Any] | None = None
                grounding = False
                if parsed is None:
                    if message.get("_finish") == "length":
                        errors.append("the reply was cut off by its length limit before the JSON object ended")
                        tokens = min(int(tokens * 1.6), 4000)   # more room once, within a bound: reasoning counts against it
                    else:
                        errors.append("the reply held no complete JSON object")
                else:
                    cleaned = _shape_check(parsed, spec, "", errors, ctx) if spec is not None else parsed
                    if not errors and ground is not None:
                        errors += ground(cleaned)
                        grounding = bool(errors)
                if not errors:
                    return cleaned
                self.invalid_replies[role] = self.invalid_replies.get(role, 0) + 1
                self.note("llm_invalid", role=role, model=model, finish=message.get("_finish"), errors=errors[:6], text=text[:300])
                if cleaned:
                    fallback = cleaned
                if grounding and attempt == 1 and not strict and not escalate:
                    return cleaned
                if not (escalate or strict or message.get("_finish") == "length"):
                    break   # one attempt per model: a second costs as much as the first and rarely does better
                feedback = ("\n\nYour previous reply could not be used: " + "; ".join(errors[:8])
                            + ". Return only the corrected JSON object" + (", kept short" if message.get("_finish") == "length" else "") + ".")
        return None if strict else fallback

    # -- the store -----------------------------------------------------------

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """One store step. Returns the store's response, or None when the call could not be made."""
        payload: dict[str, Any] = dict(self.binding)
        for attempt in range(3):
            left = self.time_left()
            if left < 2 or self.finished:
                return None
            payload["step"] = self.step + 1
            payload["action"] = {"name": name, "args": args}
            self.env.timeout = int(max(3, min(90, left - 1)))
            result = self.env.post_verbose("/environment/call", payload)
            data = result.data
            if isinstance(data, dict):
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                frame, chain = sys._getframe(1), []
                while frame and len(chain) < 5:   # the component that asked for this step: the agent's methods on the call stack
                    if frame.f_code.co_name in type(self).__dict__ and frame.f_code.co_name not in ("act", "perform"):
                        chain.append(frame.f_code.co_name)
                    frame = frame.f_back
                self.note("env", name=name, args=args, observation=_clip(data.get("observation"), 300), finished=bool(data.get("finished")),
                          by=">".join(reversed(chain)), why=self.why or (self.actor_why if "run_calls" in chain else ""), left=self.max_steps - self.step + 1,
                          reserve=self.needed(), msgs=self.msg_used, cart=dict(self.cart), choice={n: e.get("choice") for n, e in self.selection.items()})
                self.why = ""
                return {**data, "observation": self.decode(name, data.get("observation"))}
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.note("env_failure", name=name, status=status, body=body[:200])
            if status == 409:
                expected = re.search(r"expected\D{0,12}(\d+)", body)
                if expected:
                    self.step = int(expected.group(1)) - 1
                    continue
                self.finished = True
                return None
            if status == 429:
                time.sleep(1.5 + attempt)
                continue
            return None
        return None

    # -- the brief -----------------------------------------------------------

    def read_request(self) -> None:
        # two readings merged into one brief: the request (what to buy and how; the strong model, since every later judgment
        # rests on it) and the tools (which tool does which job and how; a fast model, grounded against the schemas)
        spec: Any = ("obj", {"items": ("list", ("obj", {"name": "text", "quantity": ("opt", "int"), "musts": ("list", "str"), "search": ("list", "str")}, ("name",))),
                        "budget": ("opt", "num"), "budget_unit": ("opt", "str"), "budget_scope": ("opt", ("enum", ("total", "each"))), "in_stock_only": ("opt", "bool"),
                        "pick": ("enum", ("cheapest", "most", "least", "any")), "pick_measure": ("opt", "str"), "messages": ("opt", "int"), "steps": ("opt", "int"),
                        "lookups": ("list", "str"), "rules": ("list", "str"), "wishes": ("list", "str"), "ask_first": ("list", "str"),
                        "hints": ("list", ("obj", {"item": "item", "property": "text"}, ("property",))),
                        "process": ("list", ("obj", {"step": "text", "when": ("enum", ("choose", "order", "end")), "messages": ("opt", "int")}, ("step", "when"))),
                        "quiet_decline": ("opt", "bool")}, ("items", "budget", "pick", "messages"))
        brief: dict[str, Any] = self.ask_json(BRIEF + "\n\nREQUEST:\n" + self.request, [self.models["heavy"], self.models["actor"]], 4000, role="brief", spec=spec,
                                              ground=self.ground_items, ctx={"items": 20}, escalate=True) or {}
        named = [str(e.get("name") or "the product") for e in _items(brief.get("items")) if isinstance(e, dict)] or ["the requested product"]
        lines = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(self.schemas[t['function']['name']])}" for t in self.tools]
        prompt = BRIEF_TOOLS + "\n\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUESTED ITEMS:\n" + "\n".join(f"{n}: {name}" for n, name in enumerate(named, 1))
        if self.opening is not None:
            prompt += "\n\nWHAT THE STORE SHOWS FIRST:\n" + _dump(_clip(self.opening, 300))[:3000]
        spec = ("obj", {"shelves": ("map", ("opt", "str")), "roles": ("map", ("opt", ("enum", tuple(self.schemas)))), "fixed_args": ("map", ("obj", {})),
                        "filter_args": ("opt", ("obj", {"shelf": ("opt", "str"), "cap": ("opt", "str"), "fixed": ("opt", ("obj", {}))})), 
                        "query_arg": ("opt", "str"), "value_args": ("map", ("opt", "str")), "cart_shape": ("opt", ("obj", {"lines_arg": ("opt", "str"), "id_field": ("opt", "str"), "count_field": ("opt", "str"), "whole": ("opt", "bool")}))}, ("roles",))
        brief.update(self.ask_json(prompt, [self.models["actor"], self.models["heavy"]], 4000, role="brief_tools", spec=spec, ground=self.ground_brief, escalate=True) or {})
        self.brief = brief
        roles = brief.get("roles") if isinstance(brief.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        self.cart_shape = brief.get("cart_shape") if isinstance(brief.get("cart_shape"), dict) else {}
        if self.cart_shape.get("whole") and self.roles.get("cart_add"):
            self.roles["cart_remove"] = self.roles["cart_add"]
        fixed = brief.get("fixed_args") if isinstance(brief.get("fixed_args"), dict) else {}
        for role, args in fixed.items():
            if role in self.roles and isinstance(args, dict):
                faults: list[str] = []
                props = (self.schemas[self.roles[role]].get("properties") or {})
                clean = {k: _fit_value(v, props.get(k), k, faults) for k, v in args.items() if k in props}
                if not faults and clean:
                    self.fixed[role] = clean
        f = brief.get("filter_args") if isinstance(brief.get("filter_args"), dict) else {}
        props = (self.schemas.get(self.roles.get("search") or "") or {}).get("properties") or {}
        if f.get("shelf") in props and f.get("cap") in props and {"integer", "number"} & set(_kinds(props[f["cap"]])):
            self.filter = {"shelf": f["shelf"], "cap": f["cap"], "fixed": {k: v for k, v in (f.get("fixed") or {}).items() if k in props}}
        shelves: dict[str, Any] = brief["shelves"] if isinstance(brief.get("shelves"), dict) else {}
        for number, entry in enumerate(_items(brief.get("items")), 1):
            if isinstance(entry, dict):
                self.seed += [(number, str(ph).strip()) for ph in _items(entry.get("search")) if str(ph).strip()]
                quantity = _int(entry.get("quantity"))
                self.items.append(Item(str(entry.get("name") or "the product"), str(shelves.get(str(number)) or ""), quantity if quantity and quantity > 0 else 1,
                                       [str(m).strip() for m in _items(entry.get("musts")) if str(m).strip()]))
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]
        budget = _number(brief.get("budget")) if brief.get("budget") is not None else None
        if budget and budget > 0:
            self.set_budget(budget, str(brief.get("budget_unit") or ""))
        self.pick = brief.get("pick") if brief.get("pick") in ("cheapest", "most", "least") else "any"
        limit = _int(brief.get("messages"))
        self.msg_limit = (limit if limit is not None and limit >= 0 else DEFAULT_MESSAGES) if self.roles.get("message") else 0
        steps = _int(brief.get("steps"))
        if steps is not None and 5 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in _items(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = str(entry.get("when") or "order").strip().lower()
                self.process.append({"step": str(entry["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order",
                                     "messages": max(0, _int(entry.get("messages")) or 0)})
        self.check_musts([str(x).strip() for x in _items(brief.get("lookups"))])
        for lookup in _items(brief.get("lookups")):
            self.add_ref(None, str(lookup).strip())
        for item in self.items:
            if item.shelf:
                self.labels[item.shelf.strip().lower()] = self.items.index(item) + 1
        self.note("brief", brief=brief, roles=self.roles, fixed=self.fixed)

    def check_musts(self, lookups: list[str]) -> None:
        """A model labels each must of the brief as its own condition, a repeat of another must's value under another property
        name, or the kind of product; repeats leave and kinds move into the item's name (a must's quote has to be the request's own words)."""
        rows = [(n, m) for n, item in enumerate(self.items, 1) for m in item.musts if m not in lookups]
        if not rows:
            return
        plain = _plain(self.request)
        spec = ("obj", {"musts": ("list", ("obj", {"number": "int", "kind": ("enum", ("own", "same", "kind")), "same_as": ("opt", "int"), "quote": "text"}, ("number", "kind", "quote")))})
        ground = lambda g: [f"must {e['number']}: the quote is not the request's own words" for e in g.get("musts") or [] if _plain(e["quote"]) not in plain] + \
            [f"must {e['number']}: same_as must name another must of the same item" for e in g.get("musts") or [] if e["kind"] == "same" and not (e.get("same_as") in range(1, len(rows) + 1) and e["same_as"] != e["number"] and rows[e["same_as"] - 1][0] == rows[e["number"] - 1][0])]
        got = self.ask_json(MUSTS + "\n\nREQUEST:\n" + self.request + "\n\nMUSTS:\n" + "\n".join(f"{k}. item {n} ({self.items[n - 1].name}): {m}" for k, (n, m) in enumerate(rows, 1)),
                            [self.models["heavy"], self.models["actor"]], 1500, role="musts", spec=spec, ground=ground) or {}
        gone = {e["number"]: e["kind"] for e in got.get("musts") or [] if e["kind"] != "own" and 1 <= e["number"] <= len(rows)}
        for number in sorted(gone):
            n, must = rows[number - 1]
            item = self.items[n - 1]
            if gone[number] == "kind":   # what the product is goes into its name, where the category label settles it
                value = must.split(":", 1)[-1].strip()
                item.name = item.name if _plain(value) in _plain(item.name) else f"{value} {item.name}"
                item.musts.remove(must)
            elif len(item.musts) > 1:
                item.musts.remove(must)
        self.note("musts", labels=got.get("musts"), removed=[rows[k - 1] for k in gone])

    # -- reading results -------------------------------------------------------

    def ground_items(self, brief: dict[str, Any]) -> list[str]:
        """The request half: one entry per distinct product (the code tells items apart by name)."""
        names = [str(e.get("name") or "").strip().lower() for e in _items(brief.get("items")) if isinstance(e, dict)]
        return ["two items name the same product: give one entry per distinct product (the code cannot tell same-named items apart)"] * (len(set(names)) < len(names))

    def ground_brief(self, brief: dict[str, Any]) -> list[str]:
        """The tools half: only what the schemas can check: fixed arguments, modes and named parameters must exist on the tool."""
        errors = []
        for role, args in (brief.get("fixed_args") or {}).items():
            tool = (brief.get("roles") or {}).get(role)
            props = (self.schemas.get(tool or "") or {}).get("properties") or {}
            for key, value in (args or {}).items():
                enum = (props.get(key) or {}).get("enum")
                if key not in props:
                    errors.append(f"fixed_args.{role}.{key} is not a parameter of {tool}")
                elif isinstance(enum, list) and value not in enum:
                    errors.append(f"fixed_args.{role}.{key} must be one of {enum}")
        roles, fixed = brief.get("roles") or {}, brief.get("fixed_args") or {}
        missing = [r for r in ("search", "cart_add", "order") if not roles.get(r)]
        if missing and len(self.schemas) > 2:
            errors.append(f"roles {', '.join(missing)} name no tool: every store has a tool for each of these jobs; name it")
        for one, other in (("cart_add", "cart_remove"), ("order", "decline")):
            schema = self.schemas.get(roles.get(one) or "") or {}
            modes = [k for k in schema.get("required") or [] if "enum" in ((schema.get("properties") or {}).get(k) or {})]
            if roles.get(one) and roles.get(one) == roles.get(other) and modes and not all(k in (fixed.get(r) or {}) for r in (one, other) for k in modes):
                errors.append(f"{one} and {other} share {roles[one]}: give fixed_args for both, with the value of {', '.join(modes)} that does each job")
        search = (self.schemas.get(roles.get("search") or "") or {}).get("properties") or {}
        for role, tool in roles.items():
            schema = self.schemas.get(tool or "") or {}
            props = schema.get("properties") or {}
            needed = [k for k, v in props.items() if isinstance(v, dict) and "enum" in v] if role == "search" else [k for k in schema.get("required") or [] if "enum" in (props.get(k) or {})]
            errors += [f"fixed_args.{role} must give {k}: one of {props[k]['enum']}" for k in needed if k not in (fixed.get(role) or {})]
        if brief.get("query_arg") and "string" not in _kinds(search.get(brief["query_arg"]) or {}):
            errors.append(f"query_arg {brief['query_arg']!r} is not a text parameter of the search tool")
        f = brief.get("filter_args") if isinstance(brief.get("filter_args"), dict) else {}
        errors += [f"filter_args.{k} {f[k]!r} is not a parameter of the search tool" for k in ("shelf", "cap") if f.get(k) and f[k] not in search]
        errors += [f"filter_args.fixed.{k} must be one of {search[k]['enum']}" for k, v in (f.get("fixed") or {}).items() if isinstance(search.get(k), dict) and isinstance(search[k].get("enum"), list) and v not in search[k]["enum"]]
        return errors

    def learn_keys(self, rows: list[dict[str, Any]]) -> None:
        """The job of each field of a store's records, named by a model once for each layout of record the store shows (a layout
        inside one already read needs nothing). When a job moves to another field, every listing is read again."""
        rows = [r for r in rows if isinstance(r, dict)]
        shape = frozenset(k for r in rows for k in r)
        known_id = self.keys["id"] in shape
        if not rows or any(shape <= known for known in self.shapes):
            return
        plain = sorted(k for k in shape if not isinstance(rows[0].get(k), (dict, list)))
        distinct = [k for k in plain if all(isinstance(r.get(k), str) and r[k].strip() for r in rows) and len({r[k] for r in rows}) == len(rows)]

        def ground(got: dict[str, Any]) -> list[str]:
            return ([f"id must be one of the fields distinct on every record: {distinct}"] * bool(got.get("id") and got["id"] not in distinct)
                    + [f"{job} is not a field: {got[job]}" for job in self.keys if got.get(job) and got[job] not in shape])
        spec: Any = ("obj", {job: ("opt", "str") for job in self.keys})
        got = self.ask_json(FIELDS + "\n\nRECORDS:\n" + _dump(_clip(rows[:3], 160)), [self.models["reader"], self.models["actor"]], 800, role="fields", spec=spec, ground=ground, strict=True) or {}
        self.shape_tries[shape] = self.shape_tries.get(shape, 0) + 1
        if got or self.shape_tries[shape] >= 2:
            self.shapes.append(shape)
        if not got.get("id") and not known_id:
            return
        before = dict(self.keys)
        self.keys.update({job: got[job] for job in self.keys if got.get(job) and (not self.keys[job] or (job == "id" and not known_id))})   # a job already named stays with its field
        if self.keys != before:
            for listing in self.listings.values():
                self.read_fields(listing, listing.get("raw") or {})

    def decode(self, name: str, observation: Any) -> Any:
        """A store result as data, whatever its layout: text holding JSON is read as JSON, and text holding listing records is
        split by the layout a model reads once for each tool, from that tool's longest such text. A message to or from the shopper stays text."""
        if name == self.roles.get("message"):
            return observation
        if name not in self.layouts:
            texts = self.record_texts(observation)
            if texts:
                sample = max(texts, key=len)
                spec: Any = ("obj", {"kind": ("enum", ("separators", "free", "not_records")), **{k: ("opt", "str") for k in ("record_separator", "field_separator", "pair_separator", "table_field", "table_entry_separator", "table_pair_separator")}}, ("kind",))

                def ground(got: dict[str, Any]) -> list[str]:
                    pairs = _pairs(sample, str(got.get("field_separator") or "\0"), str(got.get("pair_separator") or "\0"))
                    errors = ["these separators do not split the text into three or more named fields"] * (got.get("kind") == "separators" and len(pairs) < 3)
                    table, entry, pair = (str(got.get(k) or "") for k in ("table_field", "table_entry_separator", "table_pair_separator"))
                    if table and got.get("kind") == "separators":   # the table field must hold name/value pairs under its separators on this very text
                        cells = [c for c in str(dict(pairs).get(table, "")).split(entry or "\0") if c.strip()]
                        if table not in dict(pairs) or len(cells) < 2 or sum(1 for c in cells if pair and pair in c) < len(cells) / 2:
                            errors.append(f"table_field {table!r} is not a field of the text whose value splits by {entry!r} into two or more entries each holding {pair!r}; name the field that does, or null")
                    return errors
                got = self.ask_json(LAYOUT + "\n\nTEXT:\n" + sample[:2500], [self.models["reader"], self.models["actor"]], 800, role="layout", spec=spec, ground=ground, strict=True) or {}
                # a store writes every tool's records alike: the layout that reads the sample fullest (an attribute table, then most fields) is kept, so one tool's messy sample cannot blind that tool for the episode
                def fullness(layout: dict[str, Any]) -> tuple[bool, int]:
                    rec = self.records_in(sample, True, layout)
                    rec = rec[0] if isinstance(rec, list) and rec else rec
                    return (isinstance(rec, dict) and any(isinstance(v, dict) and v for v in rec.values()), len(rec) if isinstance(rec, dict) else 0)
                self.layouts[name] = max([got] + [l for l in self.layouts.values() if l.get("kind") == "separators"], key=fullness)
                self.note("layout", tool=name, layout=self.layouts[name], sample=sample[:160])

        def walk(node: Any, depth: int) -> Any:
            if isinstance(node, dict):
                return {k: walk(v, depth + 1) for k, v in node.items()}
            if isinstance(node, list):
                return [walk(v, depth + 1) for v in node]
            if not isinstance(node, str) or depth > 6:
                return node
            data = _parse_json(node) if node.strip()[:1] in ("{", "[") else None
            return walk(data, depth + 1) if data is not None else self.records_in(node, node in standing, self.layouts.get(name) or {})
        standing = set(self.record_texts(observation))
        return walk(observation, 0)

    def record_texts(self, node: Any, depth: int = 0) -> list[str]:
        """Text standing where a record could: the result itself, an entry of a list, the only text of an object."""
        if isinstance(node, str):
            return [node] if len(node) >= 40 and depth <= 1 else []
        if depth > 4:
            return []
        if isinstance(node, list):
            return [t for v in node for t in (self.record_texts(v, depth + 1) if not isinstance(v, str) else [v] if len(v) >= 40 else [])]
        if isinstance(node, dict):
            texts = [v for v in node.values() if isinstance(v, str)]
            return ([texts[0]] if len(texts) == 1 and len(texts[0]) >= 40 else []) + [t for v in node.values() if isinstance(v, (dict, list)) for t in self.record_texts(v, depth + 1)]
        return []

    def records_in(self, text: str, standing: bool, layout: dict[str, Any]) -> Any:
        """The records a text holds under the store's layout (one record, or a list when it holds several); else the text."""
        if layout.get("kind") == "free":
            if not standing:
                return text
            if text not in self.extracted:
                got = self.ask_json(EXTRACT + "\n\nTEXT:\n" + text[:6000], [self.models["reader"], self.models["actor"]], 3000, role="extract") or {}
                self.extracted[text] = [r for r in _items(got.get("records")) if isinstance(r, dict)]
            found = self.extracted[text]
            return (found[0] if len(found) == 1 else found) if found else text
        field, pair = layout.get("field_separator"), layout.get("pair_separator")
        tabled = next((l for l in [layout, *self.layouts.values()] if l.get("table_field")), {})   # a tool's own sample may have shown no attribute table
        if layout.get("kind") != "separators" or not field or not pair or field not in text:
            return text
        split = layout.get("record_separator")
        parts = text.split(split) if split and split in text else [text]
        records = []
        for part in parts:
            pairs = _pairs(part, field, pair)
            if len(pairs) < 3:
                return text   # text that does not follow the layout is a message
            record: dict[str, Any] = {k: _typed(v) for k, v in pairs.items()}
            table, entry, inner = tabled.get("table_field"), tabled.get("table_entry_separator"), tabled.get("table_pair_separator")
            if table and entry and inner and isinstance(record.get(table), str):
                record[table] = _pairs(record[table], entry, inner) or record[table]
            records.append(record)
        return records[0] if len(records) == 1 else records

    def stock_means(self, text: str) -> bool | None:
        """What a store's wording of availability means, read by a model once per wording."""
        if text not in self.stock_words:
            got = self.ask_json(STOCK + "\n\nTEXT: " + text, [self.models["reader"], self.models["actor"]], 600, role="stock", spec=("obj", {"buyable": ("opt", "bool")}), strict=True) or {}
            self.stock_words[text] = got.get("buyable")
        return self.stock_words[text]

    def facts_of(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """A record's attributes: the field a model named for them, its values as plain text."""
        table = record.get(self.keys["attributes"] or "")
        if not isinstance(table, dict) or not table:   # the field named for the attributes is no table on this record: the record's own table, when it holds one
            tables = [v for v in record.values() if isinstance(v, dict) and v]
            table = max(tables, key=len) if tables else None
        if not table:
            return None
        return {k: (str(v).lower() if isinstance(v, bool) else ", ".join(map(str, v)) if isinstance(v, list) else v) for k, v in table.items() if isinstance(v, (str, int, float, list))}

    def read_fields(self, listing: dict[str, Any], record: dict[str, Any]) -> None:
        for field in ("title", "category", "text"):
            key = self.keys[field]
            if key and record.get(key) not in (None, ""):
                listing[field] = str(record[key])
        if self.keys["price"] and _number(record.get(self.keys["price"])) is not None:
            listing["price"] = _number(record[self.keys["price"]])
        stock = record.get(self.keys["stock"] or "")
        if isinstance(stock, bool) or stock is None:
            listing["stock"] = stock if stock is not None else listing.get("stock")
        elif isinstance(stock, (int, float)):
            listing["stock"] = stock > 0
        else:
            listing["stock"] = self.stock_means(str(stock))

    def upsert(self, record: dict[str, Any], page: bool) -> str | None:
        ident = record.get(self.keys["id"]) if self.keys["id"] else None
        if not self.is_id(ident):
            return None
        listing = self.listings.get(ident)
        if listing is None:
            listing = {"id": ident, "title": "", "price": None, "stock": None, "category": "", "facts": None, "text": "", "opened": False, "row": {}}
            self.listings[ident] = listing
            self.seen_order.append(ident)
        listing["raw"] = {**listing.get("raw", {}), **{k: v for k, v in record.items() if not isinstance(v, (dict, list))}}
        listing["seen"] = {**listing.get("seen", {}), **{k: self.step for k in record}}
        self.read_fields(listing, record)
        facts = self.facts_of(record)
        if facts:
            listing["facts"] = facts
            listing["opened"] = True
        if page:
            listing["opened"] = True
            listing["page"] = _clip({k: v for k, v in record.items() if not isinstance(v, dict)}, 300)
        if page or facts:
            self.read_step[ident] = self.step
        else:
            listing["row"] = _clip(record, 200)
        return ident

    def listing_parts(self, node: Any, depth: int = 0) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
        """(lists of listing rows, single listing records) found in a result by their structure."""
        rows: list[list[dict[str, Any]]] = []
        records: list[dict[str, Any]] = []
        if depth > 3:
            return rows, records
        if isinstance(node, dict):
            if self.keys["id"] and self.is_id(node.get(self.keys["id"])):
                records.append(node)
                return rows, records
            for value in node.values():
                if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                    rows.append(value)
                else:
                    more = self.listing_parts(value, depth + 1)
                    rows += more[0]
                    records += more[1]
        return rows, records

    def absorb(self, name: str, observation: Any) -> list[str]:
        """Read listings out of a result and learn whether this tool shows attribute tables. Returns the ids it showed."""
        if not isinstance(observation, dict):
            return []
        rows, records = self.listing_parts(observation)
        for batch in rows + [records]:
            self.learn_keys(batch)
        if not rows and self.keys["id"] is None:   # a single record before any list: the largest object in the result
            whole = max([observation] + [v for v in observation.values() if isinstance(v, dict)], key=len)
            self.learn_keys([whole] if len(whole) >= 3 else [])
            rows, records = self.listing_parts(observation)
        shown: list[str] = []
        page_tools = ({self.roles.get("open"), self.roles.get("offer"), self.roles.get("compare")} | set(self.single_tools())) - {None}
        for batch in rows:
            for row in batch:
                ident = self.upsert(row, name in page_tools and bool(self.facts_of(row)))
                if ident:
                    shown.append(ident)
        for record in records:
            ident = self.upsert(record, name in (self.roles.get("open"), self.roles.get("compare")) or (name in page_tools and bool(self.facts_of(record))))
            if ident:
                shown.append(ident)
                if name in page_tools:
                    self.read_step[ident] = self.step      # price and stock are fresh from any page view
        return shown

    def is_id(self, value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def single_tools(self) -> list[str]:
        """Tools that take one listing id and are not a job of their own (cart, order, messaging)."""
        busy = {self.roles.get(r) for r in ("search", "cart_add", "cart_remove", "cart_view", "order", "decline", "message")} - {None}
        return [t for t in self.schemas if t not in busy and self.id_param(t)]

    def loose_lines(self, name: str, observation: Any) -> list[str]:
        """Messages riding on a result: text outside every listing record (a top-level string, a list of strings, or one inside
        an object that is not a listing), not read before, other than the store's next-page value."""
        lines: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 3 or (isinstance(node, dict) and self.keys["id"] and self.is_id(node.get(self.keys["id"]))):
                return
            if isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                if all(isinstance(v, str) for v in node):
                    lines.extend(node)
                else:
                    for value in node:
                        if not isinstance(value, dict):
                            walk(value, depth + 1)
            elif isinstance(node, str):
                lines.append(node)

        walk(observation, 0)
        page_value = str(self.searches[-1][4]) if self.searches else ""
        fresh = [l.strip() for l in dict.fromkeys(lines) if l.strip() and l.strip() not in self.lines_seen and l != page_value and l not in self.listings]
        self.lines_seen.update(fresh)
        return fresh

    # -- items and labels ----------------------------------------------------------

    def list_limit(self, tool: str | None) -> int | None:
        """How many listings one call of a several-listing view takes: the schema's maximum, else what the store has
        shown it accepts and turns down."""
        target = self.list_param(tool)
        spec = ((self.schemas.get(tool or "") or {}).get("properties") or {}).get(target or "") or {}
        if isinstance(spec.get("maxItems"), int) and spec["maxItems"] > 0:
            return spec["maxItems"]
        if tool in self.list_failed:
            return max(1, min(self.list_failed[tool] - 1, max(self.list_ok.get(tool, 1), self.list_failed[tool] - 1)))
        return None

    def list_param(self, tool: str | None) -> str | None:
        """The parameter of a tool that takes several listings at once, as the brief read it."""
        props = (self.schemas.get(tool or "") or {}).get("properties") or {}
        name = (self.brief.get("value_args") or {}).get(tool or "")
        return name if name in props and "array" in _kinds(props[name]) else None

    def read_listings(self, ids: list[str], keep: int, offer: bool = False) -> list[str]:
        """Read listings' pages, keeping `keep` steps: several in one step with a side-by-side view that shows
        attributes, else one by one with the attributes view (or the offer view for price and stock)."""
        role = "offer" if offer and self.roles.get("offer") else "open"
        ids = [i for i in dict.fromkeys(ids) if i in self.listings]
        done: list[str] = []
        compare = self.roles.get("compare")
        target = self.list_param(compare)
        if ids and not offer and compare and target and len(ids) >= 2:
            cap = self.list_limit(compare) or len(ids)
            while len(ids) >= 2 and self.steps_left() > keep and self.time_left() > WRAP_SECONDS and not self.finished:
                batch, rest = ids[:cap], ids[cap:]
                if self.call_role("compare", batch, target=target) is None:
                    break
                got = [i for i in batch if self.read_step.get(i) == self.step]
                done += got
                ids = [i for i in batch if i not in got] + rest
                if not got:
                    break
        for ident in ids:
            if self.steps_left() <= keep or self.time_left() < WRAP_SECONDS or self.finished or self.call_role(role, ident) is None:
                break
            done.append(ident)
        return done

    # -- requirements taken from another listing -------------------------------------

    def add_ref(self, number: int | None, need: str) -> None:
        if not need:
            return
        if number is None:
            number = next((n for n, it in enumerate(self.items, 1) if need in it.needs()), 0)
            if not number:
                key = _plain(need.split(":")[0])
                hit = [(n, m) for n, it in enumerate(self.items, 1) for m in it.needs() if _plain(m.split(":")[0]) == key]
                if not hit:
                    return
                number, need = hit[0]
        if not any(r["need"] == need and r["item"] == number for r in self.refs):
            self.refs.append({"item": number, "need": need, "value": "", "ids": [], "look_for": "", "key": ""})

    def pending_refs(self) -> list[dict[str, Any]]:
        return [r for r in self.refs if not r["value"] and r["need"] in self.items[r["item"] - 1].needs()]

    def resolve_refs(self) -> None:
        """Follow each unresolved reference through the records; open the links the records name; a chain that ends on
        a read page gives the requirement its value."""
        for ref in self.pending_refs():
            while True:   # each round reads what the chain names; it stops when a round adds nothing, or the free steps run out
                opened = _dump([len(self.searches)] + [(i, self.listings[i]["opened"]) for i in self.seen_order])   # a search run is a round too
                if ref["key"] == opened or self.time_left() < WRAP_SECONDS:
                    break
                ref["key"] = opened
                # the records the chain can be about: listings one of whose own field values (a seller, a maker) the requirement or
                # a page along the chain names, the links so far, else the newest; the store's strings, compared as written
                chain = " ".join([ref["need"]] + [self.record_of(i) for i in ref["ids"] if self.listings[i]["opened"]]).lower()
                mapped = {self.keys[k] for k in ("title", "category", "text", "id") if self.keys[k]}
                named = lambda i: any(isinstance(v, str) and len(v) > 3 and k not in mapped and re.search(r"(?<!\w)" + re.escape(v.lower()) + r"(?!\w)", chain)
                                      for k, v in (self.listings[i].get("raw") or {}).items())
                pool = [i for i in reversed(self.seen_order) if i in ref["ids"] or named(i)] or list(reversed(self.seen_order))
                records = [("OPENED " + self.record_of(i)) if self.listings[i]["opened"] else ("ROW " + _dump(_clip(self.listings[i].get("row"), 90))[:500]) for i in pool[:30]]
                prompt = TRAIL + "\n\nREQUIREMENT: " + ref["need"] + "\n\nRECORDS (newest first):\n" + "\n".join(records) + "\n\nSEARCHES SO FAR:\n" + (self.search_state() or "(none)")
                got = self.ask_json(prompt, [self.models["reader"]], 2500, role="trail") or {}
                if not got:
                    break
                links = [str(l.get("id") or "") for l in _items(got.get("links")) if isinstance(l, dict)]
                ref["ids"] = [i for i in links if i in self.listings]
                ref["look_for"] = str(got.get("look_for") or "").strip()
                if got.get("done") is True and str(got.get("value") or "").strip() and ref["ids"] and all(self.listings[i]["opened"] for i in ref["ids"]):
                    ref["value"] = str(got["value"]).strip()
                    self.version += 1
                    self.note("ref_resolved", need=ref["need"], value=ref["value"], ids=ref["ids"])
                    break
                todo = [i for i in ref["ids"] if not self.listings[i]["opened"]]
                if todo:
                    self.why = f"chain for {ref['need']!r}: open the links the trail names"
                    if not self.read_listings(todo[:2], self.needed()):
                        self.note("stop", where="chain", why="no free step to open the links", need=ref["need"])
                        break
                    continue
                # the chain stops at a listing not seen yet: search it in the store's words (its kind, seller and values), once,
                # only with a free step beyond the reserve plus one to read what it finds
                query, tool = self.brief.get("query_arg"), self.roles.get("search")
                args = self.role_args("search", ref["look_for"], target=query) if ref["look_for"] and query and tool else None
                mark = (tool or "") + _dump(args or {})
                if not args or mark in self.called or self.steps_left() < self.needed() + 2:
                    self.note("stop", where="chain", why="no look_for" if not args else "searched already" if mark in self.called else "fewer than reserve + 2 steps", need=ref["need"])
                    break
                self.called.add(mark)
                self.why = f"chain for {ref['need']!r}: search {ref['look_for']!r}"
                self.perform(tool, args)

    # -- money: one unit, the store's ----------------------------------------------------

    def set_budget(self, amount: float, unit: str) -> bool:
        """The budget as the shopper states it, compared with the store's prices as plain numbers. Returns True when it changed."""
        unit = " ".join(str(unit or "").lower().split()) or self.budget_unit
        if (amount, unit) == (self.budget, self.budget_unit):
            return False
        self.budget, self.budget_unit = amount, unit
        self.version += 1
        return True

    def budget_text(self) -> str:
        return f"{self.budget:g} {self.budget_unit}" if self.budget is not None else ""

    def item_of(self, ident: str) -> int:
        """The requested item a listing is of (1-based), 0 when none, decided by its category label. Which item a label holds is
        a judgment about kinds of product, so one model call places every new label seen and the code keeps to its answer."""
        label = str((self.listings.get(ident) or {}).get("category") or "").strip().lower()
        if label and label not in self.labels:   # every label seen is placed again with the new ones, so an item can move to a label that names it better
            pinned = {it.shelf.strip().lower() for it in self.items if it.shelf}
            new = sorted({str(x.get("category") or "").strip().lower() for x in self.listings.values()} - pinned - {""})
            items = "\n".join(f"{n}: {it.name}" + (f" (store label: {it.shelf})" if it.shelf else "") for n, it in enumerate(self.items, 1))
            got = self.ask_json(LABELS + "\n\nREQUESTED ITEMS:\n" + items + "\n\nLABELS:\n" + "\n".join(new), [self.models["reader"], self.models["actor"]], 1500, role="labels",
                                spec=("obj", {"labels": ("map", ("obj", {"item": ("opt", "item"), "kind": ("enum", ("names", "narrower", "wider", "other"))}, ("kind",)))}, ("labels",)), ctx={"items": len(self.items)},
                                ground=lambda g: [f"label {x!r} is not answered" for x in new if x not in {k.strip().lower() for k in g.get("labels") or {}}])
            placed = {k.strip().lower(): v for k, v in ((got or {}).get("labels") or {}).items()}
            before = dict(self.labels)
            for n, it in enumerate(self.items, 1):   # the item's label: the one naming it; only when none does, the ranges holding it
                mine = [k for k, v in placed.items() if v.get("item") == n and v["kind"] != "other" and not it.shelf]
                held = [k for k in mine if placed[k]["kind"] == "names"] or mine
                for one in new:
                    if one in held:
                        self.labels[one] = n
            for one in new:
                self.labels.setdefault(one, 0)
            self.version += any(before.get(k, self.labels[k]) != self.labels[k] for k in before if k in self.labels)
            self.note("labels", labels=self.labels)
        return self.labels.get(label, 0) if label else int(len(self.items) == 1)

    def buyable(self, ident: str) -> bool:
        listing = self.listings.get(ident) or {}
        if ident in self.gone or listing.get("stock") is False:
            return False
        price = listing.get("price")
        return not (self.budget is not None and price is not None and self.brief.get("budget_scope") != "each" and price > self.budget)

    def facets(self, number: int) -> tuple[int, dict[str, list[str]]]:
        """An item's attribute names as the store writes them over its opened listings: (listings read, name -> one value per listing carrying it)."""
        opened = [i for i in self.seen_order if self.listings[i]["opened"] and isinstance(self.listings[i].get("facts"), dict) and self.item_of(i) == number][:20]
        values: dict[str, list[str]] = {}
        for ident in opened:
            for name, value in {_plain(k): v for k, v in self.listings[ident]["facts"].items()}.items():
                if name:
                    values.setdefault(name, []).append(str(value).strip().lower()[:100])
        return len(opened), values

    # -- changes from the shopper or the store ------------------------------------

    def item_number(self, value: Any) -> int:
        number = _int(value)
        return number if number and 1 <= number <= len(self.items) else 1

    def take_notes(self, lines: list[str]) -> str:
        """Read lines riding on a result and apply every change they make. Returns a short account, "" when nothing changed."""
        if not lines or self.time_left() < 10:
            return ""
        prompt = (NOTE + "\n\nITEMS AND REQUIREMENTS IN FORCE:\n" + self.ledger(short=True) + "\n\nCART: " + (self.cart_text() or "empty")
                  + "\n\nLINES:\n" + "\n".join(lines)[:3000])
        got = self.ask_json(prompt, [self.models["reader"]], 1500, role="note") or {}
        if got.get("kind") != "change":
            return ("the shopper asks for something before the order: " + " ".join(lines)[:300]) if got.get("wait") is True else ""
        changed = self.apply_changes([(e.get("item"), e["need"].strip(), e.get("points") is True) for e in got.get("add") or []],
                                     [(e.get("item"), e["need"].strip()) for e in got.get("drop") or []], got.get("budget"))
        for entry in _items(got.get("quantity")):
            if isinstance(entry, dict) and _int(entry.get("quantity")):
                item = self.items[self.item_number(entry.get("item")) - 1]
                item.quantity = max(1, _int(entry["quantity"]) or 1)
                changed.append(f"{item.name}: quantity {item.quantity}")
        for ident in _items(got.get("gone")):
            if str(ident) in self.listings:
                self.gone.add(str(ident))
                changed.append(f"{ident} is no longer available")
        if got.get("sold_out") is True and not _items(got.get("gone")):
            changed.append("a chosen listing sold out")
            self.refind_sold()
        if got.get("price_moved") is True:
            changed.append("a price changed")
        if changed:
            self.version += 1
            self.change_step = self.step
        summary = str(got.get("summary") or "").strip()
        text = (summary + " (" + "; ".join(changed) + ")") if changed else summary
        if text:
            self.notes.append(text)
            self.note("change", lines=lines[:4], applied=changed)
        return text

    def apply_changes(self, needs: list[tuple[int, str, bool]], drops: list[tuple[int, str]], budget: Any) -> list[str]:
        """Apply needs (item, "property: constraint", points at another listing), dropped needs and a stated budget;
        returns what changed."""
        changed: list[str] = []
        many = len(self.items) > 1
        for number, need in drops:
            item = self.items[self.item_number(number) - 1]
            hit = [n for n in item.needs() if n == need] or [n for n in item.needs() if _plain(n.split(":")[0]) == _plain(need.split(":")[0])]
            item.dropped += hit
            changed += [f"dropped {n}" for n in hit]
        for number, need, points in needs:
            number = self.item_number(number)
            item = self.items[number - 1]
            item.dropped = [d for d in item.dropped if d != need]   # stated again: in force again
            if need not in item.needs():
                item.told.append(need)
                changed.append(f"item {number}: {need}" if many else need)
            if points:
                self.add_ref(number, need)
        amount = _number(budget.get("amount")) if isinstance(budget, dict) else None
        if amount and amount > 0 and self.set_budget(amount, str(budget.get("unit") or "")):
            changed.append("budget now " + self.budget_text())
        return changed

    def refind_sold(self) -> None:
        """A note said a chosen listing sold out without naming it: re-read the cart's lines. The store's word outranks an
        older reading, so a line stays only when the fresh page shows it in stock."""
        self.change_step = self.step   # stock changed: every earlier reading of it is stale
        for ident in self.cart:
            if ident in self.listings:
                self.listings[ident]["stock"] = None
        self.read_listings(list(self.cart), self.needed(), offer=True)
        for ident in self.cart:
            if self.listings.get(ident, {}).get("stock") is not True:
                self.gone.add(ident)

    def id_param(self, tool: str | None) -> str | None:
        """The parameter of a tool that takes one listing's id (or the message text), as the brief read it."""
        name = (self.brief.get("value_args") or {}).get(tool or "")
        return name if name in ((self.schemas.get(tool or "") or {}).get("properties") or {}) and name != self.list_param(tool) else None

    # -- the cart -----------------------------------------------------------------

    def ids_in(self, value: Any, depth: int = 0) -> list[str]:
        if depth > 4:
            return []
        if isinstance(value, str):
            return [value] if value in self.listings else []
        if isinstance(value, dict):
            return [i for v in value.values() for i in self.ids_in(v, depth + 1)]
        if isinstance(value, list):
            return [i for v in value for i in self.ids_in(v, depth + 1)]
        return []

    def cart_lines(self, args: dict[str, Any]) -> list[tuple[str, int | None]]:
        """(listing id, count) pairs a cart call names: line objects inside the arguments, else the top-level id and count."""
        lines: list[tuple[str, int | None]] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, dict):
                ids = [v for v in node.values() if isinstance(v, str) and v in self.listings]
                counts = [_int(v) for v in node.values() if not isinstance(v, (str, bool)) and _int(v) is not None]
                if len(ids) == 1:
                    lines.append((ids[0], counts[0] if counts else None))
                    return
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)

        walk(args, 0)
        return lines

    def is_removal(self, name: str, args: dict[str, Any]) -> bool:
        add, remove = self.roles.get("cart_add"), self.roles.get("cart_remove")
        fixed_remove = self.fixed.get("cart_remove") or {}
        if name != remove:
            return False
        if add != remove:
            return True
        if fixed_remove and all(args.get(k) == v for k, v in fixed_remove.items()):
            return True
        counts = [_int(v) for v in args.values() if not isinstance(v, (str, bool)) and _int(v) is not None]
        return bool(counts) and counts[0] < 0

    def track_cart(self, name: str, args: dict[str, Any], observation: Any) -> None:
        add, remove = self.roles.get("cart_add"), self.roles.get("cart_remove")
        if name not in (add, remove):
            return
        if self.cart_shape.get("whole"):   # the store's reply shows the cart; a reply without it changed nothing known
            self.cart_version += 1
            self.read_cart_view(observation, dict(self.cart_lines(args)))
            return
        removing = self.is_removal(name, args)
        lines = self.cart_lines(args)
        for ident, count in lines:
            if removing:
                after = self.cart.get(ident, 0) - abs(count if count is not None else self.cart.get(ident, 0))
            elif count is not None and count < 0:
                after = self.cart.get(ident, 0) + count
            else:
                after = count if count is not None else self.cart.get(ident, 0) + 1
            if after <= 0:
                self.cart.pop(ident, None)
            else:
                self.cart[ident] = after
        if lines:
            self.cart_version += 1

    def read_cart_view(self, observation: Any, asked: dict[str, int] | None = None) -> None:
        """The cart as the store shows it: the rows of its one list (an empty list is an empty cart), each line's count under
        the count field, learnt from a reply to a cart call as the one field whose values are the counts just asked for."""
        lists: list[list[Any]] = []

        def walk(node: Any, depth: int) -> None:
            if isinstance(node, dict) and depth < 3:
                for value in node.values():
                    if isinstance(value, list) and all(isinstance(v, dict) for v in value):
                        lists.append(value)
                    else:
                        walk(value, depth + 1)
        walk(observation, 0)
        if len(lists) != 1 or not self.keys["id"]:
            return
        if asked and not self.keys.get("quantity") and any(c != 1 for c in asked.values()):
            fields = [k for k in {k for r in lists[0] for k in r} if k != self.keys["id"] and all(_int(r.get(k)) == asked.get(r.get(self.keys["id"])) for r in lists[0] if self.is_id(r.get(self.keys["id"])))]
            if len(fields) == 1:
                self.keys["quantity"] = fields[0]
        cart: dict[str, int] = {}
        for row in lists[0]:
            ident = row.get(self.keys["id"])
            if self.is_id(ident):
                held = _int(row.get(self.keys["quantity"] or ""))
                cart[ident] = held if held and held > 0 else 1
        self.cart = cart
        self.cart_viewed_version = self.cart_version

    def cart_text(self) -> str:
        parts = []
        total = 0.0
        for ident, quantity in self.cart.items():
            listing = self.listings.get(ident) or {}
            price = listing.get("price")
            if price is not None:
                total += price * quantity
            stock = "SOLD OUT" if not self.buyable(ident) and (ident in self.gone or listing.get("stock") is False) else ""
            number = self.item_of(ident)
            parts.append(f"{ident} x{quantity} ({_dump(listing.get('title'))[:70]}, category {listing.get('category') or '?'}, item {number or 'none'}, price {price if price is not None else '?'}{', ' + stock if stock else ''})")
        if not parts:
            return ""
        return "; ".join(parts) + f"; total {total:g}" + (f" of budget {self.budget:g}" if self.budget is not None else "")

    # -- what the actor sees ----------------------------------------------------------

    def ledger(self, short: bool = False, rules: bool = True) -> str:
        parts = []
        for number, item in enumerate(self.items, 1):
            shown = []
            for need in item.needs():
                ref = next((r for r in self.refs if r["need"] == need and r["item"] == number), None)
                if ref and ref["value"]:
                    need += f" [read from {', '.join(ref['ids'])}: {ref['value']}]"
                elif ref:
                    need += " [value still to be read from that listing]"
                shown.append(need)
            text = f"item {number}: {item.name}" + (f" (store label {item.shelf})" if item.shelf else "") + f", quantity {item.quantity}; requirements: " + ("; ".join(shown) or "none beyond the kind of product")
            if item.free and not short:
                text += "; shopper does not mind: " + ", ".join(item.free)
            parts.append(text)
        if self.budget is not None:
            parts.append("budget: at most " + self.budget_text() + " " + ("for each item" if self.brief.get("budget_scope") == "each" else "in total"))
        if short:
            return "\n".join("- " + p for p in parts)
        parts.append({"cheapest": "pick: the cheapest fitting listing", "most": f"pick: the most {self.brief.get('pick_measure') or ''} among fitting listings",
                      "least": f"pick: the least {self.brief.get('pick_measure') or ''} among fitting listings"}.get(self.pick, "pick: any listing that fits everything"))
        if self.brief.get("in_stock_only", True) is not False:
            parts.append("in stock only")
        for wish in _items(self.brief.get("wishes")):
            parts.append(f"wish (tie-breaker only): {wish}")
        for rule in _items(self.brief.get("rules")) if rules else []:   # how to work, not what a listing must be
            parts.append(f"rule from the request: {rule}")
        for step in self.process if rules else []:   # the harness's own business, like the rules: not what a listing must be
            parts.append(f"step the request asks for ({ {'choose': 'before choosing', 'order': 'at the order: the harness does it', 'end': 'at an ending without a purchase: the harness does it'}[step['when']] }): {step['step']}")
        return "\n".join("- " + p for p in parts)

    def search_state(self) -> str:
        """Each listing search so far: its tool and arguments, what the store counts, the rows seen, the store's next-page value."""
        seen: dict[str, list[Any]] = {}
        first: dict[str, str] = {}   # rows shown -> the first call that showed them
        for tool, args, shown, count, nxt in self.searches:
            key = tool + " " + _dump(args)
            same = first.setdefault(_dump(shown), key) if shown else key
            seen[key] = [count, len(shown), nxt, f" (the same rows as {same}: the arguments it adds had no effect)" if same != key else ""]
        return "\n".join(f"{k}: matched {c}, {r} rows shown by this call, next page value {_dump(n)}{x}" for k, (c, r, n, x) in list(seen.items())[-25:])

    def status(self) -> str:
        left = self.steps_left()
        text = f"[status] step {self.step} of {self.max_steps} used. " + self.steps_text()
        if self.roles.get("message"):
            text += f" Messages: {self.msg_used} of {self.msg_limit} used (the harness sends them)."
        text += "\nRequirements in force:\n" + self.ledger()
        cart = self.cart_text()
        text += "\nCart: " + (cart or "empty")
        if self.notes:
            text += "\nChanges just received: " + " | ".join(self.notes)
            self.notes = []
        for ref in self.pending_refs():
            text += (f"\nTo resolve: \"{ref['need']}\" takes its value from another listing. Find that listing (search its kind, seller or words), open it in full, and open every listing its page refers to; the harness reads the value from the last page."
                     + (f" Next to find: {ref['look_for']}." if ref["look_for"] else ""))
        for number, item in enumerate(self.items, 1):
            labels = sorted({self.listings[i]["category"] for i in self.seen_order if self.item_of(i) == number and self.listings[i]["category"]})
            if labels:
                opened = [i for i in self.seen_order if self.item_of(i) == number and self.listings[i]["opened"] and self.buyable(i)]
                unread = sum(1 for i in self.seen_order if self.item_of(i) == number and not self.listings[i]["opened"] and self.buyable(i))
                text += (f"\nItem {number} ({item.name}) is every listing labelled {' / '.join(labels)}, whatever its title says: opened "
                         + (", ".join(f"{i} ({self.listings[i]['price']})" for i in opened[-10:]) or "none") + f"; {unread} buyable rows not opened yet.")
            elif self.listings:
                text += f"\nItem {number} ({item.name}): no listing of its category seen yet."
        state = self.search_state()
        text += ("\nSearches so far:\n" + state) if state else ""
        if left <= self.needed() + 1:
            text += "\nFINAL STEPS: order now when the cart holds a line for each item; end without a purchase only when your searches showed some item has no fitting listing."
        elif left <= 6 or self.time_left() < WRAP_SECONDS + 20:
            text += "\nFew steps or little time remain: call the order tool; the harness selects and orders."
        return text

    # -- questions to the shopper ----------------------------------------------------

    def role_args(self, role: str, value: Any = None, count: int | None = None, target: str | None = None) -> dict[str, Any] | None:
        """The arguments that make a role's tool do its job: its fixed mode, the value and the count in the parameters the brief
        named; None when they do not fit the schema."""
        tool = self.roles.get(role)
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        args: dict[str, Any] = dict(self.fixed.get(role) or {})
        lines_arg = self.cart_shape.get("lines_arg") if role in ("cart_add", "cart_remove") else None
        if lines_arg in props and value is not None:   # a list of lines: value is one id, or {id: count} for several
            wanted = value if isinstance(value, dict) else {value: count if count is not None else 1}
            id_field, count_field = self.cart_shape.get("id_field"), self.cart_shape.get("count_field")
            args[lines_arg] = [({id_field: i, **({count_field: n} if count_field else {})} if id_field else i) for i, n in wanted.items()]
            value, count = None, None
        if value is not None:
            target = target or self.id_param(tool)
            if not target:
                return None
            args[target] = value
        count_arg = self.cart_shape.get("count_field")
        if count is not None and count_arg in props:
            args[count_arg] = count
        for key in schema.get("required") or []:
            if key not in args and role == "decline" and "string" in _kinds(props.get(key)):
                args[key] = "Nothing in the store meets every requirement."
        faults: list[str] = []
        args = _fit_args(args, schema, "arguments", faults)
        return None if faults or not tool else args

    def call_role(self, role: str, value: Any = None, count: int | None = None, target: str | None = None) -> tuple[dict[str, Any], Any] | None:
        """Run a role's tool for the harness itself and read its result (listings, cart, notes). Returns (args, observation)."""
        args = self.role_args(role, value, count, target)
        if args is None or self.steps_left() <= 0 or self.finished:
            return None
        tool = self.roles[role]
        reply = self.act(tool, args)
        if reply is None:
            return None
        observation = reply.get("observation")
        shown = self.absorb(tool, observation)
        self.learn_list_limit(tool, args, shown)
        self.track_cart(tool, args, observation)
        if tool == self.roles.get("cart_view"):
            self.read_cart_view(observation)
        self.queue_notes(tool, observation)
        return args, observation

    def say(self, text: str) -> str | None:
        """Send one message to the shopper; returns their reply text, or None when it could not be sent."""
        if self.msg_used >= self.msg_limit or self.finished:
            return None
        args = self.role_args("message", text)
        if not args:
            return None
        reply = self.act(self.roles["message"], args)
        if reply is None:
            return None
        self.msg_used += 1
        observation = reply.get("observation")
        words = " ".join(_strings(observation))[:600]
        self.note("message", text=text, reply=words)
        return words

    def attributes(self, number: int) -> str:
        """An item's attribute names as the store writes them, most carried first, with how many read listings carry each and
        some of their values."""
        total, values = self.facets(number)
        lines = [f"  {k}: on {len(v)} of {total}; values: {'; '.join(dict.fromkeys(v))[:90]}" for k, v in sorted(values.items(), key=lambda x: -len(x[1]))]
        return "\n".join(lines) or "  (no listing of this item read yet)"

    def plan_picks(self, count: int, plan: dict[str, Any]) -> list[tuple[int, str]]:
        """The questions in order: the planner's, then, while the messages can carry more, the attributes most of an item's
        listings carry, taking turns between the items."""
        picks: list[tuple[int, str]] = []

        def add(number: int, label: str) -> None:
            label = _plain(label)
            settled = {_plain(n.split(":")[0]) for n in self.items[number - 1].needs()}
            values = self.facets(number)[1]
            if label and (label in values or not values) and (number, label) not in picks and label not in settled:
                picks.append((number, label))

        # what the request asks to check and what its story hints at come first when the store records it under that name
        # (the brief found the hint; the planner left "gender" out although two opened watches carried the key)
        for number, prop in [(1, str(x)) for x in _items(self.brief.get("ask_first"))] + [(self.item_number(h.get("item")), str(h.get("property") or "")) for h in _items(self.brief.get("hints")) if isinstance(h, dict)]:
            words = set(_plain(prop).split())
            keys = sorted((k for k in self.facets(number)[1] if words and words <= set(k.split())), key=lambda k: (k != _plain(prop), -len(self.facets(number)[1][k])))
            if keys:
                add(number, keys[0])
        # the planner labels every name's kind; the code keeps the properties, most carried first (the planner's own order breaks
        # ties), and keeps the other kinds out of the filler too: two runs on the same listings had ranked the same names
        # differently, and names the plan was told to skip ("condition", "model") had taken slots from real ones
        kinds = {(self.item_number(e.get("item")), _plain(str(e.get("label") or ""))): str(e.get("kind") or "property") for e in _items(plan.get("ask")) if isinstance(e, dict)}
        ranked = {n: sorted((k for k in self.facets(n)[1] if kinds.get((n, k), "property") == "property"),
                            key=lambda k, n=n: (-len(self.facets(n)[1][k]), list(kinds).index((n, k)) if (n, k) in kinds else len(kinds))) for n in range(1, len(self.items) + 1)}
        for number, label in kinds:   # a planner name the listings do not carry yet (nothing opened) still counts when nothing else is known
            if kinds[(number, label)] == "property" and not self.facets(number)[1]:
                add(number, label)
        depth = 0
        while len(picks) < count and any(depth < len(v) for v in ranked.values()):
            for number, keys in ranked.items():
                if depth < len(keys):
                    add(number, keys[depth])
            depth += 1
        return picks


    def unread(self, number: int) -> list[str]:
        """An item's buyable listings seen as rows and not opened yet."""
        return [i for i in self.seen_order if self.item_of(i) == number and not self.listings[i]["opened"] and self.buyable(i)]

    def plan_questions(self, count: int, force: bool) -> list[tuple[int, str]] | None:
        """A model chooses which of the store's attribute names to ask about, reading more listings first when it wants; None when
        it wants more than the shelf shows and the questions can wait for the actor to find them."""
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        hints = [h for h in _items(self.brief.get("hints")) if isinstance(h, dict) and str(h.get("property") or "").strip()]

        def ground(got: dict[str, Any]) -> list[str]:
            return [f"{e['label']!r} is not an attribute of item {e.get('item') or 1}: copy a name from its list" for e in got.get("ask") or []
                    if self.facets(self.item_number(e.get("item")))[1] and _plain(e["label"]) not in self.facets(self.item_number(e.get("item")))[1]]
        added, budget = "", max(2, (self.steps_left() - self.needed()) // 4)   # reads for the plan take at most a quarter of the free steps (practice: 15 single reads before asking left no steps for the cheapest pick)
        for round_ in range(3):   # a plan, and at most two more after its reads: each plan costs a model call's time (race-18 replay: 7 plans took 82 s of a 272 s episode)
            known = {(n, k) for n in range(1, len(self.items) + 1) for k in self.facets(n)[1]}
            lists = [f"item {n}: {item.name}\n" + self.attributes(n) for n, item in enumerate(self.items, 1)]
            prompt = (PLAN.replace("COUNT", str(3 * count + 3)) + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
                      + "\n\n" + self.steps_text() + "\n\nATTRIBUTES:\n" + "\n\n".join(lists))
            prompt += ("\n\nASK FIRST:\n" + "\n".join(first)) if first else ""
            prompt += ("\n\nSTORY HINTS:\n" + "\n".join(f"item {h.get('item') or 1}: {h['property']}" for h in hints)) if hints else ""
            prompt += added
            plan = self.ask_json(prompt, [self.models["heavy"], self.models["actor"]] if round_ == 0 and not self.waited else [self.models["actor"], self.models["reader"]], 4000, role="plan", ground=ground) or {}   # the first plan sets the order: Sonnet; re-plans after reads only add names: Haiku
            read, self.short = [], {}
            for e in plan.get("read") or []:
                number = self.item_number(e.get("item"))
                rows = self.unread(number)
                if len(rows) < e["count"] and not force and not self.waited:   # the shelf shows fewer: the actor searches once, and the plan waits (once an episode: each wait costs a plan call and an actor turn)
                    self.short[number] = sum(self.item_of(i) == number for i in self.seen_order)
                    self.waited.add(number)
                    continue
                self.why = f"question planner: read {e['count']} more listings of item {number} for their attribute names"
                read += self.read_listings(rows[:max(0, min(e["count"], budget - len(read)))], self.needed())
            if self.short:
                self.note("plan_wait", short=self.short)
                return None
            fresh = sorted(k for n in range(1, len(self.items) + 1) for k in self.facets(n)[1] if (n, k) not in known)
            if not read or not fresh and known or round_ == 2 or len(read) >= budget:   # nothing read, or reads that added no name: the names are known, ask from this plan
                return self.plan_picks(3 * count + 3, plan)
            added = f"\n\nYOUR LAST READS: {len(read)} listings added these new names: {', '.join(fresh) or 'none'}"

    def interview(self, force: bool) -> str | None:
        """Plan the properties to ask about from general knowledge of each product and the table of its candidates, then ask
        them one message at a time, each worded by a model for its property and read before the next is chosen. None when the
        plan waits for listings the actor has yet to find."""
        self.process_due("choose", "choose")
        count = self.msgs_free()
        picks = self.plan_questions(count, force) if count > 0 else []
        if picks is None:
            return None
        self.asked = True
        self.note("plan", picks=picks)
        # a conversation: each turn one model call reads the shopper's latest reply and writes the next message in answer to
        # it, from the planned properties still open; it ends when the model sends nothing or the messages run out
        sent: list[tuple[int, str, str, str, list[tuple[int, str]]]] = []
        heard: dict[str, Any] = {"told": [], "settled": set(), "search": []}
        while not self.finished:
            got = self.talk(sent, picks, heard["settled"])
            heard["told"] += got["told"]
            heard["settled"] |= got["settled"]
            heard["search"] += got["search"]
            nxt = got["next"]
            if not nxt or self.msgs_free() <= 0 or self.steps_left() <= self.needed(questions=False):
                self.note("stop", where="questions", why="no names left to ask" if not nxt else "no messages left" if self.msgs_free() <= 0 else "no step above the reserve",
                          unasked=[(n, l) for n, l in picks if (n, l) not in heard["settled"]][:12])
                break
            self.why = "question about " + ", ".join(f"item {n}: {x}" for n, x in nxt["pairs"])
            reply = self.say(nxt["message"])
            if reply is None:
                break
            sent.append((nxt["item"], nxt["property"], nxt["message"], reply, nxt["pairs"]))
        told = heard["told"]
        found = self.value_search(heard["search"])
        self.qa += [(n, q, a) for n, _, q, a, _ in sent]
        if told:
            self.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, _, q, a, _ in sent]
        return ("The harness asked the shopper (each message used one step):\n" + "\n".join(lines)
                + "\nNew requirements from the answers: " + ("; ".join(told) or "none")
                + ". Apply them to every pick; search again with the new values where needed." + found) if sent else ""

    def value_search(self, phrases: list[tuple[int, str]], what: str = "new") -> str:
        """Search the store with each phrase given for a stated value (the request's at the start, the shopper's answers after
        the interview). Reading is left to the stage that knows what it reads for: the question planner before the
        questions (attribute names), the selection after them (a fit)."""
        query, tool = self.brief.get("query_arg"), self.roles.get("search")
        lines: list[str] = []
        for _, phrase in dict.fromkeys(phrases):
            args = self.role_args("search", phrase, target=query) if query and tool else None
            if not args or self.steps_left() <= self.needed():
                self.note("stop", where="value_search", why=f"no free step above the reserve of {self.needed()}" if args else "no search arguments", left=[p for _, p in phrases])
                break
            self.why = f"{what} value search: {phrase}"
            self.perform(tool, args)
            lines.append(f"{phrase!r}: {len(self.searches[-1][2])} rows")
        return (f"\nThe harness searched with the {what} values: " + "; ".join(lines) + ".") if lines else ""

    def talk(self, sent: list[tuple[int, str, str, str, list[tuple[int, str]]]], picks: list[tuple[int, str]], settled: set[tuple[int, str]]) -> dict[str, Any]:
        """One turn of the conversation: a model reads what the shopper's latest reply means (applied here), then another writes
        the next message in answer to it, or None."""
        out: dict[str, Any] = {"told": [], "settled": set(), "next": None, "search": []}
        if sent:
            number, asked, question, reply, pairs = sent[-1]
            said = set(re.findall(r"[a-z0-9]+", reply.lower()))
            names = lambda e: [w for w in re.findall(r"[a-z0-9]+", _plain(str(e.get("need") or "").split(":")[0])) if len(w) > 2]
            got = self.ask_json(READ + "\n\nITEMS AND REQUIREMENTS IN FORCE:\n" + self.ledger(short=True) + f"\n\nAGENT'S MESSAGE (about {'; '.join(f'item {n}: {x}' for n, x in pairs)}): {question}\n\nSHOPPER'S REPLY: {reply}",
                                [self.models["reader"], self.models["actor"]], 3000, role="read",
                                ground=lambda g: [f"drop {e.get('need')!r}: the reply never names that property, so it gives nothing up" for e in g.get("drop") or [] if names(e) and not set(names(e)) & said]) or {}
            got["drop"] = [e for e in got.get("drop") or [] if not names(e) or set(names(e)) & said]
            meaning = str(got.get("meaning") or "unclear")
            self.meanings.append(meaning + (f": {got['wants']}" if got.get("wants") else ""))
            needs = [(e.get("item") or number, f"{e['property'].strip()}: {e['constraint'].strip()}", e.get("points") is True) for e in got.get("needs") or []]
            done = {(n, _plain(need.split(":")[0])) for n, need, _ in needs}
            for n, prop in list(done):   # the reader's wording of the property may differ from the name asked ("type" for "binder type"): the asked name of that item sharing most words with it is the one answered
                asked_ = [(m, _plain(x)) for m, x in pairs if m == n]
                if asked_ and (n, prop) not in asked_:
                    done.add(max(asked_, key=lambda a: (len(set(a[1].split()) & set(prop.split())), -asked_.index(a))))
            if meaning in ("no_preference", "not_understood") or not needs:
                # the shopper answers for the first of their properties a message names, so a reply stating nothing settles only the
                # first name: the others may have been masked by it (practice: a hidden "material" asked after a no-preference "hoodie
                # style" got a shrug), yet asking them again rarely pays (race 18: 13 of 143), so they wait behind the names not asked yet
                done |= {(n, _plain(x)) for n, x in pairs[:1]}
                self.deferred |= {(n, _plain(x)) for n, x in pairs[1:]}
            if meaning == "no_preference":
                for m, x in pairs[:1]:
                    self.items[m - 1].free += [l for n, l in picks if n == m and _plain(l) == _plain(x)]
            if got.get("crowded") is True:
                self.lists = False
            out["settled"] = done | {(n, l) for n, l in picks if (n, _plain(l)) in done}
            out["told"] = self.apply_changes(needs, [(e.get("item"), e["need"].strip()) for e in got.get("drop") or []], got.get("budget"))
            out["search"] = [(number, str(got["search"]).strip())] if str(got.get("search") or "").strip() and out["told"] else []
            self.note("read", reply=reply, got=got)
        settled = settled | out["settled"]
        open_ = [(n, l) for n, l in picks if (n, l) not in settled]
        # the items take turns, starting after the one asked last; items whose need the shopper already revealed wait
        last, many = (sent[-1][0] if sent else 0), len(self.items)
        lanes = {n: sorted([x for x in open_ if x[0] == n], key=lambda x: (n, _plain(x[1])) in self.deferred) for n in range(1, many + 1)}
        turn = sorted(lanes, key=lambda n: (any(k[0] == n for k in settled), (n - last - 1) % many))
        open_ = [lanes[n][k] for k in range(len(open_)) for n in turn if k < len(lanes[n])]
        if self.msgs_free() <= 0 or self.steps_left() <= self.needed(questions=False) or not open_:
            return out
        # how many names this message carries, all of one item: three in the first two messages, then every name left (up to 12);
        # one when a reply asked for fewer at once. (Race 18: a preference was found 21-26% of the time with up to three names asked
        # in all, 55-69% with five or six; a message naming two items was answered half as often as one about one item)
        size = (3 if len(sent) < 2 else 12) if self.lists else 1
        busy = [n for n in turn if lanes[n]]
        if len(busy) > self.msgs_free() and len(busy) > 1 and self.lists:   # more items than messages: two items share a message (race 18: the top agents' 3-item hits came this way)
            chosen = lanes[busy[0]][:2] + [x for x in lanes[busy[1]] if _plain(x[1]) not in {_plain(l) for _, l in lanes[busy[0]][:2]}][:1]
        else:
            chosen = [x for x in open_ if x[0] == open_[0][0]][:size]
        lines = []
        for n, l in chosen:
            values = [x for x in dict.fromkeys(self.facets(n)[1].get(l, [])) if x][:3]
            lines.append(f"item {n} ({self.items[n - 1].name}): {l}" + (f" (store values: {'; '.join(values)})" if values else ""))
        talked = "\n".join(f"{k}. agent (about item {n}, {l}): {q}\n   shopper: {a}\n   it meant: {m}" for k, ((n, l, q, a, _), m) in enumerate(zip(sent, self.meanings), 1))
        written = self.ask_json(TALK + "\n\nITEMS AND REQUIREMENTS IN FORCE:\n" + self.ledger(short=True) + "\n\nNAMES FOR THIS MESSAGE:\n" + "\n".join(lines)
                                + f"\n\nMESSAGES LEFT AFTER THIS ONE: {max(0, self.msgs_free() - 1)}\n\nCONVERSATION:\n" + (talked or "(nothing sent yet)"),
                                [self.models["actor"], self.models["reader"]], 2500, role="talk",
                                ground=lambda g: [f"the message must use this name word for word: {l!r}" for _, l in chosen if (g.get("next") or {}).get("message") and _plain(l) not in _plain(g["next"]["message"])]) or {}
        nxt = written.get("next")
        pairs = [(n, l) for n, l in chosen if nxt and _plain(l) in _plain(nxt["message"])]   # the names the message carries, in the order chosen
        out["next"] = {"item": pairs[0][0], "property": ", ".join(x for _, x in pairs), "pairs": pairs, "message": nxt["message"]} if nxt and pairs else None
        self.note("talk", got=written)
        return out

    def queue_notes(self, name: str, observation: Any) -> str:
        lines = self.loose_lines(name, observation)
        return self.take_notes(lines) if lines else ""

    def show(self, name: str, observation: Any, shown: list[str]) -> str:
        text = _dump(_clip(observation, 300))
        return text if len(text) <= OBS_CHARS else text[:OBS_CHARS] + f"... (cut; listings in it: {', '.join(shown[:30])})"

    def learn_list_limit(self, name: str, args: dict[str, Any], shown: list[str]) -> None:
        target = self.list_param(name)
        asked = args.get(target) if target else None
        if not isinstance(asked, list) or len(asked) < 2 or name == self.roles.get("search"):
            return
        if any(str(i) in shown for i in asked):
            self.list_ok[name] = max(self.list_ok.get(name, 0), len(asked))
        else:
            self.list_failed[name] = min(self.list_failed.get(name, len(asked)), len(asked))

    # -- the cart's own repairs ------------------------------------------------------------

    def add_line(self, ident: str, quantity: int) -> bool:
        return self.steps_left() > 1 and self.call_role("cart_add", {**self.cart, ident: quantity} if self.cart_shape.get("whole") else ident, quantity) is not None and ident in self.cart

    def dead_lines(self) -> list[str]:
        return [i for i in self.cart if i in self.gone or self.listings.get(i, {}).get("stock") is False]

    def perform(self, name: str, args: dict[str, Any]) -> str:
        reply = self.act(name, args)
        if reply is None:
            return "The episode has ended." if self.finished else "The call failed (network or server). Try again or do something else."
        observation = reply.get("observation")
        shown = self.absorb(name, observation)
        if self.lists_listings(name):
            self.learn_search(name, args, observation, shown)
        self.learn_list_limit(name, args, shown)
        self.track_cart(name, args, observation)
        if name == self.roles.get("cart_view"):
            self.read_cart_view(observation)
        text = self.show(name, observation, shown)
        lines = self.loose_lines(name, observation)
        ending = self.is_order(name, args) or self.is_decline(name, args)
        if ending and not reply.get("finished"):
            self.refusals += 1
            self.change_step = self.step
            lines = list(dict.fromkeys(lines + _strings(observation)))
            if self.cart_shape.get("whole"):
                self.cart = {}   # a refused order can empty the cart; the next whole-cart call states it again at no extra step
        if lines:
            change = self.take_notes(lines)
            if change:
                text += "\n[change read] " + change + ". Apply it before ordering."
        if not self.finished and (self.dead_lines() or (ending and not reply.get("finished"))):
            self.settle_due = True
            text += "\n[harness] the selection runs again before your next turn."
        if reply.get("finished"):
            text += " [episode finished]"
        return text

    def is_order(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("order"):
            return False
        if self.roles.get("decline") != name:
            return True
        fixed = self.fixed.get("order") or {}
        return bool(fixed) and all(args.get(k) == v for k, v in fixed.items())

    def is_decline(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("decline"):
            return False
        if self.roles.get("order") != name:
            return True
        fixed = self.fixed.get("decline") or {}
        return bool(fixed) and all(args.get(k) == v for k, v in fixed.items())

    def store_terms(self) -> None:
        """Once per requirement, and again when the store shows new values for it: a model says which of the store's attributes
        records it and which of their values meet it, from the values the read listings carry; code then holds every pick to that."""
        asks, lists = [], {}
        for n, item in enumerate(self.items, 1):
            values = {k: list(dict.fromkeys(v)) for k, v in self.facets(n)[1].items()}
            for m in item.needs():
                t = self.terms.get((n, m))
                if values and not any(r["need"] == m and r["item"] == n for r in self.refs) and (
                        t is None or set(values) - t["seen"] or {v for k in t["keys"] for v in values.get(k, [])} - set(t["meets"]) - set(t["fails"])):
                    asks.append((n, m)); lists[n] = values
        if not asks:
            return
        prompt = TERMS + "\n\nREQUEST:\n" + self.request + "\n\n" + "\n\n".join(f"item {n}: {self.items[n - 1].name}\nATTRIBUTES:\n" + "\n".join(f"  {k}: {'; '.join(v[:15])}" for k, v in lists[n].items()) for n in sorted(lists)) \
            + "\n\nREQUIREMENTS:\n" + "\n".join(f"{k}. item {n}: {m}" for k, (n, m) in enumerate(asks, 1))
        exact = lambda n, m: [k for k in lists[n] if k == _plain(m.split(":")[0])]   # an attribute spelled exactly like the property records it
        ground = lambda got: [f"requirement {e['number']}: {k!r} is not an attribute of its item" for e in got.get("terms") or []
                              if 1 <= e["number"] <= len(asks) for k in e["keys"] if _plain(k) not in lists[asks[e["number"] - 1][0]]]
        got = self.ask_json(prompt, [self.models["reader"], self.models["actor"]], 2500, role="terms", ground=ground) or {}
        for e in got.get("terms") or []:
            if 1 <= e["number"] <= len(asks):
                n, m = asks[e["number"] - 1]
                low = lambda xs: [str(x).strip().lower()[:100] for x in xs]
                self.terms[(n, m)] = {"keys": list(dict.fromkeys([_plain(k) for k in e["keys"] if _plain(k) in lists[n]] + exact(n, m))), "meets": low(e.get("meets") or []),
                                      "fails": [x for x in low(e.get("fails") or []) if x not in low(e.get("meets") or [])], "seen": set(lists[n])}
        self.note("terms", terms={f"{n}|{m}": {k: v for k, v in t.items() if k != "seen"} for (n, m), t in self.terms.items()})

    def term_check(self, ident: str) -> tuple[list[str], bool]:
        """A listing against the store's terms: the requirements its own values fail, and whether its values meet them all."""
        n, facts = self.item_of(ident), self.listings[ident].get("facts") or {}
        needs = self.items[n - 1].needs() if n else []
        faults, met = [], bool(needs)
        for m in needs:
            t = self.terms.get((n, m)) or {}
            vals = [(k, str(v).strip().lower()[:100]) for k, v in facts.items() if _plain(k) in t.get("keys", [])]
            met = met and any(v in t["meets"] for _, v in vals)
            if vals and t["meets"] and all(v in t["fails"] for _, v in vals):   # every attribute recording the property states a value that fails it (terms no value meets decide nothing)
                faults.append(f"{ident} fails {m!r}: " + ", ".join(f"its {k} states {v!r}" for k, v in vals) + ", which in this store's terms does not meet it")
        return faults, met

    def record_fields(self, ident: str, full: bool = True) -> dict[str, Any]:
        listing = self.listings.get(ident) or {}
        number = self.item_of(ident)
        kind = (f"item {number} ({self.items[number - 1].name})" if number else "not a requested item")
        data: dict[str, Any] = {"id": ident, "title": listing.get("title"), "category": listing.get("category"), "kind (by its category label)": kind,
                                "price": listing.get("price"), "in_stock": False if ident in self.gone else listing.get("stock")}
        if full:   # the store's own words, under its own field names: every field of the page (seller, remarks, notes), each key spelling apart
            mapped = {k for k in self.keys.values() if k}
            data.update({k: v for k, v in (listing.get("raw") or {}).items() if k not in mapped and k not in data and v not in (None, "")})
        if full and listing.get("facts") is not None:
            data["attributes"] = {str(k): str(v) for k, v in listing["facts"].items()}
        if full and listing.get("text"):
            data["description"] = listing["text"][:800]
        if full and not listing.get("facts") and listing.get("page"):
            data["page"] = listing["page"]
        return data

    def record_of(self, ident: str, full: bool = True) -> str:
        return _dump(_clip(self.record_fields(ident, full), 400))[:2400]

    # -- finishing checks ------------------------------------------------------------------

    def process_due(self, when: str, key: str) -> str:
        """Carry out the request's steps due at the order or the ending that take a message. Returns a change the reply made, or ""."""
        for step in self.process:
            mark = step["step"] + "|" + key
            if step["when"] != when or step["messages"] <= 0 or mark in self.process_done or self.msg_used >= self.msg_limit:
                continue
            shown = [i for i in self.seen_order if self.listings[i]["opened"] and self.buyable(i)][-6:]
            prompt = (PROCESS + "\n\nREQUEST:\n" + self.request + "\n\nSTEP: " + step["step"] + "\n\nCART: " + (self.cart_text() or ("not chosen yet" if when == "choose" else "empty (ending without a purchase)"))
                      + ("\n\nLISTINGS READ SO FAR (what the shelf offers):\n" + "\n".join(self.record_of(i, full=False) for i in shown) if when == "choose" and shown else ""))
            got = self.ask_json(prompt, [self.models["reader"]], 600, role="process") or {}
            text = str(got.get("message") or "").strip()
            if not text:
                continue
            self.why = f"request step at the {when}: {step['step']}"
            reply = self.say(text)
            if reply is not None:
                self.process_done.add(mark)
            if reply:
                change = self.take_notes([reply])
                if change:
                    return change
        return ""

    def selection_key(self) -> str:
        opened = sorted(i for i in self.listings if self.listings[i]["opened"])
        return f"{self.version}|{_dump([(i, self.listings[i].get('price'), self.listings[i].get('stock')) for i in opened])}|{_dump(sorted(self.gone))}|{len(self.seen_order)}|{_dump(sorted(self.proposed.items()))}|{_dump(sorted(self.cart))}"

    def select(self) -> dict[int, dict[str, Any]]:
        """For each item, the opened listing to buy with its evidence, the unread rows worth reading first, or a search to
        run, judged once per state from the item's opened records and its unread rows."""
        key = self.selection_key()
        if key == self.selected_key:
            return self.selection
        self.selected_key = key
        self.store_terms()
        blocks, pools, rows_of = [], {}, {}
        price = lambda i: self.listings[i]["price"] if self.listings[i]["price"] is not None else float("inf")
        for n, item in enumerate(self.items, 1):
            opened = [i for i in self.seen_order if self.listings[i]["opened"] and self.item_of(i) == n and self.buyable(i)]
            rows = [i for i in self.seen_order if not self.listings[i]["opened"] and self.item_of(i) == n and self.buyable(i)]
            held = {e.get("choice") for e in self.selection.values()} | {b for e in self.selection.values() for b in e.get("backups") or []}   # the choice and its backups stay candidates whatever their age or price
            fits = lambda i: i in held or self.term_check(i)[1]   # a listing the store's terms say meets every requirement is never pushed out by cheaper ones that do not
            opened = sorted(opened, key=lambda i: (not fits(i), price(i)))[:15] if self.pick == "cheapest" else sorted(opened[::-1], key=lambda i: i not in held)[:15]
            rows = sorted(rows, key=price)[:15] if self.pick == "cheapest" else rows[::-1][:15]
            pools[n], rows_of[n] = set(opened), set(rows)
            blocks.append(f"item {n}: {item.name}, quantity {item.quantity}\nOPENED:\n" + ("\n".join(self.record_of(i) for i in opened) or "(none)")
                          + "\nROWS (attributes not read):\n" + ("\n".join(f"{i}: {self.listings[i]['title'][:70]} | price {self.listings[i]['price']}" for i in rows) or "(none)"))

        def ground(got: dict[str, Any]) -> list[str]:
            errors = []
            for e in got.get("items") or []:
                n, choice = e["item"], e.get("choice")
                if choice and choice not in pools.get(n, ()):
                    errors.append(f"item {n}: choice {choice} is not one of its OPENED listings")
                elif choice:   # a requirement is met only by what the chosen record itself says, and never against the store's terms
                    errors += self.term_check(choice)[0]
                    fields = self.record_fields(choice)
                    fields = {_plain(k): v for k, v in [*fields.items(), *(fields.get("attributes") or {}).items()] if k != "attributes"}
                    for v in [v for v in e.get("evidence") or [] if v.get("met")]:
                        held, said = fields.get(_plain(v["source"])), " ".join(re.findall(r"\w+", str(v.get("value") or "").lower()))
                        text = " ".join(re.findall(r"\w+", str(held).lower()))
                        if held is None:
                            errors.append(f"item {n}: {choice}'s record has no field {v['source']!r} (evidence for {v['requirement']!r})")
                        elif said and f" {said} " not in f" {text} " and not (text and f" {text} " in f" {said} "):
                            errors.append(f"item {n}: {choice}'s {v['source']} does not say {v.get('value')!r}; it says {str(held)[:80]!r}")
                for call in e.get("calls") or []:
                    faults: list[str] = []
                    _fit_args(call.get("args") or {}, self.schemas.get(call["tool"]) or {}, f"call {call['tool']}", faults)
                    errors += [f"item {n}: {call['tool']} is not one of the STORE TOOLS"] * (call["tool"] not in listers) + faults
            return errors
        listers = [t for t in self.schemas if self.lists_listings(t)]
        prompt = (SELECT + "\n\nREQUIREMENTS IN FORCE (the shopper's current wishes; nothing else is required):\n" + self.ledger(rules=False) + "\n\n" + "\n\n".join(blocks)
                  + "\n\n" + self.steps_text() + "\n\nSTORE TOOLS:\n" + "\n".join(f"{t}: {_dump(self.schemas[t])}" for t in listers)
                  + (f"\nFILTER SEARCH: {self.roles.get('search')} with {_dump(self.filter['fixed'])} lists one category label ({self.filter['shelf']}) under a price ({self.filter['cap']}), without words; under a cheapest pick the harness runs it under the choice's price, so its rows are the label's cheaper listings (all of them when the count fits the pages shown); do not run it again at or above a cap SEARCHES SO FAR shows, turn its pages with the next page value instead." if self.filter else "")
                  + "\n\nSEARCHES SO FAR:\n" + (self.search_state() or "(none)"))
        terms = [f"item {n}, {m!r}: recorded under {', '.join(repr(k) for k in t['keys'])} (any spelling; one of them stating a meeting value is enough); values meeting it: {'; '.join(t['meets']) or 'none seen yet'}; not meeting it: {'; '.join(t['fails']) or 'none seen yet'}; a value under neither is judged by the reading rules"
                 for (n, m), t in self.terms.items() if m in self.items[n - 1].needs() and t["keys"] and t["meets"]]   # terms no seen value meets are not shown: the rules decide
        meeting = {n: [i for i in self.seen_order if i in pools[n] and self.term_check(i)[1]] for n in pools}
        prompt += ("\n\nSTORE TERMS:\n" + "\n".join(terms) + "".join(f"\nitem {n}: by the store's terms the attributes of these OPENED listings meet every requirement (price and stock still to check): {', '.join(v)}" for n, v in meeting.items() if v)) if terms else ""
        held_lines = {**{self.item_of(i): i for i in self.cart if self.item_of(i) and self.buyable(i)}, **self.proposed}   # a carted line that still fits stays, like the actor's own pick
        prompt += ("\n\nPROPOSED BY THE AGENT:\n" + "\n".join(f"item {n}: {i}" for n, i in sorted(held_lines.items()) if i in pools.get(n, ()))) if held_lines else ""
        got = self.ask_json(prompt, [self.models["actor"], self.models["heavy"]], 6000, role="select", ground=ground, strict=True) or {}   # a selection that failed its checks buys nothing (the room covers the model's reasoning, which counts against it)
        self.selection = {e["item"]: e for e in got.get("items") or []}
        for n, e in self.selection.items():   # a backup is another opened listing of the item, a row to read one of its rows; anything else is dropped
            e["backups"] = [b for b in dict.fromkeys(e.get("backups") or []) if b in pools.get(n, ()) and b != e.get("choice")]
            e["open"] = [r for r in e.get("open") or [] if r in rows_of.get(n, ())]
        self.note("select", selection=self.selection)
        return self.selection

    def settle(self) -> str:
        """The selection stage: read the rows the judgement asks for, then cart the chosen listing of every item and run the
        order procedure; when an item has no fitting opened listing, say what to search for."""
        self.settle_due = False
        start, sel = self.selection_key(), self.select()
        fit = lambda sel: {n: e.get("choice") for n, e in sel.items() if e.get("choice") and not any(v.get("met") is False for v in e.get("evidence") or [])}
        known = lambda chosen: (chosen, [(i, self.listings[i]["opened"]) for i in self.seen_order if self.item_of(i) not in chosen])   # choices, and what is seen or read of items without one
        while self.steps_left() > self.needed() and self.time_left() >= WRAP_SECONDS and not self.finished:
            chosen = fit(sel)
            if len(chosen) == len(self.items) and self.pick == "any":
                self.note("stop", where="settle", why="every item has a fitting choice under pick any", chosen=chosen)
                break   # every item has a fitting listing: the order goes out now, while steps remain for a refusal
            before = known(chosen)
            ran, reads = False, []
            listed = False   # under cheapest with a choice in hand, the harness lists the label under its price; the model's own searches then stop
            if self.pick == "cheapest":
                for n, c in chosen.items():
                    ran = self.list_below(n, c) or ran
                    listed = listed or (n, c) in self.glanced or any(k[0] == n for k in self.capped)
            for e in sel.values():
                for call in [] if listed else e.get("calls") or []:
                    mark = call["tool"] + _dump(call.get("args") or {})
                    named = [v for v in (call.get("args") or {}).values() if isinstance(v, str) and v in self.listings]
                    self.why = f"selection, item {e.get('item')}: {str(e.get('reason') or '')[:200]}"
                    if named:   # a call naming a listing reads it: through the reading tool, once
                        reads += [i for i in named if not self.listings[i]["opened"]]
                    elif mark not in self.called and self.steps_left() > self.needed():
                        self.called.add(mark)
                        self.perform(call["tool"], _fit_args(call.get("args") or {}, self.schemas[call["tool"]], "arguments", []))
                        ran = True
            reads = list(dict.fromkeys(reads + [r for e in sel.values() for r in e.get("open") or [] if not self.listings[r]["opened"]]))
            if self.pick == "cheapest" and chosen:
                # a fitting choice is in hand: the cheaper rows are read cheapest first, three a round and six in all while that choice
                # holds, then the order goes out, keeping the steps a refused order's replacement needs (Cairn checks 3 a line)
                price_of = lambda i: self.listings[i]["price"] if self.listings[i]["price"] is not None else float("inf")
                room = max(0, 6 - max(self.glanced.get((n, c), 0) for n, c in chosen.items()))
                reads = sorted(reads, key=price_of)[:min(3, room)]
                for n, c in chosen.items():
                    self.glanced[(n, c)] = self.glanced.get((n, c), 0) + max(len(reads), 1)   # a round with nothing to read also counts
                if not reads and not ran:
                    self.note("stop", where="settle", why="the cheaper rows allowed while this choice holds were checked; it orders now", chosen=chosen)
                    break
            self.why = "selection asked to open: " + " | ".join(f"item {e.get('item')}: {str(e.get('reason') or '')[:150]}" for e in sel.values() if e.get("open"))
            ran = bool(reads and self.read_listings(reads, self.needed())) or ran
            if not ran:
                self.note("stop", where="settle", why="the selection asked for no new call or read", chosen=chosen)
                break
            sel = self.select()
            if known(fit(sel)) == before and not any(not self.listings[r]["opened"] for e in sel.values() for r in e.get("open") or []):
                self.note("stop", where="settle", why="the round changed no choice, found nothing for an item without one, and the selection wants no row opened", chosen=fit(sel))
                break   # the round changed no choice and neither found nor read anything of an item without one: looking further stops here
        else:
            self.note("stop", where="settle", why=f"steps {self.steps_left()} left, reserve {self.needed()}, time {self.time_left():.0f}s")
        chosen = fit(sel)
        missing = [n for n in range(1, len(self.items) + 1) if n not in chosen]
        if missing:
            return "No opened listing fits " + "; ".join(f"item {n} ({self.items[n - 1].name}): {(sel.get(n) or {}).get('reason') or 'none opened yet'}"
                                                         for n in missing) + \
                ". Search and open more listings of it; the harness selects again when you call the order tool."
        text = self.complete_cart({n: [c] for n, c in chosen.items()}) or "The harness could not set the cart (too few steps, or a cart call failed). Cart now: " + (self.cart_text() or "empty")
        self.settle_due = not self.finished and self.selection_key() not in (start, self.selected_key)
        return text

    def list_below(self, number: int, choice: str) -> bool:
        """Under a cheapest pick, the store's filter search lists the choice's label under a price cap, so the listings cheaper
        than the fitting choice are seen as rows. The cap starts just under the choice's price; while more match than a few
        pages would show, it drops (to the shown price at the rank that leaves about that many), and when that leaves far
        fewer, it comes back up halfway, so the cheapest part of the label is seen whole (the pages are the selection's to
        turn). Each search spends a step above the reserve plus one to read. Returns True when a search ran."""
        tool, f = self.roles.get("search"), self.filter
        price, label = self.listings[choice].get("price"), str(self.listings[choice].get("category") or "")
        if not (tool and f and label and isinstance(price, (int, float))):
            return False
        cap, ran, low, high = float(price), False, None, None   # low: a cap that showed too few; high: one that matched too many
        for _ in range(3):
            if self.steps_left() <= self.needed() + 1 or self.finished:
                break
            cap = cap - 1 if cap.is_integer() else round(cap - 0.01, 2)
            if cap < 0 or (number, label, cap) in self.capped:
                break
            self.capped.add((number, label, cap))
            faults: list[str] = []
            args = _fit_args({**f["fixed"], f["shelf"]: label, f["cap"]: cap}, self.schemas[tool], "arguments", faults)
            if faults:
                break
            self.why = f"cheapest pick: list the label {label!r} under {cap:g}, since the choice {choice} costs {price:g}"
            self.perform(tool, args)
            ran = True
            _, _, shown, count, nxt = self.searches[-1]
            prices = [self.listings[i]["price"] for i in shown if isinstance(self.listings[i].get("price"), (int, float))]
            visible = 3 * max([len(x[2]) for x in self.searches] + [1])   # a few pages at the store's page size
            if not shown or not prices or max(prices) > cap or count is None or nxt in (None, "") and count > visible:
                break   # none cheaper, the cap does not filter, or the count is unknown
            if count > visible:   # more than a few pages: aim at the shown price at the rank leaving about that many
                high, ranked = cap, sorted(prices)
                below = float(ranked[max(0, min(len(ranked) - 1, len(ranked) * visible // count - 1))])
                nxt_cap = below + (1 if below.is_integer() else 0.01) if low is None or below > low else (low + cap) / 2
            elif high is not None and count < visible // 2:   # far fewer than the pages show: back up halfway to the cap that showed too many
                low, nxt_cap = cap, (cap + high) / 2
            else:
                break
            nxt_cap = float(int(nxt_cap)) if cap.is_integer() else round(nxt_cap, 2)
            if nxt_cap >= cap + (1 if cap.is_integer() else 0.01) and high is not None and nxt_cap >= high or not nxt_cap < cap and not (low is not None and nxt_cap > low):
                break
            cap = nxt_cap + (1 if cap.is_integer() else 0.01)   # the loop head takes one off
        return ran

    def lists_listings(self, name: str) -> bool:
        """A tool that lists listings: any tool with no other job in the store."""
        return name not in {self.roles.get(r) for r in ("open", "offer", "compare", "message", "cart_add", "cart_remove", "cart_view", "order", "decline")}

    def learn_search(self, name: str, args: dict[str, Any], observation: Any, shown: list[str]) -> None:
        """Record a listing call: a model names, once per tool, the fields of its result holding the match count and the next-page value."""
        def holder(node: Any, depth: int = 0) -> dict[str, Any]:
            found = [v for v in (node.values() if isinstance(node, dict) else []) if isinstance(v, dict)] if depth < 3 else []
            lists = isinstance(node, dict) and any(isinstance(v, list) for v in node.values())
            return node if lists else next((h for h in (holder(v, depth + 1) for v in found) if h), {})
        top = holder(observation)
        plain = {k: v for k, v in top.items() if not isinstance(v, (list, dict))}
        known = self.result_fields.setdefault(name, {"count": None, "next": None, "tries": 0})
        if plain and not known["next"] and known["tries"] < 3:   # until the next-page field is named: the last page of a search may not show it
            known["tries"] += 1
            got = self.ask_json(RESULT + "\n\nFIELDS:\n" + "\n".join(f"{k} = {_dump(v)}" for k, v in plain.items()), [self.models["reader"], self.models["actor"]], 1500, role="result", spec=("obj", {"count": ("opt", "str"), "next": ("opt", "str")}), strict=True,
                                ground=lambda g: [f"{k!r} is not a field name; the names are {list(plain)}" for k in (g.get("count"), g.get("next")) if k and k not in plain]) or {}
            known.update(count=got.get("count") or known["count"], next=got.get("next"))
        self.searches.append((name, args, shown, _int(top.get(known["count"] or "")), top.get(known["next"] or "")))

    def order_gate(self) -> str | None:
        """The order procedure for the selected cart: look at it, read again lines whose terms changed, test the code's
        facts (stock, quantities, total within the budget), then run the steps the request asks for at the order."""
        if not self.cart:
            return "Not run (no step used): the cart is empty."
        facts = self.cart_faults()
        if self.steps_left() <= 2 or self.time_left() < FORCE_SECONDS + 15:
            return ("Not run (no step used). The cart cannot be ordered as it is: " + "; ".join(facts) + ".\nCart now: " + self.cart_text()) if facts and self.steps_left() > 1 else None
        if self.roles.get("cart_view") and self.cart_viewed_version != self.cart_version and self.steps_left() > 3:
            version = self.version
            if self.call_role("cart_view") is not None and self.version != version:
                return "Not run yet. Looking at the cart showed a change: " + " | ".join(self.notes[-1:]) + "\nCart now: " + (self.cart_text() or "empty") + "\nThe harness selects again."
        self.gather_evidence()
        facts = self.cart_faults()
        if facts and self.steps_left() > self.tail() + 1:
            return "Not run (no step used). The cart cannot be ordered as it is: " + "; ".join(facts) + ".\nCart now: " + self.cart_text() + "\nSearch and open other listings of it; the harness selects again when you call the order tool."
        change = self.process_due("order", _dump(sorted(self.cart.items())))
        if change:
            return "Not run yet. The shopper's reply changed the task: " + change + "\nThe harness selects again."
        return None

    def gather_evidence(self) -> None:
        """Before an order: the cart lines read again when prices, stock or terms changed after they were read; references
        resolved."""
        fresh = [i for i in self.cart if self.read_step.get(i, -1) < self.change_step]
        if fresh:
            self.read_listings(fresh, self.needed(), offer=True)
        if self.pending_refs():
            self.resolve_refs()

    def cart_faults(self) -> list[str]:
        """What the records show for certain is wrong with the cart: dead lines, a line count other than the items', wrong
        quantities, an unknown price, a total (or a line, for a per-item budget) over the budget."""
        faults = []
        dead = [i for i in self.cart if i in self.gone or self.listings.get(i, {}).get("stock") is False]
        if dead:
            faults.append(f"{', '.join(dead)} can no longer be bought: take it out of the cart")
        faults += [f"the cart holds {len(self.cart)} line(s) for {len(self.items)} requested item(s): one line per item"] * (len(self.cart) != len(self.items))
        wanted = {i: self.items[self.item_of(i) - 1].quantity for i in self.cart if self.item_of(i)}
        faults += [f"{i} is in the cart {q} time(s); it is wanted {wanted[i]} time(s)" for i, q in self.cart.items() if i in wanted and q != wanted[i]]
        cost = {i: (self.listings.get(i, {}).get("price") or float("nan")) * q for i, q in self.cart.items()}   # nan: price not known
        each = self.brief.get("budget_scope") == "each"
        if self.budget is not None and any(c != c for c in cost.values()):
            faults.append(f"the price of {', '.join(i for i, c in cost.items() if c != c)} is not known: open it")
        elif self.budget is not None and (each and max(cost.values(), default=0) > self.budget or not each and sum(cost.values()) > self.budget):
            faults.append(f"the {'dearest line, ' + max(cost, key=lambda i: cost[i]) if each else 'total'} at {(max(cost.values()) if each else sum(cost.values())):g} is over the budget {self.budget:g}" + (" for each item" if each else ""))
        return faults

    def remove_line(self, ident: str) -> bool:
        """Take one line out of the cart (a shared add/remove tool without a mode takes a negative count)."""
        if self.steps_left() <= 1:
            return False
        if self.cart_shape.get("whole"):
            return self.call_role("cart_remove", {i: n for i, n in self.cart.items() if i != ident}) is not None
        shared = self.roles.get("cart_add") == self.roles.get("cart_remove") and not self.fixed.get("cart_remove")
        return self.call_role("cart_remove", ident, -self.cart.get(ident, 1) if shared else self.cart.get(ident, 1)) is not None

    def decline_gate(self) -> str | None:
        """An ending without a purchase claims some item has no fitting listing: the selection stage tests the claim once per
        state, and a refuted claim becomes a cart and an order."""
        if self.steps_left() > 4 and self.time_left() > WRAP_SECONDS:
            key = "decline|" + self.selection_key()
            if key not in self.passed:
                self.passed.add(key)
                text = self.settle()
                if self.finished or not text.startswith("No opened listing fits"):
                    return "Not run as an ending. " + text
        change = self.process_due("end", "end")
        return ("Not run yet. The shopper's reply changed the task: " + change) if change else None

    def valid_line(self, number: int) -> str | None:
        return next((i for i in self.cart if self.item_of(i) == number and self.buyable(i)), None)

    def complete_cart(self, named: dict[int, list[str]]) -> str | None:
        """Set the cart to the selected listing of every item at its quantity, then run the order procedure and order."""
        lines: dict[str, int] = {}
        for number, item in enumerate(self.items, 1):
            pick = (named.get(number) or [None])[0] or self.valid_line(number)
            if not pick:
                return None
            lines[pick] = item.quantity
        changes = sum(1 for i in self.cart if i not in lines) + sum(0 if self.cart.get(i) == q else 2 if i in self.cart else 1 for i, q in lines.items())
        whole = self.cart_shape.get("whole")
        if (int(self.cart != lines) if whole else changes) + 1 > self.steps_left():   # each cart change and the order: looking at the cart and the checks skip themselves when steps are short
            return None
        if whole and self.cart != lines and self.call_role("cart_add", lines) is None:
            return None
        for ident in [i for i in self.cart if i not in lines and not whole]:   # one call set the whole cart: no line is changed after it
            if not self.remove_line(ident):
                return None
        for ident, quantity in lines.items() if not whole else []:
            if self.cart.get(ident) != quantity:
                if ident in self.cart and not self.remove_line(ident):
                    return None
                if not self.add_line(ident, quantity):
                    return None
        self.note("cart_set", lines=lines)
        said = "The harness selected " + ", ".join(lines) + " and set the cart"
        hold = self.order_gate()
        if hold:
            return said + ". " + hold
        placed = self.order_now()
        return said + (" and placed the order: " + placed if placed is not None else ".")

    # -- the actor's calls ---------------------------------------------------------------

    def offered(self) -> list[dict[str, Any]]:
        return self.tools

    def ready(self) -> bool:
        """Whether rows of every item the planner wanted more listings of have been seen since it asked."""
        return all(sum(self.item_of(i) == n for i in self.seen_order) > k for n, k in self.short.items())

    def ask_before(self, force: bool = False) -> str:
        """The questions go out before the first commitment (a cart change, an order, an ending), or when the actor asks, once the
        planner has seen the listings it wanted; forced, they go out from what is known."""
        if self.asked or not self.roles.get("message") or self.msgs_free() <= 0 or self.steps_left() <= self.needed(questions=False):
            return ""
        summary = self.interview(force) if force or self.ready() else None
        if summary is None:
            return ("The harness has not sent the questions yet: the question planner needs more listings of " + ", ".join(f"item {n}" for n in self.short)
                    + " to see the attribute names it chooses from. Search for more (another page, or the item's category); the questions go out when new rows of them are seen.")
        self.talked = self.settle_due = True
        return summary or "The harness had nothing to ask the shopper."

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for call in calls:
            cid = call["id"]
            name = call["function"]["name"]
            raw = call["function"]["arguments"]
            if self.finished:
                out.append((cid, "Not run: the episode is over."))
                continue
            if name == self.roles.get("message"):
                summary = self.ask_before()
                out.append((cid, (("The harness planned and sent the questions. " if self.asked else "") + summary) if summary else
                            "Not run (no step used): the shopper has been asked already, or no messages are left. Continue with the store's tools."))
                continue
            if name not in self.schemas:
                out.append((cid, "Not run (no step used): no such tool. Tools: " + ", ".join(t["function"]["name"] for t in self.offered())))
                continue
            parsed = _parse_json(raw) if isinstance(raw, str) and raw.strip() else ({} if not raw else raw)
            if not isinstance(parsed, dict):
                out.append((cid, "Not run (no step used): the arguments were not one JSON object."))
                continue
            faults: list[str] = []
            args = _fit_args(parsed, self.schemas[name], "arguments", faults)
            if faults:
                out.append((cid, "Not run (no step used): " + "; ".join(faults[:5]) + "."))
                continue
            again = [i for i in self.ids_in(_dump(args)) if self.listings.get(i, {}).get("opened")] if name == self.roles.get("open") and not self.refusals else []
            if again:
                out.append((cid, f"Not run (no step used): {', '.join(again)} was read already; its record: " + _dump(_clip(self.listings[again[0]].get("facts") or self.listings[again[0]].get("page") or {}, 300))[:1500]))
                continue
            ending = self.is_order(name, args) or self.is_decline(name, args)
            carting = name == self.roles.get("cart_add") and not self.is_removal(name, args)
            if not ending and name != self.roles.get("cart_add") and self.steps_left() <= self.needed():
                out.append((cid, "Not run (no step used): the last steps are kept for the order or the ending: call the order tool, and the harness selects and orders."))
                continue
            if ending or carting:
                summary = self.ask_before(force=True)
                if summary:
                    out.append((cid, "Not run (no step used): the shopper is asked first. " + summary))
                    continue
            if carting:   # the actor's own pick: its call runs, and the selection honours it when it fits and the pick rule prefers no other
                self.proposed.update({self.item_of(i): i for i, _ in self.cart_lines(args) if self.item_of(i)})
            if self.is_order(name, args):
                out.append((cid, self.settle() + (" [episode finished]" if self.finished else "")))
                continue
            if self.is_decline(name, args):
                hold = self.decline_gate()
                if hold:
                    out.append((cid, hold))
                    continue
                args = self.quiet(name, args)
            target = self.list_param(name)
            limit = self.list_limit(name)
            cut = ""
            if target and isinstance(args.get(target), list) and limit and len(args[target]) > limit:
                cut = f"\n[the harness sent the first {limit}: this view takes at most {limit} listings per call; not shown yet: {', '.join(map(str, args[target][limit:]))}]"
                args = dict(args) | {target: args[target][:limit]}
            out.append((cid, self.perform(name, args) + cut))
        return out

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """An ending under a request that forbids naming alternatives carries no listing in its remark."""
        if self.brief.get("quiet_decline") is not True:
            return args
        out = dict(args)
        for key, value in args.items():
            if isinstance(value, str) and key not in (self.fixed.get("decline") or {}) and self.ids_in(value) == [] and any(i in value for i in self.listings):
                out[key] = "Nothing in the store meets every requirement."
        return out

    # -- the finish ----------------------------------------------------------------------

    def order_now(self) -> str | None:
        """Send the order (no checks: the callers ran them or have no steps left for them). Returns the store's reply."""
        args = self.role_args("order")
        if args is None:
            return None
        text = self.perform(self.roles["order"], args)
        self.note("order_sent", reply=text[:300])
        return text

    def end_with_order(self) -> bool:
        """The cart is incomplete and no steps remain to complete it: a model says whether ordering it or ending without a purchase is more likely right."""
        reasons = "; ".join(f"item {n}: {e.get('reason')}" for n, e in self.selection.items())
        got = self.ask_json(FINISH + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger(rules=False) + "\n\nCART: " + (self.cart_text() or "empty") + "\n\nWHAT WAS FOUND: " + reasons,
                            [self.models["reader"], self.models["actor"]], 800, role="finish", spec=("obj", {"end": ("enum", ("order", "decline")), "reason": ("opt", "str")}, ("end",)), strict=True) or {}
        return got.get("end") == "order"

    def finish(self, again: bool = False) -> str:
        """An incomplete cart goes through the selection stage first; then a held cart is ordered, else the episode ends
        without a purchase. With `again`, a refused order is returned for one more round."""
        if self.finished or self.steps_left() <= 0:
            return ""
        for ident in self.dead_lines():
            if self.steps_left() <= 1 or not self.remove_line(ident):
                break
        complete = lambda: sum(1 for i in self.cart if self.buyable(i)) >= len(self.items)
        if not complete() and self.steps_left() > len(self.items) + 1 and self.time_left() > FORCE_SECONDS:
            self.note("finish_settle", result=self.settle()[:300])
        if self.finished:
            return ""
        if self.cart and self.roles.get("order") and (complete() or not self.roles.get("decline") or self.end_with_order()):
            hold = self.order_gate()
            if hold:
                self.note("finish_hold", hold=hold[:300])
            text = self.order_now() if not self.finished and self.steps_left() > 0 else None
            if self.finished or (again and text):
                return "" if self.finished else text
        if self.roles.get("decline") and not self.finished and self.steps_left() > 0:
            if self.steps_left() >= 2 and self.time_left() > FORCE_SECONDS:
                self.process_due("end", "end")
            args = self.role_args("decline")
            if args is not None and not self.finished and self.steps_left() > 0:
                text = self.perform(self.roles["decline"], self.quiet(self.roles["decline"], args))
                self.note("forced_decline", reply=text[:300])
        return ""

    def push_status(self) -> None:
        """The status card goes last, as a message of its own, and the previous card leaves: the newest card holds the whole
        state, and the history before it stays exactly as it was, so the model's prompt cache keeps it."""
        for index in range(len(self.history) - 1, 1, -1):
            if self.history[index].get("role") == "user" and str(self.history[index].get("content") or "").startswith("[status] "):
                del self.history[index]
                break
        self.history.append({"role": "user", "content": self.status()})

    def trim(self) -> None:
        size = sum(len(_dump(m.get("content"))) for m in self.history)
        if size <= HISTORY_CHARS:
            return
        # oldest first, every result and harness note but the latest four, until the history fits (messages stay, so
        # each tool call keeps its reply)
        # cut at once to well under the limit, so the cached prefix is broken rarely, not on every turn
        older = [i for i, m in enumerate(self.history) if m.get("role") in ("tool", "user") and i > 1][:-4]   # the guide and the request stay whole
        for index in older:
            content = str(self.history[index].get("content") or "")
            if len(content) > 400:
                ids = [i for i in self.seen_order if i in content][:20]
                self.history[index]["content"] = content[:250] + f"... (older, shortened; listings in it: {', '.join(ids)})"
                size -= len(content) - len(self.history[index]["content"])
            if size <= HISTORY_CHARS * 0.6:
                return

    def work(self, turns: int) -> None:
        idle = misses = stalled = 0
        if self.seed and not self.finished:   # the request's own values first: a word search for each surfaces the listings that carry it
            found, self.seed = self.value_search(self.seed, "request's stated"), []
            if found:
                self.history.append({"role": "user", "content": "[harness]" + found + " Search further for an item none of them is."})
                self.push_status()
        for _ in range(turns):
            if self.finished or self.steps_left() <= 0 or self.dead or self.time_left() < FORCE_SECONDS:
                return
            if self.cost >= SPEND_HARD or self.calls >= ALL_CALLS:
                self.note("spend_stop", cost=round(self.cost, 4), calls=self.calls)
                return
            # the questions go out once every item has listings seen (the planner reads what it needs to choose them), or when the steps left only just hold them
            seen_all = all(any(self.item_of(i) == n for i in self.seen_order) for n in range(1, len(self.items) + 1))
            tight = self.steps_left() <= self.needed() + 1
            if not self.asked and self.roles.get("message") and self.msgs_free() > 0 and (seen_all and self.ready() or tight):
                summary = self.ask_before(tight)
                if summary:
                    self.history.append({"role": "user", "content": summary})
                    self.push_status()
                    continue
            if self.settle_due and (self.asked or not self.roles.get("message")):
                text = self.settle()
                if not self.finished:
                    self.history.append({"role": "user", "content": "[harness selection] " + text})
                    self.push_status()
                continue
            if self.pending_refs():
                self.resolve_refs()
            self.trim()
            message = self.infer(self.history, [self.models["actor"], self.models["reader"]], 1500, tools=self.offered(), role="actor")
            if message is None:
                misses += 1
                if misses > 2:
                    return
                continue
            calls = []
            for number, call in enumerate(message.get("tool_calls") or []):
                fn = call.get("function") if isinstance(call, dict) and isinstance(call.get("function"), dict) else {}
                raw = fn.get("arguments")
                calls.append({"id": str(call.get("id") or f"c{self.step}_{number}"), "type": "function",
                              "function": {"name": str(fn.get("name") or ""), "arguments": raw if isinstance(raw, str) else _dump(raw or {})}})
            content = message.get("content") if isinstance(message.get("content"), str) else ""
            if not calls:
                idle += 1
                self.history.append({"role": "assistant", "content": content or "(no call)"})
                self.history.append({"role": "user", "content": "Call one of the tools to continue."})
                self.push_status()
                if idle >= 3:
                    return
                continue
            idle = 0
            self.history.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            self.note("turn", text=content[:300], calls=[(c["function"]["name"], c["function"]["arguments"][:200]) for c in calls])
            self.actor_why = "actor: " + (content.strip()[:200] or "(no words)")
            opened, step = sum(1 for l in self.listings.values() if l["opened"]), self.step
            replies = self.run_calls(calls)
            # newly opened listings may hold a fit: the selection judges them now, so the first order (where the store's
            # sell-outs and changes arrive) comes as soon as a fit exists, not when the actor next calls the order tool
            self.settle_due = self.settle_due or (sum(1 for l in self.listings.values() if l["opened"]) > opened and not self.finished)
            for cid, text in replies:
                self.history.append({"role": "tool", "tool_call_id": cid, "content": text})
            self.push_status()
            # turns in a row that moved nothing in the store (each call answered or refused without a step): in the last
            # steps three of them, elsewhere six, hand over to the finish, since more turns would only cost model calls
            stalled = stalled + 1 if self.step == step and not self.finished else 0
            if stalled >= (3 if self.steps_left() <= self.tail() + 2 else 6):
                self.note("stalled", step=self.step)
                return

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_request()
        if self.opening is not None:
            self.absorb("__opening__", self.opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if self.opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + _dump(_clip(self.opening, 300))[:4000]
        intro += "\n\nStart by searching for each item and opening several listings of it in full, so the questions to the shopper can be chosen from the properties these listings carry."
        self.history = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.push_status()
        self.work(MAX_TURNS)
        if not self.finished:
            refused = self.finish(again=True)
            if refused and not self.finished and self.steps_left() > 1 and self.time_left() > FORCE_SECONDS + 15 and not self.dead:
                self.history.append({"role": "user", "content": "[harness] The order was refused: " + refused[:2000] + "\nThe harness selects again; search and open more listings when it reports that none fits."})
                self.push_status()
                self.work(10)
            if not self.finished:
                self.finish()
        self.note("end", finished=self.finished, steps=self.step, cost=round(self.cost, 5), cart=dict(self.cart), messages=self.msg_used, invalid_replies=self.invalid_replies)
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Episode(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.note("crash", error=f"{type(exc).__name__}: {exc}"[:400])
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.note("crash", error=f"{type(again).__name__}: {again}"[:400])
        return episode.trace


__all__ = ["agent_main"]
