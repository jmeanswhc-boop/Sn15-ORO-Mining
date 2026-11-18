from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

# Model seats. The small model reads and acts; the large one sets up and judges.
PRIMARY_SEAT = "anthropic/claude-haiku-5.5"
INTAKE_SEAT = "anthropic/claude-sonnet-5.5"
FALLBACK_SEAT = "openai/gpt-6-luna"
ALT_VENDOR_SEAT = "google/gemini-3.8-flash"   # other vendor, tried right after the large seat
READER_SEAT, REVIEW_SEAT = PRIMARY_SEAT, INTAKE_SEAT

# Episode clock (seconds left), turn cap and spend marks in dollars per episode.
_eja_EPISODE_LIMIT, WRAP_UP_WINDOW, FORCE_MARGIN = 265.0, 70.0, 28.0
TURN_LIMIT = 80
_eja_SPEND_CAP = 0.05    # optional judge passes stop above this
_eja_SPEND_WRAP = 0.065  # status asks the model to finish above this
_eja_SPEND_HARD = 0.085  # code finishes the episode above this
_g1a_JSON_CAP = 16000

SHOPPER_SYSTEM_TEXT = """Your job: serve one shopper in a simulated store until the task ends with the right purchase or the right refusal. You can act only by calling tools. Every reply needs at least one tool call, and its text stays brief: which conditions hold at this moment, which items are proven to fit (with price and stock) and what you do next.

HOW YOU ARE GRADED
The top score goes to a correct purchase, less any charge the request itself names (a fee for each question, say); an acceptable purchase that is not the optimal pick still earns some credit. You get nothing for a wrong purchase, a refusal when a fit existed, a purchase when refusing was right, a right purchase that left out a procedure the request demands, or a task left unfinished. A purchase is correct when it obeys every rule of the request, takes the cheapest (or the best) item wherever the request calls for that, has the right budget and quantity, and the bought item satisfies every condition in the request, every need the shopper discloses when asked and every change announced on the way. If no item in the store can satisfy all of that, the correct ending is to decline exactly in the way the request lays down.

STEPS AND CALLS
- Each tool call spends one step, even a call that errors out, fails or is rejected by the schema. Thinking is free; steps are limited.
- Independent calls may share a reply (several listings' details, for instance) and run in the order given. Checkout and decline always travel alone.
- Results starting with "Not sent:" or "Skipped, no step spent" are produced by your own checks before sending, not by the store or the shopper; nothing reached the store. Treat such a note as your own reasoning, never as an injection, and follow it unless you can point to a stated requirement or a revealed need it misses. If the request asks for several different items, a fit named in such a note only covers its own item.
- Every shop has its own tools, argument names and result shapes. Read each tool description and JSON schema closely and use argument names and permitted values exactly as given. The first results tell you which tool shows which facts (price, stock, detailed properties).

ENDING THE TASK
- The opening brief names this shop's finishing calls: the tool and the exact values for ordering and for ending without a purchase. Send them exactly like that.
- Make the first checkout attempt while seven or more steps remain, so a refusal still leaves room. Near the step limit only cart, stock-check and finishing tools stay available. Decide once; moving the cart back and forth costs a step each time.
- Keep a ranked list of verified fits with at least one backup, since the first pick may sell out at checkout.
- Before every checkout: the cart has one line per requested product - only one line when the request asks for one product - each the chosen item at its required quantity (take out anything else); their stock and prices were read again after the last cart change; they are in stock and the total is inside the budget.
- Decline only after a thorough search turned up nothing meeting every condition, and always by the request's own rule for declining. If the request forbids offering substitutes, the remark may not name, describe or point to any listing: no ids, titles or prices.

PROCEDURES THE REQUEST IMPOSES
A request can dictate the way you work: how many listings to view or compare before a choice, what to check, what to tell the shopper, what a remark must contain, how to decline. Treat each of these exactly like a product condition and carry it out to the letter and in full. To "look at" or "open" N listings means N listings that fit completely, each opened through the details tool.

WHEN THINGS CHANGE
- Price and stock can shift without notice, so check both again right before every checkout attempt; once price times quantity exceeds the current budget the item no longer fits.
- The shopper may change things along the way: the budget (up or down), a condition (dropped, swapped or added), a price going up, an item selling out; a single line may hold several of these. Such lines can turn up inside any tool result (an extra field with sentences) or with a checkout attempt. A change applies from the moment it appears: test your pick and backups against it again, and under a cheapest or best rule choose again from the new acceptable set.
- A "refusal" field in a checkout result means the order was REFUSED, even if the inner result sounds like success. The cart is unchanged and the work continues: read the refusal and apply every change it names, remove anything that is no longer acceptable (a sold-out item, say), add the next verified fit, re-check price and stock, then order again.

MESSAGES TO THE SHOPPER
- The request never states every need: a hidden need only shows up when a message asks about its property. Each message covers exactly ONE property: a short, plain, friendly question naming it with the key words the listings use (example values only to set it apart from a similar neighbouring key). Do not ask about the listing itself (title, maker, category, description) or about a key that bundles the variants on sale or several features. No instructions, no mention of lists, labels, positions or rules, never two questions at once.
- If the request leaves open a point only the shopper can settle - typically two wishes that may not both be achievable, and which one matters more - ask before you choose.
- Questions you raise yourself go in this order: first open the details of two or three listings; then the properties the story points to (who it is for, how it looks, where or how it is used); last, properties on the listings whose values differ across candidates. Skip anything the request has already decided, and stay inside the allowed number of messages.
- When a planner is running, it picks and sends the questions for you once you have opened details for two or three listings of the wanted kind; so do that early, and before any cart call whenever messages are allowed: ordering while messages are still unused misses needs the shopper has not mentioned. Answers then appear in the conversation along with their meaning; every need they reveal must be met, so search again combining it with the values the request gave.
- "Don't mind" or "anything is fine" frees that property. Once the shopper states a preference or a limit (numbers included), it stands for the rest of the task in their exact words.
- If the shopper asks you something (to confirm a change, say), it is answered for you while messages remain; either way the change binds you.

PICKING ONE FIT
- "Ideally", "would be nice", "if possible" and similar phrases mark wishes, not conditions; they only separate items that already meet everything, and a cheapest rule beats them.
- For "best" by a measure (most, largest, longest, highest...), compare every fit on that measure the same way and take the extreme; a listing with no value for the measure cannot win.
- For "cheapest", one fit proves nothing: buy the lowest-priced item among ALL items meeting every condition. Make sure every candidate row has been shown (more pages or narrower searches when the reported total exceeds the visible rows), compare the prices yourself and verify from the cheapest up. Also search the required value on its own (just its words, limited to the section where possible) and read that search to the end: titles word the rest differently, so searches with more words miss listings. Once every cheaper listing has been checked and ruled out, the first complete fit is the answer and the later fits are backups.

JUDGING WHETHER A LISTING SATISFIES A CONDITION (the outcome hangs on this)
1. Use the listing's own property for the condition as evidence. The property can be written many ways (spaces, dots, underscores, prefixes) and can appear under more than one key; any key that names it will do. Match on meaning: ignore case, accept close synonyms and singular/plural, and one entry of a comma-separated list suffices. A condition written as a value with an explanation in brackets is met by the value; the bracket just explains. The wanted word with a separate qualifier in front (a shade or a finish of that colour) still states it. Not a match: words that only share their opening letters; one compound word that contains the wanted word yet names a narrower kind ("tealwood" tells you nothing about "wood"); a broader class or a different one ("girls" does not give "women", plain "unisex" does not give "female").
2. A comma list containing the required option is a match. But a key bearing the property's own name (any spelling, singular or plural) that holds a single other value fails the listing, even when another key's list contains the wanted value (several widths under one key, a different single width under a key of the same name: fail). A list under a longer, differently named key (available widths, say) does not count against it.
3. Title and description count as evidence only for a listing without any key for the condition; for that listing a variant or option key also counts if it holds the complete wanted value (it names the version on sale: "options: teal" decides a wish for teal). If the listing's own property says otherwise, it fails, whatever the title or description claims. A word in a foreign language, or a conclusion drawn from other facts, does not state a value.
4. One property never substitutes for a neighbouring one (length tells nothing about style, and the fabric of the lining tells nothing about the main material).
5. Numbers: convert units first, then compare numerically. An edge value lies within the limit. A range passes only if the whole of it meets the limit. "covers N" means the item's own stated range must reach N, so a range that stops below N fails. If the request words the range exactly as this shop's listings word it (number plus wording), that phrase is the value to search for, and a listing that shows the figure only in another form (converted units, single grades listed) falls short.
6. Respect closed lists ("only A or B") and exclusions ("nothing containing X"). An exclusion is proven only if the listing's own key for that property shows a different value; silence proves nothing, so prefer a fit that states the property and fall back on a silent one only when no stating fit is reachable.
7. The category carries the product kind, so no property has to repeat the category name. Where the shop sorts listings under category labels, the label decides the kind: anything under the label that matches the whole wording of the request is that product, even if its title or text sounds like a spare part, an accessory or something else, and only that exact label will do (for "A or B", use a category "A & B" rather than its narrower sibling "B"). Only listings shown as available count, and price times quantity must stay inside the budget (equal is fine).

SEARCHING
- Search tools may be limited: a cap on visible rows or pages, ordering by id rather than relevance or price, all words required, filters that only work in one mode. Compare the total a search reports with the rows you actually see, and keep narrowing (sharper words, category or price filters, further pages) until you have seen the whole relevant set.
- Learn early which category labels this shop uses for the requested kind of product (listings show them). Among neighbouring labels, the right one matches the request's wording as a whole; search and buy under that one."""

INTAKE_TEXT = """OUTPUT: one JSON object on one single line - no line breaks or indentation inside it and no text before or after it.

Your reading feeds a shopping assistant: it gets the shopper's request in structured form plus a mapping from the store's tools to the jobs they perform. Below these instructions you find the store's tool list first and then the request. Sort each sentence of the request into exactly one bucket: requirement, priority, soft want, rule, budget, hint, or pure story.

TOOL MAPPING
- "roles": an object whose keys are "search", "details", "view", "message", "cart", "cart_view", "order" and "decline"; each holds the exact name of the listed tool doing that job, or null. "message" writes to the shopper; "cart" puts items into the cart or removes them; "order" checks out or buys; "decline" ends the task with no purchase. For "order" and "decline" give an object instead of a name: {"tool": exact tool name, "fixed": {argument: value}}, where "fixed" assigns every required argument that has a list of permitted values the exact value making the tool do that job ({} if there is none). If a single tool does both jobs, its two "fixed" objects differ.

WHAT IS WANTED
- "lines": only when two or more different products are requested: one object per product in the request's order, each {"product": its kind in the request's words, "requirements": its own firm conditions written as described for "requirements", "quantity": its whole units}. In that case the top-level "requirements" still holds the conditions of all products together and "quantity" is the first product's. [] when one product is requested.
- "quantity_quote": the request's exact words giving the "quantity" number when it is above 1; null otherwise.
- "quantity": whole units of one listing to buy; 1 unless the request asks for several, and an item normally sold as one unit (a set, shoes in pairs) counts as 1.
- "ask_first": points that only the shopper can settle and that should be asked before choosing - which of two wishes weighs more when the request says both may not be possible (both wishes then go into "soft_wants", not "requirements"), or anything the request says to check with the shopper; [] if none.
- "soft_wants": what the shopper would like without strictly needing it (ideally, preferably, would be nice, if possible); never put these into "requirements".
- "requirements": list of strings, one for each firm condition on the product, written "property: constraint" with the shopper's exact value, limit, exclusion or closed list. Exclusions are written "property: != value" ("!= a, b" for more than one). Numbers keep their relation: "property: <= N", "property: >= N", or "property: covers N" when the item's own stated range (the sizes, powers or settings it is offered in) has to reach N. When the request gives a value in its own word and then explains that word, the word is the value and the explanation follows in parentheses ("finish: matte (no shine at all)"); never keep only the explanation. When the request itself names the property a value belongs to, use that name ("a heavy build" becomes "build: heavy", not a guessed neighbour). Leave out: price, stock, quantity, budgets, mere wishes, background about why, when or for what the item is wanted (an event, a date, a feeling) unless it is put as a condition on the item itself, and the product kind or its category - words that belong to the category's name (a season, an occasion or an audience in it, for instance) are covered by the category already.
- "product": the kind of product asked for, in the request's own words.

PRICE, STOCK AND RANKING
- "question_cost": if the request charges for asking the shopper, {"amount": number, "per": "question" or "step", "share": the part of the task's score (0 to 1) one question costs when the request relates its price to the score, else null, "quote": exact words}; otherwise null.
- "priority_detail": the measure meant, for "best"; "" otherwise.
- "priority": "cheapest" if the request explicitly asks for the cheapest or lowest-priced fitting option; "best" if it explicitly asks for the best by some measure; "none" otherwise. A tight budget or every penny mattering is no request for the cheapest - it only explains the cap.
- "in_stock_only": true when only items really in stock / on hand are wanted.
- "currency": the unit the request prices in, or null.
- "price_cap": the highest total price permitted, as a number; null if the request names none.

LIMITS AND PROCEDURE
- "decline_no_names": true when the request's way of declining forbids offering, naming or suggesting substitutes or alternatives; false when it wants something named; null when it has no rule for declining.
- "procedures": the rules that need a tool call of their own before the end, each {"do": short description, "quote": the request's exact words, "role": the job from "roles" whose tool performs it ("message", "view", "details", "search" or "cart_view"), "count": how many such calls (1 unless a number is given), "after_pick": true when it is about the chosen item and therefore comes after putting it in the cart (confirming the choice with the shopper, for example), "before": "order", "decline" or "any"}; [] if none.
- "rules": every procedural instruction to the assistant, unchanged in the request's own words (how many listings to look at, what to check or report, how to decline, what to avoid, what to include). When in doubt whether a sentence is a rule, list it. What the shopper says they will do themselves is story, not a rule.
- "step_budget": the permitted number of moves or steps, an integer; null when not stated.
- "message_budget": how many messages the assistant may send the shopper, as an integer; null when the request is silent.
"""

UPDATE_READING_TEXT = """Reply with one JSON object and nothing around it. A shopper has just written something new; the shopping assistant needs to know what that changes. Read the line by the guide below, then fill the fields listed after it.

Reading guide:
* Read the words literally but kindly: a value named in a homely, figurative or roundabout way is still firmly wanted; count a value as excluded only when the shopper clearly asks to avoid it.
* Wishes ("ideally", "would be nice" and the like) are not needs.
* A plain "anything works", "no preference" or "don't mind" puts the asked property into "no_preference".
* Under, over, at most, at least, no more than: a comparison like that with a number is a limit; keep its unit.
* If the shopper says which of two wishes matters more, that one is a need.
* Every budget the shopper states, higher or lower, goes into "new_budget" and replaces the old one, good news too.
* A replacement appears twice: the new requirement in "needs", the old one in "dropped".
* One line may carry several changes; report all of them.

Fields:
- "answer_class": null unless the assistant's question is shown; then {"class": "reveal" (a wish for the asked property is given), "not_picky" (no preference on it), "split" (the question is read as several things, or the shopper asks which thing is meant), "unclear" (the shopper did not understand the message and wants other wording) or "other"; "key": the property the answer concerns; "value": the wanted value, reveal only; "relation": "is", "at most", "at least" or "not", reveal only; "quote": the shopper's exact words carrying it}.
- "reply": if the shopper asks the assistant something or wants a confirmation, a short plain answer to send back naming the change in substance (confirming the new value, e.g.); "" otherwise.
- "meaning": a short sentence on what the shopper means.
- "unavailable": false, unless the shopper says something can no longer be bought (gone, sold out, no stock left, its button disabled) - then true.
- "new_budget": the new highest total price as a number; null otherwise.
- "no_preference": properties the shopper does not mind about.
- "dropped": earlier requirements that no longer hold.
- "needs": firm product requirements that are new in this line, each "property: constraint" with the exact value, limit, closed list, exclusion ("property: != value") or numeric relation ("property: <= N", ">= N" or "covers N") stated. Name the property the shopper's words name, even if the assistant asked about another one.
- "requirements_now": every firm product requirement as it stands after everything said so far, rebuilt from the request, the earlier shopper lines and this line. Earlier requirements keep their earlier wording and stay unless dropped or replaced; a replaced one keeps only its newest form; each newly revealed need is added. Forms as for "needs" (a value the request names and explains keeps its own word, followed by the explanation in parentheses). Price, stock, quantity, the product category and background about why or when the item is wanted are not requirements."""

ITEM_CHECK_TEXT = """Reply with one JSON object and no other text: {"item": the item's id, "fits": true or false, "problems": [short strings, one per requirement that is not met or not shown]}

Your task: you are the last check before one candidate item is ordered, and the order only counts if the item meets every requirement listed below. Judge only from the records shown here. Set "fits" to false only for a requirement the records clearly fail or that no record mentions at all; absent stock or price records are no reason to fail it.

Further checks:
- A wish marked as not required is not a requirement.
- If records disagree (on stock or price, say), the newest one counts.
- Shown as out of stock is a fail; price times quantity must not exceed the budget (equal passes).
- The item must be of the requested kind. Where listings have category labels the label decides; it has to match the request's wording as a whole, not a narrower or neighbouring label, and a listing under that label is of the requested kind whatever its title or description suggests (even a title that sounds like a part, an add-on or another sort of item).

Requirement by requirement:
- Closed lists and exclusions have to hold. An exclusion holds only when the listing's own key for that property shows another value; report silence about it.
- Numbers: convert units before comparing; a value on the edge is inside the limit; a range passes only if all of it meets the limit. "covers N" needs the listing's own stated range to reach N, so a range stopping short of N fails. When the requirements note that the store states the value in the request's own wording, only a listing that states that wording meets it.
- Only for an item with no key for that property may title and description, or the variant or option key, decide: a wanted value stated whole there ("options: teal" for a teal colour) meets it. If the item's own property gives a different value, the requirement fails whatever the title says. A word in another language, or a conclusion from other facts, does not state the value.
- The value of a related but different property does not count.
- A key named like the property itself (any spelling, singular or plural) holding one single different value fails the item, even if another key's list includes the wanted value.
- Find the value under the item's own property for the requirement. Every key naming the property counts, whatever its spelling (spaces, dots, underscores, prefixes), and one property can appear under several keys. Compare by meaning: case is irrelevant, singular/plural and close synonyms are fine, one entry of a comma list suffices; the wanted word with a separate qualifier in front (a shade, a finish) still matches, whereas a single longer fused word that merely contains it does not; a wider or other class, or a merely similar-looking word, does not match. A requirement written as a value plus an explanation in parentheses is met by the value; the explanation only says what it means."""

QUESTION_PLAN_TEXT = """Reply with one JSON object and no other text:
{"property": "...", "question": "...", "chance": 0.0, "repeat": false, "deviate": ""}
"chance" = your estimate from 0 to 1 that the property hides an unspoken need. "deviate" = "" when the pick is one the candidate block allows; for any other pick, the reason (a clue in the request pointing to it). When nothing worth a message is left: {"property": null}.

Your task: pick the shopping assistant's next message to the shopper. Shoppers often keep a need back; such a need is always the wanted value of one single product property and only comes out when a message asks about exactly that property. So name one property and word one question about it.

What follows below: the requirements in force; possibly a "Must come from the shopper" line; earlier questions with their answers; the "Detail-sheet properties so far" block from the opened listings (each line = name, number of listings showing it, sample values; each line is a property of its own even when two names look alike, and the names inside "[also spelled: ...]" are only other spellings of that line); possibly a numbered candidate list ("Candidate properties are listed below ..."); possibly an "The previous answer seemed to cover more than one thing" note; and the request.

Never pick:
1. a property no listing has, or one asked before;
2. a property whose values are mostly the name of the requested item itself (the item word decides it already);
3. a property the request has settled - its stated requirements, price, stock, quantity - except two cases that stay open: one the request only offers as a choice between alternatives, and one stated with a broad word that the listings split into narrower values (wanting "wood" where listings show oak, pine or walnut). A single exact value that the listings use unchanged keeps it settled;
4. terms of the sale or of the listing rather than features of the item;
5. a key collecting the buyable variants, or one naming several features at once (mixed kinds of values, or long item lists): hidden needs never look like that.
Never ask for a confirmation, a choice among listed options, whatever else the shopper might want, the purpose, or the budget.

How to rank the rest: the likeliest home of an unspoken need comes first. Clues come from the request (who it is for, where and how it is used, how it should look), the requirements, earlier questions and answers, and the listing properties. An unasked "Must come from the shopper" item goes before all else. If two wishes compete over one property, ask about that property and mention both values; wishes about two different properties need two separate messages. Coverage beats rarity (a property most or all opened listings state beats one that only a few carry), and variety beats uniformity (values that differ between listings beat a value marked "(same value everywhere)"). If the candidate list is there, choose from it - from the first three if the list requires that - led by the request's clues.

How to word the question: one short, plain, open sentence (well below a hundred characters) to the shopper about that property, using the key words the listings use. If that key word also appears inside other, longer keys, add the item's own noun plus one or two values found only under this key, so it cannot be mixed up with the neighbour; otherwise no example values. One noun for the item, never two item names stuck together; when several different products are requested, name only the one product the property belongs to, in the request's own words for it. Nothing to approve, no second question, no lists, no instructions.

If the "The previous answer seemed to cover more than one thing" note is present, ask about that same property again but more precisely - add the item's noun, or if it already had it, one or two values found only under that property - in wording different from the earlier question, and set "repeat" to true; otherwise "repeat" is false."""

DECLINE_AUDIT_TEXT = """Reply with one JSON object only: {"possible_fits": [up to 5 ids of listings that seem to meet every requirement, best first], "reason": "one sentence"}.

A shopping assistant is about to tell the shopper that nothing suitable exists. Before it does, look carefully through the records once more; they come after these instructions, together with the requirements in force and the shopper's request.

How to judge a record: a requirement is met only when the listing's own key for that property holds the value (the key's spelling does not matter). A comma list containing it, a plural for a singular or a close synonym all pass; a wider or different class does not, and a key with the same name holding one single different value fails the listing. Use title or description only where no such key exists. Convert units before comparing numbers; a value exactly on a limit is fine. The price must be within the budget, and the listing must not be marked out of stock."""


# Separate proxy links for the model and for the store.
_eja_model_link = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_eja_world_link = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

_ID_SHAPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")


# ---------------------------------------------------------------- JSON text in and out

def _shorten_value(obj, room, depth=0):
    """Copy of a JSON value with long strings clipped and anything deeper than eight levels dropped to None."""
    if depth > 8:
        return None
    deeper = depth + 1
    if isinstance(obj, dict):
        return dict((name, _shorten_value(part, room, deeper)) for name, part in obj.items())
    if isinstance(obj, list):
        return [_shorten_value(part, room, deeper) for part in obj]
    if isinstance(obj, str) and len(obj) > room:
        return obj[:room] + "..."
    return obj


def _to_compact_json(obj):
    """One-line JSON of a value (str() when it cannot be encoded), clipped at the text cap."""
    try:
        out = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        out = str(obj)
    return out if len(out) <= _g1a_JSON_CAP else out[:_g1a_JSON_CAP] + "...(cut)"


_g1a_FENCE = re.compile(r"^```[A-Za-z]*")
_g1a_UNPARSED = object()


def _g1a_parse(text):
    """The decoded value, or _g1a_UNPARSED when text is not JSON."""
    try:
        return json.loads(text)
    except ValueError:
        return _g1a_UNPARSED


def _g1a_unfence(text):
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = _g1a_FENCE.sub("", text).strip()
    return text[:-3].strip() if text.endswith("```") else text


def _extract_json_dict(text):
    """Dict a model reply holds: the whole text (fences allowed), else the span from the first to the last brace."""
    if not isinstance(text, str):
        return None
    body = _g1a_unfence(text)
    whole = _g1a_parse(body)
    if whole is _g1a_UNPARSED:
        left, right = body.find("{"), body.rfind("}")
        whole = _g1a_parse(body[left:right + 1]) if 0 <= left < right else None
    return whole if isinstance(whole, dict) else None


def _gather_sentences(obj, out, depth=0):
    """Add to out each string of at least four words, walking lists (first ten entries, depth four)."""
    pending = [(depth, obj)]
    while pending:
        depth, node = pending.pop()
        if depth > 4:
            continue
        if isinstance(node, str):
            if len(node.split()) >= 4:
                out.append(node)
        elif isinstance(node, list):
            pending.extend((depth + 1, part) for part in reversed(node[:10]))


def _gather_ids(obj, out, depth=0):
    """Add to out every id-shaped string held under a key mentioning "id" (depth six at most)."""
    if depth > 6:
        return
    if isinstance(obj, dict):
        for name, part in obj.items():
            if isinstance(part, str) and "id" in name.lower() and _ID_SHAPE.match(part):
                out.add(part)
            else:
                _gather_ids(part, out, depth + 1)
    elif isinstance(obj, list):
        for part in obj:
            _gather_ids(part, out, depth + 1)


# ---------------------------------------------------------------- tool arguments checked against their schema

def _as_number(obj):
    """The value when it is a real int or float (bools excluded), else None."""
    if isinstance(obj, bool) or not isinstance(obj, (int, float)):
        return None
    return obj


def _declared_types(schema):
    """Declared "type" of a schema as a list when it is one name; any other content as it stands."""
    kind = schema.get("type")
    return [kind] if isinstance(kind, str) else kind


_g1a_SKIP = object()   # marker: a conversion does not apply
_g1a_WHOLE_RX = re.compile(r"\s*-?\d+(\.0+)?\s*")
_g1a_REAL_RX = re.compile(r"\s*-?\d+(\.\d+)?\s*")


def _g1a_kind_test(kind):
    """Membership test for one schema kind; None for a kind this code does not know."""
    if kind == "integer":
        return lambda v: _as_number(v) is not None and isinstance(v, int)
    if kind == "number":
        return lambda v: _as_number(v) is not None
    plain = {"string": str, "boolean": bool, "object": dict, "array": list}.get(kind)
    if plain is not None:
        return lambda v: isinstance(v, plain)
    if kind == "null":
        return lambda v: v is None
    return None


def _g1a_fits_kinds(obj, kinds):
    """True if the value belongs to some declared kind; unknown kinds accept anything."""
    for kind in kinds:
        test = _g1a_kind_test(kind) if isinstance(kind, str) else None
        if test is None or test(obj):
            return True
    return False


def _g1a_as_integer(v):
    if isinstance(v, float):
        return int(v) if v.is_integer() else _g1a_SKIP
    if isinstance(v, str) and _g1a_WHOLE_RX.fullmatch(v):
        return int(float(v))
    return _g1a_SKIP


def _g1a_as_real(v):
    if isinstance(v, str) and _g1a_REAL_RX.fullmatch(v):
        got = float(v)
        return int(got) if got.is_integer() else got
    return _g1a_SKIP


def _g1a_as_string(v):
    return str(v) if _as_number(v) is not None else _g1a_SKIP


def _g1a_as_flag(v):
    word = v.lower() if isinstance(v, str) else None
    return {"true": True, "false": False}.get(word, _g1a_SKIP)


_g1a_CAST = {"integer": _g1a_as_integer, "number": _g1a_as_real, "string": _g1a_as_string, "boolean": _g1a_as_flag}


def _g1a_cast(obj, kinds):
    """Move a value into a declared kind when nothing is lost; unchanged otherwise."""
    if _g1a_fits_kinds(obj, kinds):
        if isinstance(obj, float) and obj.is_integer() and "integer" in kinds:
            return int(obj)
        return obj
    for kind in kinds:
        cast = _g1a_CAST.get(kind) if isinstance(kind, str) else None
        if cast is None:
            continue
        try:
            got = cast(obj)
        except (TypeError, ValueError):
            got = _g1a_SKIP
        if got is not _g1a_SKIP:
            return got
    return obj


def _g1a_limit_notes(where, size, schema, checks):
    """Messages for the broken bounds among checks: (key, numeric bound only, test, wording)."""
    found = []
    for key, needs_number, broken, wording in checks:
        if key in schema:
            bound = schema[key]
            if (isinstance(bound, (int, float)) or not needs_number) and broken(size, bound):
                found.append("%s: %s" % (where, wording.format(bound)))
    return found


_g1a_NUMBER_LIMITS = (
    ("minimum", False, lambda m, e: m < e, "is under the minimum {}"),
    ("maximum", False, lambda m, e: m > e, "is over the maximum {}"),
    ("exclusiveMinimum", True, lambda m, e: m <= e, "has to be greater than {}"),
    ("exclusiveMaximum", True, lambda m, e: m >= e, "has to be less than {}"),
)
_g1a_TEXT_LIMITS = (("minLength", False, lambda m, e: m < e, "needs at least {} characters"),)
_g1a_LIST_LIMITS = (
    ("minItems", False, lambda m, e: m < e, "give at least {} items"),
    ("maxItems", False, lambda m, e: m > e, "give at most {} items"),
)


def _g1a_choice_notes(obj, schema, where):
    """(value, problems) after the enum check (case repair for text) and the const check."""
    found = []
    if "enum" in schema:
        allowed = schema.get("enum")
        if obj not in allowed and isinstance(obj, str):
            wanted = obj.strip().lower()
            obj = next((o for o in allowed if isinstance(o, str) and o.lower() == wanted), obj)
        if obj not in allowed:
            found.append("%s: allowed values are %s" % (where, allowed))
    if "const" in schema and obj != schema["const"]:
        found.append("%s: the only allowed value is %r" % (where, schema["const"]))
    return obj, found


def _g1a_walk_list(items, schema, where, found):
    found += _g1a_limit_notes(where, len(items), schema, _g1a_LIST_LIMITS)
    entry_shape = schema.get("items")
    if not isinstance(entry_shape, dict):
        return items
    fixed_items = []
    for index, entry in enumerate(items):
        entry, entry_notes = _validate_value(entry, entry_shape, "%s[%d]" % (where, index))
        fixed_items.append(entry)
        found += entry_notes
    return fixed_items


def _g1a_walk_object(members, schema, where, found):
    known = schema.get("properties") or {}
    found += ["%s.%s: missing (required)" % (where, name)
              for name in schema.get("required") or [] if name not in members]
    if schema.get("additionalProperties") is False:
        found += ["%s.%s: unknown argument" % (where, name) for name in list(members) if name not in known]
    result = {}
    for name, part in members.items():
        if name in known:
            part, part_notes = _validate_value(part, known[name], "%s.%s" % (where, name))
            found += part_notes
        result[name] = part
    return result


def _eja_word_cut(text, cap):
    """First cap characters of text, moved back to a word end when the cut splits a word (if a space allows)."""
    head = text[:max(0, cap)]
    splits_word = len(text) > len(head) and head[-1:].isalnum() and text[len(head)].isalnum()
    if splits_word and " " in head.strip():
        head = head.rstrip().rsplit(None, 1)[0]
    return head.rstrip() or text[:max(0, cap)]


def _validate_value(obj, schema, where):
    """Check a value against a JSON schema, repairing what is safe. Returns (value, problems)."""
    if not isinstance(schema, dict):
        return obj, []
    kinds = _declared_types(schema)
    if kinds:
        obj = _g1a_cast(obj, kinds)
        if not _g1a_fits_kinds(obj, kinds):
            return obj, [where + ": should be " + "/".join(kinds)]
    obj, found = _g1a_choice_notes(obj, schema, where)
    if isinstance(obj, str):
        found += _g1a_limit_notes(where, len(obj), schema, _g1a_TEXT_LIMITS)
        if "maxLength" in schema and len(obj) > schema.get("maxLength"):
            obj = _eja_word_cut(obj, schema.get("maxLength"))
    elif _as_number(obj) is not None:
        found += _g1a_limit_notes(where, obj, schema, _g1a_NUMBER_LIMITS)
    elif isinstance(obj, list):
        obj = _g1a_walk_list(obj, schema, where, found)
    elif isinstance(obj, dict):
        obj = _g1a_walk_object(obj, schema, where, found)
    return obj, found


# ---------------------------------------------------------------- seats, tools and the opening brief

def _g1a_strip_cache(part):
    if isinstance(part, dict):
        return {key: val for key, val in part.items() if key != "cache_control"}
    return part


def _eja_seat_messages(model, history):
    """History shaped for one seat: other vendors lose all cache marks; Anthropic seats with three or more
    history get a cache mark on the first and the last message when that one is plain text."""
    if model.startswith("anthropic/"):
        if len(history) < 3:
            return history
        shaped = [dict(entry) for entry in history]
        for entry in (shaped[0], shaped[-1]):
            body = entry.get("content")
            if isinstance(body, str) and body:
                entry["content"] = [dict(type="text", text=body, cache_control=dict(type="ephemeral"))]
        return shaped
    shaped = []
    for entry in history:
        body = entry.get("content")
        shaped.append(dict(entry, content=[_g1a_strip_cache(p) for p in body]) if isinstance(body, list) else entry)
    return shaped


def _eja_status(resp):
    fault = resp.error
    if isinstance(fault, dict):
        return fault.get("status")
    return None


_eja_ROLES_PROMPT = """Reply with nothing but one JSON object on one line: {"roles": {"search": ..., "details": ..., "view": ..., "message": ..., "cart": ..., "cart_view": ..., "order": ..., "decline": ...}}
Each value is the exact name, as listed, of the shop tool doing that job, or null when no tool does it; one tool may serve two jobs. The tools of the shop are listed below. Jobs:
- search: looks up listings
- details: all properties of one listing
- view: one listing with its price and stock
- message: sends a note to the shopper
- cart: puts a listing into the basket or takes it out
- cart_view: shows what the basket holds
- order: buys what the basket holds and ends the task
- decline: ends the task without buying"""

_eja_INFERENCE_PATH = "/inference/chat/completions"
_eja_TOKEN_FACTOR = {ALT_VENDOR_SEAT: 2, FALLBACK_SEAT: 4}    # reasoning seats get a larger allowance
_eja_RETRY_WAITS = (1.5, 4.0)
_g1a_GATEWAY = {500, 503, 504}
_eja_PASSING_FAULTS = frozenset(_g1a_GATEWAY | {429, 502})   # model faults worth a short wait
_eja_STORE_GIVE_UP = frozenset(_g1a_GATEWAY | {403, 422})
_eja_ROLE_SEATS = [INTAKE_SEAT, PRIMARY_SEAT, FALLBACK_SEAT]
# Setup readings as (seats, max_tokens): first try, larger repeat after a cut reply, last try.
_g1a_LARGE_FIRST = (INTAKE_SEAT, ALT_VENDOR_SEAT)
_eja_SETUP_SEATS = tuple((list(_g1a_LARGE_FIRST) + list(tail), room) for tail, room in (
    ((PRIMARY_SEAT, FALLBACK_SEAT), 1500), ((FALLBACK_SEAT,), 4500), ((FALLBACK_SEAT, PRIMARY_SEAT), 4500)))
# Opening-brief lines drawn from plan lists: (plan field, line opener).
_OPENING_SECTIONS = tuple(zip(
    ("soft_wants", "ask_first", "rules"),
    ("- nice to have, NOT required (only a tie-breaker): ", "- must be learned from the shopper first: ", "- rule: ")))


def _describe_tool(spec):
    """One catalogue line for a tool: name, description, argument schema."""
    schema = _to_compact_json(spec.get("parameters") or {})
    return "* {} -- {} -- argument schema: {}".format(spec["name"], spec.get("description", ""), schema)


def _eja_has_reading(plan):
    return bool(plan.get("product") or plan.get("requirements"))


def _eja_is_tool(entry_obj):
    return isinstance(entry_obj, dict) and isinstance(entry_obj.get("function"), dict)


def _eja_params(spec):
    return spec.get("parameters") or {"type": "object", "properties": {}}


def _g1a_append_fresh(target, incoming):
    """Append the non-empty strings of incoming that target lacks; return the appended ones."""
    added = []
    for entry in incoming or []:
        if isinstance(entry, str) and entry and entry not in target:
            target.append(entry)
            added.append(entry)
    return added


class _NeedList:
    """What the shopper wants right now: stated and revealed conditions, drops, free properties, budget."""

    def __init__(self):
        self.stated, self.revealed, self.dropped, self.free, self.history = [], [], [], [], []
        self.newest_lines = None
        self.budget = None
        self.heard = set()

    def _g1a_live(self):
        if self.newest_lines is not None:
            return list(self.newest_lines)
        tagged = [(self.stated, " (stated)"), (self.revealed, " (revealed by the shopper)")]
        return [line + tag for pool, tag in tagged for line in pool if line not in self.dropped]

    def render_lines(self, currency, quantity):
        tail = [] if self.budget is None else [("total price at most %s %s" % (self.budget, currency)).rstrip()]
        tail.append("quantity %s" % (quantity,))
        for label, joiner, pool in (("no longer required: ", "; ", self.dropped),
                                    ("shopper does not care about: ", ", ", self.free)):
            if pool:
                tail.append(label + joiner.join(pool))
        return "; ".join(["; ".join(self._g1a_live()) or "see the request"] + tail)

    def fold_reading(self, reading):
        """Merge one reading of a shopper line into the wants; returns notes on what changed."""
        whole_list = reading.get("requirements_now")
        if isinstance(whole_list, list) and all(isinstance(line, str) for line in whole_list):
            self.newest_lines = [line for line in whole_list if line.strip()]
        gained = _g1a_append_fresh(self.revealed, reading.get("needs"))
        _g1a_append_fresh(self.free, reading.get("no_preference"))
        lost = _g1a_append_fresh(self.dropped, reading.get("dropped"))
        changes = ["new requirement: %s" % line for line in gained] + ["dropped: %s" % line for line in lost]
        new_cap = _as_number(reading.get("new_budget"))
        if new_cap is not None and new_cap > 0 and new_cap != self.budget:
            changes.append("new budget: %s" % (new_cap,))
            self.budget = new_cap
        if reading.get("unavailable"):
            changes.append("an item is no longer available: do not order it, and take it out of the cart")
        return changes


# Per-episode state: (start value, attribute names) and (container factory, attribute names).
_eja_START = tuple(zip(
    ("", None, False, True, -1, 0.0, 0),
    ("asking queued_reply sibling_next rerank_reason unclear_next",
     "message_budget quantity_field add_signature last_cart_view answer_reading split_pending forced_basis"
     " question_cost",
     "halted key_dry cart_used audit_done order_checked decline_checked split_retried urgent_tried",
     "planner_on",
     "cart_seen_at",
     "cost",
     "step messages_used order_attempts refusals needs_revision empty_plans cart_changed_at cart_generation"
     " refused_at")))
_eja_CONTAINERS = tuple(zip(
    (dict, list, set),
    ("plan roles fixed calls_by_role cart_state units_in_cart sent_units property_menu verdicts empty_holds"
     " mentioned decline_holds faults need_homes loose_ok",
     "messages trace records recent cart_ids asked asked_props reask_later sibling_props proven_fits procedures",
     "seen_ids cart_checked qty_warned held_arms relooked faulty proc_held")))

class Episode:
    # ================================================================ construction, clocks, budgets

    def __init__(self, problem_data):
        self.start_clock = time.monotonic()
        env = problem_data["environment"]
        view = env["policy_view"]
        self.binding = dict(env["binding"])
        self.policy = view
        self.tools = list(filter(_eja_is_tool, view.get("tools") or []))
        self.param_schemas = {}
        for entry in self.tools:
            spec = entry["function"]
            self.param_schemas[spec["name"]] = _eja_params(spec)
        self.max_steps = int(view.get("max_steps") or 30)
        self.finished = bool(view.get("finished"))
        # counters take their shared start value; containers get a fresh object each
        slots = [(names, (lambda value=start: value)) for start, names in _eja_START]
        slots += [(names, make) for make, names in _eja_CONTAINERS]
        for names, produce in slots:
            for attr in names.split():
                setattr(self, attr, produce())
        self._need_list = _NeedList()

    def _g1b_note(self, kind, **extra):
        self.trace.append({"kind": kind, **extra})

    def spend_exceeded(self):
        """True once model spend passed the episode cap (optional judge passes are then skipped)."""
        return _eja_SPEND_CAP < self.cost

    def seconds_remaining(self):
        used = time.monotonic() - self.start_clock
        return _eja_EPISODE_LIMIT - used

    def remaining_steps(self):
        return self.max_steps - self.step

    def _g1b_catalog(self):
        return "\n".join(_describe_tool(entry["function"]) for entry in self.tools)

    # ================================================================ model link

    def infer(self, messages, tools, models, max_tokens):
        """Try the seats in turn; the first reply holding text or tool calls wins, tagged with its model.
        A key whose every seat refused on payment or access grounds is marked dry."""
        refused = 0
        for seat in models:
            left = self.seconds_remaining()
            if left < FORCE_MARGIN - 10:
                return None
            _eja_model_link.timeout = int(min(75, max(8, left + 15 - FORCE_MARGIN)))
            outcome = self._g1b_post_seat(seat, messages, tools, max_tokens)
            message = self._g1b_unpack(outcome.data)
            if message:
                message["model"] = seat
                return message
            refused += _eja_status(outcome) in (402, 403)
            self._g1b_note("inference_failure", model=seat,
                           error=str(outcome.error)[:300] if outcome.error else "empty")
        if models and refused == len(models):
            self.key_dry = True
            self._g1b_note("key_dry", models=list(models))
        return None

    def _g1b_post_seat(self, seat, messages, tools, max_tokens):
        """One seat; an upstream fault that tends to clear gets the same seat again after a pause."""
        body = {"model": seat, "messages": _eja_seat_messages(seat, messages), "temperature": 0,
                "max_tokens": max_tokens * _eja_TOKEN_FACTOR.get(seat, 1)}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        outcome = _eja_model_link.post_verbose(_eja_INFERENCE_PATH, body)
        pauses = iter(_eja_RETRY_WAITS)
        while outcome.data is None and _eja_status(outcome) in _eja_PASSING_FAULTS:
            pause = next(pauses, None)
            if pause is None or self.seconds_remaining() < FORCE_MARGIN + 30:
                break
            time.sleep(pause)
            outcome = _eja_model_link.post_verbose(_eja_INFERENCE_PATH, body)
        return outcome

    def _g1b_unpack(self, data):
        """First choice's message plus a "cut" flag (None when it says nothing); the call's cost is added up."""
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            return None
        spent = (data.get("usage") or {}).get("cost")
        if isinstance(spent, (int, float)):
            self.cost += float(spent)
        head = choices[0]
        message = dict(head.get("message") or {})
        message["cut"] = head.get("finish_reason") == "length"
        return message if (message.get("content") or message.get("tool_calls")) else None

    # ================================================================ world link

    def call_store(self, action):
        """Send ONE action as the next step; faults that may clear are tried again, four tries at most."""
        envelope = dict(self.binding, step=self.step + 1, action=action)
        attempt = 0
        while attempt < 4:
            attempt += 1
            left = self.seconds_remaining()
            if left < 2:
                break
            _eja_world_link.timeout = int(min(95, max(3, left - 1)))
            outcome = _eja_world_link.post_verbose("/environment/call", envelope)
            if isinstance(outcome.data, dict):
                return self._g1b_advance(outcome.data)
            if not self._g1b_should_resend(outcome.error or {}, envelope, attempt):
                break
        return None

    def _g1b_advance(self, data):
        reported = data.get("step")
        self.step = reported if isinstance(reported, int) else self.step + 1
        if data.get("finished"):
            self.finished = True
        return data

    def _g1b_should_resend(self, error, envelope, attempt):
        """Record a failed world call and tell whether another try is worth it."""
        status = error.get("status")
        text = str(error.get("body") or "")
        self._g1b_note("env_failure", status=status, body=text[:300])
        if status == 402:   # the run has no money left
            self.halted = True
            return False
        if status == 409:
            return self._g1b_resync(text, envelope)
        if status == 429:
            return self._g1b_backoff(attempt)
        return status not in _eja_STORE_GIVE_UP

    def _g1b_backoff(self, attempt):
        # spread by step and attempt so parallel episodes do not retry in lockstep
        pause = 0.6 * attempt * (1.0 + 0.15 * ((7 * self.step + 3 * attempt) % 5))
        if pause + FORCE_MARGIN >= self.seconds_remaining():
            return False
        time.sleep(pause)
        return True

    def _g1b_resync(self, text, envelope):
        """A step-number clash: stop when the episode is over, else adopt the step the world expects."""
        if "finished" in text:
            self.finished = True
            return False
        found = re.search(r"expected (\d+)", text)
        if found is None:
            return False
        self.step = int(found.group(1)) - 1
        envelope["step"] = self.step + 1
        return True

    # ================================================================ the setup reading

    def setup(self):
        catalog = self._g1b_catalog()
        story = "SHOPPER REQUEST, AS WRITTEN:\n" + str(self.policy.get("query") or "")
        shown = self.policy.get("opening")
        if shown is not None:
            story += "\n\nSTORE SCREEN AT THE START:\n" + _to_compact_json(shown)
        turn = [{"role": "user", "content": [
            {"type": "text", "text": INTAKE_TEXT + "\n\nTOOLS THIS STORE OFFERS:\n" + catalog,
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": story}]}]
        # main seat first: the backup thinks before writing and may spend its whole allowance on that
        first, roomy, last = _eja_SETUP_SEATS
        reply, self.plan = self._g1b_read_plan(turn, first)
        if reply and reply.get("cut") and not self.plan:
            self.plan = self._g1b_read_plan(turn, roomy)[1]
        if not self.key_dry and not _eja_has_reading(self.plan):
            # a failed seat, or a reading with roles only: one more try
            self._g1b_note("setup_retry", had=sorted(self.plan))
            retry = self._g1b_read_plan(turn, last)[1]
            if _eja_has_reading(retry):
                self.plan = retry
        self._g1b_take_plan(self.plan)
        if not self.roles:
            self._g1b_roles_only(catalog)

    def _g1b_read_plan(self, turn, seating):
        seats, room = seating
        reply = self.infer(turn, None, seats, room)
        content = reply.get("content") if reply else None
        return reply, _extract_json_dict(content or "") or {}

    def _g1b_roles_only(self, catalog):
        """No tool roles came with the full reading: a small call asks for the roles alone."""
        answer = self.ask_judge(_eja_ROLES_PROMPT + "\n\nTool list:\n" + catalog, 400, _eja_ROLE_SEATS) or {}
        listed = answer.get("roles")
        if isinstance(listed, dict):
            self.plan["roles"] = listed
            self.roles = {job: tool for job, tool in listed.items()
                          if isinstance(tool, str) and tool in self.param_schemas}
        self._g1b_note("roles_fallback", roles=self.roles)

    def _g1b_take_plan(self, reading):
        self.roles, self.fixed = {}, {}
        listed = reading.get("roles")
        for job, entry in (listed.items() if isinstance(listed, dict) else ()):
            if not isinstance(entry, dict):
                entry = {"tool": entry, "fixed": {}}
            tool = entry.get("tool")
            if not (isinstance(tool, str) and tool in self.param_schemas):
                continue
            self.roles[job] = tool
            if job in _CLOSING_JOBS:
                preset = entry.get("fixed")
                self.fixed[job] = dict(preset) if isinstance(preset, dict) else {}
        self._g1b_settle_closing()
        self.procedures = self._g1b_procedures(reading.get("procedures"))
        self.question_cost = self._g1b_priced_ask(reading.get("question_cost"))
        allowance = _as_number(reading.get("message_budget"))
        if allowance is not None and allowance >= 0:
            self.message_budget = int(allowance)
        wants = self._need_list
        wants.stated = [str(item) for item in reading.get("requirements") or [] if item]
        wants.budget = reading.get("price_cap")
        cap = _as_number(reading.get("step_budget"))
        if cap is not None and 0 < cap < self.max_steps:
            self.max_steps = int(cap)

    # ================================================================ small readers other parts use

    @staticmethod
    def _ejb_cut(need_line):
        """Split a "property: constraint" text at its first colon."""
        text = str(need_line)
        return text.partition(":")

    @staticmethod
    def _ejb_key_words(need_line):
        """Word set of the property half of a "property: constraint" text."""
        return _word_set(Episode._ejb_cut(need_line)[0])

    @staticmethod
    def _ejb_label(field_name):
        """A key spelled as words: a lower-to-upper step and runs of '_' or '.' become spaces."""
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(field_name))
        return re.sub(r"[_.]+", " ", spaced)

    @staticmethod
    def _ejb_short_texts(record_blob, limit):
        found = []
        for key, value in record_blob.items():
            if isinstance(value, str) and 0 < len(value) <= limit:
                found.append((key, value))
        return found

    @staticmethod
    def _ejb_ints(record_blob):
        whole = {}
        for key, value in record_blob.items():
            if isinstance(value, int) and not isinstance(value, bool):
                whole[key] = value
        return whole

    def _ejb_props(self, tool, strict=False):
        """Argument specs of a tool; strict demands that the tool exists."""
        shape = self.param_schemas[tool] if strict else self.param_schemas.get(tool) or {}
        return shape.get("properties") or {}

    def _g1b_pins(self, job):
        return self.fixed.get(job) or {}

    # ================================================================ closing jobs: checks, readings, repairs

    _g1b_REPAIR_HEAD = (
        "OUTPUT CONTRACT - a lone JSON object, no prose before or after it:\n"
        '  {"order": {"tool": name, "fixed": {argument: value}}, "decline": {"tool": name, "fixed": {...}}}\n'
        "Inside it, \"order\" names whatever completes a purchase of the cart's current contents, and "
        "\"decline\" whatever closes the session while buying nothing. Under \"fixed\" list the exact value for "
        "each required argument that is a boolean or restricted to enumerated options. A single tool may cover "
        "both jobs; then each entry pins that same set of arguments, with values that differ.\n\n"
        "Our earlier attempt at this mapping did not hold up, for these reasons:\n"
    )

    def _g1b_faults(self, job):
        """What is wrong with one closing job's mapping. Pinned values that pass are rewritten in the schema's own
        spelling as a side effect."""
        tool = self.roles.get(job)
        if not tool:
            return [f"no tool is mapped to the '{job}' job"]
        specs = self._ejb_props(tool, True)
        held = self._g1b_pins(job)
        issues = []
        for arg in self.param_schemas[tool].get("required") or []:
            if arg not in held and _is_enum_spec(specs.get(arg)):
                issues.append(f"'{job}' leaves '{arg}' unpinned, a required choice argument of {tool}")
        for arg in list(held):
            spec = specs.get(arg)
            if not isinstance(spec, dict):
                issues.append(f"'{job}' pins '{arg}', an argument {tool} does not have")
                continue
            spelled, wrong = _validate_value(held[arg], spec, arg)
            if wrong:
                issues.append(f"'{job}' pins {tool}.{arg} = {held[arg]!r}, which is not valid ("
                              + "; ".join(wrong) + ")")
            else:
                held[arg] = spelled
        issues.extend(self._g1b_overlaps(job, tool, held))
        return issues

    def _g1b_overlaps(self, job, tool, held):
        found = []
        other = "decline" if job == "order" else "order"
        if self.roles.get(other) == tool:
            rival = self._g1b_pins(other)
            if held == rival:
                found.append(f"order and decline would make the very same {tool} call")
            elif set(held) != set(rival):
                found.append(f"order and decline both use {tool} but pin different argument sets "
                             f"({sorted(held)} vs {sorted(rival)}); each side must pin the argument that "
                             "tells them apart")
        if held:
            return found
        for role, mapped in self.roles.items():
            if mapped == tool and role not in _CLOSING_JOBS:
                found.append(f"'{job}' shares {tool} with the '{role}' role and pins nothing to tell the two "
                             "uses apart")
        return found

    def _g1b_read_jobs(self, prompt, seats):
        """job -> (tool, pinned values) from one reading; tools the store lacks are ignored."""
        answer = self.ask_judge(prompt, 500, seats) or {}
        jobs = {}
        for job in _CLOSING_JOBS:
            entry = answer.get(job)
            if not isinstance(entry, dict) or entry.get("tool") not in self.param_schemas:
                continue
            pins = entry.get("fixed")
            jobs[job] = (entry["tool"], dict(pins) if isinstance(pins, dict) else {})
        return jobs

    def _g1b_choice_values(self, tool, pins):
        """Just the pinned values that select behaviour, in the schema's spelling."""
        specs = self._ejb_props(tool)
        return {arg: _validate_value(pins[arg], specs[arg], arg)[0]
                for arg in pins if _is_enum_spec(specs.get(arg))}

    def _g1b_take_jobs(self, jobs):
        for job, (tool, pins) in jobs.items():
            self.roles[job] = tool
            self.fixed[job] = pins

    def _g1b_current_jobs(self):
        return {job: (self.roles[job], self._g1b_pins(job)) for job in _CLOSING_JOBS if job in self.roles}

    def _g1b_alike(self, one, two):
        return one[0] == two[0] and self._g1b_choice_values(*one) == self._g1b_choice_values(*two)

    def _g1b_cross_read(self, catalog):
        """A second, separate reading of the closing jobs; a disagreement is put to the setup seats."""
        mine = self._g1b_current_jobs()
        rival = self._g1b_read_jobs(_CLOSING_ROLES_TEXT + "\n\nTools on offer:\n" + catalog,
                                     [PRIMARY_SEAT, FALLBACK_SEAT])
        if not rival:
            return
        shared = [job for job in mine if job in rival]
        if any(not self._g1b_alike(mine[job], rival[job]) for job in shared):
            self._g1b_referee(mine, rival, catalog)
            return
        extra = {job: pair for job, pair in rival.items() if job not in mine}
        if extra:   # no conflict: keep what only the second reading found
            self.trace.append({"kind": "finish_read", "took": extra})
            self._g1b_take_jobs(extra)

    def _g1b_referee(self, mine, rival, catalog):
        def shown(jobs):
            return _to_compact_json({job: {"tool": tool, "fixed": pins} for job, (tool, pins) in jobs.items()})
        prompt = "\n\n".join([_ROLE_REFEREE_TEXT, "Tools on offer:\n" + catalog,
                              "READING A: " + shown(mine) + "\nREADING B: " + shown(rival)])
        verdict = self._g1b_read_jobs(prompt, [INTAKE_SEAT, ALT_VENDOR_SEAT, FALLBACK_SEAT])
        self.trace.append({"kind": "finish_disagree", "setup": mine, "second": rival, "final": verdict})
        self._g1b_take_jobs(verdict)

    def _g1b_settle_closing(self):
        """Cross-read the closing jobs, then up to two repair rounds; a job still at fault is dropped."""
        catalog = self._g1b_catalog()
        self._g1b_cross_read(catalog)
        for _ in range(2):
            issues = [issue for job in _CLOSING_JOBS for issue in self._g1b_faults(job)]
            if self.key_dry or not issues:
                break
            self.trace.append({"kind": "finish_repair", "issues": issues})
            prompt = (self._g1b_REPAIR_HEAD + "\n".join("- " + issue for issue in issues)
                      + "\n\nTools on offer:\n" + catalog)
            self._g1b_take_jobs(self._g1b_read_jobs(prompt, [INTAKE_SEAT, ALT_VENDOR_SEAT, PRIMARY_SEAT,
                                                             FALLBACK_SEAT]))
        # a clash faults both sides, so judge them all before removing any
        broken = [job for job in _CLOSING_JOBS if job in self.roles and self._g1b_faults(job)]
        for job in broken:
            del self.roles[job]
            self.fixed.pop(job, None)

    def _classify_call(self, label_name, arguments):
        """The closing job ('order' / 'decline') a call performs, or None."""
        sent = arguments or {}
        for job in _CLOSING_JOBS:
            if self.roles.get(job) != label_name:
                continue
            pins = self._g1b_pins(job)
            if all(sent.get(arg) == pins[arg] for arg in pins):
                return job
        return None

    def _roles_for_log(self):
        """The roles for the log, with the jobs this store cannot do."""
        logged = dict(self.roles)
        for job in _CLOSING_JOBS:
            if job in logged:
                logged[job] = {"tool": logged[job], "fixed": self._g1b_pins(job)}
        gaps = [job for job in ("search", "details", "cart", "order", "decline") if job not in self.roles]
        if self.roles.get("message") and self._text_arguments("?") is None:
            gaps.append("message: not every required argument accepts plain text")
        logged["gaps"] = gaps
        return logged

    # ================================================================ readings checked against the request

    def _g1b_quoted(self, quote):
        def squash(text):
            return " ".join(str(text or "").lower().split())
        needle = squash(quote)
        return bool(needle) and needle in squash(self.policy.get("query"))

    def _g1b_priced_ask(self, entry):
        """A priced-question reading counts only with a number and a quote really found in the request."""
        if not isinstance(entry, dict) or _as_number(entry.get("amount")) is None:
            return None
        return entry if self._g1b_quoted(entry.get("quote")) else None

    def _g1b_procedures(self, entries):
        """Steps the request names that need a tool call other than closing or carting; quotes are checked."""
        found = []
        for entry in entries if isinstance(entries, list) else ():
            if not isinstance(entry, dict) or not self._g1b_quoted(entry.get("quote")):
                continue
            role = entry.get("role")
            if role not in self.roles or role == "cart" or role in _CLOSING_JOBS:
                continue
            count = _as_number(entry.get("count"))
            before = entry.get("before")
            found.append({
                "do": str(entry.get("do") or entry.get("quote"))[:200],
                "quote": str(entry.get("quote"))[:200],
                "role": role,
                "count": 1 if count is None else max(1, int(count)),
                "after_pick": entry.get("after_pick") is True,
                "before": before if before in ("order", "decline") else "any"})
        return found

    def _pending_procedures(self, kind):
        """Stated steps still owed before closing with `kind`."""
        owed = []
        for proc in self.procedures:
            late = proc["after_pick"]
            if late and kind != "order":
                continue
            if proc["before"] != "any" and proc["before"] != kind:
                continue
            since = self.cart_changed_at if late else 0
            done = sum(1 for at in self.calls_by_role.get(proc["role"], []) if at > since)
            if done < proc["count"]:
                owed.append(proc)
        return owed

    # ================================================================ texts for the main model

    def _g1b_rank_line(self, plan):
        rank, detail = plan.get("priority"), plan.get("priority_detail")
        if rank not in ("cheapest", "best"):
            return ("- ranking: the request asks for neither cheapest nor best, so price comparisons waste steps; "
                    "barring other instructions in it, any item within budget that meets every requirement is right")
        tail = f" ({detail})" if detail else ""
        return f"- ranking: of the items that satisfy every condition, take the {rank}" + tail

    def opening_lines(self):
        """The setup reading as lines for the opening message ('' when there is none)."""
        plan = self.plan
        if not plan:
            return ""
        rows = ["How setup understood the request (compare each line with the shopper's wording before relying on it):"]
        if plan.get("product"):
            rows.append(f"- item sought: {plan.get('product')}")
        rows.append(f"- units wanted: {plan.get('quantity') or 1}")
        if plan.get("price_cap") is not None:
            rows.append(f"- most the whole purchase may cost: {plan.get('price_cap')} "
                        f"{plan.get('currency') or ''}".rstrip())
        if plan.get("in_stock_only"):
            rows.append("- only listings with stock on hand qualify")
        rows.append(self._g1b_rank_line(plan))
        rows.extend(f"- requirement as stated: {need}" for need in plan.get("requirements") or [])
        rows.extend(self._ejc_item_lines())
        for field, lead in _OPENING_SECTIONS:
            rows.extend(lead + str(value) for value in plan.get(field) or [])
        rows.append(f"- steps allowed: {self.max_steps}")
        if self.message_budget is not None:
            rows.append(f"- messages allowed: {self.message_budget}")
        return "\n".join(rows + self._closing_lines())

    def _closing_lines(self):
        """One line per closing job: its tool and the choice values it needs."""
        purpose = {"order": "to buy what the cart holds", "decline": "to finish without buying"}
        lines = []
        for job in _CLOSING_JOBS:
            tool = self.roles.get(job)
            if not tool:
                continue
            values = self._g1b_choice_values(tool, self._g1b_pins(job))
            lines.append(f"- {purpose[job]}: call {tool}" + (" with " + _to_compact_json(values) if values else ""))
        return lines

    _g1b_LAST = "TWO STEPS OR FEWER: order the checked match sitting in the cart right now; if none is there, decline."
    _g1b_WRAP_ONE = ("TIME TO CLOSE: the cart should hold a single checked line; re-read its stock, then order - "
                     "decline instead when nothing matches.")
    _g1b_WRAP_MANY = ("TIME TO CLOSE: the cart should hold, for every requested product, a single checked line at "
                      "that product's quantity; re-read their stock, then order - decline instead when nothing "
                      "matches.")
    _g1b_EARLY = ("Plan ahead: once 7 steps remain the search and detail tools disappear, and mending a refused "
                  "order eats roughly 4 steps - cart your strongest checked candidate and buy it soon.")

    def _g1b_pressure(self, left):
        if left <= 2:
            return self._g1b_LAST
        if left <= 5 or self.seconds_remaining() < WRAP_UP_WINDOW or self.cost > _eja_SPEND_WRAP:
            return self._g1b_WRAP_MANY if self._wanted_items() else self._g1b_WRAP_ONE
        if self.order_attempts == 0 and left <= 14:
            return self._g1b_EARLY
        return ""

    def status(self):
        left = self.remaining_steps()
        allowance = "" if self.message_budget is None else f" of {self.message_budget}"
        clock = int(max(0, self.seconds_remaining() - FORCE_MARGIN))
        text = (f"[status] {left} steps left ({self.step} of {self.max_steps} used); about {clock} s on the clock; "
                f"messages sent: {self.messages_used}{allowance}.\n[requirements now] " + self.requirement_summary())
        note = self._g1b_pressure(left)
        return text + " " + note if note else text

    def requirement_summary(self):
        """The requirements in force as one text, with notes where the store spells a value like the request."""
        self._pin_stated_values()
        units = "per product, see the list below" if self._wanted_items() else self._wanted_units()
        rows = [self._need_list.render_lines(self.plan.get("currency") or "", units)]
        rows.extend(self._ejc_item_lines())
        for cond in self._list_conditions():
            spelled = self._shop_phrasing(cond)
            if not spelled:
                continue
            options = " or ".join(f'"{phrase}"' for phrase in spelled)
            rows.append(f"- {self._ejb_cut(cond)[0].strip()}: numbers computed from values phrased some other way "
                        f"are no proof here; a listing must literally state {options}, the request's own phrase, "
                        f"since that is the spelling this store uses")
        for prop, wish in self.loose_ok.items():
            rows.append(f'- {prop}: satisfied by any listing whose {prop} includes the words "{wish}", extra '
                        f'wording or not - the shopper, offered the listed "{wish}" variants, did not care which')
        return "\n".join(rows)

    @staticmethod
    def _ejb_property_values(row_map, terms):
        """Values of each fact whose key names the property (holds all its words, or is named like it)."""
        for row in row_map:
            facts = row.get("facts") or {}
            for key, value in facts.items():
                if terms <= _word_set(key) or _key_names_property(key, terms):
                    yield value

    def _shop_phrasing(self, condition, row_map=None):
        """For a "covers N" condition: the store's values for that property that hold N and that the request
        repeats word for word. Such a phrase is the value meant; a number read from other wording does not count."""
        prop, colon, wanted = self._ejb_cut(condition)
        relation = _parse_relation(wanted) if colon else None
        terms = _word_set(prop)
        if not terms or relation is None or relation[0] != "covers":
            return []
        goal = relation[1]
        request = _flatten_text(self.policy.get("query"))
        rows = self._row_table() if row_map is None else row_map
        found = []
        for value in self._ejb_property_values(rows.values(), terms):
            for text in map(str, value if isinstance(value, list) else [value]):
                for part in [text] + re.split(r"[,;|]", text):
                    flat = _flatten_text(part)
                    if not flat or flat in found or flat not in request:
                        continue
                    if goal in [float(num) for num in re.findall(r"\d+(?:\.\d+)?", flat)]:
                        found.append(flat)
        return found

    # ================================================================ JSON judges and the shopper's lines

    @staticmethod
    def _g1b_parse(reply):
        return None if reply is None else _extract_json_dict(reply.get("content") or "")

    def ask_judge(self, prompt, max_tokens=700, models=None):
        """A JSON-text call with a single user message. A reply cut before the object closed is asked again with
        three times the room; cut again, the seats after the one that answered get the last try."""
        pool = models or [REVIEW_SEAT, ALT_VENDOR_SEAT, PRIMARY_SEAT]
        turn = [{"role": "user", "content": prompt}]
        room = max_tokens
        reply = self.infer(turn, None, pool, room)
        if reply is None:
            return None
        for escalation in ("roomier", "later seats"):
            parsed = self._g1b_parse(reply)
            if parsed is not None or reply is None or not reply.get("cut"):
                return parsed
            if escalation == "roomier":
                self.trace.append({"kind": "cut_reply", "limit": room})
                room = max_tokens * 3
            else:
                who = reply.get("model")
                pool = pool[pool.index(who) + 1:] if who in pool else []
                if not pool:
                    return None
                self.trace.append({"kind": "cut_reply", "limit": room, "next": pool[0]})
            reply = self.infer(turn, None, pool, room)
        return self._g1b_parse(reply)

    def _g1b_reading_prompt(self, new_lines, question):
        # the shopper's newest words always come last, after every earlier exchange
        parts = [UPDATE_READING_TEXT,
                 "Requirements in force so far:\n" + self.requirement_summary(),
                 "The original request, for background:\n" + str(self.policy.get("query") or "")]
        history = self._need_list.history
        if history:
            parts.append("Earlier exchanges, oldest first:\n" + "\n".join(history[-12:]))
        if question:
            parts.append("Question the assistant asked:\n" + question)
        parts.append("New words from the shopper:\n" + "\n".join(new_lines))
        return "\n\n".join(parts)

    def read_shopper_lines(self, text_lines, question):
        """Fold what new shopper lines mean into the wants; returns a note for the model ('' when none)."""
        state = self._need_list
        fresh = [line for line in text_lines if line not in state.heard]
        if not fresh:
            return ""
        state.heard.update(fresh)
        if self.seconds_remaining() < WRAP_UP_WINDOW:
            return ""
        prompt = self._g1b_reading_prompt(fresh, question)
        state.history.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        reading = self.ask_judge(prompt, 1500, [READER_SEAT, REVIEW_SEAT])
        if not reading:
            return ""
        if question:
            self.answer_reading = self._g1b_answer_class(reading.get("answer_class"), " ".join(fresh))
            if not self.asking:   # an open question frees no property
                reading["no_preference"] = []
        self._ejb_refile_needs(reading, fresh)
        if question:
            self._ejb_protect_stated(reading, fresh)
        self._ejb_restore_stated(reading, fresh)
        self._ejb_drop_invented(reading, fresh)
        before = list(state.revealed)
        changes = self._g1b_fold(reading)
        self._g1b_place_needs([need for need in state.revealed if need not in before], fresh)
        answer_back = str(reading.get("reply") or "").strip()
        if answer_back and not question and self.roles.get("message"):   # our own question wants no answer
            self.queued_reply = answer_back
        note = "Meaning: " + str(reading.get("meaning") or "")
        return note + " (" + "; ".join(changes) + ")" if changes else note

    def _g1b_place_needs(self, new_needs, text_lines):
        """With several products requested, file each new need under the one product the words name, else under
        the product the asked property belongs to; left unfiled when neither tells."""
        items = self._wanted_items()
        if not items or not new_needs:
            return
        products = [item["product"] for item in items]
        fallback = _ejd_owner(self, self.asking) if self.asking else ""
        for need in new_needs:
            heard = self._ejc_stems(" ".join(list(text_lines) + [need]))
            named = [prod for prod in products if self._ejc_stems(" ".join(_ejd_own_words(prod, products))) & heard]
            home = named[0] if len(named) == 1 else fallback
            if home in products:
                self.need_homes[need] = home

    def _g1b_fold(self, reading):
        state = self._need_list
        dropped_count, budget = len(state.dropped), state.budget
        changes = state.fold_reading(reading)
        if not changes:
            return changes
        self.needs_revision += 1
        shifted = len(state.dropped) != dropped_count or state.budget != budget
        if shifted and self.plan.get("priority") == "cheapest":
            self.rerank_reason = "; ".join(state.dropped[dropped_count:]) or "the budget"
        return changes

    _g1b_CLASSES = ("reveal", "not_picky", "split", "unclear", "other")

    def _g1b_answer_class(self, raw, said):
        """The answer class, trusted only when its quote sits in the answer and a reveal names the asked property,
        a listing property or a listing value; else 'other'."""
        if not isinstance(raw, dict) or raw.get("class") not in self._g1b_CLASSES:
            return {"class": "other"}
        fields = {name: str(raw.get(name) or "") for name in ("key", "value", "relation")}
        quote = str(raw.get("quote") or "").strip().lower()
        if not quote or quote not in said.lower():
            return {"class": "other"}
        if raw["class"] == "reveal" and not self._ejb_on_topic(fields["key"], fields["value"].strip().lower()):
            return {"class": "other"}
        return {"class": raw["class"], **fields}

    def _ejb_on_topic(self, key_text, datum):
        names, short_vals = self._ejb_sheet_words()
        prop = _word_set(key_text)
        if prop:
            if prop <= _word_set(self.asking):
                return True
            if any(prop <= one for one in names):
                return True
        return bool(datum) and datum in short_vals

    def _g1c_shaky(self, prop, names):
        """True when a need's property may be misfiled: it is the one we asked about, one the shopper waved off,
        one no listing carries, or one the request already stated."""
        asked = _word_set(self.asking)
        if asked and (prop <= asked or asked <= prop):
            return True
        for waved in self._need_list.free or []:
            words = _word_set(waved)
            if words and words == prop:
                return True
        if prop not in names:
            return True
        return any(self._ejb_key_words(old) == prop for old in self._need_list.stated or [])

    @staticmethod
    def _g1c_home_in_line(line, prop_name, value_words, names):
        """The listing property a shopper line puts beside the value, in the line's own words; None if absent."""
        heard = _word_set(line)
        if _word_set(prop_name) <= heard:
            return None
        if not _words_cover(value_words, heard):
            return None
        best = None
        for cand in names:
            if cand <= heard and not cand & value_words and (best is None or len(cand) > len(best)):
                best = cand
        if best is None:
            return None
        return " ".join(tok for tok in re.findall(r"[a-z0-9]+", line.lower()) if tok in best)

    def _g1c_retarget(self, entry, lines, names):
        if not isinstance(entry, str) or ":" not in entry:
            return None
        prop_name, rest = entry.split(":", 1)
        value_words = _word_set(rest) - _word_set(prop_name)
        if not value_words or not self._g1c_shaky(_word_set(prop_name), names):
            return None
        for line in lines:
            home = self._g1c_home_in_line(line, prop_name, value_words, names)
            if home is not None:
                return home + ":" + rest
        return None

    def _ejb_refile_needs(self, read_reply, text_lines):
        """Refile a fresh need under the listing property the shopper's line names, when the reader's choice of
        property is doubtful and that line holds the value."""
        names = [one for one in self._ejb_sheet_words()[0] if one]
        if not (names and isinstance(read_reply, dict)):
            return
        renamed = {}
        for entry in read_reply.get("needs") or []:
            new_entry = self._g1c_retarget(entry, text_lines, names)
            if new_entry is not None:
                renamed[entry] = new_entry
        if renamed:
            read_reply["needs"] = [renamed.get(ent, ent) for ent in read_reply.get("needs") or []]
            rebuilt = read_reply.get("requirements_now")
            if isinstance(rebuilt, list):
                read_reply["requirements_now"] = [renamed.get(ent, ent) if isinstance(ent, str) else ent
                                                  for ent in rebuilt]
            self.trace.append({"kind": "attributed", "moved": renamed})

    def _ejb_protect_stated(self, read_reply, _lines=None):
        """Answers may add needs, but nothing the original request stated gets dropped or freed by them."""
        if not isinstance(read_reply, dict):
            return
        stated_text = [str(ent) for ent in self._need_list.stated]
        stated_props = [self._ejb_key_words(ent) for ent in stated_text]

        def on_stated(gone):
            text = str(gone)
            if any(old and (old in text or text in old) for old in stated_text):
                return True
            prop = self._ejb_key_words(text)
            return bool(prop) and prop in stated_props

        pulled = self._g1c_shield(read_reply, on_stated, [p for p in stated_props if p])
        if pulled:
            self.trace.append({"kind": "stated_guarded", "entries": pulled})
        self._g1c_hold_revealed(read_reply)

    def _g1c_hold_revealed(self, read_reply):
        """An earlier revealed need gives way only to a new value for it; a mere wave-off leaves it in place."""
        state = self._need_list
        renewed = [self._ejb_key_words(ent) for ent in read_reply.get("needs") or [] if isinstance(ent, str)]
        held = []
        for ent in state.revealed:
            prop = self._ejb_key_words(ent) if ent not in state.dropped else None
            if prop and prop not in renewed:
                held.append(ent)
        if not held:
            return
        held_props = [self._ejb_key_words(ent) for ent in held]
        pulled = self._g1c_shield(
            read_reply, lambda gone: str(gone) in held or self._g1c_nested(self._ejb_key_words(gone), held_props),
            held_props)
        rebuilt = self._g1c_rebuilt(read_reply)
        if rebuilt is not None:
            present = [self._ejb_key_words(ent) for ent in rebuilt if isinstance(ent, str)]
            rebuilt.extend(ent for ent in held if self._ejb_key_words(ent) not in present)
        if pulled:
            self.trace.append({"kind": "revealed_guarded", "entries": pulled})

    @staticmethod
    def _g1c_shield(read_reply, guarded, homes):
        """Take guarded entries out of "dropped" and free words near the homes out of "no_preference";
        return the entries taken out."""
        pulled = [str(gone) for gone in read_reply.get("dropped") or [] if guarded(gone)]
        if pulled:
            read_reply["dropped"] = [gone for gone in read_reply.get("dropped") or [] if str(gone) not in pulled]
        waved = read_reply.get("no_preference")
        if isinstance(waved, list):
            read_reply["no_preference"] = [one for one in waved if not Episode._g1c_nested(_word_set(one), homes)]
        return pulled

    def _ejb_restore_stated(self, read_reply, text_lines):
        """Return a stated requirement the rebuilt list lost, unless it was dropped or the shopper spoke of it."""
        rebuilt = self._g1c_rebuilt(read_reply)
        if rebuilt is None:
            return
        heard = self._g1c_heard(text_lines)
        released = [str(gone) for gone in read_reply.get("dropped") or []] + list(self._need_list.dropped)
        for ent in self._need_list.stated:
            prop = self._ejb_key_words(ent)
            if not prop or ent in rebuilt:
                continue
            if any(gone and (ent in gone or gone in ent) for gone in released):
                continue
            if _words_cover(prop, heard):
                continue
            others = [one for one in rebuilt if not isinstance(one, str) or self._ejb_key_words(one) != prop]
            rebuilt[:] = others + [ent]
            self.trace.append({"kind": "kept_stated", "entry": ent})

    def _ejb_drop_invented(self, read_reply, text_lines):
        """Strip rebuilt entries on a new property that neither the line nor the reading's needs mention."""
        rebuilt = self._g1c_rebuilt(read_reply)
        if rebuilt is None:
            return
        state = self._need_list
        pool = (state.stated + state.revealed + list(state.newest_lines or [])
                + list(read_reply.get("needs") or []))
        known_props = [prop for prop in (self._ejb_key_words(ent) for ent in pool) if prop]
        heard = self._g1c_heard(text_lines)
        made_up = []
        for ent in rebuilt:
            if not isinstance(ent, str) or ":" not in ent:
                continue
            prop = self._ejb_key_words(ent)
            if not prop or any(prop & other for other in known_props):
                continue
            if any(_words_match(tok, said) for tok in _word_set(ent) for said in heard):
                continue
            made_up.append(ent)
        if made_up:
            rebuilt[:] = [ent for ent in rebuilt if ent not in made_up]
            self.trace.append({"kind": "unsaid_dropped", "entries": made_up})

    # ---- tying stated values to the store's own words

    @staticmethod
    def _g1c_swap_need(state, test, new):
        for group in (state.stated, state.revealed, state.newest_lines or []):
            group[:] = [new if isinstance(ent, str) and test(ent) else ent for ent in group]

    def _g1c_barred_values(self):
        """Property words -> values the stated requirements rule out for that property."""
        barred = {}
        for ent in self._need_list.stated:
            prop_name, sep, wanted = self._ejb_cut(ent)
            if not sep or "!=" not in wanted:
                continue
            slot = frozenset(_word_set(prop_name))
            barred[slot] = barred.get(slot, set()) | _word_set(wanted.partition("!=")[2])
        return barred

    def _pin_stated_values(self):
        """Pin a stated value to a store value. If no opened listing's value for the property sits in the
        constraint while the request uses exactly one of them (not excluded), the need becomes
        "property: value (old constraint)". The value has to share a word with the constraint, unless the
        constraint both words it differently and excludes another value."""
        if not (self.records and isinstance(self.policy, dict)):
            return
        opened = [row for row in self._row_table().values() if row["opened"]]
        if not opened:
            return
        self._g1c_move_to_named(opened)
        query_words = _word_set(self.policy.get("query"))
        barred = self._g1c_barred_values()
        for ent, head, prop, wanted in self._g1c_plain_stated():
            if "(" in wanted:
                continue   # an explained value is tied already
            pos, _, neg = wanted.partition("!=")
            value = self._g1c_one_store_value(opened, prop, pos, neg, query_words,
                                              barred.get(frozenset(prop), set()))
            if value is not None:
                self._g1c_commit("anchored", ent, f"{head.strip()}: {value} ({wanted.strip()})",
                                 lambda one, old=ent: one.startswith(old))

    def _g1c_plain_stated(self):
        """(entry, property text, property words, constraint) per stated need whose constraint is no number test."""
        for ent in list(self._need_list.stated):
            head, sep, tail = self._ejb_cut(ent)
            words = _word_set(head)
            if sep and words and _parse_relation(tail) is None:
                yield ent, head, words, tail

    def _g1c_commit(self, kind, old, new, test):
        self._g1c_swap_need(self._need_list, test, new)
        self.needs_revision += 1
        self.trace.append({"kind": kind, "from": old, "to": new})

    def _g1c_one_store_value(self, opened, prop, pos, neg, query_words, barred):
        parts = set()
        for raw in self._ejb_property_values(opened, prop):
            for piece in re.split(r"[,/;|]", str(raw)):
                if _word_set(piece):
                    parts.add(piece.strip().lower())
        want = _word_set(pos)
        if not parts or any(_word_set(piece) <= want for piece in parts):
            return None
        blocked = _word_set(neg) | barred
        rephrased = bool(neg.strip() and want)
        found = []
        for piece in parts:
            words = _word_set(piece)
            if words & blocked:
                continue
            if not all(any(_words_match(tok, q) for q in query_words) for tok in words):
                continue
            if want and _words_cover(want, words):
                continue   # it names all wanted words already; pinning would only refuse its siblings
            if rephrased or any(_words_match(tok, w) for tok in words for w in want):
                found.append(piece)
        return found[0] if len(found) == 1 else None

    def _g1c_move_to_named(self, opened):
        """Move a stated need to the property its own request sentence names, when listings with that property
        state exactly the wanted value and the sentence never names the property it was filed under."""
        query = str(self.policy.get("query") or "")
        sentence_words = [_word_set(part) for part in re.split(r"(?<=[.!?])\s+", query)]
        for ent, _, prop, wanted in self._g1c_plain_stated():
            value_words = _word_set(wanted.partition("!=")[0].partition("(")[0])
            homes = [ws for ws in sentence_words if value_words and _words_cover(value_words, ws)]
            if not homes or any(_words_cover(prop, ws) for ws in homes):
                continue
            spelled = self._g1c_other_keys(opened, prop, value_words, homes)
            least = min(spelled, key=len) if spelled else None
            if least is None or any(not least <= other for other in spelled):
                continue   # nothing found, or two distinct properties carry it
            self._g1c_commit("rekeyed", ent, f"{spelled[least]}: {wanted.strip()}",
                             lambda one, old=ent: one == old)

    def _g1c_other_keys(self, opened, prop, value_words, homes):
        """Other property keys stating the value and worded in the request sentence; shortest spelling each."""
        spelled = {}
        for row in opened:
            for key, raw in row["facts"].items():
                words = _word_set(key)
                if not words or words & prop:
                    continue   # the same property spelled another way
                if not any(_words_cover(words, ws) for ws in homes):
                    continue
                if value_words not in [_word_set(piece) for piece in re.split(r"[,/;|]", str(raw))]:
                    continue
                text = " ".join(self._ejb_label(key).split()).lower()
                slot = frozenset(words)
                if len(text) < len(spelled.get(slot, text + " ")):
                    spelled[slot] = text
        return spelled

    def _ejb_sheet_words(self):
        """Property-name words and short values from detail sheets (dicts with at least five short texts)."""
        names = []
        short_vals = set()
        for blob in _iter_dicts(self.records):
            pairs = self._ejb_short_texts(blob, 70)
            if len(pairs) < 5:
                continue
            for key, text in pairs:
                short_vals.add(text.strip().lower())
                words = _word_set(self._ejb_label(key))
                if words and words not in names:
                    names.append(words)
        return names, short_vals

    def records_naming(self, unit_ref):
        """Up to four of the latest distinct records that mention this id."""
        texts = (_to_compact_json(_shorten_value(blob, 400))[:2500]
                 for record in self.records for blob in self._g1c_holders(record, unit_ref))
        unique = []
        for text in texts:
            if text not in unique:
                unique.append(text)
                if len(unique) == 8:
                    break
        return unique[-4:]

    # ---- second opinions and summaries of what the store showed

    def _verify_item(self, unit_ref, intent):
        """A second judge on one listing: a warning, or '' when it passes or no check can run."""
        if self.seconds_remaining() < WRAP_UP_WINDOW + 20 or self.spend_exceeded():
            return ""
        slot = "%s|%s" % (unit_ref, self._arm_signature())
        known = self.verdicts.get(slot)
        if known is None:
            proof = self.records_naming(unit_ref)
            if proof:
                known = self._g1c_second_check(unit_ref, intent, proof)
                self.verdicts[slot] = known
        return known if known else ""

    def _g1c_second_check(self, unit_ref, intent, proof):
        needs_now = self.requirement_summary()
        cart_text = None if self.last_cart_view is None else _to_compact_json(self.last_cart_view)[:3000]
        unprovable = self._unspoken_conditions()
        common = self._repeated_values()
        parts = [ITEM_CHECK_TEXT,
                 f"Candidate being checked, for the step '{intent}': {unit_ref}" + self._ejc_item_scope(unit_ref),
                 "Requirements as they stand now (the request plus every later answer or change):\n" + needs_now,
                 "The shopper's request, word for word:\n" + str(self.policy.get("query") or ""),
                 "All store records that mention this candidate, oldest first:\n" + "\n".join(proof)]
        if cart_text is not None:
            parts.append("Most recent view of the cart:\n" + cart_text)
        if unprovable:
            parts.append("No opened listing here mentions these conditions at all, in a key or in any text, so no "
                         "listing could ever confirm them: " + "; ".join(unprovable) + ". They alone are no reason "
                         "to turn the candidate down; judge it by the other requirements.")
        if common:
            parts.append("Values shared by several listings, with how often each occurs. Take the one that equals "
                         "the request's full wording for this candidate, not a neighbouring one:\n" + common)
        result = self.ask_judge("\n\n".join(parts), 1200)
        self.trace.append({"kind": "verify", "item": unit_ref, "result": result})
        if not result or result.get("fits") is not False:
            return ""
        reasons = "; ".join([str(one) for one in result.get("problems") or []][:5])
        return (f"The independent check of {unit_ref} did not pass: {reasons}. Pick another listing that is fully "
                "confirmed instead, unless you are certain this check is mistaken.")

    def _ejb_allowed_jobs(self):
        """The jobs whose tools stay on offer as steps run out (None: no narrowing)."""
        left = self.remaining_steps()
        for limit, jobs in self._g1c_LAST_STEPS:
            if left <= limit:
                return jobs
        if left > 7 or (self.order_attempts and not self.refusals):
            return None
        if self.refusals:
            # after a refusal, a cheaper listing seen but never opened keeps the details tool
            want_details = _rows_to_recheck(self)
        else:
            # before any order: details stay while on-target rows wait unopened and steps exceed reserve + 2
            want_details = left >= self._reserved_steps() + 2 and _eje_unfitted_rows(self, 1)
        return self._g1c_LATE_JOBS + ("details",) if want_details else self._g1c_LATE_JOBS

    def _tools_this_turn(self):
        """Tools offered this turn; the message tool stays hidden while the planner writes the questions."""
        role = self.roles.get
        menu = self.tools
        jobs = self._ejb_allowed_jobs()
        if jobs is not None and role("order") and role("cart"):
            allowed = {role(job) for job in jobs}
            menu = [spec for spec in menu if spec["function"]["name"] in allowed] or self.tools
        hidden = role("message") if self.planner_on else None
        if not hidden:
            return menu
        return [spec for spec in menu if spec["function"]["name"] != hidden]

    def _ejb_menu_sheets(self):
        """Detail sheets as label -> lower-case value (search rows, and fields seen in two record forms, left out)."""
        blobs = list(_iter_dicts(self.records))
        ids = self.seen_ids or set()
        twice = _own_listing_fields(blobs, ids)
        rows = _search_row_ids(self.records, ids)
        parent, own = self._g1c_nesting(blobs, ids)
        sheets = []
        context = []
        for blob in blobs:
            if id(blob) in rows:
                continue
            holder = next((val for val in blob.values() if _looks_like_listing(val, ids)), None)
            pairs = [(key, val) for key, val in self._ejb_short_texts(blob, 70) if (holder, key, val) not in twice]
            if len(pairs) < (3 if id(blob) in own else 5):
                continue
            page = {}
            for key, val in pairs:
                label = self._ejb_label(key).strip().lower()
                if label not in page and not _looks_like_code(val, True):
                    page[label] = val.lower()
            sheets.append(page)
            near = (blob, parent.get(id(blob)) or {})
            context.append({str(val).strip().lower() for one in near for val in one.values() if isinstance(val, str)})
        self._g1c_sheet_ctx = context
        return sheets

    def _g1c_sections_only(self, sheets):
        """Keep sheets of the requested products' own sections: a sheet naming some other section (a reference
        page, a stray search) is left out, even before any sheet of the wanted sections arrives, as its properties
        belong to another product. All sheets stay while a requested product shares no stem with a seen section."""
        products = [one["product"] for one in self._wanted_items()] or [(self.plan or {}).get("product") or ""]
        sections = {row["label"].strip().lower() for row in self._row_table().values()
                    if isinstance(row.get("label"), str)}
        wanted = set()
        for product in products:
            stems = self._ejc_stems(product)
            mine = {sec for sec in sections if stems & self._ejc_stems(sec)}
            if not mine:
                return sheets
            wanted.update(mine)
        ctx = getattr(self, "_g1c_sheet_ctx", None)
        if not isinstance(ctx, list) or len(ctx) != len(sheets):
            ctx = [set(page.values()) for page in sheets]
        kept = []
        for page, texts in zip(sheets, ctx):
            named = texts & sections
            if not named or named & wanted:
                kept.append(page)
        return sheets if len(kept) == len(sheets) else kept

    def _build_property_menu(self):
        """The property menu from listing details (counts, samples, other spellings) and the sheet count.
        Sets self.property_menu for the planner."""
        sheets = self._g1c_sections_only(self._ejb_menu_sheets())
        values_of = {}
        for page in sheets:
            for label, low in page.items():
                values_of.setdefault(label, []).append(low)
        count = {label: len(vals) for label, vals in values_of.items()}
        kinds = {label: list(dict.fromkeys(vals)) for label, vals in values_of.items()}
        root_of = self._g1c_spelling_roots(sheets, count)
        spellings = {}
        for label, root in root_of.items():
            if root != label:
                spellings.setdefault(root, []).append(label)
        product = _word_set((self.plan or {}).get("product"))
        same_all = {label for label in count if count[label] >= 3 and len(kinds[label]) == 1}

        def own_name(label):   # values mostly repeat the item's own name
            if not product:
                return False
            named = sum(1 for val in values_of[label] if product & _word_set(val))
            return 2 * named > len(values_of[label])

        chosen = [label for label in count if root_of[label] == label and not own_name(label)]
        chosen.sort(key=lambda label: (-count[label], label in same_all, label))
        chosen = chosen[:28]
        self.property_menu = {}
        lines = []
        for label in chosen:
            others = sorted(spellings.get(label, []))
            self.property_menu[label] = {"count": count[label], "values": sorted(kinds[label]),
                                         "uniform": label in same_all, "also": others}
            note = " (same value everywhere)" if label in same_all else ""
            if others:
                note += " [also spelled: " + ", ".join(others) + "]"
            lines.append(f"* {label} - on {count[label]} listings{note}; examples: " + " / ".join(kinds[label][:5]))
        return "\n".join(lines), len(sheets)

    def _repeated_values(self):
        """Short values that repeat under one key across listing rows, with counts (twelve keys at most)."""
        tally = {}
        for blob in _iter_dicts(self.records):
            if len(blob) < 4:
                continue
            for key, text in self._ejb_short_texts(blob, 40):
                if not _looks_like_code(text, False):
                    seen = tally.setdefault(str(key), {})
                    seen[text] = seen.get(text, 0) + 1
        lines = []
        for key, seen in tally.items():
            if not (2 <= len(seen) <= 8) or max(seen.values()) < 2 or sum(seen.values()) < 4:
                continue
            ranked = sorted(seen.items(), key=lambda pair: -pair[1])
            lines.append(f"* {key}: " + ", ".join(f"{text} x{times}" for text, times in ranked))
        return "\n".join(lines[:12])

    def _call_shape(self, label_name, arguments):
        """Shape of a call: tool, argument names, choice values."""
        names = ",".join(sorted(arguments))
        return f"{label_name}|{names}|{self._choice_signature(label_name, arguments)}"

    def _detect_error(self, label_name, seen):
        """(text, sure) for an error result, else ('', False). An "error" field is a sure fault; a lone text field
        on a non-message call is a possible one (it counts once the identical call gets it again)."""
        if not seen or not isinstance(seen, dict):
            return "", False
        if "error" in seen and "refusal" not in seen:
            head = dict(list(seen.items())[:2])
            return _to_compact_json(head)[:200], True
        if len(seen) == 1 and label_name != self.roles.get("message"):
            (key, val), = seen.items()
            if isinstance(val, str) and val.strip() and val not in self.seen_ids:
                return f"{key}: {val}"[:200], False
        return "", False

    def _no_hits_note(self, label_name, arguments, nested):
        """For a search that returned nothing while choice arguments were left out (and no words typed): a hint
        that this mode may need one of them."""
        found = set()
        _gather_ids(nested, found)
        if found or self._g1c_any_rows(nested):
            return ""
        props = self._ejb_props(label_name)
        typed = any(isinstance(val, str) and val.strip() and not _is_enum_spec(props.get(arg))
                    for arg, val in arguments.items())
        if typed:
            return ""   # words were searched: a plain miss
        missing = []
        for key, spec in props.items():
            if key not in arguments and isinstance(spec, dict) and spec.get("enum"):
                missing.append("%s (e.g. %s)" % (key, ", ".join(map(str, spec["enum"][:4]))))
        if not missing:
            return ""
        return ("Nothing came back. This call left out the options " + "; ".join(missing[:3]) + ". Certain search "
                "modes stay empty until one of these is set, so an empty result does not show the store lacks the item.")

    def _requested_units(self, label_name, arguments):
        """Units a cart call asks for: its free integer argument (or that argument's default), otherwise 1."""
        props = self._ejb_props(label_name, True)
        for key, spec in props.items():
            if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec:
                units = arguments[key] if key in arguments else spec.get("default", 1)
                return units if isinstance(units, int) and not isinstance(units, bool) else 1
        return 1

    def _parse_cart_units(self, seen):
        """Units per held listing from a cart view; the units field is learnt from the first line whose single
        matching integer equals the units we sent."""
        self.cart_seen_at = self.step
        per_item = {}
        for blob in _iter_dicts([seen]):
            item = next((val for val in blob.values() if isinstance(val, str) and val in self.sent_units), None)
            if item is None:
                continue
            ints = self._ejb_ints(blob)
            if self.quantity_field is None:
                same = [key for key, num in ints.items() if num == self.sent_units[item]]
                if len(same) == 1:
                    self.quantity_field = same[0]
            if self.quantity_field in ints:
                per_item[item] = ints[self.quantity_field]
        if self.quantity_field is not None:
            self.units_in_cart = per_item

    def _choice_signature(self, label_name, arguments):
        """The enum-valued arguments of a call as text (it tells an add from a remove)."""
        props = self._ejb_props(label_name, True)
        picked = {}
        for arg, val in arguments.items():
            spec = props.get(arg)
            if isinstance(spec, dict) and "enum" in spec:
                picked[arg] = val
        return _to_compact_json(picked)

    # ==== messages to the shopper ==========================================

    def _message_limit(self):
        """Messages the shopper accepts; three unless the request states a number."""
        return 3 if self.message_budget is None else self.message_budget

    def _text_arguments(self, text):
        """Arguments for the message tool holding just `text`; None when the tool also needs something else."""
        schema = self.param_schemas.get(self.roles.get("message") or "") or {}
        needed = set(schema.get("required") or [])
        fields = _ejc_text_fields(schema.get("properties") or {})
        target = next((name for name in schema.get("required") or [] if name in fields), None)
        if not target and len(fields) == 1:
            target = fields[0]
        if not target or needed - {target}:
            return None
        return {target: text}

    def _build_message_call(self, text, call_id):
        """A ready tool call sending `text` to the shopper, or None when it fails the schema."""
        args = self._text_arguments(text)
        if not args:
            return None
        tool = self.roles.get("message")
        args, issues = _validate_value(args, self.param_schemas[tool], "args")
        if issues:
            return None
        return {"id": call_id, "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}

    def _record_replies(self, replies):
        """Put tool replies into the conversation, with the status line on the last one; return the plain texts."""
        texts = []
        final = len(replies) - 1
        for pos, pair in enumerate(replies):
            texts.append(pair[1])
            content = pair[1] if pos != final else pair[1] + "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": pair[0], "content": content})
        return texts

    def _run_question_planner(self, force=False):
        """Run the question planner; True when it sent something."""
        outcome = _eje_ask(self, {"force": force})
        return outcome["exit"]

    def _send_answer(self, text):
        """Send the reply the reading drafted for a question the shopper asked us."""
        call = self._g1c_can_reply() and self._build_message_call(text, f"reply_{self.step}")
        if not call:
            return False
        self.messages.append(dict(role="assistant", tool_calls=[call], content="Sending my answer to their question."))
        sent = self._run_calls([call], planned=True)
        self._record_replies(sent)
        return True

    def _g1c_can_reply(self):
        """Room for one more message: the tool exists, the shopper's limit is not reached, steps and time remain."""
        if not self.roles.get("message"):
            return False
        room = self.remaining_steps() - _spare_steps_for(self.max_steps) if self.messages_used < self._message_limit() else 0
        return room > 0 and self.seconds_remaining() >= WRAP_UP_WINDOW

    # ==== units and the final audit ========================================

    def _units_problem(self, label_name, arguments):
        """Why a cart call's unit count is off, or '' when it matches (or the call is no add)."""
        wanted = self._wanted_units()
        if self.add_signature is not None and self._choice_signature(label_name, arguments) != self.add_signature:
            return ""
        if self._wanted_items():
            ids = [val for val in arguments.values() if isinstance(val, str) and val in self.seen_ids]
            wanted = self._ejc_line_units(ids[0]) if len(ids) == 1 else None
            if wanted is None:
                return ""
        props = self.param_schemas[label_name].get("properties") or {}
        for key, spec in props.items():
            if not isinstance(spec, dict) or spec.get("type") != "integer" or "enum" in spec:
                continue
            sent = arguments[key] if key in arguments else spec.get("default", 1)
            if isinstance(sent, int) and sent != wanted:
                return (f"unit count mismatch: this call sends {sent}, but the request wants {wanted} of a single "
                        "listing.")
        return ""

    def _audit_before_decline(self):
        """One judge pass over every record before the first decline; a hold text naming possible fits, or ''."""
        if self.audit_done:
            return ""
        if self.spend_exceeded() or self.seconds_remaining() < WRAP_UP_WINDOW + 30:
            return ""
        self.audit_done = True
        latest = _ejc_newest_within(self.records, 24000)
        if not latest:
            return ""
        sections = [("CURRENT REQUIREMENTS", self.requirement_summary()),
                    ("ORIGINAL REQUEST TEXT", str(self.policy.get("query") or "")),
                    ("RECORDS RECEIVED (NEWEST FIRST)", "\n".join(latest))]
        prompt = "\n\n".join([DECLINE_AUDIT_TEXT] + ["%s:\n%s" % pair for pair in sections])
        verdict = self.ask_judge(prompt, 1200)
        self.trace.append({"kind": "audit", "result": verdict})
        raw = (verdict or {}).get("possible_fits") or []
        named = [str(one) for one in raw if isinstance(one, (str, int))][:5]
        if not named:
            return ""
        why = verdict.get("reason")
        because = " (%s)" % why if why else ""
        return (f"Wait before declining: re-reading all records, these listings may satisfy every condition{because}: "
                f"{', '.join(named)}. Open each one for its sheet, stock and price, then buy whichever qualifies; a "
                "decline is right only once all of them are excluded.")

    # ---- small shared helpers of this part

    _g1c_LAST_STEPS = ((1, ("order", "decline")), (3, ("order", "decline", "cart")))
    _g1c_LATE_JOBS = ("order", "decline", "cart", "cart_view", "view")

    @staticmethod
    def _g1c_nested(words, homes):
        """Some home holds the words or sits inside them."""
        return bool(words) and any(words <= home or home <= words for home in homes)

    @staticmethod
    def _g1c_rebuilt(read_reply):
        rebuilt = read_reply.get("requirements_now") if isinstance(read_reply, dict) else None
        return rebuilt if isinstance(rebuilt, list) else None

    @staticmethod
    def _g1c_heard(lines):
        heard = set()
        for line in lines or []:
            heard.update(_word_set(line))
        return heard

    @staticmethod
    def _g1c_holders(root, unit_ref):
        """Dicts holding the id as a value, depth first; no descent inside a match or below level six."""
        stack = [(root, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > 6:
                continue
            if isinstance(node, dict):
                if any(isinstance(val, str) and val == unit_ref for val in node.values()):
                    yield node
                    continue
                kids = list(node.values())
            elif isinstance(node, list):
                kids = node
            else:
                continue
            stack.extend((kid, depth + 1) for kid in reversed(kids))

    @staticmethod
    def _g1c_nesting(blobs, ids):
        """(nested dict id -> its holder, ids of a listing's own nested dicts)."""
        parent = {}
        own = set()
        for blob in blobs:
            inner = [val for val in blob.values() if isinstance(val, dict)]
            for val in inner:
                parent[id(val)] = blob
            if any(_looks_like_listing(val, ids) for val in blob.values()):
                own.update(id(val) for val in inner)
        return parent, own

    @staticmethod
    def _g1c_spelling_roots(sheets, count):
        """label -> the label it respells: a shared stem and equal values on every sheet that has both."""
        order = sorted(count, key=lambda label: (-count[label], label))
        roots = []   # filled in rank order, so already sorted
        root_of = {}
        for label in order:
            for main in roots:
                both = [page for page in sheets if label in page and main in page]
                if both and _name_stems(label) & _name_stems(main) and all(page[label] == page[main] for page in both):
                    root_of[label] = main
                    break
            else:
                roots.append(label)
                root_of[label] = label
        return root_of

    @staticmethod
    def _g1c_any_rows(node):
        """A non-empty list reachable through dicts."""
        if isinstance(node, list):
            return bool(node)
        if isinstance(node, dict):
            return any(Episode._g1c_any_rows(val) for val in node.values())
        return False

    # ==== steps, epochs, units =============================================

    # ---- step reserve and knowledge tag ----

    def _reserved_steps(self):
        """Steps kept back for a hold, the finish, one refused finish and a re-read."""
        return _spare_steps_for(self.max_steps)

    def _has_step_room(self, cost):
        """An optional action may spend `cost` steps only above the reserve."""
        return self.remaining_steps() - self._reserved_steps() >= cost

    def _arm_signature(self):
        """Changes whenever the needs or the cart change."""
        return "{}.{}".format(self.needs_revision, self.cart_generation)

    def _advance_generation(self):
        self.cart_generation = self.cart_generation + 1

    # ---- what the request asks for ----

    def _wanted_units(self):
        """Units to buy; a count above one needs the request's own words behind it."""
        n = self.plan.get("quantity")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            return 1
        if n > 1 and "quantity_quote" in self.plan and not self._g1d_quote_holds(self.plan.get("quantity_quote")):
            return 1
        return n

    def _g1d_quote_holds(self, quote):
        """The quoted count occurs in the request and has a digit or a product word."""
        bare = quote.strip() if isinstance(quote, str) else ""
        if not bare or bare.lower() not in str(self.policy.get("query") or "").lower():
            return False
        has_digit = re.search(r"\d", quote) is not None
        return has_digit or bool(_word_set(quote) & _word_set(self.plan.get("product")))

    def _wanted_items(self):
        """Requested products (product, quantity, requirements) when the request names two or more; else []."""
        source = self.plan.get("lines") if isinstance(self.plan, dict) else None
        found = [_g1d_line_entry(row) for row in (source if isinstance(source, list) else [])]
        found = [entry for entry in found if entry is not None]
        return found if len(found) > 1 else []

    @staticmethod
    def _ejc_stems(text):
        words = re.findall(r"[a-z0-9]+", str(text).lower())
        return set(w.rstrip("s")[:6] for w in words if len(w) >= 3)

    def _g1d_store_stems(self, unit_ref):
        pool = set()
        for text in _g1d_texts_beside(self.records, unit_ref):
            pool.update(self._ejc_stems(text))
        return pool

    def _ejc_item_index(self, unit_ref):
        """Index of the requested product this listing is (product words count double); None on a tie or no hit."""
        many = self._wanted_items()
        if unit_ref is None or not many:
            return None
        seen = self._g1d_store_stems(unit_ref)

        def score(entry):
            value_stems = set()
            for need in entry["requirements"]:
                value_stems |= self._ejc_stems(need.partition(":")[2])
            return 2 * len(self._ejc_stems(entry["product"]) & seen) + len(value_stems & seen)

        board = [score(entry) for entry in many]
        best = max(board)
        if best <= 0 or board.count(best) != 1:
            return None
        return board.index(best)

    def _ejc_line_units(self, unit_ref):
        """Units wanted for this listing's product; None when unknown."""
        k = self._ejc_item_index(unit_ref)
        if k is None:
            return None
        return self._wanted_items()[k]["quantity"]

    def _g1d_filed_under(self, product):
        """Revealed needs still in force filed under this product."""
        dropped = self._need_list.dropped
        return [need for need, home in self.need_homes.items() if home == product and need not in dropped]

    def _g1d_foreign_needs(self, unit_ref):
        """Needs that belong only to the other requested products (empty set when unknown)."""
        many = self._wanted_items()
        k = self._ejc_item_index(unit_ref)
        if k is None:
            return set()
        mine = many[k]
        others = many[:k] + many[k + 1:]
        result = set()
        for entry in others:
            result.update(need for need in entry["requirements"] if need not in mine["requirements"])
        home_keys = [self._ejb_key_words(need) for need in self._g1d_filed_under(mine["product"])]
        elsewhere = [need for entry in others for need in self._g1d_filed_under(entry["product"])]
        result.update(elsewhere)
        stray = [kw for kw in map(self._ejb_key_words, elsewhere) if kw not in home_keys]
        result.update(c for c in self._list_conditions() if self._ejb_key_words(c) in stray)
        return result

    def _g1d_needs_text(self, entry):
        extra = ["%s (revealed by the shopper)" % need for need in self._g1d_filed_under(entry["product"])
                 if need not in entry["requirements"]]
        joined = "; ".join(entry["requirements"] + extra)
        return joined if joined else "see the request"

    def _ejc_item_lines(self):
        many = self._wanted_items()
        if not many:
            return []
        head = "- The request names %d products; each gets its own cart line at its own quantity:" % len(many)
        return [head] + ["  product %d: %s, quantity %d, requirements: %s"
                         % (n, entry["product"], entry["quantity"], self._g1d_needs_text(entry))
                         for n, entry in enumerate(many, 1)]

    def _ejc_item_scope(self, unit_ref):
        """Which requested product a listing is judged as ('' for a one-product request)."""
        many = self._wanted_items()
        if not many:
            return ""
        k = self._ejc_item_index(unit_ref)
        if k is None:
            return (" (several products are requested; hold this listing only to the requirements of the product "
                    "it actually is, never to those of the others)")
        entry = many[k]
        return (" (it is requested product %d: %s at quantity %d; only its own requirements count: %s)"
                % (k + 1, entry["product"], entry["quantity"], self._g1d_needs_text(entry)))

    def _ejc_missing_items(self):
        """(requested products without a cart line, lines held); a cart view newer than the last change adds its
        known listings. No products when it cannot be told."""
        wanted = self._wanted_items()
        held = list(self.cart_ids)
        view = self.last_cart_view
        if view is not None and not self.cart_seen_at < self.cart_changed_at:
            shown = [ref for ref in _eje_view_listings(self, view) if ref not in held]
            held += shown
        if wanted and held:
            matched = list(map(self._ejc_item_index, held))
            if len(held) < len(wanted) or None not in matched:
                return [p for k, p in enumerate(wanted) if k not in matched], len(held)
        return [], len(held)

    # ---- listing table and literal checks ----

    def _row_table(self):
        """Every listing seen, by id (later records win). Price, stock and label keys are found by value type;
        a record with five or more short texts marks its listing opened and supplies its facts."""
        nodes = _g1d_listing_nodes(self.records)
        slots = _g1d_field_roles(nodes)
        table = {}
        for ident, node in nodes:
            row = table.setdefault(ident, {"price": None, "stock": None, "label": None, "facts": {}, "opened": False})
            for slot, key, test in slots:
                if test(node.get(key)):
                    row[slot] = node[key]
            sheet = _g1d_sheet(node, ident)
            if sum(isinstance(v, str) for v in sheet.values()) >= 5:
                row["facts"].update(sheet)
                row["opened"] = True
        return table

    def _list_conditions(self):
        """Requirements in "property: constraint" form; anything else is for the money check and the reviewer."""
        self._pin_stated_values()
        book = self._need_list
        if book.newest_lines is not None:
            pool = list(book.newest_lines)
        else:
            pool = [need for need in book.stated + book.revealed if need not in book.dropped]

        def two_sided(line):
            head, _, tail = str(line).partition(":")
            return bool(head.strip()) and bool(tail.strip())

        return [line for line in pool if two_sided(line)]

    def _within_price_cap(self, listing_row):
        """Not out of stock, and price times units within the budget when both are known."""
        limit = _as_number(self._need_list.budget)
        cost_each = listing_row.get("price")
        if listing_row.get("stock") is False:
            return False
        if None in (limit, cost_each):
            return True
        products = self._wanted_items()   # several products: the smallest count stands in
        count = self._wanted_units() if not products else min(p["quantity"] for p in products)
        return limit >= cost_each * count

    def _unspoken_conditions(self, row_map=None):
        """Conditions no opened listing speaks to (no key has the property words, no text has the wanted words);
        counted only once three listings are opened."""
        table = self._row_table() if row_map is None else row_map
        sheets = [row["facts"] for row in table.values() if row["opened"]]
        if len(sheets) < 3:
            return []
        key_words = [_word_set(key) for sheet in sheets for key in sheet]
        value_words = [_g1d_said(sheet) for sheet in sheets]

        def spoken(prop, want):
            if any(prop <= kw for kw in key_words):
                return True
            return bool(want) and any(want <= vw for vw in value_words)

        quiet = []
        for line in self._list_conditions():
            parts = _g1d_parse(line)
            if parts is not None and parts[0] and not spoken(*parts):
                quiet.append(line)
        return quiet

    def _g1d_verdicts(self, row):
        """(condition, literal verdict) pairs, leaving out conditions no opened listing states."""
        skip = set(self._unspoken_conditions())
        sheet = row.get("facts") or {}
        return [(c, _literal_match(sheet, c, self._shop_phrasing(c))) for c in self._list_conditions() if c not in skip]

    def _row_passes_literal(self, listing_row):
        """False when a condition fails by its words, None when one is open, else True; a free text holding every
        wanted word counts for a property with no key."""
        sheet = listing_row.get("facts") or {}
        undecided = False
        for line, verdict in self._g1d_verdicts(listing_row):
            if verdict is None and _ejc_text_states(sheet, line):
                verdict = True
            if verdict is False:
                return False
            undecided = undecided or verdict is None
        return None if undecided else True

    def _missed_conditions(self, unit_ref, row_map):
        """Conditions the listing misses: a literal failure, or a property it never names while its section does."""
        row = row_map.get(unit_ref) or {}
        sheet = row.get("facts") or {}
        label = row.get("label")
        peers = [r for r in row_map.values() if r["opened"] and (label is None or r["label"] == label)]
        words = _g1d_said(sheet)
        skip = self._g1d_foreign_needs(unit_ref)
        missed = []
        for line in self._list_conditions():
            if line in skip:
                continue
            verdict = _literal_match(sheet, line, self._shop_phrasing(line, row_map))
            if verdict is False or (verdict is None and self._g1d_quiet_here(row, line, words, peers)):
                missed.append(line)
        return missed

    def _g1d_quiet_here(self, row, line, words, peers):
        """The listing is silent on the property while its peers state it (or one of them passes)."""
        prop, want = _g1d_parse(line) or (set(), set())
        if not (prop and want) or want <= words:
            return False
        for key in (row.get("facts") or {}):
            kw = _word_set(key)
            if prop <= kw or _key_names_property(key, prop) or (kw and kw < prop):
                return False   # named, or partly named: an open value is the reviewer's call
        voiced = [r for r in peers if any(prop <= _word_set(k) for k in r["facts"])]
        if len(voiced) >= 2 and len(voiced) * 2 >= len(peers):
            return True
        return any(r is not row and self._within_price_cap(r) and _literal_match(r["facts"], line) is True
                   for r in voiced)

    def _request_sentence_for(self, prop_name):
        """First request sentence holding every word of the property name, or ''."""
        words = _word_set(prop_name)
        request = str(self.policy.get("query") or "")
        for piece in (re.split(r"(?<=[.!?])\s+", request) if words else ()):
            if words.issubset(_word_set(piece)):
                return piece.strip()[:240]
        return ""

    # ---- evidence gathered before a decline ----

    def _decline_evidence(self):
        """Facts against a decline: reviewed full fits, near fits (one re-look each), per-condition support and
        affordable listings never opened. '' when there is nothing."""
        table = self._row_table()
        priced = {idx: row for idx, row in table.items() if row["opened"] and self._within_price_cap(row)}
        verdicts = {idx: self._g1d_verdicts(row) for idx, row in priced.items()}
        clean = [idx for idx, pairs in verdicts.items() if all(v is True for _, v in pairs)]
        passed = self._g1d_review_full(clean)
        open_notes = self._g1d_review_near(table, verdicts, clean, passed)
        conditions = [c for c, _ in next(iter(verdicts.values()), [])]
        unseen = self._g1d_unseen(table, priced, conditions)
        self.proven_fits = list(passed)
        if not (passed or open_notes or unseen):
            return ""
        notes = []
        if passed:
            notes.append("by their own stated facts every requirement is met by " + ", ".join(passed)
                         + " - a valid purchase scores while a decline with any of them on offer earns nothing, "
                         "so buy the strongest of them")
        if open_notes:
            notes.append("these opened listings break no requirement, yet each still has one unproven detail: "
                         + "; ".join(open_notes) + " - open each one's sheet (title and variant included) to settle "
                         "it, and buy one where it holds")
        if unseen:
            notes.append("priced within the limit but never opened: " + ", ".join(unseen)
                         + " - open them first" + self._ejc_read_by_view())
        if conditions and not passed:
            support = ["'%s': %s" % (c, ", ".join([idx for idx in priced if (c, True) in verdicts[idx]][:3])
                                     or "confirmed nowhere") for c in conditions]
            notes.append("confirmation so far, requirement by requirement: " + "; ".join(support))
        return "Hold on before declining: " + "; ".join(notes) + "."

    def _g1d_review_full(self, clean):
        """Up to three full fits the reviewer accepts, cart items first."""
        queue = [idx for idx in self.cart_ids if idx in clean] + [idx for idx in clean if idx not in self.cart_ids]
        passed = []
        for idx in queue:
            if len(passed) == 3:
                break
            if idx in passed:
                continue
            if not self._verify_item(idx, "matches all, verified ahead of declining"):
                passed.append(idx)
        return passed

    def _g1d_review_near(self, table, verdicts, clean, passed):
        """Near fits (no failure, few open points): two get one re-look; a passing re-look joins `passed`, the
        others come back as notes naming what stays open."""
        def unsettled(idx):
            return [c for c, v in verdicts[idx] if v is None]

        def near(idx):
            pairs = verdicts[idx]
            return (idx not in clean and pairs and not any(v is False for _, v in pairs)
                    and len(unsettled(idx)) <= max(1, len(pairs) // 2))

        def order_key(idx):
            known = table[idx]["price"]
            return len(unsettled(idx)), (known if known is not None else 1e18), idx

        queue = sorted(filter(near, verdicts), key=order_key)
        notes = []
        relooks = 2
        for idx in queue:
            if len(passed) >= 3:
                break
            if relooks > 0 and idx not in self.relooked:
                self.relooked.add(idx)
                relooks -= 1
                if not self._verify_item(idx, "almost matching; rechecked ahead of declining"):
                    passed.append(idx)
                    continue
            notes.append("%s (still unconfirmed: %s)" % (idx, " and ".join("'%s'" % c for c in unsettled(idx))))
        return notes

    def _ejc_read_by_view(self):
        """A pointer to the view tool when the details tool is withdrawn but the view tool is offered."""
        live = set(t["function"]["name"] for t in self._tools_this_turn())
        viewer = self.roles.get("view")
        reader = self.roles.get("details")
        swapped = bool(viewer and reader) and viewer in live and reader not in live
        return (" (the details tool has been withdrawn; %s is still offered and shows their data)" % viewer
                if swapped else "")

    def _g1d_unseen(self, table, priced, conditions):
        """Affordable never-opened listings of the opened ones' sections: most wanted words, then cheapest; as
        many as the free steps allow, four at most."""
        labels = {row["label"] for row in priced.values() if row["label"]}
        wanted = set()
        for c in conditions:
            wanted |= _word_set(str(c).partition(":")[2].split("!=")[0])

        pool = []
        for ref, info in table.items():
            if info["opened"] or info["price"] is None or not self._within_price_cap(info):
                continue
            if labels and info["label"] not in labels:
                continue
            pool.append((-len(wanted & _g1d_said(info.get("facts") or {})), info["price"], ref))
        pool.sort()
        free = self.remaining_steps() - self._reserved_steps() - 1
        return [ref for _, _, ref in pool[:min(max(free, 0), 4)]]

    # ---- finishing from code ----

    def _minimal_closing_args(self, label_name, text, duty=None):
        """Smallest valid arguments for a finishing tool: the job's fixed values, `text` in text fields."""
        shape = self.param_schemas.get(label_name) or {}
        fields = shape.get("properties") or {}
        built = {}
        if duty:
            built.update(self.fixed.get(duty) or {})
        for need in [k for k in (shape.get("required") or []) if k not in built]:
            filled = _g1d_fill(fields.get(need) or {}, text)
            if filled:
                built[need] = filled[0]
        if len(fields) == 1:
            (sole, desc), = fields.items()
            plain = isinstance(desc, dict) and desc.get("type") == "string" and "enum" not in desc
            if plain and sole not in built:
                built[sole] = _g1d_clip(text, desc)
        return built

    def _clean_decline_args(self, label_name, arguments):
        """Neutral remark in every free text field when names are barred or a seen id appears."""
        barred = self.plan.get("decline_no_names")
        if barred is False:
            return arguments
        if not barred and not self._g1d_mentions_seen_id(arguments):
            return arguments
        fields = (self.param_schemas.get(label_name) or {}).get("properties") or {}

        def neutral(name, current):
            desc = fields.get(name) or {}
            if isinstance(current, str) and "enum" not in desc:
                return _g1d_clip(_NOTHING_FITS_TEXT, desc)
            return current

        return {name: neutral(name, current) for name, current in arguments.items()}

    def _g1d_mentions_seen_id(self, arguments):
        for value in arguments.values():
            if isinstance(value, str) and any(tok and tok in self.seen_ids for tok in _g1d_BREAKS.split(value)):
                return True
        return False

    def _g1d_send_finish(self, duty, text):
        """Send the code-built finishing call; arguments the schema rejects are logged and not sent."""
        name = self.roles[duty]
        draft = self._minimal_closing_args(name, text, duty)
        if duty == "decline":
            draft = self._clean_decline_args(name, draft)
        good_args, problems = _validate_value(draft, self.param_schemas[name], "args")
        record = dict(kind="forced_finish", action=dict(name=name, args=good_args))
        # a refused call would still cost a step, so a broken one stays home
        record["reply"] = None if problems else self.call_store(record["action"])
        if problems:
            record["issues"] = problems
        self.trace.append(record)
        return record["reply"]

    def _listing_title(self, unit_ref):
        """The store's wordiest short text next to the id, for the shopper (who never sees ids)."""
        best = ""
        for text in _g1d_texts_beside(self.records, unit_ref):
            if len(text) <= 160 and len(text.split()) > len(best.split()):
                best = text
        return '"%s"' % best if best else "item " + unit_ref

    def _g1d_finish_duty(self):
        # an order needs an order tool, a used cart, and held lines unless no cart tool exists
        holds = bool(self.cart_ids) or not self.roles.get("cart")
        return "order" if self.roles.get("order") and self.cart_used and holds else "decline"

    def finish_now(self):
        """Finish from code: order the cart when possible, otherwise decline; an order that leaves the episode
        open is followed by a decline."""
        over = self.finished or self.halted
        if over or self.remaining_steps() <= 0 or self.seconds_remaining() < 2:
            return
        job = self._g1d_finish_duty()
        if self.roles.get(job):
            answer = self._g1d_send_finish(job, _ejc_FORCED_TEXT[job])
            if job == "order" and answer and not answer.get("finished"):
                if self.roles.get("decline") and 0 < self.remaining_steps():
                    self._g1d_send_finish("decline", _NOTHING_FITS_TEXT)

    # ---- entry points into the pipeline ----

    def _run_calls(self, calls, planned=False):
        # the pipeline checks and sends them; one reply per call, same order
        outcome = _eje_execute(self, {"calls": calls, "planned": planned})
        return outcome["replies"]

    def run(self):
        _eje_episode(self, {})
        return self.trace


# ---- module level: finishing texts, record and word helpers ----

_CLOSING_JOBS = ("order", "decline")

_CLOSING_ROLES_TEXT = """Answer with one line of JSON only, exactly in this shape:
{"order": {"tool": exact tool name, "fixed": {argument: value}}, "decline": {"tool": exact tool name, "fixed": {argument: value}}}

Filling "fixed":
- list every argument whose allowed values (a set of options, or true/false) decide what the call does, at the exact value that job requires;
- keep free-text arguments out; a tool without such arguments gets {};
- when one tool serves both jobs, both entries use the same argument names with different values.

The two jobs: an assistant runs a shop through the tools listed below, and two kinds of call bring its work to an end -
- "order" purchases whatever the basket holds right now;
- "decline" ends the work with no purchase.
Name the tool for each and the argument values that trigger it, reading the complete argument schemas: descriptions and allowed values alike. A tool that only shows the basket, or only sends the shopper a note, serves neither job."""

_ROLE_REFEREE_TEXT = """Output exactly one line of JSON and nothing else:
{"order": {"tool": exact tool name, "fixed": {argument: value}}, "decline": {"tool": exact tool name, "fixed": {argument: value}}}

You settle a dispute between READING A and READING B, both given after this text. Each one names the shop tool that buys what the basket holds now (the "order" job) and the tool that ends the job with nothing bought (the "decline" job).
1. Test both claims against the tool list: argument schemas, allowed values, descriptions.
2. Settle the right tool for each job; it may come from either reading or from neither.
3. Under "fixed", give each argument whose allowed values change what the call does, at its exact value. If one tool covers both jobs, the two entries share argument names and differ in values."""

_NOTHING_FITS_TEXT = "None of the listings I checked meets all of your conditions, so I am ending without an order."
_ejc_FORCED_TEXT = {"order": "Order the cart.", "decline": _NOTHING_FITS_TEXT}
_g1d_BREAKS = re.compile(r"[\s.,;:!?()\[\]{}<>\"'`]+")
_SKIP_NOTICE = "Skipped, no step spent"
_HELD_NOTICE = "Not sent: your own check stopped this call (the store did not), and no step was used."
_ejc_PARTS = re.compile(r"[,/;|]")
_ejc_MARKS = ("<=", ">=", "covers")


def _g1d_line_entry(row):
    """One requested product from the setup reading, or None when it names no product."""
    if not isinstance(row, dict):
        return None
    name = str(row.get("product") or "").strip()
    if not name:
        return None
    wants = row["requirements"] if "requirements" in row else row.get("requirement")
    if isinstance(wants, str):
        wants = [wants]
    elif not isinstance(wants, list):
        wants = []
    units = row.get("quantity")
    if isinstance(units, bool) or not isinstance(units, int) or units < 1:
        units = 1
    texts = [str(w).strip() for w in wants if isinstance(w, (str, int, float)) and str(w).strip()]
    return {"product": name, "quantity": units, "requirements": texts}


def _g1d_texts_beside(records, ref):
    """Text values (other than `ref`) of every record dict that holds `ref`."""
    for node in _iter_dicts(records):
        if ref in node.values():
            for value in node.values():
                if isinstance(value, str) and value != ref:
                    yield value


def _ejc_text_fields(props):
    """Free text arguments: string typed, no option list."""
    return [key for key, spec in props.items()
            if isinstance(spec, dict) and "enum" not in spec and "string" in (spec.get("type") or [])]


def _g1d_clip(text, spec):
    return text[:spec.get("maxLength", len(text))]


def _g1d_fill(spec, text):
    """A one-item tuple with the value a finishing call can put in a required argument, else ()."""
    declared = spec.get("type")
    if "string" in (declared or []):
        return (_g1d_clip(text, spec),)
    if spec.get("enum"):
        return (spec["enum"][0],)
    if declared == "boolean":
        return (True,)
    if declared in ("integer", "number"):
        floor = spec.get("minimum", 1)
        return (floor if floor > 1 else 1,)
    return ()


def _ejc_newest_within(records, limit):
    """Short forms of the records, newest first, skipping any that would pass `limit` characters in total."""
    kept, used = [], 0
    for record in reversed(records):
        piece = _to_compact_json(_shorten_value(record, 160))
        if used + len(piece) > limit:
            continue
        kept.append(piece)
        used += len(piece)
    return kept


def _g1d_parse(line):
    """(property words, value words) of "property: constraint"; None without a colon."""
    head, sep, tail = str(line).partition(":")
    if not sep:
        return None
    return _word_set(head), _word_set(tail)


def _g1d_said(sheet):
    return _word_set(" ".join(map(str, sheet.values())))


def _g1d_listing_nodes(records):
    """(id, dict) for each dict with an id-shaped text under a key containing "id" (first such key)."""
    out = []
    for node in _iter_dicts(records):
        ident = next((v for k, v in node.items()
                      if isinstance(v, str) and "id" in str(k).lower() and _ID_SHAPE.match(v)), None)
        if ident is not None:
            out.append((ident, node))
    return out


def _g1d_is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _g1d_is_flag(value):
    return isinstance(value, bool)


def _g1d_is_text(value):
    return isinstance(value, str)


def _g1d_commonest(keys):
    """Most frequent key, earliest on a tie; None when empty."""
    tally = {}
    for key in keys:
        tally[key] = tally.get(key, 0) + 1
    return max(tally, key=tally.get, default=None)


def _g1d_field_roles(found):
    """[(slot, key, test)]: price = key most often numeric, stock = key most often boolean, label = repeating
    short text key with the most values (earliest on a tie)."""
    nums, flags, shorts = [], [], {}
    for ident, node in found:
        for key, value in node.items():
            if value == ident:   # the id itself recurs, cart lines too
                pass
            elif isinstance(value, bool):
                flags.append(key)
            elif isinstance(value, (int, float)):
                nums.append(key)
            elif isinstance(value, str) and value and len(value) <= 40:
                shorts[key] = shorts.get(key, []) + [value]
    label, most = None, -1
    for key, vals in shorts.items():
        # a label key repeats its values: at least two entries per distinct value
        if len(vals) >= 2 * len(set(vals)) and len(vals) > most:
            label, most = key, len(vals)
    return [("price", _g1d_commonest(nums), _g1d_is_number),
            ("stock", _g1d_commonest(flags), _g1d_is_flag),
            ("label", label, _g1d_is_text)]


def _g1d_sheet(node, ident):
    """Short texts of a record (its id excepted), then text and list members of nested dicts."""
    sheet = {k: v for k, v in node.items() if isinstance(v, str) and v != ident and 0 < len(v) <= 120}
    for child in node.values():
        if isinstance(child, dict):
            for k, v in child.items():
                if isinstance(v, (str, list)):
                    sheet[k] = v
    return sheet


def _g1d_walk(roots):
    """Every dict and list reachable from the roots."""
    stack = list(roots)
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            kids = node.values()
        elif isinstance(node, list):
            kids = node
        else:
            continue
        yield node
        stack.extend(kid for kid in kids if isinstance(kid, (dict, list)))


def _looks_like_listing(field_value, ids):
    """A seen id shaped like a code: eight or more characters mixing letters and digits."""
    if not (isinstance(field_value, str) and len(field_value) >= 8 and field_value in ids):
        return False
    return any(ch.isdigit() for ch in field_value) and any(ch.isalpha() for ch in field_value)


def _search_row_ids(roots, ids):
    """id() of every search result row: a dict in a list next to another dict of the same keys, holding a
    listing id (a result line, not a property sheet)."""
    rows = set()
    for node in _g1d_walk(roots):
        if not isinstance(node, list):
            continue
        members = [m for m in node if isinstance(m, dict)]
        shapes = [tuple(sorted(m)) for m in members]
        for member, shape in zip(members, shapes):
            if shapes.count(shape) > 1 and any(_looks_like_listing(v, ids) for v in member.values()):
                rows.add(id(member))
    return rows

def _own_listing_fields(node_list, ids):
    """Triples (listing, key, text) that a listing repeats in more than one record layout, e.g. a search row and
    its opened page. Such text is the listing's own (its title, its maker) and not a property to ask about."""
    layouts_of = {}
    for rec in node_list:
        who = None
        for cell in rec.values():
            if _looks_like_listing(cell, ids):
                who = cell
                break
        if who is None:
            continue
        shape = tuple(sorted(rec))
        for key, cell in rec.items():
            if isinstance(cell, str) and cell not in ids:
                layouts_of.setdefault((who, key, cell), set()).add(shape)
    return {triple for triple, shapes in layouts_of.items() if len(shapes) > 1}


def _name_stems(label_name):
    """Word stems of a name: a trailing single "s" comes off longer words ("sizes" meets "size")."""
    def stem(word):
        return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word
    return {stem(word) for word in _word_set(label_name)}


def _is_enum_spec(tool_desc):
    """Schema of an argument with a closed set of choices (enum, const or boolean)."""
    if not isinstance(tool_desc, dict):
        return False
    return bool(tool_desc.get("enum")) or "const" in tool_desc or "boolean" in (_declared_types(tool_desc) or [])


def _word_set(text):
    """Lower-cased alphanumeric tokens; digits always count, letters from three characters on."""
    raw = str(text or "").lower()
    return {tok for tok in re.findall(r"[a-z0-9]+", raw) if len(tok) >= 3 or tok.isdigit()}


def _flatten_text(text):
    """Lower-cased text with all whitespace runs squeezed to single spaces."""
    return " ".join(str(text or "").lower().split())


def _words_match(have, wish):
    """Two spellings of one word: the longer starts with the shorter and adds at most two letters."""
    short, long_ = sorted((have, wish), key=len)
    return long_[:len(short)] == short and len(long_) - len(short) <= 2


def _words_cover(needed, have):
    """Each word of `needed` has some spelling among `have`."""
    return all(any(_words_match(word, other) for other in have) for word in needed)


def _key_words(key):
    """Words of a key, splitting camelCase steps as well."""
    return _word_set(re.sub(r"([a-z0-9])(?=[A-Z])", r"\1 ", str(key)))


def _key_names_property(key, need_terms):
    """The key's words are the property's words and nothing more (spelling variants allowed)."""
    parts = _key_words(key)
    if not parts or len(parts) != len(need_terms):
        return False
    return _words_cover(need_terms, parts)


def _g1e_split_wish(tail):
    """Split the right side of a condition into (value, is exclusion, value that must be absent or None).
    Bracketed notes go unless nothing is left; text after "!=" names what must not be stated."""
    body = re.sub(r"\([^)]*\)", " ", tail)
    if not _word_set(body):
        body = tail
    wanted, _mark, banned = body.partition("!=")
    if _word_set(banned):
        if _word_set(wanted):
            return wanted, False, banned
        return banned, True, None
    return body, False, None


def _g1e_pool(fact_sheet, prop_terms):
    """(members, singles): every value under keys that carry the property, split into parts; and the lone plain
    values of keys named exactly as the property."""
    members, singles = [], []
    for key, cell in fact_sheet.items():
        named = _key_names_property(key, prop_terms)
        if not (named or prop_terms <= _key_words(key)):
            continue
        items = [str(item) for item in cell] if isinstance(cell, list) else [str(cell)]
        for text in items:
            members += [bit for bit in (text, *_ejc_PARTS.split(text)) if _word_set(bit)]
        if named and not isinstance(cell, list):
            lone = [bit for bit in _ejc_PARTS.split(items[0]) if bit.strip()]
            if len(lone) == 1 and _word_set(lone[0]):
                singles.append(lone[0])
    return members, singles


_g1e_BLURB_WORDS = 5   # from this many words on, a value is a blurb or title rather than a property value


def _ejc_text_states(fact_sheet, condition):
    """True when no key carries the property, the wish is plain words, and some blurb value contains them all."""
    prop, colon, tail = str(condition).partition(":")
    value, negated, other = _g1e_split_wish(tail)
    prop_terms, wanted = _word_set(prop), _word_set(value)
    plain_wish = colon and prop_terms and wanted and not negated and other is None
    if not plain_wish or re.search(r"\d", value) or _g1e_pool(fact_sheet, prop_terms)[0]:
        return False
    for cell in fact_sheet.values():
        if not isinstance(cell, str):
            continue
        words = _word_set(cell)
        if len(words) >= _g1e_BLURB_WORDS and _words_cover(wanted, words):
            return True
    return False


def _g1e_whole_value(fact_sheet, wanted):
    """Some value (or a part of one) consists of exactly the wanted words."""
    for cell in fact_sheet.values():
        for item in (cell if isinstance(cell, list) else [cell]):
            text = str(item)
            if _word_set(text) == wanted or any(_word_set(bit) == wanted for bit in _ejc_PARTS.split(text)):
                return True
    return False


def _literal_match(facts, condition, phrasing=None):
    """What a listing's own facts say on one "property: wish" condition: True, False or None (cannot tell).

    Values come from keys whose names hold every property word, split part by part. True when a part holds all
    wanted words (or is one of the given options). False when parts exist and none fits, or when the exactly
    named keys state single other values. None when no key carries the property, the wish is a bare number,
    the parts only give a vaguer form of the wish, or the exactly named keys disagree. "!=" marks exclusions;
    `phrasing` (the store's wording of a quoted range) means a range counts only when that wording is stated."""
    prop, colon, tail = str(condition).partition(":")
    value, negated, other = _g1e_split_wish(tail)
    if other is not None and _literal_match(facts, f"{prop}: != {other}") is False:
        return False
    prop_terms, wanted = _word_set(prop), _word_set(value)
    if not (colon and prop_terms and wanted):
        return None
    members, singles = _g1e_pool(facts, prop_terms)
    has_digit = re.search(r"\d", value) is not None
    relation = _parse_relation(value)
    if not members:
        # no own key: only a whole value elsewhere (the variant on sale, say) can still state it
        if negated or has_digit or relation is not None:
            return None
        return True if _g1e_whole_value(facts, wanted) else None
    if negated:
        banned = [words for words in (_word_set(bit) for bit in _ejc_PARTS.split(value)) if words]
        return not any(_words_cover(words, _word_set(part)) for part in members for words in banned)
    if relation is not None:
        if not phrasing:
            return _g1e_span_fits(relation, members)
        return any(quote in _flatten_text(part) for part in members for quote in phrasing)

    def holds(part):
        return _words_cover(wanted, _word_set(part))

    if has_digit:
        return True if any(holds(part) for part in members) else None

    def near(part):   # holds the wish, or is a vaguer form of it
        return holds(part) or _words_cover(_word_set(part), wanted)

    if singles and not any(near(one) for one in singles):
        return False
    mixed = not all(near(one) for one in singles)
    if any(holds(part) for part in members):
        return None if mixed else True
    if mixed or any(near(part) for part in members):
        return None
    return False


def _parse_relation(wish):
    """(mark, number, magnitudes) from a wish led by "<=", ">=" or "covers"; "+/-" or the plus-minus sign asks
    for magnitudes. None when the mark or the number is missing."""
    text = wish.strip()
    lowered = text.lower()
    for mark in _ejc_MARKS:
        if lowered.startswith(mark):
            break
    else:
        return None
    found = re.search(r"\d+(?:\.\d+)?", text[len(mark):])
    if found is None:
        return None
    return mark, float(found.group()), ("+/-" in text or "\u00b1" in text)


def _g1e_span_fits(rel, members):
    """Numbers in the parts against a relation. "covers": the stated span must include the value; "<=" / ">=":
    True when all numbers comply, False when none does, None when mixed or no number is stated."""
    mark, limit, magnitude = rel
    signed = [float(num) for part in members for num in re.findall(r"-?\d+(?:\.\d+)?", part)]
    if not signed:
        return None
    seen = [abs(num) for num in signed] if magnitude else signed
    if mark == "covers":
        if magnitude and min(signed) < 0:   # a span with signs has to reach both -value and +value
            return min(signed) <= -limit and max(signed) >= limit
        return min(seen) <= limit <= max(seen)
    ok = [(num <= limit) if mark == "<=" else (num >= limit) for num in seen]
    if all(ok):
        return True
    return None if any(ok) else False


def _cart_listing_id(session, label_name, arguments):
    """Listing a cart call targets: its first single-id argument; with several cart lines, the first one not
    yet carted, otherwise the last line."""
    props = session.param_schemas[label_name].get("properties") or {}
    direct = list(_eje_cart_ids_in(props, arguments))
    if direct:
        return direct[0]
    line_ids = [listing for listing, _qty in _eje_cart_lines(props, arguments)]
    for listing in line_ids:
        if listing not in session.cart_ids:
            return listing
    return line_ids[-1] if line_ids else None


def _looks_like_code(text, need_separator):
    """A sku-like token: letters, digits, "-" and "_" only, with at least one digit (and a separator if asked)."""
    if not (re.fullmatch(r"[A-Za-z0-9_\-]+", text) and re.search(r"\d", text)):
        return False
    return not need_separator or "-" in text or "_" in text


def _g1e_lines_only(value):
    """A non-empty list made of strings only."""
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) for item in value)


def _unwrap_text(field_value, nest_level=0):
    """Strip world wrappers: a dict of two entries, one a list of text lines and the other the wrapped result.
    Goes four layers deep at most and returns (collected lines, innermost value)."""
    if nest_level < 4 and isinstance(field_value, dict) and len(field_value) == 2:
        first, second = field_value.values()
        first_lines, second_lines = _g1e_lines_only(first), _g1e_lines_only(second)
        if first_lines != second_lines:
            lines, wrapped = (first, second) if first_lines else (second, first)
            more, core = _unwrap_text(wrapped, nest_level + 1)
            return list(lines) + more, core
    return [], field_value


def _looks_like_refusal(field_value):
    """A store refusal: a flat dict with one to three values, one of them a code-like token."""
    if not isinstance(field_value, dict) or len(field_value) not in (1, 2, 3):
        return False
    cells = list(field_value.values())
    if any(isinstance(cell, (dict, list)) for cell in cells):
        return False
    tokens = [cell.strip() for cell in cells if isinstance(cell, str)]
    return any(tok and " " not in tok and (tok.lower() != tok or "_" in tok or any(ch.isdecimal() for ch in tok))
               for tok in tokens)


def _iter_dicts(roots):
    """All dicts inside the records, parents before children in document order, down to depth six."""
    for root in roots:
        pending = [(root, 0)]
        while pending:
            node, depth = pending.pop()
            if depth > 6:
                continue
            if isinstance(node, dict):
                yield node
                kids = [cell for cell in node.values() if isinstance(cell, (dict, list))]
            elif isinstance(node, list):
                kids = node
            else:
                continue
            pending.extend((kid, depth + 1) for kid in reversed(kids))

# ================================================================ episode stages
# A stage gets (session, record), may set record["exit"] to stop its run, and hands the record back.

_Record = dict[str, Any]   # the record a run of stages passes along


def _g1e_first_message(run: Episode) -> str:
    """Opening user message: the shopper's request, the setup reading, the status line, the opening store view."""
    request = str(run.policy.get("query") or "")
    reading = run.opening_lines()
    view = run.policy.get("opening")
    status = run.status()
    sections = ["Request from the shopper:\n" + request]
    if reading:
        sections.append(reading)
    sections.append(status)
    if view is not None:
        sections.append("What the store showed first:\n" + _to_compact_json(view))
    return "\n\n".join(sections)


def _g1e_open(run: Episode) -> None:
    """Read the request, log the plan and seed the conversation; the opening view joins the records."""
    run.setup()
    run.trace.append({"kind": "plan", "plan": run.plan, "roles": run._roles_for_log()})
    opening = _g1e_first_message(run)
    shown = run.policy.get("opening")
    if shown is not None:
        run.records.append(shown)
        _gather_ids(shown, run.seen_ids)
    run.messages = [{"role": "system", "content": SHOPPER_SYSTEM_TEXT}, {"role": "user", "content": opening}]


def _stage_start_episode(session: Episode, carry: _Record) -> _Record:
    """One-time setup, skipped when the episode is already over."""
    if session.finished:
        carry["exit"] = "already finished"
    else:
        _g1e_open(session)
    return carry


class _g1e_Tally:
    """Counters kept across model turns: replies without calls, failed model calls, issued call ids."""

    __slots__ = ("quiet", "misses", "seq")

    def __init__(self):
        self.quiet = 0
        self.misses = 0
        self.seq = 0


def _stage_loop_turns(session: Episode, carry: _Record) -> _Record:
    tally = _g1e_Tally()
    for _round in range(TURN_LIMIT):
        if _eje_turn(session, {"carry": tally}).get("exit") == "stop":
            break
    return carry


def _stage_settle_episode(session: Episode, carry: _Record) -> _Record:
    if session.finished or session.halted:
        return carry
    session.finish_now()
    return carry


def _stage_close_episode(session: Episode, carry: _Record) -> _Record:
    elapsed = round(time.monotonic() - session.start_clock, 1)
    session.trace.append({"kind": "end", "finished": session.finished, "steps": session.step,
                          "cost": round(session.cost, 5), "seconds": elapsed})
    return carry


# ================================================================ turn stages

def _g1e_end_turn(run: Episode, carry: _Record, final: bool) -> _Record:
    """Leave the turn; when final, code finishes the episode and no further turn follows."""
    if not final:
        carry["exit"] = "next"
        return carry
    run.finish_now()
    carry["exit"] = "stop"
    return carry


def _g1e_turn_blocked(run: Episode) -> str | None:
    """"stop" when no turn may run, "finish" when code must finish now, None when the model may go on."""
    if run.finished or run.halted or run.remaining_steps() <= 0:
        return "stop"
    # no usable key, the clock nearly out, or spend past the hard line: code finishes (never a pricier model)
    if run.key_dry or run.seconds_remaining() < FORCE_MARGIN or run.cost > _eja_SPEND_HARD:
        return "finish"
    return None


def _stage_turn_guard(session: Episode, carry: _Record) -> _Record:
    blocked = _g1e_turn_blocked(session)
    if blocked == "finish":
        return _g1e_end_turn(session, carry, True)
    if blocked == "stop":
        carry["exit"] = "stop"
    return carry


def _stage_send_queued(session: Episode, carry: _Record) -> _Record:
    text = session.queued_reply
    if text:
        session.queued_reply = ""
        if session._send_answer(text):
            carry["exit"] = "next"
    return carry


def _stage_run_planner(session: Episode, carry: _Record) -> _Record:
    if session.planner_on and session._run_question_planner():
        carry["exit"] = "next"
    return carry


def _stage_call_model(session: Episode, carry: _Record) -> _Record:
    answer = session.infer(session.messages, session._tools_this_turn(),
                           [PRIMARY_SEAT, FALLBACK_SEAT, PRIMARY_SEAT], 2500)
    if answer is not None:
        carry["reply"] = answer
        return carry
    tally = carry["carry"]
    tally.misses += 1
    return _g1e_end_turn(session, carry, tally.misses >= 2 or session.key_dry)


def _g1e_tidy(raw: _Record, seq: int) -> _Record:
    """One proposed call with an id and JSON-string arguments."""
    fn = raw.get("function") or {}
    args = fn.get("arguments")
    text_args = args if isinstance(args, str) else json.dumps(args or {})
    return {"id": raw.get("id") or f"call_{seq}", "type": "function",
            "function": {"name": fn.get("name"), "arguments": text_args}}


def _stage_take_calls(session: Episode, carry: _Record) -> _Record:
    """Normalise the proposed calls and the reply text."""
    tally, answer = carry["carry"], carry["reply"]
    proposed = [raw for raw in answer.get("tool_calls") or [] if isinstance(raw, dict)]
    tidy = []
    for raw in proposed:
        tally.seq += 1
        tidy.append(_g1e_tidy(raw, tally.seq))
    said = answer.get("content") or ""
    carry.update(calls=tidy, said=said if isinstance(said, str) else _to_compact_json(said))
    return carry


def _stage_commit_turn(session: Episode, carry: _Record) -> _Record:
    """A cart, order or decline call first gives the planner one forced chance per (ledger, menu, asked) state."""
    duties = session.roles
    wanted = {call["function"]["name"] for call in carry["calls"]}
    decisive = any(duties.get(duty) is not None and duties.get(duty) in wanted for duty in ("cart", "order", "decline"))
    if not (session.planner_on and decisive):
        return carry
    ledger = session.needs_revision
    state_key = (ledger, session._build_property_menu()[1], len(session.asked))
    if session.forced_basis != state_key:
        session.forced_basis = state_key
        moved = session._run_question_planner(force=True) and ledger != session.needs_revision
        if moved:
            carry["exit"] = "next"   # the requirements changed: the model decides again
    return carry


_g1e_NUDGE = "Every reply must call a tool; the last one did not. Continue.\n"


def _stage_idle_turn(session: Episode, carry: _Record) -> _Record:
    tally = carry["carry"]
    if carry["calls"]:
        tally.quiet = 0
        return carry
    tally.quiet += 1
    session.messages.append({"role": "assistant", "content": carry["said"] or "(no tool call)"})
    session.messages.append({"role": "user", "content": _g1e_NUDGE + session.status()})
    return _g1e_end_turn(session, carry, tally.quiet >= 3)


def _stage_act_turn(session: Episode, carry: _Record) -> _Record:
    calls, said = carry["calls"], carry["said"]
    session.trace.append({"kind": "model", "content": said[:2000], "calls": [call["function"] for call in calls]})
    session.messages.append({"role": "assistant", "content": said or None, "tool_calls": calls})
    replies = session._run_calls(calls)
    session._record_replies(replies)
    return carry


# ================================================================ question planner

def _spare_steps_for(max_steps: int) -> int:
    """Steps kept back for a hold, the finish, and a refused finish plus its re-read."""
    return max(5, round(max_steps * 0.11))


# Chance that asking about a property carrying a hidden need brings the need out (pooled over past worlds); times
# the planner's own chance, it prices a question when the request charges for asking.
_ejd_REVEAL_ODDS = 0.9

_g1e_FOLLOW_UP = 2   # steps to act on one revealed need: a narrow search and one opened listing

# A short leading word of a property name, followed by more name.
_ejd_LEAD_WORD = re.compile(r"[^A-Za-z0-9]*([A-Za-z0-9]+)[^A-Za-z0-9]+(?=[A-Za-z0-9])")


def _token_list(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _same_tokens(a: str, b: str) -> bool:
    """One property name, ignoring case, spacing and punctuation."""
    return _token_list(a) == _token_list(b)


def _g1e_contains_run(longer: list[str], shorter: list[str]) -> bool:
    """`shorter` (at least two words) appears as a contiguous run in `longer`."""
    size = len(shorter)
    if size < 2:
        return False
    return any(longer[start:start + size] == shorter for start in range(len(longer) - size + 1))


def _ejd_asked_before(store_run: Episode, prop_name: str) -> bool:
    """Already asked: this name, or a name of two or more words that contains it or is contained in it."""
    mine = _token_list(prop_name)

    def related(other: list[str]) -> bool:
        return other == mine or _g1e_contains_run(mine, other) or _g1e_contains_run(other, mine)

    return any(related(_token_list(done)) for done in store_run.asked_props)


def _g1e_held_messages(run: Episode) -> int:
    """Messages to the shopper that pending stated procedures will need."""
    return sum(proc["count"] for proc in run._pending_procedures("order") if proc["role"] == "message")


def _ejd_clean_want(wish: str) -> set[str]:
    return _word_set(re.sub(r"\(.*?\)", " ", wish))


def _g1e_settled(run: Episode, prop_name: str, info: _Record) -> bool:
    """A requirement in force already decides this menu property: it names it (or an alias), or its wanted words
    sit inside one of the property's values. A loose requirement leaves the property open."""
    names = [_word_set(prop_name), *(_word_set(alias) for alias in info.get("also") or [])]
    for cond in run._list_conditions():
        head, _sep, wish = str(cond).partition(":")
        head_words = _word_set(head)
        if head_words and any(name and (name <= head_words or head_words <= name) for name in names):
            if prop_name not in _ejd_loose_props(run):
                return True
            continue
        wanted = _ejd_clean_want(wish)
        if wanted and any(wanted <= _word_set(val) for val in info.get("values") or []):
            return True
    return False


def _ejd_ranking(store_run: Episode) -> list[str]:
    """Menu properties still worth asking, in a stable order built from opened listings only (reruns of one
    request see the same list). Dropped: asked, settled, variant lines. Order: loose first, label-like and
    single-valued last, then by coverage, value groups, value count and name."""
    menu = store_run.property_menu
    open_props = [name for name in menu
                  if not (_ejd_asked_before(store_run, name) or _g1e_settled(store_run, name, menu[name])
                          or _g1e_variant_line(menu[name]))]
    loose = set(_ejd_loose_props(store_run)) & set(open_props)

    def rank(name: str):
        info = menu[name]
        return (name not in loose, _g1e_label_like(name, info), info.get("uniform", False), -info["count"],
                -_g1e_value_groups(info), -len(info["values"]), name)

    return sorted(open_props, key=rank)


def _g1e_label_like(prop_name: str, info: _Record) -> bool:
    """Values read like labels rather than single options: one joins options with "/", or most of them repeat a
    word of the property name. Hidden needs rarely sat on such lines, so they rank last."""
    vals = info.get("values") or []
    if any("/" in val for val in vals):
        return True
    stems = _name_stems(prop_name)
    if not stems:
        return False
    repeats = sum(1 for val in vals if stems & _name_stems(val))
    return 2 * repeats > len(vals)


def _g1e_value_groups(info: _Record) -> int:
    """Count of truly different values: values that share any token (unit marks included) merge into one."""
    groups: list[set[str]] = []
    for val in info.get("values") or []:
        tokens = set(_token_list(val))
        merged, kept = set(tokens), []
        for group in groups:
            if group & tokens:
                merged |= group
            else:
                kept.append(group)
        groups = kept + [merged]
    return len(groups)


def _g1e_loose_value(wish: str, values: list[str]) -> bool:
    """The wish matches no value (nor a part of one) as written, yet its words fit two or more values."""
    if "!=" in wish:
        return False
    phrase = _flatten_text(re.sub(r"\(.*?\)", " ", wish))
    wanted = _word_set(phrase)
    if not wanted:
        return False
    whole = {_flatten_text(val) for val in values}
    parts = {_flatten_text(bit) for val in values for bit in re.split(r"[,;/|]", str(val)) if bit.strip()}
    if phrase in whole or phrase in parts:
        return False   # stated as is, alone or inside a list
    return sum(1 for val in whole if wanted <= _word_set(val)) >= 2


def _ejd_loose_props(store_run: Episode) -> dict[str, str]:
    """Menu property -> loose wish for requirements in force. Values are pooled over all menu names that share a
    word with the requirement's property; only the best covered of them is kept (one question per requirement)."""
    menu = store_run.property_menu
    loose: dict[str, str] = {}
    for cond in store_run._list_conditions():
        head, _sep, wish = str(cond).partition(":")
        head_words = _word_set(head)
        if not head_words:
            continue
        near = [name for name in menu if head_words & _word_set(name)]
        if not near:
            continue
        pooled = [val for name in near for val in menu[name].get("values") or []]
        if _g1e_loose_value(wish, pooled):
            best = max(near, key=lambda name: (menu[name]["count"], -len(name)))
            loose[best] = wish.strip()
    return loose


def _g1e_variant_line(info: _Record) -> bool:
    """A seller's variant line: three or more listings, a distinct value on each, mostly "/"-joined options."""
    vals = info.get("values") or []
    listings = info.get("count") or 0
    if listings < 3 or len(vals) != listings:
        return False
    return sum(1 for val in vals if "/" in val) * 2 > len(vals)


def _g1e_ready_to_ask(run: Episode, sheets: int, forced: bool) -> bool:
    """Ask now when there are two fact sets (one if forced), when the request says to ask first and nothing was
    asked, or late in the episode with two fact sets or in a world without a details tool."""
    if sheets >= (1 if forced else 2):
        return True
    if run.plan.get("ask_first") and not run.asked and not run.urgent_tried:
        return True
    if run.step < round(run.max_steps * 0.22):
        return False
    return sheets >= 2 or not run.roles.get("details")


def _g1e_hold(run: Episode, carry: _Record, switch_off: bool) -> _Record:
    """No question this time; switch_off also stops the planner for the rest of the episode."""
    if switch_off:
        run.planner_on = False
    carry["exit"] = False
    return carry


def _g1e_gate(run: Episode, forced: bool):
    """(switch_off, None) when no question goes out now; (None, record fields) when one may."""
    tool = run.roles.get("message")
    cap = run._message_limit()
    sent = run.messages_used
    if not tool or sent >= cap:
        return True, None
    cap -= _g1e_held_messages(run)   # stated procedures get their messages first
    if sent >= cap:
        return False, None
    menu, sheets = run._build_property_menu()
    if not _g1e_ready_to_ask(run, sheets, forced):
        return False, None
    slack = run.remaining_steps() - _spare_steps_for(run.max_steps)
    if slack < cap - sent + _g1e_FOLLOW_UP or run.seconds_remaining() < WRAP_UP_WINDOW + 40:
        return True, None
    return None, {"tool": tool, "cap": cap, "menu": menu, "sheets": sheets}


def _stage_question_gate(session: Episode, carry: _Record) -> _Record:
    """Whether the planner may ask now; if so the record gets the message tool, the cap and the fact menu."""
    switch_off, fields = _g1e_gate(session, carry["force"])
    if fields is None:
        return _g1e_hold(session, carry, switch_off)
    carry.update(fields)
    return carry


def _ejd_sibling_pick(store_run: Episode, carry: _Record, sibling: str) -> _Record | None:
    """Next after a brushed-off answer: the overlapping property, worded by the planner if it picks that one."""
    last = store_run.asked_props[-1]
    hint = (f'Ask about "{sibling}" this time, calling it by that name next to the product noun. Why: the answer '
            f'on "{last}" brushed it aside, and as the two names overlap, that answer may have been about this one.')
    drafted = _draft_plan_once(store_run, carry, hint)
    on_target = drafted is not None and _same_tokens(drafted["property"], sibling)
    chosen = drafted if on_target else dict(property=sibling, question=_ejd_plain(store_run, sibling),
                                            repeat=False, chance=None)
    return _ejd_worth(store_run, chosen) and chosen or None


def _ejd_regular_pick(store_run: Episode, carry: _Record) -> _Record | None:
    def draft(note, *focus):
        return _draft_plan_once(store_run, carry, note, *focus)

    pending = store_run.split_pending
    choice = draft("" if not pending else
                   f'The previous answer seemed to cover more than one thing. Property asked: "{pending[0]}"; '
                   f'question as sent: "{pending[1]}".')
    fresh = choice is not None and not choice["repeat"]
    if fresh and _ejd_names_sample(store_run, choice):
        choice.update(question=_ejd_plain(store_run, choice["property"]))   # sample values only for a sibling key
    if pending and not _ejd_sharper(choice, pending):
        # no sharper question: treat the property as asked and go on
        store_run.split_pending = None
        store_run.asked_props.append(pending[0])
        if not fresh:
            choice = draft("")
    if choice is None and store_run.reask_later:
        retry = store_run.reask_later.pop(0)
        choice = draft(f'"{retry}" may be asked once more; this time name the item by its own noun.', retry)
    return choice if choice is not None and _ejd_worth(store_run, choice) else None


def _ejd_unclear_pick(store_run: Episode, prop_name: str) -> _Record | None:
    """Single re-ask after a reply that missed the question: our plain question naming the product. If it would
    repeat the earlier wording, the property counts as asked and nothing is sent."""
    plain = _ejd_plain(store_run, prop_name)
    question = _ejd_with_stated_value(store_run, prop_name, plain)
    tag = "[%s] " % prop_name
    previous = next((entry for entry in reversed(store_run.asked) if entry.startswith(tag)), None)
    if previous is not None:
        quoted = previous.split('"')[1] if '"' in previous else ""
        if _token_list(quoted) == _token_list(question):
            store_run.asked_props.append(prop_name)
            return None
    retry = dict(property=prop_name, question=question, repeat=True, chance=None)
    return _ejd_worth(store_run, retry) and retry or None

def _g1f_planned_pick(ep: Episode, box: _Record, sibling: str) -> _Record | None:
    """The neighbour of a waved-off property when its turn is due, else the planner's regular choice."""
    neighbour_due = (bool(sibling) and not ep.split_pending
                     and not _ejd_asked_before(ep, sibling) and _ejd_turn_due(ep, sibling))
    if not neighbour_due:
        return _ejd_regular_pick(ep, box)
    got = _ejd_sibling_pick(ep, box, sibling)
    if got is not None:
        ep.sibling_props.append(got["property"])
    return got


def _stage_question_draft(ep: Episode, box: _Record) -> _Record:
    box["picks"] = []
    sibling, ep.sibling_next = ep.sibling_next, ""
    unclear, ep.unclear_next = ep.unclear_next, ""
    chosen = _ejd_unclear_pick(ep, unclear) if unclear else None
    if chosen is None and ep.planner_on:
        chosen = _g1f_planned_pick(ep, box, sibling)
    if chosen is not None:
        box["picks"] = [chosen]
    return box


def _g1f_shortlist(ep: Episode, order: list[str]) -> str:
    if ep.asked_props:
        head = 'Candidate properties are listed below. A pick from outside the list needs its reason in "deviate".'
    else:
        head = ('Candidate properties are listed below, strongest first. This first question must take one of the '
                'top three; any other pick needs its reason in "deviate".')
    vague = _ejd_loose_props(ep)
    rows = [head]
    for rank, name in enumerate(order[:8], 1):
        note = vague.get(name)
        extra = "" if note is None else ' - the requested "%s" appears in several listing values; ask which one' % note
        rows.append("(%d) %s%s" % (rank, name, extra))
    return "\n".join(rows)


def _g1f_planner_prompt(ep: Episode, box: _Record, extra: str, steer: list[str] | None, summary: str) -> str:
    parts = [QUESTION_PLAN_TEXT]
    if steer:
        parts.append(_g1f_shortlist(ep, steer))
    parts.append("Shopper's request, verbatim:\n" + str(ep.policy.get("query") or ""))
    parts.append("Requirements in force (current):\n" + summary)
    needed = ep.plan.get("ask_first")
    if needed:
        parts.append("Must come from the shopper: " + "; ".join(map(str, needed)))
    if ep.asked:
        parts.append("Questions sent earlier, each with its reply:\n" + "\n".join(ep.asked))
    parts.append("Detail-sheet properties so far (each line: name / listings showing it / sample values):\n"
                 + (box["menu"] or "(none yet)"))
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


def _g1f_hold_to_list(ep: Episode, steer: list[str], reply: _Record, prop_name: str, phrasing: str) -> tuple[str, str]:
    """Accept the planner's property when it sits in the allowed window of candidates (top three for the first
    question) or names a candidate with a stated reason; else fall back to the leading candidate."""
    def listed(names: list[str]) -> bool:
        return any(_same_tokens(prop_name, cand) for cand in names)

    span = 8 if ep.asked_props else 3
    if listed(steer[:span]) or (str(reply.get("deviate") or "").strip() and listed(steer)):
        return prop_name, phrasing
    lead = steer[0]
    fallback = _ejd_plain(ep, lead)
    ep.trace.append(dict(kind="ask_ranked", property=lead, dropped=reply.get("property")))
    return lead, fallback


def _ejd_with_stated_value(ep: Episode, prop_name: str, phrasing: str) -> str:
    """A loose requested value the question leaves out gets named in the question."""
    match = next(((label, value) for label, value in _ejd_loose_props(ep).items()
                  if _same_tokens(prop_name, label)), None)
    if match is None or _word_set(match[1]) <= _word_set(phrasing):
        return phrasing
    said = _g1f_spoken(ep, prop_name)
    owner = _ejd_owner(ep, prop_name)
    held = f"{said} of the {owner}" if owner else said
    return f"You said {match[1]} for the {held}: which {said} exactly do you mean?"


def _g1f_planner_reply(ep: Episode, box: _Record, extra: str, steer: list[str] | None, summary: str) -> _Record:
    """The planner's (first) proposal as a dict; {} when the call gives nothing."""
    got = ep.ask_judge(_g1f_planner_prompt(ep, box, extra, steer, summary), 2000,
                            [READER_SEAT, REVIEW_SEAT]) or {}   # smaller budgets got cut off
    batch = got.get("questions")
    return batch[0] if isinstance(batch, list) and batch and isinstance(batch[0], dict) else got


def _g1f_new_enough(ep: Episode, prop_name: str, phrasing: str, again_for: str) -> bool | None:
    """Whether the pick repeats the open re-ask; None when the pick brings nothing new."""
    anchor = ep.split_pending[0] if ep.split_pending else again_for
    repeat = bool(anchor) and _same_tokens(prop_name, anchor)
    if not again_for:
        return None if (not repeat and _ejd_asked_before(ep, prop_name)) else repeat
    # a second try has to name the item's noun
    names_noun = repeat and bool(_word_set(ep.plan.get("product")) & _word_set(phrasing))
    return repeat if names_noun else None


def _draft_plan_once(ep: Episode, box: _Record, extra: str, again_for: str = "") -> _Record | None:
    """One planner call; the resulting pick, or None when nothing new comes of it."""
    summary = ep.requirement_summary()
    ranked = _ejd_rotation(ep, _ejd_ranking(ep))
    steer = None if (again_for or ep.split_pending or not ranked) else ranked
    reply = _g1f_planner_reply(ep, box, extra, steer, summary)
    prop_name, phrasing = (str(reply.get(k) or "").strip() for k in ("property", "question"))
    if steer:
        prop_name, phrasing = _g1f_hold_to_list(ep, steer, reply, prop_name, phrasing)
    if not prop_name or not phrasing:
        return None
    phrasing = _ejd_with_stated_value(ep, prop_name, _g1f_single_product(ep, prop_name, phrasing))
    repeat = _g1f_new_enough(ep, prop_name, phrasing, again_for)
    if repeat is None:
        return None
    odds = _as_number(reply.get("chance"))
    if odds is not None:
        odds = min(1.0, max(0.0, odds))
    return {"property": prop_name, "question": phrasing, "repeat": repeat, "chance": odds}


def _g1f_alike(a: str, b: str) -> bool:
    """Two name words for one thing: equal, or the shorter one (4+ letters) opens the longer."""
    if len(a) > len(b):
        a, b = b, a
    return a == b or (len(a) >= 4 and b.startswith(a))


def _g1f_sibling(ep: Episode, prop_name: str) -> str:
    """Best unasked menu neighbour of prop_name by shared name words (a match of both first words scores two,
    words carried by half the menu or more score nothing; ties go to the property on more listings)."""
    menu = ep.property_menu
    if len(menu) < 2:
        return ""
    words = {name: _token_list(name) for name in menu}
    own = _token_list(prop_name)
    products = _g1f_products(ep)
    slot = _ejd_product_slot(ep, prop_name, products) if len(products) >= 2 else None

    def widespread(term: str) -> bool:
        return 2 * sum(1 for name in menu if any(_g1f_alike(term, w) for w in words[name])) >= len(menu)

    def weight(pos: int, term: str) -> int:
        if widespread(term) or not any(_g1f_alike(term, w) for w in own):
            return 0
        return 2 if pos == 0 and _g1f_alike(term, own[0]) else 1

    best, best_score = "", 0
    for name in list(menu):
        if _same_tokens(name, prop_name) or _ejd_asked_before(ep, name):
            continue
        if slot is not None and _ejd_product_slot(ep, name, products) != slot:
            continue   # never carry a reply across to another product's property
        score = sum(weight(pos, term) for pos, term in enumerate(words[name]))
        if not score:
            continue
        if score > best_score or (score == best_score
                                  and menu[name]["count"] > menu.get(best, {}).get("count", 0)):
            best, best_score = name, score
    return best


def _g1f_spoken(ep: Episode, prop_name: str) -> str:
    """The property name as said to the shopper: a short lead word that heads two or more other listing names
    is a key prefix and is left out, unless the rest would be another property's name."""
    found = _ejd_LEAD_WORD.match(prop_name)
    if found is None:
        return prop_name
    lead, rest = found.group(1), prop_name[found.end():].strip()
    if len(lead) > 3 and len(_token_list(rest)) < 2:
        return prop_name
    names = [n for key, entry in ep.property_menu.items() for n in (key, *(entry.get("also") or []))]
    if any(_same_tokens(rest, n) for n in names):
        return prop_name
    lead = lead.lower()
    heads = {tuple(_token_list(n)) for n in names if not _same_tokens(n, prop_name)}
    return rest if sum(1 for t in heads if len(t) >= 2 and t[0] == lead) >= 2 else prop_name


def _g1f_products(ep: Episode) -> list[str]:
    """Several requested products: the setup reading's lines, or the plan's product phrase cut at ; and ,."""
    lines = [entry["product"] for entry in getattr(ep, "_wanted_items", list)()]
    if lines:
        return lines
    pieces = [bit.strip() for bit in re.split(r"[;,]", str(ep.plan.get("product") or ""))]
    pieces = [bit for bit in pieces if bit]
    return pieces if len(pieces) > 1 else []


def _ejd_own_words(segment: str, products: list[str]) -> set[str]:
    """Words of one product that none of the other products uses."""
    shared: set[str] = set()
    for other in products:
        if other != segment:
            shared |= _word_set(other)
    return _word_set(segment) - shared


def _g1f_owner_by_items(ep: Episode, aliases: set[str], items: list[_Record]) -> str:
    """Vote over recorded listings that carry the property: the requested item most of them belong to."""
    votes: dict[int, int] = {}
    for blob in _iter_dicts(ep.records):
        listing = next((v for v in blob.values() if isinstance(v, str) and v in ep.seen_ids), None)
        if listing is None:
            continue
        labels = {ep._ejb_label(key).strip().lower() for part in _iter_dicts([blob]) for key in part}
        if not aliases & labels:
            continue
        slot = ep._ejc_item_index(listing)
        if slot is not None:
            votes[slot] = votes.get(slot, 0) + 1
    if not votes:
        return ""
    peak = max(votes.values())
    leaders = [slot for slot, n in votes.items() if n == peak]
    return items[leaders[0]]["product"] if len(leaders) == 1 else ""


def _g1f_owner_by_sheets(ep: Episode, aliases: set[str], products: list[str]) -> str:
    """The product whose own words the property's detail sheets show most (earlier product on a tie), with
    leading words the sheets never show trimmed."""
    pool: set[str] = set()
    for sheet in ep._ejb_menu_sheets():
        if aliases & set(sheet):
            for value in sheet.values():
                pool |= _word_set(value)
    hits = [len(_ejd_own_words(seg, products) & pool) for seg in products]
    best = max(range(len(products)), key=lambda pos: (hits[pos], -pos))
    if not hits[best]:
        return ""
    words = products[best].split()
    while len(words) > 1 and _word_set(words[0]) and not _word_set(words[0]) & pool:
        del words[0]
    return " ".join(words)


def _ejd_owner(ep: Episode, prop_name: str) -> str:
    """For a request naming several products: the one the property belongs to, in the request's words; ""
    for a single product or when no single one owns it."""
    products = _g1f_products(ep)
    if not products:
        return ""
    entry = ep.property_menu.get(prop_name) or {}
    aliases = {prop_name.strip().lower()} | {alias.lower() for alias in entry.get("also") or []}
    items = getattr(ep, "_wanted_items", list)()
    if items:
        return _g1f_owner_by_items(ep, aliases, items)
    return _g1f_owner_by_sheets(ep, aliases, products)


def _g1f_single_product(ep: Episode, prop_name: str, phrasing: str) -> str:
    """With several products the question must name exactly one, the property's owner; else our plain question."""
    owner = _ejd_owner(ep, prop_name)
    if not owner:
        return phrasing
    products = _g1f_products(ep)
    heard = _word_set(phrasing)
    named = sum(1 for seg in products if _ejd_own_words(seg, products) & heard)
    if named == 1 and _word_set(owner) & heard:
        return phrasing
    return _ejd_plain(ep, prop_name)


def _ejd_product_slot(ep: Episode, prop_name: str, products: list[str]) -> int | None:
    """Index of the property's owning product; None when no single product owns it."""
    owner = _ejd_owner(ep, prop_name)
    if owner in products:
        return products.index(owner)
    words = _word_set(owner)
    if not words:
        return None
    return next((pos for pos, seg in enumerate(products) if words <= _word_set(seg)), None)


def _g1f_behind(ep: Episode, products: list[str]) -> set[int] | None:
    """Product slots with the fewest questions so far, once asking is uneven across several products."""
    if len(products) < 2 or not ep.asked_props:
        return None
    tally = [0] * len(products)
    for done in ep.asked_props:
        slot = _ejd_product_slot(ep, done, products)
        if slot is not None:
            tally[slot] += 1
    low = min(tally)
    behind = {pos for pos, n in enumerate(tally) if n == low}
    return behind if len(behind) < len(products) else None


def _ejd_turn_due(ep: Episode, prop_name: str) -> bool:
    """True unless the property belongs to a product that has had more questions than another."""
    products = _g1f_products(ep)
    behind = _g1f_behind(ep, products)
    if behind is None:
        return True
    slot = _ejd_product_slot(ep, prop_name, products)
    return slot is None or slot in behind


def _ejd_rotation(ep: Episode, ranked: list[str]) -> list[str]:
    """Several products: keep, in rank order, the candidates of products asked least (and loose stated values);
    hidden needs usually sit on two products, and uncovering both paid about twice as often in the race worlds.
    The full ranking when the filter leaves nothing."""
    products = _g1f_products(ep)
    behind = _g1f_behind(ep, products)
    if behind is None:
        return ranked
    vague = _ejd_loose_props(ep)
    kept = [name for name in ranked if name in vague or _ejd_product_slot(ep, name, products) in behind]
    return kept if kept else ranked


def _ejd_names_product(ep: Episode, prop_name: str, question: str) -> bool:
    """The question mentions, by a word over three letters, the owning product (else the requested product)."""
    noun = _ejd_owner(ep, prop_name) or str(ep.plan.get("product") or "")
    long_stems = {stem for stem in _name_stems(noun) if len(stem) > 3}
    return bool(long_stems & _name_stems(question))


def _ejd_plain(ep: Episode, prop_name: str) -> str:
    """Our own question about one property, naming the owning product, or a short (two-word) product noun."""
    said = _g1f_spoken(ep, prop_name)
    owner = _ejd_owner(ep, prop_name)
    if not owner:
        noun = str(ep.plan.get("product") or "").strip()
        owner = noun if noun and len(noun.split()) <= 2 else ""
    if owner:
        return f"Do you have a preference for the {said} of the {owner}?"
    return f"Do you have a preference for the {said}?"


def _ejd_names_sample(ep: Episode, pick: _Record) -> bool:
    """The question quotes one of the property's sample values (beyond values the request itself names), and
    no longer menu name contains the property's words."""
    menu, prop_name = ep.property_menu, pick["property"]
    entry = menu.get(prop_name) or next((menu[key] for key in menu if _same_tokens(key, prop_name)), None)
    own = _word_set(prop_name)
    if not entry or any(own < _word_set(key) for key in menu):
        return False
    heard = _word_set(pick["question"]) - own - _word_set(ep.plan.get("product"))
    requested = _word_set((getattr(ep, "policy", None) or {}).get("query"))
    for value in entry["values"]:
        vw = _word_set(value)
        if vw and vw <= heard and not vw <= requested:
            return True
    return False


def _ejd_sharper(pick: _Record | None, pending: tuple[str, str]) -> bool:
    """A re-ask after a split reply: the same property in different words."""
    if pick is None or not pick["repeat"]:
        return False
    return _same_tokens(pick["property"], pending[0]) and _token_list(pick["question"]) != _token_list(pending[1])


def _ejd_worth(ep: Episode, pick: _Record) -> bool:
    """Priced questions go out only while expected gain beats their share; a price without a usable share
    leaves room for one question. A refusal here also stops the planner."""
    price = ep.question_cost
    if not price:
        return True
    share = _as_number(price.get("share"))
    if share is not None and 0 <= share <= 1:
        odds = pick["chance"]
        ok = (0.5 if odds is None else odds) * _ejd_REVEAL_ODDS > share
    else:
        ok = not ep.asked_props and not ep.split_pending
    if not ok:
        ep.planner_on = False
    return ok


def _g1f_idle_planner(ep: Episode, box: _Record) -> None:
    """Nothing sendable this round: below two open fact sets there is no menu yet, so the planner only pauses;
    it stops for good after three empty rounds or once two sheets were open."""
    ep.empty_plans += 1
    ep.urgent_tried = True
    if ep.empty_plans >= 3 or box.get("sheets", 0) >= 2:
        ep.planner_on = False
    box["exit"] = False


def _stage_question_build(ep: Episode, box: _Record) -> _Record:
    prepared = []
    for pick in box["picks"]:
        wording = str(pick.get("question")).strip()
        call = ep._build_message_call(wording, "ask_%d_%d" % (ep.messages_used + 1, ep.step))
        if call is not None:
            prepared.append((call, (str(pick.get("property")), wording, bool(pick.get("repeat")))))
    if not prepared:
        _g1f_idle_planner(ep, box)
        return box
    calls, topics = zip(*prepared)
    box["calls"], box["topics"] = list(calls), list(topics)
    return box


_g1f_REPLY_NOTE = ("\n[agent note] Their answer was added, word for word, as a requirement. Good next step: a "
                   "search combining it with the product type, and afterwards a fresh check of the chosen item "
                   "and of each listing already opened.")


def _stage_question_send(ep: Episode, box: _Record) -> _Record:
    outgoing, subjects = box["calls"], box["topics"]
    revision = ep.needs_revision
    about = "; ".join(topic[0] for topic in subjects)
    ep.messages.append(dict(role="assistant", content="Planned question(s) to the shopper on: %s." % about,
                                 tool_calls=outgoing))
    ep.asking, ep.answer_reading = subjects[0][0], None
    replies = ep._record_replies(ep._run_calls(outgoing, planned=True))
    for (prop_name, question, repeat), text in zip(subjects, replies):
        ep.asked.append('[%s] "%s" => %s' % (prop_name, question, text[:400]))
        verdict = (ep.answer_reading or {}).get("class", "other")
        _g1f_file_answer(ep, prop_name, question, repeat, verdict)
    ep.asking = ""
    if ep.needs_revision != revision and ep.messages and ep.messages[-1].get("role") == "tool":
        ep.messages[-1]["content"] += _g1f_REPLY_NOTE
    ep.planner_on = box["cap"] > ep.messages_used
    box["exit"] = True
    return box


def _g1f_later(ep: Episode, prop_name: str) -> None:
    if prop_name not in ep.reask_later:
        ep.reask_later.append(prop_name)


def _g1f_waved_off(ep: Episode, prop_name: str, question: str) -> None:
    """A not-picky reply: accept any listed value for a loose wish the question named, queue one neighbour,
    and keep a late re-ask when the reply may have meant another property."""
    wish = _ejd_loose_props(ep).get(prop_name)
    asked = _word_set(question)
    if wish and _ejd_clean_want(wish) <= asked:
        ep.loose_ok[prop_name] = wish
    # neighbours do not chain: a waved-off neighbour opens none of its own
    ep.sibling_next = "" if prop_name in ep.sibling_props else _g1f_sibling(ep, prop_name)
    label = _word_set(prop_name)
    if not (asked & _word_set(ep.plan.get("product")) or (len(label) >= 2 and label <= asked)):
        _g1f_later(ep, prop_name)


def _g1f_file_answer(ep: Episode, prop_name: str, question: str, repeat: bool, verdict: str) -> None:
    """Close, reopen or park the property according to the reader's class for its reply."""
    if repeat:
        ep.split_retried = True
        ep.split_pending = None
    elif verdict == "split":
        ep.split_pending = (prop_name, question)   # one narrower re-ask follows
        return
    elif verdict == "unclear":
        if not _ejd_names_product(ep, prop_name, question):
            ep.unclear_next = prop_name   # re-ask once naming the product; still open
            return
        # the product was already named, so new words seldom help: one late try after the others
        _g1f_later(ep, prop_name)
    elif verdict == "not_picky":
        _g1f_waved_off(ep, prop_name, question)
    if not _ejd_asked_before(ep, prop_name):
        ep.asked_props.append(prop_name)


# ================================================================ call screening (a gate returns None or a reply)

def _gate_unknown_tool(ep: Episode, ctx: _Record, req: _Record) -> str | None:
    name = req["name"]
    if name in ep.param_schemas:
        return None
    return "%s: no tool is called '%s'. Tools available now: %s." % (_SKIP_NOTICE, name, ", ".join(sorted(ctx["allowed"])))


def _gate_not_offered(ep: Episode, ctx: _Record, req: _Record) -> str | None:
    name = req["name"]
    if ctx["planned"] or name in ctx["allowed"]:
        return None
    if not (ep.planner_on and name == ctx["message"]):
        return "%s: the step limit is close; only %s remain." % (_SKIP_NOTICE, ", ".join(sorted(ctx["allowed"])))
    return (_SKIP_NOTICE + ": skip messaging - shopper questions are sent for you and their replies arrive on their "
            "own. Spend this turn on the listings instead: read details, put a good fit in the cart, then order.")


def _gate_bad_json(ep: Episode, ctx: _Record, req: _Record) -> str | None:
    raw = req["raw"]
    if isinstance(raw, dict):
        parsed = raw
    else:
        parsed = _extract_json_dict(raw) if (raw or "").strip() else {}
    req["args"] = parsed
    return None if parsed is not None else _SKIP_NOTICE + ": the arguments could not be read; send one JSON object."


def _gate_schema_misfit(ep: Episode, ctx: _Record, req: _Record) -> str | None:
    fixed, issues = _validate_value(req["args"], ep.param_schemas[req["name"]], "args")
    req["args"] = fixed
    if issues:
        return _SKIP_NOTICE + ": schema check failed - " + "; ".join(issues[:6])
    return None


def _gate_finish_alone(ep: Episode, ctx: _Record, req: _Record) -> str | None:
    if ctx["count"] > 1 and req["name"] in ctx["finishing"]:
        return (_SKIP_NOTICE + ": an order or decline call must be the only call in its reply. Let the other "
                "results come back, then send it by itself.")
    return None


def _gate_known_fault(ep: Episode, ctx: _Record, req: _Record) -> str | None:
    name, args = req["name"], req["args"]
    if not args or name in ctx["finishing"]:
        return None
    seen_fault = ep._call_shape(name, args) in ep.faulty or name + _to_compact_json(args) in ep.faulty
    if not seen_fault:
        return None
    return ("%s: calls shaped like this have failed twice on %s with one and the same error. Fix the arguments "
            "to its schema to the letter, or pick some other tool." % (_SKIP_NOTICE, name))


def _gate_tool_quota(ep: Episode, ctx: _Record, req: _Record) -> str | None:
    budget = ep.message_budget
    if budget is None or req["name"] != ctx["message"]:
        return None
    if ep.messages_used + ctx["queued_messages"] >= budget:
        return _SKIP_NOTICE + ": the message allowance is used up; the shopper cannot be messaged again."
    return None


def _gate_repeat_call(ep: Episode, ctx: _Record, req: _Record) -> str | None:
    name = req["name"]
    sig = req["sig"] = name + _to_compact_json(req["args"])
    exempt = name in (ctx["message"], ctx["cart"]) or name in ctx["finishing"]
    if exempt or sig not in ep.recent:
        return None
    return (_SKIP_NOTICE + ": this repeats your last call exactly. Its result above is unchanged; change the "
            "arguments or do something else.")


def _g1f_requested_ids(req: _Record) -> list[str]:
    return [value for value in req["args"].values() if isinstance(value, str)]


def _gate_late_details(ep: Episode, ctx: _Record, req: _Record) -> str | None:
    """Close to the end after a refusal, details reads are kept for cheaper seen listings still unopened (see
    _tools_this_turn); a read of any other listing is skipped and the allowed ones are named."""
    if req["name"] != ep.roles.get("details"):
        return None
    if ep.remaining_steps() > 7 or not ep.refusals:
        return None
    allowed = _rows_to_recheck(ep)
    if not allowed or any(value in allowed for value in _g1f_requested_ids(req)):
        return None
    return ("%s: this close to the end a details read is only for a cheaper listing never opened: %s. For the item "
            "you hold, the cart view already shows price and stock." % (_SKIP_NOTICE, ", ".join(allowed)))


def _gate_late_unfitted(ep: Episode, ctx: _Record, req: _Record) -> str | None:
    """Close to the end with no order tried, details reads stay open only while some product lacks a fit (see
    _ejb_allowed_jobs): on its unopened on-target rows, one at a time, down to the reserve plus two."""
    tool = ep.roles.get("details")
    left = ep.remaining_steps()
    can_finish = ep.roles.get("order") and ep.roles.get("cart") and not ep.order_attempts
    if req["name"] != tool or left > 7 or ep.refusals or not can_finish:
        return None
    in_flight = sum(1 for _, name, _ in ctx.get("ready") or [] if name == tool)
    allowed = _eje_unfitted_rows(ep, 2)
    if not allowed:
        return _SKIP_NOTICE + ": " + _format_note("reserve_late")
    if left - in_flight >= ep._reserved_steps() + 2 and any(v in allowed for v in _g1f_requested_ids(req)):
        return None
    return ("%s: this close to the end one last details read remains, for a product still without a fit: one of "
            "%s." % (_SKIP_NOTICE, ", ".join(allowed)))


def _gate_cart_review(ep: Episode, ctx: _Record, req: _Record) -> str | None:
    tool, args = req["name"], req["args"]
    if tool != ctx["cart"] or ep.remaining_steps() <= 3:
        return None
    add_sig = ep.add_signature
    if add_sig is not None and add_sig != ep._choice_signature(tool, args):
        return None   # this cart call is not an add
    target = _cart_listing_id(ep, tool, args)
    if not target:
        return None
    tag = "%s|%s" % (target, ep._arm_signature())
    if tag in ep.cart_checked:
        return None
    ep.cart_checked.add(tag)
    doubt = ep._verify_item(target, "being added to the cart")
    if not doubt:
        return None
    return "%s. %s If you are sure it fits regardless, resend it unchanged." % (_SKIP_NOTICE, doubt)


def _gate_unit_count(ep: Episode, ctx: _Record, req: _Record) -> str | None:
    if req["name"] != ctx["cart"]:
        return None
    issue = ep._units_problem(req["name"], req["args"])
    if issue and req["sig"] not in ep.qty_warned:
        ep.qty_warned.add(req["sig"])
        return "%s: %s If you are sure, resend it unchanged." % (_SKIP_NOTICE, issue)
    return None


def _gate_stacked_add(ep: Episode, ctx: _Record, req: _Record) -> str | None:
    """Adding a listing the cart already holds may stack units: hold such an add once per cart state, stating
    the units the cart shows and the units the request wants."""
    tool, args = req["name"], req["args"]
    if tool != ctx["cart"] or ep.add_signature is None:
        return None
    if ep._choice_signature(tool, args) != ep.add_signature:
        return None
    held = next((v for v in args.values() if isinstance(v, str) and v in ep.cart_ids), None)
    if held is None:
        return None
    target = ep._wanted_units()
    if ep._wanted_items():
        target = ep._ejc_line_units(held) or max(line["quantity"] for line in ep._wanted_items())
    fresh_view = ep.cart_seen_at >= ep.cart_changed_at
    shown = ep.units_in_cart.get(held) if fresh_view else None
    if shown is not None and shown + ep._requested_units(tool, args) <= target:
        return None   # the wanted count is not reached yet
    tag = "readd %s %s" % (held, ep.cart_changed_at)
    if tag in ep.held_arms:
        return None
    ep.held_arms.add(tag)
    view = "" if shown is None else " (the newest cart view shows %s)" % shown
    return ("%s Note: %s is already in the cart%s. Adding it again may raise the units above %s, the count the "
            "request asks for. Either order the cart as it is, or remove the item first if the count has to "
            "change. Resend unchanged only if more units really are wanted." % (_HELD_NOTICE, held, view, target))


def _gate_reread_after_refusal(ep: Episode, ctx: _Record, req: _Record) -> str | None:
    """After a refused order the next order waits until the held items' price and stock were read again."""
    if ep._classify_call(req["name"], req["args"]) != "order":
        return None
    if not ep.refusals or not ep.cart_ids or not ep._has_step_room(1):
        return None
    cutoff = max(ep.refused_at, ep.cart_changed_at)
    watched = ep.cart_ids if ep._wanted_items() else ep.cart_ids[-1:]
    stale = [lid for lid in watched if ep.mentioned.get(lid, -1) <= cutoff]
    tag = "reread %s %s" % (ep.refused_at, ep.cart_changed_at)
    if not stale or tag in ep.held_arms:
        return None
    ep.held_arms.add(tag)
    return ("%s Hold the order until %s gets a fresh look: since the refusal (or the newest cart edit) nobody has "
            "read its price, stock or cart units. One tool call that shows those - units appear in the cart view - "
            "comes first; then order. Already done just now? Resend as it is." % (_HELD_NOTICE, ", ".join(stale)))


# ================================================================ price rows and paged searches

def _find_cheaper_rows(ep, pick: str, row_map: dict[str, _Record], unopened_only: bool,
                     ties: bool = False) -> list[str]:
    anchor = row_map.get(pick) or {}
    ceiling, section = anchor.get("price"), anchor.get("label")
    if ceiling is None:
        return []
    found = []
    for rid, row in row_map.items():
        price = row["price"]
        if rid == pick or price is None:
            continue
        if not (price <= ceiling if ties else price < ceiling):
            continue
        if not ep._within_price_cap(row):
            continue
        if section is not None and row["label"] != section:
            continue
        if ep._row_passes_literal(row) is False:
            continue
        if unopened_only and row["opened"]:
            continue
        found.append((price, rid))
    return [rid for _, rid in sorted(found)]


def _rows_to_recheck(ep, row_map: dict[str, _Record] | None = None) -> list[str]:
    """After a refused order on a cheapest request: unopened in-budget rows in the held listing's section at or
    below its latest price, cheapest first, as many as steps allow (a price rise or a lost listing can make one
    of them the cheapest fit, and only an opened row can be checked)."""
    held = ep.cart_ids[-1] if ep.cart_ids else None
    if held is None or not ep.refusals:
        return []
    if (getattr(ep, "plan", None) or {}).get("priority") != "cheapest":
        return []
    left = ep.remaining_steps()
    if left < 4:   # the reserve exists for this case: open, swap and order take four steps
        return []
    table = row_map if row_map is not None else ep._row_table()
    return _find_cheaper_rows(ep, held, table, True, True)[:max(1, min(4, left - 3))]


def _find_paging_args(ep, tool: str) -> tuple[str | None, int | None]:
    """The search tool's paging argument: an optional integer with a maximum (the first row shown)."""
    schema = ep.param_schemas.get(tool) or {}
    required = set(schema.get("required") or [])
    for key, spec in (schema.get("properties") or {}).items():
        if key in required or not isinstance(spec, dict):
            continue
        if spec.get("type") == "integer" and "maximum" in spec:
            return key, int(spec["maximum"])
    return None, None


def _g1f_row_json(row: _Record) -> str:
    return json.dumps(row, sort_keys=True)


def _g1f_row_has(row: _Record, text: str) -> bool:
    return any(isinstance(cell, str) and cell == text for cell in row.values())


class _ejd_PagedSearch:
    """One search (its arguments without the paging one) and what its pages have shown so far."""

    def __init__(self, query: _Record):
        self.query = query
        self.total = self.size = 0
        self.seen: set[int] = set()
        self.holes: list[int] = []
        self.lead_block = None
        self.in_section = False

    def take(self, start: int, shown: list[_Record], totals: list[int], label: str | None) -> None:
        count = len(shown)
        self.total = max([self.total, *totals])
        self.size = max(self.size, count)
        self.seen.update(start + i for i in range(count))
        if not start and self.lead_block is None:
            self.lead_block = tuple(map(_g1f_row_json, shown))
        if not self.in_section:
            self.in_section = label is None or any(_g1f_row_has(row, label) for row in shown)

    def term_set(self) -> set[str]:
        return _word_set(" ".join(map(str, self.query.values())))

    def key(self) -> str:
        return _g1f_row_json(self.query)


def _g1f_search_step(entry: _Record, tool: str) -> tuple[_Record | None, object] | None:
    """(arguments, observation) of a recorded world step whose first action used the tool, else None."""
    reply = entry.get("reply")
    if entry.get("kind") != "env" or not isinstance(reply, dict):
        return None
    lead = (entry.get("actions") or [{}])[0]
    return (lead.get("args"), reply.get("observation")) if lead.get("name") == tool else None


def _ejd_search_pages(ep, tool: str, offset: str):
    """Yield (query without paging, first row index, inner result) per recorded call of the search tool."""
    for entry in ep.trace:
        step = _g1f_search_step(entry, tool)
        if step is None:
            continue
        params = dict(step[0] or {})
        first_row = params.pop(offset, 0)
        inner = _unwrap_text(step[1])[1]
        if isinstance(first_row, int) and isinstance(inner, dict):
            yield params, first_row, inner



# ================================================================ reading back what the search tool did

def _g1g_ints(body, floor):
    """Whole numbers (no booleans) of a reply at or above `floor`: candidate match counts."""
    return [n for n in body.values() if isinstance(n, int) and not isinstance(n, bool) and n >= floor]


def _g1g_table(body):
    """The longest list made only of dicts in a reply (first such on a tie), or []."""
    best = []
    for part in body.values():
        if isinstance(part, list) and part and all(isinstance(cell, dict) for cell in part) and len(part) > len(best):
            best = part
    return best


def _g1g_body(entry):
    """The parsed result dict of one env trace entry, or None."""
    answer = entry.get("reply")
    if not isinstance(answer, dict):
        return None
    parsed = _unwrap_text(answer.get("observation"))[1]
    return parsed if isinstance(parsed, dict) else None


def _g1g_calls(agent):
    """[(trace entry, sent args)] for every recorded call of the search tool, oldest first."""
    search_tool = agent.roles.get("search")
    if not search_tool:
        return []
    found = []
    for entry in agent.trace:
        if entry.get("kind") == "env":
            first = (entry.get("actions") or [{}])[0]
            if first.get("name") == search_tool:
                found.append((entry, first.get("args") or {}))
    return found


def _g1g_answered(agent):
    """[(sent args, result dict, row list)] for search calls that came back with a dict result."""
    out = []
    for entry, sent in _g1g_calls(agent):
        parsed = _g1g_body(entry)
        if parsed is not None:
            out.append((sent, parsed, _g1g_table(parsed)))
    return out


def _g1g_props(agent, tool):
    if not tool:
        return {}
    spec = agent.param_schemas.get(tool) or {}
    return spec.get("properties") or {}


def _g1g_ctx(agent):
    """(search tool or None, its offset argument or None, its argument specs)."""
    name = agent.roles.get("search")
    if not name:
        return None, None, {}
    return name, _find_paging_args(agent, name)[0], _g1g_props(agent, name)


def _g1g_listing_id(cells):
    """The first id-shaped text under a key naming an id."""
    for name, val in cells.items():
        if isinstance(val, str) and "id" in str(name).lower() and _ID_SHAPE.match(val):
            return val
    return None


def _g1g_group_pages(run, tool, offset, label):
    """Paged searches keyed by their arguments without the offset, in first-seen order."""
    groups = {}
    for query, start, parsed in _ejd_search_pages(run, tool, offset):
        shown = _g1g_table(parsed)
        counts = _g1g_ints(parsed, len(shown))
        if not shown or not counts:
            continue
        sig = json.dumps(query, sort_keys=True)
        pager = groups.get(sig)
        if pager is None:
            pager = groups[sig] = _ejd_PagedSearch(query)
        pager.take(start, shown, counts, label)
    return list(groups.values())


def _g1g_vocab(run):
    """(words of the wanted values, words a search may type and stay on target): the wanted words, the product
    and every store wording that carries a wanted value under a field naming its property."""
    try:
        conds = run._list_conditions()
    except AttributeError:   # request not read yet
        conds = []
    wanted = set()
    for cond in conds:
        wanted |= _word_set(str(cond).split(":", 1)[-1])
    allowed = wanted | _word_set((getattr(run, "plan", None) or {}).get("product"))
    try:
        conds = run._list_conditions()
        listings = run._row_table()
    except AttributeError:
        return wanted, allowed
    pairs = []
    for cond in conds:
        prop, _, val = str(cond).partition(":")
        pairs.append((_word_set(prop), _word_set(val)))
    pairs = [(prop, val) for prop, val in pairs if val]
    for entry in listings.values():
        for name, text in (entry.get("facts") or {}).items():
            for prop, val in pairs:
                if (prop <= _word_set(name) or _key_names_property(name, prop)) and val & _word_set(text):
                    allowed |= _word_set(text)
    return wanted, allowed


def _partial_search_views(session, label: str | None, singular: bool = False) -> list[tuple[int, _Record, int, int]]:
    """Offset-paged searches with rows still unseen, as (total, next-page args, rows seen,
    pages left). Kept: shows a `label` row, has a page size, has a gap before one page past the cap. Same lead
    block and total -> one, fewest words wins. Typed searches outside the wanted/product/store vocabulary drop out
    while an on-target one exists (`singular`: by stems). Order: wanted-word coverage, then total."""
    name = session.roles.get("search")
    offset, cap = _find_paging_args(session, name) if name else (None, None)
    if offset is None:
        return []
    pagers = _g1g_group_pages(session, name, offset, label)
    wanted, allowed = _g1g_vocab(session)
    props = _g1g_props(session, name)

    def free_words(pager):   # words typed into free-text fields; choice fields do not count
        return _word_set(" ".join(str(val) for name, val in pager.query.items()
                                  if isinstance(val, str) and not (props.get(name) or {}).get("enum")))

    live = []
    for pager in pagers:
        pager.holes = [pos for pos in range(min(pager.total, cap + pager.size)) if pos not in pager.seen]
        if pager.holes and pager.in_section and pager.size:
            live.append(pager)
    live.sort(key=lambda pager: (len(pager.term_set()), pager.key()))
    kept = []
    for pager in live:
        twin = [other for other in kept
                if other.lead_block and other.lead_block == pager.lead_block and other.total == pager.total]
        if not twin:
            kept.append(pager)
    if singular:
        stems = _name_stems(" ".join(allowed))
        on_target = [p for p in kept if free_words(p) and _name_stems(" ".join(free_words(p))) <= stems]
    else:
        on_target = [p for p in kept if free_words(p) and free_words(p) <= allowed]
    if wanted and on_target:   # untyped browsing stays, off-target typing steps aside
        kept = [p for p in kept if p in on_target or not free_words(p)]
    rows = []
    for pager in kept:
        ahead = dict(pager.query)
        ahead[offset] = pager.holes[0]
        rows.append(((-len(wanted & pager.term_set()), pager.total, json.dumps(ahead, sort_keys=True)),
                     (pager.total, ahead, len(pager.seen), -(-len(pager.holes) // pager.size))))
    rows.sort(key=lambda row: row[0])
    return [view for _, view in rows]


def _eje_bare_values(agent) -> list[str]:
    """Required values in the request's words, cut before a bracket, comma, semicolon or exclusion, unique, at most
    33 characters; number relations are checked, not searched, so they are skipped."""
    try:
        conds = agent._list_conditions()
    except AttributeError:   # request not read yet
        return []
    out = []
    for cond in conds:
        text = str(cond).partition(":")[2]
        if _parse_relation(text) is not None:
            continue
        head = re.split(r"[(,;]|!=", text, maxsplit=1)[0].strip().strip("\"'.")
        if head and len(head) <= 33 and head not in out:
            out.append(head)
    return out


def _g1g_alone_searched(agent, value):
    """True once a search typed some text whose words all belong to `value`."""
    mine = _word_set(value)
    for _, sent in _g1g_calls(agent):
        for val in sent.values():
            if isinstance(val, str):
                words = _word_set(val)
                if words and words <= mine:
                    return True
    return False


def _g1g_free_typed(sent, paging, props):
    """Words in one search's free-text arguments (no choice fields, no paging argument)."""
    words = set()
    for name, val in sent.items():
        spec = props.get(name)
        if name == paging or not isinstance(val, str) or (isinstance(spec, dict) and "enum" in spec):
            continue
        words |= _word_set(val)
    return words


def _g1g_row_hits(agent):
    """One record per listing row a search showed: (id, wanted values typed, match total, offset position,
    position in its page)."""
    _, paging, props = _g1g_ctx(agent)
    targets = [w for w in (_word_set(v) for v in _eje_bare_values(agent)) if w]
    hits = []
    for sent, parsed, shown in _g1g_answered(agent):
        first = sent.get(paging, 0) if paging else 0
        if isinstance(first, bool) or not isinstance(first, int):
            first = 0
        typed = _g1g_free_typed(sent, paging, props)
        named = sum(1 for w in targets if w <= typed)
        total = max(_g1g_ints(parsed, len(shown)), default=len(shown))
        for spot, cells in enumerate(shown):
            ident = _g1g_listing_id(cells)
            if ident is not None:
                hits.append((ident, named, total, first + spot, spot))
    return hits


def _g1g_best_named(agent):
    """id -> (most wanted values typed by a search showing it, earliest offset position)."""
    best = {}
    for ident, named, _, at, _ in _g1g_row_hits(agent):
        old = best.get(ident)
        best[ident] = (named, at) if old is None else (max(old[0], named), min(old[1], at))
    return best


def _g1g_sharpest(agent):
    """id -> least of (minus wanted values typed, match total, page position) over the searches showing it."""
    best = {}
    for ident, named, total, _, spot in _g1g_row_hits(agent):
        mark = (-named, total, spot)
        if ident not in best or mark < best[ident]:
            best[ident] = mark
    return best


def _g1g_differing(key, mine, theirs, here, there):
    """Classify the one differing argument of two identical-row searches; None if not comparable."""
    if mine is None or theirs is None:
        val, holder = (mine, here) if theirs is None else (theirs, there)
        return True, "%s %s" % (key, val), '%s "%s"' % (key, val), _word_set(val), holder
    if not isinstance(mine, str) or not isinstance(theirs, str):
        return None
    a, b = _word_set(mine), _word_set(theirs)
    if a < b:
        holder, wide, wide_words, extra = there, theirs, b, b - a
    elif b < a:
        holder, wide, wide_words, extra = here, mine, a, a - b
    else:   # unrelated words, same rows: the store ignored the argument
        return True, key, '%s (both "%s" and "%s" returned identical rows)' % (key, mine, theirs), a ^ b, here
    quoted = '"%s"' % " ".join(w for w in re.findall(r"[a-z0-9]+", wide.lower()) if w in extra)
    return False, quoted, quoted, extra, holder


def _eje_idle_parts(agent) -> list[str]:
    """Arguments the store seems to ignore. A pair of calls with equal rows and total that differ in one
    non-paging argument, where a row lacks its words, yields a note; whole arguments lead, newest first, one per
    tag."""
    paging = _g1g_ctx(agent)[1]
    earlier = []
    whole, partial, tags = [], [], set()
    for sent, parsed, shown in _g1g_answered(agent):
        if not shown:
            continue
        now = dict(sent)
        stamp = json.dumps([max(_g1g_ints(parsed, len(shown)), default=None), shown], sort_keys=True)
        stem_rows = [_name_stems(" ".join(str(v) for v in cells.values() if not isinstance(v, (dict, list))))
                     for cells in shown]
        for before, before_stamp in earlier:
            if before_stamp != stamp:
                continue
            keys = [k for k in set(now) | set(before) if now.get(k) != before.get(k)]
            if len(keys) != 1 or keys[0] == paging:
                continue
            got = _g1g_differing(keys[0], now.get(keys[0]), before.get(keys[0]), now, before)
            if got is None:
                continue
            entire, tag, wording, words, holder = got
            if not words or tag in tags:
                continue
            probe = _name_stems(" ".join(words))
            if all(probe <= stems for stems in stem_rows):
                continue
            tags.add(tag)
            (whole if entire else partial).insert(0, "%s in %s" % (wording, json.dumps(holder, sort_keys=True)))
        earlier.append((now, stamp))
    return whole + partial


def _eje_view_completes(agent, view: tuple[int, dict[str, Any], int, int]) -> bool:
    """True when paging this view up to the paging maximum would show every one of its matches."""
    tool = agent.roles.get("search")
    cap = _find_paging_args(agent, tool)[1] if tool else None
    if cap is None:
        return False
    total, _, seen, left = view
    size = -(-(total - seen) // left) if left else 0
    return total <= cap + max(size, 1)


def _eje_page_list(views) -> str:
    """Each unfinished search by its arguments, with its read count."""
    parts = []
    for total, args, seen, _ in views:
        parts.append("%s (%s of %s rows read)" % (json.dumps(args, sort_keys=True), seen, total))
    return "; ".join(parts)


def _eje_shape(text) -> str:
    """The text with each digit run turned into one mark, so a handed-out page value and an invented one match."""
    return re.sub(r"\d+", "#", str(text))


def _eje_page_keys(agent) -> list[str]:
    """Optional, non-choice integer or string arguments of the search tool (where a page position can go)."""
    schema = agent.param_schemas.get(agent.roles.get("search")) or {}
    required = set(schema.get("required") or [])
    out = []
    for name, field in (schema.get("properties") or {}).items():
        if name in required or not isinstance(field, dict) or "enum" in field:
            continue
        if field.get("type") in ("integer", "string"):
            out.append(name)
    return out


def _eje_offered_next(body, kind, start) -> list:
    """What a reply offers for its next page: for int, whole numbers above `start` and under the largest number;
    otherwise short spaceless top-level texts."""
    plain = [v for v in (body or {}).values() if isinstance(v, (int, str)) and not isinstance(v, bool)]
    if kind is not int:
        return [v for v in plain if isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_.:=+/-]{1,64}", v)]
    nums = [v for v in plain if isinstance(v, int)]
    if not nums:
        return []
    low, top = start or 0, max(nums)
    return [v for v in nums if low < v < top]


def _eje_prior_page(agent, key, sent_args):
    """(value of `key`, result) of the newest answered search equal to `sent_args` outside `key`, or None."""
    others = {k: v for k, v in sent_args.items() if k != key}
    for entry, past in reversed(_g1g_calls(agent)):
        parsed = _g1g_body(entry)
        if parsed is not None and {k: v for k, v in past.items() if k != key} == others:
            return past.get(key), parsed
    return None


def _eje_text_page_key(agent) -> str | None:
    """Name of the string argument used as a page token: only ever filled with handed-out values (or
    the one never used yet). None unless a single name fits and some reply handed out a value."""
    specs = _g1g_props(agent, agent.roles.get("search"))
    names = [k for k in _eje_page_keys(agent) if specs[k].get("type") == "string"]
    given, good, bad = set(), set(), set()
    for entry, sent in _g1g_calls(agent):
        for k in names:
            if k in sent:
                if isinstance(sent[k], str) and sent[k] in given:
                    good.add(k)
                else:
                    bad.add(k)
        given.update(_eje_offered_next(_g1g_body(entry), str, 0))
    choice = [k for k in names if k in good and k not in bad]
    if not choice:
        choice = [k for k in names if k not in good and k not in bad]
    if len(choice) == 1 and given:
        return choice[0]
    return None


def _g1g_token_views(agent, label):
    """Like the offset views, for stores whose next page is a token the reply hands out; size comes
    from the first page."""
    name = agent.roles.get("search")
    if not name or _find_paging_args(agent, name)[0] is not None:
        return []
    key = _eje_text_page_key(agent)
    if key is None:
        return []
    book = {}
    for entry, sent in _g1g_calls(agent):
        parsed = _g1g_body(entry)
        if parsed is None:
            continue
        shown = _g1g_table(parsed)
        base_args = {k: v for k, v in sent.items() if k != key}
        sig = json.dumps(base_args, sort_keys=True)
        if sig not in book:
            book[sig] = {"args": base_args, "seen": set(), "total": 0, "size": len(shown), "hit": False, "next": None}
        rec = book[sig]
        rec["seen"].update(json.dumps(cells, sort_keys=True) for cells in shown)
        rec["total"] = max([rec["total"]] + _g1g_ints(parsed, len(shown)))
        if not rec["hit"]:
            rec["hit"] = label is None or any(
                label in [v for v in cells.values() if isinstance(v, str)] for cells in shown)
        fresh = [v for v in _eje_offered_next(parsed, str, 0) if v != sent.get(key)]
        rec["next"] = fresh[0] if len(fresh) == 1 else None
    out = []
    for rec in book.values():
        got = len(rec["seen"])
        if rec["next"] is None or not rec["hit"] or not rec["size"] or got >= rec["total"]:
            continue
        ahead = dict(rec["args"])
        ahead[key] = rec["next"]
        out.append((rec["total"], ahead, got, -(-(rec["total"] - got) // rec["size"])))
    out.sort(key=lambda view: (view[3], json.dumps(view[1], sort_keys=True)))
    return out


# ================================================================ hold notes shown to the model

# templates keyed by note name; braces are filled per call
_eje_NOTES = dict(
    # cheapest-pick checks
    relax="Prices must be compared again: {change} is no longer as before, and the request takes only the "
          "cheapest acceptable item. Query the catalogue for the product once more leaving out what was "
          "dropped, and weigh the results.{seen} When your pick is still cheapest, resend exactly the same "
          "order call.",
    relax_seen=" Lower-priced rows seen before that may fit now: {ids}.",
    cheap_rows="Cheapest fit required. Listed but never opened, same section, in budget and cheaper than your "
               "pick: {ids}. Go through them in this order; the first one fitting every requirement is the buy. "
               "If none fits, resend exactly the same order call.",
    cheap_pages="Cheapest fit required. Rows carry prices, and these queries still have matches nobody has seen: "
                "{pages}. Page on through them, open lower-priced in-budget rows of that section, and buy the "
                "cheapest full fit; when your pick stays lowest, resend exactly the same order call.",
    cheap_bare="Cheapest fit required. So far every query mixed extra words into the required values, and that "
               "hides items the store words otherwise. Run a query of just {values}{scope}; read all its pages. "
               "Afterwards open in-budget rows below your pick's price and buy the cheapest full fit - or, when "
               "your pick is still lowest, resend exactly the same order call.",
    cheap_scope=" (inside section \"{label}\", when the tool can limit to it)",
    # cart removal
    release="Likely keep {target}: priced {price}{within}, it still meets the request on its own data. Leave "
            "it in place unless some message says it is gone or not wanted; resending the same cart call "
            "removes it anyway.",
    release_budget=", under the limit of {budget}",
    # held-item checks
    fit_unread="Nothing about {held} is confirmed: its details were never opened. Open them, check each "
               "requirement, then buy; if you just did, resend the same order call.",
    fit_money="{held} {why} by the latest data - something moved mid-session. Open its listing once more. If "
              "confirmed, drop it and go for the next item still acceptable (redo the price comparison if "
              "cheapest is wanted). The same order call is for after a fresh look proves it fine again.",
    fit_sold="is now listed as out of stock",
    fit_dear="now costs {price} each, above the total budget of {budget}",
    fit_unmet="Unconfirmed for {held}: {unmet}. Its own listing either gives a different value or says nothing "
              "where most of its section does.{pointer} The order should go to an item whose listing confirms "
              "each requirement; resend the same order call only if {held} really does.",
    fit_others=" Opened items that meet every requirement: {ids}.",
    fit_words=" The request's own phrasing: {phrases}. Search with that phrasing plus the product and open the "
              "items that mention it.",
    # search steering
    page_arg="Page jump spotted. Value {value} was never handed out: for its next page the store's latest "
             "reply to this query offered {offered}, and anything else skips rows or repeats them. To read on "
             "send {call}; resending this call jumps anyway.",
    flood="Searches sent: {count}. Listings opened: none. Look at what is already on screen before typing "
          "more queries - start with these unopened rows from the wanted sections, best-aimed query first: "
          "{ids}.{mixed} After that, query only for gaps they leave; resending this exact call searches "
          "anyway.",
    flood_mixed="A note on mode: rows from the worded queries landed mostly outside the chosen section, so that "
                "choice appears to have no effect there; joining one required value with the product noun keeps "
                "results in place - say {example}.",
    page_fresh="Page jump on a first query. Nothing like this query was sent before, so page value {value} "
               "belongs elsewhere and would start it past its opening rows. Send {call} to get page one; "
               "resending this call jumps anyway.",
    # decline checks
    reach_rows="Wait - first open {ids}. Queries that named every requirement listed them, and nobody has read "
               "their details. Resend the same decline call if none of them fits.",
    reach_pages="Hold on before declining: {advice}. If nothing new fits, resend the same decline call.",
    reach_read="paging would still reach every unread row of some queries ({pages}); open each row whose text "
               "conflicts with no requirement",
    reach_idle="{ignored} made no difference to the results, with it or without it, so those rows cannot tell "
               "you anything on that point - narrow another way: those words leading or standing alone, a "
               "different filter, a different mode",
    # message form
    ask_none="there is no question mark, and messages are limited. Message the shopper only to ask about one "
             "property, in one question.",
    ask_many="it holds {marks} question marks, but each shopper reply answers one point. Ask about one property "
             "only.",
    # finishing
    unfit="By this store's tool rules, that {tool} call neither buys nor declines. {forms}",
    empty="The cart is empty. Add your pick first, or decline.",
    items="There are {wanted} products in the request, each needing its own cart line and quantity; the cart has "
          "{lines} line(s). Still missing a line: {missing}. Add a checked fit for each, then order.",
    step_open="A step the request sets is still open: {do} (\"{quote}\"). Do it first; finish afterwards.",
    step_ask="Due under the request: {do}. Code itself sends that message to the shopper; you finish once its "
             "answer is read.",
    count="Unit mismatch on {held}: the cart view says {shown}, the request wants {wanted}. Before "
          "ordering, delete that line and add it back with the correct count.",
    decline_again="Stop: {ids} - each one passes by its own listing data, and the reviewer agrees. Giving up earns "
                  "nothing when such items are on hand, so cart the best of them and complete the purchase. A "
                  "decline is right only once you find a stated need one of them fails; quote that need.",
    reserve="the last step is reserved for a purchase or a decline.",
    swap="Code bought instead of declining. Why: {pick} passes each requirement per its listing data and "
         "per the reviewer, and an order outranks a refusal - so it went into the cart (when absent) and "
         "the order was placed. What the store answered: {said}",
    swap_cut="Code carted {pick} instead of declining, since its listing data and the reviewer both say it "
             "passes each requirement. Before any order, resolve what the store answered: {said}",
    reserve_late="with only a few steps left, they are kept back for carting, checking stock and finishing.",
    held_over="Skipped at no cost because {stop} Deal with that result first.",
    closed="Not run: out of steps, or the episode has ended.",
    unreached="The store never answered this call (connection or server trouble). Try it again.",
    why_unreached="an earlier store request in this batch got no answer.",
    why_fault="an earlier call in this batch already failed.",
    why_refused="the finish attempt before it was refused.",
    why_rejected="a finish attempt before it was bounced.",
    why_news="the result before it brought news from the shopper.",
    fault_twice="Blocked: {tool} has failed twice with '{code}', so every call of this shape is now stopped. "
                "Either shape the arguments to the schema or get there through a different tool.",
    spent="This tool's allowance is used up; no more calls to it.",
    order_refused="ORDER REFUSED - nothing ended and no cart line moved. Work through it: take on board every "
                  "change the refusal states; take out lines now unfit (an item gone out of stock, say); confirm "
                  "stock and price of whatever replaces them; place the order again.",
    decline_refused="DECLINE REFUSED - the session keeps going. Fold in every change the reply mentions, carry on "
                    "looking, and close out later.",
    recheck="Cheapest fit required. Shown earlier yet never opened, same section, inside the budget and no "
            "dearer than your pick's latest price: {priced}. Climb from the lowest price, testing every "
            "requirement on each, and only then accept a higher price.",
    rejected="The finish did not go through; the store's reply says why. Fix that and finish again.",
    news="This result came with news from the shopper - take it into account: {heard}",
)


def _format_note(key: str, **field_specs) -> str:
    template = _eje_NOTES[key]
    return template.format(**field_specs)


def _held_note(key: str, **field_specs) -> str:
    """A hold note: the call is kept back and no step is spent."""
    return "%s %s" % (_HELD_NOTICE, _format_note(key, **field_specs))


# ================================================================ order and cart gates

def _g1g_orders(agent, call):
    return agent._classify_call(call["name"], call["args"]) == "order"


def _g1g_price_lead(agent, listings, held):
    """Next lead for a lower price: unopened cheaper rows, then pages that finish a search, then a
    search on a bare value; returns (kind, data) or None."""
    room = max(1, min(4, agent.remaining_steps() - agent._reserved_steps() - 1))
    named = _g1g_best_named(agent)
    lower = _find_cheaper_rows(agent, held, listings, True)
    lower = sorted(lower, key=lambda ident: (-named.get(ident, (0, 0))[0], listings[ident]["price"]))[:room]
    if lower:
        return "rows", lower
    section = (listings.get(held) or {}).get("label")
    fits = lambda view: agent._has_step_room(view[3] + 2)
    partial = [view for view in _partial_search_views(agent, section) if fits(view)]
    finishing = [view for view in partial if _eje_view_completes(agent, view)]
    finishing += [view for view in _g1g_token_views(agent, section) if fits(view)]
    untried = [val for val in _eje_bare_values(agent) if not _g1g_alone_searched(agent, val)]
    if finishing:
        return "pages", finishing[:2]
    if not untried and partial:
        return "pages", partial[:2]
    if untried and agent._has_step_room(4):
        return "bare", (untried[:2], section)
    return None


def _g1g_fit_hint(agent, held, listings, unmet):
    """A pointer for the model when the held listing misses something: opened fits of its section, else the
    request's own wording of the missed properties."""
    section = listings[held].get("label")
    good = []
    for ident, facts in listings.items():
        if ident != held and facts["opened"] and agent._within_price_cap(facts):
            same = section is None or facts["label"] == section
            if same and agent._row_passes_literal(facts) is True:
                good.append(ident)
    if good:
        return _format_note("fit_others", ids=", ".join(good[:3]))
    quotes = []
    for miss in unmet:
        said = agent._request_sentence_for(str(miss).partition(":")[0])
        if said and said not in quotes:
            quotes.append(said)
    if not quotes:
        return ""
    return _format_note("fit_words", phrases=" | ".join('"%s"' % said for said in quotes[:2]))


def _eje_g_relax(agent, gate_ctx, call):
    """A cheapest order after a dropped condition or new budget is held once so prices get compared again."""
    reason = agent.rerank_reason
    if not reason or not _g1g_orders(agent, call) or not agent.cart_ids:
        return None
    agent.rerank_reason = ""   # spent whether or not the hold fires
    if not agent._has_step_room(3):
        return None
    lower = _find_cheaper_rows(agent, agent.cart_ids[-1], agent._row_table(), False)
    seen = _format_note("relax_seen", ids=", ".join(lower[:4])) if lower[:4] else ""
    return _held_note("relax", change=reason, seen=seen)


def _eje_g_cheapest(agent, gate_ctx, call):
    """Cheapest-priority order: hold once per state when a lower price could still turn up."""
    wants_low = agent.plan.get("priority") == "cheapest"
    if not (wants_low and _g1g_orders(agent, call) and agent.cart_ids):
        return None
    listings = agent._row_table()
    mark = "cheapest %s %d" % (agent._arm_signature(), len(listings))   # new rows give a new state
    if mark in agent.held_arms or not agent._has_step_room(2):
        return None
    lead = _g1g_price_lead(agent, listings, agent.cart_ids[-1])
    if lead is None:
        return None
    agent.held_arms.add(mark)
    kind, data = lead
    if kind == "bare":
        values, section = data
        fields = {"values": " / ".join('"%s"' % val for val in values),
                  "scope": _format_note("cheap_scope", label=section) if section else ""}
    elif kind == "rows":
        fields = {"ids": ", ".join(data)}
    else:
        fields = {"pages": _eje_page_list(data)}
    return _held_note("cheap_" + kind, **fields)


def _eje_g_release(agent, gate_ctx, call):
    """Taking the held item out after a change is held once while its own facts and price still pass."""
    name, args = call["name"], call["args"]
    adding = agent.add_signature
    if name != gate_ctx["cart"] or adding is None or agent._choice_signature(name, args) == adding:
        return None   # not the cart tool, or an add
    target = _cart_listing_id(agent, name, args)
    mark = "keep %s %s" % (target, agent._arm_signature())
    if not agent.cart_generation or target not in agent.cart_ids:
        return None
    if mark in agent.held_arms or not agent._has_step_room(1):
        return None
    facts = agent._row_table().get(target)
    if not facts or not agent._within_price_cap(facts):
        return None
    if agent._row_passes_literal(facts) is not True:
        return None
    agent.held_arms.add(mark)
    cap = agent._need_list.budget
    return _held_note("release", target=target, price=facts["price"],
                      within="" if cap is None else _format_note("release_budget", budget=cap))


def _g1g_check_line(agent, held, listings):
    """The hold note for one held line, once per state; None when it passes."""
    mark = "fit %s %s" % (held, agent._arm_signature())
    if mark in agent.held_arms:
        return None
    facts = listings.get(held) or {}
    extra = {}
    if not facts.get("opened"):
        key = "fit_unread"
    elif agent._within_price_cap(facts):
        unmet = agent._missed_conditions(held, listings)
        if not unmet:
            return None
        key = "fit_unmet"
        extra = {"unmet": "; ".join(unmet), "pointer": _g1g_fit_hint(agent, held, listings, unmet)}
    else:   # newest data: unavailable or over budget
        key = "fit_money"
        if facts.get("stock") is False:
            extra = {"why": _format_note("fit_sold")}
        else:
            extra = {"why": _format_note("fit_dear", price=facts.get("price"), budget=agent._need_list.budget)}
    agent.held_arms.add(mark)
    return _held_note(key, held=held, **extra)


def _eje_g_fit(agent, gate_ctx, call):
    """Order gate: per state, hold once for an unread, unavailable, over-budget or short line."""
    if not (_g1g_orders(agent, call) and agent.cart_ids and agent._has_step_room(2)):
        return None
    listings = agent._row_table()
    queue = list(agent.cart_ids[:-1]) if agent._wanted_items() else []
    queue.append(agent.cart_ids[-1])
    return next((note for note in (_g1g_check_line(agent, held, listings) for held in queue) if note), None)


# ================================================================ unopened leads

def _eje_unopened_hits(agent, needed, limit):
    """Never-opened, not ruled-out ids shown by searches naming at least `needed` required values, best first."""
    if not needed or limit <= 0:
        return []
    listings = agent._row_table()
    if not listings:
        return []
    ranked = []
    for ident, (named, at) in _g1g_best_named(agent).items():
        facts = listings.get(ident)
        usable = facts is not None and named >= needed and not facts["opened"] and facts["stock"] is not False
        if usable and agent._within_price_cap(facts) and agent._row_passes_literal(facts) is not False:
            ranked.append((-named, at, ident))
    ranked.sort()
    return [ident for _, _, ident in ranked]


def _g1g_sections(agent):
    """Requested section labels in lower case; chosen enum values that match seen labels, widened by
    stem match for any product left uncovered. Empty set means unknown."""
    props = _g1g_ctx(agent)[2]
    picked = set()
    for _, sent in _g1g_calls(agent):
        for name, val in sent.items():
            spec = props.get(name)
            if isinstance(val, str) and isinstance(spec, dict) and spec.get("enum"):
                picked.add(val.strip().lower())
    seen = set()
    for entry in agent._row_table().values():
        if isinstance(entry.get("label"), str):
            seen.add(entry["label"].strip().lower())
    chosen = {lb for lb in seen if lb in picked}
    goods = [item["product"] for item in agent._wanted_items()] or [(agent.plan or {}).get("product") or ""]
    for good in goods:
        stems = agent._ejc_stems(good)
        if not stems or any(stems & agent._ejc_stems(lb) for lb in chosen):
            continue
        chosen |= {lb for lb in seen if stems & agent._ejc_stems(lb)}
    return chosen


def _g1g_take_turns(lanes, limit, keep):
    """Round robin over the lanes: each turn a lane gives its first row `keep` accepts (rows refused are spent)."""
    out = []
    while len(out) < limit and any(lanes):
        for lane in lanes:
            while lane and len(out) < limit:
                head = lane.pop(0)
                if keep(head):
                    out.append(head)
                    break
    return out


def _g1g_leads(agent, limit, only=None, typed_only=False):
    """Leads to open next, up to `limit`: unopened rows that fit the budget, sit in a requested section and are
    not ruled out by a stated value. Ranking: more wanted values typed, then the narrowest search, price, position;
    sections alternate. `only` limits to those product indexes, `typed_only` to rows a typed value reached."""
    listings = agent._row_table()
    sections = _g1g_sections(agent)
    pool = []
    for ident, mark in _g1g_sharpest(agent).items():
        facts = listings.get(ident)
        if (typed_only and not mark[0]) or facts is None or facts["opened"]:
            continue
        section = (str(facts.get("label") or "")).lower().strip()
        if sections and section not in sections:
            continue
        if not agent._within_price_cap(facts):
            continue
        if agent._row_passes_literal(facts) is False:
            continue
        cost = facts.get("price")
        if not isinstance(cost, (int, float)):
            cost = float("inf")
        pool.append(((mark[0], mark[1], cost, mark[2]), section, ident))
    by_section = {}
    for entry in sorted(pool):
        by_section.setdefault(entry[1], []).append(entry[2])
    keep = (lambda ident: True) if only is None else (lambda ident: agent._ejc_item_index(ident) in only)
    return _g1g_take_turns(list(by_section.values()), limit, keep)


def _eje_unfitted_rows(agent, limit) -> list[str]:
    """Unopened on-target rows for requested products that have no opened fit yet."""
    listings = agent._row_table()
    items = agent._wanted_items()
    covered = set()
    for ident, facts in listings.items():
        if facts["opened"] and agent._within_price_cap(facts) and not agent._missed_conditions(ident, listings):
            covered.add(agent._ejc_item_index(ident) if items else 0)
    open_items = set(range(len(items) or 1)) - covered
    if not open_items:
        return []
    return _g1g_leads(agent, limit, open_items if items else None, typed_only=True)


# ================================================================ search flood gate

_g1g_FLOOD_AFTER = 6   # searches with nothing opened before a new one is held once


def _g1g_read_count(agent, gate_ctx):
    """Detail or compare reads sent so far, including ones already let through in this batch."""
    readers = {agent.roles.get(role) for role in ("details", "view")}
    readers.discard(None)
    count = 0
    for entry in agent.trace:
        if entry.get("kind") == "env" and (entry.get("actions") or [{}])[0].get("name") in readers:
            count += 1
    return count + len([1 for queued in gate_ctx.get("ready") or [] if queued[1] in readers])


def _g1g_paging_repeat(agent, sent):
    """True when the call is an earlier search again with only its paging values changed."""
    page_keys = {_g1g_ctx(agent)[1], _eje_text_page_key(agent)}
    page_keys.discard(None)
    history = _g1g_calls(agent)
    given = set()
    for entry, _ in history:
        given.update(_eje_offered_next(_g1g_body(entry), str, 0))
    page_keys |= {k for k, v in sent.items() if isinstance(v, str) and v in given}
    if not page_keys & set(sent):
        return False
    strip = lambda args: {k: v for k, v in args.items() if k not in page_keys}
    target = strip(sent)
    return any(strip(past) == target for _, past in history)


def _g1g_noun(label):
    """The longest plain word (over two letters, no apostrophe) of a section label; the later one on a tie."""
    noun = ""
    for word in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", label):
        if "'" not in word and len(word) > 2 and len(word) >= len(noun):
            noun = word
    return noun.lower()


def _g1g_mixed_hint(agent):
    """Suggested wording when the section filter seems ignored by worded searches (most of their rows
    are strays): the latest such text plus the section noun, cut to fit; '' when not."""
    _, offset, props = _g1g_ctx(agent)
    page_keys = {offset, _eje_text_page_key(agent)}
    page_keys.discard(None)
    listings, sections = agent._row_table(), _g1g_sections(agent)
    strays_all = shown_all = 0
    hint = ""
    for sent, _, shown in _g1g_answered(agent):
        if not shown:
            continue
        worded, chosen = [], []
        for k, v in sent.items():
            if not isinstance(v, str):
                continue
            if (props.get(k) or {}).get("enum"):
                if v.strip().lower() in sections:
                    chosen.append(v.strip().lower())
            elif k not in page_keys and v.strip():
                worded.append((k, v))
        if not worded or not chosen:
            continue
        strays = 0
        for cells in shown:
            lb = str((listings.get(_g1g_listing_id(cells)) or {}).get("label") or "").strip().lower()
            if lb and lb != chosen[0]:
                strays += 1
        strays_all += strays
        shown_all += len(shown)
        noun = _g1g_noun(chosen[0])
        k, text = worded[0]
        if noun and strays * 2 > len(shown) and noun not in _word_set(text):
            limit = (props.get(k) or {}).get("maxLength")
            space = limit - len(noun) - 1 if isinstance(limit, int) else len(text)
            hint = _eja_word_cut(text.strip(), space) + " " + noun if space > 0 else ""
    if shown_all and strays_all * 2 > shown_all:
        return hint
    return ""


def _eje_g_flood(agent, gate_ctx, call):
    """Hold a fresh search once when six or more went out and nothing was opened; paging passes."""
    tool = agent.roles.get("search")
    if not tool or call["name"] != tool:
        return None
    if gate_ctx["planned"] or "flood" in agent.held_arms:
        return None
    queued = [job for job in gate_ctx.get("ready") or [] if job[1] == tool]
    count = len(_g1g_calls(agent)) + len(queued)
    if count < _g1g_FLOOD_AFTER:
        return None
    if _g1g_read_count(agent, gate_ctx) or not agent._has_step_room(3):
        return None
    if _g1g_paging_repeat(agent, call.get("args") or {}):
        return None
    leads = _g1g_leads(agent, 4)
    if not leads:
        return None
    agent.held_arms.add("flood")
    hint = _g1g_mixed_hint(agent)
    return _held_note("flood", count=count, ids=", ".join(leads),
                      mixed=_format_note("flood_mixed", example=json.dumps(hint)) if hint else "")

def _g1h_gate_reach(ep, gctx, call):
    """Hold a decline once per state while rows of on-target searches are still unopened, or while such a search
    can still be paged to its end; the hold also names search parts the store ignored."""
    if ep._classify_call(call["name"], call["args"]) != "decline":
        return None
    memo = "reach decline " + ep._arm_signature()
    if memo in ep.held_arms or not ep._has_step_room(3) or ep.seconds_remaining() < WRAP_UP_WINDOW + 30:
        return None
    wanted = sum(1 for value in _eje_bare_values(ep) if _word_set(value))
    room = min(4, ep.remaining_steps() - ep._reserved_steps() - 2)
    unopened = _eje_unopened_hits(ep, wanted, room)
    if unopened:
        ep.held_arms.add(memo)
        return _held_note("reach_rows", ids=", ".join(unopened[:room])) + ep._ejc_read_by_view()
    pages = []
    for view in _partial_search_views(ep, None, singular=True):
        if _eje_view_completes(ep, view) and ep._has_step_room(view[3] + 2):
            pages.append(view)
    idle = _eje_idle_parts(ep)[:2]
    if not pages:
        return None
    ep.held_arms.add(memo)
    tips = [_format_note("reach_read", pages=_eje_page_list(pages[:2]))]
    if idle:
        tips.append(_format_note("reach_idle", ignored="; ".join(idle)))
    return _held_note("reach_pages", advice="; also ".join(tips))


def _g1h_without(args, key):
    return {name: value for name, value in (args or {}).items() if name != key}


def _g1h_gate_page(ep, gctx, call):
    """Hold, once per call, a search page value that the latest reply of the same search did not offer, and name
    the one it did offer (any other value skips or repeats rows)."""
    tool = ep.roles.get("search")
    sent = call.get("args") or {}
    if not tool or call["name"] != tool or not isinstance(sent, dict) or not ep._has_step_room(2):
        return None
    text_key = _eje_text_page_key(ep)
    number_key = _find_paging_args(ep, tool)[0]
    for key in _eje_page_keys(ep):
        value = sent.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            continue
        kind = int if isinstance(value, int) else str
        if kind is int and key != number_key:
            continue
        rest = _g1h_without(sent, key)
        prior = _eje_prior_page(ep, key, sent)
        if prior is None:
            if kind is str and key == text_key and value.strip():
                memo = "fresh " + tool + _to_compact_json(sent)
                if memo in ep.held_arms:
                    return None
                ep.held_arms.add(memo)
                return _held_note("page_fresh", value=json.dumps(value), call=_to_compact_json(rest))
            continue
        if prior[0] == value:
            continue
        # one reply paging the same search several times: only its first page is checked
        if any(name == tool and _g1h_without(args, key) == rest for _, name, args in gctx.get("ready") or []):
            continue
        last = prior[0]
        start = last if isinstance(last, int) and not isinstance(last, bool) else 0
        offered = _eje_offered_next(prior[1], kind, start)
        if kind is int and value <= start:
            continue   # stepping back is deliberate
        if kind is str and key != text_key:
            offered = [token for token in offered if _eje_shape(token) == _eje_shape(value)]
        if len(offered) != 1 or value in offered:
            continue
        memo = "page " + tool + _to_compact_json(sent)
        if memo in ep.held_arms:
            return None
        ep.held_arms.add(memo)
        fixed = dict(sent)
        fixed[key] = offered[0]
        return _held_note("page_arg", value=json.dumps(value), offered=json.dumps(offered[0]),
                          call=_to_compact_json(fixed))
    return None


def _g1h_gate_one_question(ep, gctx, call):
    """A message written by the model carries exactly one question mark."""
    if gctx["planned"] or call["name"] != gctx["message"] or ep._pending_procedures("order"):
        return None
    marks = sum(text.count("?") for text in (call["args"] or {}).values() if isinstance(text, str))
    if marks == 1:
        return None
    detail = _format_note("ask_many", marks=marks) if marks else _format_note("ask_none")
    return "%s: %s" % (_SKIP_NOTICE, detail)


# a proposed call meets these gates in this order; the first one to answer with a note holds or skips it
_g1h_GATES = (
    _gate_unknown_tool, _gate_not_offered, _gate_bad_json, _gate_schema_misfit, _gate_finish_alone, _gate_known_fault,
    _gate_tool_quota, _gate_repeat_call, _g1h_gate_page, _eje_g_flood, _gate_late_details, _gate_late_unfitted,
    _g1h_gate_one_question, _gate_stacked_add, _gate_cart_review, _gate_unit_count,
    _eje_g_release, _gate_reread_after_refusal, _eje_g_relax, _eje_g_cheapest, _eje_g_fit,
    _g1h_gate_reach,
)

_g1h_ORDER_LIST = _HELD_NOTICE + (
    " Resend the order only after all five points hold; mend any that does not."
    " Procedures - each one the request set is done."
    " Lines - the basket carries a single entry for each requested product (just one entry if one product was"
    " requested), and every entry is your chosen listing in its wanted quantity."
    " Suitability - that listing's own attributes satisfy each stated requirement, each shopper-revealed need and"
    " each announced change."
    " Cost and availability - looked up again since your last basket edit, with a tool that displays them: in"
    " stock, and units multiplied by price inside today's budget."
    " Best pick - when the request wanted the cheapest or the best, no listing you could reach suits it better."
)

_g1h_DECLINE_LIST = _HELD_NOTICE + (
    " Anything suitable should be bought instead. A decline is permanent, so send that tool again only if no"
    " candidate works and you are sure that: the data of each candidate itself was opened; queries changed their"
    " wording, filters and page; and each candidate breaks budget, availability, a stated requirement or a"
    " shopper-revealed need. Remember that matching keys may use other spellings and close synonyms are matches."
)

# a stage puts an outcome name here to leave its pipeline early; the runner always removes it
_g1h_WAY = "_g1h_way"


# ---------------------------------------------------------------- pipeline runner

def _g1h_pipe(ep, rec, stages, outs=()):
    """Run `stages` in order, each taking and returning the record. Returns (record, way): "exit" as soon as a
    stage has put a value under rec["exit"], else the outcome a stage named (it must be one of `outs` and ends
    this pipeline), else "done" after the last stage."""
    for stage in stages:
        rec = stage(ep, rec)
        way = rec.pop(_g1h_WAY, "done")
        if "exit" in rec:
            return rec, "exit"
        if way != "done":
            if way not in outs:
                raise KeyError((stage.__name__, way))
            return rec, way
    return rec, "done"


# ---------------------------------------------------------------- executing proposed calls

def _g1h_gate_context(ep, calls, planned):
    """What every gate may read about this batch."""
    roles = ep.roles
    allowed = {spec["function"]["name"] for spec in ep._tools_this_turn()}
    ends = {roles.get("order"), roles.get("decline")}
    ends.discard(None)
    return {"planned": planned, "allowed": allowed, "finishing": ends, "message": roles.get("message"),
            "cart": roles.get("cart"), "count": len(calls), "queued_messages": 0}


def _g1h_screen(ep, job):
    """Every proposed call meets the gates; a held or skipped call gets the gate's note as its reply."""
    ctx = _g1h_gate_context(ep, job["calls"], job["planned"])
    passed = []
    job["ctx"], job["replies"], job["ready"] = ctx, [], passed
    ctx["ready"] = passed   # what this batch already let through
    for proposed in job["calls"]:
        func = proposed.get("function") or {}
        probe = {"id": proposed["id"], "name": func.get("name"), "raw": func.get("arguments")}
        note = None
        for gate in _g1h_GATES:
            note = gate(ep, ctx, probe)
            if note is not None:
                break
        if note is not None:
            job["replies"].append((probe["id"], note))
            continue
        if probe["name"] == ctx["message"]:
            ctx["queued_messages"] += 1
        passed.append((probe["id"], probe["name"], probe["args"]))
    return job


def _g1h_hold(job, call_id, note):
    """Keep the finishing call back (answered with `note` when one is given); nothing is left to send."""
    if note is not None:
        job["replies"].append((call_id, note))
    job["ready"] = []
    job[_g1h_WAY] = "held"
    return job


def _g1h_f_pick(ep, job):
    """Finish checks run only when the first surviving call is a finishing tool."""
    ready = job["ready"]
    if not ready or ready[0][1] not in job["ctx"]["finishing"]:
        job[_g1h_WAY] = "pass"
        return job
    ident, tool, args = ready[0]
    job["finish"] = {"id": ident, "name": tool, "args": args, "kind": ep._classify_call(tool, args)}
    return job


def _g1h_f_unfit(ep, job):
    """Values that fit neither finishing job: both forms are explained once per call, then the call goes."""
    fin = job["finish"]
    if fin["kind"] is not None:
        return job
    tool = fin["name"]
    doubled = any(job not in _CLOSING_JOBS and bound == tool for job, bound in ep.roles.items())
    memo = "unfit " + tool + _to_compact_json(fin["args"])
    if doubled or memo in ep.held_arms:
        job[_g1h_WAY] = "pass"
        return job
    ep.held_arms.add(memo)
    return _g1h_hold(job, fin["id"], _held_note("unfit", tool=tool, forms=" ".join(ep._closing_lines())))


def _g1h_f_empty(ep, job):
    """An order on an empty cart is always held; after three holds for one cart state, code finishes."""
    if job["finish"]["kind"] != "order" or not ep.roles.get("cart") or ep.cart_ids:
        return job
    if ep.remaining_steps() <= 1:
        return job
    stamp = ep.cart_changed_at
    tries = ep.empty_holds.get(stamp, 0) + 1
    ep.empty_holds[stamp] = tries
    if tries <= 3:
        return _g1h_hold(job, job["finish"]["id"], _held_note("empty"))
    ep.finish_now()
    return _g1h_hold(job, None, None)


def _g1h_f_steps(ep, job):
    """A stated step still open before this finish: code sends a confirmation question when a message is left;
    any other open step holds the finish once and names it. Otherwise the kind of finish picks the next checks."""
    fin = job["finish"]
    kind = fin["kind"]
    spare = ep.remaining_steps() - 2
    for proc in ep._pending_procedures(kind):
        memo = (proc["quote"], kind, ep.cart_changed_at)
        if spare <= 0 or memo in ep.proc_held:
            continue
        ep.proc_held.add(memo)
        if proc["role"] != "message":
            return _g1h_hold(job, fin["id"], _held_note("step_open", do=proc["do"], quote=proc["quote"]))
        if ep.messages_used >= ep._message_limit() or not ep.cart_ids or ep.queued_reply:
            continue
        if ep.remaining_steps() <= 4:
            continue
        chosen = ep.cart_ids if ep._wanted_items() else ep.cart_ids[-1:]
        item_name = " and ".join(ep._listing_title(idx) for idx in chosen)
        ep.queued_reply = "Before I place the order: I have chosen {item_name}. Shall I go ahead with it?".format(
            item_name=item_name)
        return _g1h_hold(job, fin["id"], _held_note("step_ask", do=proc["do"]))
    job[_g1h_WAY] = kind
    return job


def _g1h_f_count(ep, job):
    """Hold an order once per cart state when the latest cart view shows a different unit count."""
    if ep._wanted_items():
        pairs = [(idx, ep._ejc_line_units(idx)) for idx in ep.cart_ids]
    else:
        pairs = [(idx, ep._wanted_units()) for idx in ep.cart_ids[-1:]]
    if not pairs or ep.cart_seen_at < ep.cart_changed_at:
        return job
    for listing, wanted in pairs:
        shown = ep.units_in_cart.get(listing)
        if shown is None or wanted is None or shown == wanted or ep.remaining_steps() <= 2:
            continue
        memo = "count %s %s" % (listing, ep.cart_changed_at)
        if memo in ep.held_arms:
            continue
        ep.held_arms.add(memo)
        return _g1h_hold(job, job["finish"]["id"], _held_note("count", held=listing, shown=shown, wanted=wanted))
    return job


def _g1h_f_items(ep, job):
    """Several products asked: an order whose cart has no line for one of them is held once per cart state."""
    if job["finish"]["kind"] != "order" or ep.remaining_steps() <= 2:
        return job
    missing, lines = ep._ejc_missing_items()
    memo = "items %s" % ep.cart_changed_at
    if not missing or memo in ep.held_arms:
        return job
    ep.held_arms.add(memo)
    named = "; ".join("%s (quantity %d)" % (item["product"], item["quantity"]) for item in missing)
    note = _held_note("items", wanted=len(ep._wanted_items()), lines=lines, missing=named)
    return _g1h_hold(job, job["finish"]["id"], note)


def _g1h_f_review(ep, job):
    """The first order after a cart change is held once, for a listing warning or the checklist."""
    if ep.order_checked or ep.remaining_steps() <= 2 or ep.seconds_remaining() <= WRAP_UP_WINDOW:
        return job
    ep.order_checked = True
    chosen = ep.cart_ids if ep._wanted_items() else ep.cart_ids[-1:]
    warning = next((text for text in (ep._verify_item(idx, "about to be ordered") for idx in chosen) if text), "")
    note = "%s %s" % (_HELD_NOTICE, warning) if warning else _g1h_ORDER_LIST
    return _g1h_hold(job, job["finish"]["id"], note)


def _g1h_f_decline_again(ep, job):
    """A decline repeated while reviewed listings still pass is held one more time; after that, code may buy."""
    fin = job["finish"]
    arm = "decline " + ep._arm_signature()
    fin["arm"] = arm
    if arm not in ep.held_arms or not ep.proven_fits:
        return job
    used = ep.decline_holds.get(arm, 0)
    if used < 1 and ep._has_step_room(2):
        ep.decline_holds[arm] = used + 1
        return _g1h_hold(job, fin["id"], _held_note("decline_again", ids=", ".join(ep.proven_fits)))
    swapped = _g1h_buy_instead(ep, job["ctx"])
    if swapped is None:
        return job
    return _g1h_hold(job, fin["id"], swapped)


def _g1h_props(ep, tool):
    return (ep.param_schemas.get(tool) or {}).get("properties") or {}


def _g1h_single_listing(ep, props, args):
    """The one known listing a cart call names, or None."""
    named = {listing for listing, _ in _eje_cart_lines(props, args) if listing in ep.seen_ids}
    return named.pop() if len(named) == 1 else None


def _g1h_asked_sections(ep):
    """Section labels of opened listings the request points at: a choice value the searches sent, or a label
    sharing the product's own words."""
    tool = ep.roles.get("search")
    shape = _g1h_props(ep, tool)
    choices = set()
    for entry in ep.trace:
        if entry.get("kind") != "env":
            continue
        for action in entry.get("actions") or []:
            if action.get("name") != tool:
                continue
            for key, value in (action.get("args") or {}).items():
                if isinstance(value, str) and (shape.get(key) or {}).get("enum"):
                    choices.add(str(value).strip().lower())
    product = ep._ejc_stems(ep.plan.get("product") or "")
    labels = {row["label"] for row in ep._row_table().values() if row["opened"] and isinstance(row["label"], str)}
    return {label for label in labels if label.strip().lower() in choices or ep._ejc_stems(label) & product}


def _g1h_add_template(ep):
    """The latest cart call, sent or proposed as an add, that names exactly one listing: (args, listing) or None."""
    tool = ep.roles.get("cart")
    props = _g1h_props(ep, tool)
    proposed = []
    for msg in ep.messages:
        if msg.get("role") != "assistant":
            continue
        for spec in msg.get("tool_calls") or []:
            func = spec.get("function") or {}
            if func.get("name") == tool:
                proposed.append(_extract_json_dict(func.get("arguments") or ""))
    for args in reversed(proposed):
        if not isinstance(args, dict):
            continue
        if ep.add_signature is not None and ep._choice_signature(tool, args) != ep.add_signature:
            continue
        listing = _g1h_single_listing(ep, props, args)
        if listing is not None:
            return args, listing
    return None


def _g1h_swap(node, old, new, units):
    """Copy of a cart call with listing `old` replaced by `new`; the lone whole number beside it becomes `units`."""
    if isinstance(node, list):
        return [_g1h_swap(part, old, new, units) for part in node]
    if not isinstance(node, dict):
        return new if node == old else node
    whole = [key for key, value in node.items() if isinstance(value, int) and not isinstance(value, bool)]
    beside = len(whole) == 1 and old in node.values()
    out = {}
    for key, value in node.items():
        out[key] = units if beside and key == whole[0] else _g1h_swap(value, old, new, units)
    return out


def _g1h_swap_plan(ep, cart):
    """What code would buy in place of a decline: (pick, add call or None, fits), or None when nothing qualifies.
    A fit is a proven listing of a requested section within the price cap whose own data pass."""
    table, wanted_sections = ep._row_table(), _g1h_asked_sections(ep)
    fits = []
    for idx in ep.proven_fits:
        row = table.get(idx) if idx in table else None
        if row is None or row["label"] not in wanted_sections or not ep._within_price_cap(row):
            continue
        if ep._row_passes_literal(row) is True:
            fits.append(idx)
    basket = ep.cart_ids
    if basket:
        if len(basket) == 1 and basket[0] in fits:
            return basket[0], None, fits   # the cart already holds just that fit
        return None
    if not fits:
        return None
    cheapest = min(fits, key=lambda idx: (table[idx]["price"] is None, table[idx]["price"] or 0, idx))
    base_call = _g1h_add_template(ep)
    if base_call is None:
        return None
    swapped = _g1h_swap(base_call[0], base_call[1], cheapest, ep._wanted_units())
    add, problems = _validate_value(swapped, ep.param_schemas[cart], "args")
    if problems or _g1h_single_listing(ep, _g1h_props(ep, cart), add) != cheapest:
        return None
    return cheapest, add, fits


def _g1h_buy_instead(ep, gctx):
    """A decline repeated after both holds while listings of the requested section still pass their own data and
    the reviewer: code orders the cheapest such fit instead (one requested product; an empty cart, or a cart
    holding only that fit). Returns the note for the declined call, or None to let the decline go."""
    cart, order = ep.roles.get("cart"), ep.roles.get("order")
    if ep._wanted_items() or not (cart and order) or ep._pending_procedures("order"):
        return None
    plan = _g1h_swap_plan(ep, cart)
    if plan is None:
        return None
    pick, add, fits = plan
    finale, problems = _validate_value(ep._minimal_closing_args(order, _ejc_FORCED_TEXT["order"], "order"),
                                       ep.param_schemas[order], "args")
    steps_needed = 2 if add else 1
    if problems or ep.remaining_steps() < steps_needed:
        return None
    ep.trace.append({"kind": "decline_swapped", "pick": pick, "fits": fits})
    replies = []
    if add is not None:
        first_text, why = _g1h_send(ep, gctx, cart, add)
        replies.append(first_text)
        if why or ep.finished or pick not in ep.cart_ids:
            return _format_note("swap_cut", pick=pick, said=" | ".join(replies))
    replies.append(_g1h_send(ep, gctx, order, finale)[0])
    return _format_note("swap", pick=pick, said=" | ".join(replies))


def _g1h_f_decline_first(ep, job):
    """The first decline per state is held for the proof; the very first also runs the audit or the checklist."""
    fin = job["finish"]
    arm = fin["arm"]
    if arm in ep.held_arms or not ep._has_step_room(2) or ep.seconds_remaining() <= WRAP_UP_WINDOW + 30:
        return job
    ep.held_arms.add(arm)
    first = not ep.decline_checked
    ep.decline_checked = True
    doubt = ep._decline_evidence() or (ep._audit_before_decline() if first else "")
    if doubt:
        return _g1h_hold(job, fin["id"], "%s %s" % (_HELD_NOTICE, doubt))
    if first:
        return _g1h_hold(job, fin["id"], _g1h_DECLINE_LIST)
    return job


def _g1h_f_scrub(ep, job):
    """A decline that goes out drops listing names from its remark when the request wants that."""
    fin = job["finish"]
    cleaned = ep._clean_decline_args(fin["name"], fin["args"])
    job["ready"][0] = (fin["id"], fin["name"], cleaned)
    return job


def _g1h_reserve(ep, job):
    """Calls that do not finish may not take the last step: it is kept for the finish."""
    ready, ends = job["ready"], job["ctx"]["finishing"]
    if not ready or not ends or ready[0][1] in ends:
        return job
    keep = ep.remaining_steps() - 1
    if keep >= len(ready):
        return job
    keep = max(0, keep)
    skip = _SKIP_NOTICE + ": " + _format_note("reserve")
    job["replies"] += [(ident, skip) for ident, _, _ in ready[keep:]]
    job["ready"] = ready[:keep]
    return job


def _g1h_send(ep, gctx, tool, args):
    """One call as one step: (reply text, reason to hold back the rest of the reply or "")."""
    action = {"name": tool, "args": args}
    step_no = ep.step + 1
    reply = ep.call_store(action)
    ep.trace.append({"kind": "env", "step": step_no, "actions": [action], "reply": reply})
    if reply is None:
        return _format_note("unreached"), _format_note("why_unreached")
    seen = _eje_observe(ep, {"name": tool, "args": args, "obs": reply.get("observation"),
                                "closing": bool(reply.get("finished")), "ctx": gctx})
    return seen["text"], seen.get("halt") or ""


def _g1h_dispatch(ep, job):
    """Send the surviving calls one step each. Once a result closes, refuses, errs or carries world lines, the
    rest of the reply is held back so the model decides again."""
    stop = ""
    for ident, tool, args in job["ready"]:
        if stop:
            text = _format_note("held_over", stop=stop)
        elif ep.finished or ep.halted or ep.remaining_steps() <= 0:
            text = _format_note("closed")
        else:
            text, stop = _g1h_send(ep, job["ctx"], tool, args)
        job["replies"].append((ident, text))
    return job


_g1h_FINISH_SHARED = (_g1h_f_pick, _g1h_f_unfit, _g1h_f_empty, _g1h_f_steps)
_g1h_FINISH_BY_KIND = {
    "order": (_g1h_f_count, _g1h_f_items, _g1h_f_review),
    "decline": (_g1h_f_decline_again, _g1h_f_decline_first, _g1h_f_scrub),
}
_g1h_SENDING = (_g1h_reserve, _g1h_dispatch)


def _g1h_route_batch(ep, job):
    """Chooses the path of a screened batch: no finishing call first -> reserve and send; a finishing call ->
    the shared finish checks, then the checks of its kind, then reserve and send. A hold skips the sending."""
    job, way = _g1h_pipe(ep, job, _g1h_FINISH_SHARED, ("pass", "held", "order", "decline"))
    if way in ("exit", "held"):
        return job
    if way != "pass":
        job, way = _g1h_pipe(ep, job, _g1h_FINISH_BY_KIND[way], ("held",))
        if way != "done":
            return job
    return _g1h_pipe(ep, job, _g1h_SENDING)[0]


def _g1h_collate(ep, job):
    """Replies go back in the order of the proposed calls."""
    rank = {proposed["id"]: pos for pos, proposed in enumerate(job["calls"])}
    job["replies"].sort(key=lambda pair: rank.get(pair[0], 0))
    return job


# ---------------------------------------------------------------- reading one result

def _g1h_o_tally(ep, seen):
    """Bookkeeping for one sent call: counters, recent shapes, records, and when each id was last mentioned."""
    tool, args, ctx = seen["name"], seen["args"], seen["ctx"]
    kind = ep._classify_call(tool, args)
    seen["kind"] = kind
    if tool == ctx["message"]:
        ep.messages_used += 1
    if kind == "order":
        ep.order_attempts += 1
    for role, bound in ep.roles.items():
        if bound == tool and role not in _CLOSING_JOBS:
            ep.calls_by_role.setdefault(role, []).append(ep.step)
    if tool == ctx["cart"]:
        ep.cart_used = True
        ep.order_checked = False
    if tool in ctx["finishing"] or tool in (ctx["cart"], ctx["message"]):
        ep.recent = []
    else:
        ep.recent = ep.recent[-1:] + [tool + _to_compact_json(args)]
    ep.records.append(seen["obs"])
    ids = set()
    _gather_ids(seen["obs"], ids)
    ep.mentioned.update({idx: ep.step for idx in ids})
    return seen


def _g1h_o_fault(ep, seen):
    """Errors are counted per call shape (a lone text answer only for the identical call); a second one marks
    the shape as faulty."""
    tool, args = seen["name"], seen["args"]
    code, certain = ep._detect_error(tool, seen["obs"])
    seen["fault"] = code
    if not code:
        return seen
    shape = ep._call_shape(tool, args) if certain else tool + _to_compact_json(args)
    tally = ep.faults.get((shape, code), 0) + 1
    ep.faults[(shape, code)] = tally
    if certain:
        seen["halt"] = _format_note("why_fault")
    if tally >= 2 and args and tool not in seen["ctx"]["finishing"]:
        ep.faulty.add(shape)
        seen["fault_note"] = _format_note("fault_twice", tool=tool, code=code)
    return seen


def _eje_cart_ids_in(props, dispatched):
    """Free-text one-token argument values of a cart call: the listing ids it names."""
    out = []
    for key, value in dispatched.items():
        spec = props.get(key)
        if not isinstance(value, str) or (isinstance(spec, dict) and "enum" in spec):
            continue
        token = value.strip()
        if token and " " not in token:
            out.append(value)
    return out


def _eje_cart_lines(props, dispatched):
    """The listing ids a cart call names, each with the units its own line asks for (None when it gives none):
    free one-token values of the call itself, then those of every line object inside a list at any depth (one
    call setting several lines, also when an object wraps the list); an object's single integer is its units."""
    found = [(listing, None) for listing in _eje_cart_ids_in(props, dispatched)]

    def descend(spec, node, depth):
        if depth > 6:
            return
        if not isinstance(spec, dict):
            spec = {}
        if isinstance(node, dict):
            fields = spec.get("properties") or {}
            for key, part in node.items():
                descend(fields.get(key), part, depth + 1)
            return
        if not isinstance(node, list):
            return
        member = spec.get("items")
        if not isinstance(member, dict):
            member = {}
        for line in node:
            if isinstance(line, dict):
                whole = [part for part in line.values() if isinstance(part, int) and not isinstance(part, bool)]
                units = whole[0] if len(whole) == 1 else None
                found.extend((listing, units) for listing in _eje_cart_ids_in(member.get("properties") or {}, line))
            descend(member, line, depth + 1)

    for key, value in dispatched.items():
        if isinstance(value, (dict, list)):
            descend(props.get(key), value, 1)
    return found


def _eje_view_listings(ep, seen):
    """Known listing ids (seen in earlier results) that a cart view's records name, in their order."""
    named = []
    for blob in _iter_dicts([seen]):
        for cell in blob.values():
            if isinstance(cell, str) and cell not in named and cell in ep.seen_ids:
                named.append(cell)
    return named


def _g1h_o_cart(ep, seen):
    """Track the cart. The enum values of the first cart call mean adding; free one-token values are ids, also
    inside the line objects of a list argument. A cart view naming known listings is the cart as it stands."""
    tool, args = seen["name"], seen["args"]
    if tool == seen["ctx"]["cart"]:
        mode = ep._choice_signature(tool, args)
        if ep.add_signature is None:
            ep.add_signature = mode
        adding = mode == ep.add_signature
        for listing, units in _eje_cart_lines(ep.param_schemas[tool].get("properties") or {}, args):
            ep.cart_state[listing] = mode
            others = [idx for idx in ep.cart_ids if idx != listing]
            if not adding:
                ep.cart_ids = others
                ep.units_in_cart.pop(listing, None)
                continue
            ep.cart_ids = others + [listing]
            ep.sent_units[listing] = units if units is not None else ep._requested_units(tool, args)
        ep.cart_changed_at = ep.step
    if tool == ep.roles.get("cart_view"):
        ep.last_cart_view = seen["obs"]
        shown = _eje_view_listings(ep, seen["obs"])
        if shown:   # a view naming no known listing leaves the tracked cart alone
            kept = [idx for idx in ep.cart_ids if idx in shown]
            ep.cart_ids = kept + [idx for idx in shown if idx not in ep.cart_ids]
        ep._parse_cart_units(seen["obs"])
    return seen


def _g1h_o_read(ep, seen):
    """Unwrap world lines, note a spent allowance, and name what kind of result this is."""
    raw = seen["obs"]
    lines, body = _unwrap_text(raw)
    seen["lines"], seen["inner"] = lines, body
    open_finish = bool(seen["kind"]) and not seen["closing"] and not ep.finished
    flagged = isinstance(raw, dict) and "refusal" in raw   # the runtime marks its own refusals
    seen["refused"] = flagged or (open_finish and not _looks_like_refusal(body))
    seen["notes"] = []
    if isinstance(raw, (dict, list)):
        _gather_ids(raw, ep.seen_ids)
    if isinstance(body, dict) and body.get("error") == "budget_exceeded":
        if seen["name"] == ep.roles.get("message") or ep.message_budget is not None:
            cap = ep.message_budget
            ep.message_budget = ep.messages_used if cap is None else min(cap, ep.messages_used)
        seen["notes"].append(_format_note("spent"))
    if seen["refused"]:
        seen[_g1h_WAY] = "refused"
    elif open_finish:
        seen[_g1h_WAY] = "rejected"
    elif lines and seen["name"] != ep.roles.get("message"):
        seen[_g1h_WAY] = "news"
    return seen


def _g1h_o_refused(ep, seen):
    """A refused finish: a new state, the order may be checked again, and cheaper unread rows are named."""
    seen["halt"] = _format_note("why_refused")
    ep.refusals += 1
    ep.refused_at = ep.step
    ep._advance_generation()
    ep.order_checked = False
    declined = seen["kind"] == "decline"
    seen["notes"].append(_format_note("decline_refused" if declined else "order_refused"))
    if seen["kind"] != "order":
        return seen
    rows = ep._row_table()
    cheaper = _rows_to_recheck(ep, rows) if rows else []
    if cheaper:
        priced = ", ".join("%s (%s)" % (idx, rows[idx]["price"]) for idx in cheaper)
        seen["notes"].append(_format_note("recheck", priced=priced))
    return seen


def _g1h_o_rejected(ep, seen):
    """A finishing call answered with a fault code."""
    seen["halt"] = seen.get("halt") or _format_note("why_rejected")
    seen["notes"].append(_format_note("rejected"))
    return seen


def _g1h_o_news(ep, seen):
    """World lines wrapped around an ordinary result."""
    heard = []
    _gather_sentences(seen["lines"], heard)
    if heard:
        ep._advance_generation()
        seen["halt"] = seen.get("halt") or _format_note("why_news")
        seen["notes"].append(_format_note("news", heard=" | ".join(heard[:4])))
    return seen


_g1h_READ_FORKS = {"refused": (_g1h_o_refused,), "rejected": (_g1h_o_rejected,), "news": (_g1h_o_news,),
                   "done": ()}


def _g1h_route_result(ep, seen):
    """Chooses how a result is read: a refused finish, a rejected finish, world news, or nothing more."""
    seen, way = _g1h_pipe(ep, seen, (_g1h_o_read,), ("refused", "rejected", "news"))
    if way == "exit":
        return seen
    return _g1h_pipe(ep, seen, _g1h_READ_FORKS[way])[0]


def _g1h_o_tail(ep, seen):
    """Closing notes: a repeated fault, an empty search, the end of the episode."""
    notes = seen["notes"]
    if seen.get("fault_note"):
        notes.append(seen["fault_note"])
    if seen["name"] == ep.roles.get("search") and not seen.get("fault"):
        empty = ep._no_hits_note(seen["name"], seen["args"], seen["inner"])
        if empty:
            notes.append(empty)
    if seen["closing"]:
        seen["halt"] = "the episode has ended."
    seen["note"] = " ".join(notes)
    return seen


def _g1h_append_note(seen, extra):
    if extra:
        seen["note"] = " ".join(part for part in (seen["note"], extra) if part)


def _g1h_o_listen(ep, seen):
    """The shopper's words go to the interpreter: a message's answer (world lines on it read on their own), or
    the lines riding on any other result plus the sentences of a refusal."""
    body = seen["inner"]
    heard = []
    _gather_sentences(seen["lines"], heard)
    question = None
    if seen["name"] == seen["ctx"]["message"]:
        if heard:   # world lines on an answer are never the answer itself
            _g1h_append_note(seen, ep.read_shopper_lines(heard, None))
            heard = []
        if isinstance(body, str):
            heard.append(body)
        elif isinstance(body, dict) and "error" not in body and not _looks_like_refusal(body):
            _gather_sentences([part for part in body.values() if isinstance(part, (str, list))], heard)
        question = " ".join(str(part) for part in seen["args"].values() if isinstance(part, str))
    elif seen["refused"] and isinstance(body, dict):
        heard.extend(part for part in body.values() if isinstance(part, str) and len(part.split()) >= 4)
    if heard:
        _g1h_append_note(seen, ep.read_shopper_lines(heard, question))
    return seen


def _g1h_o_render(ep, seen):
    """The reply text the model sees."""
    text = _to_compact_json(seen["obs"])
    if seen["closing"]:
        text += " (this ended the episode)"
    if seen["note"]:
        text += "\n[agent note] " + seen["note"]
    seen["text"] = text
    return seen


# ---------------------------------------------------------------- the five flows

_g1h_EPISODE = (_stage_start_episode, _stage_loop_turns, _stage_settle_episode, _stage_close_episode)
_g1h_TURN = (_stage_turn_guard, _stage_send_queued, _stage_run_planner, _stage_call_model, _stage_take_calls,
             _stage_commit_turn, _stage_idle_turn, _stage_act_turn)
_g1h_ASK = (_stage_question_gate, _stage_question_draft, _stage_question_build, _stage_question_send)
_g1h_EXECUTE = (_g1h_screen, _g1h_route_batch, _g1h_collate)
_g1h_OBSERVE = (_g1h_o_tally, _g1h_o_fault, _g1h_o_cart, _g1h_route_result, _g1h_o_tail, _g1h_o_listen,
                _g1h_o_render)


def _eje_episode(session, carry):
    return _g1h_pipe(session, carry, _g1h_EPISODE)[0]


def _eje_turn(session, carry):
    return _g1h_pipe(session, carry, _g1h_TURN)[0]


def _eje_ask(session, carry):
    return _g1h_pipe(session, carry, _g1h_ASK)[0]


def _eje_execute(session, carry):
    return _g1h_pipe(session, carry, _g1h_EXECUTE)[0]


def _eje_observe(session, carry):
    return _g1h_pipe(session, carry, _g1h_OBSERVE)[0]


def _g1h_note_crash(episode, exc):
    """Log an exception that escaped the run."""
    episode.trace.append({"kind": "crash", "error": ("%s: %s" % (type(exc).__name__, exc))[:500]})


def _eje_rescue(episode, exc):
    """After a crash: log it, finish from code while the episode is still open, and return the trace."""
    _g1h_note_crash(episode, exc)
    try:
        if not (episode.finished or episode.halted):
            episode.finish_now()
    except Exception as again:
        _g1h_note_crash(episode, again)
    return episode.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode = Episode(problem_data)
    try:
        return episode.run()
    except Exception as exc:
        return _eje_rescue(episode, exc)


__all__ = ["agent_main"]
