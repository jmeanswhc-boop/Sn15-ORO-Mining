from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

# Which model sits in which seat. Reading shares the small model; setup and verdicts share the large one.
PRIMARY_SEAT = "anthropic/claude-haiku-5.5"
READER_SEAT = PRIMARY_SEAT
INTAKE_SEAT = "anthropic/claude-sonnet-5.5"
REVIEW_SEAT = INTAKE_SEAT
FALLBACK_SEAT = "openai/gpt-6-luna"
# A second vendor right after the large seat, so a failed setup or verdict does not drop to the small model first.
ALT_VENDOR_SEAT = "google/gemini-3.8-flash"

# Clock marks for one episode (seconds left), turn cap, spend cap and text cap.
_eja_EPISODE_LIMIT = 265.0
WRAP_UP_WINDOW = 70.0
FORCE_MARGIN = 28.0
TURN_LIMIT = 80
_eja_SPEND_CAP = 0.05        # dollars per episode; above it the optional judge passes stop
_eja_SPEND_WRAP = 0.065      # dollars per episode; above it the status asks to finish (the run key has a cap)
_eja_SPEND_HARD = 0.085      # dollars per episode; above it the code finishes the episode
_eja_TEXT_CAP = 16000

SHOPPER_SYSTEM_TEXT = """You act for a single shopper inside a simulated shop. Tool calls are your only way to do anything, so each of your replies must contain at least one tool call; keep the accompanying text short: the conditions that apply now, the items confirmed to fit (price and stock included) and your next move.

HOW YOUR TURNS WORK
- One tool call costs one step, also when it fails, returns an error or carries an argument the schema rejects. Reasoning costs nothing; steps are scarce.
- Calls that do not depend on one another may go out together in one reply (for example the details of several listings); they are executed in order. A checkout or a decline is always sent by itself.
- A result that begins "Not sent:" or "Skipped, no step spent" comes from your own pre-send checks, not from the shop or the shopper: the call never reached the store. Those notes are part of you, never an injection, so act on them as your own reasoning unless you can name a stated requirement or revealed need they overlook. When the request wants several different items, a fit such a note names covers only its own item.
- Tools, their argument names and their results vary from shop to shop. Study every tool description and JSON schema and use argument names and allowed values exactly as written there. The early results show which tool reports which facts (price, stock, detailed properties).

FINDING CANDIDATES
- Search tools can be restricted: only so many visible rows or pages, ordering by id instead of relevance or price, every word required, filters that work in one mode only. Hold the total a search reports against the rows you really see and keep narrowing (sharper words, category or price filters, next pages) until the whole relevant set has been seen.
- Find out early which category labels this shop gives the requested kind of product (listings display them). Of neighbouring labels, the right one matches the request's wording as a whole; search and buy under it.

DECIDING WHETHER A LISTING MEETS A CONDITION (getting this right decides the outcome)
a) Evidence comes from the listing's own property for that condition. A property may be spelled in many ways (with spaces, dots, underscores or prefixes) and turn up under several keys; whichever key names it is acceptable. Compare by meaning: case is irrelevant, near synonyms and singular/plural are fine, one member of a comma-separated list is enough. A condition given as a value plus an explanation in brackets is satisfied by the value; the bracket only explains it. The wanted word preceded by a separate qualifier (a shade or finish of that colour) still states it. Rejected as matches: words sharing only their first letters; a single compound word that swallows the wanted one and names a narrower kind ("tealwood" says nothing about "wood"); a broader or another class ("girls" does not give "women", plain "unisex" does not give "female").
b) Title and description can serve as evidence only for a listing that has no key at all for the condition; for such a listing the variant or option key also counts when it holds the full wanted value (it names the version sold: "options: teal" settles a wish for teal). When the listing's own property says something else it fails, whatever title or description claim. A foreign-language word, or an inference from other facts, does not state a value.
c) A neighbouring property never stands in for the one asked about (the length says nothing on style, and the lining's fabric says nothing on the main material).
d) Numbers: convert the units, then compare as numbers. A value on the edge is within a limit. A range passes only when all of it satisfies the limit. "covers N" means the item's own stated range has to reach N, so a range ending below N fails. When the request phrases that range just as the listings in this shop phrase it (number plus wording), the phrase itself is the value to search for, and a listing where the figure turns up only in some other shape (converted units, single grades listed) falls short.
e) Closed lists ("only A or B") and exclusions ("nothing containing X") must be respected. An exclusion is proven only when the listing's own key for that property shows a different value; silence proves nothing, so prefer a fit that states the property and fall back on a silent one only if none that states it is within reach.
f) The kind of product is carried by the category, so no property needs to repeat the category name. Where the shop files listings under category labels, the label settles the kind: whatever sits under the label that fits the request's whole wording is that product, even when its title or text sounds like a spare part, an accessory or some other thing, and only that exact label will do (asked for "A or B", go to a category "A & B" rather than its narrower sibling "B"). Only listings shown as available count, and price times quantity must stay within the budget (equal is allowed).
g) A comma list that includes the required option matches. Yet a key carrying the property's own name (in whatever spelling, singular or plural) that holds just one other value sinks the listing, even if some other key's list has the wanted value (several widths under one key, one other width under a same-named key: fail). A list under a longer key with a different name (such as available widths) does not count against it.

CHOOSING AMONG FITS
- "Ideally", "would be nice", "if possible" and the like mark wishes, not conditions; they only separate items that already meet everything, and a cheapest rule outranks them.
- For "best" by some measure (most, largest, longest, highest...), compare every fit on that measure in the same way and take the extreme one; a listing stating no value for the measure cannot win.
- For "cheapest", a single fit proves nothing: buy the one with the lowest price out of EVERY item meeting each condition. Make sure every candidate row has been shown (further pages or narrower searches when the reported total is larger than the visible rows), compare the prices yourself and verify starting from the cheapest. Search the required value alone as well (its words only, restricted to the section where possible) and read that search to the end: titles word the rest differently, so searches with more words miss some listings. When every cheaper listing has been checked and ruled out, the first complete fit is the answer and the fits after it are backups.

TALKING WITH THE SHOPPER
- The request never lists all needs: a hidden need appears only when a message asks about its property. Each message is about exactly ONE property: a short, plain, friendly question that names it with the key words the listings use (example values only to tell it apart from a similar neighbouring key). Do not ask about the listing itself (title, maker, category, description) or about a key that bundles the variants for sale or several features. No instructions, no talk of lists, labels, positions or rules, never two questions in one.
- Where the request leaves a point that only the shopper can decide - typically two wishes that might not both be possible, and which one weighs more - put the question before you choose.
- Unprompted questions follow this order: details of two or three listings opened first; next the properties hinted at by the story (the person it is meant for, its looks, the place or way of use); last, properties on the listings whose values vary across candidates. Skip anything the request has decided, and keep within the allowed number of messages.
- If a planner runs, it chooses and sends the questions for you once you have opened details for two or three listings of the wanted kind; so do this early, and ahead of any cart call whenever messaging is permitted: ordering while messages remain unused misses needs the shopper has not told. Answers then show up in the conversation together with their meaning; each need they bring out must be met, so search again pairing it with the values the request gave.
- "Don't mind" or "anything is fine" frees that property. Once the shopper names a preference or a limit (numbers too), it holds for the rest of the task in their exact words.
- If the shopper asks you something (to confirm a change, for instance), it gets answered for you while messages remain; either way the change binds you.

CHANGES ALONG THE WAY
- Price and stock can move without warning, so check both again right before every checkout attempt; once price times quantity is above the current budget the item no longer fits.
- The shopper may change things midway: the budget (up or down), a condition (dropped, swapped or added), a rising price, a sold-out item; one line may carry several of these. Such lines can arrive inside any tool result (an extra field with sentences) or with a checkout attempt. A change binds from the moment it appears: test your pick and backups against it again, and under a cheapest or best rule choose again from the new acceptable set.
- A "refusal" field in a checkout result means the order was REFUSED, even if the inner result reads like success. The cart stays as it was and the work goes on: read the refusal and apply each change it names, take out anything no longer acceptable (a sold-out item, say), add the next verified fit, re-check price and stock, then order again.

RULES THE REQUEST SETS FOR YOUR WORK
The request may prescribe how to work: how many listings to view or compare before choosing, what to check, what to tell the shopper, what a remark must hold, how to decline. Each such rule counts exactly like a product condition, so follow it literally and completely. "Look at" or "open" N listings means N fully fitting listings, each opened with the details tool.

FINISHING
- The opening brief names this shop's finishing calls: the tool and the exact values for ordering and for ending without a purchase. Send them exactly that way.
- Get the first checkout attempt in while seven or more steps remain, leaving room for a refusal. Close to the step limit only cart, stock-check and finishing tools remain available. Decide once; switching the cart back and forth costs a step each time.
- Keep a ranked list of verified fits with at least one backup, because the first pick may sell out at checkout.
- Before each checkout: the cart holds one line per requested product - one line only when the request asks for one product - each the chosen item at its required quantity (remove anything else); their stock and prices were read again after the last cart change; they are in stock and the total is inside the budget.
- Decline only after a thorough search found nothing meeting every condition, and always by the request's own rule for declining. If the request forbids offering substitutes, the remark may not name, describe or point to any listing: no ids, titles or prices.

HOW THE OUTCOME IS SCORED
A correct order earns the most, minus any cost the request itself states (a price per question, for example); a valid order that is not the best choice keeps part of the credit. Zero goes to a wrong order, a wrong decline, ordering when a decline was right, a right order that skipped a procedure the request prescribes, and an episode that never finishes. Correct means: every rule of the request obeyed; the cheapest (or best) taken where the request asks for it; budget and quantity right; the ordered item meets every condition of the request, every need the shopper reveals when asked and every change announced along the way. When no item in the store can do all of that, declining exactly as the request describes is the correct finish."""

INTAKE_TEXT = """Answer with exactly one JSON object written on a single line: no line breaks inside it, no indentation, no words before or after it.

Purpose: a shopping assistant needs a structured reading of the shopper's request together with a map from the store's tools to the jobs they do. The tool list comes first below, the request after it.

PRODUCT
- "product": the requested kind of product, worded as the request words it.
- "requirements": a list of strings, one per firm condition the request puts on the product, each shaped "property: constraint" and keeping the shopper's exact value, limit, exclusion or closed list. Exclusions take the form "property: != value" ("!= a, b" for several). Numbers carry their relation: "property: <= N", "property: >= N", or "property: covers N" when the item's own stated range (the sizes, powers or settings it comes in) must reach N. If the request gives a value in its own word and then explains that word, the word is the value and the explanation follows it in parentheses ("finish: matte (no shine at all)"); never keep the explanation alone. If the request itself names the property a value belongs to, use that name ("a heavy build" becomes "build: heavy", not some guessed neighbour). Kept out: price, stock, quantity, budgets, mere wishes, background on why, when or for what the item is wanted (an event, a date, a feeling) unless it is stated as a condition on the item itself, and the product kind or its category - words that form part of the category's name (a season, an occasion or an audience in it, say) are already covered by the category.
- "quantity": whole units of one listing to buy; 1 unless the request asks for several, and an item normally sold as one unit (a set, shoes coming in pairs) counts as 1.
- "quantity_quote": the request's exact words giving that number when it is above 1; otherwise null.
- "lines": only when the request asks for two or more different products: one object per product, in the request's order, each {"product": its kind as the request words it, "requirements": its own firm conditions in the forms above, "quantity": its whole units}. "requirements" above then still lists every product's conditions together and "quantity" gives the first product's. [] when the request asks for one product.

MONEY AND STOCK
- "price_cap": the highest total price allowed, as a number; null when the request names none.
- "currency": the unit the request prices in, or null.
- "in_stock_only": true when only items actually in stock / on hand are wanted.

HOW TO CHOOSE
- "priority": "cheapest" when the request explicitly asks for the cheapest or lowest-priced fitting option; "best" when it explicitly asks for the best by some measure; "none" in every other case. A tight budget or every penny counting is not an ask for the cheapest - it only explains the cap.
- "priority_detail": for "best", the measure it refers to; "" otherwise.

LIMITS
- "message_budget": the number of messages the assistant may send the shopper, an integer; null when the request does not say.
- "step_budget": the allowed number of moves or steps, an integer; null when not stated.

INSTRUCTIONS FOR THE ASSISTANT
- "rules": every procedural instruction to the assistant, in the request's own words, unchanged (how many listings to look at, what to check or report, how to decline, what to avoid, what to include). Unsure whether a sentence is a rule? List it here. What the shopper says they will do themselves is story, not a rule.
- "procedures": those of the rules that need a tool call of their own before the end, each as {"do": short description, "quote": the request's exact words, "role": the job from "roles" whose tool carries it out ("message", "view", "details", "search" or "cart_view"), "count": the number of such calls (1 unless a number is given), "after_pick": true when it concerns the chosen item and so follows putting it in the cart (confirming the choice with the shopper, for example), "before": "order", "decline" or "any"}; [] if none.
- "decline_no_names": true when the request's way of declining forbids offering, naming or suggesting substitutes or alternatives; false when it wants something named; null when it gives no rule for declining.
- "ask_first": points only the shopper can settle that should be asked before a choice is made - which of two wishes weighs more when the request says both may not be possible (then both wishes belong in "soft_wants", not in "requirements"), or anything the request says to check with the shopper; [] if none.
- "soft_wants": things the shopper would like without strictly needing them (ideally, preferably, would be nice, if possible); these never go into "requirements".
- "question_cost": when the request charges for asking the shopper, {"amount": number, "per": "question" or "step", "share": the fraction of the task's score (0 to 1) one question takes when the request relates its price to the score, else null, "quote": exact words}; null otherwise.

TOOLS
- "roles": an object with the keys "search", "details", "view", "message", "cart", "cart_view", "order" and "decline", each mapped to the exact name of the listed tool doing that job, or null. "message" writes to the shopper; "cart" puts items into the cart or takes them out; "order" checks out or buys; "decline" ends the task without a purchase. "order" and "decline" take an object rather than a name: {"tool": exact tool name, "fixed": {argument: value}}, where "fixed" sets every required argument that has a list of allowed values to the exact value making the tool do that job ({} when there is none). When one tool serves both jobs, its two "fixed" objects differ.

Account for every sentence of the request as exactly one of these: requirement, priority, soft want, rule, budget, hint, or pure story.
"""

UPDATE_READING_TEXT = """A shopper has just said something new. Work out what it changes for the shopping assistant and answer with a single JSON object, with nothing around it, holding these fields:

- "meaning": one short sentence saying what the shopper means.
- "needs": firm product requirements that are new in this line, each "property: constraint" with the exact value, limit, closed list, exclusion ("property: != value") or numeric relation ("property: <= N", ">= N" or "covers N") given. Use the property the shopper's words name, even if the assistant asked about a different one.
- "dropped": earlier requirements that no longer apply.
- "no_preference": properties the shopper does not mind about.
- "new_budget": the new highest total price as a number; otherwise null.
- "unavailable": false, unless the shopper reports that something cannot be bought any more (it is gone, sold out, has no stock left, its button is disabled) - then true.
- "requirements_now": the full list of firm product requirements as it stands after everything said up to now, rebuilt from the request, the earlier shopper lines and this one. Each earlier requirement keeps its earlier wording and stays unless dropped or replaced; a replaced one keeps only its newest form; every newly revealed need joins the list. Same forms as in "needs" (a value the request names and explains keeps its own word, the explanation in parentheses after it). Price, stock, quantity, the product category and background on why or when the item is wanted are not requirements.
- "reply": when the shopper asks the assistant something or wants a confirmation, a short plain answer to send back that names the change in substance (confirming the new value, e.g.); "" otherwise.
- "answer_class": null unless the assistant's question is shown; then {"class": "reveal" (a wish for the property asked about is given), "not_picky" (no preference on it), "split" (the question is taken as several things, or the shopper asks which thing is meant), "unclear" (the shopper did not understand the message and asks for other wording) or "other"; "key": the property the answer is about; "value": the wanted value, reveal only; "relation": "is", "at most", "at least" or "not", reveal only; "quote": the shopper's exact words carrying it}.

How to read the line:
* Several changes can sit in one line; report every one of them.
* A replacement shows up twice: the new requirement under "needs", the old one under "dropped".
* Any budget the shopper states, higher or lower, goes to "new_budget" and replaces the old one, good news included.
* When the shopper says which of two wishes matters more, that one is a need.
* Under, over, at most, at least, no more than: such a comparison with a number makes a limit; keep its unit.
* A bare "anything works", "no preference" or "don't mind" puts the asked property in "no_preference".
* Wishes ("ideally", "would be nice" and the like) are not needs.
* Take the words at face value yet kindly: a value named in a homely, figurative or roundabout way is still firmly wanted; treat a value as excluded only if the shopper clearly asks to stay away from it."""

ITEM_CHECK_TEXT = """Decide whether one candidate item may be ordered. The order only counts when the item meets every requirement listed below, and you are the final check before it is placed; judge from the records shown here and nothing else.

Reply with a single JSON object, no other text: {"item": the item's id, "fits": true or false, "problems": [short strings, one for each requirement not met or not shown]}

Set "fits" to false only for a requirement the records clearly fail or that no record mentions at all. Missing stock or price records are no reason to fail the item.

Matching a requirement:
- Look for the value under the item's own property for that requirement. Every key naming the property counts, however it is spelled (spaces, dots, underscores, prefixes), and a property can sit under several keys. Match by meaning: case does not matter, singular/plural and close synonyms are fine, one element of a comma list is enough; the wanted word with a separate qualifier before it (a shade, a finish) still matches, while one longer fused word that merely contains it does not; a wider or different class, or a word that only looks alike, does not match. A requirement written as a value followed by an explanation in parentheses is met by that value; the explanation only tells what it means.
- If a key named like the property itself (any spelling, singular or plural) gives one single different value, the item fails, even when another key's list contains the wanted value.
- A related but different property's value does not count.
- Title and description, or the variant or option key, may decide only for an item with no key for that property: a wanted value stated whole there ("options: teal" for a teal colour) meets it. When the item's own property gives another value, the requirement fails whatever the title says. A word in another language, or a conclusion drawn from other facts, does not state the value.
- Numbers: convert units before comparing; an edge value is inside the limit; a range passes only if all of it meets the limit. "covers N" requires the listing's own stated range to reach N, so a range ending short of N fails. If the requirements note that the store states the value in the request's own wording, only a listing stating that wording meets it.
- Closed lists and exclusions must hold. An exclusion holds only when the listing's own key for that property shows another value; silence about it is to be reported.

Other checks:
- Kind of item: it must be of the requested kind. Where listings carry category labels the label decides; it must match the request's wording as a whole, not a narrower or neighbouring label, and a listing under that label is of the requested kind whatever its title or description suggests (even a title that reads like a part, an add-on or another sort of item).
- Stock and money: shown as out of stock is a fail; price times quantity may not go over the budget (equal passes).
- When records disagree (on stock or price, say), the newest one wins.
- A wish marked as not required is not a requirement."""

QUESTION_PLAN_TEXT = """You help a shopping assistant decide its next message to the shopper. Shoppers often hold back a need, and such a need is always the wanted value of one single product property; it only surfaces when a message asks about exactly that property. So the job is to name one property and phrase one question about it.

Material you get below: the requirements in force; perhaps a "Must come from the shopper" line; the questions sent earlier with what came back; the "Detail-sheet properties so far" block from the opened listings (each line = name, number of listings showing it, sample values; every line counts as its own property even if two names look alike, and names inside "[other spellings: ...]" are just other spellings of that line); maybe a numbered candidate list ("These are the candidates ..."); maybe an "Answer judged to mix several things" note; and the request.

Ruled out:
1. properties absent from every listing, and properties asked before;
2. properties whose values are mostly the name of the requested item itself (the item word already decides them);
3. properties the request has settled - its stated requirements, price, stock, quantity - with two exceptions that remain open: one the request only offers as a choice between alternatives, and one stated with a broad word the listings divide into narrower values (wanting "wood" where listings show oak, pine or walnut). One exact value that the listings use unchanged keeps it settled;
4. terms of the sale or of the listing, as opposed to features of the item;
5. a key gathering the buyable variants, or one naming several features together (mixed kinds of values, or long item lists): hidden needs never take that shape.
Never ask for a confirmation, a choice among listed options, whatever else the shopper might want, the purpose, or the budget.

Ranking what remains: the likeliest home of an unspoken need wins. Clues come from the request (for whom, where and how it is used, how it should look), the requirements, earlier questions and answers, and the listing properties. An unasked "Must come from the shopper" item comes before everything. If two wishes compete over a single property, ask about that property and mention both values; wishes about two different properties need two separate messages. Coverage beats rarity (a property most or all opened listings state wins over one only a few carry), and variety beats uniformity (values that differ between listings win over a value marked "(identical on all listings)"). When the candidate list is present, choose from it - from the first three if the list demands it - guided by the request's clues.

The question itself: a single short, plain, open sentence (far under a hundred characters) put to the shopper on that property, using the key words the listings use. Where that key word also sits inside other, longer keys, add the item's own noun plus one or two values that occur only under this key, so it cannot be confused with the neighbour; otherwise no example values. One noun for the item, never two item names glued together; when the request is for several different products, name only the one product the property belongs to, in the request's own words for it. Nothing to approve, no second question, no lists, no instructions.

When the "Answer judged to mix several things" note is there, ask about that same property again but more precisely - add the item's noun, or if it already had it, one or two values found only under that property - with wording that differs from the earlier question, and set "repeat" to true; otherwise "repeat" is false.

Answer format, a single JSON object and no other text:
{"property": "...", "question": "...", "chance": 0.0, "repeat": false, "deviate": ""}
"chance" = your estimate between 0 and 1 that the property hides an unspoken need. "deviate" = "" when the pick is one the candidate block allows; for any other pick, the reason (a clue in the request pointing to it). If nothing worth a message remains: {"property": null}."""

DECLINE_AUDIT_TEXT = """Nothing suitable - that is what a shopping assistant is about to report to the shopper. Take one more careful look at the records first. After these instructions come the requirements in force, the shopper's request and the listing records.

Judging a record: count a requirement as met only when the listing's own key for that property carries the value (how the key is spelled does not matter). A comma list holding it, a plural for a singular or a near synonym all pass; a wider or other class does not, and an identically named key holding one single different value sinks the listing. Fall back on title or description only where no such key exists. Convert units before any numeric comparison; a value exactly on a limit is acceptable. Price inside the budget; the listing may not be marked out of stock.

Answer with one JSON object only: {"possible_fits": [up to 5 ids of listings that look like they meet every requirement, best first], "reason": "one sentence"}."""


# Model calls and store calls go through two separate proxy links.
_eja_model_link = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_eja_world_link = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

_ID_SHAPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")


# ---------------------------------------------------------------- JSON values as text

def _shorten_value(field_value, limit, nest_level=0):
    """Shortened copy of a JSON value: strings over limit are cut, anything nested past eight levels is None."""
    def shrink(branch, level):
        if level > 8:
            return None
        if isinstance(branch, str):
            return branch if len(branch) <= limit else branch[:limit] + "..."
        if isinstance(branch, dict):
            return {key: shrink(nested, level + 1) for key, nested in branch.items()}
        if isinstance(branch, list):
            return [shrink(nested, level + 1) for nested in branch]
        return branch
    return shrink(field_value, nest_level)


_eja_ENCODER = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))


def _to_compact_json(field_value):
    """Single-line JSON text of value, cut at the text cap."""
    try:
        text = _eja_ENCODER.encode(field_value)
    except (TypeError, ValueError):
        text = str(field_value)
    if len(text) > _eja_TEXT_CAP:
        text = text[:_eja_TEXT_CAP] + "...(cut)"
    return text


_eja_FENCE_HEAD = re.compile(r"^```[A-Za-z]*")


def _eja_load_object(text):
    """(could be decoded, the decoded dict or None)."""
    try:
        decoded = json.loads(text)
    except ValueError:
        return False, None
    return True, (decoded if isinstance(decoded, dict) else None)


def _extract_json_dict(text):
    """The JSON object a model wrote: plain, inside a code fence, or between the first and last brace."""
    if not isinstance(text, str):
        return None
    body = text.strip()
    if body.startswith("```"):
        body = _eja_FENCE_HEAD.sub("", body).strip()
        if body.endswith("```"):
            body = body[:-3].strip()
    decoded, found = _eja_load_object(body)
    if decoded:
        return found
    opened = body.find("{")
    closed = body.rfind("}")
    if opened == -1 or closed <= opened:
        return None
    return _eja_load_object(body[opened:closed + 1])[1]


def _gather_sentences(field_value, out, nest_level=0):
    """Append every string of four words or more, looking into nested lists (ten entries, four levels)."""
    if nest_level > 4:
        return
    if isinstance(field_value, list):
        for element in field_value[:10]:
            _gather_sentences(element, out, nest_level + 1)
    elif isinstance(field_value, str) and len(field_value.split()) > 3:
        out.append(field_value)


def _gather_ids(field_value, out, nest_level=0):
    """Put into out each id-like string found under a key whose name mentions an id (six levels at most)."""
    stack = [(nest_level, field_value)]
    while stack:
        level, branch = stack.pop()
        if level > 6:
            continue
        below = []
        if isinstance(branch, list):
            below = list(branch)
        elif isinstance(branch, dict):
            for key, nested in branch.items():
                if not isinstance(nested, str) or "id" not in key.lower() or not _ID_SHAPE.match(nested):
                    below.append(nested)
                    continue
                out.add(nested)
        stack.extend((level + 1, nested) for nested in below)


# ---------------------------------------------------------------- tool arguments against their JSON schema

def _as_number(field_value):
    """value itself when it is an int or float other than a bool; None otherwise."""
    if field_value is True or field_value is False:
        return None
    return field_value if isinstance(field_value, (int, float)) else None


def _declared_types(arg_shape):
    """The schema's declared types as a list (or whatever non-string the schema holds there)."""
    declared = arg_shape.get("type")
    if isinstance(declared, str):
        return [declared]
    return declared


_NO_CONVERSION = object()   # a conversion that does not apply
_eja_INT_TEXT = re.compile(r"\s*-?\d+(\.0+)?\s*")
_eja_NUM_TEXT = re.compile(r"\s*-?\d+(\.\d+)?\s*")
_eja_PLAIN_KINDS = {"string": str, "boolean": bool, "object": dict, "array": list}
_eja_KNOWN_KINDS = frozenset(["integer", "number", "null", *_eja_PLAIN_KINDS])


def _eja_kind_holds(kind, field_value):
    if kind == "null":
        return field_value is None
    if kind in _eja_PLAIN_KINDS:
        return isinstance(field_value, _eja_PLAIN_KINDS[kind])
    real = _as_number(field_value) is not None
    return real and isinstance(field_value, int) if kind == "integer" else real


def _eja_admits(field_value, type_names):
    """True when value passes one declared kind; a kind this code does not know admits everything."""
    for kind in type_names:
        if not isinstance(kind, str) or kind not in _eja_KNOWN_KINDS or _eja_kind_holds(kind, field_value):
            return True
    return False


def _eja_to_integer(field_value):
    if isinstance(field_value, float):
        return int(field_value) if field_value.is_integer() else _NO_CONVERSION
    if isinstance(field_value, str) and _eja_INT_TEXT.fullmatch(field_value):
        return int(float(field_value))
    return _NO_CONVERSION


def _eja_to_number(field_value):
    if not isinstance(field_value, str) or not _eja_NUM_TEXT.fullmatch(field_value):
        return _NO_CONVERSION
    amount = float(field_value)
    return int(amount) if amount.is_integer() else amount


def _eja_to_text(field_value):
    return _NO_CONVERSION if _as_number(field_value) is None else str(field_value)


def _eja_to_flag(field_value):
    term = field_value.lower() if isinstance(field_value, str) else ""
    if term == "true":
        return True
    if term == "false":
        return False
    return _NO_CONVERSION


_eja_CONVERTERS = {"integer": _eja_to_integer, "number": _eja_to_number, "string": _eja_to_text, "boolean": _eja_to_flag}


def _eja_coerce(field_value, arg_shape):
    """value moved to a declared type when that loses nothing; else unchanged."""
    type_names = _declared_types(arg_shape)
    if not type_names:
        return field_value
    if _eja_admits(field_value, type_names):
        whole = isinstance(field_value, float) and field_value.is_integer()
        return int(field_value) if whole and "integer" in type_names else field_value
    for kind in type_names:
        if not isinstance(kind, str) or kind not in _eja_CONVERTERS:
            continue
        try:
            moved = _eja_CONVERTERS[kind](field_value)
        except (TypeError, ValueError):
            continue
        if moved is not _NO_CONVERSION:
            return moved
    return field_value


# Bounds per measured thing: (schema key, bound must be numeric, broken when, message pattern).
_eja_BOUNDS = {
    "number": (
        ("minimum", False, lambda got, edge: got < edge, "is under the minimum {}"),
        ("maximum", False, lambda got, edge: got > edge, "is over the maximum {}"),
        ("exclusiveMinimum", True, lambda got, edge: got <= edge, "has to be greater than {}"),
        ("exclusiveMaximum", True, lambda got, edge: got >= edge, "has to be less than {}"),
    ),
    "text": (("minLength", False, lambda got, edge: got < edge, "needs at least {} characters"),),
    "list": (
        ("minItems", False, lambda got, edge: got < edge, "give at least {} items"),
        ("maxItems", False, lambda got, edge: got > edge, "give at most {} items"),
    ),
}


def _eja_bound_issues(measure, arg_shape, group, trail):
    found = []
    for key, numeric_only, broken, pattern in _eja_BOUNDS[group]:
        if key not in arg_shape:
            continue
        edge = arg_shape[key]
        if numeric_only and not isinstance(edge, (int, float)):
            continue
        if broken(measure, edge):
            found.append("%s: %s" % (trail, pattern.format(edge)))
    return found


def _eja_pick_option(field_value, arg_shape, trail, issues):
    """Enum (with a case-blind repair for text) and const checks."""
    if "enum" in arg_shape:
        choices = arg_shape.get("enum")
        if field_value not in choices:
            if isinstance(field_value, str):
                lowered = field_value.strip().lower()
                for option in choices:
                    if isinstance(option, str) and option.lower() == lowered:
                        field_value = option
                        break
            if field_value not in choices:
                issues.append("%s: allowed values are %s" % (trail, choices))
    if "const" in arg_shape:
        only = arg_shape["const"]
        if field_value != only:
            issues.append("%s: the only allowed value is %r" % (trail, only))
    return field_value


def _eja_check_members(field_value, arg_shape, trail, issues):
    """Required and unknown names of an object, then each known member in turn."""
    member_list = arg_shape.get("properties") or {}
    for label_name in arg_shape.get("required") or []:
        if label_name not in field_value:
            issues.append("%s.%s: missing (required)" % (trail, label_name))
    if arg_shape.get("additionalProperties") is False:
        for label_name in list(field_value):
            if label_name not in member_list:
                issues.append("%s.%s: unknown argument" % (trail, label_name))
    out = {}
    for label_name, nested in field_value.items():
        if label_name in member_list:
            nested, inner_issues = _validate_value(nested, member_list[label_name], "%s.%s" % (trail, label_name))
            issues.extend(inner_issues)
        out[label_name] = nested
    return out


def _eja_check_entries(field_value, arg_shape, trail, issues):
    """Count bounds of a list, then each entry against the item schema."""
    issues.extend(_eja_bound_issues(len(field_value), arg_shape, "list", trail))
    shape = arg_shape.get("items")
    if not isinstance(shape, dict):
        return field_value
    out = []
    for at, element in enumerate(field_value):
        element, entry_issues = _validate_value(element, shape, "%s[%d]" % (trail, at))
        out.append(element)
        issues.extend(entry_issues)
    return out


def _eja_word_cut(text, cap):
    """Text cut to `cap` characters; a cut inside a later word goes back to the end of the word before it."""
    cut = text[:max(0, cap)]
    if len(text) > len(cut) and cut[-1:].isalnum() and text[len(cut)].isalnum() and cut.strip().count(" "):
        cut = cut.rstrip().rsplit(None, 1)[0]
    return cut.rstrip() or text[:max(0, cap)]


def _validate_value(field_value, arg_shape, trail):
    """Check value against a JSON schema, repairing what is safe; returns (value, list of problems)."""
    if not isinstance(arg_shape, dict):
        return field_value, []
    field_value = _eja_coerce(field_value, arg_shape)
    type_names = _declared_types(arg_shape)
    if type_names and not _eja_admits(field_value, type_names):
        return field_value, [trail + ": should be " + "/".join(type_names)]
    issues = []
    field_value = _eja_pick_option(field_value, arg_shape, trail, issues)
    if isinstance(field_value, str):
        issues.extend(_eja_bound_issues(len(field_value), arg_shape, "text", trail))
        cap = arg_shape.get("maxLength")
        if "maxLength" in arg_shape and len(field_value) > cap:
            field_value = _eja_word_cut(field_value, cap)
    if _as_number(field_value) is not None:
        issues.extend(_eja_bound_issues(field_value, arg_shape, "number", trail))
    if isinstance(field_value, list):
        field_value = _eja_check_entries(field_value, arg_shape, trail, issues)
    if isinstance(field_value, dict):
        field_value = _eja_check_members(field_value, arg_shape, trail, issues)
    return field_value, issues


# ---------------------------------------------------------------- seats, tools and the brief

def _eja_unmarked(part):
    if not isinstance(part, dict):
        return part
    clean = dict(part)
    clean.pop("cache_control", None)
    return clean


def _eja_seat_messages(model, messages):
    """The history as one seat should get it: Anthropic seats with a longer history get cache marks on the first
    and last plain-text messages; seats of other vendors get every cache mark removed."""
    if not model.startswith("anthropic/"):
        cleaned = []
        for msg_text in messages:
            parts = msg_text.get("content")
            if isinstance(parts, list):
                msg_text = dict(msg_text, content=list(map(_eja_unmarked, parts)))
            cleaned.append(msg_text)
        return cleaned
    if len(messages) < 3:
        return messages
    marked = list(map(dict, messages))
    for at in (0, len(marked) - 1):
        text = marked[at].get("content")
        if text and isinstance(text, str):
            marked[at]["content"] = [dict(type="text", text=text, cache_control=dict(type="ephemeral"))]
    return marked


def _eja_status(result_info):
    fault = result_info.error
    return fault.get("status") if isinstance(fault, dict) else None


_eja_ROLES_PROMPT = """Here is the tool list of a web shop. For every job below, find the tool doing it and give its name exactly as listed (null when none does; the same tool may cover two jobs):
search - finds listings
details - one listing's full set of properties
view - a listing together with its price and stock
message - writes a note to the shopper
cart - adds a listing to the basket or removes it
cart_view - displays the basket
order - buys the basket's contents, ending the task
decline - ends the task with no purchase
Output a single one-line JSON object, nothing else: {"roles": {"search": ..., "details": ..., "view": ..., "message": ..., "cart": ..., "cart_view": ..., "order": ..., "decline": ...}}"""

_eja_TOKEN_FACTOR = {FALLBACK_SEAT: 4, ALT_VENDOR_SEAT: 2}   # seats that think before they write get more room
_eja_PASSING_FAULTS = frozenset([429, 500, 502, 503, 504])     # model faults that tend to clear in seconds
_eja_RETRY_WAITS = (1.5, 4.0)
_eja_STORE_GIVE_UP = frozenset([403, 422, 500, 503, 504])
_eja_INFERENCE_PATH = "/inference/chat/completions"
_eja_SETUP_SEATS = (
    # (seats, max_tokens) for the first reading, the roomier repeat after a cut reply, and the last try
    ([INTAKE_SEAT, ALT_VENDOR_SEAT, PRIMARY_SEAT, FALLBACK_SEAT], 1500),
    ([INTAKE_SEAT, ALT_VENDOR_SEAT, FALLBACK_SEAT], 4500),
    ([INTAKE_SEAT, ALT_VENDOR_SEAT, FALLBACK_SEAT, PRIMARY_SEAT], 4500),
)
_eja_ROLE_SEATS = [INTAKE_SEAT, PRIMARY_SEAT, FALLBACK_SEAT]
_OPENING_SECTIONS = (
    ("soft_wants", "- nice to have, NOT required (only a tie-breaker): "),
    ("ask_first", "- must be learned from the shopper first: "),
    ("rules", "- rule: "),
)


def _describe_tool(tool_desc):
    """Catalogue line of one tool: name, description and argument schema."""
    return "* %s -- %s -- argument schema: %s" % (tool_desc["name"], tool_desc.get("description", ""),
                                                  _to_compact_json(tool_desc.get("parameters") or {}))


def _eja_has_reading(plan):
    return bool(plan.get("product") or plan.get("requirements"))


def _eja_is_tool(element):
    nested = element.get("function") if isinstance(element, dict) else None
    return isinstance(nested, dict)


def _eja_params(tool_desc):
    return tool_desc.get("parameters") or dict(type="object", properties={})


# ---------------------------------------------------------------- tool roles read from the tool list itself
# Every world renames its tools and arguments, so this reading uses argument shapes first, then words that the
# descriptions and names of two tools share. A job it cannot settle is left out and the model's reading stands.

def _eja_name_words(text):
    """Words of a tool name or description: separators and lower-to-upper case steps split words."""
    return _word_set(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", re.sub(r"[_.-]+", " ", str(text or ""))))


def _eja_arg_kind(arg_spec, depth=0):
    """Shape of one argument: choice, count, number, text, ids (a list of texts), lines (listing lines, each a text
    and a whole number, in a list or inside an object), or '' for anything else."""
    if not isinstance(arg_spec, dict) or depth > 4:
        return ""
    if _is_enum_spec(arg_spec):
        return "choice"
    kinds = _declared_types(arg_spec)
    kinds = kinds if isinstance(kinds, list) else []
    for type_name, shape in (("integer", "count"), ("number", "number"), ("string", "text")):
        if type_name in kinds:
            return shape
    inner = arg_spec.get("items") if "array" in kinds else arg_spec if "object" in kinds else None
    if not isinstance(inner, dict):
        return ""
    if inner is not arg_spec:
        member = _eja_arg_kind(inner, depth + 1)
        if member in ("text", "lines"):
            return "ids" if member == "text" else "lines"
    members = inner.get("properties")
    parts = [_eja_arg_kind(sub, depth + 1) for sub in members.values()] if isinstance(members, dict) else []
    return "lines" if "lines" in parts or {"text", "count"} <= set(parts) else ""


def _eja_tool_shapes(tools):
    """name -> argument shapes, required names, room of each text argument, words and description sentences."""
    shapes = {}
    for tool_desc in (tk["function"] for tk in tools if _eja_is_tool(tk)):
        params = _eja_params(tool_desc)
        props = params.get("properties") if isinstance(params, dict) else None
        if not (isinstance(tool_desc.get("name"), str) and isinstance(props, dict)):
            continue   # a tool this reading cannot take apart says nothing
        must = params.get("required") if isinstance(params.get("required"), list) else []
        said = str(tool_desc.get("description") or "")
        pieces = (_eja_name_words(piece) for piece in re.split(r"[.!?]+", said))
        shapes[tool_desc["name"]] = dict(
            kinds={arg: _eja_arg_kind(sub) for arg, sub in props.items()}, props=props,
            must={arg for arg in must if isinstance(arg, str) and arg in props},
            words=_eja_name_words(tool_desc["name"]) | _eja_name_words(said), said=said,
            sentences={frozenset(piece) for piece in pieces if len(piece) > 1})
    return shapes


def _eja_text_room(arg_spec):
    """Longest text an argument takes (no stated maximum counts as no limit)."""
    room = arg_spec.get("maxLength") if isinstance(arg_spec, dict) else None
    return room if isinstance(room, int) and not isinstance(room, bool) else float("inf")


def _eja_id_room(shape):
    """Room of the listing id a cart tool takes: its one text argument, or the text inside its listing lines."""
    for arg, kind in shape["kinds"].items():
        spec = shape["props"][arg]
        if kind == "text":
            return _eja_text_room(spec)
        while kind == "lines" and isinstance(spec, dict):
            spec = spec.get("items") if isinstance(spec.get("items"), dict) else spec
            inner = spec.get("properties") if isinstance(spec.get("properties"), dict) else {}
            texts = [sub for sub in inner.values() if _eja_arg_kind(sub) == "text"]
            if texts:
                return _eja_text_room(texts[0])
            spec = next((sub for sub in inner.values() if _eja_arg_kind(sub) == "lines"), None)
    return None


def _eja_is_cart(shape):
    """A tool putting listings in the cart: listing lines, or one required text (the id) next to a whole number."""
    kinds = shape["kinds"]
    texts = [arg for arg, kind in kinds.items() if kind == "text"]
    if "lines" in kinds.values():
        return True
    return len(texts) == 1 and texts[0] in shape["must"] and "count" in kinds.values()


def _eja_pick_cart(shapes, common):
    """The cart tool: the one cart-shaped tool; of several, the one taking listing lines, else the one sharing most
    words with the tools that need no argument (the cart view among them)."""
    carts = [name for name, shape in shapes.items() if _eja_is_cart(shape)]
    if len(carts) < 2:
        return carts[0] if carts else None
    lined = [name for name in carts if "lines" in shapes[name]["kinds"].values()]
    if len(lined) == 1:
        return lined[0]
    bare = set().union(*(shape["words"] for shape in shapes.values() if not shape["kinds"])) - common
    return _eja_unique_top(carts, lambda name: len(shapes[name]["words"] & bare))


def _eja_unique_top(names, score):
    """The one name scoring above every other one (and above zero), or None."""
    ranked = sorted(names, key=score, reverse=True)
    if not ranked or score(ranked[0]) <= 0 or (len(ranked) > 1 and score(ranked[1]) == score(ranked[0])):
        return None
    return ranked[0]


def _eja_value_sides(said, values, marks):
    """For a choice of two values the description names: (the value whose stretch of the description, from it up
    to the next value, shares more cart words, the other value); None when the description does not tell."""
    low = said.lower()
    spots = []
    for choice_value in values:
        hit = re.search(r"\b%s\b" % re.escape(str(choice_value).lower()), low)
        if not hit:
            return None
        spots.append((hit.start(), hit.end(), choice_value))
    spots.sort()
    reach = {}
    for pos, (_, end, choice_value) in enumerate(spots):
        stop = spots[pos + 1][0] if pos + 1 < len(spots) else len(low)
        reach[choice_value] = len(_eja_name_words(low[end:stop]) & marks)
    first, second = values
    if reach[first] == reach[second]:
        return None
    return (first, second) if reach[first] > reach[second] else (second, first)


def _eja_closing_choice(shape, marks):
    """A tool ending the task both ways by one required choice of two values (other arguments optional text):
    (order pins, decline pins), or None."""
    picks = [arg for arg in shape["must"] if shape["kinds"][arg] == "choice"]
    rest = [kind for arg, kind in shape["kinds"].items() if arg not in picks]
    if len(picks) != 1 or len(shape["must"]) != 1 or any(kind != "text" for kind in rest):
        return None
    values = shape["props"][picks[0]].get("enum")
    if not isinstance(values, list) or len(values) != 2 or not all(isinstance(fv, str) for fv in values):
        return None
    sides = _eja_value_sides(shape["said"], values, marks)
    return None if sides is None else ({picks[0]: sides[0]}, {picks[0]: sides[1]})


def _eja_shape_roles(tools):
    """What the tool list itself says about the roles: job -> list of (tool, pinned values or None). One entry is a
    sure reading; several mean the job is one of those tools. A job left out is one this reading cannot settle."""
    shapes = _eja_tool_shapes(tools)
    counts = {}
    for shape in shapes.values():
        for tk in shape["words"]:
            counts[tk] = counts.get(tk, 0) + 1
    common = {tk for tk, seen in counts.items() if 2 * seen > len(shapes)}   # words most tools carry tell nothing
    found = {}
    cart = _eja_pick_cart(shapes, common)
    if cart is None:
        return found
    found["cart"] = [(cart, None)]
    marks = shapes[cart]["words"] - common
    overlap = lambda name: len(shapes[name]["words"] & marks)
    left = [name for name in shapes if name != cart]
    closing = {name: _eja_closing_choice(shapes[name], marks) for name in left}
    closing = {name: pins for name, pins in closing.items() if pins}
    if len(closing) == 1:
        both, pins = next(iter(closing.items()))
        found["order"], found["decline"] = [(both, pins[0])], [(both, pins[1])]
        left.remove(both)
    wide = [name for name in left if len(shapes[name]["kinds"]) > 1
            and set(shapes[name]["kinds"].values()) & {"text", "choice"}]
    search = _eja_unique_top(wide, lambda name: len(shapes[name]["kinds"]))
    if search:
        found["search"] = [(search, None)]
    _eja_single_roles(shapes, cart, [name for name in left if name != search], overlap, found)
    _eja_bare_roles(shapes, [name for name in left if name != search], overlap, found)
    return found


def _eja_single_roles(shapes, cart, left, overlap, found):
    """Tools of one argument: a list of ids compares; a required text is the shopper message when it takes far
    longer texts than a listing id, otherwise it opens one listing (or takes it out of a cart that only adds)."""
    lone = [name for name in left if len(shapes[name]["kinds"]) == 1]
    lists = [name for name in lone if "ids" in shapes[name]["kinds"].values()]
    texts = [name for name in lone if "text" in shapes[name]["kinds"].values() and shapes[name]["must"]]
    room = {name: _eja_text_room(next(iter(shapes[name]["props"].values()))) for name in texts}
    id_room = _eja_id_room(shapes[cart])
    if id_room is None and len(texts) > 1:
        id_room = min(room.values())
    wordy = [name for name in texts if id_room is not None and room[name] > 2 * id_room]
    if len(wordy) == 1:
        found["message"] = [(wordy[0], None)]
    short = [name for name in texts if name not in wordy]
    if "choice" not in shapes[cart]["kinds"].values() and len(short) > 1:
        remover = _eja_unique_top(short, overlap)
        if remover:
            found["cart_remove"] = [(remover, None)]
            short.remove(remover)
    if len(short) == 1:
        found["details"] = [(short[0], None)]
    if len(lists) == 1:
        found["view"] = [(lists[0], None)]
    elif not lists and len(short) == 2:   # one listing shown two ways: the two tools share details and view
        found["details"] = found["view"] = [(name, None) for name in short]


def _eja_bare_roles(shapes, left, overlap, found):
    """Tools needing no argument (an optional note at most): the cart view and the two closing tools. A sentence
    two of them alone share marks those two as the closing pair; otherwise the one sharing most cart words is the
    view. Of the closing pair, the one sharing more cart words orders."""
    bare = [name for name in left if not shapes[name]["must"] and len(shapes[name]["kinds"]) <= 1
            and set(shapes[name]["kinds"].values()) <= {"text"}]
    if "order" in found:
        if len(bare) == 1:
            found["cart_view"] = [(bare[0], None)]
        return
    pairs = []
    for pos, one in enumerate(bare):
        for other in bare[pos + 1:]:
            joint = shapes[one]["sentences"] & shapes[other]["sentences"]
            elsewhere = set().union(*(shapes[name]["sentences"] for name in shapes if name not in (one, other)))
            if joint - elsewhere:
                pairs.append((one, other))
    if len(pairs) == 1:
        closers = list(pairs[0])
    elif len(bare) == 3:
        top = _eja_unique_top(bare, overlap)
        closers = [name for name in bare if name != top] if top else []
    else:
        closers = []
    views = [name for name in bare if name not in closers]
    if len(closers) == 2 and len(views) == 1:
        found["cart_view"] = [(views[0], None)]
    buyer = _eja_unique_top(closers, overlap) if len(closers) == 2 else None
    if buyer:
        found["order"] = [(buyer, {})]
        found["decline"] = [(next(name for name in closers if name != buyer), {})]


def _eja_job_fits(duty, shape):
    """Whether a tool has the argument shape a job needs (a model reading naming a tool of the wrong shape is off)."""
    kinds, must = shape["kinds"], shape["must"]
    needed_text = any(kinds[arg] == "text" for arg in must)
    if duty == "search":
        return bool(set(kinds.values()) & {"text", "choice"})
    if duty in ("details", "cart_remove"):
        return needed_text
    if duty == "view":
        return needed_text or "ids" in kinds.values()
    if duty == "message":
        return "text" in kinds.values()
    if duty == "cart":
        return bool(set(kinds.values()) & {"text", "lines"})
    if duty == "cart_view":
        return not must
    if duty in _CLOSING_JOBS:
        return all(kinds[arg] in ("choice", "text") for arg in must)
    return True


class _NeedList:
    """The shopper's wants as they stand: stated and revealed conditions, drops, free properties, budget."""

    def __init__(self):
        for bucket in ("stated", "revealed", "dropped", "free", "history"):
            setattr(self, bucket, [])
        self.newest_lines = self.budget = None
        self.heard = set()

    def render_lines(self, currency, quantity):
        def alive(pool, tag):
            return [element + tag for element in pool if element not in self.dropped]
        if self.newest_lines is None:
            entry_list = alive(self.stated, " (stated)") + alive(self.revealed, " (revealed by the shopper)")
        else:
            entry_list = list(self.newest_lines)
        chunks = ["; ".join(entry_list) or "see the request"]
        if self.budget is not None:
            chunks.append(("total price at most %s %s" % (self.budget, currency)).rstrip())
        chunks.append("quantity %s" % (quantity,))
        tails = ((self.dropped, "no longer required: ", "; "), (self.free, "shopper does not care about: ", ", "))
        chunks.extend(opener + glue.join(pool) for pool, opener, glue in tails if pool)
        return "; ".join(chunks)

    @staticmethod
    def _eja_add_new(target, incoming):
        """Extend target with the new non-empty strings of incoming; return just those."""
        new_lines = []
        for element in incoming or []:
            if isinstance(element, str) and element and element not in target:
                target.append(element)
                new_lines.append(element)
        return new_lines

    def fold_reading(self, parsed_reading):
        """Fold one reading of a shopper line into the wants; returns notes on what changed."""
        current = parsed_reading.get("requirements_now")
        if isinstance(current, list) and not any(not isinstance(element, str) for element in current):
            self.newest_lines = list(filter(str.strip, current))
        notes = list(map("new requirement: ".__add__, self._eja_add_new(self.revealed, parsed_reading.get("needs"))))
        self._eja_add_new(self.free, parsed_reading.get("no_preference"))
        notes += map("dropped: ".__add__, self._eja_add_new(self.dropped, parsed_reading.get("dropped")))
        cap = _as_number(parsed_reading.get("new_budget"))
        if cap is not None and cap > 0 and cap != self.budget:
            self.budget = cap
            notes.append("new budget: %s" % (cap,))
        if parsed_reading.get("unavailable"):
            notes.append("an item is no longer available: do not order it, and take it out of the cart")
        return notes


# Fresh per-episode state: starting values by group, and containers made new for every episode.
_eja_START = (
    (0, "step messages_used order_attempts refusals needs_revision cart_changed_at cart_generation refused_at empty_plans"),
    (0.0, "cost"),
    (-1, "cart_seen_at"),
    (True, "planner_on"),
    (False, "halted order_checked decline_checked key_dry cart_used audit_done split_retried urgent_tried"),
    (None, "message_budget quantity_field add_signature last_cart_view answer_reading split_pending forced_basis"
           " question_cost"),
    ("", "asking sibling_next queued_reply rerank_reason unclear_next"),
)
_eja_CONTAINERS = (
    (set, "seen_ids cart_checked qty_warned held_arms relooked faulty proc_held"),
    (list, "cart_ids records recent asked asked_props reask_later sibling_props proven_fits procedures messages trace"),
    (dict, "cart_state units_in_cart sent_units property_menu verdicts empty_holds mentioned decline_holds faults roles"
           " fixed calls_by_role plan need_homes loose_ok"),
)


class Episode:
    def __init__(self, problem_data):
        self.start_clock = time.monotonic()
        env = problem_data["environment"]
        self.binding = dict(env["binding"])
        self.policy = view = env["policy_view"]
        self.tools = [element for element in view.get("tools") or [] if _eja_is_tool(element)]
        specs = [tool["function"] for tool in self.tools]
        self.param_schemas = {tool_desc["name"]: _eja_params(tool_desc) for tool_desc in specs}
        self.max_steps = int(view.get("max_steps") or 30)
        self.finished = bool(view.get("finished"))
        for start, names in _eja_START:
            for label_name in names.split():
                setattr(self, label_name, start)
        for make, names in _eja_CONTAINERS:
            for label_name in names.split():
                setattr(self, label_name, make())
        self._need_list = _NeedList()

    def _eja_log(self, kind, **field_specs):
        self.trace.append(dict(kind=kind, **field_specs))

    # ------------------------------------------------------------ limits

    def spend_exceeded(self):
        """Model spend above the episode cap: optional judge passes are skipped."""
        return self.cost > _eja_SPEND_CAP

    def seconds_remaining(self):
        return _eja_EPISODE_LIMIT - (time.monotonic() - self.start_clock)

    def remaining_steps(self):
        return self.max_steps - self.step

    # ------------------------------------------------------------ model calls

    def infer(self, messages, tools, models, max_tokens):
        """Ask the seats in order and return the first reply with text or tool calls, tagged with its model.
        If every seat was refused for payment or access, the key counts as dry."""
        payment_refusals = 0
        for model in models:
            remaining = self.seconds_remaining()
            if remaining < FORCE_MARGIN - 10:
                return None
            _eja_model_link.timeout = int(min(75, max(8, remaining + 15 - FORCE_MARGIN)))
            result_info = self._eja_ask_seat(model, messages, tools, max_tokens)
            reply_obj = self._eja_answer_of(result_info.data)
            if reply_obj:
                reply_obj["model"] = model
                return reply_obj
            if _eja_status(result_info) in (402, 403):
                payment_refusals += 1
            reason = "empty" if not result_info.error else str(result_info.error)[:300]
            self._eja_log("inference_failure", model=model, error=reason)
        if models and payment_refusals == len(models):
            self.key_dry = True
            self._eja_log("key_dry", models=list(models))
        return None

    def _eja_ask_seat(self, model, messages, tools, max_tokens):
        """One seat's call; a passing upstream fault gets the same seat again after a short wait."""
        body = dict(model=model, messages=_eja_seat_messages(model, messages), temperature=0,
                    max_tokens=max_tokens * _eja_TOKEN_FACTOR.get(model, 1))
        if tools:
            body.update(tools=tools, tool_choice="auto")
        result_info = _eja_model_link.post_verbose(_eja_INFERENCE_PATH, body)
        for wait in _eja_RETRY_WAITS:
            passing = result_info.data is None and _eja_status(result_info) in _eja_PASSING_FAULTS
            if not passing or self.seconds_remaining() < FORCE_MARGIN + 30:
                break
            time.sleep(wait)
            result_info = _eja_model_link.post_verbose(_eja_INFERENCE_PATH, body)
        return result_info

    def _eja_answer_of(self, data):
        """The first choice's message with a "cut" flag, or None when it is empty; the call's cost is counted."""
        if not isinstance(data, dict) or not data.get("choices"):
            return None
        usage = data.get("usage") or {}
        if isinstance(usage.get("cost"), (int, float)):
            self.cost += float(usage["cost"])
        first = data["choices"][0]
        msg_text = dict(first.get("message") or {})
        msg_text["cut"] = first.get("finish_reason") == "length"
        if msg_text.get("content") or msg_text.get("tool_calls"):
            return msg_text
        return None

    # ------------------------------------------------------------ store calls

    def call_store(self, action):
        """Send ONE action as the next step (one action per request), retrying faults that may pass."""
        wrapper = dict(self.binding, step=self.step + 1, action=action)
        for attempt in range(1, 5):
            remaining = self.seconds_remaining()
            if remaining < 2:
                return None
            _eja_world_link.timeout = int(min(95, max(3, remaining - 1)))
            result_info = _eja_world_link.post_verbose("/environment/call", wrapper)
            if isinstance(result_info.data, dict):
                return self._eja_took_step(result_info.data)
            if not self._eja_store_fault(result_info.error or {}, wrapper, attempt):
                return None
        return None

    def _eja_took_step(self, data):
        counted = data.get("step")
        if isinstance(counted, int):
            self.step = counted
        else:
            self.step += 1
        if data.get("finished"):
            self.finished = True
        return data

    def _eja_store_fault(self, error, wrapper, attempt):
        """Log a failed store call; True when sending it again is worth it."""
        status = error.get("status")
        body = str(error.get("body") or "")
        self._eja_log("env_failure", status=status, body=body[:300])
        if status == 402:   # the run is out of money
            self.halted = True
            return False
        if status == 429:   # rate limited: wait, spread by step and attempt so parallel episodes do not collide
            spread = 1.0 + 0.15 * ((7 * self.step + 3 * attempt) % 5)
            wait = 0.6 * attempt * spread
            if self.seconds_remaining() <= wait + FORCE_MARGIN:
                return False
            time.sleep(wait)
            return True
        if status != 409:
            return status not in _eja_STORE_GIVE_UP
        if "finished" in body:
            self.finished = True
            return False
        expected = re.search(r"expected (\d+)", body)
        if not expected:
            return False
        self.step = int(expected.group(1)) - 1
        wrapper["step"] = self.step + 1
        return True

    # ------------------------------------------------------------ reading the request

    def setup(self):
        tool_catalog = "\n".join(map(_describe_tool, (tool["function"] for tool in self.tools)))
        query, opening = self.policy.get("query") or "", self.policy.get("opening")
        story = "THE SHOPPER'S REQUEST:\n" + str(query)
        if opening is not None:
            story = story + "\n\nWHAT THE STORE SHOWED AT THE START:\n" + _to_compact_json(opening)
        lead_block = dict(type="text", text=INTAKE_TEXT + "\n\nTHE STORE'S TOOLS:\n" + tool_catalog,
                    cache_control=dict(type="ephemeral"))
        turn = [dict(role="user", content=[lead_block, dict(type="text", text=story)])]
        # the main model goes before the backup: the backup reasons first and can use up its whole allowance
        first, roomy, last = _eja_SETUP_SEATS
        reply_obj, self.plan = self._eja_read_once(turn, first)
        if not self.plan and reply_obj and reply_obj.get("cut"):
            reply_obj, self.plan = self._eja_read_once(turn, roomy)
        if not (self.key_dry or _eja_has_reading(self.plan)):
            # nothing usable yet (a failed seat, or roles only): one more reading
            self._eja_log("setup_retry", had=sorted(self.plan))
            again = self._eja_read_once(turn, last)[1]
            if _eja_has_reading(again):
                self.plan = again
        if not self._eja_adopt(self.plan):
            self._eja_roles_alone(tool_catalog)

    def _eja_read_once(self, turn, seating):
        seat_list, tokens = seating
        reply_obj = self.infer(turn, None, seat_list, tokens)
        text = (reply_obj.get("content") or "") if reply_obj else ""
        return reply_obj, _extract_json_dict(text) or {}

    def _eja_roles_alone(self, tool_catalog):
        """The full reading named no tool roles: a small call for the roles only."""
        reply = self.ask_judge(_eja_ROLES_PROMPT + "\n\nTOOLS:\n" + tool_catalog, 400, _eja_ROLE_SEATS) or {}
        named = reply.get("roles")
        if isinstance(named, dict):
            self.plan["roles"] = named
            self._eja_take_roles(named)
        self._eja_log("roles_fallback", roles=self.roles)

    def _eja_take_roles(self, named):
        """The model's role map (tools of this list only), held against the tool list's own reading, then the closing
        jobs confirmed. Returns how many roles the model named validly."""
        self.roles, self.fixed = {}, {}
        for duty, element in (named.items() if isinstance(named, dict) else ()):
            tool_desc = element if isinstance(element, dict) else {"tool": element, "fixed": {}}
            tool = tool_desc.get("tool")
            if not isinstance(tool, str) or tool not in self.param_schemas:
                continue
            self.roles[duty] = tool
            if duty in _CLOSING_JOBS:
                preset = tool_desc.get("fixed")
                self.fixed[duty] = dict(preset) if isinstance(preset, dict) else {}
        named_ok = len(self.roles)
        self._eja_check_roles()
        self._confirm_closing_roles()
        return named_ok

    def _eja_check_roles(self):
        """Hold the model's roles against what the tool list itself shows. A job the model left empty, or gave a tool
        of the wrong shape, takes the list's reading when that reading is sure (or leaves one tool not yet bound);
        closing jobs keep the model's tool but take the list's pinned values when the model pinned none. A model
        reading of the right shape stays even where the list reads otherwise: the difference is only logged."""
        read = _eja_shape_roles(self.tools)
        shapes = _eja_tool_shapes(self.tools)
        changed, differs = {}, {}
        for duty, options in read.items():
            mine = self.roles.get(duty)
            names = [tool for tool, _ in options]
            if mine in names:
                pins = options[names.index(mine)][1]
                if duty in _CLOSING_JOBS and pins and not self.fixed.get(duty):
                    self.fixed[duty] = dict(pins)
                    changed[duty] = {"tool": mine, "fixed": pins}
                continue
            if mine and (mine not in shapes or _eja_job_fits(duty, shapes[mine])):
                differs[duty] = {"model": mine, "list": names}
                continue
            taken = {tool for job, tool in self.roles.items() if job != duty and job not in _CLOSING_JOBS}
            free = [opt for opt in options if opt[0] not in taken] if len(options) > 1 else options
            if len(free) != 1:
                continue
            self.roles[duty] = free[0][0]
            if duty in _CLOSING_JOBS:
                self.fixed[duty] = dict(free[0][1] or {})
            changed[duty] = {"was": mine, "now": free[0][0]}
        self._eja_log("roles_code", read={duty: [tool for tool, _ in opts] for duty, opts in read.items()},
                      changed=changed, differs=differs)

    def _eja_adopt(self, parsed_reading):
        """Take the full reading of the request; returns how many roles the model named validly."""
        named_ok = self._eja_take_roles(parsed_reading.get("roles"))
        self.procedures = self._stated_procedures(parsed_reading.get("procedures"))
        self.question_cost = self._keep_priced_ask(parsed_reading.get("question_cost"))
        messages_allowed = _as_number(parsed_reading.get("message_budget"))
        if messages_allowed is not None and messages_allowed >= 0:
            self.message_budget = int(messages_allowed)
        wants = self._need_list
        wants.stated = list(map(str, filter(None, parsed_reading.get("requirements") or [])))
        wants.budget = parsed_reading.get("price_cap")
        steps_allowed = _as_number(parsed_reading.get("step_budget"))
        if steps_allowed is not None and 0 < steps_allowed < self.max_steps:
            self.max_steps = int(steps_allowed)
        return named_ok

    # ---------------------------------------------------------------- small readers shared by this part

    @staticmethod
    def _ejb_cut(need_line):
        """A "property: constraint" text split at its first colon."""
        return str(need_line).partition(":")

    @staticmethod
    def _ejb_key_words(need_line):
        """Words of the property half of a "property: constraint" text."""
        return _word_set(str(need_line).partition(":")[0])

    @staticmethod
    def _ejb_label(field_name):
        """A key as words: separators become spaces and a lower-to-upper case step starts a new word."""
        return re.sub(r"[_.]+", " ", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(field_name)))

    @staticmethod
    def _ejb_short_texts(record_blob, limit):
        return [(label_key, cell) for label_key, cell in record_blob.items() if isinstance(cell, str) and 0 < len(cell) <= limit]

    @staticmethod
    def _ejb_ints(record_blob):
        return {label_key: cell for label_key, cell in record_blob.items() if isinstance(cell, int) and not isinstance(cell, bool)}

    def _ejb_props(self, tool, strict=False):
        """A tool's argument specs (strict: the tool must be known)."""
        arg_shape = self.param_schemas[tool] if strict else (self.param_schemas.get(tool) or {})
        return arg_shape.get("properties") or {}

    def _ejb_pins(self, duty):
        return self.fixed.get(duty) or {}

    # ---------------------------------------------------------------- closing jobs and the steps the request names

    _ejb_REPAIR_HEAD = (
        "Tool mapping fix. Answer shape (JSON only, nothing around it):\n"
        '{"order": {"tool": name, "fixed": {argument: value}}, "decline": {"tool": name, "fixed": {...}}}\n'
        "Key meanings -\n"
        "  order: tool plus pinned arguments that purchase what sits in the cart\n"
        "  decline: tool plus pinned arguments that stop without purchasing\n"
        "  fixed: argument -> exact value, for every required argument limited to listed options or a boolean; "
        "a tool used for both jobs gets the same arguments pinned to distinct values in each entry\n\n"
        "Why the earlier mapping failed:\n"
    )

    def _ejb_job_faults(self, duty):
        """Everything wrong with how one closing job is mapped. Valid pinned values are rewritten in the
        schema's own spelling on the way."""
        tool = self.roles.get(duty)
        if not tool:
            return [f"the '{duty}' job has no tool"]
        arg_specs = self._ejb_props(tool, True)
        pinned = self._ejb_pins(duty)
        must = self.param_schemas[tool].get("required") or []
        out = [f"{tool} needs the choice argument '{a}', which '{duty}' leaves unpinned"
               for a in must if _is_enum_spec(arg_specs.get(a)) and a not in pinned]
        for arg in list(pinned):
            tool_desc = arg_specs.get(arg)
            if not isinstance(tool_desc, dict):
                out.append(f"unknown argument '{arg}' pinned by '{duty}' ({tool} has no such argument)")
                continue
            spelled, wrong = _validate_value(pinned[arg], tool_desc, arg)
            if not wrong:
                pinned[arg] = spelled
                continue
            out.append(f"invalid pin for '{duty}': {tool}.{arg} = {pinned[arg]!r} (" + "; ".join(wrong) + ")")
        return out + self._ejb_clashes(duty, tool, pinned)

    def _ejb_clashes(self, duty, tool, pinned):
        notes = []
        rival = {"order": "decline"}.get(duty, "order")
        if self.roles.get(rival) == tool:
            theirs = self._ejb_pins(rival)
            if theirs == pinned:
                notes.append(f"order and decline resolve to one identical call of {tool}")
            elif set(theirs) != set(pinned):
                notes.append(f"order/decline share {tool} while pinning unlike argument sets {sorted(pinned)} and "
                             f"{sorted(theirs)}; the distinguishing argument has to be pinned on each side")
        if not pinned:
            for rname in self.roles:
                if self.roles[rname] == tool and rname not in _CLOSING_JOBS:
                    notes.append(f"{tool} doubles as the '{rname}' tool for '{duty}' without any pinned value to "
                                 "separate the two uses")
        return notes

    def _ejb_read_jobs(self, prompt, seat_list):
        """job -> (tool, pinned values) from one model reading; unknown tools are ignored."""
        reply = self.ask_judge(prompt, 500, seat_list) or {}
        selected = ((duty, reply.get(duty)) for duty in _CLOSING_JOBS)
        return {duty: (raw_obj["tool"], dict(raw_obj["fixed"]) if isinstance(raw_obj.get("fixed"), dict) else {})
                for duty, raw_obj in selected if isinstance(raw_obj, dict) and raw_obj.get("tool") in self.param_schemas}

    def _ejb_choice_values(self, tool, pins):
        """Only the pinned values that select behaviour, normalised to the schema's spelling."""
        arg_specs = self._ejb_props(tool)
        kept = [arg for arg in pins if _is_enum_spec(arg_specs.get(arg))]
        return {arg: _validate_value(pins[arg], arg_specs[arg], arg)[0] for arg in kept}

    def _ejb_adopt_jobs(self, read_reply):
        for duty in read_reply:
            self.roles[duty], self.fixed[duty] = read_reply[duty]

    def _ejb_current_jobs(self):
        return {duty: (self.roles[duty], self._ejb_pins(duty)) for duty in _CLOSING_JOBS if duty in self.roles}

    def _ejb_same_job(self, one, two):
        if one[0] != two[0]:
            return False
        return self._ejb_choice_values(*one) == self._ejb_choice_values(*two)

    def _ejb_second_opinion(self, tool_catalog):
        """Read the closing jobs again on their own; a disagreement goes to the setup seats to decide."""
        own_row = self._ejb_current_jobs()
        other = self._ejb_read_jobs(_CLOSING_ROLES_TEXT + "\n\nTool list:\n" + tool_catalog, [PRIMARY_SEAT, FALLBACK_SEAT])
        if not other:
            return
        if any(not self._ejb_same_job(own_row[j], other[j]) for j in own_row if j in other):
            self._ejb_arbitrate(own_row, other, tool_catalog)
            return
        extra = {j: other[j] for j in other if j not in own_row}
        if extra:   # the readings agree; take what only the second one found
            self.trace.append({"kind": "finish_read", "took": extra})
            self._ejb_adopt_jobs(extra)

    def _ejb_arbitrate(self, own_row, other, tool_catalog):
        def as_text(read_reply):
            return _to_compact_json({j: {"tool": pair[0], "fixed": pair[1]} for j, pair in read_reply.items()})
        settle = "\n\n".join([_ROLE_REFEREE_TEXT, "Tool list:\n" + tool_catalog,
                              "First reading: " + as_text(own_row) + "\nSecond reading: " + as_text(other)])
        final = self._ejb_read_jobs(settle, [INTAKE_SEAT, ALT_VENDOR_SEAT, FALLBACK_SEAT])
        self.trace.append({"kind": "finish_disagree", "setup": own_row, "second": other, "final": final})
        self._ejb_adopt_jobs(final)

    def _confirm_closing_roles(self):
        """Second reading of the closing jobs, then at most two repair rounds; whatever still fails is removed."""
        tool_catalog = "\n".join(_describe_tool(need_line["function"]) for need_line in self.tools)
        self._ejb_second_opinion(tool_catalog)
        rounds = 0
        while rounds < 2:
            rounds += 1
            faults = sum((self._ejb_job_faults(j) for j in _CLOSING_JOBS), [])
            if self.key_dry or not faults:
                break
            self.trace.append({"kind": "finish_repair", "issues": faults})
            snip = self._ejb_REPAIR_HEAD + "\n".join("- " + fobj for fobj in faults) + "\n\nAvailable tools:\n" + tool_catalog
            self._ejb_adopt_jobs(self._ejb_read_jobs(snip, [INTAKE_SEAT, ALT_VENDOR_SEAT, PRIMARY_SEAT, FALLBACK_SEAT]))
        # both sides of a clash go together, so collect first
        for duty in [j for j in _CLOSING_JOBS if j in self.roles and self._ejb_job_faults(j)]:
            del self.roles[duty]
            self.fixed.pop(duty, None)

    def _classify_call(self, label_name, arguments):
        """Which closing job a call performs ('order' / 'decline'), or None."""
        dispatched = arguments or {}
        for duty in (j for j in _CLOSING_JOBS if self.roles.get(j) == label_name):
            pins = self._ejb_pins(duty)
            if not any(dispatched.get(a) != pins[a] for a in pins):
                return duty
        return None

    def _roles_for_log(self):
        """Roles for the log, plus the jobs this store cannot do."""
        logged = dict(self.roles)
        for duty in _CLOSING_JOBS:
            if duty in logged:
                logged[duty] = {"tool": logged[duty], "fixed": self._ejb_pins(duty)}
        lacking = [j for j in ("search", "details", "cart", "order", "decline") if j not in self.roles]
        if self.roles.get("message") and self._text_arguments("?") is None:
            lacking.append("message: not every required argument accepts plain text")
        logged["gaps"] = lacking
        return logged

    def _ejb_said_in_request(self, quote):
        fold = lambda snip: " ".join(str(snip or "").lower().split())
        needle = fold(quote)
        return bool(needle) and needle in fold(self.policy.get("query"))

    def _keep_priced_ask(self, element):
        """Keep a priced-asking reading only if it has a number and its quote really is in the request."""
        if isinstance(element, dict) and _as_number(element.get("amount")) is not None:
            if self._ejb_said_in_request(element.get("quote")):
                return element
        return None

    def _stated_procedures(self, entry_list):
        """Request-stated steps needing a non-closing, non-cart tool call, each with its quote checked."""
        steps = []
        for need_line in entry_list if isinstance(entry_list, list) else []:
            if not (isinstance(need_line, dict) and self._ejb_said_in_request(need_line.get("quote"))):
                continue
            rname = need_line.get("role")
            if rname not in self.roles or rname in _CLOSING_JOBS or rname == "cart":
                continue
            times, when = _as_number(need_line.get("count")), need_line.get("before")
            snip = need_line.get("do") or need_line.get("quote")
            steps.append(dict(
                do=str(snip)[:200], quote=str(need_line.get("quote"))[:200], role=rname,
                count=1 if times is None else max(1, int(times)),
                after_pick=need_line.get("after_pick") is True,
                before=when if when in ("order", "decline") else "any"))
        return steps

    def _pending_procedures(self, kind):
        """Stated steps still owed before closing with `kind`."""
        owed = []
        for step_proc in self.procedures:
            late = step_proc["after_pick"]
            if step_proc["before"] not in ("any", kind) or (late and kind != "order"):
                continue
            floor = self.cart_changed_at if late else 0
            if len([pager for pager in self.calls_by_role.get(step_proc["role"], []) if pager > floor]) < step_proc["count"]:
                owed.append(step_proc)
        return owed

    # ---------------------------------------------------------------- texts for the main model

    _ejb_NO_RANK = ("- ranking: none found (neither cheapest nor best); unless the request says otherwise, every "
                    "item within budget that satisfies all requirements is right, and comparing prices just burns steps")

    def opening_lines(self):
        """The setup reading as lines for the opening message ('' without one)."""
        plan = self.plan
        if not plan:
            return ""
        rank, detail = plan.get("priority"), plan.get("priority_detail")
        text_lines = ["Initial parse of the request - verify each line against the original wording:"]
        if plan.get("product"):
            text_lines.append(f"- product: {plan.get('product')}")
        text_lines.append(f"- quantity: {plan.get('quantity') or 1}")
        text_lines.extend(f"- stated requirement: {needed}" for needed in plan.get("requirements") or [])
        text_lines.extend(self._ejc_item_lines())
        if plan.get("price_cap") is not None:
            text_lines.append(f"- upper limit for the total price: {plan.get('price_cap')} {plan.get('currency') or ''}".rstrip())
        if plan.get("in_stock_only"):
            text_lines.append("- in-stock (on hand) listings are the only option")
        if rank in ("cheapest", "best"):
            text_lines.append(f"- ranking: pick the {rank} candidate among those meeting all conditions" + (f" ({detail})" if detail else ""))
        else:
            text_lines.append(self._ejb_NO_RANK)
        for field_name, opener in _OPENING_SECTIONS:
            text_lines.extend(opener + str(unit_ref) for unit_ref in plan.get(field_name) or [])
        if self.message_budget is not None:
            text_lines.append(f"- message allowance: {self.message_budget}")
        text_lines.append(f"- step allowance: {self.max_steps}")
        return "\n".join(text_lines + self._closing_lines())

    _ejb_PURPOSE = {"order": "buying the cart's contents", "decline": "ending with no purchase"}

    def _closing_lines(self):
        """One line per closing job: the tool and the choice values it needs."""
        out = []
        for duty in _CLOSING_JOBS:
            tool = self.roles.get(duty)
            if tool:
                choice = self._ejb_choice_values(tool, self._ejb_pins(duty))
                suffix = " with " + _to_compact_json(choice) if choice else ""
                out.append(f"- {self._ejb_PURPOSE[duty]}: {tool}{suffix}")
        return out

    _ejb_LAST = "LAST STEPS - verified fit already in the cart: order it this turn. No such fit: decline."
    _ejb_FINISH = ("FINISH NOW: one verified line only in the cart; look at its stock again; then place the order, "
                   "or decline when no line fits.")
    _ejb_FINISH_MANY = ("FINISH NOW: one verified line per requested product in the cart, each at its own quantity; "
                        "look at their stock again; then place the order, or decline when no line fits.")
    _ejb_FEW = ("Budget check: about 4 steps go into fixing a refused order, and at 7 remaining the search and "
                "detail tools are withdrawn; place the strongest verified candidate in the cart and buy it before long.")

    def status(self):
        remaining = self.remaining_steps()
        cap = "" if self.message_budget is None else f" of {self.message_budget}"
        clock = int(max(0, self.seconds_remaining() - FORCE_MARGIN))
        snip = (f"[status] steps: {self.step} used of {self.max_steps}, {remaining} left; messages sent: "
                f"{self.messages_used}{cap}; time left: about {clock} s.\n[requirements now] " + self.requirement_summary())
        tiers = ((lambda: remaining <= 2, self._ejb_LAST),
                 (lambda: remaining <= 5 or self.seconds_remaining() < WRAP_UP_WINDOW or self.cost > _eja_SPEND_WRAP,
                  self._ejb_FINISH_MANY if self._wanted_items() else self._ejb_FINISH),
                 (lambda: self.order_attempts == 0 and remaining <= 14, self._ejb_FEW))
        return snip + next((" " + note for test, note in tiers if test()), "")

    def requirement_summary(self):
        """The requirements in force as one text, with notes on values the store words like the request."""
        self._pin_stated_values()
        many = self._wanted_items()
        text_lines = [self._need_list.render_lines(self.plan.get("currency") or "",
                                                   "per product as listed below" if many else self._wanted_units())]
        text_lines.extend(self._ejc_item_lines())
        for clause in self._list_conditions():
            phrasing = self._shop_phrasing(clause)
            if phrasing:
                quoted = " or ".join(f'"{term_word}"' for term_word in phrasing)
                text_lines.append(f"- {self._ejb_cut(clause)[0].strip()}: the request's own phrase {quoted} is how this store "
                             f"spells the value; a listing qualifies only by stating it, and numbers derived from "
                             f"differently worded values do not count")
        for prop_name, wish in self.loose_ok.items():
            text_lines.append(f'- {prop_name}: asked which of the listed values holding "{wish}" was meant, the shopper '
                              f'did not mind; a listing whose {prop_name} holds the words "{wish}" meets it, '
                              f'whatever else the value adds')
        return "\n".join(text_lines)

    @staticmethod
    def _ejb_property_values(row_map, need_terms):
        """Values of every fact whose key names the property (all words held, or named like it)."""
        for listing_row in row_map:
            for field_name, datum in (listing_row.get("facts") or {}).items():
                if need_terms <= _word_set(field_name) or _key_names_property(field_name, need_terms):
                    yield datum

    def _shop_phrasing(self, condition, row_map=None):
        """For a "covers N" condition: store values for that property holding N that the request repeats word for
        word. Such a phrase is the value meant; a number taken from other wording does not satisfy it."""
        pname, sep, wanted = self._ejb_cut(condition)
        rel = _parse_relation(wanted) if sep else None
        need_terms = _word_set(pname)
        if not need_terms or rel is None or rel[0] != "covers":
            return []
        target, request = rel[1], _flatten_text(self.policy.get("query"))
        hits = []
        row_index = self._row_table() if row_map is None else row_map
        for datum in self._ejb_property_values(row_index.values(), need_terms):
            for snip in map(str, datum if isinstance(datum, list) else [datum]):
                for piece in [snip, *re.split(r"[,;|]", snip)]:
                    flat = _flatten_text(piece)
                    if not flat or flat in hits or flat not in request:
                        continue
                    if target in [float(cnt) for cnt in re.findall(r"\d+(?:\.\d+)?", flat)]:
                        hits.append(flat)
        return hits

    # ---------------------------------------------------------------- JSON judges and shopper lines

    def ask_judge(self, prompt, max_tokens=700, models=None):
        """A JSON-text call with one user message. When the reply was cut before the object closed, ask again with
        three times the room, and if that is cut too, move on to the seats after the one that answered."""
        seat_list = models or [REVIEW_SEAT, ALT_VENDOR_SEAT, PRIMARY_SEAT]
        ask = lambda who, room: self.infer([{"role": "user", "content": prompt}], None, who, room)
        read = lambda raw_obj: None if raw_obj is None else _extract_json_dict(raw_obj.get("content") or "")
        first = ask(seat_list, max_tokens)
        if first is None:
            return None
        parsed = read(first)
        if parsed is not None or not first.get("cut"):
            return parsed
        roomy = max_tokens * 3
        self.trace.append({"kind": "cut_reply", "limit": max_tokens})
        second = ask(seat_list, roomy)
        parsed = read(second)
        if parsed is not None or second is None or not second.get("cut"):
            return parsed
        who = second.get("model")
        rest = seat_list[seat_list.index(who) + 1:] if who in seat_list else []
        if not rest:
            return None
        self.trace.append({"kind": "cut_reply", "limit": roomy, "next": rest[0]})
        return read(ask(rest, roomy))

    def _ejb_reading_prompt(self, new_lines, question):
        past = self._need_list.history
        sections = [("Background - the request as first written", str(self.policy.get("query") or "")),
                    ("Standing requirements up to now", self.requirement_summary()),
                    ("Conversation so far, earliest first", "\n".join(past[-12:]) if past else None),
                    ("Assistant's question", question or None),
                    ("The shopper's new words", "\n".join(new_lines))]
        return "\n\n".join([UPDATE_READING_TEXT] + [f"{title}:\n{body}" for title, body in sections if body is not None])

    def read_shopper_lines(self, text_lines, question):
        """Fold the meaning of new shopper lines into the wants; returns a note for the model ('' if none)."""
        need_state = self._need_list
        new_lines = [ln for ln in text_lines if ln not in need_state.heard]
        if not new_lines:
            return ""
        need_state.heard.update(new_lines)
        if self.seconds_remaining() < WRAP_UP_WINDOW:
            return ""
        prompt = self._ejb_reading_prompt(new_lines, question)
        need_state.history.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(new_lines))
        read_reply = self.ask_judge(prompt, 1500, [READER_SEAT, REVIEW_SEAT])
        if not read_reply:
            return ""
        if question:
            self.answer_reading = self._ejb_checked_answer(read_reply.get("answer_class"), " ".join(new_lines))
            if not self.asking:   # a general question frees no property
                read_reply["no_preference"] = []
        chain = (self._ejb_refile_needs, self._ejb_protect_stated if question else None,
                 self._ejb_restore_stated, self._ejb_drop_invented)
        for screen in chain:
            if screen is not None:
                screen(read_reply, new_lines)
        known = list(need_state.revealed)
        changes = self._ejb_fold(read_reply)
        self._ejb_home_needs([ent for ent in need_state.revealed if ent not in known], new_lines)
        reply = str(read_reply.get("reply") or "").strip()
        if reply and not question and self.roles.get("message"):   # our own question needs no answer back
            self.queued_reply = reply
        tail = " (" + "; ".join(changes) + ")" if changes else ""
        return "Meaning: " + str(read_reply.get("meaning") or "") + tail

    def _ejb_home_needs(self, new_needs, text_lines):
        """Several products requested: file each newly revealed need under the one product the shopper's words
        name, else under the product the asked property belongs to; left unfiled when neither tells."""
        items = self._wanted_items()
        if not items or not new_needs:
            return
        products = [el["product"] for el in items]
        owner = _ejd_owner(self, self.asking) if self.asking else ""
        for ent in new_needs:
            said = self._ejc_stems(" ".join(list(text_lines) + [ent]))
            named = [seg for seg in products if self._ejc_stems(" ".join(_ejd_own_words(seg, products))) & said]
            home = named[0] if len(named) == 1 else owner
            if home in products:
                self.need_homes[ent] = home

    def _ejb_fold(self, read_reply):
        need_state = self._need_list
        dropped_before, budget_before = len(need_state.dropped), need_state.budget
        changes = need_state.fold_reading(read_reply)
        if changes:
            self.needs_revision += 1
            moved = (len(need_state.dropped), need_state.budget) != (dropped_before, budget_before)
            if moved and self.plan.get("priority") == "cheapest":
                self.rerank_reason = "; ".join(need_state.dropped[dropped_before:]) or "the budget"
        return changes

    _ejb_CLASSES = ("reveal", "not_picky", "split", "unclear", "other")

    def _ejb_checked_answer(self, raw_obj, said):
        """The answer class, trusted only when its quote is in the answer and a reveal names the asked property, a
        listing property, or a listing value; otherwise 'other'."""
        if not isinstance(raw_obj, dict) or raw_obj.get("class") not in self._ejb_CLASSES:
            return {"class": "other"}
        field_specs = {label_name: str(raw_obj.get(label_name) or "") for label_name in ("key", "value", "relation")}
        quote = str(raw_obj.get("quote") or "").strip().lower()
        trusted = bool(quote) and quote in said.lower()
        if trusted and raw_obj["class"] == "reveal":
            trusted = self._ejb_on_topic(field_specs["key"], field_specs["value"].strip().lower())
        return {"class": raw_obj["class"], **field_specs} if trusted else {"class": "other"}

    def _ejb_on_topic(self, key_text, datum):
        tag_names, values = self._ejb_sheet_words()
        field_name = _word_set(key_text)
        if field_name and (field_name <= _word_set(self.asking) or any(field_name <= tag for tag in tag_names)):
            return True
        return bool(datum) and datum in values

    def _ejb_doubtful(self, have, tag_names):
        """A need's property is open to doubt: the asked one, one waved off, none the listings carry, or a stated one."""
        asked = _word_set(self.asking)
        if asked and (have <= asked or asked <= have):
            return True
        if have in [fobj for fobj in (_word_set(el) for el in self._need_list.free or []) if fobj]:
            return True
        given_reqs = [self._ejb_key_words(need_entry) for need_entry in self._need_list.stated or []]
        return have not in tag_names or have in given_reqs

    @staticmethod
    def _ejb_line_home(ln, pname, datum, tag_names):
        """The listing property a shopper line names next to the value, spelled with the line's words."""
        said = _word_set(ln)
        if _word_set(pname) <= said or not _words_cover(datum, said):
            return None
        found_keys = [term_word for term_word in tag_names if term_word <= said and not term_word & datum]
        if not found_keys:
            return None
        widest = max(found_keys, key=len)
        return " ".join(term_word for term_word in re.findall(r"[a-z0-9]+", ln.lower()) if term_word in widest)

    def _ejb_refiled(self, need_line, text_lines, tag_names):
        if not (isinstance(need_line, str) and ":" in need_line):
            return None
        pname, constraint = need_line.split(":", 1)
        datum = _word_set(constraint) - _word_set(pname)
        if not datum or not self._ejb_doubtful(_word_set(pname), tag_names):
            return None
        homes = (self._ejb_line_home(ln, pname, datum, tag_names) for ln in text_lines)
        return next((home + ":" + constraint for home in homes if home is not None), None)

    def _ejb_refile_needs(self, read_reply, text_lines):
        """Move a new need to the listing property the shopper's own line names, when the reader's property is
        doubtful and the line carries the value."""
        tag_names = [term_word for term_word in self._ejb_sheet_words()[0] if term_word]
        if not tag_names or not isinstance(read_reply, dict):
            return
        moved = {}
        for need_line in read_reply.get("needs") or []:
            target = self._ejb_refiled(need_line, text_lines, tag_names)
            if target is not None:
                moved[need_line] = target
        if not moved:
            return
        swap = lambda ent: moved.get(ent, ent)
        read_reply["needs"] = [swap(ent) for ent in read_reply.get("needs") or []]
        now = read_reply.get("requirements_now")
        if isinstance(now, list):
            read_reply["requirements_now"] = [swap(ent) if isinstance(ent, str) else ent for ent in now]
        self.trace.append({"kind": "attributed", "moved": moved})

    def _ejb_protect_stated(self, read_reply, _lines=None):
        """An answer to our question may add needs but never drops or frees what the request itself stated."""
        if not isinstance(read_reply, dict):
            return
        given_reqs = [str(ent) for ent in self._need_list.stated]
        arg_specs = [self._ejb_key_words(ent) for ent in given_reqs]

        def touches(snip):
            if any(pager in snip or snip in pager for pager in given_reqs if pager):
                return True
            label_name = self._ejb_key_words(snip)
            return bool(label_name) and label_name in arg_specs

        guarded = [str(dd) for dd in read_reply.get("dropped") or [] if touches(str(dd))]
        if guarded:
            read_reply["dropped"] = [dd for dd in read_reply.get("dropped") or [] if str(dd) not in guarded]
            self.trace.append({"kind": "stated_guarded", "entries": guarded})
        free = read_reply.get("no_preference")
        if isinstance(free, list):
            overlaps = lambda term_set: any(prop_terms and (prop_terms <= term_set or term_set <= prop_terms) for prop_terms in arg_specs)
            read_reply["no_preference"] = [fobj for fobj in free if not (_word_set(fobj) and overlaps(_word_set(fobj)))]
        self._ejb_keep_revealed(read_reply)

    def _ejb_keep_revealed(self, read_reply):
        """A need the shopper revealed earlier is replaced only by a new value for it: an answer that just waves the
        property off neither drops nor frees it, and the rebuilt list keeps it."""
        need_state = self._need_list
        fresh = [self._ejb_key_words(ent) for ent in read_reply.get("needs") or [] if isinstance(ent, str)]
        held = [ent for ent in need_state.revealed if ent not in need_state.dropped and self._ejb_key_words(ent)
                and self._ejb_key_words(ent) not in fresh]
        if not held:
            return
        homes = [self._ejb_key_words(ent) for ent in held]
        near = lambda term_set: bool(term_set) and any(term_set <= home or home <= term_set for home in homes)
        lost = [str(dd) for dd in read_reply.get("dropped") or [] if str(dd) in held or near(self._ejb_key_words(dd))]
        if lost:
            read_reply["dropped"] = [dd for dd in read_reply.get("dropped") or [] if str(dd) not in lost]
        free = read_reply.get("no_preference")
        if isinstance(free, list):
            read_reply["no_preference"] = [fobj for fobj in free if not near(_word_set(fobj))]
        relist = self._ejb_rebuilt(read_reply)
        if relist is not None:
            have = [self._ejb_key_words(ent) for ent in relist if isinstance(ent, str)]
            relist.extend(ent for ent in held if self._ejb_key_words(ent) not in have)
        if lost:
            self.trace.append({"kind": "revealed_guarded", "entries": lost})

    @staticmethod
    def _ejb_rebuilt(read_reply):
        relist = read_reply.get("requirements_now") if isinstance(read_reply, dict) else None
        return relist if isinstance(relist, list) else None

    @staticmethod
    def _ejb_spoken(text_lines):
        spoken = set()
        for ln in text_lines or []:
            spoken |= _word_set(ln)
        return spoken

    def _ejb_restore_stated(self, read_reply, text_lines):
        """Put back a stated requirement the rebuilt list lost, unless it was dropped or the line spoke of it."""
        relist = self._ejb_rebuilt(read_reply)
        if relist is None:
            return
        spoken = self._ejb_spoken(text_lines)
        gone = [str(dd) for dd in read_reply.get("dropped") or []] + list(self._need_list.dropped)
        lifted = lambda need_line: any(need_line in dd or dd in need_line for dd in gone if dd)
        for need_line in self._need_list.stated:
            pname = self._ejb_key_words(need_line)
            if not pname or need_line in relist or lifted(need_line) or _words_cover(pname, spoken):
                continue   # missing property, still present, dropped, or spoken about by the shopper
            relist[:] = [need_entry for need_entry in relist if not isinstance(need_entry, str) or self._ejb_key_words(need_entry) != pname] + [need_line]
            self.trace.append({"kind": "kept_stated", "entry": need_line})

    def _ejb_drop_invented(self, read_reply, text_lines):
        """Remove rebuilt entries on a brand-new property that neither the line nor the reading's needs mention."""
        relist = self._ejb_rebuilt(read_reply)
        if relist is None:
            return
        need_state = self._need_list
        known = need_state.stated + need_state.revealed + list(need_state.newest_lines or []) + list(read_reply.get("needs") or [])
        arg_specs = [prop_terms for prop_terms in map(self._ejb_key_words, known) if prop_terms]
        spoken = self._ejb_spoken(text_lines)

        def invented(need_line):
            if not (isinstance(need_line, str) and ":" in need_line):
                return False
            pname = self._ejb_key_words(need_line)
            if not pname or any(pname & prop_terms for prop_terms in arg_specs):
                return False
            return not any(_words_match(term_word, pager) for term_word in _word_set(need_line) for pager in spoken)

        made_up = [need_line for need_line in relist if invented(need_line)]
        if made_up:
            relist[:] = [need_entry for need_entry in relist if need_entry not in made_up]
            self.trace.append({"kind": "unsaid_dropped", "entries": made_up})

    # ---------------------------------------------------------------- stated values tied to the store's words

    @staticmethod
    def _ejb_rewrite_wants(need_state, old_matches, new):
        for bucket in (need_state.stated, need_state.revealed, need_state.newest_lines or []):
            bucket[:] = [new if isinstance(need_entry, str) and old_matches(need_entry) else need_entry for need_entry in bucket]

    def _ejb_exclusions(self):
        """Property words -> the values the stated requirements exclude for it."""
        barred = {}
        for need_line in self._need_list.stated:
            pname, sep, wanted = self._ejb_cut(need_line)
            if sep and "!=" in wanted:
                slot_name = frozenset(_word_set(pname))
                barred[slot_name] = barred.get(slot_name, set()) | _word_set(wanted.partition("!=")[2])
        return barred

    def _pin_stated_values(self):
        """Tie a stated value to a store value. When none of the opened listings' values for the property is in the
        constraint, yet the request uses exactly one of them (outside an exclusion), the requirement becomes
        "property: value (old constraint)". The value must share a word with the constraint, unless the constraint
        both words the value differently and excludes another one."""
        if not self.records or not isinstance(self.policy, dict):
            return
        opened = [listing_row for listing_row in self._row_table().values() if listing_row["opened"]]
        if not opened:
            return
        self._ejb_rekey(opened)
        request = _word_set(self.policy.get("query"))
        barred = self._ejb_exclusions()
        need_state = self._need_list
        for need_line in list(need_state.stated):
            pname, sep, wanted = self._ejb_cut(need_line)
            key_words = _word_set(pname)
            plain = sep and key_words and "(" not in wanted and _parse_relation(wanted) is None
            if not plain:
                continue   # numbers keep their relation check; explained values are already tied
            positive, _, excluded = wanted.partition("!=")
            pick = self._ejb_store_pick(opened, key_words, positive, excluded, request,
                                        barred.get(frozenset(key_words), set()))
            if pick is None:
                continue
            anchored = f"{pname.strip()}: {pick} ({wanted.strip()})"
            self._ejb_rewrite_wants(need_state, lambda need_entry, old=need_line: need_entry.startswith(old), anchored)
            self.needs_revision += 1
            self.trace.append({"kind": "anchored", "from": need_line, "to": anchored})

    def _ejb_store_pick(self, opened, need_terms, positive, excluded, request, barred):
        member_list = set()
        for datum in self._ejb_property_values(opened, need_terms):
            member_list.update(prop_terms.strip().lower() for prop_terms in re.split(r"[,/;|]", str(datum)) if _word_set(prop_terms))
        want_words = _word_set(positive)
        if not member_list or any(_word_set(mt) <= want_words for mt in member_list):
            return None
        off = _word_set(excluded) | barred
        reworded = bool(excluded.strip() and want_words)
        picks = []
        for member in member_list:
            term_set = _word_set(member)
            if term_set & off or not all(any(_words_match(term_word, q) for q in request) for term_word in term_set):
                continue
            if want_words and _words_cover(want_words, term_set):
                continue   # it already states every wanted word: pinning it would only refuse its siblings
            if reworded or any(_words_match(term_word, prop_terms) for term_word in term_set for prop_terms in want_words):
                picks.append(member)
        return picks[0] if len(picks) == 1 else None

    def _ejb_rekey(self, opened):
        """Re-file a stated requirement under the property its own request sentence names, when listings of that
        property state exactly the wanted value and the sentence never names the filed property."""
        need_state = self._need_list
        sentences = [_word_set(pager) for pager in re.split(r"(?<=[.!?])\s+", str(self.policy.get("query") or ""))]
        for need_line in list(need_state.stated):
            pname, sep, wanted = self._ejb_cut(need_line)
            key_words = _word_set(pname)
            datum = _word_set(wanted.partition("!=")[0].partition("(")[0])
            if not (sep and key_words and datum) or _parse_relation(wanted) is not None:
                continue
            homes = [term_set for term_set in sentences if _words_cover(datum, term_set)]
            if not homes or any(_words_cover(key_words, term_set) for term_set in homes):
                continue
            found_keys = self._ejb_named_keys(opened, key_words, datum, homes)
            if not found_keys:
                continue
            shortest = min(found_keys, key=len)
            if not all(shortest <= other for other in found_keys):
                continue   # two different properties carry it: ambiguous
            moved = f"{found_keys[shortest]}: {wanted.strip()}"
            self._ejb_rewrite_wants(need_state, lambda need_entry, old=need_line: need_entry == old, moved)
            self.needs_revision += 1
            self.trace.append({"kind": "rekeyed", "from": need_line, "to": moved})

    def _ejb_named_keys(self, opened, need_terms, datum, homes):
        """Other property keys stating the value whose words the request sentence uses, shortest spelling each."""
        found_keys = {}
        for listing_row in opened:
            for field_name, snip in listing_row["facts"].items():
                key_words = _word_set(field_name)
                if not key_words or key_words & need_terms:
                    continue   # the same property in another spelling
                if not any(_words_cover(key_words, term_set) for term_set in homes):
                    continue
                if datum not in [_word_set(part) for part in re.split(r"[,/;|]", str(snip))]:
                    continue
                plain = " ".join(self._ejb_label(field_name).split()).lower()
                slot_name = frozenset(key_words)
                if slot_name not in found_keys or len(plain) < len(found_keys[slot_name]):
                    found_keys[slot_name] = plain
        return found_keys

    def _ejb_sheet_words(self):
        """Property-name words and short values found on detail sheets (dicts with five or more short texts)."""
        tag_names = []
        values = set()
        for record_blob in _iter_dicts(self.records):
            texts = self._ejb_short_texts(record_blob, 70)
            if len(texts) < 5:
                continue
            values.update(tk.strip().lower() for _, tk in texts)
            for term_set in (_word_set(self._ejb_label(label_key)) for label_key, _ in texts):
                if term_set and term_set not in tag_names:
                    tag_names.append(term_set)
        return tag_names, values

    @staticmethod
    def _ejb_mentions(record_blob, unit_ref, nest_level):
        """Dicts holding `item` as a value, depth first; the search stops inside a match and below six levels."""
        if nest_level > 6:
            return
        if isinstance(record_blob, dict):
            if any(isinstance(cell, str) and cell == unit_ref for cell in record_blob.values()):
                yield record_blob
                return
            kids = list(record_blob.values())
        elif isinstance(record_blob, list):
            kids = record_blob
        else:
            return
        for kid in kids:
            yield from Episode._ejb_mentions(kid, unit_ref, nest_level + 1)

    def records_naming(self, unit_ref):
        """The last (up to four) distinct records that mention this id."""
        found = []
        for record in self.records:
            for record_blob in self._ejb_mentions(record, unit_ref, 0):
                snip = _to_compact_json(_shorten_value(record_blob, 400))[:2500]
                if snip not in found:
                    found.append(snip)
                if len(found) >= 8:
                    return found[-4:]
        return found[-4:]

    # ---------------------------------------------------------------- code-side checks and summaries of the store

    def _verify_item(self, unit_ref, intent):
        """A second-opinion judge on one listing: a warning text, or '' when it passes or cannot be judged."""
        if self.seconds_remaining() < WRAP_UP_WINDOW + 20 or self.spend_exceeded():
            return ""
        memo = f"{unit_ref}|{self._arm_signature()}"
        cached = self.verdicts.get(memo)
        if cached is None:
            evidence = self.records_naming(unit_ref)
            if evidence:
                cached = self.verdicts[memo] = self._ejb_verdict(unit_ref, intent, evidence)
        return cached or ""

    def _ejb_verdict(self, unit_ref, intent, evidence):
        in_force = self.requirement_summary()
        cart = None if self.last_cart_view is None else _to_compact_json(self.last_cart_view)[:3000]
        silent = self._unspoken_conditions()
        recurring = self._repeated_values()
        blocks = [ITEM_CHECK_TEXT,
                  "Original wording of the request:\n" + str(self.policy.get("query") or ""),
                  "Current requirements, combining the request with later answers and changes:\n" + in_force,
                  f"Listing under review ({intent}): {unit_ref}" + self._ejc_item_scope(unit_ref),
                  "Every store record naming this listing, in order received:\n" + "\n".join(evidence)]
        if cart is not None:
            blocks.append("Latest cart view:\n" + cart)
        if silent:
            blocks.append("Conditions with no trace in any opened listing here (neither a key nor any text covers "
                          "them, hence no listing is able to prove them): " + "; ".join(silent) + ". Do not reject "
                          "the listing over these by themselves; decide on the remaining requirements.")
        if recurring:
            blocks.append("Labels that recur over several listings, with counts. For this listing use the label equal "
                          "to the full request wording, never an adjacent one:\n" + recurring)
        result_info = self.ask_judge("\n\n".join(blocks), 1200)
        self.trace.append({"kind": "verify", "item": unit_ref, "result": result_info})
        if not (result_info and result_info.get("fits") is False):
            return ""
        gaps = [str(prop_terms) for prop_terms in result_info.get("problems") or []][:5]
        own = self._row_table().get(unit_ref) or {}
        every = [str(prop_terms) for prop_terms in result_info.get("problems") or []]
        if own.get("opened") and self._row_passes_literal(own) is True \
                and _ejc_qualified_only(own["facts"], self._list_conditions(), every):
            self.trace.append({"kind": "verify_kept", "item": unit_ref})   # a qualified value of its own key
            return ""
        return (f"Second check on {unit_ref} failed: " + "; ".join(gaps)
                + ". Choose a different, fully verified listing instead, except when you are sure this check got it wrong.")

    def _ejb_allowed_jobs(self):
        """The jobs whose tools stay on offer as steps run out (None: no narrowing)."""
        remaining = self.remaining_steps()
        if remaining <= 1:
            return ("order", "decline")
        if remaining <= 3:
            return ("order", "decline", "cart")
        if remaining > 7 or (self.order_attempts and not self.refusals):
            return None
        late = ("order", "decline", "cart", "cart_view", "view")
        # after a refusal, a cheaper listing seen but never opened keeps the details tool
        if self.refusals:
            return late + ("details",) if _rows_to_recheck(self) else late
        # no order tried and a product without a fit while on-target rows wait unopened: details down to reserve + 2
        if remaining >= self._reserved_steps() + 2 and _eje_unfitted_rows(self, 1):
            return late + ("details",)
        return late

    def _tools_this_turn(self):
        """The tools offered this turn; the message tool is hidden while the planner sends the questions."""
        jobs = self._ejb_allowed_jobs()
        tool_of = self.roles.get
        offered = self.tools
        if jobs is not None and tool_of("order") and tool_of("cart"):
            names = set(map(tool_of, jobs))
            offered = [tk for tk in offered if tk["function"]["name"] in names] or self.tools
        muted = tool_of("message") if self.planner_on else None
        return [tk for tk in offered if tk["function"]["name"] != muted] if muted else offered

    def _ejb_menu_sheets(self):
        """Detail sheets as label -> lower-case value (search rows and fields shown in two record forms left out)."""
        node_list = list(_iter_dicts(self.records))
        ids = self.seen_ids or set()
        dual = _own_listing_fields(node_list, ids)
        row_ids = _search_row_ids(self.records, ids)
        owned = set()   # a listing's own nested property dict counts even when short
        above = {}   # a nested dict -> the record holding it
        for record_blob in node_list:
            above.update((id(child), record_blob) for child in record_blob.values() if isinstance(child, dict))
            if any(_looks_like_listing(el, ids) for el in record_blob.values()):
                owned.update(id(child) for child in record_blob.values() if isinstance(child, dict))
        sheets = []
        self._ejb_sheet_texts = []   # per sheet: the lower-case texts of its record and of the record above it
        for record_blob in node_list:
            if id(record_blob) in row_ids:
                continue
            owner = next((cell for cell in record_blob.values() if _looks_like_listing(cell, ids)), None)
            texts = [(label_key, cell) for label_key, cell in self._ejb_short_texts(record_blob, 70) if (owner, label_key, cell) not in dual]
            if len(texts) < (3 if id(record_blob) in owned else 5):
                continue
            fact_sheet = {}
            for field_name, snip in texts:
                tag = self._ejb_label(field_name).strip().lower()
                if tag not in fact_sheet and not _looks_like_code(snip, True):
                    fact_sheet[tag] = snip.lower()
            sheets.append(fact_sheet)
            around = [record_blob, above.get(id(record_blob)) or {}]
            self._ejb_sheet_texts.append({str(cell).strip().lower() for blob in around for cell in blob.values()
                                          if isinstance(cell, str)})
        return sheets

    def _ejb_on_target_sheets(self, sheets):
        """Detail sheets of the requested products' own sections: a sheet naming another section (a reference page,
        a stray search) leaves the question menu, even when no sheet of the wanted sections is in yet: its
        properties belong to another product. Every sheet stays while some requested product shares no word stem
        with any section seen."""
        products = [el["product"] for el in self._wanted_items()] or [(self.plan or {}).get("product") or ""]
        labels = {rw["label"].strip().lower() for rw in self._row_table().values() if isinstance(rw.get("label"), str)}
        homes = set()
        for product in products:
            stems = self._ejc_stems(product)
            hit = {lb for lb in labels if stems & self._ejc_stems(lb)}
            if not hit:
                return sheets
            homes |= hit
        words = getattr(self, "_ejb_sheet_texts", None)
        if not isinstance(words, list) or len(words) != len(sheets):
            words = [set(page.values()) for page in sheets]
        named = [said & labels for said in words]
        keep = [page for page, said in zip(sheets, named) if not said or said & homes]
        if len(keep) == len(sheets):
            return sheets
        return keep

    @staticmethod
    def _ejb_spellings(sheets, freq):
        """label -> the label it is another spelling of: shared stem and equal values wherever both appear."""
        rank = lambda label_key: (-freq[label_key], label_key)
        home = {}
        roots = set()
        for tag in sorted(freq, key=rank):
            def same(main):
                both = [page for page in sheets if tag in page and main in page]
                return bool(both) and bool(_name_stems(tag) & _name_stems(main)) and all(page[tag] == page[main] for page in both)
            match = next((main for main in sorted(roots, key=rank) if same(main)), None)
            if match is None:
                roots.add(tag)
                match = tag
            home[tag] = match
        return home

    def _build_property_menu(self):
        """The property menu from listing details (counts, samples, other spellings) and the number of sheets.
        Sets self.menu_info for the planner."""
        sheets = self._ejb_on_target_sheets(self._ejb_menu_sheets())
        seen = {}
        for fact_sheet in sheets:
            for tag, low in fact_sheet.items():
                seen.setdefault(tag, []).append(low)
        freq = {tag: len(vals) for tag, vals in seen.items()}
        distinct = {tag: list(dict.fromkeys(vals)) for tag, vals in seen.items()}
        home = self._ejb_spellings(sheets, freq)
        aliases = {}
        for tag in home:
            if home[tag] != tag:
                aliases.setdefault(home[tag], []).append(tag)
        product = _word_set((self.plan or {}).get("product"))

        def item_named(label_key: str) -> bool:   # mostly the item's own name in its values
            hits = sum(1 for cell in seen[label_key] if product & _word_set(cell))
            return bool(product) and 2 * hits > len(seen[label_key])

        uniform = {label_key for label_key in freq if freq[label_key] >= 3 and len(distinct[label_key]) == 1}
        top = sorted((label_key for label_key in freq if home[label_key] == label_key and not item_named(label_key)),
                     key=lambda label_key: (-freq[label_key], label_key in uniform, label_key))[:28]
        self.property_menu = {label_key: {"count": freq[label_key], "values": sorted(distinct[label_key]), "uniform": label_key in uniform,
                              "also": sorted(aliases.get(label_key, []))} for label_key in top}
        text_lines = []
        for label_key in top:
            same = " (identical on all listings)" if label_key in uniform else ""
            also = " [other spellings: " + ", ".join(sorted(aliases[label_key])) + "]" if aliases.get(label_key) else ""
            text_lines.append(f"* {label_key}: {freq[label_key]} listings{same}{also}; sample values " + " / ".join(distinct[label_key][:5]))
        return "\n".join(text_lines), len(sheets)

    def _repeated_values(self):
        """Short values repeating under one key across listing rows, with counts (at most twelve keys)."""
        per_key = {}
        for record_blob in (cnt for cnt in _iter_dicts(self.records) if len(cnt) >= 4):
            for field_name, snip in self._ejb_short_texts(record_blob, 40):
                if _looks_like_code(snip, False):
                    continue
                bucket = per_key.setdefault(str(field_name), {})
                bucket[snip] = 1 + bucket.get(snip, 0)

        def worth(counts):
            return 2 <= len(counts) <= 8 and max(counts.values()) >= 2 and sum(counts.values()) >= 4

        shown = [f"* {field_name}: " + ", ".join(f"{cell} ({cnt}x)" for cell, cnt in sorted(ch.items(), key=lambda pair: -pair[1]))
                 for field_name, ch in per_key.items() if worth(ch)]
        return "\n".join(shown[:12])

    def _call_shape(self, label_name, arguments):
        """Shape of a call: tool, argument names, choice values."""
        return "|".join((label_name, ",".join(sorted(arguments)), self._choice_signature(label_name, arguments)))

    def _detect_error(self, label_name, seen):
        """(text, sure) for an error result, else ('', False). An "error" field is a sure fault; a single text field
        on a non-message call is only a possible one (it counts when the identical call gets it again)."""
        nothing = ("", False)
        if not isinstance(seen, dict) or not seen:
            return nothing
        if "error" in seen and "refusal" not in seen:
            return _to_compact_json(dict(list(seen.items())[:2]))[:200], True
        # a yes/no field that read true in this tool's earlier replies and reads false now: the call did not go
        # through, whatever the reply's other keys are called
        held_true = self.__dict__.setdefault("_ejf_true_flags", {}).setdefault(label_name, set())
        # only short status replies (no rows, no listing ids): a listing's own yes/no facts differ between listings
        blobs = [blob for blob in [seen] + [cell for cell in seen.values() if isinstance(cell, dict)]
                 if len(blob) <= 3 and not any(isinstance(cell, list) or cell in self.seen_ids
                                               for cell in blob.values() if not isinstance(cell, dict))]
        flipped = [key for blob in blobs for key, cell in blob.items() if cell is False and key in held_true]
        if flipped and "refusal" not in seen and label_name != self.roles.get("message"):
            return _to_compact_json(dict(list(seen.items())[:2]))[:200], True
        held_true.update(key for blob in blobs for key, cell in blob.items() if cell is True)
        if len(seen) > 1 or label_name == self.roles.get("message"):
            return nothing
        field_name, datum = next(iter(seen.items()))
        lone_text = isinstance(datum, str) and datum.strip() and datum not in self.seen_ids
        return (f"{field_name}: {datum}"[:200], False) if lone_text else nothing

    @staticmethod
    def _ejb_has_rows(record_blob):
        """A non-empty list reachable through dicts."""
        if isinstance(record_blob, list):
            return bool(record_blob)
        if isinstance(record_blob, dict):
            return any(Episode._ejb_has_rows(cell) for cell in record_blob.values())
        return False

    def _no_hits_note(self, label_name, arguments, nested):
        """For a search that returned nothing while choice arguments were left out (and no words were typed): a
        hint that the mode may need one of them."""
        ids = set()
        _gather_ids(nested, ids)
        if ids or self._ejb_has_rows(nested):
            return ""
        arg_specs = self._ejb_props(label_name)
        for arg, given in arguments.items():
            if isinstance(given, str) and given.strip() and not _is_enum_spec(arg_specs.get(arg)):
                return ""   # words were searched: an ordinary miss
        choices = {label_key: pager["enum"] for label_key, pager in arg_specs.items() if label_key not in arguments and isinstance(pager, dict) and pager.get("enum")}
        if not choices:
            return ""
        unset = ["%s (e.g. %s)" % (label_key, ", ".join(map(str, vals[:4]))) for label_key, vals in choices.items()]
        return ("Zero results. Options this call omitted: " + "; ".join(unset[:3]) + ". Some search modes return "
                "nothing until one of them is given, so this emptiness says nothing about whether the store carries the item.")

    def _requested_units(self, label_name, arguments):
        """Units a cart call requests: its free integer argument (or its default), otherwise 1."""
        arg_specs = self._ejb_props(label_name, True)
        counter = next((label_key for label_key, pager in arg_specs.items()
                        if isinstance(pager, dict) and pager.get("type") == "integer" and "enum" not in pager), None)
        if counter is None:
            return 1
        units = arguments[counter] if counter in arguments else arg_specs[counter].get("default", 1)
        return units if isinstance(units, int) and not isinstance(units, bool) else 1

    def _parse_cart_units(self, seen):
        """Units per held listing from a cart view; the units field is learned from the first line whose single
        matching integer equals the units we sent."""
        self.cart_seen_at = self.step
        text_lines = []
        for record_blob in _iter_dicts([seen]):
            held = next((cell for cell in record_blob.values() if isinstance(cell, str) and cell in self.sent_units), None)
            if held is not None:
                text_lines.append((held, self._ejb_ints(record_blob)))
        counts = {}
        for held, ints in text_lines:
            if self.quantity_field is None:
                matches = [label_key for label_key in ints if ints[label_key] == self.sent_units[held]]
                if len(matches) == 1:
                    self.quantity_field = matches[0]
            if self.quantity_field in ints:
                counts[held] = ints[self.quantity_field]
        if self.quantity_field is not None:
            self.units_in_cart = counts

    def _choice_signature(self, label_name, arguments):
        """The enum-valued arguments of a call as text (it tells an add from a remove)."""
        arg_specs = self._ejb_props(label_name, True)
        chosen = {}
        for arg in arguments:
            tool_desc = arg_specs.get(arg)
            if isinstance(tool_desc, dict) and "enum" in tool_desc:
                chosen[arg] = arguments[arg]
        return _to_compact_json(chosen)

    # ==== shopper messages =================================================

    def _message_limit(self):
        """Messages the shopper will take; three unless the request gives a number."""
        limit = self.message_budget
        return 3 if limit is None else limit

    def _text_arguments(self, text):
        """Arguments for the message tool carrying only `text`; None when the tool also requires something else."""
        tool_spec = self.param_schemas.get(self.roles.get("message") or "") or {}
        must = list(tool_spec.get("required") or [])
        writable = _ejc_text_fields(tool_spec.get("properties") or {})
        slot_name = next((label_name for label_name in must if label_name in writable), None)
        if not slot_name and len(writable) == 1:
            slot_name = writable[0]
        if slot_name and not set(must) - {slot_name}:
            return {slot_name: text}
        return None

    def _build_message_call(self, text, call_id):
        """A ready tool call that sends `text` to the shopper, or None when it would not pass the schema."""
        payload = self._text_arguments(text)
        if not payload:
            return None
        sender = self.roles.get("message")
        payload, problems = _validate_value(payload, self.param_schemas[sender], "args")
        if problems:
            return None
        body = {"name": sender, "arguments": json.dumps(payload)}
        return {"id": call_id, "type": "function", "function": body}

    def _record_replies(self, replies):
        """Add tool replies to the conversation, the status line riding on the last; return the bare texts."""
        last = len(replies) - 1
        for cnt, (call_id, said) in enumerate(replies):
            shown = said + "\n" + self.status() if cnt == last else said
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": shown})
        return [said for _, said in replies]

    def _run_question_planner(self, force=False):
        """Let the question planner run; True when it sent something."""
        return _eje_ask(self, {"force": force})["exit"]

    def _send_answer(self, text):
        """Send a reply that the reading prepared for a question the shopper put to us."""
        call = self._build_message_call(text, f"reply_{self.step}") if self._ejc_may_reply() else None
        if call is None:
            return False
        self.messages.append({"role": "assistant", "tool_calls": [call], "content": "Sending my answer to their question."})
        dispatched = self._run_calls([call], planned=True)
        self._record_replies(dispatched)
        return True

    def _ejc_may_reply(self):
        if not self.roles.get("message"):
            return False
        if self.messages_used >= self._message_limit():
            return False
        spare = self.remaining_steps() - _spare_steps_for(self.max_steps)
        return spare > 0 and self.seconds_remaining() >= WRAP_UP_WINDOW

    # ==== units, the last-chance audit =====================================

    def _units_problem(self, label_name, arguments):
        """Why a cart call's unit count is wrong, or '' when it matches (or the call is not an add)."""
        units = self._wanted_units()
        if self.add_signature is not None and self.add_signature != self._choice_signature(label_name, arguments):
            return ""
        if self._wanted_items():
            named = [fv for fv in arguments.values() if isinstance(fv, str) and fv in self.seen_ids]
            units = self._ejc_line_units(named[0]) if len(named) == 1 else None
            if units is None:
                return ""
        counters = [(kname, pager) for kname, pager in (self.param_schemas[label_name].get("properties") or {}).items()
                    if isinstance(pager, dict) and pager.get("type") == "integer" and "enum" not in pager]
        for field, tool_desc in counters:
            dispatched = arguments[field] if field in arguments else tool_desc.get("default", 1)
            if isinstance(dispatched, int) and dispatched != units:
                return f"wrong count: {dispatched} here, yet {units} of one single listing are wanted per the request."
        return ""

    def _audit_before_decline(self):
        """A single judge pass over all records before the first decline; a hold text naming possible fits, or ''."""
        if self.audit_done or self.spend_exceeded() or self.seconds_remaining() < WRAP_UP_WINDOW + 30:
            return ""
        self.audit_done = True
        excerpts = _ejc_newest_within(self.records, 24000)
        if not excerpts:
            return ""
        request = str(self.policy.get("query") or "")
        verdict = self.ask_judge("\n\n".join([
            DECLINE_AUDIT_TEXT,
            "REQUIREMENTS NOW IN FORCE:\n" + self.requirement_summary(),
            "REQUEST AS WRITTEN:\n" + request,
            "SEEN SO FAR (LATEST AT THE TOP):\n" + "\n".join(excerpts),
        ]), 1200)
        self.trace.append({"kind": "audit", "result": verdict})
        named = (verdict or {}).get("possible_fits") or []
        picks = [str(el) for el in named if isinstance(el, (str, int))][:5]
        if not picks:
            return ""
        because = verdict.get("reason")
        return ("Hold on before declining: a second look at every record suggests these listings might meet all "
                "conditions" + (" (%s)" % because if because else "") + ": " + ", ".join(picks) + ". Purchase whichever "
                "qualifies once each has been opened for its sheet, availability and cost; declining becomes "
                "correct only after every one is ruled out.")

    # ==== steps, epochs, units =============================================

    def _reserved_steps(self):
        """Steps held back for a hold, the finish, and one refused finish plus a re-read."""
        return _spare_steps_for(self.max_steps)

    def _has_step_room(self, cost):
        """True when an optional action of `cost` steps leaves the reserve intact."""
        return self.remaining_steps() - cost >= self._reserved_steps()

    def _arm_signature(self):
        """Tag of the current state of knowledge; it moves on every change, refusal or world line."""
        return "%s.%s" % (self.needs_revision, self.cart_generation)

    def _advance_generation(self):
        self.cart_generation += 1

    def _wanted_units(self):
        """Units to buy. A count above one stands only when the request's own words tie it to the product."""
        count = self.plan.get("quantity")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            return 1
        if count == 1 or "quantity_quote" not in self.plan:
            return count
        return count if self._ejc_count_quoted(self.plan.get("quantity_quote")) else 1

    def _wanted_items(self):
        """The products asked for when the request names two or more: dicts with "product", "requirements" and
        "quantity" from the setup reading. [] for a one-product request, which keeps the single-line path."""
        plan = self.plan if isinstance(self.plan, dict) else {}
        raw = plan.get("lines")
        items = []
        for element in raw if isinstance(raw, list) else ():
            if not isinstance(element, dict) or not str(element.get("product") or "").strip():
                continue
            needs = element.get("requirements", element.get("requirement"))
            needs = [needs] if isinstance(needs, str) else needs if isinstance(needs, list) else []
            count = element.get("quantity")
            ok = isinstance(count, int) and not isinstance(count, bool) and count >= 1
            items.append({"product": str(element["product"]).strip(), "quantity": count if ok else 1,
                          "requirements": [str(fv).strip() for fv in needs if isinstance(fv, (str, int, float)) and str(fv).strip()]})
        return items if len(items) >= 2 else []

    @staticmethod
    def _ejc_stems(text):
        return {tk.rstrip("s")[:6] for tk in re.findall(r"[a-z0-9]+", str(text).lower()) if len(tk) > 2}

    def _ejc_item_index(self, unit_ref):
        """Which requested product a listing is: the one whose product words and wanted values share the most
        word stems with the store's own texts on that listing; None on a tie, no overlap or one product."""
        items = self._wanted_items()
        if not items or unit_ref is None:
            return None
        have = set()
        for branch in _iter_dicts(self.records):
            if unit_ref in branch.values():
                for fv in branch.values():
                    if isinstance(fv, str) and fv != unit_ref:
                        have |= self._ejc_stems(fv)
        scores = []
        for element in items:
            words = self._ejc_stems(element["product"])
            values = set().union(*[self._ejc_stems(rw.partition(":")[2]) for rw in element["requirements"]] or [set()])
            scores.append(2 * len(words & have) + len(values & have))
        top = max(scores)
        return scores.index(top) if top > 0 and scores.count(top) == 1 else None

    def _ejc_line_units(self, unit_ref):
        """Units wanted for this listing's own product; None when it cannot be told."""
        index = self._ejc_item_index(unit_ref)
        return None if index is None else self._wanted_items()[index]["quantity"]

    def _ejc_other_items_needs(self, unit_ref):
        """Stated conditions, and revealed needs filed under a product, that belong only to the other requested
        products (empty when unknown)."""
        items, index = self._wanted_items(), self._ejc_item_index(unit_ref)
        if index is None:
            return set()
        own = set(items[index]["requirements"])
        foreign = {rw for k, el in enumerate(items) if k != index for rw in el["requirements"] if rw not in own}
        mine = [self._ejb_key_words(rw) for rw in self._ejc_homed(items[index]["product"])]
        away = [rw for k, el in enumerate(items) if k != index for rw in self._ejc_homed(el["product"])]
        keys = [self._ejb_key_words(rw) for rw in away if self._ejb_key_words(rw) not in mine]
        return foreign | set(away) | {rw for rw in self._list_conditions() if self._ejb_key_words(rw) in keys}

    def _ejc_homed(self, product):
        """Revealed needs still in force that were filed under this requested product."""
        state = self._need_list
        return [rw for rw, home in self.need_homes.items() if home == product and rw not in state.dropped]

    def _ejc_own_needs(self, element):
        """A requested product's stated requirements followed by the revealed needs filed under it."""
        homed = [rw + " (revealed by the shopper)" for rw in self._ejc_homed(element["product"])
                 if rw not in element["requirements"]]
        return "; ".join(element["requirements"] + homed) or "see the request"

    def _ejc_item_lines(self):
        many = self._wanted_items()
        if not many:
            return []
        out = [f"- {len(many)} products are requested; the cart needs one line for each, at its own quantity:"]
        for k, element in enumerate(many, 1):
            needs = self._ejc_own_needs(element)
            out.append(f"  product {k}: {element['product']}, quantity {element['quantity']}, requirements: {needs}")
        return out

    def _ejc_item_scope(self, unit_ref):
        """For several requested products: which one a listing is judged as ('' for one product)."""
        many = self._wanted_items()
        if not many:
            return ""
        index = self._ejc_item_index(unit_ref)
        if index is None:
            return (" (the request asks for several products; judge a listing only against the requirements of "
                    "the product it is, never against the other products')")
        element = many[index]
        return (" (requested product %d: %s, quantity %d; only its own requirements apply: %s)"
                % (index + 1, element["product"], element["quantity"], self._ejc_own_needs(element)))

    def _ejc_missing_items(self):
        """(requested products with no cart line, lines held): by count, named by matching; no products when it
        cannot be told. A cart view newer than the last cart change adds the known listings it shows."""
        items, held = self._wanted_items(), list(self.cart_ids)
        if self.last_cart_view is not None and self.cart_seen_at >= self.cart_changed_at:
            held += [idx for idx in _eje_view_listings(self, self.last_cart_view) if idx not in held]
        if not items or not held:
            return [], len(held)
        covered = [self._ejc_item_index(idx) for idx in held]
        missing = [el for k, el in enumerate(items) if k not in covered]
        if len(held) >= len(items) and None in covered:
            return [], len(held)
        return missing, len(held)

    def _ejc_count_quoted(self, quote):
        """The quoted count is in the request and carries a digit or a product word."""
        if not isinstance(quote, str) or not quote.strip():
            return False
        if quote.strip().lower() not in str(self.policy.get("query") or "").lower():
            return False
        if re.search(r"\d", quote):
            return True
        return bool(_word_set(self.plan.get("product")) & _word_set(quote))

    # ==== listing table ====================================================

    def _row_table(self):
        """Every listing seen, keyed by id, later records overriding earlier ones.

        Field roles come from value types: price = the numeric key most listings carry, stock = the boolean key
        most carry, label = the short text key whose values repeat. Facts are the short texts of a record (and of
        its nested dicts); a record with five or more text facts marks the listing opened."""
        found = _ejc_listing_nodes(self.records)
        slots = _ejc_field_roles(found)
        book = {}
        for ident, carry in found:
            element = book.get(ident)
            if element is None:
                element = book[ident] = dict(price=None, stock=None, label=None, facts={}, opened=False)
            for slot_name, key, accepts in slots:
                if accepts(carry.get(key)):
                    element[slot_name] = carry[key]
            fact_sheet = _ejc_sheet(carry, ident)
            if len([fv for fv in fact_sheet.values() if isinstance(fv, str)]) >= 5:
                element["facts"].update(fact_sheet)
                element["opened"] = True
        return book

    def _list_conditions(self):
        """Requirements written as "property: constraint". Lines without both halves (a budget put in words, say)
        are left to the money check and the reviewer."""
        self._pin_stated_values()
        w = self._need_list
        pool = list(w.newest_lines) if w.newest_lines is not None else [rw for rw in w.stated + w.revealed if rw not in w.dropped]
        if w.newest_lines is not None:
            pool = [text_line for text_line in pool if not self._ejc_restates_limits(text_line)]
        return [text_line for text_line in pool if all(half.strip() for half in str(text_line).partition(":")[::2])]

    def _ejc_restates_limits(self, text_line):
        """A line the shopper reading copied from the summary, not a product condition: neither stated nor revealed,
        no stated or revealed line names its property, and each figure in it is the budget or a unit count."""
        w = self._need_list
        prop_name, _, constraint = str(text_line).partition(":")
        figures = {float(fv) for fv in re.findall(r"\d+(?:\.\d+)?", constraint)}
        limits = {float(fv) for fv in [_as_number(w.budget), self._wanted_units()]
                  + [el["quantity"] for el in self._wanted_items()] if isinstance(fv, (int, float))}
        if not figures or not figures <= limits or text_line in w.stated + w.revealed:
            return False
        own = _word_set(prop_name)
        return not any(own and _word_set(str(rw).partition(":")[0]) == own for rw in w.stated + w.revealed)

    def _within_price_cap(self, listing_row):
        """In stock (or unknown) and, when both are known, price times units within the budget."""
        if listing_row.get("stock") is False:
            return False
        cap, cost = _as_number(self._need_list.budget), listing_row.get("price")
        many = self._wanted_items()   # several products: the row's own count is unknown here; take the smallest
        units = min(el["quantity"] for el in many) if many else self._wanted_units()
        return True if cap is None or cost is None else cost * units <= cap

    def _unspoken_conditions(self, row_map=None):
        """Conditions no opened listing speaks about: no key carries the property's words and no listing's text
        carries the wanted words. Such a condition cannot set listings apart. Counted only from three opened ones."""
        row_index = self._row_table() if row_map is None else row_map
        shown = [rw["facts"] for rw in row_index.values() if rw["opened"]]
        if len(shown) < 3:
            return []
        named = [_word_set(kname) for fact_sheet in shown for kname in fact_sheet]
        voiced = [_ejc_said(fact_sheet) for fact_sheet in shown]
        quiet = []
        for text_line in self._list_conditions():
            parsed = _ejc_parse(text_line)
            if parsed is None or not parsed[0]:
                continue
            prop_terms, vw = parsed
            if any(prop_terms <= kname for kname in named) or (vw and any(vw <= tk for tk in voiced)):
                continue
            quiet.append(text_line)
        return quiet

    def _condition_verdicts(self, listing_row):
        """(condition, literal verdict) for this listing, conditions no opened listing states left out."""
        quiet = set(self._unspoken_conditions())
        fact_sheet = listing_row.get("facts") or {}
        pairs = []
        for text_line in self._list_conditions():
            if text_line not in quiet:
                pairs.append((text_line, _literal_match(fact_sheet, text_line, self._shop_phrasing(text_line))))
        return pairs

    def _row_passes_literal(self, listing_row):
        """False if any condition fails by its words, None if one cannot be told, else True. A listing with no key
        for a property counts as stating it when one of its free texts (the title, say) holds every wanted word."""
        fact_sheet = listing_row.get("facts") or {}
        outcomes = [True if fv is None and _ejc_text_states(fact_sheet, text_line) else fv
                    for text_line, fv in self._condition_verdicts(listing_row)]
        for decisive in (False, None):
            if any(fv is decisive for fv in outcomes):
                return decisive
        return True

    def _missed_conditions(self, unit_ref, row_map):
        """Conditions this listing's opened details miss: a value that fails literally, or a property it never
        mentions although most opened listings of its section do (likely another kind of item with the same noun)."""
        me = row_map.get(unit_ref) or {}
        fact_sheet = me.get("facts") or {}
        kind = me.get("label")
        section = [rw for rw in row_map.values() if rw["opened"] and (kind is None or rw["label"] == kind)]
        said = _ejc_said(fact_sheet)
        missed = []
        foreign = self._ejc_other_items_needs(unit_ref)
        for text_line in self._list_conditions():
            if text_line in foreign:
                continue
            result_info = _literal_match(fact_sheet, text_line, self._shop_phrasing(text_line, row_map))
            if result_info is False or (result_info is None and self._ejc_silent_among_peers(me, text_line, said, section)):
                missed.append(text_line)
        return missed

    def _ejc_silent_among_peers(self, me, text_line, said, section):
        """The listing never names the property, while its section's opened listings do (or one of them passes)."""
        prop_terms, vw = _ejc_parse(text_line) or (set(), set())
        fact_sheet = me.get("facts") or {}
        if not prop_terms or not vw or vw <= said:
            return False
        if any(prop_terms <= _word_set(kname) or _key_names_property(kname, prop_terms)
               or (_word_set(kname) and _word_set(kname) < prop_terms) for kname in fact_sheet):
            return False   # the property is named, or a key holds part of its name; an open value is for the reviewer
        speakers = [rw for rw in section if any(prop_terms <= _word_set(kname) for kname in rw["facts"])]
        if len(speakers) >= 2 and 2 * len(speakers) >= len(section):
            return True
        for other in speakers:
            if other is not me and self._within_price_cap(other) and _literal_match(other["facts"], text_line) is True:
                return True
        return False

    def _request_sentence_for(self, prop_name):
        """The first sentence of the request that holds every word of the property name ('' when none does)."""
        prop_terms = _word_set(prop_name)
        if prop_terms:
            for sentence in re.split(r"(?<=[.!?])\s+", str(self.policy.get("query") or "")):
                if prop_terms <= _word_set(sentence):
                    return sentence.strip()[:240]
        return ""

    # ==== evidence before a decline ========================================

    def _decline_evidence(self):
        """What the listings' own facts say before a decline: full fits (re-checked by the reviewer), near fits
        (one bounded re-look each), per-condition evidence, and affordable listings never opened. '' if nothing."""
        row_index = self._row_table()
        affordable = {idx: rw for idx, rw in row_index.items() if rw["opened"] and self._within_price_cap(rw)}
        marks = {}
        for idx, rw in affordable.items():   # several products: a listing answers only to its own product's needs
            foreign = self._ejc_other_items_needs(idx)
            marks[idx] = [(ch, fv) for ch, fv in self._condition_verdicts(rw) if ch not in foreign]
        sure = [idx for idx in affordable if all(fv is True for _, fv in marks[idx])]
        good = self._ejc_confirm(sure)
        near_notes = self._ejc_relook_near(row_index, marks, sure, good)
        text_lines = list(dict.fromkeys(ch for got in marks.values() for ch, _ in got))
        unopened = self._ejc_unopened_picks(row_index, affordable, text_lines)
        self.proven_fits = list(good)
        if not (good or near_notes or unopened):
            return ""
        parts = []
        if good:
            parts.append("going by what they state themselves, all requirements are met by " + ", ".join(good)
                         + " - declining while any of them is on offer is worth nothing whereas a valid purchase "
                         "counts, so purchase the strongest one")
        if near_notes:
            parts.append("no requirement is broken by these opened listings, though one detail on each is still "
                         "unproven: " + "; ".join(near_notes) + " - settle that open point for each by opening its "
                         "sheet (title and variant included) and purchase one where it holds")
        if unopened:
            parts.append("never opened yet priced within the limit: " + ", ".join(unopened)
                         + " - open them first" + self._ejc_read_by_view())
        if text_lines and not good:
            per_line = []
            for ch in text_lines:
                backers = [idx for idx in affordable if (ch, True) in marks[idx]][:3]
                per_line.append(f"'{ch}': " + (", ".join(backers) or "unconfirmed everywhere"))
            parts.append("what has been confirmed, requirement by requirement: " + "; ".join(per_line))
        return "Hold on before declining: " + "; ".join(parts) + "."

    def _ejc_confirm(self, sure):
        """Up to three full fits the reviewer does not object to, held items first."""
        queue = [idx for idx in self.cart_ids if idx in sure]
        queue += [idx for idx in sure if idx not in queue]
        good = []
        for candidate in queue:
            if len(good) >= 3:
                break
            if candidate not in good and not self._verify_item(candidate, "matches all, verified ahead of declining"):
                good.append(candidate)
        return good

    def _ejc_relook_near(self, row_index, marks, sure, good):
        """Near fits (nothing fails, at most a few open): two of them get one re-look each; a re-look that passes
        joins `good`, the rest come back as notes naming what is still open."""
        def still_open(idx):
            return [ch for ch, fv in marks[idx] if fv is None]

        def price_of(idx):
            known = row_index[idx]["price"]
            return 1e18 if known is None else known

        close_fits = [idx for idx, got in marks.items() if idx not in sure and got and all(fv is not False for _, fv in got)
                and len(still_open(idx)) <= max(1, len(got) // 2)]
        close_fits.sort(key=lambda idx: (len(still_open(idx)), price_of(idx), idx))
        notes = []
        relooks_left = 2
        for candidate in close_fits:
            if len(good) >= 3:
                break
            if relooks_left and candidate not in self.relooked:
                self.relooked.add(candidate)
                relooks_left -= 1
                if not self._verify_item(candidate, "almost matching; rechecked ahead of declining"):
                    good.append(candidate)
                    continue
            pending = " and ".join("'%s'" % ch for ch in still_open(candidate))
            notes.append(f"{candidate} (still unconfirmed: {pending})")
        return notes

    def _ejc_read_by_view(self):
        """When the details tool is withdrawn this late but the view tool is still offered: a pointer to it."""
        offered = {tk["function"]["name"] for tk in self._tools_this_turn()}
        details, view = self.roles.get("details"), self.roles.get("view")
        if not view or view not in offered or not details or details in offered:
            return ""
        return f" (the details tool is withdrawn now; {view} is still offered and shows their data)"

    def _ejc_unopened_picks(self, row_index, affordable, text_lines):
        """Listings never opened, as many as the free steps allow (at most four): rows of the requested products
        that still lack a fit, from the sharpest searches typing a required value; failing those, affordable rows of
        the opened ones' sections, most wanted words first, then cheapest."""
        spare = max(0, self.remaining_steps() - self._reserved_steps() - 1)
        aimed = _eje_unfitted_rows(self, min(spare, 4))
        if aimed:
            return aimed
        sections = {rw["label"] for rw in affordable.values() if rw["label"]}
        wanted = set()
        for ch in text_lines:
            wanted.update(_word_set(str(ch).partition(":")[2].split("!=")[0]))
        ranked = sorted(
            (-len(wanted & _ejc_said(rw.get("facts") or {})), rw["price"], idx) for idx, rw in row_index.items()
            if not rw["opened"] and rw["price"] is not None and self._within_price_cap(rw)
            and (not sections or rw["label"] in sections))
        return [element[2] for element in ranked[:min(spare, 4)]]

    # ==== finishing from code ==============================================

    def _minimal_closing_args(self, label_name, text, duty=None):
        """Minimal valid arguments for a finishing tool: the job's fixed values plus `text` in text fields."""
        arg_shape = self.param_schemas.get(label_name) or {}
        field_specs = arg_shape.get("properties") or {}
        made = dict(self.fixed.get(duty) or {}) if duty else {}
        for key in arg_shape.get("required") or []:
            if key in made:
                continue
            ok, field_value = _ejc_default_for(field_specs.get(key) or {}, text)
            if not ok and duty == "order":
                ok, field_value = _ejc_cart_lines(self, field_specs.get(key) or {})
            if ok:
                made[key] = field_value
        if len(field_specs) == 1:
            only, tool_desc = next(iter(field_specs.items()))
            if only not in made and isinstance(tool_desc, dict) and tool_desc.get("type") == "string" and "enum" not in tool_desc:
                made[only] = _ejc_clip(text, tool_desc)
        return made

    def _clean_decline_args(self, label_name, arguments):
        """Put the neutral remark in every free text field when names are not allowed or a seen id shows up."""
        rule = self.plan.get("decline_no_names")
        if rule is False or not (rule or self._ejc_names_seen_id(arguments)):
            return arguments
        field_specs = (self.param_schemas.get(label_name) or {}).get("properties") or {}
        out = {}
        for key, field_value in arguments.items():
            tool_desc = field_specs.get(key) or {}
            free = isinstance(field_value, str) and "enum" not in tool_desc
            out[key] = _ejc_clip(_NOTHING_FITS_TEXT, tool_desc) if free else field_value
        return out

    def _ejc_names_seen_id(self, arguments):
        texts = [fv for fv in arguments.values() if isinstance(fv, str)]
        return any(tk and tk in self.seen_ids for fv in texts for tk in _WORD_BREAKS.split(fv))

    def _send_closing(self, duty, text):
        """Send the finishing call for `job` built from code; a call the schema rejects is logged, not sent."""
        tool = self.roles[duty]
        draft = self._minimal_closing_args(tool, text, duty)
        if duty == "decline":
            draft = self._clean_decline_args(tool, draft)
        action = {"name": tool, "args": draft}
        action["args"], issues = _validate_value(draft, self.param_schemas[tool], "args")
        log = {"kind": "forced_finish", "action": action}
        if issues:   # the store would refuse it and still count the step
            log.update(reply=None, issues=issues)
            self.trace.append(log)
            return None
        log["reply"] = self.call_store(action)
        self.trace.append(log)
        return log["reply"]

    def _listing_title(self, unit_ref):
        """The store's own name for a listing, for a message to the shopper (who never sees ids): the text with
        the most words shown next to its id; the id only when nothing was shown."""
        texts = [fv for branch in _iter_dicts(self.records) if unit_ref in branch.values() for fv in branch.values()
                 if isinstance(fv, str) and fv != unit_ref and len(fv) <= 160 and fv.split()]
        top_choice = max(texts, key=lambda fv: len(fv.split()), default="")
        return '"%s"' % top_choice if top_choice else "item " + unit_ref

    def finish_now(self):
        """Finish from code: order what the cart holds when an order is possible, otherwise decline; a sent order
        that leaves the episode open is followed by a decline."""
        if self.finished or self.halted or self.remaining_steps() <= 0 or self.seconds_remaining() < 2:
            return
        has = self.roles.get
        duty = "order" if has("order") and self.cart_used and (self.cart_ids or not has("cart")) else "decline"
        if not has(duty):
            return
        result_info = self._send_closing(duty, _ejc_FORCED_TEXT[duty])
        if duty != "order" or not result_info or result_info.get("finished"):
            return
        if has("decline") and self.remaining_steps() > 0:
            self._send_closing("decline", _NOTHING_FITS_TEXT)

    # ==== entry points into the state machine ==============================

    def _run_calls(self, calls, planned=False):
        """Check and send a batch of calls (the model's, or code's own when planned); replies in call order."""
        return _eje_execute(self, {"calls": calls, "planned": planned})["replies"]

    def run(self):
        _eje_episode(self, {})
        return self.trace


# ======================================================================
# Finishing jobs, fixed texts, shared word and record helpers.
# ======================================================================

_CLOSING_JOBS = ("order", "decline")

_CLOSING_ROLES_TEXT = """Reply with a single line of JSON and nothing around it, shaped exactly like this:
{"order": {"tool": exact tool name, "fixed": {argument: value}}, "decline": {"tool": exact tool name, "fixed": {argument: value}}}

What the two entries mean: the tools further down belong to a shop and are used by an assistant. Exactly two kinds of call bring the job to an end -
- "order" buys what is currently in the basket;
- "decline" ends it without buying anything.
For each, name the tool and the argument values that make it happen, judged from the full argument schemas: their descriptions and the values each one allows.

How to fill "fixed":
- include every argument whose permitted values (an option list, or true/false) select the behaviour of the call, set to the exact value that job needs;
- leave free-text arguments out, and give {} when the tool has no such argument;
- if a single tool serves both jobs, give both entries the same argument names with different values.
A tool that merely displays the basket, or merely sends the shopper a note, is neither of the two."""

_ROLE_REFEREE_TEXT = """You referee a disagreement. READING A and READING B (after this text) each claim to know which of a shop's tools buys the current basket contents - the job called "order" - and which closes the job with nothing purchased - the job called "decline".

Steps:
1. Compare each claim with the tool list: argument schemas, permitted values, descriptions.
2. Decide the true tool for each job. Taking it from A, from B, or from neither is all fine.
3. Under "fixed" list the arguments whose permitted values switch what the call does, each at its exact value. When a single tool handles both jobs, the two entries carry the same argument names with different values.

Output only this, as one line of JSON with no other text:
{"order": {"tool": exact tool name, "fixed": {argument: value}}, "decline": {"tool": exact tool name, "fixed": {argument: value}}}"""

_NOTHING_FITS_TEXT = "Closing with no purchase: no listing I could verify satisfies every stated requirement."
_ejc_FORCED_TEXT = {"order": "Order the cart.", "decline": _NOTHING_FITS_TEXT}
_WORD_BREAKS = re.compile(r"[\s.,;:!?()\[\]{}<>\"'`]+")
_SKIP_NOTICE = "Skipped, no step spent"
_HELD_NOTICE = "Not sent: your own check stopped this call (the store did not), and no step was used."
_ejc_PARTS = re.compile(r"[,/;|]")
_ejc_MARKS = ("<=", ">=", "covers")


def _ejc_text_fields(props):
    """Names of free text arguments (string typed, no option list)."""
    out = []
    for key, tool_desc in props.items():
        if isinstance(tool_desc, dict) and "enum" not in tool_desc and "string" in (tool_desc.get("type") or []):
            out.append(key)
    return out


def _ejc_clip(text, tool_desc):
    cap = tool_desc.get("maxLength", len(text))
    return text[:cap]


def _ejc_default_for(tool_desc, text):
    """(True, value) for a required argument a finishing call can fill on its own; (False, None) otherwise."""
    kind = tool_desc.get("type")
    if "string" in (kind or []):
        return True, _ejc_clip(text, tool_desc)
    if tool_desc.get("enum"):
        return True, tool_desc["enum"][0]
    if kind == "boolean":
        return True, True
    if kind in ("integer", "number"):
        return True, max(1, tool_desc.get("minimum", 1))
    return False, None


def _ejc_cart_lines(store_run, tool_desc):
    """(True, lines) for a required list a forced order can fill from the tracked cart: its ids as plain texts,
    or one entry per id with the id in the entry's one free text field and the held count in its whole-number one."""
    entry = tool_desc.get("items") if tool_desc.get("type") == "array" else None
    held = list(store_run.cart_ids)
    if not isinstance(entry, dict) or not held:
        return False, None
    if "string" in (entry.get("type") or []) and "enum" not in entry:
        return True, held
    inner = entry.get("properties") if isinstance(entry.get("properties"), dict) else {}
    texts = _ejc_text_fields(inner)
    counts = [key for key, sub in inner.items() if isinstance(sub, dict) and sub.get("type") == "integer"]
    if len(texts) != 1 or len(counts) > 1:
        return False, None
    lines = []
    for idx in held:
        line = {texts[0]: idx}
        if counts:
            line[counts[0]] = max(1, int(store_run.units_in_cart.get(idx) or 1))
        lines.append(line)
    return True, lines


def _ejc_newest_within(records, limit):
    """Short forms of the records, newest first, skipping any that would push the total past `limit` characters."""
    kept, size = [], 0
    for carry in records[::-1]:
        piece = _to_compact_json(_shorten_value(carry, 160))
        if size + len(piece) <= limit:
            kept.append(piece)
            size += len(piece)
    return kept


def _ejc_parse(text_line):
    """(property words, value words) of a "property: constraint" line; None without the colon."""
    lead_block, colon, tail = str(text_line).partition(":")
    return (_word_set(lead_block), _word_set(tail)) if colon else None


def _ejc_said(fact_sheet):
    """Every word a facts dict states in its values."""
    return _word_set(" ".join(str(fv) for fv in fact_sheet.values()))


def _ejc_listing_nodes(records):
    """(listing id, dict) for each dict holding an id-like text under a key whose name contains "id"."""
    out = []
    for carry in _iter_dicts(records):
        for key, cell in carry.items():
            if isinstance(cell, str) and "id" in str(key).lower() and _ID_SHAPE.match(cell):
                out.append((cell, carry))
                break
    return out


def _ejc_is_number(field_value):
    return isinstance(field_value, (int, float)) and not isinstance(field_value, bool)


def _ejc_field_roles(found):
    """[(slot, key, test)] for the three slots (cost, availability, label): the key most often holding a number,
    a boolean, and a short text whose values repeat (earliest key on a tie)."""
    numeric, boolean, texts = [], [], {}
    for ident, carry in found:
        for key, cell in carry.items():
            if cell == ident:   # a listing's own id shows up in many places, cart lines too
                continue
            if isinstance(cell, bool):
                boolean.append(key)
            elif isinstance(cell, (int, float)):
                numeric.append(key)
            elif isinstance(cell, str) and 0 < len(cell) <= 40:
                texts.setdefault(key, []).append(cell)
    repeated = [kname for kname, vals in texts.items() if len(vals) >= 2 * len(set(vals))]
    label = max(repeated, key=lambda kname: len(texts[kname]), default=None)
    return [("price", _ejc_most(numeric), _ejc_is_number),
            ("stock", _ejc_most(boolean), lambda fv: isinstance(fv, bool)),
            ("label", label, lambda fv: isinstance(fv, str))]


def _ejc_most(keys):
    """The most frequent entry, the one seen first on a tie; None for no entries."""
    return max(dict.fromkeys(keys), key=keys.count, default=None)


def _ejc_sheet(carry, ident):
    """Short text values of a record (not its id), then text and list members of its nested dicts."""
    fact_sheet = {}
    for key, cell in carry.items():
        if isinstance(cell, str) and cell != ident and 0 < len(cell) <= 120:
            fact_sheet[key] = cell
    for child in carry.values():
        if isinstance(child, dict):
            fact_sheet.update((kname, fv) for kname, fv in child.items() if isinstance(fv, (str, list)))
    return fact_sheet


def _ejc_containers(roots):
    """Every dict and list reachable from the roots (any order)."""
    todo = list(roots)
    while todo:
        branch = todo.pop()
        if isinstance(branch, list):
            yield branch
            todo += branch
        elif isinstance(branch, dict):
            yield branch
            todo += [fv for fv in branch.values() if isinstance(fv, (dict, list))]


def _looks_like_listing(field_value, ids):
    """A seen listing id that looks like a code (eight or more characters mixing letters and digits), not a word
    or a short measure found under a key that happens to contain the letters."""
    if isinstance(field_value, str) and len(field_value) >= 8 and field_value in ids:
        return any(map(str.isdigit, field_value)) and any(map(str.isalpha, field_value))
    return False


def _search_row_ids(roots, ids):
    """id() of every result row: a dict in a list where another dict has the same keys, holding a listing id.
    Such rows show the listing in a search result, not its property sheet."""
    selected = set()
    for branch in _ejc_containers(roots):
        if isinstance(branch, dict):
            continue
        dicts = [mt for mt in branch if isinstance(mt, dict)]
        layouts = [tuple(sorted(mt)) for mt in dicts]
        for member, layout in zip(dicts, layouts):
            if layouts.count(layout) >= 2 and any(_looks_like_listing(fv, ids) for fv in member.values()):
                selected.add(id(member))
    return selected


def _own_listing_fields(node_list, ids):
    """(listing id, key, value) for plain text fields that one listing shows in two or more record shapes (a result
    row and the opened page): its own fields such as title or maker, not properties worth asking about."""
    forms = {}
    for carry in node_list:
        owner = next((fv for fv in carry.values() if _looks_like_listing(fv, ids)), None)
        if owner is None:
            continue
        layout = tuple(sorted(carry))
        for key, cell in carry.items():
            if isinstance(cell, str) and cell not in ids:
                forms.setdefault((owner, key, cell), set()).add(layout)
    return {triple for triple, layouts in forms.items() if len(layouts) >= 2}


def _name_stems(label_name):
    """Words of a property name with a plural "s" dropped, so "styles" and "style" meet."""
    out = set()
    for term in _word_set(label_name):
        plural = len(term) > 3 and term[-1] == "s" and term[-2] != "s"
        out.add(term[:-1] if plural else term)
    return out


def _is_enum_spec(tool_desc):
    """An argument that selects from fixed options: an option list, a constant, or a boolean."""
    if not isinstance(tool_desc, dict):
        return False
    if tool_desc.get("enum") or "const" in tool_desc:
        return True
    return "boolean" in (_declared_types(tool_desc) or [])


def _word_set(text):
    """Lower-case tokens: every number, and words of at least three characters."""
    found = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return {tk for tk in found if tk.isdigit() or len(tk) > 2}


def _flatten_text(text):
    """Lower case, whitespace runs folded to one space: for word-for-word phrase checks."""
    chunks = str(text or "").lower().split()
    return " ".join(chunks)


def _words_match(have, wish):
    """Equal words, or one is the other plus up to two letters at the end (a plural)."""
    a, b = (have, wish) if len(have) <= len(wish) else (wish, have)
    return b.startswith(a) and len(b) <= len(a) + 2


def _words_cover(needed, have):
    """True when each needed word appears among `have` in some spelling."""
    for term in needed:
        if not any(_words_match(term, other) for other in have):
            return False
    return True


def _key_words(key):
    """A key's words; a lower-to-upper case step inside a key starts a new word."""
    return _word_set(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(key)))


def _key_names_property(key, need_terms):
    """A key whose words are exactly the property's words, allowing spellings."""
    call_args = _key_words(key)
    return bool(call_args) and len(call_args) == len(need_terms) and _words_cover(need_terms, call_args)


def _ejc_constraint(tail):
    """(value text, is exclusion, other value that must be absent or None) from the right side of a condition.
    A bracketed explanation is dropped when words remain; "!=" marks values that must not be stated."""
    plain = re.sub(r"\([^)]*\)", " ", tail)
    if not _word_set(plain):
        plain = tail
    keep, _, ban = plain.partition("!=")
    if not _word_set(ban):
        return plain, False, None
    if not _word_set(keep):
        return ban, True, None
    return keep, False, ban


def _ejc_pool(fact_sheet, prop_terms):
    """Members stated under keys carrying the property (lists and comma lists split up), and the single plain
    values of keys named exactly like it."""
    stated, singles = [], []
    for key, cell in fact_sheet.items():
        exact = _key_names_property(key, prop_terms)
        if exact or prop_terms <= _key_words(key):
            stated += [str(fv) for fv in (cell if isinstance(cell, list) else [cell])]
        if exact and not isinstance(cell, list):
            chunks = [p for p in _ejc_PARTS.split(str(cell)) if p.strip()]
            if len(chunks) == 1 and _word_set(chunks[0]):
                singles.append(chunks[0])
    member_list = [p for fv in stated for p in [fv] + _ejc_PARTS.split(fv) if _word_set(p)]
    return member_list, singles


_ejc_FREE_TEXT_WORDS = 5   # a value this long is free text (a title or blurb), not one property's value


def _ejc_text_states(fact_sheet, condition):
    """No key carries the property, the wanted value is plain words, and a free-text value holds them all."""
    lead_block, colon, tail = str(condition).partition(":")
    target, negated, absent = _ejc_constraint(tail)
    prop_terms, vw = _word_set(lead_block), _word_set(target)
    if not (colon and prop_terms and vw) or negated or absent is not None or re.search(r"\d", target):
        return False
    if _ejc_pool(fact_sheet, prop_terms)[0]:
        return False
    for cell in fact_sheet.values():
        said = _word_set(cell) if isinstance(cell, str) else set()
        if len(said) >= _ejc_FREE_TEXT_WORDS and _words_cover(vw, said):
            return True
    return False


def _ejc_stated_whole(fact_sheet, vw):
    """Some value anywhere (or one member of it) has exactly the wanted words."""
    for cell in fact_sheet.values():
        for fv in (cell if isinstance(cell, list) else [cell]):
            text = str(fv)
            if any(_word_set(p) == vw for p in [text] + _ejc_PARTS.split(text)):
                return True
    return False


def _ejc_qualified_hit(fact_sheet, condition):
    """The listing's own key holds every wanted word of a plain-word condition together with other words (a
    shade or a finish beside them, say), and no member has just the wanted words: met by the matching rules,
    though a strict reading may doubt it."""
    lead_block, _, tail = str(condition).partition(":")
    target, negated, absent = _ejc_constraint(tail)
    prop_terms, vw = _word_set(lead_block), _word_set(target)
    if not (prop_terms and vw) or negated or absent is not None or re.search(r"\d", target):
        return False
    if _literal_match(fact_sheet, condition) is not True:
        return False
    member_sets = [_word_set(mt) for mt in _ejc_pool(fact_sheet, prop_terms)[0]]
    if any(ws == vw for ws in member_sets):
        return False
    return any(_words_cover(vw, ws) and not _words_cover(ws, vw) for ws in member_sets)


def _ejc_qualified_only(fact_sheet, conditions, problems):
    """Every objection of the second check names only conditions this listing meets by a qualified value of its
    own key (see _ejc_qualified_hit). An objection naming none of them (kind, money, a reference) still counts."""
    if not problems or not conditions:
        return False
    for complaint in problems:
        said = _word_set(complaint)
        named = [text_line for text_line in conditions
                 if _word_set(str(text_line).partition(":")[0]) and _words_cover(_word_set(str(text_line).partition(":")[0]), said)]
        if not named or not all(_ejc_qualified_hit(fact_sheet, text_line) for text_line in named):
            return False
    return True


def _literal_match(facts, condition, phrasing=None):
    """Literal verdict of a listing's own facts on one "property: constraint" condition.

    Keys carry the property when their name holds all its words; their values are pooled member by member.
    True when a member holds every wanted word (or is one of the listed options); False when members exist and
    none does, or the exactly named keys state only single other values; None when no key carries it, the
    constraint is a plain number, the members only give a less specific form of the wanted words, or exactly
    named keys disagree. Exclusions use "!=". With `wording` (the store's phrasing of a range the
    request quotes) a range is met only by stating that phrasing."""
    lead_block, colon, tail = str(condition).partition(":")
    target, negated, absent = _ejc_constraint(tail)
    if absent is not None and _literal_match(facts, f"{lead_block}: != {absent}") is False:
        return False
    prop_terms, vw = _word_set(lead_block), _word_set(target)
    if not (colon and prop_terms and vw):
        return None
    member_list, singles = _ejc_pool(facts, prop_terms)
    numeric = re.search(r"\d", target) is not None
    if not member_list:   # nothing under the property's own key: a whole value elsewhere (the variant sold) may say it
        if negated or numeric or _parse_relation(target) is not None:
            return None
        return True if _ejc_stated_whole(facts, vw) else None
    if negated:
        bans = [_word_set(p) for p in _ejc_PARTS.split(target) if _word_set(p)]
        return not any(_words_cover(b, _word_set(mt)) for mt in member_list for b in bans)
    rel = _parse_relation(target)
    if rel is not None:
        if phrasing:
            return any(phrase in _flatten_text(mt) for mt in member_list for phrase in phrasing)
        return _numbers_fit(rel, member_list)
    if numeric:
        return True if any(_words_cover(vw, _word_set(mt)) for mt in member_list) else None

    def states(part):   # the member holds every wanted word
        return _words_cover(vw, _word_set(part))

    def touches(part):   # it holds them, or is a less specific form of them
        return states(part) or _words_cover(_word_set(part), vw)
    if singles and not any(touches(pager) for pager in singles):
        return False   # the property's own keys state only other values
    split = not all(touches(pager) for pager in singles)
    if any(states(mt) for mt in member_list):
        return None if split else True
    return None if split or any(touches(mt) for mt in member_list) else False   # a less specific value says too little


def _parse_relation(wish):
    """The reading's number mark as (mark, value, both signs): mark is "<=", ">=" or "covers"; "+/-" (or the
    plus-minus sign) means magnitudes are compared. None without a mark and a number."""
    text = wish.strip()
    sign = next((mt for mt in _ejc_MARKS if text.lower().startswith(mt)), None)
    if sign is None:
        return None
    number = re.search(r"\d+(?:\.\d+)?", text[len(sign):])
    if number is None:
        return None
    return sign, float(number.group()), ("+/-" in text or "\u00b1" in text)


def _numbers_fit(rel, member_list):
    """Stated numbers against a relation: "covers" needs the stated span to include the value; "<=" / ">=" are
    True when every number is on the right side, False when none is, None when they disagree."""
    sign, target_value, both = rel
    seen = [float(el) for mt in member_list for el in re.findall(r"-?\d+(?:\.\d+)?", mt)]
    if both:
        seen = [abs(el) for el in seen]
    if not seen:
        return None
    if sign == "covers":
        signed = [float(el) for mt in member_list for el in re.findall(r"-?\d+(?:\.\d+)?", mt)]
        if both and any(el < 0 for el in signed):   # a signed span must reach both -value and +value
            return min(signed) <= -target_value and max(signed) >= target_value
        return min(seen) <= target_value <= max(seen)
    hits = sum(1 for el in seen if (el <= target_value if sign == "<=" else el >= target_value))
    if hits == len(seen):
        return True
    return None if hits else False


def _cart_listing_id(session, label_name, arguments):
    """The listing a cart call points at: its first free one-token argument; for a call setting several lines, the
    first line not in the cart yet (else the last line)."""
    field_specs = session.param_schemas[label_name].get("properties") or {}
    own = list(_eje_cart_ids_in(field_specs, arguments))
    if own:
        return own[0]
    lines = [listing for listing, _ in _eje_cart_lines(field_specs, arguments)]
    fresh = [listing for listing in lines if listing not in session.cart_ids]
    return fresh[0] if fresh else (lines[-1] if lines else None)


def _looks_like_code(text, need_separator):
    """A code-like value such as a sku: only letters, digits, dashes and underscores, with a digit in it."""
    if not re.search(r"\d", text) or not re.fullmatch(r"[A-Za-z0-9_\-]+", text):
        return False
    return ("-" in text or "_" in text) if need_separator else True


def _is_text_list(field_value):
    """A non-empty list holding only strings."""
    if not isinstance(field_value, list) or not field_value:
        return False
    return all(isinstance(fv, str) for fv in field_value)


def _unwrap_text(field_value, nest_level=0):
    """Peel world wrappers off a reply: a two-entry dict whose one entry is a list of text lines and whose other
    entry is the inner result. Up to four wrappers deep; returns (all lines, innermost result)."""
    if nest_level >= 4 or not isinstance(field_value, dict) or len(field_value) != 2:
        return [], field_value
    (_, v1), (_, v2) = field_value.items()
    first, second = _is_text_list(v1), _is_text_list(v2)
    if first == second:
        return [], field_value
    said, rest = (v1, v2) if first else (v2, v1)
    deeper, nested = _unwrap_text(rest, nest_level + 1)
    return list(said) + deeper, nested


def _looks_like_refusal(field_value):
    """A refusal code from the store: a flat dict of at most three values, one a code-like single token."""
    if not isinstance(field_value, dict) or not 1 <= len(field_value) <= 3:
        return False
    if any(isinstance(fv, (dict, list)) for fv in field_value.values()):
        return False
    for raw_obj in field_value.values():
        token = raw_obj.strip() if isinstance(raw_obj, str) else ""
        if token and " " not in token and (token != token.lower() or "_" in token or re.search(r"\d", token)):
            return True
    return False


def _iter_dicts(roots):
    """Every dict in the records, in document order (a parent before its children), at most six levels deep."""
    def visit(branch, nest_level):
        if nest_level > 6:
            return
        if isinstance(branch, dict):
            yield branch
            kids = [fv for fv in branch.values() if isinstance(fv, (dict, list))]
        elif isinstance(branch, list):
            kids = branch
        else:
            return
        for kid in kids:
            yield from visit(kid, nest_level + 1)

    for root in roots:
        yield from visit(root, 0)

# ================================================================ episode handlers
# Each handler takes (ep, rec), may put a value under rec["exit"] to end its run early, and returns rec.

_Record = dict[str, Any]   # the record a run of handlers passes along

def _ejd_opening_text(store_run: Episode) -> str:
    """The first user message: the request, the setup reading, the store's opening view and the status line."""
    blocks = ["The shopper wrote:\n" + str(store_run.policy.get("query") or "")]
    parsed_reading = store_run.opening_lines()
    if parsed_reading:
        blocks.append(parsed_reading)
    first_view = store_run.policy.get("opening")
    if first_view is not None:
        blocks.append("Store view at the opening:\n" + _to_compact_json(first_view))
    blocks.append(store_run.status())
    return "\n\n".join(blocks)


def _stage_start_episode(session: Episode, carry: _Record) -> _Record:
    """Read the request once and seed the model's conversation."""
    if not session.finished:
        session.setup()
        session.trace.append(dict(kind="plan", plan=session.plan, roles=session._roles_for_log()))
        seed = _ejd_opening_text(session)
        view = session.policy.get("opening")
        if view is not None:
            session.records.append(view)
            _gather_ids(view, session.seen_ids)
        session.messages = [dict(role="system", content=SHOPPER_SYSTEM_TEXT), dict(role="user", content=seed)]
    else:
        carry["exit"] = "already finished"
    return carry


class _ejd_TurnState:
    """Passed from one model turn to the next: replies without a call, failed model calls, call numbering."""

    __slots__ = ("idle", "failures", "counter")

    def __init__(self):
        self.idle = self.failures = self.counter = 0


def _stage_loop_turns(session: Episode, carry: _Record) -> _Record:
    state = _ejd_TurnState()
    remaining = TURN_LIMIT
    while remaining > 0:
        remaining -= 1
        if _eje_turn(session, {"carry": state}).get("exit") == "stop":
            break
    return carry


def _stage_settle_episode(session: Episode, carry: _Record) -> _Record:
    if not (session.finished or session.halted):
        session.finish_now()
    return carry


def _stage_close_episode(session: Episode, carry: _Record) -> _Record:
    took = time.monotonic() - session.start_clock
    session.trace.append(dict(kind="end", finished=session.finished, steps=session.step, cost=round(session.cost, 5),
                         seconds=round(took, 1)))
    return carry


# ================================================================ turn handlers

def _ejd_close_turn(store_run: Episode, carry: _Record, give_up: bool) -> _Record:
    """End this turn: with give_up the episode is finished from code, else the next turn starts."""
    if give_up:
        store_run.finish_now()
    carry["exit"] = "stop" if give_up else "next"
    return carry


def _stage_turn_guard(session: Episode, carry: _Record) -> _Record:
    if session.finished or session.halted or session.remaining_steps() <= 0:
        carry["exit"] = "stop"
        return carry
    # without a working key, with almost no time left or past the spend line, the finish comes from code
    if session.key_dry or session.seconds_remaining() < FORCE_MARGIN or session.cost > _eja_SPEND_HARD:
        return _ejd_close_turn(session, carry, True)   # spend: finish sooner only, never another model
    return carry


def _stage_send_queued(session: Episode, carry: _Record) -> _Record:
    queued = session.queued_reply
    if not queued:
        return carry
    session.queued_reply = ""
    if session._send_answer(queued):
        carry["exit"] = "next"
    return carry


def _stage_run_planner(session: Episode, carry: _Record) -> _Record:
    asked_now = session.planner_on and session._run_question_planner()
    if asked_now:
        carry["exit"] = "next"
    return carry


def _stage_call_model(session: Episode, carry: _Record) -> _Record:
    seat_list = [PRIMARY_SEAT, FALLBACK_SEAT, PRIMARY_SEAT]
    reply_obj = session.infer(session.messages, session._tools_this_turn(), seat_list, 2500)
    if reply_obj is None:
        state = carry["carry"]
        state.failures += 1
        return _ejd_close_turn(session, carry, state.failures >= 2 or session.key_dry)
    carry["reply"] = reply_obj
    return carry


def _ejd_tidy_call(raw_obj: _Record, number: int) -> _Record:
    """One proposed call with an id and its arguments as a JSON string."""
    func = raw_obj.get("function") or {}
    arguments = func.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {})
    return {"id": raw_obj.get("id") or "call_%d" % number, "type": "function",
            "function": {"name": func.get("name"), "arguments": arguments}}


def _stage_take_calls(session: Episode, carry: _Record) -> _Record:
    """Give each proposed call an id and string arguments; make the reply text a string."""
    state, reply_obj = carry["carry"], carry["reply"]
    tidy = []
    for raw_obj in reply_obj.get("tool_calls") or []:
        if isinstance(raw_obj, dict):
            state.counter += 1
            tidy.append(_ejd_tidy_call(raw_obj, state.counter))
    text = reply_obj.get("content") or ""
    carry["calls"] = tidy
    carry["said"] = text if isinstance(text, str) else _to_compact_json(text)
    return carry


def _ejd_decisive(store_run: Episode, proposed: list[_Record]) -> bool:
    """Whether a proposed call goes to the cart, order or decline tool."""
    jobs = [store_run.roles.get(duty) for duty in ("cart", "order", "decline")]
    names = {ch["function"]["name"] for ch in proposed}
    return any(tool is not None and tool in names for tool in jobs)


def _stage_commit_turn(session: Episode, carry: _Record) -> _Record:
    """Before a cart, order or decline call the planner gets one forced try per ledger/menu/asked state."""
    if not (session.planner_on and _ejd_decisive(session, carry["calls"])):
        return carry
    basis = (session.needs_revision, session._build_property_menu()[1], len(session.asked))
    if basis != session.forced_basis:
        session.forced_basis = basis
        before = session.needs_revision
        dispatched = session._run_question_planner(force=True)
        if dispatched and before != session.needs_revision:
            carry["exit"] = "next"   # requirements moved: let the model choose again
    return carry


_ejd_NO_CALL = "Calling a tool is required in every reply, and this one had none. Go on.\n"


def _stage_idle_turn(session: Episode, carry: _Record) -> _Record:
    state = carry["carry"]
    if carry["calls"]:
        state.idle = 0
        return carry
    state.idle += 1
    session.messages += [
        {"role": "assistant", "content": carry["said"] or "(no tool call)"},
        {"role": "user", "content": _ejd_NO_CALL + session.status()},
    ]
    return _ejd_close_turn(session, carry, state.idle >= 3)


def _stage_act_turn(session: Episode, carry: _Record) -> _Record:
    proposed, text = carry["calls"], carry["said"]
    session.trace.append(dict(kind="model", content=text[:2000], calls=[ch["function"] for ch in proposed]))
    session.messages.append(dict(role="assistant", content=text or None, tool_calls=proposed))
    session._record_replies(session._run_calls(proposed))
    return carry


# ================================================================ question planner

def _spare_steps_for(max_steps: int) -> int:
    """Steps held back for a hold, the finish, and one refused finish followed by a re-read."""
    return max(5, round(0.11 * max_steps))


# Odds that asking about a property which holds a hidden need brings that need back (pooled over earlier worlds).
# Multiplied by the planner's own chance, it values a question when the request puts a price on asking.
_ejd_REVEAL_ODDS = 0.9

# Steps that following up one revealed need takes: a narrow search plus one opened listing.
_ejd_FOLLOW_STEPS = 2

# A short leading word of a property name, followed by more name.
_ejd_LEAD_WORD = re.compile(r"[^A-Za-z0-9]*([A-Za-z0-9]+)[^A-Za-z0-9]+(?=[A-Za-z0-9])")


def _token_list(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _same_tokens(a: str, b: str) -> bool:
    """Same property name once case, spaces and punctuation are ignored."""
    return _token_list(a) == _token_list(b)


def _ejd_holds_run(longer: list[str], shorter: list[str]) -> bool:
    """The shorter name (two words or more) sits whole and in order inside the longer one."""
    width = len(shorter)
    return width >= 2 and any(longer[pos:pos + width] == shorter for pos in range(len(longer) - width + 1))


def _ejd_asked_before(store_run: Episode, prop_name: str) -> bool:
    """Asked already: the same name, or a sibling name that holds or is held by an asked name of two words or more."""
    own = _token_list(prop_name)
    for done in store_run.asked_props:
        theirs = _token_list(done)
        if own == theirs or _ejd_holds_run(own, theirs) or _ejd_holds_run(theirs, own):
            return True
    return False


def _ejd_reserved_messages(store_run: Episode) -> int:
    """Messages kept for stated procedures that are done by writing to the shopper."""
    return sum(step_proc["count"] for step_proc in store_run._pending_procedures("order") if step_proc["role"] == "message")


def _ejd_clean_want(wish: str) -> set[str]:
    return _word_set(re.sub(r"\(.*?\)", " ", wish)) - _word_set("!=")


def _ejd_decided(store_run: Episode, prop_name: str, details_map: _Record) -> bool:
    """True when a requirement in force already settles this menu property: it names the property (or an alias),
    or its value words fall inside one of the property's values."""
    spellings = [_word_set(prop_name)] + [_word_set(alias) for alias in details_map.get("also") or []]
    for clause in store_run._list_conditions():
        lead_block, _, wish = str(clause).partition(":")
        head_words = _word_set(lead_block)
        if head_words and any(pager and (pager <= head_words or head_words <= pager) for pager in spellings):
            return True
        needed = _ejd_clean_want(wish)
        if needed and any(needed <= _word_set(field_value) for field_value in details_map.get("values") or []):
            return True
    return False


def _ejd_ranking(store_run: Episode) -> list[str]:
    """Askable menu properties in a fixed order drawn from the opened listings only, so reruns of one request see
    the same choices. Left out: asked, decided or variant lines. Sort: label-like and single-valued last; then
    wider coverage, more value groups, more values, name. A name holding or held by an asked name waits until no
    other is left: such repeats brought a need about half as often as a new property."""
    details_map = store_run.property_menu
    open_props = [p for p in details_map
                  if not _ejd_asked_before(store_run, p) and not _ejd_decided(store_run, p, details_map[p])
                  and not _ejd_variant_line(details_map[p])]

    def order_key(p: str):
        element = details_map[p]
        return (_ejd_label_like(p, element), element.get("uniform", False), -element["count"],
                -_ejd_value_groups(element), -len(element["values"]), p)

    ranked = sorted(open_props, key=order_key)
    fresh = [p for p in ranked if not _ejd_family_asked(store_run, p)]
    return fresh or ranked


def _ejd_family_asked(store_run: Episode, prop_name: str) -> bool:
    """The name, or an asked name, sits whole and in order inside the other (a word of four letters or more may
    open a longer one): "storage" in "storage type", "pack type" in "packaging type"."""
    own = _token_list(prop_name)
    for done in store_run.asked_props:
        theirs = _token_list(done)
        short, long_ = (own, theirs) if len(own) <= len(theirs) else (theirs, own)
        width = len(short)
        if width and any(all(_ejd_word_alike(short[k], long_[pos + k]) for k in range(width))
                         for pos in range(len(long_) - width + 1)):
            return True
    return False


def _ejd_label_like(prop_name: str, details_map: _Record) -> bool:
    """Values that read as labels instead of one option each: a slash joins options somewhere, or most values
    reuse a word of the property name. Hidden needs sat on such lines far less often, so they go last."""
    vals = details_map.get("values") or []
    if any("/" in fv for fv in vals):
        return True
    own = _name_stems(prop_name)
    self_named = sum(1 for fv in vals if own & _name_stems(fv)) if own else 0
    return self_named * 2 > len(vals)


def _ejd_value_groups(details_map: _Record) -> int:
    """Number of really different values: values linked by a shared word (unit marks too) count as one."""
    pools: list[set[str]] = []
    for field_value in details_map.get("values") or []:
        ws = set(_token_list(field_value))
        linked = [p for p in pools if p & ws]
        pools = [p for p in pools if not p & ws] + [ws.union(*linked)]
    return len(pools)


def _ejd_loose(wish: str, values: list[str]) -> bool:
    """A stated value that equals no listing value yet whose words fit two or more of them."""
    phrase = _flatten_text(re.sub(r"\(.*?\)", " ", wish))
    needed = _word_set(phrase) - _word_set("!=")
    if "!=" in wish or not needed:
        return False
    flat_values = {_flatten_text(fv) for fv in values}
    members = {_flatten_text(mt) for fv in values for mt in re.split(r"[,;/|]", str(fv)) if mt.strip()}
    if phrase in flat_values or phrase in members:
        return False   # the value is stated as it is, alone or as one element of a list
    return len([fv for fv in flat_values if needed <= _word_set(fv)]) >= 2


def _ejd_loose_props(store_run: Episode) -> dict[str, str]:
    """Menu property -> loose value, for requirements in force. Values pool over every menu name sharing a word
    with the requirement's property; only the best-covered such name is reported, one question per requirement."""
    found: dict[str, str] = {}
    menu = store_run.property_menu
    for clause in store_run._list_conditions():
        lead_block, _, wish = str(clause).partition(":")
        head_words = _word_set(lead_block)
        if not head_words:
            continue
        close_fits = [label_name for label_name in menu if head_words & _word_set(label_name)]
        if not close_fits:
            continue
        if _ejd_loose(wish, [fv for label_name in close_fits for fv in menu[label_name].get("values") or []]):
            found[max(close_fits, key=lambda label_name: (menu[label_name]["count"], -len(label_name)))] = wish.strip()
    return found


def _ejd_variant_line(details_map: _Record) -> bool:
    """A seller's variant line: a new value on each of three or more listings, mostly slash-joined options."""
    values = details_map.get("values") or []
    seen_on = details_map.get("count") or 0
    if seen_on < 3 or len(values) != seen_on:
        return False
    return 2 * len([fv for fv in values if "/" in fv]) > len(values)


def _ejd_ask_moment(store_run: Episode, sheets: int, forced: bool) -> bool:
    """Enough to ask about: two fact sets (one when forced), a stated must-ask with nothing asked yet, or a late
    step with two fact sets or a world that never shows details as fact sets."""
    if sheets >= (1 if forced else 2):
        return True
    if store_run.plan.get("ask_first") and not store_run.asked and not store_run.urgent_tried:
        return True
    if store_run.step < round(0.22 * store_run.max_steps):
        return False
    return sheets >= 2 or not store_run.roles.get("details")


def _stage_question_gate(session: Episode, carry: _Record) -> _Record:
    """May the planner ask now? When yes, keep the tool, the message cap and the fact menu on the record."""
    tool = session.roles.get("message")
    limit = session._message_limit()
    used = session.messages_used
    verdict = None   # None: go ahead; False: not now; True: not now and switch the planner off
    if not tool or used >= limit:
        verdict = True
    else:
        limit -= _ejd_reserved_messages(session)   # messages a stated procedure needs go first
        if used >= limit:
            verdict = False
    if verdict is None:
        menu, sheets = session._build_property_menu()
        if not _ejd_ask_moment(session, sheets, carry["force"]):
            verdict = False
        else:
            spare = session.remaining_steps() - _spare_steps_for(session.max_steps)
            if spare < limit - used + _ejd_FOLLOW_STEPS or session.seconds_remaining() < WRAP_UP_WINDOW + 40:
                verdict = True
    if verdict is None:
        carry.update(tool=tool, cap=limit, menu=menu, sheets=sheets)
        return carry
    if verdict:
        session.planner_on = False
    carry["exit"] = False
    return carry


def _ejd_sibling_pick(store_run: Episode, carry: _Record, sibling: str) -> _Record | None:
    """After a waved-off answer: ask the neighbouring property next, in the planner's words when it names it."""
    hint = (f'Target property for this question: "{sibling}" (call it by its own name, together with the product '
            f'noun). Reason: the reply about "{store_run.asked_props[-1]}" waved it away, and the two names overlap - '
            "that reply could have been meant here.")
    pick = _draft_plan_once(store_run, carry, hint)
    if pick is None or not _same_tokens(pick["property"], sibling):
        pick = {"property": sibling, "question": _ejd_plain(store_run, sibling), "repeat": False, "chance": None}
    return pick if _ejd_worth(store_run, pick) else None


def _ejd_regular_pick(store_run: Episode, carry: _Record) -> _Record | None:
    split = store_run.split_pending
    note = ""
    if split:
        note = f'Answer judged to mix several things. Asked property: "{split[0]}". Wording used: "{split[1]}".'
    choice = _draft_plan_once(store_run, carry, note)
    new_lines = choice is not None and not choice["repeat"]
    if new_lines and _ejd_names_sample(store_run, choice):
        choice["question"] = _ejd_plain(store_run, choice["property"])   # sample values only where a sibling key needs them
    if split and not _ejd_sharper(choice, split):
        # nothing sharper came back: count the property as asked and move on
        store_run.asked_props.append(split[0])
        store_run.split_pending = None
        if not new_lines:
            choice = _draft_plan_once(store_run, carry, "")
    if choice is None and store_run.reask_later:
        again = store_run.reask_later.pop(0)
        choice = _draft_plan_once(store_run, carry, f'One more try is allowed for "{again}", naming '
                                        "the item's own noun this time.", again)
    return choice if choice is not None and _ejd_worth(store_run, choice) else None


def _ejd_unclear_pick(store_run: Episode, prop_name: str) -> _Record | None:
    """The one re-ask after a reply that did not understand the question: our own plain question, which names the
    product; dropped (the property then counts as asked) when it would repeat the earlier wording."""
    question = _ejd_plain(store_run, prop_name)
    before = [ent for ent in store_run.asked if ent.startswith("[%s] " % prop_name)]
    if before and _token_list(question) == _token_list(before[-1].split('"')[1] if '"' in before[-1] else ""):
        store_run.asked_props.append(prop_name)
        return None
    pick = {"property": prop_name, "question": question, "repeat": True, "chance": None}
    return pick if _ejd_worth(store_run, pick) else None


def _stage_question_draft(session: Episode, carry: _Record) -> _Record:
    carry["picks"] = []
    sibling = session.sibling_next
    session.sibling_next = ""
    unclear = session.unclear_next
    session.unclear_next = ""
    pick = _ejd_unclear_pick(session, unclear) if unclear else None
    if pick is None and session.planner_on:
        if sibling and not session.split_pending and not _ejd_asked_before(session, sibling) and _ejd_turn_due(session, sibling):
            pick = _ejd_sibling_pick(session, carry, sibling)
            if pick is not None:
                session.sibling_props.append(pick["property"])
        else:
            pick = _ejd_regular_pick(session, carry)
    if pick is not None:
        carry["picks"] = [pick]
    return carry


def _ejd_candidate_block(store_run: Episode, order: list[str]) -> str:
    if store_run.asked_props:
        rule = "These are the candidates. Choosing outside them is allowed only with a reason in \"deviate\"."
    else:
        rule = ("These are the candidates, strongest first; the first question has to be among the top three. "
                "Choosing outside that is allowed only with a reason in \"deviate\".")
    text_lines = [rule]
    for cnt, prop_name in enumerate(order[:8], 1):
        text_lines.append(f"[{cnt}] {prop_name}")
    return "\n".join(text_lines)


def _ejd_planner_text(store_run: Episode, carry: _Record, extra: str, steer: list[str] | None, requirement_summary: str) -> str:
    musts = store_run.plan.get("ask_first")
    blocks = [QUESTION_PLAN_TEXT, "Request (exact text):\n" + str(store_run.policy.get("query") or ""),
              "Requirements in force now:\n" + requirement_summary]
    if musts:
        blocks.append("Must come from the shopper: " + "; ".join(map(str, musts)))
    if store_run.asked:
        blocks.append("Earlier questions with replies:\n" + "\n".join(store_run.asked))
    blocks.append("Detail-sheet properties so far, as name / number of listings / sample values:\n"
                  + (carry["menu"] or "(none yet)"))
    if steer:
        blocks.append(_ejd_candidate_block(store_run, steer))
    if extra:
        blocks.append(extra)
    return "\n\n".join(blocks)


def _ejd_first_pick(reply: _Record) -> _Record:
    many = reply.get("questions")
    if isinstance(many, list) and many and isinstance(many[0], dict):
        return many[0]
    return reply


def _ejd_text_of(raw_obj: _Record, key: str) -> str:
    return str(raw_obj.get(key) or "").strip()


def _ejd_keep_on_list(store_run: Episode, steer: list[str], raw_obj: _Record, prop_name: str, phrasing: str) -> tuple[str, str]:
    """The planner's choice when it is a candidate (the first question: one of the top three) or comes with a
    reason; otherwise the top candidate in plain words."""
    window = steer[:8] if store_run.asked_props else steer[:3]
    if any(_same_tokens(prop_name, kname) for kname in window):
        return prop_name, phrasing
    if _ejd_text_of(raw_obj, "deviate") and any(_same_tokens(prop_name, kname) for kname in steer):
        return prop_name, phrasing
    top = steer[0]
    plain = _ejd_plain(store_run, top)
    store_run.trace.append(dict(kind="ask_ranked", property=top, dropped=raw_obj.get("property")))
    return top, plain


def _draft_plan_once(session: Episode, carry: _Record, extra: str, again_for: str = "") -> _Record | None:
    """Run the planner once; its pick, or None when it offers nothing new."""
    requirement_summary = session.requirement_summary()
    ranked = _ejd_rotation(session, _ejd_ranking(session))
    steer = ranked if ranked and not again_for and not session.split_pending else None
    prompt = _ejd_planner_text(session, carry, extra, steer, requirement_summary)
    raw_obj = _ejd_first_pick(session.ask_judge(prompt, 2000, [READER_SEAT, REVIEW_SEAT]) or {})   # 800 tokens got cut off
    prop_name, phrasing = _ejd_text_of(raw_obj, "property"), _ejd_text_of(raw_obj, "question")
    if steer:
        prop_name, phrasing = _ejd_keep_on_list(session, steer, raw_obj, prop_name, phrasing)
    if not (prop_name and phrasing):
        return None
    phrasing = _ejd_one_product(session, prop_name, phrasing)
    redo = session.split_pending[0] if session.split_pending else again_for
    repeat = bool(redo) and _same_tokens(prop_name, redo)
    if again_for:
        if not (repeat and _word_set(session.plan.get("product")) & _word_set(phrasing)):
            return None   # a second try must name the item's noun
    elif not repeat and _ejd_asked_before(session, prop_name):
        return None
    odds = _as_number(raw_obj.get("chance"))
    if odds is not None:
        odds = min(1.0, max(0.0, odds))
    return dict(property=prop_name, question=phrasing, repeat=repeat, chance=odds)


def _ejd_word_alike(a: str, b: str) -> bool:
    """Name words for one thing: identical, or the shorter (4+ letters) begins the longer."""
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 4 and long_.startswith(short)


def _ejd_sibling(store_run: Episode, prop_name: str) -> str:
    """The unasked menu property whose name shares most words with prop (first word matching first word counts
    twice; words found in half the menu names or more do not count); "" when none shares any."""
    menu = list(store_run.property_menu)
    if len(menu) < 2:
        return ""
    split = {label_name: _token_list(label_name) for label_name in menu}
    own_row = _token_list(prop_name)
    products = _ejd_products(store_run)
    own_slot = _ejd_product_slot(store_run, prop_name, products) if len(products) >= 2 else None

    def too_common(term: str) -> bool:
        carriers = sum(1 for label_name in menu if any(_ejd_word_alike(term, el) for el in split[label_name]))
        return carriers * 2 >= len(menu)

    top_choice, top_score = "", 0
    for label_name in menu:
        if _same_tokens(label_name, prop_name) or _ejd_asked_before(store_run, label_name):
            continue
        if _ejd_family_asked(store_run, label_name) or _ejd_decided(store_run, label_name, store_run.property_menu[label_name]):
            continue   # a repeat of an asked name, or a property the request states
        if own_slot is not None and _ejd_product_slot(store_run, label_name, products) != own_slot:
            continue   # a reply about one product is never meant for another product's property
        score = 0
        for pos, term in enumerate(split[label_name]):
            if too_common(term) or not any(_ejd_word_alike(term, el) for el in own_row):
                continue
            score += 2 if pos == 0 and _ejd_word_alike(term, own_row[0]) else 1
        if not score:
            continue
        if score > top_score or (score == top_score
                                  and store_run.property_menu[label_name]["count"] > store_run.property_menu.get(top_choice, {}).get("count", 0)):
            top_choice, top_score = label_name, score
    return top_choice


def _ejd_spoken(store_run: Episode, prop_name: str) -> str:
    """The name used with the shopper: lone letters dropped; a lead word of at most three letters that also opens
    two or more other names on the listings is a key prefix, so it is dropped."""
    clean = " ".join(tk for tk in prop_name.split() if len(tk) > 1 or tk.isdigit())
    if clean and clean != prop_name:
        return _ejd_spoken(store_run, clean)   # a lone letter is what is left of a possessive in a key
    opener = _ejd_LEAD_WORD.match(prop_name)
    if opener is None:
        return prop_name
    shorter = prop_name[opener.end():].strip()
    if len(opener.group(1)) > 3 and len(_token_list(shorter)) < 2:
        return prop_name
    every = [cnt for kname, element in store_run.property_menu.items() for cnt in [kname, *(element.get("also") or [])]]
    if any(_same_tokens(shorter, cnt) for cnt in every):
        return prop_name   # the shorter form is some other property's name
    prefix = opener.group(1).lower()
    starts = {tuple(_token_list(cnt)) for cnt in every if not _same_tokens(cnt, prop_name)}
    return shorter if len([tk for tk in starts if len(tk) >= 2 and tk[0] == prefix]) >= 2 else prop_name


def _ejd_products(store_run: Episode) -> list[str]:
    """The request's products when it asks for several: the setup reading's product lines, else the plan's product
    phrase split at its list marks."""
    lined = [item["product"] for item in getattr(store_run, "_wanted_items", list)()]
    if lined:
        return lined
    parts = [seg.strip() for seg in re.split(r"[;,]", str(store_run.plan.get("product") or "")) if seg.strip()]
    return parts if len(parts) >= 2 else []


def _ejd_own_words(segment: str, products: list[str]) -> set[str]:
    """Words of one product that no other product of the request uses."""
    others = set().union(*[_word_set(seg) for seg in products if seg != segment])
    return _word_set(segment) - others


def _ejd_owner(store_run: Episode, prop_name: str) -> str:
    """In a request for several products: the one whose words the listings carrying the property show most, in
    the request's words (leading joiners the listings never show trimmed); "" for a single product or no match."""
    products = _ejd_products(store_run)
    if not products:
        return ""
    names = {prop_name.strip().lower(), *[al.lower() for al in (store_run.property_menu.get(prop_name) or {}).get("also") or []]}
    items = getattr(store_run, "_wanted_items", list)()
    if items:   # the requested item the listings carrying the property belong to, by their own texts
        votes: dict[int, int] = {}
        for record_blob in _iter_dicts(store_run.records):
            listing = next((cell for cell in record_blob.values() if isinstance(cell, str) and cell in store_run.seen_ids), None)
            if listing is None:
                continue
            keys = {store_run._ejb_label(kname).strip().lower() for part in _iter_dicts([record_blob]) for kname in part}
            index = store_run._ejc_item_index(listing) if names & keys else None
            if index is not None:
                votes[index] = votes.get(index, 0) + 1
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            return items[ranked[0][0]]["product"]
        return ""
    pool: set[str] = set()
    for sheet in store_run._ejb_menu_sheets():
        if names & set(sheet):
            for cell in sheet.values():
                pool |= _word_set(cell)
    scored = [(len(_ejd_own_words(seg, products) & pool), -pos, seg) for pos, seg in enumerate(products)]
    top = max(scored)
    if not top[0]:
        return ""
    words = top[2].split()
    while len(words) > 1 and _word_set(words[0]) and not _word_set(words[0]) & pool:
        words.pop(0)
    return " ".join(words)


def _ejd_one_product(store_run: Episode, prop_name: str, phrasing: str) -> str:
    """In a request for several products the question names exactly one, the property's own; otherwise our own
    plain question takes its place."""
    products = _ejd_products(store_run)
    owner = _ejd_owner(store_run, prop_name) if products else ""
    if not owner:
        return phrasing
    said = _word_set(phrasing)
    named = [seg for seg in products if _ejd_own_words(seg, products) & said]
    if len(named) == 1 and _word_set(owner) & said:
        return phrasing
    return _ejd_plain(store_run, prop_name)


def _ejd_product_slot(store_run: Episode, prop_name: str, products: list[str]) -> int | None:
    """Position of the property's product among the request's products; None when no single one owns it."""
    said = _ejd_owner(store_run, prop_name)
    if said in products:
        return products.index(said)
    owner = _word_set(said)
    return next((pos for pos, seg in enumerate(products) if owner and owner <= _word_set(seg)), None)


def _ejd_least_asked(store_run: Episode, products: list[str]) -> set[int] | None:
    """In a request for several products, once one has been asked about: the products asked least so far; None
    while every product has had as many questions as the others."""
    if len(products) < 2 or not store_run.asked_props:
        return None
    asks = [0] * len(products)
    for done in store_run.asked_props:
        slot = _ejd_product_slot(store_run, done, products)
        if slot is not None:
            asks[slot] += 1
    fewest = min(asks)
    least = {pos for pos, n in enumerate(asks) if n == fewest}
    return None if len(least) == len(products) else least


def _ejd_turn_due(store_run: Episode, prop_name: str) -> bool:
    """The property's product is one of those asked least (always true for one product or an unknown owner)."""
    products = _ejd_products(store_run)
    least = _ejd_least_asked(store_run, products)
    slot = _ejd_product_slot(store_run, prop_name, products) if least is not None else None
    return slot is None or slot in least


def _ejd_rotation(store_run: Episode, ranked: list[str]) -> list[str]:
    """In a request for several products the next question goes to a product asked least so far: its candidates
    stay, in rank order. Such requests usually hide needs on two products, and in the
    race worlds revealing both paid about twice as often as revealing one. All candidates when none is left."""
    products = _ejd_products(store_run)
    least = _ejd_least_asked(store_run, products)
    if least is None:
        return ranked
    kept = [p for p in ranked if _ejd_product_slot(store_run, p, products) in least]
    return kept or ranked


def _ejd_names_product(store_run: Episode, prop_name: str, question: str) -> bool:
    """The question names the property's product (in a request for several), else the requested product, by a
    word of four letters or more, or by the first or last word of the product's name ("ram" in "ram kits")."""
    noun = _ejd_owner(store_run, prop_name) or str(store_run.plan.get("product") or "")
    words = [tk for tk in _token_list(noun) if len(tk) > 2]
    edges = _name_stems(" ".join(words[:1] + words[-1:]))
    return bool(({tk for tk in _name_stems(noun) if len(tk) > 3} | edges) & _name_stems(question))


def _ejd_plain(store_run: Episode, prop_name: str) -> str:
    """Our own one-property question; the item noun is added when the product name is short, and in a request for
    several products the one product the property belongs to."""
    said = _ejd_spoken(store_run, prop_name)
    owner = _ejd_owner(store_run, prop_name)
    if owner:
        return f"Do you have a preference for the {said} of the {owner}?"
    noun = str(store_run.plan.get("product") or "").strip()
    if not noun or len(noun.split()) > 2:
        return f"Do you have a preference for the {said}?"
    return f"Do you have a preference for the {said} of the {noun}?"


def _ejd_names_sample(store_run: Episode, pick: _Record) -> bool:
    """The question quotes a sample value of its property, while no longer menu name contains the property's
    words (only then do values tell it apart from a neighbour)."""
    menu, prop_name = store_run.property_menu, pick["property"]
    element = menu.get(prop_name) or next((menu[kname] for kname in menu if _same_tokens(kname, prop_name)), None)
    own = _word_set(prop_name)
    if not element or any(own < _word_set(kname) for kname in menu):
        return False
    spoken = _word_set(pick["question"]) - own - _word_set(store_run.plan.get("product"))
    wished = _word_set((getattr(store_run, "policy", None) or {}).get("query"))   # values the request names are fine
    samples = [_word_set(field_value) for field_value in element["values"]]
    return any(vw and vw <= spoken and not vw <= wished for vw in samples)


def _ejd_sharper(pick: _Record | None, pending: tuple[str, str]) -> bool:
    """A follow-up to a split answer: same property, different wording from the question that split."""
    if pick is None or not pick["repeat"] or not _same_tokens(pick["property"], pending[0]):
        return False
    return _token_list(pick["question"]) != _token_list(pending[1])


def _ejd_worth(store_run: Episode, pick: _Record) -> bool:
    """With a price on questions, ask only while the expected gain beats the share it costs; a money-only price
    allows a single question. A no also switches the planner off."""
    price = store_run.question_cost
    if not price:
        return True
    share = _as_number(price.get("share"))
    if share is None or not 0 <= share <= 1:
        allowed = not store_run.asked_props and not store_run.split_pending
    else:
        odds = 0.5 if pick["chance"] is None else pick["chance"]
        allowed = odds * _ejd_REVEAL_ODDS > share
    if not allowed:
        store_run.planner_on = False
    return allowed


def _stage_question_build(session: Episode, carry: _Record) -> _Record:
    ready = []
    for pick in carry["picks"]:
        text = str(pick.get("question")).strip()
        msg = session._build_message_call(text, "ask_%d_%d" % (session.messages_used + 1, session.step))
        if msg is not None:
            ready.append((msg, (str(pick.get("property")), text, bool(pick.get("repeat")))))
    if ready:
        carry["calls"] = [msg for msg, _ in ready]
        carry["topics"] = [topic for _, topic in ready]
        return carry
    # nothing to send: with under two fact sets open there is no menu yet, so wait for one rather than
    # switching questions off for good
    session.empty_plans += 1
    session.urgent_tried = True
    if carry.get("sheets", 0) >= 2 or session.empty_plans >= 3:
        session.planner_on = False
    carry["exit"] = False
    return carry


_ejd_ANSWER_NOTE = ("\n[agent note] Requirement list updated with the shopper's reply, kept in their wording. "
                    "Suggested next move: a search pairing that value with the product type; afterwards re-verify "
                    "the current pick and each opened listing against it.")


def _ejd_log_and_file(store_run: Episode, subject: tuple[str, str, bool], text: str) -> None:
    """Keep the exchange for the planner, then file the property by the class the reader gave the answer."""
    prop_name, question, repeat = subject
    store_run.asked.append('[%s] "%s" => %s' % (prop_name, question, text[:400]))
    heard = (store_run.answer_reading or {}).get("class", "other")
    _ejd_file_answer(store_run, prop_name, question, repeat, heard)


def _ejd_flag_new_requirement(store_run: Episode) -> None:
    """Tell the model, on the last tool reply, that the answer became a requirement."""
    last = store_run.messages[-1] if store_run.messages else {}
    if last.get("role") == "tool":
        last["content"] += _ejd_ANSWER_NOTE


def _stage_question_send(session: Episode, carry: _Record) -> _Record:
    outgoing = carry["calls"]
    subjects = carry["topics"]
    version_before = session.needs_revision
    note = "Asking the shopper (planned) about %s." % "; ".join(tk[0] for tk in subjects)
    session.messages.append(dict(role="assistant", content=note, tool_calls=outgoing))
    session.asking, session.answer_reading = subjects[0][0], None
    texts = session._record_replies(session._run_calls(outgoing, planned=True))
    for subject, text in zip(subjects, texts):
        _ejd_log_and_file(session, subject, text)
    session.asking = ""
    if session.needs_revision != version_before:
        _ejd_flag_new_requirement(session)
    session.planner_on = carry["cap"] > session.messages_used
    carry["exit"] = True
    return carry


def _ejd_file_answer(store_run: Episode, prop_name: str, text: str, repeat: bool, heard: str) -> None:
    """Close, reopen or park the property by how its answer was classed."""
    if repeat:
        store_run.split_retried = True
        store_run.split_pending = None
    elif heard == "split":
        store_run.split_pending = (prop_name, text)   # one narrower re-ask of this property
        return
    elif heard == "unclear":
        if not _ejd_names_product(store_run, prop_name, text):
            store_run.unclear_next = prop_name   # one re-ask naming the product; the property stays open
            return
        # the product was named already: a re-ask in new words rarely lands, so other properties go first and
        # this one gets its one late try only when none is left
        if prop_name not in store_run.reask_later:
            store_run.reask_later.append(prop_name)
    elif heard == "not_picky":
        wish = _ejd_loose_props(store_run).get(prop_name)
        if wish and _ejd_clean_want(wish) <= _word_set(text):
            store_run.loose_ok[prop_name] = wish   # asked which listed value was meant; any of them will do
        # one neighbour per asked property: a waved-off neighbour does not open a neighbour of its own
        store_run.sibling_next = "" if prop_name in store_run.sibling_props else _ejd_sibling(store_run, prop_name)
        spoken, label_name = _word_set(text), _word_set(prop_name)
        anchored = bool(spoken & _word_set(store_run.plan.get("product"))) or (len(label_name) >= 2 and label_name <= spoken)
        if not anchored and prop_name not in store_run.reask_later:
            store_run.reask_later.append(prop_name)   # a lone word may have landed on a neighbouring property
    if not _ejd_asked_before(store_run, prop_name):
        store_run.asked_props.append(prop_name)


# ================================================================ call screening (each gate: None or a reply text)

def _gate_unknown_tool(session: Episode, gate_ctx: _Record, unit_ref: _Record) -> str | None:
    if unit_ref["name"] in session.param_schemas:
        return None
    tools = ", ".join(sorted(gate_ctx["allowed"]))
    return f"{_SKIP_NOTICE}: unknown tool '{unit_ref['name']}'. Callable now: {tools}."


def _gate_not_offered(session: Episode, gate_ctx: _Record, unit_ref: _Record) -> str | None:
    if gate_ctx["planned"] or unit_ref["name"] in gate_ctx["allowed"]:
        return None
    if session.planner_on and unit_ref["name"] == gate_ctx["message"]:
        return (f"{_SKIP_NOTICE}: no need to message - questions to the shopper go out automatically and replies are "
                "delivered unasked. Use this turn on the listings: open details, cart a good fit, then place the "
                "order.")
    return _SKIP_NOTICE + ": near the step limit, what is left is " + ", ".join(sorted(gate_ctx["allowed"])) + "."


def _gate_bad_json(session: Episode, gate_ctx: _Record, unit_ref: _Record) -> str | None:
    raw_obj = unit_ref["raw"]
    if isinstance(raw_obj, dict):
        parsed = raw_obj
    elif (raw_obj or "").strip():
        parsed = _extract_json_dict(raw_obj)
    else:
        parsed = {}
    unit_ref["args"] = parsed
    if parsed is None:
        return _SKIP_NOTICE + ": unreadable arguments (a single JSON object is expected)."
    return None


def _gate_schema_misfit(session: Episode, gate_ctx: _Record, unit_ref: _Record) -> str | None:
    unit_ref["args"], issues = _validate_value(unit_ref["args"], session.param_schemas[unit_ref["name"]], "args")
    if not issues:
        return None
    return _SKIP_NOTICE + ": argument problems against the tool schema: " + "; ".join(issues[:6])


def _gate_finish_alone(session: Episode, gate_ctx: _Record, unit_ref: _Record) -> str | None:
    if gate_ctx["count"] <= 1 or unit_ref["name"] not in gate_ctx["finishing"]:
        return None
    return (_SKIP_NOTICE + ": finishing calls (order or decline) travel alone. Wait for the other results, then send it "
            "in a reply of its own.")


def _gate_known_fault(session: Episode, gate_ctx: _Record, unit_ref: _Record) -> str | None:
    label_name, arguments = unit_ref["name"], unit_ref["args"]
    if not arguments or label_name in gate_ctx["finishing"]:
        return None
    if session._call_shape(label_name, arguments) not in session.faulty and label_name + _to_compact_json(arguments) not in session.faulty:
        return None
    return (f"{_SKIP_NOTICE}: {label_name} returned one identical error for two calls of this form; send arguments that obey "
            "its schema exactly, or turn to a different tool.")


def _gate_tool_quota(session: Episode, gate_ctx: _Record, unit_ref: _Record) -> str | None:
    limit = session.message_budget
    if limit is None or unit_ref["name"] != gate_ctx["message"]:
        return None
    if session.messages_used + gate_ctx["queued_messages"] < limit:
        return None
    return _SKIP_NOTICE + ": message allowance exhausted - no more messages to the shopper."


def _gate_repeat_call(session: Episode, gate_ctx: _Record, unit_ref: _Record) -> str | None:
    label_name = unit_ref["name"]
    unit_ref["sig"] = label_name + _to_compact_json(unit_ref["args"])
    if label_name in (gate_ctx["message"], gate_ctx["cart"]) or label_name in gate_ctx["finishing"] or unit_ref["sig"] not in session.recent:
        return None
    return (_SKIP_NOTICE + ": duplicate of your previous call. Nothing changed and the earlier result is still above; "
            "vary the arguments or move on.")


def _gate_late_details(session: Episode, gate_ctx: _Record, unit_ref: _Record) -> str | None:
    """Late after a refusal, details are offered only for cheaper seen listings not opened yet (see
    _tools_this_turn); opening a different listing then is skipped, with the allowed ones named."""
    late_retry = unit_ref["name"] == session.roles.get("details") and session.remaining_steps() <= 7 and session.refusals
    if not late_retry:
        return None
    allowed = _rows_to_recheck(session)
    asked_for = [fv for fv in unit_ref["args"].values() if isinstance(fv, str)]
    if not allowed or any(fv in allowed for fv in asked_for):
        return None
    return (_SKIP_NOTICE + ": so near the end, only cheaper never-opened listings may get a details read, namely "
            + ", ".join(allowed) + ". The item already held? Its price and stock are in the cart view.")


def _gate_late_unfitted(session: Episode, gate_ctx: _Record, unit_ref: _Record) -> str | None:
    """Late with no order tried, details stay offered only while a product has no fit (see _ejb_allowed_jobs):
    then just for its unopened on-target rows, one read at a time down to the reserve plus two."""
    tool = session.roles.get("details")
    remaining = session.remaining_steps()
    narrowed = session.roles.get("order") and session.roles.get("cart") and not session.order_attempts
    if unit_ref["name"] != tool or remaining > 7 or session.refusals or not narrowed:
        return None
    queued = len([1 for _, name, _ in gate_ctx.get("ready") or [] if name == tool])
    allowed = _eje_unfitted_rows(session, 2)
    asked_for = [fv for fv in unit_ref["args"].values() if isinstance(fv, str)]
    if allowed and remaining - queued >= session._reserved_steps() + 2 and any(fv in allowed for fv in asked_for):
        return None
    if not allowed:
        return _SKIP_NOTICE + ": " + _format_note("reserve_late")
    return (_SKIP_NOTICE + ": so near the end, one more details read is left, for a product with no fit yet, "
            "namely one of " + ", ".join(allowed) + ".")


def _ejd_cart_target(store_run, label_name: str, arguments: _Record) -> str | None:
    """The listing a cart call adds (see _cart_listing_id)."""
    return _cart_listing_id(store_run, label_name, arguments)


def _gate_cart_review(session: Episode, gate_ctx: _Record, unit_ref: _Record) -> str | None:
    tool, call_args = unit_ref["name"], unit_ref["args"]
    if tool != gate_ctx["cart"] or session.remaining_steps() <= 3:
        return None
    sig = session.add_signature
    if sig is not None and sig != session._choice_signature(tool, call_args):
        return None   # not an add
    target = _ejd_cart_target(session, tool, call_args)
    tag = "%s|%s" % (target, session._arm_signature()) if target else ""
    if not tag or tag in session.cart_checked:
        return None
    session.cart_checked.add(tag)
    doubt = session._verify_item(target, "being added to the cart")
    if doubt:
        return _SKIP_NOTICE + ". " + doubt + " Sure it fits anyway? Then send it again as is."
    return None


def _gate_unit_count(session: Episode, gate_ctx: _Record, unit_ref: _Record) -> str | None:
    if unit_ref["name"] != gate_ctx["cart"]:
        return None
    issue = session._units_problem(unit_ref["name"], unit_ref["args"])
    if not issue or unit_ref["sig"] in session.qty_warned:
        return None
    session.qty_warned.add(unit_ref["sig"])
    return f"{_SKIP_NOTICE}: {issue} Sure anyway? Then send it again as is."


def _gate_stacked_add(session: Episode, gate_ctx: _Record, unit_ref: _Record) -> str | None:
    """A second add of a listing the cart holds may stack units: hold it once per cart state, giving the count the
    cart shows and the count the request wants."""
    tool, call_args = unit_ref["name"], unit_ref["args"]
    if tool != gate_ctx["cart"] or session.add_signature is None:
        return None
    if session._choice_signature(tool, call_args) != session.add_signature:
        return None
    in_cart = [fv for fv in call_args.values() if isinstance(fv, str) and fv in session.cart_ids]
    if not in_cart:
        return None
    held, needed = in_cart[0], session._wanted_units()
    if session._wanted_items():
        needed = session._ejc_line_units(held) or max(el["quantity"] for el in session._wanted_items())
    units = None
    if session.cart_seen_at >= session.cart_changed_at:   # the cart view is newer than the last change
        units = session.units_in_cart.get(held)
    if units is not None and units + session._requested_units(tool, call_args) <= needed:
        return None   # still short of the wanted count
    tag = "readd %s %s" % (held, session.cart_changed_at)
    if tag in session.held_arms:
        return None
    session.held_arms.add(tag)
    shown = "" if units is None else " (%s per the latest cart view)" % units
    return ("%s Careful, %s is in the cart already%s. Putting it in again can push the unit count past %s, which "
            "is what the request wants. Order it as it stands, or first remove it if the count must change. Only if "
            "extra units are truly meant, send it again as is." % (_HELD_NOTICE, held, shown, needed))


def _gate_reread_after_refusal(session: Episode, gate_ctx: _Record, unit_ref: _Record) -> str | None:
    """After a refusal, a new order waits until the held item's price and stock have been read once more."""
    if session._classify_call(unit_ref["name"], unit_ref["args"]) != "order":
        return None
    if not (session.refusals and session.cart_ids and session._has_step_room(1)):
        return None
    since = max(session.refused_at, session.cart_changed_at)
    stale = [idx for idx in (session.cart_ids if session._wanted_items() else session.cart_ids[-1:])
             if session.mentioned.get(idx, -1) <= since]
    tag = "reread %s %s" % (session.refused_at, session.cart_changed_at)
    if not stale or tag in session.held_arms:
        return None
    held = ", ".join(stale)
    session.held_arms.add(tag)
    return ("%s Order paused: no re-check of %s (price, stock, units in cart) has happened since the refusal or "
            "the newest cart change. Look it up with a tool that shows those - the cart view lists units - and then "
            "order. Send it again as is if that read just happened." % (_HELD_NOTICE, held))


# ================================================================ price rows and paged searches

def _find_cheaper_rows(session, pick: str, row_map: dict[str, _Record], unopened_only: bool,
                     ties: bool = False) -> list[str]:
    ref = row_map.get(pick) or {}
    top, label = ref.get("price"), ref.get("label")
    if top is None:
        return []

    def under(price) -> bool:
        return price <= top if ties else price < top

    hits = []
    for rid, listing_row in row_map.items():
        if rid == pick or listing_row["price"] is None or not under(listing_row["price"]):
            continue
        if not session._within_price_cap(listing_row) or (label is not None and listing_row["label"] != label):
            continue
        if session._row_passes_literal(listing_row) is False or (unopened_only and listing_row["opened"]):
            continue
        hits.append((listing_row["price"], rid))
    hits.sort()
    return [rid for _, rid in hits]


def _rows_to_recheck(session, row_map: dict[str, _Record] | None = None) -> list[str]:
    """After a refused order on a cheapest request: unopened in-budget rows of the held listing's section that
    cost no more than its latest price, cheapest first, as many as the steps allow (a price rise or a lost
    listing may have made one of them the cheapest fit, and only an opened one can be checked)."""
    target_value = (getattr(session, "plan", None) or {}).get("priority")
    held = session.cart_ids[-1] if session.cart_ids else None
    # the reserve is kept for exactly this (one refused finish, then a re-read): open, swap and order need four
    if not session.refusals or held is None or target_value != "cheapest" or session.remaining_steps() < 4:
        return []
    spare = session.remaining_steps() - 3
    found = _find_cheaper_rows(session, held, session._row_table() if row_map is None else row_map, True, True)
    return found[:max(1, min(4, spare))]


def _find_paging_args(session, tool: str) -> tuple[str | None, int | None]:
    """The search tool's paging argument: an optional integer with a maximum (first row to show)."""
    spec_of = session.param_schemas.get(tool) or {}
    must = set(spec_of.get("required") or [])
    for key, tool_desc in (spec_of.get("properties") or {}).items():
        paging = isinstance(tool_desc, dict) and key not in must and tool_desc.get("type") == "integer"
        if paging and "maximum" in tool_desc:
            return key, int(tool_desc["maximum"])
    return None, None


class _ejd_PagedSearch:
    """One search, keyed by its arguments minus the paging one, with what its pages have shown so far."""

    def __init__(self, query: _Record):
        self.query = query
        self.total = 0
        self.seen: set[int] = set()
        self.size = 0
        self.in_section = False
        self.lead_block = None
        self.holes: list[int] = []

    def take(self, start: int, shown: list[_Record], totals: list[int], label: str | None) -> None:
        self.total = max([self.total] + totals)
        self.seen.update(range(start, start + len(shown)))
        self.size = max(self.size, len(shown))
        if self.lead_block is None and start == 0:
            self.lead_block = tuple(json.dumps(rw, sort_keys=True) for rw in shown)
        if label is None or any(label in [fv for fv in rw.values() if isinstance(fv, str)] for rw in shown):
            self.in_section = True

    def term_set(self) -> set[str]:
        return _word_set(" ".join(str(fv) for fv in self.query.values()))

    def key(self) -> str:
        return json.dumps(self.query, sort_keys=True)


def _ejd_search_pages(store_run, tool: str, offset: str):
    """Yield (query without paging, first row index, inner result) for each recorded call of the search tool."""
    for element in store_run.trace:
        reply = element.get("reply")
        if element.get("kind") != "env" or not isinstance(reply, dict):
            continue
        first = (element.get("actions") or [{}])[0]
        if first.get("name") != tool:
            continue
        query = dict(first.get("args") or {})
        start = query.pop(offset, 0)
        body = _unwrap_text(reply.get("observation"))[1]
        if isinstance(body, dict) and isinstance(start, int):
            yield query, start, body


def _ejd_collect_searches(store_run, tool: str, offset: str, label: str | None) -> list[_ejd_PagedSearch]:
    by_query: dict[str, _ejd_PagedSearch] = {}
    for query, start, body in _ejd_search_pages(store_run, tool, offset):
        tables = [fv for fv in body.values() if isinstance(fv, list) and fv and all(isinstance(rw, dict) for rw in fv)]
        shown = max(tables, key=len, default=[])
        totals = [fv for fv in body.values() if isinstance(fv, int) and not isinstance(fv, bool) and fv >= len(shown)]
        if shown and totals:
            sig = json.dumps(query, sort_keys=True)
            if sig not in by_query:
                by_query[sig] = _ejd_PagedSearch(query)
            by_query[sig].take(start, shown, totals, label)
    return list(by_query.values())


def _ejd_search_vocab(store_run) -> tuple[set[str], set[str]]:
    """Words of the wanted values, and every word a search may type and stay on target: those, the product, and
    the store's own wording of a required value ("square" written as "square neck")."""
    try:
        conds = store_run._list_conditions()
    except AttributeError:   # the request has not been read yet
        conds = []
    wanted = set().union(*(_word_set(str(ch).split(":", 1)[-1]) for ch in conds))
    vocab = wanted | _word_set((getattr(store_run, "plan", None) or {}).get("product"))
    try:
        conds, row_index = store_run._list_conditions(), store_run._row_table()
    except AttributeError:
        conds, row_index = [], {}
    pairs = [(_word_set(p), _word_set(w)) for p, _, w in (str(ch).partition(":") for ch in conds)]
    for listing_row in row_index.values():
        facts = (listing_row.get("facts") or {}).items()
        for needed, field_value in pairs:
            if field_value:
                for key, text in facts:
                    if (needed <= _word_set(key) or _key_names_property(key, needed)) and field_value & _word_set(text):
                        vocab |= _word_set(text)
    return wanted, vocab


def _partial_search_views(session, label: str | None, singular: bool = False) -> list[tuple[int, _Record, int, int]]:
    """Searches whose matching rows were not all shown: (total, args of the next page, rows seen, pages left).
    The total is the highest count in the reply that covers its page; reach stops one page past the paging
    maximum; only views that showed a row of the section count; the next page starts at the first unseen row.
    Searches with identical first pages are one view (fewest words kept). A search typing words beyond the wanted
    values, the product and the store's wording of them is off target and offered only when no on-target one is
    left; with `singular`, a plural is compared as its singular."""
    tool = session.roles.get("search")
    if not tool:
        return []
    offset, limit = _find_paging_args(session, tool)
    if offset is None:
        return []
    searches = _ejd_collect_searches(session, tool, offset, label)
    wanted, vocab = _ejd_search_vocab(session)
    tool_desc = (session.param_schemas.get(tool) or {}).get("properties") or {}

    def typed(pager: _ejd_PagedSearch) -> set[str]:   # free-text words only; choice values are not typed
        return _word_set(" ".join(str(fv) for kname, fv in pager.query.items()
                                  if isinstance(fv, str) and not (tool_desc.get(kname) or {}).get("enum")))

    pending = []
    for pager in searches:
        pager.holes = [idx for idx in range(min(pager.total, limit + pager.size)) if idx not in pager.seen]
        if pager.holes and pager.in_section and pager.size:
            pending.append(pager)
    pending.sort(key=lambda pager: (len(pager.term_set()), pager.key()))
    unique: list[_ejd_PagedSearch] = []
    for pager in pending:
        if not any(u.lead_block and u.lead_block == pager.lead_block and u.total == pager.total for u in unique):
            unique.append(pager)
    if singular:
        roots = _name_stems(" ".join(vocab))
        aimed = [pager for pager in unique if typed(pager) and _name_stems(" ".join(typed(pager))) <= roots]
    else:
        aimed = [pager for pager in unique if typed(pager) and typed(pager) <= vocab]
    if wanted and aimed:   # plain browsing views stay; off-target typed searches step aside
        unique = [pager for pager in unique if pager in aimed or not typed(pager)]
    scored = []
    for pager in unique:
        step_on = dict(pager.query)
        step_on[offset] = pager.holes[0]
        remaining = (len(pager.holes) + pager.size - 1) // pager.size
        scored.append((-len(wanted & pager.term_set()), pager.total, json.dumps(step_on, sort_keys=True),
                       (pager.total, step_on, len(pager.seen), remaining)))
    scored.sort(key=lambda tk: tk[:3])   # searches naming more wanted values first
    return [tk[3] for tk in scored]

import enum


# ---------------------------------------------------------------- search history readers

def _eje_search_calls(agent):
    """Yield (trace entry, arguments) for each recorded call of the search tool."""
    tool = agent.roles.get("search")
    if not tool:
        return
    for element in agent.trace:
        if element.get("kind") != "env":
            continue
        dispatched = (element.get("actions") or [{}])[0]
        if dispatched.get("name") == tool:
            yield element, dispatched.get("args") or {}


def _eje_search_pages(agent):
    """Yield (arguments, inner result, row list) for each search call whose reply was a dict result.
    The row list is the longest list of dicts in the result (empty when there is none)."""
    for element, sent_args in _eje_search_calls(agent):
        reply = element.get("reply")
        if not isinstance(reply, dict):
            continue
        body = _unwrap_text(reply.get("observation"))[1]
        if not isinstance(body, dict):
            continue
        tables = [fv for fv in body.values() if isinstance(fv, list) and fv and all(isinstance(rw, dict) for rw in fv)]
        yield sent_args, body, max(tables, key=len, default=[])


def _eje_bare_values(agent, conditions=None) -> list[str]:
    """Required values as the request words them, ready to search on: text before any bracket, comma, semicolon
    or exclusion; number relations are left out because they are checked, not searched. `conditions` narrows the
    lines read (all requirements in force by default)."""
    try:
        conditions = agent._list_conditions() if conditions is None else conditions
    except AttributeError:   # the request has not been read yet
        return []
    selected = []
    for condition in conditions:
        constraint = str(condition).partition(":")[2]
        if _parse_relation(constraint) is None:
            lead_block = re.split(r"[(,;]|!=", constraint, maxsplit=1)[0]
            selected.append(lead_block.strip().strip("\"'."))
    return list(dict.fromkeys(fv for fv in selected if fv and len(fv) <= 33))


def _eje_searched_bare(agent, field_value: str) -> bool:
    """True once some search typed nothing beyond the value's own words."""
    own = _word_set(field_value)
    for _, sent_args in _eje_search_calls(agent):
        typed_sets = (_word_set(tk) for tk in sent_args.values() if isinstance(tk, str))
        if any(typed and typed <= own for typed in typed_sets):
            return True
    return False


def _eje_typed_words(sent_args, paging, props) -> set[str]:
    """Words of the free-text arguments of one search (enum choices and the paging argument excluded)."""
    term_set: set[str] = set()
    for key, text in sent_args.items():
        tool_desc = props.get(key)
        if key != paging and isinstance(text, str) and not (isinstance(tool_desc, dict) and "enum" in tool_desc):
            term_set |= _word_set(text)
    return term_set


def _eje_seen_sources(agent) -> dict[str, tuple[int, int]]:
    """Map each listing id a search showed to (the most required values typed by any search that showed it, first
    position it held in a reply)."""
    tool = agent.roles.get("search")
    paging = _find_paging_args(agent, tool)[0] if tool else None
    props = ((agent.param_schemas.get(tool) or {}).get("properties") or {}) if tool else {}
    wanted = [w for w in map(_word_set, _eje_bare_values(agent)) if w]
    top_choice: dict[str, tuple[int, int]] = {}
    for sent_args, _, row_map in _eje_search_pages(agent):
        base = sent_args.get(paging, 0) if paging else 0
        if isinstance(base, bool) or not isinstance(base, int):
            base = 0
        typed = _eje_typed_words(sent_args, paging, props)
        score = len([w for w in wanted if w <= typed])
        for position, listing_row in enumerate(row_map, base):
            ident = next((fv for kname, fv in listing_row.items()
                          if isinstance(fv, str) and "id" in str(kname).lower() and _ID_SHAPE.match(fv)), None)
            if ident is not None:
                old_score, old_pos = top_choice.get(ident, (0, position))
                top_choice[ident] = (max(old_score, score), min(old_pos, position))
    return top_choice


def _eje_idle_part(key, own_row, theirs, here, there):
    """Compare one differing argument of two searches that gave identical rows.
    Returns (unused as a whole, tag, wording, its words, the call holding it) or None."""
    if own_row is None or theirs is None:
        field_value, owner = (theirs, there) if own_row is None else (own_row, here)
        return True, f"{key} {field_value}", f'{key} "{field_value}"', _word_set(field_value), owner
    if not (isinstance(own_row, str) and isinstance(theirs, str)):
        return None
    owner = here
    if _word_set(own_row) < _word_set(theirs):   # let `mine` be the wider one
        owner, own_row, theirs = there, theirs, own_row
    big, small = _word_set(own_row), _word_set(theirs)
    if small < big:
        extra = big - small
        quoted = '"' + " ".join(w for w in re.findall(r"[a-z0-9]+", own_row.lower()) if w in extra) + '"'
        return False, quoted, quoted, extra, owner
    # different words, identical rows: the store ignored the argument itself
    return True, key, '%s (both "%s" and "%s" returned identical rows)' % (key, own_row, theirs), big ^ small, owner


def _eje_idle_parts(agent) -> list[str]:
    """Search arguments the store ignored: two search calls differing in one non-paging argument that returned the
    same total and rows, where some row lacks the argument's words. Wholly ignored arguments come first, newest
    first within each group, one note per tag."""
    tool = agent.roles.get("search")
    paging = _find_paging_args(agent, tool)[0] if tool else None
    history: list[tuple[dict[str, Any], str]] = []
    groups: dict[bool, list[str]] = {True: [], False: []}
    tag_names: set[str] = set()
    for raw_args, body, row_map in _eje_search_pages(agent):
        if not row_map:
            continue
        here = dict(raw_args)
        totals = [fv for fv in body.values() if isinstance(fv, int) and not isinstance(fv, bool) and fv >= len(row_map)]
        stamp = json.dumps([max(totals, default=None), row_map], sort_keys=True)
        row_stems = [_name_stems(" ".join(str(fv) for fv in rw.values() if not isinstance(fv, (dict, list)))) for rw in row_map]
        for there, there_stamp in history:
            if there_stamp != stamp:
                continue
            differing = [kname for kname in set(here) | set(there) if here.get(kname) != there.get(kname)]
            if len(differing) != 1 or differing[0] == paging:
                continue
            key = differing[0]
            found = _eje_idle_part(key, here.get(key), there.get(key), here, there)
            if found is None:
                continue
            whole, tag, phrasing, term_set, owner = found
            if not term_set or tag in tag_names or all(_name_stems(" ".join(term_set)) <= pager for pager in row_stems):
                continue
            tag_names.add(tag)
            groups[whole].insert(0, f"{phrasing} in {json.dumps(owner, sort_keys=True)}")
        history.append((here, stamp))
    return groups[True] + groups[False]


def _eje_view_completes(agent, view: tuple[int, dict[str, Any], int, int]) -> bool:
    """True when paging this search to the paging argument's limit would show all of its matches."""
    tool = agent.roles.get("search")
    limit = _find_paging_args(agent, tool)[1] if tool else None
    if limit is None:
        return False
    total, _, read, pages = view
    per_page = -((read - total) // pages) if pages else 0
    return total <= limit + max(per_page, 1)


def _eje_page_list(views) -> str:
    """Name each unfinished search by its arguments, with how many of its rows were read."""
    return "; ".join("%s (%s of %s rows read)" % (json.dumps(view[1], sort_keys=True), view[2], view[0])
                     for view in views)


def _eje_shape(text) -> str:
    """A value with each digit run made one mark: a page value the store hands out and one made up look alike."""
    return re.sub(r"[0-9]+", "#", str(text))


def _eje_page_keys(agent) -> list[str]:
    """Optional free (non-choice) number or text arguments of the search tool: where a page position can go."""
    spec_of = agent.param_schemas.get(agent.roles.get("search")) or {}
    must = set(spec_of.get("required") or [])
    return [key for key, tool_desc in (spec_of.get("properties") or {}).items()
            if key not in must and isinstance(tool_desc, dict) and "enum" not in tool_desc
            and tool_desc.get("type") in ("integer", "string")]


def _eje_offered_next(body, kind, start) -> list:
    """Values a search reply offers for its next page: short spaceless top-level text, or whole numbers past this
    page's start and below the reply's largest count."""
    found = [fv for fv in (body or {}).values() if isinstance(fv, (int, str)) and not isinstance(fv, bool)]
    if kind is int:
        numbers = [fv for fv in found if isinstance(fv, int)]
        if not numbers:
            return []
        return [fv for fv in numbers if (start or 0) < fv < max(numbers)]
    return [fv for fv in found if isinstance(fv, str) and re.fullmatch(r"[A-Za-z0-9_.:=+/-]{1,64}", fv)]


def _eje_reply_body(element):
    reply = element.get("reply")
    body = _unwrap_text(reply.get("observation"))[1] if isinstance(reply, dict) else None
    return body if isinstance(body, dict) else None


def _eje_prior_page(agent, key, sent_args):
    """(value `key` had, reply) of the latest search with the same arguments apart from `key`, or None."""
    rest = {kname: fv for kname, fv in sent_args.items() if kname != key}
    found = None
    for element, past in _eje_search_calls(agent):
        body = _eje_reply_body(element)
        if body is not None and {kname: fv for kname, fv in past.items() if kname != key} == rest:
            found = (past.get(key), body)
    return found


def _eje_text_page_key(agent) -> str | None:
    """The text argument a store pages by: used only with values earlier search replies handed out (or, before any
    use, the single never-used one), and never with a value of the model's own."""
    keys = [key for key in _eje_page_keys(agent)
            if ((agent.param_schemas.get(agent.roles.get("search")) or {}).get("properties") or {})[key].get("type")
            == "string"]
    handed: set[str] = set()
    used, free = set(), set()
    for element, sent_args in _eje_search_calls(agent):
        for key in keys:
            if key in sent_args:
                (used if isinstance(sent_args[key], str) and sent_args[key] in handed else free).add(key)
        handed |= set(_eje_offered_next(_eje_reply_body(element), str, 0))
    picks = [key for key in keys if key in used and key not in free] or [key for key in keys if key not in used | free]
    return picks[0] if len(picks) == 1 and handed else None


def _eje_token_views(agent, label) -> list[tuple[int, dict[str, Any], int, int]]:
    """Unfinished searches of a store that pages by handed-out text: (total, args of the next page, rows seen,
    pages left), for searches that showed a row of section `label`; a page holds as many rows as the first."""
    tool = agent.roles.get("search")
    key = _eje_text_page_key(agent) if tool and _find_paging_args(agent, tool)[0] is None else None
    if key is None:
        return []
    state: dict[str, list] = {}
    for element, sent_args in _eje_search_calls(agent):
        body = _eje_reply_body(element)
        if body is None:
            continue
        tables = [fv for fv in body.values() if isinstance(fv, list) and fv and all(isinstance(rw, dict) for rw in fv)]
        shown = max(tables, key=len, default=[])
        query = {kname: fv for kname, fv in sent_args.items() if kname != key}
        sig = json.dumps(query, sort_keys=True)
        rec = state.setdefault(sig, [query, set(), 0, len(shown), False, None])
        rec[1].update(json.dumps(rw, sort_keys=True) for rw in shown)
        rec[2] = max([rec[2]] + [fv for fv in body.values() if isinstance(fv, int) and not isinstance(fv, bool)
                                 and fv >= len(shown)])
        rec[4] = rec[4] or label is None or any(label in [fv for fv in rw.values() if isinstance(fv, str)] for rw in shown)
        tokens = [fv for fv in _eje_offered_next(body, str, 0) if fv != sent_args.get(key)]
        rec[5] = tokens[0] if len(tokens) == 1 else None
    views = []
    for query, rows, total, size, in_section, token in state.values():
        if token is not None and in_section and size and len(rows) < total:
            step_on = dict(query)
            step_on[key] = token
            views.append((total, step_on, len(rows), -(-(total - len(rows)) // size)))
    return sorted(views, key=lambda view: (view[3], json.dumps(view[1], sort_keys=True)))


# ---------------------------------------------------------------- notes for the model

# one template per note the gates, finish checks and result reading give the model; fields in braces
_eje_NOTES = {
    "relax": "Price check due: {change} is different now, and only the least expensive acceptable item will do. "
             "Before buying, query the catalogue for the product again, minus whatever was dropped, and weigh what "
             "comes back.{seen} Should your current pick still be the least expensive match, submit the very same "
             "order call once more.",
    "relax_seen": " Earlier results with a lower price that could qualify after this change: {ids}.",
    "cheap_rows": "Least expensive acceptable item wanted. Lower-priced, within budget, same section, shown in results "
                  "yet never inspected: {ids}. Inspect them in this sequence; purchase the first one matching "
                  "everything, or submit the very same order call once more when none matches.",
    "cheap_pages": "Least expensive acceptable item wanted, and result rows carry prices, but some queries hold "
                   "matches not displayed so far: {pages}. Page further through them, inspect the lower-priced "
                   "in-budget rows from that same section, then purchase the least expensive full match (or submit "
                   "the very same order call once more when your pick stays lowest).",
    "cheap_bare": "Least expensive acceptable item wanted, yet no query so far used a required value on its own. Query "
                  "with nothing except {values}{scope}, then page through all results, because added words hide "
                  "items phrased differently. Next inspect in-budget rows priced under your pick and purchase the "
                  "least expensive full match (or submit the very same order call once more when your pick stays "
                  "lowest).",
    "cheap_scope": " - restricted to section \"{label}\" if the tool supports that",
    "release": "Taking {target} out is probably a mistake: judged by its own data it still satisfies the request, "
               "priced {price}{within}. Unless a message says it is unavailable or unwanted, leave it where it is; "
               "to drop it regardless, submit that same cart call again.",
    "release_budget": " (spending limit {budget})",
    "fit_unread": "The data sheet of {held} remains unread, so nothing about its properties is confirmed. Read "
                  "it, test all requirements against it, then buy; if you read it a moment ago, submit the same "
                  "order call again.",
    "fit_money": "Fresh data: {held} {why}. Something shifted mid-session, so re-inspect it. When this is confirmed, "
                 "drop it and buy the next item still acceptable (comparing prices anew if the least expensive is "
                 "wanted). Only after a new inspection proves it acceptable again should the same order call be "
                 "submitted.",
    "fit_sold": "shows as sold out",
    "fit_dear": "has gone up to {price} apiece, more than the {budget} allowed overall",
    "fit_unmet": "Missing in the own data of {held}: {unmet} - either its value disagrees, or it is silent about that "
                 "property where most items in its section speak.{pointer} Buy an item whose data confirm each "
                 "requirement; submit the same order call again only when {held} genuinely satisfies them.",
    "fit_others": " Inspected items confirming all requirements: {ids}.",
    "fit_words": " Wording the request itself uses: {phrases}. Combine that wording with the product in a query and "
                 "inspect items mentioning it.",
    "page_arg": "Paging value {value} was not offered: the latest reply to this same query handed out {offered} for "
                "its next page, and any other value skips or repeats rows. Send {call} to read on; to jump anyway, "
                "send this call again.",
    "flood": "{count} searches so far and not one listing opened: more searching rarely beats reading what is "
             "already shown. Open these unopened rows of the wanted sections first (sharpest search first): {ids}.{mixed} "
             "Then search again only for what they leave open; to search anyway, send this call again.",
    "flood_mixed": " Most rows of the worded searches came from other sections, so in that mode the section choice "
                   "seems not to apply; a wording that joins a required value with the product's own noun stays "
                   "in the section, for example {example}.",
    "page_fresh": "This query has not been sent before, so the page value {value} belongs to another query and would "
                  "start this one past its first rows. Send {call} for its first page; to jump anyway, send this "
                  "call again.",
    "reach_early": "{count} searches so far, and no opened listing meets every requirement yet. {advice} Open the rows "
                   "there whose text contradicts no requirement before trying new wordings; to search anyway, send "
                   "this call again.",
    "early_pages": "Searches that typed a required value still hold unread matches the store will page out: {pages} - "
                   "read them on to their end.",
    "early_words": "Most rows of the worded searches came from other sections; a required value joined with the "
                   "product's own noun keeps closer to it: {needles}.",
    "best_probe": "Best by {detail} wanted, and listing texts seldom lead with that figure. Figures for it seen so far "
                  "(lowest and highest per unit): {ends}. Search a required value together with such a figure, and "
                  "with figures beyond these ends in the same unit (for example {needles}); inspect what comes back, "
                  "then buy the best full match, or send the same order call again when your pick stays best.",
    "page_empty": "An empty {arg} names no page. For the first page leave {arg} out: {call}.",
    "page_none": "Paging value {value} was never handed out: the latest reply to this same query offered no value "
                 "for a next page, so a made-up one returns nothing. Rows past those shown are reached by narrowing "
                 "the query instead (add a required value or the product's own noun).{words} To send it anyway, send "
                 "this call again.",
    "page_words": " Wordings that keep to the section: {needles}.",
    "reach_rows": "Not yet - before giving up, inspect these: {ids}. Queries naming each requirement listed them, yet "
                  "their data sheets stay unread. If none turns out acceptable, submit the same decline call again.",
    "reach_pages": "Not yet - before giving up: {advice}. If nothing fresh qualifies, submit the same decline call "
                   "again.",
    "reach_read": "a few queries still have unread rows that paging would fully reach ({pages}); inspect any row "
                  "whose text contradicts no requirement",
    "reach_idle": "adding or removing {ignored} made no difference to what came back, so those rows reveal nothing "
                  "on it - narrow by other means (lead with those words or use them alone, switch filter or mode)",
    "ask_none": "no question mark appears, and messages are rationed. Write to the shopper only when raising a "
                "single property, as a single question.",
    "ask_many": "{marks} question marks found; a shopper reply covers a single point per message. Restrict it to one "
                "property.",
    "unfit": "Under this store's tool rules, that {tool} call is neither a purchase nor a decline. {forms}",
    "empty": "Your cart holds no item. Place your pick in it first - or decline.",
    "items": "The request asks for {wanted} products, one cart line each at its own quantity, and the cart holds "
             "{lines} line(s). No line yet for: {missing}. Put a verified fit for each in the cart, then order.",
    "step_open": "The request sets a step not done yet: {do} (\"{quote}\"). Perform it; finishing comes after.",
    "step_ask": "Pending per the request: {do}. Code sends the shopper that message itself; once the reply is read, "
                "finish.",
    "count": "Wrong unit number for {held}: {shown} in the cart view against {wanted} wanted. Repair it before "
             "buying - drop the line, re-add it with the correct number.",
    "decline_again": "Giving up scores zero while acceptable items exist. Still acceptable by own data and by the "
                     "reviewer: {ids}. Cart the strongest and buy it; decline solely after discovering a stated "
                     "requirement that one of them violates, and cite it.",
    "reserve": "the final move is spoken for: only a purchase or a decline may use it.",
    "swap": "Decline not sent: {pick} still meets every requirement by its own data and by the reviewer, and a "
            "purchase is worth more than giving up, so code put it in the cart (unless it was there) and placed the "
            "order. Store replies: {said}",
    "swap_cut": "Decline not sent: {pick} still meets every requirement by its own data and by the reviewer, so code "
                "put it in the cart; the reply needs handling before the order: {said}",
    "reserve_late": "near the step limit, the remaining steps are kept for the cart, the stock check and the finish.",
    "held_over": "Skipped without cost because {stop} Respond to that outcome first.",
    "closed": "Not executed: no steps left or the episode already closed.",
    "unreached": "Connection or server trouble: the store never answered this call. Retry it.",
    "why_unreached": "a store request earlier in this batch went unanswered.",
    "why_fault": "the batch already hit a fault on a prior call.",
    "why_refused": "the preceding finish attempt met a refusal.",
    "why_rejected": "a finish attempt ahead of it bounced.",
    "why_news": "the preceding result carried shopper news.",
    "fault_twice": "Blocked from here on: any call shaped like this one, since {tool} failed twice with '{code}'."
                   " Rebuild the arguments to fit the schema, or reach the goal with another tool.",
    "spent": "Allowance for this tool exhausted - no further calls to it.",
    "order_refused": "ORDER REFUSED. The session continues; the cart is untouched. To do: honour each change in the "
                     "refusal, drop cart lines that became unacceptable (say, something sold out), re-verify the "
                     "substitute's stock and price, and buy again.",
    "decline_refused": "DECLINE REFUSED. Not over yet: take in whatever changes the reply lists, keep looking, "
                       "and close out afterwards.",
    "recheck": "Least expensive acceptable item wanted: these in-budget items from the same section, costing at most "
               "what your pick last cost, were displayed but never inspected: {priced}. Go from the lowest "
               "price upward, testing each against every requirement, before settling for something pricier.",
    "rejected": "Finishing failed: the store gave a reason in its reply. Deal with that cause and close out "
                "again.",
    "news": "Shopper news arrived with this result - work it in: {heard}",
}


def _format_note(key: str, **field_specs) -> str:
    return _eje_NOTES[key].format(**field_specs)


def _held_note(key: str, **field_specs) -> str:
    """A hold note: the call is kept back without spending a step."""
    return _HELD_NOTICE + " " + _format_note(key, **field_specs)


# ---------------------------------------------------------------- price, release, fit and reach gates

def _eje_is_order(agent, call) -> bool:
    return agent._classify_call(call["name"], call["args"]) == "order"


def _eje_g_relax(agent, gate_ctx, call):
    """On a cheapest request, a dropped condition or a new budget earns one hold to compare prices again."""
    if not agent.rerank_reason or not _eje_is_order(agent, call) or not agent.cart_ids:
        return None
    change = agent.rerank_reason
    agent.rerank_reason = ""
    if not agent._has_step_room(3):
        return None
    lower = _find_cheaper_rows(agent, agent.cart_ids[-1], agent._row_table(), False)[:4]
    seen = _format_note("relax_seen", ids=", ".join(lower)) if lower else ""
    return _held_note("relax", change=change, seen=seen)


def _eje_cheap_lead(agent, row_map, held):
    """What could still show a lower price for the held pick: ("rows", ids), ("pages", views), ("bare", values)
    or None."""
    limit = max(1, min(4, agent.remaining_steps() - agent._reserved_steps() - 1))
    sources = _eje_seen_sources(agent)
    ranked = sorted(_find_cheaper_rows(agent, held, row_map, True),
                    key=lambda idx: (-sources.get(idx, (0, 0))[0], row_map[idx]["price"]))
    if ranked[:limit]:
        return "rows", ranked[:limit]
    label = (row_map.get(held) or {}).get("label")
    open_views = [fv for fv in _partial_search_views(agent, label) if agent._has_step_room(fv[3] + 2)]
    complete = [fv for fv in open_views if _eje_view_completes(agent, fv)]   # paging these shows every match
    complete += [fv for fv in _eje_token_views(agent, label) if agent._has_step_room(fv[3] + 2)]   # handed-out pages
    bare = [fv for fv in _eje_bare_values(agent) if not _eje_searched_bare(agent, fv)]
    pending = complete or ([] if bare else open_views)
    if pending:
        return "pages", pending[:2]
    if bare and agent._has_step_room(4):
        return "bare", (bare[:2], label)
    return None


def _eje_g_cheapest(agent, gate_ctx, call):
    """A cheapest order waits once per state while cheaper listings of its section, unread pages or a search on a
    bare required value could still turn up a lower price."""
    if agent.plan.get("priority") != "cheapest" or not _eje_is_order(agent, call) or not agent.cart_ids:
        return None
    row_map = agent._row_table()
    arm_key = f"cheapest {agent._arm_signature()} {len(row_map)}"   # newly shown rows open a fresh state
    if arm_key in agent.held_arms or not agent._has_step_room(2):
        return None
    opener = _eje_cheap_lead(agent, row_map, agent.cart_ids[-1])
    if opener is None:
        return None
    agent.held_arms.add(arm_key)
    form, what = opener
    if form == "rows":
        return _held_note("cheap_rows", ids=", ".join(what))
    if form == "pages":
        return _held_note("cheap_pages", pages=_eje_page_list(what))
    values, label = what
    return _held_note("cheap_bare", values=" / ".join('"%s"' % fv for fv in values),
                     scope=_format_note("cheap_scope", label=label) if label else "")


def _eje_measure_ends(agent, detail) -> list[str]:
    """Figures with their unit as the store writes them ("512gb"), for the property a best-by request ranks on: units
    come from opened sheets' keys naming it; any shown text adds figures in those units. The lowest and highest
    figure per unit, the unit seen most first (at most two units)."""
    terms = _word_set(detail)
    found: dict[str, dict[str, float]] = {}
    for facts in [rw["facts"] for rw in agent._row_table().values() if rw["opened"]]:
        for key, text in facts.items():
            if isinstance(text, str) and terms & _word_set(key):
                for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)", text.lower()):
                    found.setdefault(unit, {})[num + unit] = float(num)
    if not found:
        return []
    for _, _, shown in _eje_search_pages(agent):
        for rw in shown:
            for text in [fv for fv in rw.values() if isinstance(fv, str)]:
                for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)\b", text.lower()):
                    if unit in found:
                        found[unit][num + unit] = float(num)
    ends = []
    for unit in sorted(found, key=lambda un: -len(found[un]))[:2]:
        ranked = sorted(found[unit], key=lambda tk: found[unit][tk])
        ends += list(dict.fromkeys([ranked[0], ranked[-1]]))
    return ends


def _eje_g_best_probe(agent, gate_ctx, call):
    """A best-by-property order waits once per state while no search typed a required value together with a figure
    of that property: listing texts rarely lead with it, so such searches reach items a plain one buries."""
    plan = agent.plan or {}
    if plan.get("priority") != "best" or not _eje_is_order(agent, call) or not agent.cart_ids:
        return None
    arm_key = "best " + agent._arm_signature()
    if arm_key in agent.held_arms or not agent._has_step_room(4):
        return None
    detail = str(plan.get("priority_detail") or "")
    ends, values = _eje_measure_ends(agent, detail), [fv for fv in _eje_bare_values(agent) if _word_set(fv)]
    key, cap = _eje_text_key(agent)
    if not ends or not values or key is None:
        return None
    units = {re.sub(r"[0-9.]+", "", tk) for tk in ends}
    for _, args in _eje_search_calls(agent):
        text = str(args.get(key) or "").lower()
        figures = {unit for _, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)", text)}
        if figures & units and any(_word_set(fv) <= _word_set(text) for fv in values):
            return None   # probed already
    agent.held_arms.add(arm_key)
    needles = [_eja_word_cut(values[0] + " " + tk, cap or 64) for tk in ends[:2]]
    return _held_note("best_probe", detail=detail or "the measure", ends=", ".join(ends),
                      needles=" / ".join(json.dumps(fv) for fv in needles))


def _eje_g_release(agent, gate_ctx, call):
    """Removing the held item after a change waits once while its own facts and price still pass."""
    tool, dispatched = call["name"], call["args"]
    if tool != gate_ctx["cart"] or agent.add_signature is None:
        return None
    if agent._choice_signature(tool, dispatched) == agent.add_signature:
        return None   # an add, not a removal
    target = _cart_listing_id(agent, tool, dispatched)
    arm_key = f"keep {target} {agent._arm_signature()}"
    if not agent.cart_generation or target not in agent.cart_ids or arm_key in agent.held_arms or not agent._has_step_room(1):
        return None
    facts = agent._row_table().get(target)
    if not facts or not agent._within_price_cap(facts) or agent._row_passes_literal(facts) is not True:
        return None
    agent.held_arms.add(arm_key)
    budget = agent._need_list.budget
    within = "" if budget is None else _format_note("release_budget", budget=budget)
    return _held_note("release", target=target, price=facts["price"], within=within)


def _eje_fit_pointer(agent, held, row_map, unmet) -> str:
    """Where to look next when the held listing misses a requirement."""
    section = row_map[held].get("label")
    passing = []
    for ident, facts in row_map.items():
        if ident == held or not facts["opened"] or not agent._within_price_cap(facts):
            continue
        if (section is None or facts["label"] == section) and agent._row_passes_literal(facts) is True:
            passing.append(ident)
    if passing:
        return _format_note("fit_others", ids=", ".join(passing[:3]))
    phrases = [ch for ch in dict.fromkeys(agent._request_sentence_for(str(u).partition(":")[0]) for u in unmet) if ch]
    if not phrases:
        return ""
    return _format_note("fit_words", phrases=" | ".join('"%s"' % ch for ch in phrases[:2]))


def _eje_g_fit(agent, gate_ctx, call):
    """An order waits once per state while the held listing is unread, out of money or stock, or misses a stated
    requirement in its own details."""
    if not _eje_is_order(agent, call) or not agent.cart_ids or not agent._has_step_room(2):
        return None
    row_map = agent._row_table()
    for held in (agent.cart_ids[:-1] if agent._wanted_items() else []):
        note = _eje_g_fit_line(agent, held, row_map)
        if note:
            return note
    return _eje_g_fit_line(agent, agent.cart_ids[-1], row_map)


def _eje_g_fit_line(agent, held, row_map):
    """The fit hold for one held line, once per state (None when it passes)."""
    arm_key = f"fit {held} {agent._arm_signature()}"
    if arm_key in agent.held_arms:
        return None
    facts = row_map.get(held) or {}
    if not facts.get("opened"):
        note = _held_note("fit_unread", held=held)
    elif not agent._within_price_cap(facts):   # newest records show it unavailable or above the budget
        if facts.get("stock") is False:
            why = _format_note("fit_sold")
        else:
            why = _format_note("fit_dear", price=facts.get("price"), budget=agent._need_list.budget)
        note = _held_note("fit_money", held=held, why=why)
    else:
        unmet = agent._missed_conditions(held, row_map)
        if not unmet:
            return None
        note = _held_note("fit_unmet", held=held, unmet="; ".join(unmet),
                         pointer=_eje_fit_pointer(agent, held, row_map, unmet))
    agent.held_arms.add(arm_key)
    return note


def _eje_unopened_hits(agent, needed, limit):
    """Ids from searches that named every required value, never opened and not ruled out, best first."""
    row_map = agent._row_table() if needed and limit > 0 else {}
    if not row_map:
        return []
    picks = []
    for ident, (score, position) in _eje_seen_sources(agent).items():
        facts = row_map.get(ident)
        if score < needed or facts is None or facts["opened"] or facts["stock"] is False:
            continue
        if agent._within_price_cap(facts) and agent._row_passes_literal(facts) is not False:
            picks.append((-score, position, ident))
    return [ident for _, _, ident in sorted(picks)]


# searches sent with no listing opened yet; the next new search after this many waits once, naming rows to open
_eje_FLOOD_SEARCHES = 6


def _eje_wanted_sections(agent) -> set[str]:
    """Section labels (lower case) of the requested products: labels of seen listings equal to a choice value some
    search sent; a product none of those covers adds the labels sharing a word stem with it. Empty: unknown."""
    tool = agent.roles.get("search")
    props = ((agent.param_schemas.get(tool) or {}).get("properties") or {}) if tool else {}
    chosen = set()
    for _, sent_args in _eje_search_calls(agent):
        for key, fv in sent_args.items():
            if isinstance(fv, str) and isinstance(props.get(key), dict) and props[key].get("enum"):
                chosen.add(fv.strip().lower())
    labels = {rw["label"].strip().lower() for rw in agent._row_table().values() if isinstance(rw.get("label"), str)}
    exact = {lb for lb in labels if lb in chosen}
    products = [el["product"] for el in agent._wanted_items()] or [(agent.plan or {}).get("product") or ""]
    for product in products:
        stems = agent._ejc_stems(product)
        if stems and not any(stems & agent._ejc_stems(lb) for lb in exact):
            exact |= {lb for lb in labels if stems & agent._ejc_stems(lb)}
    return exact


def _eje_page_total(body, shown) -> int:
    counts = [fv for fv in body.values() if isinstance(fv, int) and not isinstance(fv, bool) and fv >= len(shown)]
    return max(counts, default=len(shown))


def _eje_row_ident(listing_row):
    return next((fv for kname, fv in listing_row.items()
                 if isinstance(fv, str) and "id" in str(kname).lower() and _ID_SHAPE.match(fv)), None)


def _eje_sharp_sources(agent) -> dict[str, tuple[int, int, int]]:
    """Each listing id a search showed -> (minus the most required values a search showing it typed, the fewest
    matches such a search reported, first position): the sharpest search that reached a row ranks it."""
    tool = agent.roles.get("search")
    paging = _find_paging_args(agent, tool)[0] if tool else None
    props = ((agent.param_schemas.get(tool) or {}).get("properties") or {}) if tool else {}
    wanted = [w for w in map(_word_set, _eje_bare_values(agent)) if w]
    best: dict[str, tuple[int, int, int]] = {}
    for sent_args, body, shown in _eje_search_pages(agent):
        typed = _eje_typed_words(sent_args, paging, props)
        rank = (-len([w for w in wanted if w <= typed]), _eje_page_total(body, shown))
        for position, listing_row in enumerate(shown):
            ident = _eje_row_ident(listing_row)
            if ident is not None:
                best[ident] = min(best.get(ident, rank + (position,)), rank + (position,))
    return best


def _eje_aimed_rows(agent, limit, only=None, typed_only=False) -> list[str]:
    """Unopened, available, in-budget listings of the requested sections that no stated value rules out, best lead
    first: shown by a search typing more required values, then by the sharpest such search, then cheaper. With
    several products the sections take turns; `only` keeps rows of those product indexes; `typed_only` keeps rows
    some search reached by typing a required value."""
    row_map = agent._row_table()
    sections = _eje_wanted_sections(agent)
    ranked = []
    for ident, (minus, total, position) in _eje_sharp_sources(agent).items():
        facts = row_map.get(ident)
        if facts is None or facts["opened"] or (typed_only and not minus):
            continue
        label = str(facts.get("label") or "").strip().lower()
        if (sections and label not in sections) or not agent._within_price_cap(facts):
            continue
        if agent._row_passes_literal(facts) is False:
            continue
        price = facts["price"] if isinstance(facts.get("price"), (int, float)) else float("inf")
        ranked.append(((minus, total, price, position), label, ident))
    queues: dict[str, list[str]] = {}
    for _, label, ident in sorted(ranked):
        queues.setdefault(label, []).append(ident)
    picks: list[str] = []
    while len(picks) < limit and any(queues.values()):
        for label, waiting in queues.items():
            while waiting and len(picks) < limit:
                ident = waiting.pop(0)
                if only is None or agent._ejc_item_index(ident) in only:
                    picks.append(ident)
                    break
    return picks


def _eje_fitless_items(agent) -> set:
    """Indexes of the requested products (0 for a single one) with no opened fit yet: in budget, nothing missed."""
    row_map = agent._row_table()
    many = agent._wanted_items()
    fitted = set()
    for ident, facts in row_map.items():
        if facts["opened"] and agent._within_price_cap(facts) and not agent._missed_conditions(ident, row_map):
            fitted.add(agent._ejc_item_index(ident) if many else 0)
    return set(range(len(many) or 1)) - fitted


def _eje_unfitted_rows(agent, limit) -> list[str]:
    """For requested products with no opened fit yet (in budget, nothing missed), their unopened on-target rows."""
    still = _eje_fitless_items(agent)
    if not still:
        return []
    return _eje_aimed_rows(agent, limit, still if agent._wanted_items() else None, typed_only=True)


def _eje_reads_sent(agent, gate_ctx) -> int:
    """Detail or side-by-side reads sent so far, this batch's let-through ones included."""
    tools = {agent.roles.get("details"), agent.roles.get("view")} - {None}
    sent = [(el.get("actions") or [{}])[0].get("name") for el in agent.trace if el.get("kind") == "env"]
    sent += [name for _, name, _ in gate_ctx.get("ready") or []]
    return len([name for name in sent if name in tools])


def _eje_next_page(agent, sent_args) -> bool:
    """True when the call repeats an earlier search apart from its paging values."""
    tool = agent.roles.get("search")
    keys = {_find_paging_args(agent, tool)[0], _eje_text_page_key(agent)} - {None}
    handed: set = set()
    for element, _ in _eje_search_calls(agent):
        handed.update(_eje_offered_next(_eje_reply_body(element), str, 0))
    keys |= {key for key, fv in sent_args.items() if isinstance(fv, str) and fv in handed}
    if not keys & set(sent_args):
        return False
    rest = {key: fv for key, fv in sent_args.items() if key not in keys}
    return any({key: fv for key, fv in past.items() if key not in keys} == rest for _, past in _eje_search_calls(agent))


def _eje_section_noun(label: str) -> str:
    """The longest plain word of a section label (a later one on a tie): the product's own noun."""
    words = [w for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", label) if "'" not in w and len(w) > 2]
    return max(reversed(words), key=len).lower() if words else ""


def _eje_mixed_example(agent) -> str:
    """When the worded searches that chose a section mostly showed rows of other sections, a sharper wording of the
    latest one: its words plus the section's noun, cut by whole words to the argument's length ('' otherwise)."""
    tool = agent.roles.get("search")
    props = ((agent.param_schemas.get(tool) or {}).get("properties") or {}) if tool else {}
    page_keys = {_find_paging_args(agent, tool)[0], _eje_text_page_key(agent)} - {None}
    row_map, sections = agent._row_table(), _eje_wanted_sections(agent)
    off = total = 0
    example = ""
    for sent_args, _, shown in _eje_search_pages(agent):
        texts = [(key, fv) for key, fv in sent_args.items() if isinstance(fv, str) and key not in page_keys
                 and not (props.get(key) or {}).get("enum") and fv.strip()]
        picked = [fv.strip().lower() for key, fv in sent_args.items()
                  if isinstance(fv, str) and (props.get(key) or {}).get("enum") and fv.strip().lower() in sections]
        if not texts or not picked or not shown:
            continue
        labels = [str((row_map.get(_eje_row_ident(rw)) or {}).get("label") or "").strip().lower() for rw in shown]
        strays = len([lb for lb in labels if lb and lb != picked[0]])
        off, total = off + strays, total + len(shown)
        noun = _eje_section_noun(picked[0])
        key, words = texts[0]
        if strays * 2 > len(shown) and noun and noun not in _word_set(words):
            cap = (props.get(key) or {}).get("maxLength")
            room = cap - len(noun) - 1 if isinstance(cap, int) else len(words)
            example = (_eja_word_cut(words.strip(), room) + " " + noun) if room > 0 else ""
    return example if total and off * 2 > total else ""


def _eje_g_flood(agent, gate_ctx, call):
    """A new search waits once while many searches went out and not one listing has been opened, naming the
    best unopened rows of the requested sections (next pages of a sent search pass)."""
    tool = agent.roles.get("search")
    if not tool or call["name"] != tool or gate_ctx["planned"] or "flood" in agent.held_arms:
        return None
    sent = len(list(_eje_search_calls(agent))) + len([1 for _, name, _ in gate_ctx.get("ready") or [] if name == tool])
    if sent < _eje_FLOOD_SEARCHES or _eje_reads_sent(agent, gate_ctx) or not agent._has_step_room(3):
        return None
    if _eje_next_page(agent, call.get("args") or {}):
        return None
    picks = _eje_aimed_rows(agent, 4)
    if not picks:
        return None
    agent.held_arms.add("flood")
    example = _eje_mixed_example(agent)
    mixed = _format_note("flood_mixed", example=json.dumps(example)) if example else ""
    return _held_note("flood", count=sent, ids=", ".join(picks), mixed=mixed)


def _eje_text_key(agent) -> tuple[str | None, int | None]:
    """The search tool's free-text argument (no choice list, not a paging one) and its length cap: of several, the
    one sent with spaces most often, then the longest allowed."""
    tool = agent.roles.get("search")
    props = ((agent.param_schemas.get(tool) or {}).get("properties") or {}) if tool else {}
    paging = {_find_paging_args(agent, tool)[0], _eje_text_page_key(agent)} - {None}
    keys = [key for key, tk in props.items() if isinstance(tk, dict) and tk.get("type") == "string"
            and not tk.get("enum") and key not in paging]
    if not keys:
        return None, None
    spaced = {key: len([1 for _, args in _eje_search_calls(agent) if " " in str(args.get(key) or "").strip()])
              for key in keys}
    key = max(keys, key=lambda kname: (spaced[kname], props[kname].get("maxLength") or 0))
    cap = props[key].get("maxLength")
    return key, cap if isinstance(cap, int) else None


def _eje_mixed_labels(agent) -> list[str]:
    """When the worded searches that chose a requested section mostly showed rows of other sections: those sections,
    latest first ([] otherwise)."""
    tool = agent.roles.get("search")
    props = ((agent.param_schemas.get(tool) or {}).get("properties") or {}) if tool else {}
    page_keys = {_find_paging_args(agent, tool)[0], _eje_text_page_key(agent)} - {None}
    row_map, sections = agent._row_table(), _eje_wanted_sections(agent)
    off = total = 0
    chosen: list[str] = []
    for sent_args, _, shown in _eje_search_pages(agent):
        worded = any(isinstance(fv, str) and fv.strip() and key not in page_keys and not (props.get(key) or {}).get("enum")
                     for key, fv in sent_args.items())
        picked = [fv.strip().lower() for key, fv in sent_args.items()
                  if isinstance(fv, str) and (props.get(key) or {}).get("enum") and fv.strip().lower() in sections]
        if not worded or not picked or not shown:
            continue
        labels = [str((row_map.get(_eje_row_ident(rw)) or {}).get("label") or "").strip().lower() for rw in shown]
        off, total = off + len([lb for lb in labels if lb and lb != picked[0]]), total + len(shown)
        chosen = [picked[0]] + [lb for lb in chosen if lb != picked[0]]
    return chosen if total and off * 2 > total else []


def _eje_value_needles(agent) -> list[str]:
    """Searches whose rows mostly leave the section: up to two wordings that keep to it - a required value of that
    section's product with the section's noun, else the value alone - within the text argument's length, short
    (at most three words of value) and never sent before. With several products, a section of one still lacking a
    fit is taken, and only its own requirements."""
    key, cap = _eje_text_key(agent)
    labels = _eje_mixed_labels(agent) if key is not None else []
    many, still = agent._wanted_items(), _eje_fitless_items(agent)
    conditions = None
    if many:
        owner = {lb: [k for k, el in enumerate(many) if agent._ejc_stems(el["product"]) & agent._ejc_stems(lb)]
                 for lb in labels}
        labels = [lb for lb in labels if len(owner[lb]) == 1 and owner[lb][0] in still]
        if labels:
            element = many[owner[labels[0]][0]]
            conditions = element["requirements"] + agent._ejc_homed(element["product"])
    noun = _eje_section_noun(labels[0]) if labels else ""
    if not noun:
        return []
    sent = [_name_stems(fv) for _, args in _eje_search_calls(agent) for kname, fv in args.items()
            if kname == key and isinstance(fv, str)]
    out: list[str] = []
    for field_value in _eje_bare_values(agent, conditions):
        if len(field_value.split()) > 3:
            continue
        for needle in (field_value + " " + noun, field_value):
            if (cap is None or len(needle) <= cap) and _name_stems(needle) not in sent and needle not in out:
                out.append(needle)
                break
    return out[:2]


def _eje_on_section(agent, query, keys, sections) -> bool:
    """True unless most rows the search `query` (paging keys left out) showed sit outside the requested sections."""
    row_map = agent._row_table()
    labels = []
    for sent_args, _, shown in _eje_search_pages(agent):
        if {kname: fv for kname, fv in sent_args.items() if kname not in keys} == query:
            labels += [str((row_map.get(_eje_row_ident(rw)) or {}).get("label") or "").strip().lower() for rw in shown]
    labels = [lb for lb in labels if lb]
    return not sections or not labels or 2 * len([lb for lb in labels if lb in sections]) >= len(labels)


def _eje_value_views(agent) -> list[tuple[int, dict[str, Any], int, int]]:
    """Unfinished searches that typed a required value and mostly stayed in the section, read to the end of what
    the store pages (not only those paging would finish): handed-out pages and fully reachable ones first."""
    tool = agent.roles.get("search")
    if not tool:
        return []
    number_key = _find_paging_args(agent, tool)[0]
    keys = {number_key, _eje_text_page_key(agent)} - {None}
    props = (agent.param_schemas.get(tool) or {}).get("properties") or {}
    wanted, sections = _ejd_search_vocab(agent)[0], _eje_wanted_sections(agent)
    tokens = _eje_token_views(agent, None)
    _, limit = _find_paging_args(agent, tool)
    picked = []
    for view in tokens + _partial_search_views(agent, None):
        per_page = max(1, (view[0] - view[2]) // max(view[3], 1))
        reach = (limit + per_page) if limit is not None and view not in tokens else _eje_EARLY_PAGES * per_page
        if view[0] > 2 * reach:
            continue   # too wide to read through: narrowing serves better than paging
        query = {kname: fv for kname, fv in view[1].items() if kname not in keys}
        if not wanted & _eje_typed_words(query, number_key, props) or not _eje_on_section(agent, query, keys, sections):
            continue
        if agent._has_step_room(min(view[3], 2) + 2):
            picked.append((view not in tokens and not _eje_view_completes(agent, view), view))
    return [view for _, view in sorted(picked, key=lambda tk: tk[0])][:2]


# a store paging by handed-out text: pages a narrow search may still be read through
_eje_EARLY_PAGES = 5
# free steps the early push needs beyond the reserve: paging on, opening a find, and buying it
_eje_EARLY_ROOM = 8


def _eje_g_reach_early(agent, gate_ctx, call):
    """A new search waits once per state when many searches went out and no opened listing fits a requested product
    yet, while searches typing a required value still hold unread pages, or worded searches mostly left the
    section (then the note names wordings that keep to it). Next pages of a sent search pass."""
    tool = agent.roles.get("search")
    if not tool or call["name"] != tool or gate_ctx["planned"]:
        return None
    arm_key = "reach early " + agent._arm_signature()
    if arm_key in agent.held_arms or not agent._has_step_room(_eje_EARLY_ROOM):
        return None
    sent = len(list(_eje_search_calls(agent))) + len([1 for _, name, _ in gate_ctx.get("ready") or [] if name == tool])
    if sent < _eje_FLOOD_SEARCHES or _eje_next_page(agent, call.get("args") or {}) or not _eje_fitless_items(agent):
        return None
    views, needles = _eje_value_views(agent), _eje_value_needles(agent)
    if not views and not needles:
        return None
    agent.held_arms.add(arm_key)
    advice = []
    if views:
        advice.append(_format_note("early_pages", pages=_eje_page_list(views)))
    if needles:
        advice.append(_format_note("early_words", needles=" / ".join(json.dumps(fv) for fv in needles)))
    return _held_note("reach_early", count=sent, advice=" ".join(advice))


def _eje_g_reach(agent, gate_ctx, call):
    """A decline waits once per state while rows from on-target searches are unopened or such searches can still be
    paged to their end; the note also lists search parts the store ignored."""
    kind = agent._classify_call(call["name"], call["args"])
    if kind != "decline":
        return None
    arm_key = f"reach {kind} {agent._arm_signature()}"
    if arm_key in agent.held_arms or not agent._has_step_room(3) or agent.seconds_remaining() < WRAP_UP_WINDOW + 30:
        return None
    needed = len([fv for fv in _eje_bare_values(agent) if _word_set(fv)])
    limit = min(4, agent.remaining_steps() - agent._reserved_steps() - 2)
    hits = _eje_unopened_hits(agent, needed, limit)
    if hits:
        agent.held_arms.add(arm_key)
        return _held_note("reach_rows", ids=", ".join(hits[:limit])) + agent._ejc_read_by_view()
    readable = [fv for fv in _partial_search_views(agent, None, singular=True)
                if _eje_view_completes(agent, fv) and agent._has_step_room(fv[3] + 2)][:2]
    ignored = _eje_idle_parts(agent)[:2]
    needles = _eje_value_needles(agent) if agent.roles.get("search") in gate_ctx["allowed"] else []
    if not readable and not needles:
        return None
    agent.held_arms.add(arm_key)
    advice = [_format_note("reach_read", pages=_eje_page_list(readable))] if readable else []
    if needles:
        advice.append(_format_note("early_words", needles=" / ".join(json.dumps(fv) for fv in needles)))
    if ignored:
        advice.append(_format_note("reach_idle", ignored="; ".join(ignored)))
    return _held_note("reach_pages", advice="; also ".join(advice))


def _eje_g_page(agent, gate_ctx, call):
    """A search paging with a value the latest reply of that same search did not hand out waits once per call,
    naming the one it did (another value skips or repeats rows)."""
    tool = agent.roles.get("search")
    sent_args = call.get("args") or {}
    if not tool or call["name"] != tool or not isinstance(sent_args, dict) or not agent._has_step_room(2):
        return None
    text_key = _eje_text_page_key(agent)
    number_key = _find_paging_args(agent, tool)[0]
    for key in _eje_page_keys(agent):
        field_value = sent_args.get(key)
        kind = int if isinstance(field_value, int) and not isinstance(field_value, bool) else str
        if not isinstance(field_value, (int, str)) or isinstance(field_value, bool) or (kind is int and key != number_key):
            continue
        if kind is str and key == text_key and not field_value.strip():
            # an empty paging value names no page: the first page comes without the argument at all
            memo = "empty " + tool + _to_compact_json(sent_args)
            if memo in agent.held_arms:
                return None
            agent.held_arms.add(memo)
            first = {kname: fv for kname, fv in sent_args.items() if kname != key}
            return _held_note("page_empty", arg=key, call=_to_compact_json(first))
        prior = _eje_prior_page(agent, key, sent_args)
        if prior is None and kind is str and key == text_key and field_value.strip():
            memo = "fresh " + tool + _to_compact_json(sent_args)
            if memo in agent.held_arms:
                return None
            agent.held_arms.add(memo)
            first = {kname: fv for kname, fv in sent_args.items() if kname != key}
            return _held_note("page_fresh", value=json.dumps(field_value), call=_to_compact_json(first))
        if prior is None or prior[0] == field_value:
            continue
        if any(name == tool and {k: v for k, v in (args or {}).items() if k != key} ==
               {k: v for k, v in sent_args.items() if k != key} for _, name, args in gate_ctx.get("ready") or []):
            continue   # several pages of one search in one reply: only the first is checked
        start = prior[0] if isinstance(prior[0], int) and not isinstance(prior[0], bool) else 0
        offered = _eje_offered_next(prior[1], kind, start)
        if kind is int and field_value <= start:
            continue   # going back is a choice, not a slip
        if kind is str and key != text_key:
            offered = [fv for fv in offered if _eje_shape(fv) == _eje_shape(field_value)]
        if not offered and kind is str and _eje_made_up_page(agent, key, field_value):
            memo = "none " + tool + _to_compact_json(sent_args)
            if memo in agent.held_arms:
                return None
            agent.held_arms.add(memo)
            needles = _eje_value_needles(agent)
            words = _format_note("page_words", needles=" / ".join(json.dumps(fv) for fv in needles)) if needles else ""
            return _held_note("page_none", value=json.dumps(field_value), words=words)
        if len(offered) != 1 or field_value in offered:
            continue
        memo = "page " + tool + _to_compact_json(sent_args)
        if memo in agent.held_arms:
            return None
        agent.held_arms.add(memo)
        step_on = dict(sent_args)
        step_on[key] = offered[0]
        return _held_note("page_arg", value=json.dumps(field_value), offered=json.dumps(offered[0]),
                          call=_to_compact_json(step_on))
    return None


def _eje_made_up_page(agent, key, field_value) -> bool:
    """True for a text paging value no search reply ever handed out that does not read as search text: one word,
    holding a digit or a seen listing id, or sharing no word with the request, its conditions or other typed text."""
    value = field_value.strip()
    if not value or " " in value or key == _eje_text_key(agent)[0]:
        return False
    handed: set = set()
    typed: set = _word_set(agent.policy.get("query")) | _word_set(" ".join(agent._list_conditions()))
    for element, sent_args in _eje_search_calls(agent):
        handed.update(_eje_offered_next(_eje_reply_body(element), str, 0))
        typed |= _word_set(" ".join(str(fv) for kname, fv in sent_args.items() if kname != key and isinstance(fv, str)))
    if value in handed:
        return False   # a value the store gave out always passes
    return bool(re.search(r"[0-9]", value)) or value in agent.seen_ids or not (_word_set(value) & typed)


def _eje_g_ask_form(agent, gate_ctx, call):
    """A message the model writes must hold exactly one question mark."""
    if gate_ctx["planned"] or call["name"] != gate_ctx["message"] or agent._pending_procedures("order"):
        return None
    marks = sum(fv.count("?") for fv in (call["args"] or {}).values() if isinstance(fv, str))
    if marks == 1:
        return None
    return _SKIP_NOTICE + ": " + (_format_note("ask_many", marks=marks) if marks else _format_note("ask_none"))


# every gate a proposed call passes through, in order; the first note returned holds or skips the call
_eje_GATES = (
    _gate_unknown_tool, _gate_not_offered, _gate_bad_json, _gate_schema_misfit, _gate_finish_alone, _gate_known_fault,
    _gate_tool_quota, _gate_repeat_call, _eje_g_page, _eje_g_flood, _eje_g_reach_early, _gate_late_details,
    _gate_late_unfitted, _eje_g_ask_form, _gate_stacked_add, _gate_cart_review, _gate_unit_count,
    _eje_g_release, _gate_reread_after_refusal, _eje_g_relax, _eje_g_cheapest, _eje_g_best_probe, _eje_g_fit,
    _eje_g_reach,
)

_eje_ORDER_CHECK = _HELD_NOTICE + (
    " Check this list first, fix whatever is not yet true, and then send the order call again."
    " 1) Cart: exactly one line per requested product (one line for a single product), each the chosen listing"
    " at its quantity."
    " 2) Fit: the listing's own properties satisfy each stated requirement, each need the shopper revealed and each"
    " announced change."
    " 3) Price and stock: read again after your latest cart change, using a tool that shows them; available, and"
    " quantity times price within the current budget."
    " 4) Choice: when the cheapest or the best was requested, nothing you could find fits better."
    " 5) Rules: every procedure the request set has been followed."
)

_eje_DECLINE_CHECK = _HELD_NOTICE + (
    " Declining cannot be reversed. Repeat it only once three things are certain - each candidate's own data were"
    " read; queries varied in words, filters and pages; and every candidate breaks the budget, the stock, a stated"
    " requirement or a revealed need. Matching keys may be spelled differently, and near-synonyms count. Buy"
    " whatever fits; if nothing does, call the decline tool once more."
)

# record key a handler sets to choose its outcome; the machine removes it again
_ROUTE_KEY = "_eje_route"


# ---------------------------------------------------------------- execution handlers

def _eje_context(agent, calls, planned) -> dict[str, Any]:
    """What every gate may read about this batch of calls."""
    offered = {tk["function"]["name"] for tk in agent._tools_this_turn()}
    roles = agent.roles
    return {"planned": planned, "allowed": offered,
            "finishing": {roles.get("order"), roles.get("decline")} - {None},
            "message": roles.get("message"), "cart": roles.get("cart"),
            "count": len(calls), "queued_messages": 0}


def _eje_first_note(agent, gate_ctx, probe):
    for gate in _eje_GATES:
        note = gate(agent, gate_ctx, probe)
        if note is not None:
            return note
    return None


def _eje_x_screen(agent: Episode, duty: dict[str, Any]) -> dict[str, Any]:
    """Run every proposed call through the gates; held or skipped calls get their note as the reply."""
    gate_ctx = duty["ctx"] = _eje_context(agent, duty["calls"], duty["planned"])
    duty["replies"], duty["ready"] = [], []
    gate_ctx["ready"] = duty["ready"]   # calls already let through in this batch
    for proposed in duty["calls"]:
        func = proposed.get("function") or {}
        probe = {"id": proposed["id"], "name": func.get("name"), "raw": func.get("arguments")}
        note = _eje_first_note(agent, gate_ctx, probe)
        if note is not None:
            duty["replies"].append((probe["id"], note))
            continue
        if probe["name"] == gate_ctx["message"]:
            gate_ctx["queued_messages"] += 1
        duty["ready"].append((probe["id"], probe["name"], probe["args"]))
    return duty


def _hold_job(duty: dict[str, Any], call_id: str | None, note: str | None) -> dict[str, Any]:
    """Drop the finishing call, answer it with the full `note` (when given) and route past the other checks."""
    if note is not None:
        duty["replies"].append((call_id, note))
    duty["ready"] = []
    duty[_ROUTE_KEY] = "held"
    return duty


def _eje_f_pick(agent: Episode, duty: dict[str, Any]) -> dict[str, Any]:
    """The finish checks apply only when the first surviving call is a finishing tool."""
    queue = duty["ready"]
    if queue and queue[0][1] in duty["ctx"]["finishing"]:
        ident, tool, dispatched = queue[0]
        duty["finish"] = {"id": ident, "name": tool, "args": dispatched, "kind": agent._classify_call(tool, dispatched)}
    else:
        duty[_ROUTE_KEY] = "pass"
    return duty


def _eje_f_unfit(agent: Episode, duty: dict[str, Any]) -> dict[str, Any]:
    """A finishing tool whose values fit neither job: explain both forms once per call, then let it through."""
    closing = duty["finish"]
    if closing["kind"] is not None:
        return duty
    tool = closing["name"]
    shared = any(job_name not in _CLOSING_JOBS and bound == tool for job_name, bound in agent.roles.items())
    memo = "unfit " + tool + _to_compact_json(closing["args"])
    if shared or memo in agent.held_arms:
        duty[_ROUTE_KEY] = "pass"
        return duty
    agent.held_arms.add(memo)
    forms = " ".join(agent._closing_lines())
    return _hold_job(duty, closing["id"], _held_note("unfit", tool=tool, forms=forms))


def _eje_f_empty(agent: Episode, duty: dict[str, Any]) -> dict[str, Any]:
    """An order with an empty cart is held every time; past three holds per cart state, code finishes."""
    if duty["finish"]["kind"] != "order" or not agent.roles.get("cart") or agent.cart_ids:
        return duty
    if agent.remaining_steps() <= 1:
        return duty
    stamp = agent.cart_changed_at
    agent.empty_holds[stamp] = tries = agent.empty_holds.get(stamp, 0) + 1
    if tries > 3:
        agent.finish_now()
        return _hold_job(duty, None, None)
    return _hold_job(duty, duty["finish"]["id"], _held_note("empty"))


def _eje_f_items(agent: Episode, duty: dict[str, Any]) -> dict[str, Any]:
    """Several products asked: an order whose cart lacks a line for one of them is held once per cart state,
    naming the products with no line."""
    if duty["finish"]["kind"] != "order" or agent.remaining_steps() <= 2:
        return duty
    missing, lines = agent._ejc_missing_items()
    memo = f"items {agent.cart_changed_at}"
    if not missing or memo in agent.held_arms:
        return duty
    agent.held_arms.add(memo)
    named = "; ".join("%s (quantity %d)" % (el["product"], el["quantity"]) for el in missing)
    return _hold_job(duty, duty["finish"]["id"], _held_note("items", wanted=len(agent._wanted_items()),
                                                         lines=lines, missing=named))


def _eje_f_steps(agent: Episode, duty: dict[str, Any]) -> dict[str, Any]:
    """A stated step still open before this finish: a confirmation question goes out from code when a message is
    left, any other step holds the finish once and names it."""
    closing = duty["finish"]
    kind = closing["kind"]
    spare = agent.remaining_steps() - 2
    for step_proc in agent._pending_procedures(kind):
        memo = (step_proc["quote"], kind, agent.cart_changed_at)
        if spare <= 0 or memo in agent.proc_held:
            continue
        agent.proc_held.add(memo)
        if step_proc["role"] != "message":
            return _hold_job(duty, closing["id"], _held_note("step_open", do=step_proc["do"], quote=step_proc["quote"]))
        if (agent.messages_used >= agent._message_limit() or not agent.cart_ids or agent.queued_reply
                or agent.remaining_steps() <= 4):
            continue
        item_name = " and ".join(agent._listing_title(idx) for idx in
                                 (agent.cart_ids if agent._wanted_items() else agent.cart_ids[-1:]))
        agent.queued_reply = "Before I place the order: I have chosen {item_name}. Shall I go ahead with it?".format(
            item_name=item_name)
        return _hold_job(duty, closing["id"], _held_note("step_ask", do=step_proc["do"]))
    duty[_ROUTE_KEY] = kind
    return duty


def _eje_f_count(agent: Episode, duty: dict[str, Any]) -> dict[str, Any]:
    """Hold an order once per cart state when the latest cart view shows another unit count."""
    if agent._wanted_items():
        lines = [(idx, agent._ejc_line_units(idx)) for idx in agent.cart_ids]
    else:
        lines = [(idx, agent._wanted_units()) for idx in agent.cart_ids[-1:]]
    if not lines or agent.cart_seen_at < agent.cart_changed_at:
        return duty
    for held, wanted in lines:
        shown = agent.units_in_cart.get(held)
        memo = f"count {held} {agent.cart_changed_at}"
        if shown is None or wanted is None or shown == wanted or agent.remaining_steps() <= 2 or memo in agent.held_arms:
            continue
        agent.held_arms.add(memo)
        return _hold_job(duty, duty["finish"]["id"], _held_note("count", held=held, shown=shown, wanted=wanted))
    return duty


def _eje_repriced_keep(agent: Episode) -> bool:
    """After a refused order on a request with no cheapest or best goal: the one held listing, untouched since,
    was read again after the refusal, its price moved, and it is in stock, within the budget and meets every
    condition by its own data. Such a pick stays right, so its re-order skips the checklist (which tends to
    send the driver after another listing)."""
    if not agent.refusals or agent._wanted_items() or len(agent.cart_ids) != 1:
        return False
    if (agent.plan or {}).get("priority") in ("cheapest", "best") or agent.cart_changed_at > agent.refused_at:
        return False
    held = agent.cart_ids[0]
    facts = agent._row_table().get(held) or {}
    if agent.mentioned.get(held, -1) <= agent.refused_at or facts.get("stock") is not True:
        return False
    if not agent._within_price_cap(facts) or agent._row_passes_literal(facts) is not True:
        return False
    found = _ejc_listing_nodes(agent.records)
    _, key, accepts = _ejc_field_roles(found)[0]
    prices = {carry[key] for ident, carry in found if ident == held and key and accepts(carry.get(key))}
    return len(prices) > 1


def _eje_f_review(agent: Episode, duty: dict[str, Any]) -> dict[str, Any]:
    """The first order after each cart change is held once for a review or the checklist."""
    if agent.order_checked or agent.remaining_steps() <= 2 or agent.seconds_remaining() <= WRAP_UP_WINDOW:
        return duty
    agent.order_checked = True
    warning = ""
    for held in (agent.cart_ids if agent._wanted_items() else agent.cart_ids[-1:]):
        warning = agent._verify_item(held, "about to be ordered")
        if warning:
            break
    if warning:
        return _hold_job(duty, duty["finish"]["id"], f"{_HELD_NOTICE} {warning}")
    if _eje_repriced_keep(agent):
        return duty
    return _hold_job(duty, duty["finish"]["id"], _eje_ORDER_CHECK)


def _eje_f_decline_again(agent: Episode, duty: dict[str, Any]) -> dict[str, Any]:
    """A repeated decline while reviewed listings still pass is held one more time."""
    closing = duty["finish"]
    closing["arm"] = arm_key = "decline " + agent._arm_signature()
    count = agent.decline_holds.get(arm_key, 0)
    if arm_key not in agent.held_arms or not agent.proven_fits:
        return duty
    if count < 1 and agent._has_step_room(2):
        agent.decline_holds[arm_key] = count + 1
        return _hold_job(duty, closing["id"], _held_note("decline_again", ids=", ".join(agent.proven_fits)))
    done = _eje_buy_instead(agent, duty["ctx"])
    return duty if done is None else _hold_job(duty, closing["id"], done)


def _eje_asked_sections(agent) -> set[str]:
    """Section labels of opened listings that the request points at: a choice value the searches sent, or the
    product's own words."""
    tool = agent.roles.get("search")
    shape = (agent.param_schemas.get(tool) or {}).get("properties") or {}
    sent = set()
    for entry in agent.trace:
        for action in (entry.get("actions") or []) if entry.get("kind") == "env" else []:
            if action.get("name") == tool:
                sent |= {str(fv).strip().lower() for kname, fv in (action.get("args") or {}).items()
                         if isinstance(fv, str) and (shape.get(kname) or {}).get("enum")}
    product = agent._ejc_stems(agent.plan.get("product") or "")
    labels = {rw["label"] for rw in agent._row_table().values() if rw["opened"] and isinstance(rw["label"], str)}
    return {lb for lb in labels if lb.strip().lower() in sent or agent._ejc_stems(lb) & product}


def _eje_add_template(agent):
    """The latest cart call naming one listing that was sent or proposed as an add: (args, listing) or None."""
    tool = agent.roles.get("cart")
    props = (agent.param_schemas.get(tool) or {}).get("properties") or {}
    proposed = [_extract_json_dict((tk.get("function") or {}).get("arguments") or "")
                for msg in agent.messages if msg.get("role") == "assistant" for tk in msg.get("tool_calls") or []
                if (tk.get("function") or {}).get("name") == tool]
    for args in reversed([fv for fv in proposed if isinstance(fv, dict)]):
        if agent.add_signature is not None and agent._choice_signature(tool, args) != agent.add_signature:
            continue
        lines = {listing for listing, _ in _eje_cart_lines(props, args) if listing in agent.seen_ids}
        if len(lines) == 1:
            return args, lines.pop()
    return None


def _eje_swap_listing(branch, old, new, units):
    """A copy of a cart call with the listing id replaced; the single whole number next to it becomes `units`."""
    if isinstance(branch, list):
        return [_eje_swap_listing(fv, old, new, units) for fv in branch]
    if not isinstance(branch, dict):
        return new if branch == old else branch
    counts = [kname for kname, fv in branch.items() if isinstance(fv, int) and not isinstance(fv, bool)]
    near = old in branch.values() and len(counts) == 1
    return {kname: units if near and kname == counts[0] else _eje_swap_listing(fv, old, new, units)
            for kname, fv in branch.items()}


def _eje_cart_all_proven(agent) -> bool:
    """Several products: the cart holds one line per requested product, each a fit the decline proof confirmed
    against its own product's requirements, at its own unit count where shown, all together within the budget."""
    many, held = agent._wanted_items(), list(agent.cart_ids)
    indexes = [agent._ejc_item_index(idx) for idx in held]
    if len(held) != len(many) or None in indexes or len(set(indexes)) != len(many) or agent._ejc_missing_items()[0]:
        return False
    row_map = agent._row_table()
    if any(idx not in agent.proven_fits or idx not in row_map or row_map[idx]["price"] is None for idx in held):
        return False
    units = [many[k]["quantity"] for k in indexes]
    if any(agent.units_in_cart.get(idx) not in (None, count) for idx, count in zip(held, units)):
        return False
    cap = _as_number(agent._need_list.budget)
    return cap is None or sum(row_map[idx]["price"] * count for idx, count in zip(held, units)) <= cap


def _eje_buy_instead(agent, gate_ctx):
    """A decline repeated after both holds while listings of the requested section still pass their own data and
    the reviewer: code orders the cheapest such fit instead (one requested product; an empty cart, or a cart
    holding just that fit), or the cart itself when it holds a proven fit for each of several products. Returns
    the note for the declined call, or None to let the decline go."""
    tool, order = agent.roles.get("cart"), agent.roles.get("order")
    if not (tool and order) or agent._pending_procedures("order"):
        return None
    if agent._wanted_items():
        if not _eje_cart_all_proven(agent):
            return None
        closing, issues = _validate_value(agent._minimal_closing_args(order, _ejc_FORCED_TEXT["order"], "order"),
                                          agent.param_schemas[order], "args")
        if issues or agent.remaining_steps() < 1:
            return None
        agent.trace.append({"kind": "decline_swapped", "pick": list(agent.cart_ids), "fits": list(agent.proven_fits)})
        return _format_note("swap", pick=", ".join(agent.cart_ids), said=_eje_send(agent, gate_ctx, order, closing)[0])
    row_map, sections = agent._row_table(), _eje_asked_sections(agent)
    fits = [idx for idx in agent.proven_fits if idx in row_map and row_map[idx]["label"] in sections
            and agent._within_price_cap(row_map[idx]) and agent._row_passes_literal(row_map[idx]) is True]
    held = [idx for idx in agent.cart_ids if idx in fits]
    if held and len(agent.cart_ids) == 1:
        pick, add = held[0], None
    elif fits and not agent.cart_ids:
        pick = min(fits, key=lambda idx: (row_map[idx]["price"] is None, row_map[idx]["price"] or 0, idx))
        template = _eje_add_template(agent)
        if template is None:
            return None
        add, issues = _validate_value(_eje_swap_listing(template[0], template[1], pick, agent._wanted_units()),
                                      agent.param_schemas[tool], "args")
        if issues or _eje_one_listing(agent, tool, add) != pick:
            return None
    else:
        return None
    closing, issues = _validate_value(agent._minimal_closing_args(order, _ejc_FORCED_TEXT["order"], "order"),
                                      agent.param_schemas[order], "args")
    if issues or agent.remaining_steps() < (2 if add else 1):
        return None
    agent.trace.append({"kind": "decline_swapped", "pick": pick, "fits": fits})
    said = []
    if add is not None:
        text, stop = _eje_send(agent, gate_ctx, tool, add)
        said.append(text)
        if stop or agent.finished or pick not in agent.cart_ids:
            return _format_note("swap_cut", pick=pick, said=" | ".join(said))
    said.append(_eje_send(agent, gate_ctx, order, closing)[0])
    return _format_note("swap", pick=pick, said=" | ".join(said))


def _eje_one_listing(agent, tool, args):
    """The one listing a cart call names, or None."""
    props = (agent.param_schemas.get(tool) or {}).get("properties") or {}
    lines = {listing for listing, _ in _eje_cart_lines(props, args) if listing in agent.seen_ids}
    return lines.pop() if len(lines) == 1 else None


def _eje_f_decline_first(agent: Episode, duty: dict[str, Any]) -> dict[str, Any]:
    """The first decline per state is held for the proof; the very first one also runs the audit or the checklist."""
    closing = duty["finish"]
    arm_key = closing["arm"]
    if arm_key in agent.held_arms or not agent._has_step_room(2) or agent.seconds_remaining() <= WRAP_UP_WINDOW + 30:
        return duty
    agent.held_arms.add(arm_key)
    first, agent.decline_checked = not agent.decline_checked, True
    doubt = agent._decline_evidence() or (agent._audit_before_decline() if first else "")
    if doubt:
        return _hold_job(duty, closing["id"], f"{_HELD_NOTICE} {doubt}")
    if not first:
        return duty
    return _hold_job(duty, closing["id"], _eje_DECLINE_CHECK)


def _eje_f_scrub(agent: Episode, duty: dict[str, Any]) -> dict[str, Any]:
    """A decline that goes out carries a remark without listing names when the request wants that."""
    closing = duty["finish"]
    duty["ready"][0] = (closing["id"], closing["name"], agent._clean_decline_args(closing["name"], closing["args"]))
    return duty


def _eje_x_reserve(agent: Episode, duty: dict[str, Any]) -> dict[str, Any]:
    """Non-finishing calls may not use the last step: it stays free for the finish."""
    queue, finishing = duty["ready"], duty["ctx"]["finishing"]
    if not queue or not finishing or queue[0][1] in finishing:
        return duty
    spare = agent.remaining_steps() - 1
    if spare >= len(queue):
        return duty
    cut = max(0, spare)
    note = _SKIP_NOTICE + ": " + _format_note("reserve")
    duty["replies"].extend((ident, note) for ident, _, _ in queue[cut:])
    duty["ready"] = queue[:cut]
    return duty


def _eje_send(agent, gate_ctx, tool, dispatched):
    """Send one call as one step; return (reply text, reason to hold back the rest or "")."""
    action = {"name": tool, "args": dispatched}
    number = agent.step + 1
    reply_obj = agent.call_store(action)
    agent.trace.append({"kind": "env", "step": number, "actions": [action], "reply": reply_obj})
    if reply_obj is None:
        return _format_note("unreached"), _format_note("why_unreached")
    read = _eje_observe(agent, {"name": tool, "args": dispatched, "obs": reply_obj.get("observation"),
                                "closing": bool(reply_obj.get("finished")), "ctx": gate_ctx})
    return read["text"], read.get("halt") or ""


def _eje_x_dispatch(agent: Episode, duty: dict[str, Any]) -> dict[str, Any]:
    """Send the surviving calls one per step. After a result that closes, refuses, errs or brings world lines,
    the rest of the reply is withheld so the model decides again."""
    blocked = ""
    for ident, tool, dispatched in duty["ready"]:
        if blocked:
            text = _format_note("held_over", stop=blocked)
        elif agent.finished or agent.halted or agent.remaining_steps() <= 0:
            text = _format_note("closed")
        else:
            text, blocked = _eje_send(agent, duty["ctx"], tool, dispatched)
        duty["replies"].append((ident, text))
    return duty


def _eje_x_collate(agent: Episode, duty: dict[str, Any]) -> dict[str, Any]:
    """Put the replies back in the order of the proposed calls."""
    place = {}
    for cnt, proposed in enumerate(duty["calls"]):
        place[proposed["id"]] = cnt
    duty["replies"].sort(key=lambda pair: place.get(pair[0], 0))
    return duty


# ---------------------------------------------------------------- observation handlers

def _eje_o_count(agent: Episode, result_map: dict[str, Any]) -> dict[str, Any]:
    """Bookkeeping for one sent call: counters, recent shapes, records and when each id was last mentioned."""
    tool, dispatched, gate_ctx = result_map["name"], result_map["args"], result_map["ctx"]
    result_map["kind"] = agent._classify_call(tool, dispatched)
    if tool == gate_ctx["message"]:
        agent.messages_used += 1
    if result_map["kind"] == "order":
        agent.order_attempts += 1
    for role, bound in agent.roles.items():
        if bound == tool and role not in _CLOSING_JOBS:
            agent.calls_by_role.setdefault(role, []).append(agent.step)
    if tool == gate_ctx["cart"]:
        agent.cart_used = True
        agent.order_checked = False
    resets = tool in gate_ctx["finishing"] or tool in (gate_ctx["cart"], gate_ctx["message"])
    agent.recent = [] if resets else agent.recent[-1:] + [tool + _to_compact_json(dispatched)]
    agent.records.append(result_map["obs"])
    ids: set[str] = set()
    _gather_ids(result_map["obs"], ids)
    agent.mentioned.update(dict.fromkeys(ids, agent.step))
    return result_map


def _eje_o_fault(agent: Episode, result_map: dict[str, Any]) -> dict[str, Any]:
    """Count errors per call shape (a lone text answer only for the identical call); twice holds the shape."""
    tool, dispatched = result_map["name"], result_map["args"]
    code, certain = agent._detect_error(tool, result_map["obs"])
    result_map["fault"] = code
    if not code:
        return result_map
    shape = agent._call_shape(tool, dispatched) if certain else tool + _to_compact_json(dispatched)
    tally = agent.faults.get((shape, code), 0) + 1
    agent.faults[(shape, code)] = tally
    if certain:
        result_map["halt"] = _format_note("why_fault")
    if tally >= 2 and dispatched and tool not in result_map["ctx"]["finishing"]:
        agent.faulty.add(shape)
        result_map["fault_note"] = _format_note("fault_twice", tool=tool, code=code)
    return result_map


def _eje_cart_ids_in(props, dispatched):
    """Free-text single-token argument values of a cart call: the listing ids it names."""
    for key, field_value in dispatched.items():
        tool_desc = props.get(key)
        if isinstance(field_value, str) and not (isinstance(tool_desc, dict) and "enum" in tool_desc):
            token = field_value.strip()
            if token and " " not in token:
                yield field_value


def _eje_cart_lines(props, dispatched):
    """The listing ids a cart call names, each with the units its own line asks for (None when it gives none): free
    one-token values of the call itself, and those of every line object inside a list at any depth (one call
    setting several cart lines, also when an object wraps the list), where the object's single integer is that
    line's units."""
    found = [(field_value, None) for field_value in _eje_cart_ids_in(props, dispatched)]

    def walk(tool_desc, branch, depth):
        if depth > 6:
            return
        tool_desc = tool_desc if isinstance(tool_desc, dict) else {}
        if isinstance(branch, dict):
            inner = tool_desc.get("properties") or {}
            for key, cell in branch.items():
                walk(inner.get(key), cell, depth + 1)
        elif isinstance(branch, list):
            member = tool_desc.get("items") if isinstance(tool_desc.get("items"), dict) else {}
            for line in branch:
                if isinstance(line, dict):
                    counts = [cell for cell in line.values() if isinstance(cell, int) and not isinstance(cell, bool)]
                    for listing in _eje_cart_ids_in(member.get("properties") or {}, line):
                        found.append((listing, counts[0] if len(counts) == 1 else None))
                walk(member, line, depth + 1)

    for key, field_value in dispatched.items():
        if isinstance(field_value, (dict, list)):
            walk(props.get(key), field_value, 1)
    return found


def _eje_view_listings(agent: Episode, seen) -> list[str]:
    """Known listing ids (ids seen in earlier results) that a cart view's records name, in their order."""
    out: list[str] = []
    for record_blob in _iter_dicts([seen]):
        for cell in record_blob.values():
            if isinstance(cell, str) and cell in agent.seen_ids and cell not in out:
                out.append(cell)
    return out


def _eje_o_cart(agent: Episode, result_map: dict[str, Any]) -> dict[str, Any]:
    """Track the cart: the enum values of the first cart call mean adding; free one-token values are ids, also inside
    the line objects of a list argument. A cart view that names known listings is the cart as it stands."""
    tool, dispatched = result_map["name"], result_map["args"]
    if tool == result_map["ctx"]["cart"]:
        mode = agent._choice_signature(tool, dispatched)
        if agent.add_signature is None:
            agent.add_signature = mode
        adding = mode == agent.add_signature
        props = agent.param_schemas[tool].get("properties") or {}
        lines = _eje_cart_lines(props, dispatched)
        if adding and mode == _to_compact_json({}) and len(lines) > len(list(_eje_cart_ids_in(props, dispatched))):
            # a list of lines on a cart tool with no mode choice sets the whole cart: listings it leaves out are gone
            named = [field_value for field_value, _ in lines]
            for gone in [idx for idx in agent.cart_ids if idx not in named]:
                agent.cart_ids.remove(gone)
                agent.units_in_cart.pop(gone, None)
        for field_value, line_units in lines:
            agent.cart_state[field_value] = mode
            rest = [idx for idx in agent.cart_ids if idx != field_value]
            if adding:
                agent.cart_ids = rest + [field_value]
                agent.sent_units[field_value] = (line_units if line_units is not None
                                                 else agent._requested_units(tool, dispatched))
            else:
                agent.cart_ids = rest
                agent.units_in_cart.pop(field_value, None)
        agent.cart_changed_at = agent.step
    if tool == agent.roles.get("cart_view"):
        agent.last_cart_view = result_map["obs"]
        shown = _eje_view_listings(agent, result_map["obs"])
        if shown:   # a view naming no known listing leaves the tracked cart as it is
            agent.cart_ids = [idx for idx in agent.cart_ids if idx in shown] + [
                idx for idx in shown if idx not in agent.cart_ids]
        agent._parse_cart_units(result_map["obs"])
    return result_map


def _eje_o_read(agent: Episode, result_map: dict[str, Any]) -> dict[str, Any]:
    """Unwrap world lines, note a spent allowance and choose what kind of result this is."""
    raw_obj = result_map["obs"]
    result_map["lines"], result_map["inner"] = text_lines, body = _unwrap_text(raw_obj)
    unsettled = bool(result_map["kind"]) and not result_map["closing"] and not agent.finished
    marked = isinstance(raw_obj, dict) and "refusal" in raw_obj   # the runtime's own refusal marker
    result_map["refused"] = marked or (unsettled and not _looks_like_refusal(body))
    result_map["notes"] = []
    if isinstance(raw_obj, (dict, list)):
        _gather_ids(raw_obj, agent.seen_ids)
    if isinstance(body, dict) and body.get("error") == "budget_exceeded":
        tool = result_map["name"]   # another known tool's meter says nothing about the message allowance
        if tool == agent.roles.get("message") or (agent.message_budget is not None and tool not in agent.roles.values()):
            cap = agent.message_budget
            agent.message_budget = agent.messages_used if cap is None else min(cap, agent.messages_used)
        result_map["notes"].append(_format_note("spent"))
    if result_map["refused"]:
        result_map[_ROUTE_KEY] = "refused"
    elif unsettled:
        result_map[_ROUTE_KEY] = "rejected"
    elif text_lines and result_map["name"] != agent.roles.get("message"):
        result_map[_ROUTE_KEY] = "news"
    return result_map


def _eje_o_refused(agent: Episode, result_map: dict[str, Any]) -> dict[str, Any]:
    """A refused finish: new state, the order may be checked again, and cheaper unread rows are named."""
    kind = result_map["kind"]
    result_map["halt"] = _format_note("why_refused")
    agent.refusals += 1
    agent.refused_at = agent.step
    agent._advance_generation()
    agent.order_checked = False
    result_map["notes"].append(_format_note("decline_refused" if kind == "decline" else "order_refused"))
    if kind != "order":
        return result_map
    row_map = agent._row_table()
    lower = _rows_to_recheck(agent, row_map) if row_map else []
    if lower:
        priced = ", ".join("%s (%s)" % (idx, row_map[idx]["price"]) for idx in lower)
        result_map["notes"].append(_format_note("recheck", priced=priced))
    return result_map


def _eje_o_rejected(agent: Episode, result_map: dict[str, Any]) -> dict[str, Any]:
    """A finishing call answered with a fault code."""
    if not result_map.get("halt"):
        result_map["halt"] = _format_note("why_rejected")
    result_map["notes"].append(_format_note("rejected"))
    return result_map


def _eje_o_news(agent: Episode, result_map: dict[str, Any]) -> dict[str, Any]:
    """World lines wrapped around an ordinary result."""
    heard: list[str] = []
    _gather_sentences(result_map["lines"], heard)
    if not heard:
        return result_map
    agent._advance_generation()
    if not result_map.get("halt"):
        result_map["halt"] = _format_note("why_news")
    result_map["notes"].append(_format_note("news", heard=" | ".join(heard[:4])))
    return result_map


def _eje_o_tail(agent: Episode, result_map: dict[str, Any]) -> dict[str, Any]:
    """Last notes: a repeated fault, an empty search, the end of the episode."""
    notes = result_map["notes"]
    if result_map.get("fault_note"):
        notes.append(result_map["fault_note"])
    if result_map["name"] == agent.roles.get("search") and not result_map.get("fault"):
        empty = agent._no_hits_note(result_map["name"], result_map["args"], result_map["inner"])
        if empty:
            notes.append(empty)
    if result_map["closing"]:
        result_map["halt"] = "the episode has ended."
    result_map["note"] = " ".join(notes)
    return result_map


def _eje_add_note(result_map: dict[str, Any], extra: str) -> None:
    if extra:
        result_map["note"] = " ".join(p for p in (result_map["note"], extra) if p)


def _eje_o_listen(agent: Episode, result_map: dict[str, Any]) -> dict[str, Any]:
    """Have the shopper's words interpreted: the answer to a message (with any world lines read separately), or
    the lines riding on any other result plus a refusal's own sentences."""
    body = result_map["inner"]
    heard: list[str] = []
    _gather_sentences(result_map["lines"], heard)
    question = None
    if result_map["name"] == result_map["ctx"]["message"]:
        if heard:   # world lines on the answer are read on their own, never as the answer
            _eje_add_note(result_map, agent.read_shopper_lines(heard, None))
            heard = []
        if isinstance(body, str):
            heard.append(body)
        elif isinstance(body, dict) and "error" not in body and not _looks_like_refusal(body):
            _gather_sentences([fv for fv in body.values() if isinstance(fv, (str, list))], heard)
        question = " ".join(str(fv) for fv in result_map["args"].values() if isinstance(fv, str))
    elif result_map["refused"] and isinstance(body, dict):
        heard += [fv for fv in body.values() if isinstance(fv, str) and len(fv.split()) >= 4]
    if heard:
        _eje_add_note(result_map, agent.read_shopper_lines(heard, question))
    return result_map


def _eje_o_render(agent: Episode, result_map: dict[str, Any]) -> dict[str, Any]:
    """The tool reply text the model sees."""
    parts = [_to_compact_json(result_map["obs"])]
    if result_map["closing"]:
        parts.append(" (this ended the episode)")
    if result_map["note"]:
        parts.append("\n[agent note] " + result_map["note"])
    result_map["text"] = "".join(parts)
    return result_map


# ---------------------------------------------------------------- the stage machine

class _Stage(enum.Enum):
    """Every state a record can be in. Each run starts at one entry state and stops at END."""
    # episode
    OPEN = "open"
    TURNS = "turns"
    SETTLE = "settle"
    CLOSE = "close"
    # one model turn
    GUARD = "guard"
    ANSWER = "answer"
    PLANNER = "planner"
    INFER = "infer"
    CALLS = "calls"
    COMMIT = "commit"
    IDLE = "idle"
    ACT = "act"
    # one planned question
    ASK_GATE = "ask_gate"
    ASK_DRAFT = "ask_draft"
    ASK_BUILD = "ask_build"
    ASK_SEND = "ask_send"
    # executing proposed calls
    SCREEN = "screen"
    FIN_PICK = "fin_pick"
    FIN_UNFIT = "fin_unfit"
    FIN_EMPTY = "fin_empty"
    FIN_STEPS = "fin_steps"
    FIN_COUNT = "fin_count"
    FIN_ITEMS = "fin_items"
    FIN_REVIEW = "fin_review"
    FIN_DECLINE_AGAIN = "fin_decline_again"
    FIN_DECLINE_FIRST = "fin_decline_first"
    FIN_SCRUB = "fin_scrub"
    RESERVE = "reserve"
    DISPATCH = "dispatch"
    COLLATE = "collate"
    # reading one result
    O_COUNT = "o_count"
    O_FAULT = "o_fault"
    O_CART = "o_cart"
    O_READ = "o_read"
    O_REFUSED = "o_refused"
    O_REJECTED = "o_rejected"
    O_NEWS = "o_news"
    O_TAIL = "o_tail"
    O_LISTEN = "o_listen"
    O_RENDER = "o_render"
    END = "end"


_St = _Stage

_STAGE_HANDLERS = {
    _St.OPEN: _stage_start_episode, _St.TURNS: _stage_loop_turns, _St.SETTLE: _stage_settle_episode, _St.CLOSE: _stage_close_episode,
    _St.GUARD: _stage_turn_guard, _St.ANSWER: _stage_send_queued, _St.PLANNER: _stage_run_planner,
    _St.INFER: _stage_call_model, _St.CALLS: _stage_take_calls, _St.COMMIT: _stage_commit_turn, _St.IDLE: _stage_idle_turn,
    _St.ACT: _stage_act_turn,
    _St.ASK_GATE: _stage_question_gate, _St.ASK_DRAFT: _stage_question_draft, _St.ASK_BUILD: _stage_question_build,
    _St.ASK_SEND: _stage_question_send,
    _St.SCREEN: _eje_x_screen, _St.FIN_PICK: _eje_f_pick, _St.FIN_UNFIT: _eje_f_unfit,
    _St.FIN_EMPTY: _eje_f_empty, _St.FIN_STEPS: _eje_f_steps, _St.FIN_COUNT: _eje_f_count, _St.FIN_ITEMS: _eje_f_items,
    _St.FIN_REVIEW: _eje_f_review, _St.FIN_DECLINE_AGAIN: _eje_f_decline_again,
    _St.FIN_DECLINE_FIRST: _eje_f_decline_first, _St.FIN_SCRUB: _eje_f_scrub,
    _St.RESERVE: _eje_x_reserve, _St.DISPATCH: _eje_x_dispatch, _St.COLLATE: _eje_x_collate,
    _St.O_COUNT: _eje_o_count, _St.O_FAULT: _eje_o_fault, _St.O_CART: _eje_o_cart,
    _St.O_READ: _eje_o_read, _St.O_REFUSED: _eje_o_refused, _St.O_REJECTED: _eje_o_rejected,
    _St.O_NEWS: _eje_o_news, _St.O_TAIL: _eje_o_tail, _St.O_LISTEN: _eje_o_listen,
    _St.O_RENDER: _eje_o_render,
}


def _eje_build_table() -> dict[tuple[_Stage, str], _Stage]:
    """(state, outcome) -> next state. "done" follows each line in order, "exit" (a stage put a value under
    "exit") always ends the run; the finish checks and the result reading also branch on named outcomes."""
    S = _St
    text_lines = (
        (S.OPEN, S.TURNS, S.SETTLE, S.CLOSE, S.END),
        (S.GUARD, S.ANSWER, S.PLANNER, S.INFER, S.CALLS, S.COMMIT, S.IDLE, S.ACT, S.END),
        (S.ASK_GATE, S.ASK_DRAFT, S.ASK_BUILD, S.ASK_SEND, S.END),
        (S.SCREEN, S.FIN_PICK, S.FIN_UNFIT, S.FIN_EMPTY, S.FIN_STEPS),   # then by the kind of finish
        (S.FIN_COUNT, S.FIN_ITEMS, S.FIN_REVIEW, S.RESERVE, S.DISPATCH, S.COLLATE, S.END),
        (S.FIN_DECLINE_AGAIN, S.FIN_DECLINE_FIRST, S.FIN_SCRUB, S.RESERVE),
        (S.O_COUNT, S.O_FAULT, S.O_CART, S.O_READ, S.O_TAIL, S.O_LISTEN, S.O_RENDER, S.END),
        (S.O_REFUSED, S.O_TAIL),
        (S.O_REJECTED, S.O_TAIL),
        (S.O_NEWS, S.O_TAIL),
    )
    row_index: dict[tuple[_Stage, str], _Stage] = {}
    for text_line in text_lines:
        for here, there in zip(text_line, text_line[1:]):
            row_index[(here, "done")] = there
    for stage in _STAGE_HANDLERS:
        row_index[(stage, "exit")] = S.END
    for check in (S.FIN_UNFIT, S.FIN_EMPTY, S.FIN_STEPS, S.FIN_COUNT, S.FIN_ITEMS, S.FIN_REVIEW,
                  S.FIN_DECLINE_AGAIN, S.FIN_DECLINE_FIRST):
        row_index[(check, "held")] = S.COLLATE   # nothing is left to reserve or send
    row_index.update({
        (S.FIN_PICK, "pass"): S.RESERVE,
        (S.FIN_UNFIT, "pass"): S.RESERVE,
        (S.FIN_STEPS, "order"): S.FIN_COUNT,
        (S.FIN_STEPS, "decline"): S.FIN_DECLINE_AGAIN,
        (S.O_READ, "refused"): S.O_REFUSED,
        (S.O_READ, "rejected"): S.O_REJECTED,
        (S.O_READ, "news"): S.O_NEWS,
    })
    return row_index


_eje_TABLE = _eje_build_table()


def _run_stages(start: _Stage, agent: Episode, carry: dict[str, Any]) -> dict[str, Any]:
    """Run handlers from `start`, moving by the table on each handler's outcome, until END."""
    stage = start
    while stage is not _St.END:
        carry = _STAGE_HANDLERS[stage](agent, carry)
        result_info = carry.pop(_ROUTE_KEY, "done")
        if "exit" in carry:
            result_info = "exit"
        stage = _eje_TABLE[(stage, result_info)]
    return carry


def _eje_episode(session: Episode, carry: dict[str, Any]) -> dict[str, Any]:
    return _run_stages(_St.OPEN, session, carry)


def _eje_turn(session: Episode, carry: dict[str, Any]) -> dict[str, Any]:
    return _run_stages(_St.GUARD, session, carry)


def _eje_ask(session: Episode, carry: dict[str, Any]) -> dict[str, Any]:
    return _run_stages(_St.ASK_GATE, session, carry)


def _eje_execute(session: Episode, carry: dict[str, Any]) -> dict[str, Any]:
    return _run_stages(_St.SCREEN, session, carry)


def _eje_observe(session: Episode, carry: dict[str, Any]) -> dict[str, Any]:
    return _run_stages(_St.O_COUNT, session, carry)


def _eje_log_crash(episode: Episode, exc: BaseException) -> None:
    """Record an exception that escaped the run."""
    text = "%s: %s" % (type(exc).__name__, exc)
    episode.trace.append({"kind": "crash", "error": text[:500]})


def _eje_rescue(episode: Episode, exc: BaseException) -> list[dict[str, Any]]:
    """After a crash: log it, finish from code when the episode is still open, and hand back the trace."""
    _eje_log_crash(episode, exc)
    try:
        if not (episode.finished or episode.halted):
            episode.finish_now()
    except Exception as again:
        _eje_log_crash(episode, again)
    return episode.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode = Episode(problem_data)
    try:
        return episode.run()
    except Exception as exc:
        return _eje_rescue(episode, exc)


__all__ = ["agent_main"]
