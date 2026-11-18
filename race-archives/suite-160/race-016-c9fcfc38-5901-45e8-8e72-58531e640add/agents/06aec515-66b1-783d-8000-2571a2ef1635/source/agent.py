"""Shopping agent: one episode, one shopper, the tools the world hands out.

Dựng lại thiết kế từ agent public KYO, viết lại bằng cấu trúc và chữ của ta.

The quick model makes the tool calls. Ordinary code tracks the board
(products, needs, money limit, listings met so far), decides which questions go
to the shopper and blocks steps that would be wasted. The stronger model turns
the request into a work order at the start, and is consulted again to review a
cart ahead of checkout and to review the case for closing with no purchase.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agent.proxy_client import ProxyClient

FAST_MODEL = "anthropic/claude-haiku-5.5"
DEEP_MODEL = "anthropic/claude-sonnet-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"

MAIN_TURNS = 42
TURN_CAP = 60
COST_EASE = 0.055
COST_CAP = 0.060
BACKOFF_S = (4, 10, 20, 35)
BACKOFF_BUDGET_S = 45
FIELD_CHARS = 450
ROW_AGAIN_CHARS = 60
OBS_CHARS = 15000
CHAT_CHARS = 150000
DUTY_ROUNDS = 3

# Section labels the prompt builders write in front of each block of context.
LABEL_TOOLS = "TOOLS"
LABEL_MUST_ASK = "MUST ASK"
LABEL_LOG = "LOG"
LABEL_FULL = "FULL"
LABEL_ROW = "ROW-ONLY"
KEY_MATCH_TAG = "[key match"
DUTY_WAIT_KEY = "wait_on"


_RULES_FULL = """Matching rules:
- Pass: the listing's attribute covering that property gives the value. Key spelling is irrelevant, one member of a list suffices, case, singular/plural and common synonyms are tolerated, a value whose words include the asked value's words passes ("dusty rose" for rose), units get converted, bounds are inclusive, a span must lie fully inside its bound.
- Decisive: an attribute named for the requirement's own property; if its value differs, the listing fails. An attribute on some other property, however close (cover colour versus lining colour), counts for nothing.
- With no attribute for the property, the asked value inside another attribute, the title or the description is enough; a figure for a different quantity is not (thickness versus length), nor a bare number with no stated meaning.
- A category name expressing a requirement's value satisfies it; the category name must name the complete requested product.
- A rule to differ from another listing's value passes unless some record shows exactly that value."""


_RULES_SHORT = """Matching rules:
- An attribute named for the requirement's own property decides (key spelling is irrelevant, one member of a list suffices, synonyms and unit conversion are fine, a value whose words include the asked value's words passes, as "dusty rose" for rose, bounds are inclusive). An attribute on some other property, however close (cover colour versus lining colour), counts for nothing.
- With no attribute for the property, the asked value inside another attribute, the title or the description is enough; a bare number with no stated meaning proves nothing.
- A category name expressing a requirement's value satisfies it; the category name must name the item.
- A rule to differ from another listing's value passes unless some record shows exactly that value.
- Unavailable listings and listings over the budget never qualify."""


GUIDE = f"""Role: you are the purchaser acting for a single shopper at a web shop. The shop's tools are your only hands.

Goal. One of two endings is correct:
(a) an order containing each requested product at its requested count, where every chosen listing meets each requirement still active when the order goes in, and the sum stays within the money limit; or
(b) when the shop truly carries nothing that meets every requirement, a close through the shop's no-purchase tool, worded as the request instructs.

Using the shop
1. A tool call spends one step; steps are capped, thinking is not. Every shop has its own tool names, parameter names and result layouts, so read every tool's description and parameter schema, then send exactly the parameter names and values it accepts.
2. Search may be restricted: by category, by a text mode that reads only the first few words, by tiny result pages, by a continuation token or offset. If the answer depends on the full set, set the reported match total against the rows already in hand and continue using the returned token or offset verbatim. Text queries do best when short: value first, then the noun. An empty filtered result normally means a wrong filter value or a wrong filter parameter; retry using the category exactly as rows print it, or switch to a different filter parameter.
3. Result rows are previews. Each shop divides what it knows about a listing differently: one tool may return the attributes (properties), another just price, stock and blurb, and a multi-listing view may return either. When the helper code finds the tool that returns attributes, it names that tool on the board. Judge property requirements from those attributes, so fetch them for every listing you might order; a single call covering several listings is cheaper if it returns what you need. Reading an unchanged listing twice is waste.
4. Questions for the shopper are sent by the helper code, never by you. Answers show up in the chat and get copied to the board.
5. Some requests also demand side tasks: an approval, a message, a check or a read beforehand. The board lists them. Those due right before ordering or closing are run by the helper code the moment you invoke the checkout or closing tool, so do not stall for them and never close without an order on their account; any other side task you do yourself with shop tools (product questions still go via the helper code).

Deciding whether a listing qualifies
- Product kind comes from the listing's category name. Only a category naming the requested product as a whole qualifies; an adjacent category never does.
- For a property requirement, look for the listing's attribute covering that property; it passes when that attribute gives the value. Attribute-name spelling is irrelevant (spacing, underscores, camelCase, prefixes); one member of a list split by commas or slashes suffices. Compare values word by word: case, singular/plural and common synonyms are tolerated, and a value whose words include the asked value's words passes ("dusty rose" passes rose; "extra firm feel" passes firm). Different words fail: a short form or code differs from the full word, and "rose gold" is not "blush pink". Convert units first; bounds are inclusive; a span must lie fully within a bound.
- An attribute whose name is the requirement's property (that name or a common synonym) is decisive: if it says otherwise the listing fails, regardless of other attributes or the title. An attribute on some other property, however close (cover colour versus lining colour), is no evidence either way. Under each opened listing the helper adds {KEY_MATCH_TAG}] lines naming attributes that match a requirement.
- With no attribute for the property, the asked value found inside another attribute (a feature or style list that includes it, say), the title or the description is enough. A figure for a different quantity is not (thickness versus length), nor a number whose text never says what it measures.
- If the category name itself expresses a requirement's value (an audience or season in the name), the category satisfies it.
- A requirement can reference another listing instead of a value ("matching the X of <shop>'s item that has <property>", "no more than N from the X shown on ...'s page"). Settle it first: locate that listing (searching by its category, brand, seller or wording), open its full page with a single-listing or multi-listing tool (preview rows are insufficient), and should that page point onward to another listing for the property, locate and open it as well; the final page supplies the value. An order is right only if you opened every page in that chain. If the asked value is anything other than that listing's value, the requirement rules out only listings with exactly that value, and by itself it is never grounds to close without ordering.
- If the listing you plan to order points you to another page about itself, look at that page first.
- If one requested product must go with another (say, an add-on for a gadget), select the pair jointly, relying on the text of the listings (compatible model lists, a series or model code appearing on both pages). If no add-on meeting its own requirements suits your chosen gadget, swap to another qualifying gadget that such an add-on suits; never close without ordering while this swap is untested.
- Obey exclusions and closed lists. Soft wishes ("if possible", "a bonus") only break ties.
- Never order a listing the shop reports as unavailable. Price times count, summed over all lines, must not exceed the money limit; equal is allowed.

Choosing among qualifying listings
- Cheapest requested (lowest price or lowest total): the order needs the lowest-priced listing of ALL qualifying ones. Collect the full qualifying set (page on: rows are often unsorted by price, and the helper itself reads extra pages of short results), test from the lowest price upward, and keep the first that fully qualifies. If search offers a price ceiling, put it under the price of your best pick so far, leaving only cheaper rivals.
- Highest or lowest of a measure requested: rank every qualifying listing on it and keep the top (or bottom) one; listings that do not state the measure are out.
- In every other case any fully qualifying listing is correct: don't spend steps chasing a better one.
- Add each product to the cart the moment you have opened a fully qualifying listing (for cheapest, most or least: after the comparison), then continue. Cart lines often sell out soon after adding, and adding early keeps steps free for a substitute.
- Different products get one cart line apiece, each at its own count.

Mid-task changes
- Shopper or shop may change the task: a new money limit, requirements added or dropped, prices moving, listings selling out. Such news comes inside tool results, often as a refused order; it counts whichever tool returned it, and the board shows the requirements in force. After a refusal the episode continues: apply all changes, fix the cart, order again.
- Shortly before ordering, re-read the chosen listing's record: price and stock can move silently.

Ending
- Make the cart exactly the chosen lines (remove extras), view it once, then call the checkout tool alone.
- Close without ordering only after real searching shows no listing meets everything. A low step count proves nothing; in that case check out the strongest cart available. Word the close as the request instructs; any remark stays short and names no listing.
- Every reply calls one tool or more. Text stays brief: known facts, then the next action."""


BRIEF = """Below are a shop's tools and a shopper's request. Write a work order for a buying agent as ONE compact JSON object (single line, no other text). Fields:

items: list, one object per separate product to purchase:
  name: the product kind, in the request's wording
  shelf: the shop's category name for it, copied character for character, if a tool exposes category names; else ""
  quantity: integer count of that one listing (default 1; a pair or set sold as one listing is 1)
  musts: list of hard conditions on this product, each formatted "property: constraint" with the precise value, bound, exclusion or closed list
  Writing musts:
  * Call the property by the term the request uses ("meant for a rooftop terrace" -> "setting: rooftop terrace"; "a 90-centimetre cord" -> "cord length: 90 cm"); never substitute a neighbouring property.
  * When the value must be taken from some other listing, write ONE must keeping that listing's whole description ("cover colour: unlike the cover colour of the umbrella Quellmore sells with a carved handle"). Everything describing that other listing (seller, kind, the trailing "with <property> <value>") stays inside it and never becomes a must of the product itself.
  * Price, budget, stock and quantity are never musts.
  * Background (who it is for, where it goes, past trouble, the occasion) becomes a must only if it spells out a property value or bound. Words belonging to the product's name (a season, audience or use attached to it) go in name, not musts.
  * Products joined by "and" inside one "I want ..." sharing one quantity (frequently the name of a department) form ONE item; split only when each has a separate quantity or separate conditions.
budget: the maximum total spend, numeric, in the unit the request uses; else null
budget_unit: that unit spelled out (e.g. "pence"), else null
budget_scope: "total" (limit for the whole order), "each" (per item), or null (no limit)
in_stock_only: true, except when the request plainly accepts unavailable listings
pick: "cheapest" when the cheapest price or smallest sum is wanted; "most" / "least" when an extreme of a measure is wanted; "any" otherwise. Remarks about a tight wallet do not make a pick rule.
pick_measure: name of the measure being pushed to an extreme; "" otherwise
messages: integer cap on messages to the shopper, or null if unstated
steps: integer cap on tool calls (also called steps or moves), or null
lookups: the musts (verbatim) that need a value looked up on another listing the request refers to; else []
rules: each instruction on working method or on finishing, 12 words or fewer apiece
wishes: nice-to-haves only
ask_first: points the request explicitly wants confirmed with the shopper ahead of the choice; [] if none
process: side tasks the request demands besides finding and buying (e.g. obtaining approval from the shopper, informing or showing them something, confirming a point with them, a prior read of something, use of a named tool), each {"step": description in 15 words or fewer, "when": "choose" | "order" | "end", "messages": number of shopper messages it takes, 0 if none}. "choose" means before the products are picked, "order" right before checkout, "end" right before closing without a purchase. Not side tasks: product conditions, budget, stock, message or step caps, how to close when nothing fits. [] when there are none.
process_hint: true when some sentence of the request asks for an action besides finding and buying (an approval, a message, a confirmation, a read beforehand), even one you left out of process; false otherwise
quiet_decline: true if the request forbids suggesting or naming alternatives in case nothing fits; false if alternatives are requested; null when not mentioned
roles: object from job to exact tool name (null if no tool covers the job). Jobs:
  search = finds listings
  open = returns the attributes (properties) of a single listing (if the shop has two single-listing tools, choose the one described as giving specs, attributes or properties)
  offer = returns one listing's live price and stock, only when that is a separate tool from open
  compare = returns full records of several listings
  ask = messages the shopper
  cart = edits the cart
  cart_view = displays the cart
  buy = performs checkout
  refuse = finishes with no purchase
buy_args: constant arguments under which the buy tool performs checkout (if the tool takes a mode parameter, that mode); {} otherwise
refuse_args: constant arguments under which the refuse tool finishes with no purchase; {} otherwise
"""


_PLAN_BODY = """Pick the questions a buying agent will put to its shopper.
Background: a shopper keeps a few firm preferences out of the written request, each concerning a single property of a single requested product. Only a question that mentions that precise property surfaces such a preference, and a reply takes up at most one property, most often the earliest one named, so what leads a question matters most.
Input: the request; the items, each with the points the request already fixes; a numbered list of the attribute keys found when listings in each item's category were opened, showing how many listings have the key and sample values. A hidden preference may sit on a property no listed key covers.
Priority:
 (0) Things the request tells you to confirm with the shopper (shown under {must_ask} if any), each expressed as the property it concerns.
 (1) Hinted properties. The story often hints at a preference without stating it, so go through it sentence by sentence and ask what each detail would make this person care about in the product: a line on something beyond the shopper's judgement or left to you, a person whose taste decides, the look of the product or the place it will live, who will use or wear it, what it will contain or fit, the event or routine it serves, a past purchase that went wrong. For every such line give the single property it hints at and which item, in story order, with the line in "hint". Prefer a property as named among the listed keys of that item; any listed key can be the one hinted, technical ones included. Use the plain property name: a wearer hint -> gender; a looks hint -> colour or finish, whichever is still open in the request.
 (2) After that, listed keys, best first: keys that appear on the majority of that item's listings come first, and among those the ones a buyer of this kind of product decides on early. Keys on few listings still count when nothing better is left.
Leave out: properties already fixed by the request for that item; a synonym of a property already picked; keys listing purchase variants (mixed sizes, colours, pack counts); identifiers; warranty, shipping and similar logistics. For several items: all hinted properties first, then rotate between items.
label = plain name of the property; for a listed key, every word of the key, plainly spelled ("pole height" for poleHeight, "cover texture" for cover_texture, never trimmed to "height" or "texture"); name only, no sample values, no brackets, never two properties.
Return {limit} when that many properties remain, best first, as ONE JSON object only: {{"ask": [{{"item": <its number>, "label": "...", "hint": "<up to 8 words from the story it rests on, or empty>"}}]}}"""


SHOPPER_PICTURE = """A shopper wrote the request below and a buying agent has opened some listings. Before any question is chosen, picture this person.
Read the story, wording and constraints, and the kind of product each item is. For each item say who is likely buying it and for what, and which properties of the listed ones would change which product this person takes, most likely first. Name properties as they are listed. Say what you are unsure about, at most 120 words. Output ONE JSON object only: {"notes": "..."}"""


def plan_prompt(limit):
    """Question-planning instructions allowing at most ``limit`` questions."""
    return _PLAN_BODY.format(must_ask=LABEL_MUST_ASK, limit=limit)


HEAR = """The shopper answered questions from a buying agent. Read the answers. Output ONE JSON object only, with:
needs: [{"item": <its number>, "need": "<attr>: <rule>", "points": <whether the value must be looked up on another listing: true or false>}]
drop: [{"item": <its number>, "property": "<requirement given up by the shopper>"}]
free: [{"item": <its number>, "property": "<property of a question answered with indifference>"}]
crowded: [question numbers whose answer complains about too many topics in one message or asks for one question at a time; not answers that just say they didn't follow]
unclear: [question numbers whose answer says the shopper did not catch or follow the question, or asks for it again; not answers that say they do not mind, which go under free]
budget: the revised overall limit (a number) or null
How to fill needs:
- Every firm preference stated, with exact value or bound and unit ("at most 45 cm", "round", "not nylon"). Label it with whichever property the answer mentions, which need not be the one asked about; answers about one shared property merge into a single need. Keep the value verbatim, nouns and units included ("at least 2 pockets", never "at least 2").
- An answer that sets the value by reference to another listing (similar to, different from or close to the value on a seller's product, which is singled out through some of its properties) gives ONE need, for the named property, with points set to true; its value copies the answer exactly, from the comparing word through the end of the description, keeping whatever it mentions about a listing named on that page; no trimming, no paraphrase. Properties that only describe the other product never become this item's needs.
- Folksy or roundabout phrasing that conveys a value still counts as stating it."""


NOTE = """A note appeared while a buying agent was working (its source may be the shop or the shopper). Output ONE JSON object only:
kind: "change" if it alters the task (budget, requirements, prices, stock, or the order getting refused); "error" if it only reports a bad or rejected call; otherwise "other"
budget: revised overall limit, using the unit of the request; null if none
gone: [ids of listings reported unavailable]
add: [{"item": <its number>, "need": "<attr>: <rule>"}]
drop: [{"item": <its number>, "need": "<requirement now unwanted>"}]
sold_out: true when the note reports that a picked listing sold out, vanished or cannot be purchased any more, named or not
price_moved: true when the note reports a price change
summary: a brief sentence
Treat a replaced requirement as drop (old) plus add (new). Report every change."""


CHECK = ("""Final review before a buying agent places its order: from the records below, judge whether the cart is exactly correct.
""" + _RULES_FULL + """
Cart rules: one separate line per requested item, at its requested quantity; no extra lines; no line shown as unavailable; the sum of price times quantity across lines within the budget. When a listing has several records, the newest counts.
Under a cheapest/most/least pick, scan the remaining listings too: a listing that clearly meets all requirements and wins on that rule over the cart's choice makes the cart wrong.
Output ONE JSON object only: {"ok": true|false, "problems": [short concrete problems], "better": [ids of listings to swap into the cart, strongest first]}. Return ok=false only when the records plainly show the problem, or when no record covers a requirement.""")


READING = _RULES_SHORT


AUDIT = ("""A buying agent intends to close without ordering, believing nothing qualifies. Test the listing records below against the requirements.
""" + READING + f"""
{LABEL_ROW} records come from search previews only; their attributes are unread, so a property the preview does not show can neither pass nor fail them.
Output ONE JSON object only: {{"fits": {{"<item number>": [ids that seem to meet everything, cheapest first, up to 4]}}, "open": [{LABEL_ROW} ids of a requested product type, affordable within the budget, where opening the page could prove a fit, most promising first, up to 6], "reason": "<one sentence>"}}""")


UNMIX = """Context: a request was turned into hard requirements on a product. A few borrow a value from a separate listing that the request mentions, and the words identifying the separate listing (type, seller, properties attached by "with ...") are at times wrongly taken as requirements of the purchase. Judging by the request's wording, tell which entries under OTHER REQUIREMENTS merely identify the separate listing. Output ONE JSON object only: {"describe_other": [those requirements, verbatim; [] when each is a genuine requirement on the product]}"""


TRAIL = f"""Checkout is next for a buying agent. Some requirements do not state a value but borrow it from a different listing, and one listing's page may point onward to yet another listing for it. Every listing on every chain must be fully opened by the agent before the order is right. Trace each requirement's chain, beginning with the wording of the requirement itself and moving link to link through the pages' text, using the records below ({LABEL_FULL} = page opened by the agent; {LABEL_ROW} = search preview only, page unread).
Output ONE JSON object only: {{"chains": [{{"need": <the requirement, verbatim>, "links": [{{"id": <id of the record below that is this link, or "">, "about": <the way the requirement or the preceding page describes it>, "seek": <when id is empty, the few words to search for that listing: its seller or brand and the values it is described by, nothing else>}}], "done": <true if the final link is a {LABEL_FULL} page that gives the value of the property itself, otherwise false>, "value": <that value as the final page gives it, or "">}}]}}
Give links in order; end the chain at the earliest link whose page is unopened or missing; beyond that point the text is unknown."""


_DUTY_BODY = """A buying agent's next move is to {stage}. The request demands the side tasks listed below. Using the numbered {log} of the agent's actions, judge for each task due before it can {stage} whether it is complete for the present cart. Complete means: some {log} line shows it carried out, and, for a task touching the cart or order, the line must postdate the latest cart change and its reply must approve. Questions about product properties never complete a task.
If a task is still open, give the single next shop call that performs it, a tool listed under {tools} together with arguments. A shopper message is sent with the messaging tool: one brief plain-language sentence or question limited to that task, naming no product property or value the task doesn't require. Give no call when all due tasks are complete, or when the remaining messages or steps cannot complete the open one.
Output ONE JSON object only: {{"steps": [{{"step": <task>, "done": true|false, "lines": [{log} line numbers proving it]}}], "call": null or {{"tool": "<name>", "args": {{...}}}}, "{wait}": <what a reply wants changed first, or "">, "needs": [{{"item": <its number>, "need": "<attr>: <rule>"}} for every firm requirement stated in such a reply]}}"""


def duty_prompt(stage):
    """The side-task check run before ``stage`` (e.g. "place the order")."""
    return _DUTY_BODY.format(stage=stage, log=LABEL_LOG, tools=LABEL_TOOLS, wait=DUTY_WAIT_KEY)


SWAP = ("""A listing in a buying agent's cart just sold out. Look through the remaining listing records below for substitutes for that cart line.
""" + READING + """
Output ONE JSON object only: {"fits": [{"id": <the listing's id>, "price": <numeric price exactly as recorded>}, one per listing that appears to satisfy all requirements of the item on the sold-out line, up to 6], "reason": "<one sentence>"}""")


SHAPE = """Below is a result returned by a shop tool. Point out its parts using the precise key names found in the result and in the tool's parameters. Output ONE JSON object only: {"rows": <key of the array of listings>, "count": <key with the overall match count (not the number shown), or null>, "next": <key whose value requests the next page, or null>, "next_arg": <tool parameter that receives it, or null>, "price": <row-level key carrying the price, or null>, "stock": <row-level key telling whether it is purchasable right now, or null>, "price_cap": <tool parameter that sets the highest price a row may have, or null>}"""


# ---------------------------------------------------------------------------
# small pure helpers


def dump(value):
    """Compact JSON text, or ``str`` when the value cannot be encoded."""
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    return text


def clip(value, limit, depth=0):
    """Shorten every string in a nested value to ``limit`` characters; give up below depth 7."""
    if depth > 7:
        return "..."
    if isinstance(value, list):
        return [clip(part, limit, depth + 1) for part in value]
    if isinstance(value, dict):
        return {name: clip(part, limit, depth + 1) for name, part in value.items()}
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "..."
    return value


_FENCED = re.compile(r"^```[A-Za-z]*\s*(.*?)\s*```$", re.S)


def _decode_dict(text):
    """(parsed, result): parsed is False when the text is not JSON at all."""
    try:
        got = json.loads(text)
    except ValueError:
        return False, None
    return True, (got if isinstance(got, dict) else None)


def json_obj(text):
    """The JSON object in a model reply: whole text, fenced block, or the outermost braces."""
    if not isinstance(text, str):
        return None
    body = text.strip()
    fenced = _FENCED.match(body)
    if fenced:
        body = fenced.group(1)
    parsed, got = _decode_dict(body)
    if parsed:
        return got
    start, stop = body.find("{"), body.rfind("}")
    if start < 0 or start >= stop:
        return None
    return _decode_dict(body[start:stop + 1])[1]


def plain(key):
    """A key in lower-case words: camelCase, snake_case, dots and dashes split apart."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", spaced).strip().lower()


def _singular(word):
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def stems(text):
    """The set of plain words in ``text`` with a trailing plural 's' dropped."""
    return frozenset(map(_singular, plain(text).split()))


def names_match(prop, key):
    """Whether a key's word set names a property's word set."""
    if not (prop and key):
        return False
    if prop <= key:
        return True
    return len(key) > 1 and key <= prop


def _one_edit(a, b):
    """Whether two words are the same or differ by one inserted, dropped or replaced character."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:] or (len(a) == len(b) and a[i + 1:] == b[i + 1:])


def _folded(given, asked):
    """The given words, each long one swapped for the asked word it misspells by a single character."""
    return frozenset(next((a for a in asked if len(a) >= 5 and len(w) >= 5 and _one_edit(w, a)), w) for w in given)


FIT_MARKS = ("=", "⊃", "⊂", "≠")
_RULE_WORDING = re.compile(r"[,;]|\b(?:not|no|non|only|or|and|without|except|any|nothing|either)\b", re.I)
_MEMBER_SPLIT = re.compile(r"[,/;|]|\s+and\s+", re.I)


def value_fit(asked, given):
    """How a listing's value compares with an asked one: "=" the same words (or one member of a list), "⊃" the asked
    words among more, "⊂" only part of the asked words, "≠" other words; "?" when the asked value carries a number
    (bounds and units are the model's to judge), reads as a rule rather than one value, or either side is empty."""
    want = stems(asked)
    if not want or any(ch.isdigit() for ch in str(asked)) or _RULE_WORDING.search(str(asked)):
        return "?"
    best = "?"
    for part in _MEMBER_SPLIT.split(str(given)) + [str(given)]:
        have = _folded(stems(part), want)
        if not have:
            continue
        mark = "=" if have == want else "⊃" if want < have else "⊂" if have < want else "≠"
        if best == "?" or FIT_MARKS.index(mark) < FIT_MARKS.index(best):
            best = mark
    return best


def items(value):
    """A value as a list: lists pass through, a non-empty string or dict becomes a one-entry list."""
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, (str, dict)) and value else []


_WHOLE_TEXT = re.compile(r"\s*\d+(\.0+)?\s*")


def whole(value):
    """A whole number from an int, an integral float or a digit string; None otherwise."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str) and _WHOLE_TEXT.fullmatch(value):
        return int(float(value))
    return None


def looks_id(value, named=False):
    """A 3-64 character token without spaces; it also needs a digit unless its key says id."""
    if not isinstance(value, str):
        return False
    core = value.strip()
    if " " in core or not 3 <= len(core) <= 64:
        return False
    return True if named else re.search(r"\d", value) is not None


def _says_id(key):
    return "id" in str(key).lower()


def learn_id_key(rows):
    """Pick the row identifier: id-like and distinct on every row; a name containing id ranks first, then longer values."""
    first = rows[0]
    ranked = []
    for key in first:
        named = _says_id(key)
        raw = [row.get(key) for row in rows]
        if not all(looks_id(v, named) for v in raw):
            continue
        cleaned = [v.strip() for v in raw]
        if len(set(cleaned)) != len(cleaned):
            continue
        ranked.append(((named, sum(map(len, cleaned)) / len(cleaned)), key))
    if not ranked:
        return None
    return max(ranked, key=lambda pair: pair[0])[1]


def _record_has(record, ident):
    if not isinstance(record, dict):
        return False
    return ident in [str(v).strip() for v in record.values() if isinstance(v, str)]


def in_rows(node, ident, depth=0):
    """Whether some list inside ``node`` contains a record carrying ``ident`` as a value."""
    if depth > 5:
        return False
    if isinstance(node, dict):
        return any(in_rows(part, ident, depth + 1) for part in node.values())
    if not isinstance(node, list):
        return False
    if any(_record_has(part, ident) for part in node):
        return True
    return any(in_rows(part, ident, depth + 1) for part in node)


def _scalar(value):
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def fact_block(node):
    """The child object of ``node`` with the most scalar fields (at least two); first one wins ties."""
    if not isinstance(node, dict):
        return None
    best = None
    for child in node.values():
        if not isinstance(child, dict) or len(child) < 2:
            continue
        flat = {k: v for k, v in child.items() if _scalar(v)}
        if len(flat) < 2:
            continue
        if best is None or len(flat) > len(best):
            best = flat
    return best


# ---------------------------------------------------------------------------
# bending tool arguments toward a JSON schema


def kinds(schema):
    """The type names a schema declares, as a list."""
    declared = schema.get("type")
    if isinstance(declared, str):
        return [declared]
    return list(filter(lambda name: isinstance(name, str), declared or []))


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_TYPE_TESTS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": _is_number,
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def fits_kind(value, kind):
    """Whether a value has the JSON type ``kind``; unknown type names accept anything."""
    test = _TYPE_TESTS.get(kind)
    return True if test is None else test(value)


_NUMBER_TEXT = re.compile(r"\s*-?\d+(\.\d+)?\s*")
_MISSING = object()


def _to_number(value, kind):
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if not (isinstance(value, str) and _NUMBER_TEXT.fullmatch(value)):
        return _MISSING
    number = float(value)
    return int(number) if kind == "integer" or number.is_integer() else number


def _to_flag(value):
    word = value.strip().lower() if isinstance(value, str) else None
    return word == "true" if word in ("true", "false") else _MISSING


_CONVERTERS = {
    "integer": lambda v: _to_number(v, "integer"),
    "number": lambda v: _to_number(v, "number"),
    "string": lambda v: str(v) if _is_number(v) else _MISSING,
    "boolean": _to_flag,
    "array": lambda v: _MISSING if v is None or isinstance(v, list) else [v],
}


def _convert(value, kind):
    """``value`` turned into ``kind`` when a safe conversion exists, else the marker _MISSING."""
    turn = _CONVERTERS.get(kind)
    return _MISSING if turn is None else turn(value)


class Fitter:
    """Coerces a value to fit a JSON schema; problems it cannot repair go into ``faults``."""

    def __init__(self, faults):
        self.faults = faults

    def note(self, text):
        self.faults.append(text)

    def fit(self, value, schema, where):
        if not isinstance(schema, dict):
            return value
        wanted = kinds(schema)
        if wanted and not self._typed(value, wanted):
            value = self._retype(value, wanted)
            if not self._typed(value, wanted):
                self.note(f"{where} must be of type {'/'.join(wanted)}")
                return value
        value = self._choices(value, schema, where)
        if isinstance(value, str):
            value = self._text(value, schema, where)
        if _is_number(value):
            self._range(value, schema, where)
        if isinstance(value, list):
            value = self._array(value, schema, where)
        if isinstance(value, dict):
            value = self._object(value, schema, where)
        return value

    @staticmethod
    def _typed(value, wanted):
        return any(fits_kind(value, kind) for kind in wanted)

    @staticmethod
    def _retype(value, wanted):
        for kind in wanted:
            turned = _convert(value, kind)
            if turned is not _MISSING:
                return turned
        return value

    def _choices(self, value, schema, where):
        allowed = schema.get("enum")
        if not isinstance(allowed, list) or value in allowed:
            return value
        if isinstance(value, str):
            wanted = value.strip().lower()
            same = next((opt for opt in allowed if isinstance(opt, str) and opt.lower() == wanted), _MISSING)
            if same is not _MISSING:
                value = same
        if value not in allowed:
            self.note(f"{where} must be one of {allowed}")
        return value

    def _text(self, value, schema, where):
        longest, shortest = schema.get("maxLength"), schema.get("minLength")
        if isinstance(longest, int) and len(value) > longest:
            value = value[:longest]
        if isinstance(shortest, int) and len(value) < shortest:
            self.note(f"{where} is shorter than allowed")
        return value

    def _range(self, value, schema, where):
        low, high = schema.get("minimum"), schema.get("maximum")
        if isinstance(low, (int, float)) and value < low:
            self.note(f"{where} may not go under {low}")
        if isinstance(high, (int, float)) and value > high:
            self.note(f"{where} may not go over {high}")

    def _array(self, value, schema, where):
        most, least, inner = schema.get("maxItems"), schema.get("minItems"), schema.get("items")
        if isinstance(most, int) and len(value) > most:
            self.note(f"{where} takes no more than {most} entries")
        if isinstance(least, int) and len(value) < least:
            self.note(f"{where} takes no fewer than {least} entries")
        if not isinstance(inner, dict):
            return value
        return [self.fit(part, inner, f"{where}[{n}]") for n, part in enumerate(value)]

    def _object(self, value, schema, where):
        props = schema.get("properties")
        if not isinstance(props, dict):
            props = {}
        if schema.get("additionalProperties") is False:
            for name in [n for n in value if n not in props]:
                self.note(f"{where}.{name} is not an accepted argument")
        for name in schema.get("required") or []:
            if name not in value:
                self.note(f"{where}.{name} is missing")
        return {name: (self.fit(part, props.get(name), f"{where}.{name}") if name in props else part)
                for name, part in value.items()}


def fit_args(value, schema, where, faults):
    """``value`` bent toward ``schema``; problems that cannot be repaired are appended to ``faults``."""
    return Fitter(faults).fit(value, schema, where)


def fitted_call(host, tool, args):
    """A store call with its arguments fitted to the tool's schema; None when they do not fit, nothing came back,
    or the episode closed with it."""
    faults = []
    shaped = fit_args(args, host.schemas.get(tool), "args", faults)
    if faults:
        return None
    reply = host.call(tool, shaped)
    return None if reply is None or host.finished else reply


# ---- state, model access, store access, episode clock ----


CLOCK_SOFT_S = 235.0
CLOCK_HARD_S = 262.0
CLOCK_MARGIN_S = 6.0
STORE_WAIT_FLOOR_S = 10.0
MODEL_WAIT_FLOOR_S = 8.0
STORE_END_S = 295.0
TIME_WARN_S = 90.0


class Clock:
    """Wall time of one episode, counted from the moment the clock is made."""

    def __init__(self, soft_s=CLOCK_SOFT_S, hard_s=CLOCK_HARD_S, now=None):
        self.now = now or time.monotonic
        self.soft_s = soft_s
        self.hard_s = hard_s
        self.began = self.now()

    def used_s(self):
        return self.now() - self.began

    def left_s(self):
        return self.hard_s - self.used_s()

    def soft_left_s(self):
        return self.soft_s - self.used_s()

    def gone(self):
        return self.left_s() <= 0

    def room_for(self, seconds):
        """True while a pause of this length still leaves the margin before the hard limit."""
        return seconds + CLOCK_MARGIN_S < self.left_s()

    def timeout(self, usual, floor, end_s=None):
        """The usual timeout of a request, shortened when less time than that is left before the end mark."""
        left = (self.hard_s if end_s is None else end_s) - self.used_s()
        return max(floor, min(usual, left - 2.0))


@dataclass(eq=False)
class Item:
    name: str
    shelf: str
    qty: int
    musts: list[str]
    told: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    free: list[str] = field(default_factory=list)


@dataclass
class Spend:
    cost: float = 0.0
    calls: int = 0
    loop_calls: int = 0
    waited: int = 0
    misses: int = 0
    no_credit: bool = False
    broke: bool = False

    CALL_CAP = 60

    def out_of_calls(self):
        return self.calls >= self.CALL_CAP

    def stopped(self):
        return self.broke or self.no_credit or self.out_of_calls()


@dataclass
class Talk:
    cap: int = 0
    used: int = 0
    kept: int = 0
    duty_msgs: int = 0
    duties: list[dict[str, Any]] = field(default_factory=list)
    duty_done: set[tuple[str, str]] = field(default_factory=set)
    record: list[str] = field(default_factory=list)
    record_steps: list[int] = field(default_factory=list)
    asked: bool = False
    heard: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def free(self):
        """Messages open to the interviewer once the ones held back for duties at the order or the end are set aside."""
        still_held = self.kept - self.duty_msgs
        return self.cap - self.used - (still_held if still_held > 0 else 0)

    def due(self, stage, mark):
        if (stage, mark) in self.duty_done:
            return False
        return any(duty["when"] == stage for duty in self.duties)

    def finish_steps(self, mark):
        """Steps saved for the end: one for the checkout, plus one while an order duty still has a message to send."""
        speaks = any(duty["when"] == "order" and duty["messages"] > 0 for duty in self.duties)
        if speaks and self.used < self.cap and self.due("order", mark):
            return 2
        return 1


@dataclass
class Roles:
    jobs: dict[str, str] = field(default_factory=dict)
    shows_facts: dict[str, bool] = field(default_factory=dict)
    id_args: set[tuple[str, str]] = field(default_factory=set)
    compare_ok: int = 0
    compare_bad: int = 0

    def page_tools(self):
        """Views that count as a listing's own page: the offer view when it is separate from the attributes view,
        otherwise the single-listing view; the side-by-side view always counts."""
        own = self.jobs.get("offer") or self.jobs.get("open")
        found = set()
        for tool in (own, self.jobs.get("compare")):
            if tool:
                found.add(tool)
        return found

    def facts_tool(self, target):
        """The single-listing tool used for attributes: one known to return them, else the reading's pick."""
        for tool, showed in self.shows_facts.items():
            if showed and target(tool):
                return tool
        named = self.jobs.get("open")
        if self.shows_facts.get(named or "") is False or not target(named):
            return None
        return named

    def compare_facts(self):
        return self.shows_facts.get(self.jobs.get("compare") or "")

    def compare_bare(self, target):
        """The side-by-side view cannot be trusted for attributes: it has shown none, or it is untried while
        another view already gives them (or a separate offer view exists)."""
        state = self.compare_facts()
        if state is False:
            return True
        if state is not None:
            return False
        other = self.facts_tool(target) or ""
        return self.shows_facts.get(other) is True or bool(self.jobs.get("offer"))


@dataclass
class Books:
    epoch: int = 0
    basket: dict[str, int] = field(default_factory=dict)
    basket_view: Any = None
    basket_seen: int = -1
    basket_moved: int = -1
    basket_touched: bool = False
    basket_said: str = ""
    best_basket: list[str] = field(default_factory=list)
    dropped: set[str] = field(default_factory=set)
    offer_seen: dict[str, int] = field(default_factory=dict)
    gone: dict[str, int] = field(default_factory=dict)
    sold_lines: set[str] = field(default_factory=set)
    note_sold: bool = False
    held: int = 0
    refused_at: int = -1
    shortlist: list[str] = field(default_factory=list)
    refuse_blocks: int = 0
    pushed: bool = False
    late_held: bool = False
    refused_mark: str = ""
    held_marks: set[str] = field(default_factory=set)
    vouched: str | None = None
    queued: list[str] = field(default_factory=list)
    buys: int = 0
    pointers: list[tuple[int, str]] = field(default_factory=list)
    pointed: dict[str, str] = field(default_factory=dict)

    def mark(self):
        lines = sorted(self.basket.items())
        try:
            return json.dumps(lines, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(lines)


@dataclass
class Pages:
    replay_cache: dict[str, tuple[int, int, str]] = field(default_factory=dict)
    recent: list[str] = field(default_factory=list)
    turned_down: dict[str, tuple[str, int, str]] = field(default_factory=dict)


def _model_link():
    return ProxyClient(timeout=50, max_retries=1, retry_delay=1)


def _store_link():
    return ProxyClient(timeout=95, max_retries=3, retry_delay=1)


class Model:
    """Chat completions through the proxy: tiers tried in order, short paced resends, spend kept in Spend."""

    PAUSES = (4, 10, 20, 35)
    PAUSE_BUDGET = 45
    USUAL_TIMEOUT = 50.0
    RESEND = frozenset((402, 408, 425, 429))

    def __init__(self, spend, clock, trace, link=None,
                 sleep=None, jitter=None):
        self.spend = spend
        self.clock = clock
        self.trace = trace
        self.link = link if link is not None else _model_link()
        self.sleep = sleep or time.sleep
        self.jitter = jitter or self._spread

    @staticmethod
    def _spread():
        """A number in [0, 1) taken from the clock, so paced resends of parallel episodes do not line up."""
        return (time.monotonic() * 7919.0) % 1.0

    def _worth_resending(self, status):
        if status is None or status in self.RESEND:
            return True
        return isinstance(status, int) and status >= 500

    def _resends_allowed(self, status):
        # a timed-out request already used up its own wait, so it goes out once more at most
        return 1 if status in (None, 408) else len(self.PAUSES)

    def turn(self, messages, tiers, max_tokens, tools=None):
        refused = 0
        for tier in tiers:
            if self.spend.out_of_calls() or self.clock.gone():
                return None
            self.spend.calls += 1
            reply, status = self._try_tier(tier, messages, max_tokens, tools)
            if reply is not None:
                return reply
            refused += status == 402
        # a large request can be refused for credit while a cheaper tier still goes through
        if refused == len(tiers) and any(tier != DEEP_MODEL for tier in tiers):
            self.spend.no_credit = True
        return None

    def _try_tier(self, tier, messages, max_tokens, tools):
        """One tier: the request, then paced resends while the failure is a passing one and time and the pause budget allow."""
        resent = 0
        reply, status = self._once(tier, messages, max_tokens, tools)
        while reply is None and self._may_resend(status, resent):
            pause = self.PAUSES[resent]
            self.spend.waited += pause
            resent += 1
            self.sleep(pause * (0.8 + 0.4 * self.jitter()))
            reply, status = self._once(tier, messages, max_tokens, tools)
        return reply, status

    def _may_resend(self, status, resent):
        if not self._worth_resending(status) or resent >= self._resends_allowed(status):
            return False
        pause = self.PAUSES[resent]
        return self.spend.waited + pause <= self.PAUSE_BUDGET and self.clock.room_for(pause)

    @staticmethod
    def _with_cache_marks(tier, messages):
        """For Anthropic tiers the first and the last message are marked for prompt caching."""
        if len(messages) < 2 or not tier.startswith("anthropic/"):
            return messages
        marked = list(map(dict, messages))
        for message in {id(marked[0]): marked[0], id(marked[-1]): marked[-1]}.values():
            text = message.get("content")
            if text and isinstance(text, str):
                message["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        return marked

    def _once(self, tier, messages, max_tokens, tools):
        """One request: (message, 200), or (None, status) where 0 means the reply came back empty."""
        body = {"model": tier, "messages": self._with_cache_marks(tier, messages), "max_tokens": max_tokens, "temperature": 0}
        body.update({"tools": tools, "tool_choice": "auto"} if tools else {})
        self.link.timeout = self.clock.timeout(self.USUAL_TIMEOUT, MODEL_WAIT_FLOOR_S)
        result = self.link.post_verbose("/inference/chat/completions", body)
        data = result.data if isinstance(result.data, dict) else None
        if data and data.get("choices"):
            return self._read_reply(tier, data)
        return self._failed(tier, result, data)

    def _read_reply(self, tier, data):
        spent = (data.get("usage") or {}).get("cost")
        if isinstance(spent, (int, float)):
            self.spend.cost += float(spent)
        message = data["choices"][0].get("message") or {}
        if message.get("content") or message.get("tool_calls"):
            return message, 200
        self.trace.append({"kind": "llm_fail", "model": tier, "status": 0})
        return None, 0

    def _failed(self, tier, result, data):
        failure = result.error or {}
        status = failure.get("status")
        inner = (data or {}).get("error")
        if isinstance(inner, dict):
            # an error carried inside a normal reply, under the provider's own code
            status, failure = whole(inner.get("code")), {"body": inner.get("message")}
        self.trace.append({"kind": "llm_fail", "model": tier, "status": status, "body": str(failure.get("body") or "")[:200]})
        return None, status

    def json(self, prompt, tiers, max_tokens=1800):
        """Ask one tier at a time for a JSON object; move to the next tier when the reply holds none."""
        for tier in tiers:
            reply = self.turn([{"role": "user", "content": prompt}], [tier], max_tokens)
            parsed = json_obj((reply or {}).get("content"))
            if parsed is not None:
                return parsed
            if self.spend.stopped():
                return None
        return None


class World:
    """Store calls for one episode: the step counter, the finished flag and the step-conflict repairs."""

    ATTEMPTS = 4
    USUAL_TIMEOUT = 95.0
    GIVE_UP = frozenset((400, 401, 402, 403, 404, 422))
    CLOSED_WORDS = ("finish", "quarantin", "finaliz")
    SKIP_WORDS = ("reuse", "different action")

    def __init__(self, binding, spend, clock, trace,
                 finished=False, link=None, after=None):
        self.binding = binding
        self.spend = spend
        self.clock = clock
        self.trace = trace
        self.link = link if link is not None else _store_link()
        self.after = after
        self.step = 0
        self.finished = finished

    def _payload(self, name, args):
        sent = dict(self.binding)
        sent["step"] = self.step + 1
        sent["action"] = {"name": name, "args": args}
        return sent

    def _took(self, name, args, data):
        number = data.get("step")
        self.step = number if isinstance(number, int) else self.step + 1
        if data.get("finished"):
            self.finished = True
        if self.after is not None:
            self.after(name, args, data.get("observation"))
        return data

    def _repair(self, body):
        """After a step conflict: True to send again with a fixed step, False to stop, None when the episode closed."""
        low = body.lower()
        if any(word in low for word in self.CLOSED_WORDS):
            self.finished = True
            return None
        wanted = re.search(r"expected (\d+)", body)
        if wanted:
            self.step = int(wanted.group(1)) - 1
            return True
        if any(word in low for word in self.SKIP_WORDS):
            self.step += 1
            return True
        return False

    def call(self, name, args):
        for _ in range(self.ATTEMPTS):
            self.link.timeout = self.clock.timeout(self.USUAL_TIMEOUT, STORE_WAIT_FLOOR_S, STORE_END_S)
            result = self.link.post_verbose("/environment/call", self._payload(name, args))
            if isinstance(result.data, dict):
                return self._took(name, args, result.data)
            failure = result.error or {}
            status = failure.get("status")
            text = str(failure.get("body") or "")
            self.trace.append({"kind": "env_fail", "status": status, "body": text[:200]})
            if status == 409:
                if self._repair(text):
                    continue
                return None
            if status in self.GIVE_UP:
                if status == 402:
                    self.spend.broke = True
                return None
        return None


# Intake (reading the request) and Board (the notes shown to the model each turn).


# A second, heavier reading of only the "process" part of the request, made when the first reading
# found no step for checkout or ending but flagged that the request seems to ask for one.
RECHECK_DUTIES = True
# Closing arguments read from the request are checked against the tool's schema; a mismatch is asked again.
CHECK_FINISH_ARGS = True
# The task's own step limit comes in the view; a smaller count read from the request is used only
# when that number is written in the request itself.
STEPS_NEED_SOURCE = False

DUTIES = """Go over the shopper's request once more and list only the errands it sets apart from locating and paying for the right goods: a yes from the shopper before payment, letting the shopper know the total or the listing picked, settling a detail with them, viewing a certain page first, or working through a named tool. Invented examples: "send me the final basket to approve before paying" is one errand at "order" costing 1 message; "check the warranty page before choosing" is one errand at "choose" costing 0 messages.
Not errands: conditions on the goods, the money limit, availability, caps on messages or steps, and how to finish if nothing qualifies.
Reply with ONE JSON object only: {"process": [one entry per errand, {"step": the errand in 15 words or fewer, "when": "choose" (ahead of picking goods), "order" (right ahead of the checkout) or "end" (right ahead of finishing with no purchase), "messages": shopper messages the errand uses, 0 if it uses none}], "messages": the message allowance the request states (an integer), or null if it states none}. Give "process": [] when the request sets no errand."""

SPLIT_FINISH = """One store tool, {tool}, either places the order or finishes with no purchase, depending on what it is given. Its parameters: {schema}
Reply with ONE JSON object only: {{"buy_args": the fixed arguments that place the order, "refuse_args": the fixed arguments that finish with no purchase}}. Give only the arguments that pick between the two."""

STAGE_WORDS = {
    "choose": "before picking products",
    "order": "at checkout (done for you when you check out)",
    "end": "when finishing without a purchase (done for you then)",
}


def _tidy(*parts):
    return " ".join(" ".join(str(p) for p in parts).split())


def _positive_number(value):
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if value > 0 else None


def _stage_of(value):
    stage = str(value or "").strip().lower()
    return stage if stage in STAGE_WORDS else "order"


def _read_duties(raw):
    found = []
    for entry in items(raw):
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("step") or "").strip()
        if text:
            found.append({"step": text, "when": _stage_of(entry.get("when")),
                          "messages": max(0, whole(entry.get("messages")) or 0)})
    return found


PICK_SHELVES = True

SHELF = """A shop files its listings under the category names below. Pick the one that names the product "{product}" from the request as a whole, not an accessory, a part or a neighbouring kind.
{names}
Reply with ONE JSON object only: {{"shelf": "<one of the names above, copied exactly>"}}"""


def _enum_labels(schemas):
    """Category-like names offered by the tools: string enum options with a letter, in schema order, no repeats."""
    out = []
    todo = list(schemas.values())
    while todo:
        node = todo.pop(0)
        if not isinstance(node, dict):
            continue
        for option in node.get("enum") or []:
            if isinstance(option, str) and re.search(r"[A-Za-z]", option) and option not in out:
                out.append(option)
        todo.extend(v for v in node.values() if isinstance(v, dict))
    return out


def _covering(product, names):
    """(name, words left over) for each name whose words hold all of the product's longer words, fewest left over first."""
    wanted = {w for w in stems(product) if len(w) > 3}
    if not wanted:
        return []
    found = [(n, len(stems(n) - wanted)) for n in names if wanted <= stems(n)]
    return sorted(found, key=lambda pair: pair[1])


@dataclass
class Brief:
    """What the request asks for, as read once at the start. Other keys of the reading stay in raw."""
    raw: dict = field(default_factory=dict)
    items: list = field(default_factory=list)
    jobs: dict = field(default_factory=dict)
    pointers: list = field(default_factory=list)
    budget: object = None
    talk_cap: int = 0
    talk_kept: int = 0
    steps: int = 30
    duties: list = field(default_factory=list)
    rechecked: bool = False

    def get(self, key, default=None):
        return self.raw.get(key, default)

    def __getitem__(self, key):
        return self.raw[key]

    def __setitem__(self, key, value):
        self.raw[key] = value


def _item_from(entry):
    count = whole(entry.get("quantity"))
    musts = [str(m) for m in items(entry.get("musts")) if m]
    return Item(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), count if count and count > 0 else 1, musts)


class Intake:
    """Reads the request against the tool list once, at the start, and fills a Brief."""

    def __init__(self, model, tools, schemas, request, log):
        self.model = model
        self.tools = tools
        self.schemas = schemas
        self.request = request
        self.log = log

    def _broke(self):
        return self.model.spend.no_credit

    def _menu(self):
        rows = []
        for tool in self.tools:
            spec = tool["function"]
            rows.append("- " + spec["name"] + ": " + str(spec.get("description", ""))
                        + " | parameters: " + dump(spec.get("parameters") or {}))
        return "\n".join(rows)

    def read(self, view):
        text = BRIEF + "\nTOOLS:\n" + self._menu() + "\n\nREQUEST:\n" + self.request
        if view.get("opening") is not None:
            text += "\n\nOPENING NOTE FROM THE STORE:\n" + dump(clip(view.get("opening"), 1500))
        raw, heavy_read = self._first_reading(text)
        brief = Brief(raw=raw, steps=self._view_steps(view))
        brief.jobs = {job: tool for job, tool in (raw.get("roles") or {}).items()
                      if isinstance(tool, str) and tool in self.schemas} if isinstance(raw.get("roles"), dict) else {}
        self._finish_args(brief)
        self._goods(brief)
        self._shelves(brief)
        self._link_pointers(brief)
        self.untangle(brief)
        brief.budget = _positive_number(raw.get("budget"))
        if RECHECK_DUTIES and not heavy_read:
            self._recheck_duties(brief)
        self._limits(brief)
        return brief

    def _first_reading(self, text):
        # the light model reads first; the heavy one only when no product or no checkout tool came back
        raw = self.model.json(text, [FAST_MODEL], 4000) or {}
        jobs = raw.get("roles")
        whole_enough = items(raw.get("items")) and isinstance(jobs, dict) and jobs.get("buy")
        if whole_enough or self._broke():
            return raw, False
        again = self.model.json(text, [DEEP_MODEL, FAST_MODEL], 2500)
        return (again, True) if again else (raw, False)

    def _view_steps(self, view):
        given = view.get("max_steps")
        count = whole(given)
        if count is None and isinstance(given, float):
            count = int(given)
        return count if count and count > 0 else 30

    def _finish_args(self, brief):
        raw = brief.raw
        for key in ("buy_args", "refuse_args"):
            if not isinstance(raw.get(key), dict):
                raw[key] = {}
        tool = brief.jobs.get("buy")
        if not tool or tool != brief.jobs.get("refuse"):
            return
        split = raw["buy_args"] and raw["refuse_args"] and raw["buy_args"] != raw["refuse_args"]
        if split and self._fit(tool, raw["buy_args"]) and self._fit(tool, raw["refuse_args"]):
            return
        ask = SPLIT_FINISH.format(tool=tool, schema=dump(self.schemas.get(tool)))
        got = self.model.json(ask, [FAST_MODEL, DEEP_MODEL], 600) or {}
        place, leave = got.get("buy_args"), got.get("refuse_args")
        if isinstance(place, dict) and isinstance(leave, dict) and place != leave:
            raw["buy_args"], raw["refuse_args"] = place, leave

    def _fit(self, tool, args):
        """Whether fixed closing arguments are accepted by the tool's schema (other required fields aside)."""
        if not CHECK_FINISH_ARGS:
            return True
        faults = []
        fit_args(args, self.schemas.get(tool), "args", faults)
        return not [f for f in faults if not f.endswith(" is missing")]

    def _goods(self, brief):
        entries = [e for e in items(brief.raw.get("items")) if isinstance(e, dict)]
        brief.items.extend(_item_from(e) for e in entries)
        if not brief.items:
            brief.items.append(Item("the requested product", "", 1, []))

    def _shelves(self, brief):
        """With category names in a tool's schema: the names holding all of the product's own words with the fewest
        words left over are the best fits. The reading's category stays when it is one of them (or nothing fits);
        otherwise a single best fit is taken, and the light model picks between the best fits and the reading's own."""
        names = _enum_labels(self.schemas) if PICK_SHELVES else []
        for good in brief.items if names else []:
            exact = next((n for n in names if n.lower() == good.shelf.strip().lower()), "")
            ranked = _covering(good.name, names)
            best = [n for n, extra in ranked if extra == ranked[0][1]]
            if not ranked or exact in best:
                chosen = exact
            else:
                options = best + ([exact] if exact else [])
                chosen = options[0] if len(options) == 1 else self._tie(good, options, exact)
            if chosen and chosen != good.shelf:
                self.log.append({"kind": "shelf", "item": good.name, "was": good.shelf, "now": chosen})
                good.shelf = chosen

    def _tie(self, good, best, exact):
        ask = SHELF.format(product=good.name, names="\n".join("- " + n for n in best)) + "\n\nREQUEST:\n" + self.request
        got = self.model.json(ask, [FAST_MODEL], 300) or {}
        pick = str(got.get("shelf") or "").strip()
        return pick if pick in best else (exact or best[0])

    def _link_pointers(self, brief):
        for raw_text in items(brief.raw.get("lookups")):
            text = str(raw_text).strip()
            exact = any(text in good.musts for good in brief.items)
            if not exact:
                # copied loosely from the musts: take the must about the same property instead
                label = plain(text.split(":")[0])
                text = next((m for good in brief.items for m in good.musts
                             if plain(m.split(":")[0]) == label), text)
            owner = 1
            for number, good in enumerate(brief.items, 1):
                if text in good.musts:
                    owner = number
                    break
            if text and (owner, text) not in brief.pointers:
                brief.pointers.append((owner, text))

    def untangle(self, brief):
        """Conditions that merely identify the other listing a pointer names, mistaken for conditions on the
        product itself, are moved inside that pointer."""
        for number, good in enumerate(brief.items, 1):
            aimed = [need for owner, need in brief.pointers if owner == number and need in good.musts]
            others = [m for m in good.musts if m not in aimed]
            if aimed and others:
                self._unmix_item(brief, number, good, aimed, others)

    def _unmix_item(self, brief, number, good, aimed, others):
        text = "".join([UNMIX, "\n\nREQUEST:\n", self.request, "\n\nPRODUCT: ", good.name,
                        "\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n", "\n".join(aimed),
                        "\nOTHER REQUIREMENTS:\n", "\n".join(others)])
        got = self.model.json(text, [DEEP_MODEL, FAST_MODEL], 500) or {}
        named = {str(x).strip() for x in items(got.get("describe_other"))}
        moved = [m for m in others if m in named]
        if not moved:
            return
        head = aimed[0]
        merged = f"{head} (that listing: {'; '.join(moved)})"
        good.musts = [merged if m == head else m for m in good.musts if m not in moved]
        brief.pointers = [(number, merged) if pair == (number, head) else pair for pair in brief.pointers]
        self.log.append({"kind": "unmix", "item": number, "moved": moved})

    def _recheck_duties(self, brief):
        raw = brief.raw
        first = _read_duties(raw.get("process"))
        if any(d["when"] in ("order", "end") for d in first):
            return
        if raw.get("process_hint") is not True or self._broke():
            return
        got = self.model.json(DUTIES + "\n\nREQUEST:\n" + self.request, [DEEP_MODEL], 1200)
        if not isinstance(got, dict):
            return
        second = _read_duties(got.get("process"))
        brief.rechecked = True
        self.log.append({"kind": "duties_recheck", "first": len(first), "second": clip(second, 120)})
        if second:
            raw["process"] = second
        if whole(raw.get("messages")) is None and whole(got.get("messages")) is not None:
            raw["messages"] = got.get("messages")

    def _limits(self, brief):
        raw = brief.raw
        asks = bool(brief.jobs.get("ask"))
        cap = whole(raw.get("messages"))
        if cap is not None and cap >= 0:
            brief.talk_cap = cap
        elif asks:
            brief.talk_cap = 3
        brief.steps = self._step_cap(whole(raw.get("steps")), brief.steps)
        brief.duties = _read_duties(raw.get("process"))
        # messages kept back for the steps due at checkout or at the ending: the largest such need
        wants = [d["messages"] for d in brief.duties if d["when"] in ("order", "end") and d["messages"]]
        if asks:
            brief.talk_kept = min(max(wants), brief.talk_cap) if wants else 0

    def _step_cap(self, asked, steps):
        """A smaller step limit read from the request replaces the task's own (10 at least)."""
        if asked is None or not 10 <= asked < steps:
            return steps
        if STEPS_NEED_SOURCE and not re.search(r"(?<!\d)" + str(asked) + r"(?!\d)", self.request):
            return steps
        return asked


class Board:
    """The running notes shown to the model: what to buy (text) and where the episode stands (status).

    Reads the driver's live state: items, budget, pointers, pointed, duties, brief (the other keys of the
    reading), step, max_steps, left(), keep_for_finish(), buys, basket_touched, needs_now(), shortlist,
    notes, facts_tool(), page_tools(), shows_facts, jobs."""

    def __init__(self, desk):
        self.desk = desk

    def _product_line(self, number, good, several):
        pointed = self.desk.pointed
        wanted = []
        for need in good.musts + good.told:
            if need in good.dropped:
                continue
            wanted.append(need + (" (value read off the page it points to: " + pointed[need] + ")" if need in pointed else ""))
        line = ("product " + str(number) + " - " if several else "product - ") + good.name
        if good.shelf:
            line += " [category label: " + good.shelf + "]"
        line += ", how many: " + str(good.qty) + "; must have: "
        line += "; ".join(wanted) if wanted else "nothing past being this kind of product"
        if good.free:
            line += "; shopper is fine with any " + ", ".join(good.free)
        return line

    def _choice_line(self, brief):
        pick = brief.get("pick")
        if pick == "cheapest":
            return "choose: the lowest-priced listing that meets everything"
        if pick in ("most", "least"):
            return _tidy("choose: the fitting listing with the", pick, brief.get("pick_measure") or "")
        return "choose: any listing that meets everything"

    def _where_facts(self):
        desk = self.desk
        source = desk.facts_tool()
        trusted = source and (desk.shows_facts.get(source) is True or desk.jobs.get("offer"))
        if not trusted:
            return None
        pages = desk.page_tools()
        silent = sorted(t for t in pages | {desk.jobs.get("offer") or ""} if t and desk.shows_facts.get(t) is False)
        others = sorted(pages - {source})
        bits = ["a listing's attributes are shown by " + source]
        bits += [", ".join(silent) + " show no attributes"] if silent else []
        bits += ["a listing's own page (what following a chain of pages or checking a page means) opens with "
                 + " or ".join(others)] if others else []
        return "; ".join(bits)

    def text(self):
        desk = self.desk
        brief = desk.brief
        goods = desk.items
        lines = [self._product_line(n, g, len(goods) > 1) for n, g in enumerate(goods, 1)]
        if desk.budget is not None:
            reach = "per product" if brief.get("budget_scope") == "each" else "for the whole order"
            lines.append(_tidy("spend limit:", desk.budget, brief.get("budget_unit") or "", reach))
        lines.append(self._choice_line(brief))
        if brief.get("in_stock_only", True) is not False:
            lines.append("only listings available now")
        lines += ["nice to have (only to break a tie): " + str(w) for w in items(brief.get("wishes"))]
        lines += ["the request says: " + str(r) for r in items(brief.get("rules"))]
        if brief.get("quiet_decline") is True:
            lines.append("if nothing meets everything: finish without naming or proposing any listing")
        lines += ["requested step, " + STAGE_WORDS[d["when"]] + ": " + d["step"] for d in desk.duties]
        facts = self._where_facts()
        if facts:
            lines.append(facts)
        return "\n".join("* " + line for line in lines)

    def _pace(self, remaining):
        desk = self.desk
        if remaining <= desk.keep_for_finish() + 1:
            return (" LAST STEPS: once the cart has a line for every product, check out now; finish without buying"
                    " only if your searches proved some product has no fitting listing (being short of steps proves nothing).")
        unbought = desk.buys == 0
        if remaining <= 6 or (unbought and remaining <= len(desk.items) + 5):
            return " Only a few steps remain: put the chosen lines in the cart and check out now."
        if unbought and remaining <= 12:
            return " Steps are running thin: settle each pick, put it in the cart and finish with a checkout within a few steps."
        if unbought and not desk.basket_touched and desk.step >= desk.max_steps // 2:
            return (" Half of the steps are gone and the cart is still empty: put each product's fitting listing"
                    " in the cart now (stock can still run out).")
        return ""

    def status(self):
        desk = self.desk
        remaining = desk.left()
        out = "[desk] " + str(desk.step) + "/" + str(desk.max_steps) + " steps used, " + str(remaining) + " remain."
        out += self._pace(remaining)
        current = {need for _, need in desk.needs_now()}
        unread = [need for _, need in desk.pointers if need in current and need not in desk.pointed]
        if unread and desk.buys == 0:
            out += ("\n[look these up first: open the listing each one refers to and every listing its page sends you on to;"
                    " the final page holds the value] " + " | ".join(unread[:3]))
        if desk.shortlist:
            out += "\n[shortlist to review] " + ", ".join(desk.shortlist)
        if desk.notes:
            out += "\n[recent changes] " + " | ".join(desk.notes[-4:])
        split = desk.legwork.budget_line()
        if split:
            out += "\n[steps per product] " + split
        return out + self._time_line() + "\n[board]\n" + self.text()

    def _time_line(self):
        desk = self.desk
        if not desk.paced:
            return ""
        seconds = desk.clock.left_s()
        if seconds >= TIME_WARN_S:
            return ""
        return f"\n[time] about {max(0, int(seconds))} seconds of working time remain: settle the cart and finish now."


# ---- listings met so far, and the layout of row-list results ----

SKIP_KNOWN = True       # a property already asked or settled is neither asked again nor offered to the planner
WRAPPED_RECORDS = True  # a single listing record wrapped under any key of a result counts as shown
def _stock_turn(record, node):
    """-1 when a true flag of the record turns false in the new copy, +1 when a false one turns true, else 0."""
    pairs = [(record.get(k), v) for k, v in node.items()]
    if any(was is True and now is False for was, now in pairs):
        return -1
    return 1 if any(was is False and now is True for was, now in pairs) else 0


class Catalog:
    """Everything learned about store listings from tool results: one merged record per listing id, the full
    attribute record of listings whose page was read, and which fields each viewing tool has shown."""

    def __init__(self, schemas, gone, epoch, wrapped=True):
        self.schemas = schemas
        self.gone = gone
        self.epoch = epoch
        self.wrapped = wrapped
        self.seen = {}
        self.opened_facts = {}
        self.order_seen = []
        self.pages_read_ids = set()
        self.unavailable = set()
        self.id_key = None
        self.price_key = None
        self.stock_key = None
        self.tool_keys = {}
        self.view_keys = {}

    def id_of(self, node, in_list=False, depth=0):
        """The listing id a record carries, or None when the node does not look like a listing."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and looks_id(node.get(self.id_key), True):
            return node[self.id_key].strip()
        holds_dict = any(isinstance(v, dict) for v in node.values())
        if not (in_list or holds_dict or depth == 0):
            return None
        key = learn_id_key([node])
        if not key:
            return None
        value = node[key].strip()
        return value if "id" in str(key).lower() or value in self.seen else None

    def _rows_list(self, node):
        """Whether a list reads as result rows; learns the id key from it when the current one does not fit."""
        records = [r for r in node if isinstance(r, dict) and len(r) >= 3]
        if not records or 2 * len(records) < len(node):
            return False
        known = self.id_key
        if known and all(looks_id(r.get(known), True) for r in records):
            return True
        learned = learn_id_key(records)
        if learned:
            self.id_key = learned
        return learned is not None

    def _merge(self, ident, node):
        if ident not in self.seen:
            self.seen[ident] = {}
            self.order_seen.append(ident)
        record = self.seen[ident]
        turn = _stock_turn(record, node)
        if turn < 0:
            self.unavailable.add(ident)
        elif turn > 0:
            self.unavailable.discard(ident)
        record.update(node)
        if fact_block(node):
            self.opened_facts[ident] = dict(record)
        elif ident in self.opened_facts:
            self.opened_facts[ident].update(node)

    def take(self, observation):
        """Fold every listing record found in a tool result into the books."""
        stack = [(observation, 0, False)]
        while stack:
            node, depth, in_list = stack.pop()
            if depth > 6:
                continue
            if isinstance(node, list):
                as_rows = self._rows_list(node)
                stack.extend((child, depth + 1, as_rows) for child in reversed(node))
            elif isinstance(node, dict):
                ident = self.id_of(node, in_list, depth)
                if ident:
                    self._merge(ident, node)
                else:
                    stack.extend((child, depth + 1, False) for child in reversed(list(node.values())))

    def sold_out(self, ident):
        return ident in self.unavailable or self.gone(ident)

    def labels(self):
        """Lower-cased enum options anywhere in the tool schemas: the store's own category names."""
        out = set()
        todo = [(schema, 0) for schema in self.schemas.values()]
        while todo:
            schema, depth = todo.pop()
            if depth > 6 or not isinstance(schema, dict):
                continue
            out.update(o.strip().lower() for o in schema.get("enum") or [] if isinstance(o, str))
            todo.extend((v, depth + 1) for v in schema.values() if isinstance(v, dict))
        return out

    def category_key(self):
        """The row key that names a listing's category: short text on most rows, few distinct values, store labels first."""
        values = {}
        carried = {}
        for record in self.seen.values():
            for key, value in record.items():
                if not isinstance(value, str) or not 0 < len(value) <= 40 or re.search(r"\d{3}", value):
                    continue
                values.setdefault(key, set()).add(value)
                carried[key] = carried.get(key, 0) + 1
        floor = 0.8 * max(1, len(self.seen))
        known = self.labels()

        def rank(key):
            options = values[key]
            on_label = sum(v.strip().lower() in known for v in options) / len(options)
            return (-on_label, len(options), -carried[key], -sum(map(len, options)) / len(options))

        fit = [k for k in values if carried[k] >= floor and len(values[k]) <= 40]
        return min(fit, key=rank) if fit else None

    @staticmethod
    def _wanted(items):
        out = []
        for item in items:
            shelf = item.shelf or ""
            long = {w for w in stems(shelf) | stems(item.name) if len(w) >= 4}
            out.append((plain(shelf), plain(item.name), long))
        return out

    @staticmethod
    def _asked_for(label, wanted):
        flat = plain(label)
        long = {w for w in stems(label) if len(w) >= 4}
        return any(flat in (shelf, name) or long & words for shelf, name, words in wanted)

    def key_table(self, items):
        """Rows of (category, attribute name, listings showing it, up to four sample values) over opened listings,
        categories of the requested items only when any match, biggest category first, most carried attribute first."""
        board = self._opened_by_label()
        wanted = self._wanted(items)
        labels = [label for label in board if self._asked_for(label, wanted)] or list(board)
        rows = []
        for label in sorted(labels, key=lambda name: board[name][0], reverse=True):
            kept = []
            for name, per in sorted(board[label][1].items(), key=lambda pair: len(pair[1]), reverse=True):
                twin = any(name.startswith(prev + " ") and per == other for prev, other in kept)
                if not (twin or self._noisy(per)):
                    kept.append((name, per))
            for name, per in kept[:20]:
                samples = list(dict.fromkeys(per.values()))[:4]
                rows.append((label, name, len(per), [text[:40] for text in samples]))
        return rows

    def _opened_by_label(self):
        """Per category label of opened listings: [listings opened, {attribute name: {listing id: its value}}]."""
        cat = self.category_key()
        board = {}
        for ident in self.order_seen:
            record = self.opened_facts.get(ident)
            if record:
                label = str(record.get(cat) or "listings") if cat else "listings"
                entry = board.setdefault(label, [0, {}])
                entry[0] += 1
                facts = fact_block(record) or {}
                for name, value in zip(map(plain, facts), facts.values()):
                    if name and ident not in entry[1].setdefault(name, {}):
                        entry[1][name][ident] = str(value).strip().lower()
        return board

    @staticmethod
    def _noisy(per):
        """An attribute no shopper would pick by: long text, variant lists, or a different long value on every listing."""
        texts = list(per.values())
        if not texts:
            return True
        mean = sum(map(len, texts)) / len(texts)
        if mean > 40 or 2 * sum(" / " in t for t in texts) >= len(texts):
            return True
        return len(texts) >= 4 and mean > 12 and len(set(texts)) == len(texts)

    def groups(self, items):
        """Listing ids by category, the requested items' categories when any match, largest first, one per item."""
        cat = self.category_key()
        by_label = {}
        for ident in self.order_seen:
            label = str(self.seen[ident].get(cat) or "") if cat else ""
            by_label.setdefault(label, []).append(ident)
        wanted = self._wanted(items)
        pool = [ids for label, ids in by_label.items() if self._asked_for(label, wanted)] or list(by_label.values())
        pool.sort(key=len, reverse=True)
        return pool[: max(1, len(items))]

    def single_target(self, tool):
        """The one free-text parameter a tool needs, when that is all it requires."""
        if not tool:
            return None
        schema = self.schemas.get(tool) or {}
        needed = schema.get("required") or []
        free = [k for k, v in (schema.get("properties") or {}).items() if isinstance(v, dict) and "enum" not in v and "string" in kinds(v)]
        target = next((k for k in needed if k in free), None)
        if target is None and len(free) == 1:
            target = free[0]
        if not target or any(k != target for k in needed):
            return None
        return target

    def price_of(self, ident):
        """The newest price seen for a listing, read under the price key named for result rows."""
        if not self.price_key:
            return None
        value = (self.seen.get(ident) or {}).get(self.price_key)
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            return float(value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def buyable(self, ident):
        """False once anything marked the listing gone, or its stock field reads false or zero."""
        if self.sold_out(ident):
            return False
        stock = (self.seen.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        if stock is False:
            return False
        return not (isinstance(stock, (int, float)) and not isinstance(stock, bool) and stock <= 0)

    def _listed(self, rows):
        out = []
        for row in rows if isinstance(rows, list) else []:
            ident = self.id_of(row, True) if isinstance(row, dict) else None
            if ident:
                out.append((ident, row))
        return out

    def records_in(self, observation):
        """Listing records a result shows: itself, rows of its lists, and, one level down, rows of nested lists
        and single records wrapped under any key."""
        if isinstance(observation, list):
            return self._listed(observation)
        if not isinstance(observation, dict):
            return []
        ident = self.id_of(observation)
        if ident:
            return [(ident, observation)]
        out = []
        for value in observation.values():
            if isinstance(value, list):
                out += self._listed(value)
            elif isinstance(value, dict):
                inner = self.id_of(value, True) if self.wrapped else None
                if inner:
                    out.append((inner, value))
                for sub in value.values():
                    out += self._listed(sub)
        return out

    def note_view(self, tool, observation):
        """Record which fields this tool shows and which each listing has shown since the store last changed."""
        now = self.epoch()
        for ident, record in self.records_in(observation):
            fields = set(record)
            self.tool_keys.setdefault(tool, set()).update(fields)
            when, shown = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (now, fields | (shown if when == now else set()))

    def shown_already(self, tool, ident):
        """True when every field this tool shows was already shown for the listing in the current store state."""
        fields = self.tool_keys.get(tool)
        if not fields or self.gone(ident):
            return False
        when, shown = self.view_keys.get(ident, (-1, set()))
        return when == self.epoch() and fields <= shown


def _row_lists(observation):
    """Keys of a result whose value is a list of two or more records."""
    if not isinstance(observation, dict):
        return []
    return [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]


def _named_parts(reply, observation, rows, params):
    """The key names a model gave for a result's parts, each kept only when it really is there."""
    first = observation[rows][0]

    def key_in(slot, inside):
        key = reply.get(slot)
        return key if isinstance(key, str) and key in inside else None
    count = key_in("count", observation)
    if count and (isinstance(observation[count], bool) or not isinstance(observation[count], int)):
        count = None
    return dict(rows=rows, count=count, next=key_in("next", observation), next_arg=key_in("next_arg", params),
                price=key_in("price", first), stock=key_in("stock", first), price_cap=key_in("price_cap", params))


class ResultShape:
    """Learns, a single time for each tool, how its row-list results are laid out: where the rows sit, where the
    overall match count sits, which key continues to a further page and which parameter accepts it, and which row
    fields hold price and availability."""

    def __init__(self, catalog, ask, log):
        self.catalog = catalog
        self.ask = ask
        self.log = log
        self.known = {}

    def of(self, tool, observation):
        if tool in self.known:
            return self.known[tool]
        lists = _row_lists(observation)
        if not lists:
            return None
        self.known[tool] = None
        schema = self.catalog.schemas.get(tool) or {}
        preview = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = "\n\n".join([SHAPE, "Tool " + tool + ", parameters " + dump(schema),
                              "Result, trimmed to its first rows:\n" + dump(clip(preview, 120))[:4000]])
        reply = self.ask(prompt, [FAST_MODEL, DEEP_MODEL], 1500) or {}
        rows = reply.get("rows")
        if not rows or rows not in lists:
            return None
        shape = _named_parts(reply, observation, rows, schema.get("properties") or {})
        self.known[tool] = shape
        self.catalog.price_key = self.catalog.price_key or shape["price"]
        self.catalog.stock_key = self.catalog.stock_key or shape["stock"]
        self.log({"kind": "shape", "tool": tool, "shape": shape})
        return shape

    def paging(self, tool, observation):
        """(matches overall, rows returned now, the paging parameter, the value that continues, None once finished)."""
        shape = self.of(tool, observation)
        ready = shape and shape["count"] and shape["next"] and shape["next_arg"] and isinstance(observation, dict)
        if not ready:
            return None
        rows, total, token = (observation.get(shape[k]) for k in ("rows", "count", "next"))
        whole_count = isinstance(total, int) and not isinstance(total, bool)
        usable_token = token is None or (isinstance(token, (str, int)) and not isinstance(token, bool))
        if not (isinstance(rows, list) and rows and whole_count and usable_token):
            return None
        return total, len(rows), shape["next_arg"], token

    def price_sorted(self, observation):
        prices = [self.catalog.price_of(i) for i, _ in self.catalog.records_in(observation)]
        if len(prices) < 3 or any(p is None for p in prices):
            return False
        return all(a <= b for a, b in zip(prices, prices[1:]))


Shape = ResultShape


# ---- steps taken without a model turn ----

PAGE_READ_CAP = 6      # extra search pages read without a model turn, per episode
CHEAPER_LOOKS = 3      # cheaper same-kind rows whose attributes get read before a lowest-price checkout
LINEUP_LOOKS = 5       # listings opened one by one when no side-by-side view shows attributes
LINEUP_SPARE = 22      # steps that must stay unspent while that lineup is opened
LANE_FLOOR = 3         # listings of each item group opened before the questions are chosen
LANE_SPARE = 14        # steps that must stay unspent while those are opened
SEARCH_STOP = False    # pause a repeated empty code-word search; off by default
RESCAN_CHEAPER = True  # search again under a price ceiling before a lowest-price checkout
EARLY_ASK = False      # two rounds of questions, the first in the opening steps
MAIN_WAVE_STEP = 6     # the main round starts by this step at the latest

ORDERED_PICKS = ("cheapest", "most", "least")
_CODE_RX = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-./]*")
_NAMED_NUMBER_RX = re.compile(r"\b[A-Z][a-z]+\s+(\d{3,})\b")


def _no_price(value):
    return float("inf") if value is None else value


def _later_page(given):
    """True when the call itself already asked for a page past the first."""
    if isinstance(given, int) and not isinstance(given, bool):
        return given not in (0, 1)
    return bool(str(given or "").strip())


def _code_words(args):
    """Code-like words in a call's text arguments: letters mixed with two or more digits, or a
    capitalised name followed by a bare number of three or more digits."""
    found = []
    for value in args.values():
        if not isinstance(value, str):
            continue
        for word in _CODE_RX.findall(value):
            digits = sum(ch.isdigit() for ch in word)
            if digits >= 2 and digits < len(word) and len(word) >= 4 and any(ch.isalpha() for ch in word):
                found.append(word)
        found += [m.group(0) for m in _NAMED_NUMBER_RX.finditer(value)]
    return found


class Legwork:
    def __init__(self, host):
        self.d = host
        self.pages_read = 0
        self.last_pages = []
        self.wider_done = False
        self.cheaper_done = set()
        self.empty_runs = {}
        self.found_by = {}
        self.rescanned = set()

    # ---------- shared ----------
    def room(self, keep=None):
        """Steps left above `keep` (by default the finish's own steps plus four)."""
        d = self.d
        return d.left() - (d.keep_for_finish() + 4 if keep is None else keep)

    def _fitted(self, tool, args):
        faults = []
        shaped = fit_args(args, self.d.schemas.get(tool), "args", faults)
        return None if faults else shaped

    def _facts_view(self):
        d = self.d
        tool = d.facts_tool()
        target = d.single_target(tool)
        return (tool, target) if tool and target else None

    def _lowest(self):
        return self.d.brief.get("pick") == "cheapest"

    def observe(self, tool, result):
        """Take in a self-made call's result; a note inside it is read and queued. Returns what the note changed."""
        d = self.d
        d.take(result)
        lines = d.loose_lines(result)
        if not lines:
            return ""
        meaning = d.read_note(lines)
        if meaning:
            d.queued.append(f"(from {tool}) {meaning}")
        return meaning

    # ---------- search pages ----------
    def pages(self, tool, args, obs):
        """After a search under a cheapest/most/least pick: a short remainder is read here, a page per step;
        a long one is pointed out so the agent narrows the query instead."""
        d = self.d
        self.last_pages = []
        self.note_search(tool, args, obs)
        self.note_found(tool, args, obs)
        layout = self._rest_of(tool, args, obs)
        if layout is None:
            return ""
        total, per_page, param, token = layout
        need = -(-(total - per_page) // per_page)
        roomy = need <= 2 and self.pages_read + need <= PAGE_READ_CAP and d.left() - d.keep_for_finish() - need >= 12
        if not roomy:
            return (f"\n(note) {total} listings match this search but only {per_page} came back, and not by price. "
                    f"The pick has to compare every match, so either narrow the query to the needed value "
                    f"or fetch the next page with {param}={dump(token)}.")
        parts = self._fetch_rest(tool, args, param, token, need)
        d.trace.append({"kind": "pages", "tool": tool, "read": len(parts)})
        return "\n(the rest of this search, fetched automatically at one step per page)\n" + "\n".join(parts) if parts else ""

    def _rest_of(self, tool, args, obs):
        """(total, rows shown, page parameter, its next value) when an ordered pick's search has more pages to read."""
        d = self.d
        pick = d.brief.get("pick")
        layout = d.paging(tool, obs) if pick in ORDERED_PICKS else None
        if not layout:
            return None
        total, per_page, param, token = layout
        if token is None or total <= per_page or _later_page(args.get(param)):
            return None
        return None if pick == "cheapest" and d.price_sorted(obs) else layout

    def _fetch_rest(self, tool, args, param, token, need):
        d = self.d
        parts = []
        while token is not None and len(parts) < need:
            call = self._fitted(tool, {**args, param: token})
            reply = None if call is None else d.call(tool, call)
            if reply is None or d.finished:
                break
            self.pages_read += 1
            result = reply.get("observation")
            self.observe(tool, result)
            self.last_pages.append(result)
            text = d.show(result)
            d.replay_cache[tool + dump(call)] = (d.epoch, d.step, text)
            parts.append(f"{param}={dump(token)}: {text}")
            following = d.paging(tool, result)
            token = following[3] if following else None
        return parts

    def ranked_here(self, results):
        """Lowest-price pick: the buyable rows of a result (and its fetched pages) by price, since stores do not sort them."""
        d = self.d
        if not self._lowest() or not d.price_key:
            return ""
        ids = {i for result in results for i, _ in d.records_in(result)}
        priced = [(d.price_of(i), i) for i in ids if d.buyable(i) and d.price_of(i) is not None]
        if len(priced) < 2:
            return ""
        top = sorted(priced)[:6]
        return "\n(note) buyable rows above, lowest price first: " + ", ".join(f"{i} ({p:g})" for p, i in top)

    def look_wider(self):
        """Once per episode, ahead of a lowest-price checkout with steps to spare: title search misses fitting
        listings named in other words, so the agent is asked to search again under other names."""
        d = self.d
        if self.wider_done or not self._lowest() or d.left() < 14 + 2 * len(d.items):
            return ""
        self.wider_done = True
        priced = [f"{i} at {d.price_of(i):g}" for i in d.basket_ids() if d.price_of(i) is not None]
        line = f" (the cart has {'; '.join(priced)})" if priced else ""
        return (f"Not run (this step was not spent). There are {d.left()} steps left and the lowest price is wanted. "
                "Search matches words in listing titles, while a needed value usually sits only in the attributes, "
                "so searching by that value turns up few of the listings that fit; results are also not sorted by price. "
                "Before checking out, search for each item a few more times by other names: its plain short name, "
                "near synonyms, and words taken from titles of listings that fit. Keep only buyable rows priced under "
                f"what the cart pays{line}, read the attributes of the cheapest of them, swap in one that fits, then check out.")

    # ---------- reading attributes ----------
    def open_facts(self, ids, keep):
        """Read attributes, one step each, of listings no view has shown in full; stop with `keep` steps left."""
        d = self.d
        view = self._facts_view()
        if not view:
            return []
        tool, target = view
        todo = [i for i in ids if i in d.seen and i not in d.opened_facts and not d.sold_out(i)]
        read = []
        for ident in todo:
            if d.left() <= keep or d.finished or d.shows_facts.get(tool) is False:
                break
            call = self._fitted(tool, {target: ident})
            if call is None:
                break
            reply = d.call(tool, call)
            if reply is None:
                break
            result = reply.get("observation")
            self.observe(tool, result)
            d.note_view(tool, result)
            if isinstance(result, dict) and d.id_of(result):
                d.shows_facts[tool] = d.shows_facts.get(tool, False) or bool(fact_block(result))
            read.append(f"{ident}: " + d.show(result)[:1500])
        if read:
            d.trace.append({"kind": "fill_facts", "n": len(read)})
        return read

    def lineup(self):
        """Unread buyable listings to open, alternating between the item groups, at most two per group and
        never past eight opened in a group; lowest price first under a lowest-price pick."""
        d = self.d
        lanes = [self._lane_picks(ids) for ids in d.groups()]
        return [lane[turn] for turn in (0, 1) for lane in lanes if len(lane) > turn]

    def _lane_picks(self, ids):
        d = self.d
        room = max(0, min(2, 8 - len([i for i in ids if i in d.opened_facts])))
        pool = [i for i in ids if i not in d.opened_facts and d.buyable(i)]
        if self._lowest():
            pool = sorted(pool, key=lambda i: _no_price(d.price_of(i)))
        return pool[:room]

    def _open_lineup(self):
        if self._facts_view():
            self.open_facts(self.lineup()[:LINEUP_LOOKS], LINEUP_SPARE)

    def fill_lanes(self):
        """Every item group gets a few listings opened before the questions are chosen, since a property can only
        be chosen once a listing has shown it."""
        d = self.d
        for ids in d.groups():
            short = LANE_FLOOR - len([i for i in ids if i in d.opened_facts])
            pool = [i for i in ids if i not in d.opened_facts and d.buyable(i)]
            if short > 0 and pool:
                self.open_facts(pool[:short], LANE_SPARE)

    def widen(self):
        """Open more listings in one side-by-side step so the planner's key counts rest on more than a couple;
        when that view shows no attributes, open them one by one instead."""
        d = self.d
        tool = d.jobs.get("compare")
        many = d.side_arg() if tool else None
        if not many or d.compare_bare():
            self._open_lineup()
        elif d.left() > 12:
            self._widen_side(tool, many)

    def _widen_side(self, tool, many):
        d = self.d
        limit = ((d.schemas.get(tool) or {}).get("properties") or {})[many].get("maxItems")
        limit = limit if isinstance(limit, int) else 4
        groups = d.groups()
        each = max(1, limit // max(1, len(groups)))
        chosen = [i for ids in groups for i in [j for j in ids if j not in d.opened_facts][:each]][:limit]
        call = self._fitted(tool, {many: chosen}) if len(chosen) >= 2 else None
        before = len(d.opened_facts)
        reply = d.call(tool, call) if call is not None else None
        if reply is None:
            return
        result = reply.get("observation")
        self.observe(tool, result)
        d.note_view(tool, result)
        rows = d.records_in(result)
        if rows:
            d.shows_facts[tool] = d.shows_facts.get(tool, False) or any(fact_block(r) for _, r in rows)
        d.trace.append({"kind": "widen", "ids": chosen, "opened": len(d.opened_facts) - before})
        if rows and d.shows_facts.get(tool) is False:
            self._open_lineup()
        elif len(d.opened_facts) == before and len(chosen) > 2 and d.left() > 12:
            self._retry_pair(tool, many, chosen[:2])

    def _retry_pair(self, tool, many, pair):
        d = self.d
        retry = self._fitted(tool, {many: pair})
        again = d.call(tool, retry) if retry is not None else None
        if again is not None:
            self.observe(tool, again.get("observation"))

    def cheaper(self):
        """Lowest-price pick, before checkout: per cart line, up to two unread buyable rows of the same kind priced
        below it get their attributes read, so the final check can compare them. Once per cart state."""
        d = self.d
        mark = d.basket_mark() + f"#{d.epoch}"
        kind_key = d.category_key() if self._lowest() else None
        if not kind_key or mark in self.cheaper_done:
            return []
        self.cheaper_done.add(mark)
        self.rescan()
        basket = d.basket_ids()
        wanted = []
        for line in basket:
            for ident in self._under(line, basket, kind_key)[:2]:
                if ident not in wanted:
                    wanted.append(ident)
        return self.open_facts(wanted[:CHEAPER_LOOKS], d.keep_for_finish() + 4)

    def _under(self, line, basket, kind_key):
        """Unread buyable rows of a basket line's kind priced below it, cheapest first."""
        d = self.d
        def kind_of(ident):
            return str((d.seen.get(ident) or {}).get(kind_key) or "").strip().lower()
        ceiling, kind = d.price_of(line), kind_of(line)
        if ceiling is None or not kind:
            return []
        rows = [i for i in d.order_seen if i not in basket and i not in d.opened_facts and d.buyable(i)]
        return sorted((i for i in rows if kind_of(i) == kind and _no_price(d.price_of(i)) < ceiling),
                      key=lambda i: d.price_of(i) or 0.0)

    def note_found(self, tool, args, obs):
        """Remember, per listing, the search that first showed it."""
        if tool == self.d.jobs.get("search"):
            for ident, _ in self.d.records_in(obs):
                self.found_by.setdefault(ident, dict(args))

    def rescan(self):
        """Lowest-price pick, before checkout: the search that found a basket line is run again under a price
        ceiling just below that line's price, so cheaper rows of the same search come into view (once per line and price)."""
        d = self.d
        tool = d.jobs.get("search")
        shape = d.shape.known.get(tool or "") or {}
        cap = shape.get("price_cap")
        if not (RESCAN_CHEAPER and cap and self._lowest()):
            return
        for ident in d.basket_ids()[:2]:
            price, args = d.price_of(ident), self.found_by.get(ident)
            mark = f"{ident}@{price}"
            if price is None or args is None or mark in self.rescanned or d.left() < d.keep_for_finish() + 8:
                continue
            self.rescanned.add(mark)
            ceiling = price - 1 if float(price).is_integer() else round(price - 0.01, 2)
            plain_args = {k: v for k, v in args.items() if k != shape.get("next_arg")}
            call = self._fitted(tool, {**plain_args, cap: ceiling}) if ceiling > 0 else None
            reply = d.call(tool, call) if call is not None else None
            if reply is None or d.finished:
                continue
            result = reply.get("observation")
            self.observe(tool, result)
            d.trace.append({"kind": "rescan", "line": ident, "cap": ceiling, "rows": len(d.records_in(result))})

    # ---------- talking ----------
    def ready(self):
        """Whether the interviewer should run now. Early asking: a short round on hinted properties in the first
        steps, the main round once a few listings were opened (or by step six); otherwise one round once enough
        listings were opened or a quarter of the steps went by."""
        d = self.d
        if d.asked or not d.jobs.get("ask") or d.talk_free() <= 0 or d.left() <= 8:
            return False
        opened = len(d.opened_facts)
        if not EARLY_ASK:
            lanes = d.groups()
            filled = len(lanes) >= len(d.items) and all(len([i for i in ids if i in d.opened_facts]) >= 1 for ids in lanes)
            return (opened >= 3 * len(d.items) + 1 and filled) or d.step >= max(12, d.max_steps // 3)
        if d.interviewer.log.rounds == 0 and d.step <= FIRST_WAVE_LAST_STEP:
            return True
        return opened >= min(3, len(d.items) + 1) or d.step >= MAIN_WAVE_STEP

    def talk_args(self, text):
        """Arguments for the messaging tool: the text goes in its free-text field, other required fields take
        their first allowed value; None when the schema cannot be met."""
        d = self.d
        schema = d.schemas.get(d.jobs.get("ask") or "") or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        free = [k for k, v in props.items() if isinstance(v, dict) and "string" in kinds(v) and "enum" not in v]
        slot = next((k for k in required if k in free), None) or (free[0] if free else None)
        if not slot:
            return None
        args = {slot: text}
        for key in required:
            if key in args:
                continue
            choices = (props.get(key) or {}).get("enum")
            if not (isinstance(choices, list) and choices):
                return None
            args[key] = choices[0]
        faults = []
        shaped = fit_args(args, schema, "args", faults)
        return None if faults else shaped

    # ---------- empty code-word searches ----------
    def _lane(self, tool, args):
        """Which part of the store a search looks in: its arguments restricted to a fixed list of values."""
        props = (self.d.schemas.get(tool) or {}).get("properties") or {}
        fixed = sorted((k, str(v)) for k, v in args.items() if isinstance(props.get(k), dict) and props[k].get("enum"))
        return tool + "|" + "|".join(f"{k}={v}" for k, v in fixed)

    def note_search(self, tool, args, obs):
        """Count empty searches that carried a code-like word, per lane; a code-word search that finds rows
        clears the count (searches without codes leave it as it is)."""
        d = self.d
        if not SEARCH_STOP or tool != d.jobs.get("search") or not isinstance(args, dict) or not _code_words(args):
            return
        lane = self._lane(tool, args)
        if d.records_in(obs):
            self.empty_runs[lane] = 0
        else:
            self.empty_runs[lane] = self.empty_runs.get(lane, 0) + 1

    def budget_line(self):
        """Steps per requested item and groups seen with nothing opened yet (shown when several items are asked)."""
        d = self.d
        n = len(d.items)
        if n < 3:
            return ""
        each = max(0, d.max_steps - d.keep_for_finish() - 4) // n
        groups = d.groups()
        bare = sum(1 for ids in groups if not any(i in d.opened_facts for i in ids))
        line = f"About {each} steps per requested item."
        if bare:
            line += f" {bare} of {len(groups)} listing groups have no listing opened yet."
        return line


# ---- shopper questions, request steps before an ending, notes inside results ----

FIRST_WAVE_LAST_STEP = 3  # an early round of questions only happens this early in the episode
FIRST_WAVE_MAX = 2        # questions in that early round
STOP_AFTER_QUESTIONS = False
PICTURE_FIRST = False
PICKS_PER_MESSAGE = 8     # properties named in one message while the shopper takes that well
PLAN_MAX = 25             # properties asked of the plan
ONE_BY_ONE = re.compile(r"\b(one (by|at a|thing|question)|in turn|separately|too many|in one go|first (one|thing|question)|just the (first|one)|the rest(?! of))\b", re.I)
RESEND_UNCLEAR = False
RESENDS_PER_EPISODE = 2   # questions sent again word for word after a reply that did not take them in
SAMPLE_MAX_CHARS = 30


def _above_zero(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _item_index(raw, size, fallback=1):
    """1-based item number from a model reply, kept inside the item list."""
    got = whole(raw)
    return min(max(1, got or fallback or 1), size)


def _name_all(labels):
    """Property names as one phrase: "a, b or c"."""
    return " or ".join(x for x in (", ".join(labels[:-1]), labels[-1]) if x)


def _one_property(label):
    """A label naming two properties ("x and y", "x / y", "x (y)") keeps only the first."""
    head = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(label or ""))[0]
    return head.strip().rstrip("?.")


def live_needs(items):
    """Every requirement still in force, as (item number, requirement)."""
    return [(n, need) for n, item in enumerate(items, 1) for need in item.musts + item.told if need not in item.dropped]


def spoken(observation):
    """The words a reply carries: a plain string as is, else every multi-word string inside it."""
    if isinstance(observation, str):
        return observation
    bits = []
    stack = [(observation, 0)]
    while stack:
        node, level = stack.pop()
        if level > 4:
            continue
        if isinstance(node, str):
            if len(node.split()) >= 2:
                bits.append(node)
        elif isinstance(node, dict):
            stack.extend((v, level + 1) for v in reversed(list(node.values())))
        elif isinstance(node, list):
            stack.extend((v, level + 1) for v in reversed(node))
    return " ".join(bits)[:800] if bits else dump(observation)[:400]


# ---------- sending a message ----------
def send_questions(host, questions, floor=6):
    """Send (item number, label, text) messages one by one; returns (item number, text, reply words) for each one sent.
    Stops when steps fall to `floor`, the message allowance is used up, or the store reports an error."""
    tool = host.jobs["ask"]
    out = []
    for number, _label, text in questions:
        if host.finished or host.left() <= floor or host.talk_used >= host.talk_cap:
            break
        args = host.legwork.talk_args(text)
        if not args:
            break
        reply = host.world.call(tool, args)
        if reply is None:
            break
        host.talk_used += 1
        answer = reply.get("observation")
        words = spoken(answer)
        out.append((number, text, words))
        host.record.append(f"step {host.step}: asked the shopper {text!r}, reply {words[:300]!r} (cart at the time: {host.basket_mark()})")
        host.record_steps.append(host.step)
        host.trace.append({"kind": "ask", "q": text, "a": words[:300]})
        if isinstance(answer, dict) and answer.get("error"):
            break
    return out


def _item_of(host, entry):
    return host.items[_item_index(entry.get("item"), len(host.items)) - 1]


def _take_need(host, entry, told):
    need = str(entry.get("need") or "").strip()
    if not need:
        return
    number = _item_index(entry.get("item"), len(host.items))
    item = host.items[number - 1]
    if need not in item.told:
        item.told.append(need)
        told.append(f"item {number}: {need}" if len(host.items) > 1 else need)
    if entry.get("points") is True and (number, need) not in host.pointers:
        host.pointers.append((number, need))


def _take_drop(host, entry, told):
    if not str(entry.get("property") or "").strip():
        return
    item = _item_of(host, entry)
    target = plain(str(entry["property"]).split(":")[0])
    matched = [m for m in item.musts + item.told if m not in item.dropped and plain(m.split(":")[0]) == target]
    item.dropped.extend(matched)
    told.extend("dropped " + m for m in matched)


def _take_free(host, entry, told):
    if entry.get("property"):
        _item_of(host, entry).free.append(str(entry["property"]))


def take_answers(host, heard, told):
    """Write what the replies state into the items; `told` gathers the new requirements for the summary."""
    for key, apply in (("needs", _take_need), ("drop", _take_drop), ("free", _take_free)):
        for entry in items(heard.get(key)):
            if isinstance(entry, dict):
                apply(host, entry, told)
    if _above_zero(heard.get("budget")):
        host.budget = heard["budget"]


# ---------- questions about the products ----------
@dataclass
class Pick:
    item: int
    label: str
    hinted: bool = False


@dataclass
class AskLog:
    rounds: int = 0                       # rounds of questions already run
    asked: list[Pick] = field(default_factory=list)
    resent: set[str] = field(default_factory=set)
    unclear_left: list[tuple[int, str]] = field(default_factory=list)


class Interviewer:
    """Questions to the shopper. The default settings give one round in which each message names several properties
    of one item until a reply wants one question at a time.

    Optional behaviour, off by default:
      two_waves      - a short early round on properties the request itself hints at, then the main round
      resend_unclear - a question whose reply says it was not understood goes again word for word (reads HEAR "unclear")
      skip_known     - a property already asked, or already holding a requirement, is not asked again for any item
      steer_on_free  - two "don't mind" replies in a row on one item push that item's unhinted picks to the back
    """

    def __init__(self, host, two_waves=False, resend_unclear=False,
                 skip_known=False, steer_on_free=False):
        self.host = host
        self.two_waves = two_waves
        self.resend_unclear = resend_unclear
        self.skip_known = skip_known
        self.steer_on_free = steer_on_free
        self.log = AskLog()
        self.started = False

    # -- state the driver reads --
    @property
    def done(self):
        """No further round will be run."""
        if not self.started:
            return False
        return not self.two_waves or self.log.rounds >= 2

    def free_messages(self):
        """Messages open to questions: the ones the request's own steps at the order or ending need stay back."""
        h = self.host
        return h.talk_cap - h.talk_used - max(0, h.talk_kept - h.duty_msgs)

    # -- main entry --
    def run(self):
        self.started = True
        early = self.two_waves and self.log.rounds == 0 and self.host.step <= FIRST_WAVE_LAST_STEP
        self.log.rounds = self.log.rounds + 1 if self.two_waves else 2
        if early:
            return self._early_round()
        if self.two_waves and self.log.rounds == 1:
            # too late for an early round: this is the only one
            self.log.rounds = 2
        return self._main_round()

    # -- rounds --
    def _early_round(self):
        budget = self.free_messages()
        spare = 1 if self.host.talk_cap >= 4 else 0
        room = min(FIRST_WAVE_MAX, budget - spare)
        if room <= 0:
            return ""
        picks = [p for p in self._choose(budget, widen=False) if p.hinted][:room]
        if not picks:
            return ""
        sent, told = self._ask_and_hear(picks, readers=(DEEP_MODEL, FAST_MODEL))
        if self.resend_unclear and self.log.unclear_left and self.free_messages() > 1:
            sent += self._resend(told)
        return self._summary(sent, told, more_coming=True)

    def _main_round(self):
        """Each message names several properties of one item, likeliest first; with a single message open it names one.
        An item that gave a firm answer waits until the others have had their turn. Once a reply wants one question at
        a time, that message's properties go back to the front of their lane and every later message names one."""
        count = self.free_messages()
        if count <= 0:
            return ""
        picks = self._choose(min(PLAN_MAX, count * PICKS_PER_MESSAGE + 1), widen=True)
        lanes = {}
        for pick in sorted(picks, key=lambda p: not p.hinted):
            lanes.setdefault(pick.item, []).append(pick)
        sent, told, settled = [], [], set()
        size = PICKS_PER_MESSAGE if count >= 2 else 1
        served = dict.fromkeys(lanes, 0)
        while self.free_messages() > 0 and any(lanes.values()):
            waiting = [n for n, lane in lanes.items() if lane]
            fresh = [n for n in waiting if n not in settled] or waiting
            number = min(fresh, key=served.get)
            served[number] += 1
            room = max(1, min(size, -(-sum(map(len, lanes.values())) // self.free_messages())))
            group, lanes[number] = lanes[number][:room], lanes[number][room:]
            got, heard = self._ask_group(number, group, told)
            if not got:
                break
            sent += got
            if items(heard.get("needs")):
                settled.add(number)
                said = [stems(str(e.get("need") or "").split(":")[0]) for e in items(heard["needs"]) if isinstance(e, dict)]
                left = [x for x in group if not any(names_match(stems(x.label), s) for s in said)]
                if len(left) < len(group):
                    lanes[number] = left + lanes[number]
            elif len(group) > 1 and (items(heard.get("crowded")) or ONE_BY_ONE.search(got[-1][2])):
                size = 1
                lanes[number] = group + lanes[number]
        return self._summary(sent, told, more_coming=False) if sent else ""

    def _ask_group(self, number, group, told):
        """One message about the given properties of one item; returns what was sent and how the reply reads.
        Indifference to a message naming several properties says nothing about any one of them."""
        text = self.question(number, _name_all([p.label for p in group]))
        while len(group) > 1 and not self.host.legwork.talk_args(text):
            group = group[:-1]
            text = self.question(number, _name_all([p.label for p in group]))
        out = send_questions(self.host, [(number, text, text)])
        if not out:
            return [], {}
        self.log.asked.extend(p for p in group if p not in self.log.asked)
        heard = self.read_replies(out, (FAST_MODEL, DEEP_MODEL))
        if len(group) > 1:
            heard = {**heard, "free": []}
        take_answers(self.host, heard, told)
        return out, heard

    def _lead_questions(self, lead, rest, told, count):
        """Send the lead questions, read the replies, then use the message kept back."""
        got, heard = self._send_and_read(lead, (DEEP_MODEL, FAST_MODEL))
        if not got:
            return []
        take_answers(self.host, heard, told)
        if self.steer_on_free:
            rest = self._steer(got, heard, rest)
        if count >= 2 and self.free_messages() > 0:
            return got + self._last_message(got, heard, lead, rest, told)
        return got

    # -- choosing properties --
    def _choose(self, wanted, widen):
        h = self.host
        if widen:
            h.legwork.widen()
            h.legwork.fill_lanes()
        table = h.catalog.key_table(h.items)
        prompt = self._plan_prompt(wanted, table)
        if PICTURE_FIRST:
            prompt += self._picture(prompt)
        plan = h.model.json(prompt, [DEEP_MODEL, FAST_MODEL], 3000) or {}
        first = {stems(x) for x in items(h.brief.get("ask_first")) if str(x).strip()}
        picks = []
        for raw in items(plan.get("ask")):
            if not isinstance(raw, dict):
                continue
            number = _item_index(raw.get("item") or 1, len(h.items))
            label = _one_property(raw.get("label"))
            if not label or len(label.split()) > 5:
                continue
            if any(p.item == number and plain(p.label) == plain(label) for p in picks + self.log.asked):
                continue
            if self.skip_known and self._known(number, label):
                continue
            hinted = bool(str(raw.get("hint") or "").strip()) or stems(label) in first
            picks.append(Pick(number, label, hinted))
        h.trace.append({"kind": "plan", "ask": clip(plan.get("ask"), 120)})
        return self._top_up(picks, wanted, table)

    def _picture(self, plan_text):
        """Notes on who is buying, written before the questions are chosen; nothing when no notes come back."""
        h = self.host
        facts = plan_text.split("REQUEST:\n", 1)[-1]
        got = h.model.json(SHOPPER_PICTURE + "\n\nREQUEST:\n" + facts, [FAST_MODEL, DEEP_MODEL], 400) or {}
        notes = str(got.get("notes") or "").strip()
        h.trace.append({"kind": "picture", "notes": clip(notes, 200)})
        return "\n\nNOTES ON THE SHOPPER (use them to order the properties):\n" + notes if notes else ""

    def _top_up(self, picks, wanted, table):
        """A plan shorter than the messages open is filled from the opened listings' keys, most carried first,
        taking turns between the items; a message left unsent can surface nothing."""
        h = self.host
        queues = [[row[1] for row in table if h.catalog._asked_for(row[0], h.catalog._wanted([it]))]
                  for it in h.items]
        turn = len(picks)
        while len(picks) < wanted and any(queues):
            number = turn % len(queues) + 1
            turn += 1
            while queues[number - 1]:
                label = queues[number - 1].pop(0)
                if self._fits_pick(number, label, picks):
                    picks.append(Pick(number, label))
                    break
        return picks

    def _fits_pick(self, number, label, picks):
        if len(label.split()) > 5 or any(p.item == number and stems(p.label) == stems(label) for p in picks + self.log.asked):
            return False
        item = self.host.items[number - 1]
        return not any(stems(m.split(":")[0]) == stems(label) for m in item.musts + item.told if m not in item.dropped)

    def _plan_prompt(self, wanted, table):
        h = self.host
        settled = [f"item {n}: {it.name} - already fixed by the request: " + ("; ".join(it.musts) or "nothing")
                   for n, it in enumerate(h.items, 1)]
        rows = []
        current = None
        if self.skip_known:
            known = self._known_roots()
            table = [row for row in table if stems(row[1]) not in known]
        for number, (label, name, carried, values) in enumerate(table, 1):
            if label != current:
                rows.append(f"category {label!r}:")
                current = label
            rows.append(f"  {number}. {name} (on {carried}): " + " | ".join(values))
        parts = [plan_prompt(wanted), "REQUEST:\n" + h.request, "ITEMS:\n" + "\n".join(settled)]
        first = [str(x) for x in items(h.brief.get("ask_first")) if str(x).strip()]
        if first:
            parts.append(LABEL_MUST_ASK + " (topics the request wants raised with the shopper):\n" + "\n".join(first))
        parts.append("ATTRIBUTE KEYS ON OPENED LISTINGS:\n" + ("\n".join(rows) or "(no listing opened in full so far)"))
        if self.log.asked:
            parts.append("QUESTIONS ALREADY SENT (pick none of these again, under any name):\n"
                         + "\n".join(f"item {p.item}: {p.label}" for p in self.log.asked))
        if self.skip_known and self.log.asked:
            calm = [f"item {n}: {prop}" for n, it in enumerate(h.items, 1) for prop in it.free]
            if calm:
                parts.append("THE SHOPPER DID NOT MIND ABOUT (prefer a different kind of property):\n" + "\n".join(calm))
        return "\n\n".join(parts)

    def _known_roots(self):
        """Word sets of properties already asked about, or already holding a requirement, on any item."""
        roots = {stems(p.label) for p in self.log.asked}
        for it in self.host.items:
            roots.update(stems(m.split(":")[0]) for m in it.musts + it.told if m not in it.dropped)
        return roots

    def _known(self, number, label):
        """Asked before for any item (a shopper repeats an old answer to a same-named question), or already a requirement."""
        roots = stems(label)
        if any(stems(p.label) == roots for p in self.log.asked):
            return True
        item = self.host.items[number - 1]
        return any(stems(m.split(":")[0]) == roots for m in item.musts + item.told if m not in item.dropped)

    def _steer(self, got, heard, rest):
        calm = {(_item_index(e.get("item"), len(self.host.items)), stems(e.get("property")))
                for e in items(heard.get("free")) if isinstance(e, dict) and e.get("property")}
        streak, last_item = 0, None
        tired = set()
        for number, text, _ in got:
            pick = next((p for p in self.log.asked if p.item == number and self.question(p.item, p.label) == text), None)
            is_calm = pick is not None and (number, stems(pick.label)) in calm
            streak = streak + 1 if is_calm and number == last_item else (1 if is_calm else 0)
            last_item = number
            if streak >= 2:
                tired.add(number)
        if not tired:
            return rest
        keep = [p for p in rest if p.item not in tired or p.hinted]
        later = [p for p in rest if p.item in tired and not p.hinted]
        return keep + later

    # -- wording --
    def question(self, number, label, samples=None):
        thing = self.host.items[number - 1].name
        if samples:
            return f"What {label} would you like for the {thing}, for example {samples[0]} or {samples[1]}?"
        return f"What {label} would you like for the {thing}?"

    def samples(self, number, label):
        """A pair of values seen on listings for this property, so a reworded question shows what it is about."""
        h = self.host
        target = plain(label)
        already = " ".join(h.items[number - 1].musts).lower()
        for _cat, name, _carried, values in h.catalog.key_table(h.items):
            if name == target:
                return [v for v in values if v and v not in already and len(v) <= SAMPLE_MAX_CHARS][:2]
        return []

    # -- sending and reading --
    def _send_and_read(self, picks, readers):
        h = self.host
        out = send_questions(h, [(p.item, p.label, self.question(p.item, p.label)) for p in picks])
        self.log.asked.extend(picks[: len(out)])
        if not out:
            return [], {}
        heard = self.read_replies(out, readers)
        if self.resend_unclear:
            for i in (whole(x) for x in items(heard.get("unclear"))):
                if i and 1 <= i <= len(out) and out[i - 1][1] not in self.log.resent:
                    self.log.unclear_left.append((out[i - 1][0], out[i - 1][1]))
        return out, heard

    def _ask_and_hear(self, picks, readers):
        told = []
        got, heard = self._send_and_read(picks, readers)
        if got:
            take_answers(self.host, heard, told)
        return got, told

    def _resend(self, told):
        """Send the oldest misunderstood question again, unchanged; once per question, a few times per episode."""
        if len(self.log.resent) >= RESENDS_PER_EPISODE or not self.log.unclear_left:
            return []
        number, text = self.log.unclear_left.pop(0)
        self.log.resent.add(text)
        out = send_questions(self.host, [(number, "", text)])
        if out:
            take_answers(self.host, self.read_replies(out, (FAST_MODEL, DEEP_MODEL)), told)
        return out

    def _last_message(self, got, heard, lead,
                      rest, told):
        """The message kept back: a misunderstood question first (when enabled), then a question that wanted to be
        asked alone (with two sample values when known), else the next pick."""
        if self.resend_unclear and self.log.unclear_left:
            return self._resend(told)
        nxt = None
        crowded = [got[i - 1] for i in (whole(x) for x in items(heard.get("crowded"))) if i and 1 <= i <= len(got)]
        for number, text, _ in crowded:
            label = next((p.label for p in lead if p.item == number and self.question(p.item, p.label) == text), None)
            shown = self.samples(number, label) if label else []
            if label:
                nxt = (number, label, self.question(number, label, shown if len(shown) >= 2 else None))
                break
        if nxt is None and rest:
            p = rest[0]
            nxt = (p.item, p.label, self.question(p.item, p.label))
        if nxt is None:
            return []
        out = send_questions(self.host, [nxt])
        if out:
            self.log.asked.append(Pick(nxt[0], nxt[1]))
            take_answers(self.host, self.read_replies(out, (FAST_MODEL, DEEP_MODEL)), told)
        return out

    def read_replies(self, sent, readers):
        h = self.host
        listing = "\n".join(f"item {n}: {it.name}" for n, it in enumerate(h.items, 1))
        pairs = "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        prompt = HEAR + "\n\nITEMS:\n" + listing + "\n\nQUESTIONS AND ANSWERS:\n" + pairs
        return h.model.json(prompt, list(readers), 1500) or {}

    # -- what the driver gets back --
    def _summary(self, sent, told, more_coming):
        h = self.host
        if told:
            h.epoch += 1
        if not sent:
            return ""
        text = ["Questions sent to the shopper so far in this round (one step each):"]
        text += [f"Q: {q}\nA: {a}" for _, q, a in sent]
        text.append("Firm requirements these replies add: " + ("; ".join(told) if told else "none")
                    + ". Keep every choice to them, searching again with the new values where needed.")
        linked = [need for _, need in h.pointers if any(need in t for t in told)]
        if linked:
            text.append("Some of them take their value from another listing: " + " | ".join(linked)
                        + ". Locate that listing and open its page in full, then open each listing that page points to;"
                          " the value is on the last page reached, and no order should go in before all of them are open.")
        if more_coming:
            text.append("More questions will follow once a few listings have been opened.")
        out = "\n".join(text)
        if h.queued:
            out += "\n[store notes seen meanwhile] " + " | ".join(h.queued)
            h.queued = []
        return out


# ---------- the request's own steps ----------
@dataclass
class DutyRun:
    """One pass of the step check before an ending."""
    stage: str
    due: list[dict[str, Any]]
    menu: list[str]
    closing: set[str]
    done: list[str] = field(default_factory=list)
    pushed: bool = False
    over: bool = False


class Gate:
    """Errands the request ties to the checkout ("order") or to a no-purchase ending ("end"), carried out one store
    call at a time as the model directs, until every errand is covered for the basket in its current state."""

    def __init__(self, host, reread_when_empty=False):
        self.host = host
        self.reread_when_empty = reread_when_empty
        self.reread_done = False

    def pass_(self, stage):
        h = self.host
        self._maybe_reread(stage)
        due = [d for d in h.duties if d["when"] == stage]
        if h.finished or not due or not h.duty_due(stage):
            return True, ""
        closing = {h.jobs.get("buy"), h.jobs.get("refuse")} - {None}
        run = DutyRun(stage, due, self._tool_menu(closing), closing)
        for _ in range(DUTY_ROUNDS):
            changed = self._one_round(run)
            if changed is not None:
                return False, changed
            if run.over:
                break
        if run.done:
            return True, "Before going on, these steps the request asks for were carried out:\n" + "\n".join(run.done)
        return True, ""

    def _one_round(self, run):
        """Ask which step call is next and make it; returns the reply text when a shopper's answer changes the task."""
        h = self.host
        tiers = [DEEP_MODEL, FAST_MODEL] if run.pushed else [FAST_MODEL, DEEP_MODEL]
        reply = h.model.json(self._prompt(run.stage, run.due, run.menu), tiers, 2000)
        if reply is None:
            run.over = True
            return None
        told = []
        wait_on = ""
        if run.done:
            # only replies to the gate's own calls are read here; earlier ones belong to the interviewer
            take_answers(h, {"needs": reply.get("needs")}, told)
            wait_on = str(reply.get(DUTY_WAIT_KEY) or "").strip()
        call = reply.get("call")
        call = call if isinstance(call, dict) else None
        h.trace.append({"kind": "gate", "stage": run.stage, "call": clip(call, 200), "steps": clip(reply.get("steps"), 120),
                        "wait_on": wait_on[:200], "told": told})
        if told or wait_on:
            h.epoch += 1
            return self._changed(run.done, told, wait_on)
        run.over = not (self._proceed(run, call) if call else self._no_call(run, reply))
        return None

    def _no_call(self, run, reply):
        """No call named: nudge once when steps lack proof and a message can still go, else take the steps as done."""
        h = self.host
        if not (run.done or run.pushed) and self._can_push():
            open_steps = self._unproven(run.stage, run.due, reply)
            if open_steps:
                run.pushed = True
                self._push(open_steps)
                return True
        h.duty_done.add((run.stage, h.basket_mark() + f"#{h.epoch}"))
        return False

    def _proceed(self, run, call):
        h = self.host
        name = str(call.get("tool") or "")
        blocked = name not in h.schemas or name in run.closing or h.left() < 2
        if blocked or (name == h.jobs.get("ask") and h.talk_used >= h.talk_cap):
            return False
        line = self._do(name, call)
        if line is not None:
            run.done.append(line)
        return line is not None

    def _maybe_reread(self, stage):
        """Once, at the first checkout, when no step was read from the request but the brief says it may name one."""
        h = self.host
        if not self.reread_when_empty or self.reread_done or stage != "order" or h.duties:
            return
        if h.brief.get("process_hint") is not True or not hasattr(h, "reread_duties"):
            return
        self.reread_done = True
        h.reread_duties()

    def _tool_menu(self, closing):
        out = []
        for t in self.host.tools:
            fn = t["function"]
            if fn["name"] in closing:
                continue
            out.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {dump(fn.get('parameters') or {})}")
        return out

    def _prompt(self, stage, due, menu):
        h = self.host
        what = "place the order" if stage == "order" else "end without a purchase"
        record = "\n".join(f"{i}. {line}" for i, line in enumerate(h.record, 1)) or "(nothing yet)"
        parts = [duty_prompt(what),
                 "REQUEST:\n" + h.request,
                 "STEPS OWED AT THIS POINT:\n" + "\n".join(f"- {d['step']}" for d in due),
                 "REQUIREMENTS LEDGER:\n" + h.board(),
                 "CART AT THIS MOMENT: " + h.basket_text(),
                 LABEL_LOG + " (shopper messages and step calls, numbered):\n" + record]
        if h.notes:
            parts.append("LATEST STORE NOTES:\n" + "\n".join(h.notes[-4:]))
        last = "order" if stage == "order" else "ending"
        parts.append(f"MESSAGES LEFT: {max(0, h.talk_cap - h.talk_used)}   STEPS LEFT: {h.left()} (one of them goes to the {last} itself)")
        parts.append(LABEL_TOOLS + ":\n" + "\n".join(menu))
        return "\n\n".join(parts)

    def _unproven(self, stage, due, reply):
        """Steps the reply leaves without a record line that proves them. At the order, a step that needs a message
        is about the final cart, so only lines made after the cart last changed count."""
        h = self.host
        strict = stage == "order" and any(d["messages"] > 0 for d in due)
        steps_at = h.record_steps
        valid = {n for n in range(1, len(h.record) + 1)
                 if not strict or (n <= len(steps_at) and steps_at[n - 1] > h.basket_moved)}

        def proven(entry):
            cited = (whole(n) or 0 for n in items(entry.get("lines")))
            return entry.get("done") is True and any(n in valid for n in cited)
        return [e for e in items(reply.get("steps")) if isinstance(e, dict) and not proven(e)]

    def _can_push(self):
        h = self.host
        return bool(h.jobs.get("ask")) and h.talk_used < h.talk_cap and h.left() >= 2

    def _push(self, open_steps):
        h = self.host
        h.trace.append({"kind": "gate_unproven", "steps": clip(open_steps, 120)})
        names = "; ".join(str(e.get("step") or "") for e in open_steps)[:200]
        h.record.append(f"(check: nothing recorded after the cart last changed at step {h.basket_moved} shows {names} as done;"
                        f" {h.talk_cap - h.talk_used} message(s) and {h.left()} steps remain, so name the call that does it)")
        h.record_steps.append(-1)

    def _do(self, name, call):
        """Run one step call; returns the line for the agent, or None when it could not go out."""
        h = self.host
        faults = []
        args = fit_args(call.get("args") if isinstance(call.get("args"), dict) else {}, h.schemas[name], "args", faults)
        if faults:
            return None
        if name == h.jobs.get("ask"):
            text = next((v for v in args.values() if isinstance(v, str)), "")
            out = send_questions(h, [(1, "", text)], 1)
            if not out:
                return None
            h.duty_msgs += 1
            return f"Q: {text}\nA: {out[0][2]}"
        reply = h.world.call(name, args)
        if reply is None:
            return None
        seen = reply.get("observation")
        h.catalog.take(seen)
        h.record.append(f"step {h.step}: {name} {dump(args)} -> {h.show(seen)[:300]} (cart at the time: {h.basket_mark()})")
        h.record_steps.append(h.step)
        return f"{name} {dump(args)} -> {h.show(seen)[:400]}"

    @staticmethod
    def _changed(done, told, wait_on):
        head = ("Before going on, these steps the request asks for were carried out:\n" + "\n".join(done) + "\n") if done else ""
        asks = "; ".join(told + ([wait_on] if wait_on else []))
        return head + "The shopper's reply changes the task: " + asks + ". Adjust the cart to it, then try the checkout again."


# ---------- notes inside store results ----------
class Notes:
    """Free-text lines a store result carries outside its listing records, read for changes to the task."""

    def __init__(self, host):
        self.host = host

    def lines(self, observation):
        """Sentences of five words or more at the top of a result that is not itself a listing record, plus those
        in its string lists and (six words or more) in its non-record sub-objects; lines read before are skipped."""
        h = self.host
        found = []
        if not isinstance(observation, dict) or h.catalog.id_of(observation):
            return found
        for value in observation.values():
            if isinstance(value, str):
                if len(value.split()) >= 5:
                    found.append(value)
            elif isinstance(value, list):
                if value and all(isinstance(v, str) for v in value):
                    found += [v for v in value if len(v.split()) >= 5]
            elif isinstance(value, dict) and not h.catalog.id_of(value, False, 1):
                found += [v for v in value.values() if isinstance(v, str) and len(v.split()) >= 6]
        return [line for line in found if line not in h.heard]

    def read(self, lines):
        h = self.host
        h.heard.update(lines)
        prompt = "\n\n".join([NOTE, "ITEMS WITH THEIR CURRENT REQUIREMENTS:\n" + h.board(), "NOTE TEXT:\n" + "\n".join(lines)])
        got = h.model.json(prompt, [FAST_MODEL, BACKUP_MODEL], 1200) or {}
        gone = [str(g) for g in items(got.get("gone")) if isinstance(g, (str, int))]
        self._mark_sold(gone, got, lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changes = self._changes(got, gone)
        summary = str(got.get("summary") or "").strip()
        if changes or summary:
            h.epoch += 1
            h.recent = []
            h.notes.append(summary or "; ".join(changes))
        tail = " (" + "; ".join(changes) + ")" if changes else ""
        return (summary + tail).strip()

    def _mark_sold(self, gone, got, lines):
        h = self.host
        h.unavailable.update(g for g in gone if g in h.seen)
        h.note_sold = bool(gone) or got.get("sold_out") is True
        if h.note_sold and len(lines) == 1:
            # a note made only of the sell-out is kept: the same words later mean the same thing
            h.sold_lines.update(lines)

    def _changes(self, got, gone):
        h = self.host
        out = []
        budget = got.get("budget")
        if _above_zero(budget) and budget != h.budget:
            h.budget = budget
            out.append(f"budget now {budget}")
        out += self._drops(got) + self._adds(got)
        out += ["unavailable: " + ", ".join(gone)] if gone else []
        out += ["a price changed"] if got.get("price_moved") is True else []
        return out

    def _entries(self, got, key):
        return [e for e in items(got.get(key)) if isinstance(e, dict) and str(e.get("need") or "").strip()]

    def _drops(self, got):
        goods = self.host.items
        out = []
        for entry in self._entries(got, "drop"):
            item = goods[_item_index(entry.get("item"), len(goods)) - 1]
            target = str(entry["need"]).strip()
            active = [m for m in item.musts + item.told if m not in item.dropped]
            matched = [m for m in active if m == target] or \
                [m for m in active if plain(m.split(":")[0]) == plain(target.split(":")[0])][:1]
            item.dropped.extend(matched or [target])
            out.append("dropped " + target)
        return out

    def _adds(self, got):
        goods = self.host.items
        out = []
        for entry in self._entries(got, "add"):
            item = goods[_item_index(entry.get("item"), len(goods)) - 1]
            need = str(entry["need"]).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            out.append("added " + need)
        return out


# ---- Pre-finish review: order check, decline review, pointer walk, sold-out swap, basket completion. ----


class Lens:
    """How one review lists the listings it has seen for a model prompt.

    depth: which record counts as read in full ("page": its page was opened, "facts": its attributes are known,
    "" for an untagged list); full/thin: (clip, cut) for read and row-only records."""

    def __init__(self, depth, full, thin=(0, 0)):
        self.depth = depth
        self.full = full
        self.thin = thin


LENS_RIVALS = Lens("", (160, 900))
LENS_POINTERS = Lens("page", (300, 1400), (80, 300))
LENS_SWAP = Lens("", (200, 1200))
LENS_DECLINE = Lens("facts", (200, 1200), (120, 500))


def _moved_keys(before, now):
    """Plain fields of a record whose value differs from an earlier copy of it."""
    return [k for k, v in now.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]


class Review:
    """Checks run before the agent finishes, all reading the same driver state.

    The driver (``host``) supplies: seen, opened_facts, order_seen, pages_read_ids, basket, basket_view,
    basket_seen, basket_moved, epoch, jobs, shortlist, vouched, pointers, pointed, items, brief, request,
    schemas, cost, step, trace, finished, unavailable, gone, offer_seen, refused_at, id_args, buys; and the
    methods left(), board(), needs_now(), ident(), price_of(), single_target(), facts_tool(), compare_target(),
    compare_room(), compare_bare(), call(), observe(), show(), ask_json(), category_key(), fill_facts(),
    note_cart(), cart_sets(), duty_check()."""

    WALK_ROUNDS = 4
    WALK_STOPS = 2
    WALK_OPENS = 6
    ROOM = 8
    STAYS = 3

    def __init__(self, host, stay=True):
        self.h = host
        self.stay = stay
        self.asked = set()
        self.verdicts = {}
        self.declined_once = False
        self.walks = 0
        self.walk_stops = 0
        self.settled = False
        self.fits_by_item = {}
        self.stays_used = set()
        self.plan = CartPlan(host, self)

    # ---- shared views of the basket and the listings ----
    def fresh_view(self):
        h = self.h
        return h.basket_view is not None and h.basket_seen >= h.basket_moved

    def basket_text(self):
        if self.fresh_view():
            return dump(clip(self.h.basket_view, 300))[:3000]
        return "basket as tracked from the cart calls made (listing: units): " + dump(self.h.basket)

    def basket_ids(self):
        if not self.fresh_view():
            return list(self.h.basket)
        shown = dump(self.h.basket_view)
        return [i for i in self.h.seen if i in shown]

    def record(self, ident):
        found = self.h.opened_facts.get(ident) or self.h.seen.get(ident)
        if not found:
            return "(nothing on record)"
        return dump(clip(found, 400))[:3000]

    def sold(self, ident):
        return ident in self.h.unavailable or ident in self.h.gone

    def key(self, ids):
        return "|".join(sorted(ids)) + "#" + str(self.h.epoch)

    def verdict(self, ids):
        return self.verdicts.get(self.key(ids))

    def _read_record(self, ident, depth):
        h = self.h
        if depth == "page":
            return (h.opened_facts.get(ident) or h.seen.get(ident)) if ident in h.pages_read_ids else None
        if depth == "facts":
            return h.opened_facts.get(ident) if ident in h.opened_facts else None
        return h.opened_facts.get(ident) or h.seen.get(ident)

    def records(self, lens, cap, skip=None, order=None):
        """One line per listing seen, newest first unless an order is given."""
        out = []
        queue = order if order is not None else list(reversed(self.h.order_seen))
        for ident in queue:
            if skip is not None and skip(ident):
                continue
            got = self._read_record(ident, lens.depth)
            if not lens.depth:
                out.append(dump(clip(got, lens.full[0]))[:lens.full[1]])
            elif got:
                out.append(LABEL_FULL + " " + dump(clip(got, lens.full[0]))[:lens.full[1]])
            else:
                out.append(LABEL_ROW + " " + dump(clip(self.h.seen.get(ident), lens.thin[0]))[:lens.thin[1]])
            if len(out) >= cap:
                break
        return out

    def rivals(self, skip, cap=30):
        h = self.h
        queue = list(reversed(h.order_seen))
        if h.brief.get("pick") == "cheapest":
            def rank(ident):
                price = h.price_of(ident)
                return (ident not in h.opened_facts, price if price is not None else float("inf"))
            queue.sort(key=rank)
        lines = self.records(LENS_RIVALS, cap, lambda i: i in skip or self.sold(i), queue)
        return "\n".join(lines)

    def _brief_head(self):
        return "\n\nSHOPPER REQUEST:\n" + self.h.request + "\n\nREQUIREMENTS NOW IN FORCE:\n" + self.h.board()

    # ---- evidence line under a reply that shows attribute blocks ----
    def proof(self, observation):
        h = self.h
        found = self._opened_in(observation)
        wanted = h.needs_now()
        rows = [row for row in (self._proof_row(name, wanted) for name in found[:12]) if row] if wanted else []
        if not rows:
            return ""
        return (KEY_MATCH_TAG + "] (attribute keys sharing a word with a requirement's property; a key about some other part "
                "or aspect settles nothing, and a differently named key can still carry the same property; the mark after a "
                "value compares it with the asked value: = states it, ⊃ has it among more words, ⊂ lacks part of it, ≠ other "
                "words) " + " | ".join(rows))

    def _opened_in(self, observation):
        """Ids of listing records with an attribute block inside a result, in reading order."""
        h = self.h
        found = []
        stack = [(observation, 0)]
        while stack:
            node, depth = stack.pop()
            children = []
            if depth <= 6 and isinstance(node, list):
                children = node
            elif depth <= 6 and isinstance(node, dict):
                name = h.ident(node, False, depth)
                if name and name not in found and fact_block(node):
                    found.append(name)
                else:
                    children = list(node.values())
            stack.extend((child, depth + 1) for child in reversed(children))
        return found

    def _proof_row(self, name, wanted):
        h = self.h
        attrs = fact_block(h.opened_facts.get(name) or {}) or {}
        tag = len(h.items) > 1
        bits = []
        for number, need in wanted[:8]:
            prop, _, asked = need.partition(":")
            prop = prop.strip()
            hits = []
            for k, v in attrs.items():
                if names_match(stems(prop), stems(k)):
                    mark = value_fit(asked, v)
                    hits.append(f"{k}={str(v)[:50]}" + (f" [{mark}]" if mark != "?" else ""))
            if hits:
                bits.append((f"(item {number}) " if tag else "") + prop + " -> " + ", ".join(hits[:3]))
        return name + ": " + "; ".join(bits) if bits else ""

    # ---- read listings whose attributes state the asked values ----
    def item_of(self, ident):
        """The number of the requested product a listing belongs to, read off its category label; None when unclear."""
        h = self.h
        slot = h.category_key()
        label = str((h.seen.get(ident) or {}).get(slot) or "") if slot else ""
        if not label:
            return 1 if len(h.items) == 1 else None
        hits = [n for n, one in enumerate(Catalog._wanted(h.items), 1) if Catalog._asked_for(label, [one])]
        return hits[0] if len(hits) == 1 else None

    def fit_table(self, ident, number):
        """Per requirement of product `number`, the listing's best-fitting attribute named for the property:
        {property: (mark, key, value, asked value)}; a property no key is named for is left out."""
        h = self.h
        attrs = fact_block(h.opened_facts.get(ident) or {}) or {}
        table = {}
        for n, need in h.needs_now() if number else []:
            prop, _, asked = need.partition(":")
            prop, asked = prop.strip(), asked.strip()
            if n != number or not asked:
                continue
            for key, value in attrs.items():
                mark = value_fit(asked, value) if names_match(stems(prop), stems(key)) else "?"
                if mark != "?" and (prop not in table or FIT_MARKS.index(mark) < FIT_MARKS.index(table[prop][0])):
                    table[prop] = (mark, key, str(value), asked)
        return table

    def exact_fits(self, number, exclude, room):
        """Read listings of product `number`, live and outside `exclude`, whose every judged attribute states the asked
        value and whose price at the requested count stays within `room` (None: no limit); lowest price first."""
        h = self.h
        count = h.items[number - 1].qty
        out = []
        for ident in h.opened_facts:
            if ident in exclude or self.sold(ident) or not h.buyable(ident) or self.item_of(ident) != number:
                continue
            marks = [row[0] for row in self.fit_table(ident, number).values()]
            price = h.price_of(ident)
            over = room is not None and price is not None and price * count > room
            if marks and not over and all(mark == "=" for mark in marks):
                out.append((price if price is not None else float("inf"), ident))
        return [ident for _, ident in sorted(out)]

    def room_for(self, skip):
        """What the spend limit leaves beside the basket lines outside `skip`; None without a limit."""
        h = self.h
        if h.budget is None:
            return None
        return h.budget - sum((h.price_of(i) or 0) * n for i, n in h.basket.items() if i not in skip)

    # ---- order: independent check of the basket (or of planned lines) ----
    def order(self, ids=None, cap=COST_EASE):
        h = self.h
        planned = ids is not None
        lines = ids if planned else self.basket_ids()
        mark = self.key(lines)
        if not lines or mark in self.asked or h.cost > cap:
            return ""
        self.asked.add(mark)
        got = h.ask_json(self._order_prompt(lines, planned), [DEEP_MODEL, FAST_MODEL], 1500)
        h.trace.append({"kind": "order_review", "cart": lines, "got": got})
        refused = bool(got) and got.get("ok") is False
        self.verdicts[mark] = bool(got) and not refused
        return self._order_text(got, lines) if refused else ""

    def _order_prompt(self, lines, planned):
        h = self.h
        if planned:
            units = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(h.items, 1))
            basket = ("BASKET (proposed; treat each listing below as already carted at its requested count: "
                      + units + "):\n" + ", ".join(lines))
        else:
            basket = "BASKET:\n" + self.basket_text()
        gone_flag = "NO LONGER SOLD (the store reported it unavailable). "
        shown = "\n".join(i + ": " + (gone_flag if self.sold(i) else "") + self.record(i) for i in lines)
        parts = [CHECK + self._brief_head(), basket, "WHAT THE STORE SHOWED FOR EACH BASKET LISTING:\n" + shown]
        if h.brief.get("pick") in ("cheapest", "most", "least"):
            parts.append("OTHER LISTINGS SEEN (most recent first):\n" + self.rivals(lines, 15))
        return "\n\n".join(parts)

    def _order_text(self, got, lines):
        h = self.h
        h.shortlist = [i for i in h.shortlist if i not in lines]
        faults = [str(p) for p in items(got.get("problems"))][:5]
        better = [str(b) for b in items(got.get("better"))]
        swaps = [b for b in better if b in h.seen and not self.sold(b)][:4]
        text = "An independent review finds the basket not ready: " + "; ".join(faults or ["compare it with the requirements"])
        text += (". Better candidates: " + ", ".join(swaps)) if swaps else ""
        return text + ". Correct the basket (or re-read the records if the review misread them), then check out."

    # ---- trail: requirements whose value sits on another listing ----
    def _chain_need(self, chain, pending):
        named = str(chain.get("need") or "").strip()
        for p in pending:
            if p == named:
                return p
        return pending[0] if len(pending) == 1 else None

    def _walk_round(self, pending, got):
        """Sort each traced chain: a value read off opened pages, links still to open, or a link nobody has found."""
        reach = []
        lost = []
        for chain in [c for c in items(got.get("chains")) if isinstance(c, dict)]:
            need = self._chain_need(chain, pending)
            if need:
                self._sort_chain(chain, need, reach, lost)
        return reach, lost

    def _sort_chain(self, chain, need, reach, lost):
        h = self.h
        links = [link for link in items(chain.get("links")) if isinstance(link, dict)]
        ids = [str(link.get("id") or "").strip() for link in links]
        value = str(chain.get("value") or "").strip()
        if value and links and set(ids) <= h.pages_read_ids and chain.get("done") is True:
            h.pointed[need] = value
            h.epoch += 1
            return
        unread = [i for i in ids if i in h.seen and i not in h.pages_read_ids]
        if unread:
            reach.extend(dict.fromkeys(i for i in unread if i not in reach))
            return
        gap = next((link for link, i in zip(links, ids) if i not in h.seen), {})
        about = str(gap.get("about") or "")
        lost.append(f"{need} -> find {about or 'the listing it refers to'}")

    def _open_refs(self):
        h = self.h
        live = {need for _, need in h.needs_now()}
        return [need for _, need in h.pointers if need in live and need not in h.pointed]

    def trail(self):
        """Before a checkout, open every page along each requirement's pointer chain; a link nobody has found
        stops the checkout (at most twice). Returns what was read and whether to stop."""
        h = self.h
        live = {need for _, need in h.needs_now()}
        pending = [need for _, need in h.pointers if need in live and need not in h.pointed]
        tool = self.offer_tool()
        target = h.single_target(tool)
        read = []
        lost = []
        while pending and self.walks < self.WALK_ROUNDS and h.cost < COST_CAP:
            reach, lost = self._walk_ask(pending)
            pending = [p for p in pending if p not in h.pointed]
            if h.finished or not (reach and tool and target) or h.left() < len(reach) + 4:
                break
            read += self._open_pages(tool, target, reach[:max(0, min(3, self.WALK_OPENS - len(read)))])
        return self._trail_text(read, lost)

    def _walk_ask(self, pending):
        h = self.h
        self.walks += 1
        prompt = "\n\n".join([TRAIL, "REQUIREMENTS WHOSE VALUE IS ON ANOTHER LISTING:\n" + "\n".join(pending),
                               "LISTINGS (most recent first):\n" + "\n".join(self.records(LENS_POINTERS, 60))])
        got = h.ask_json(prompt, [FAST_MODEL, DEEP_MODEL], 1800) or {}
        h.trace.append({"kind": "pointer_walk", "pending": pending, "got": clip(got, 150)})
        return self._walk_round(pending, got)

    def _open_pages(self, tool, target, ids):
        h = self.h
        out = []
        for ident in ids:
            reply = fitted_call(h, tool, {target: ident})
            if reply is None:
                break
            shown = reply.get("observation")
            h.observe(tool, shown)
            out.append(f"{ident}: " + h.show(shown)[:1500])
        return out

    def _trail_text(self, read, lost):
        h = self.h
        parts = []
        if read:
            parts.append("[pages a requirement refers to, opened before checkout]\n" + "\n".join(read) + "\n")
        if h.pointed:
            parts.append("[values taken from the referred pages] " + "; ".join(f"{k} -> {v}" for k, v in h.pointed.items()) + "\n")
        stop = bool(lost) and self.walk_stops < self.WALK_STOPS and h.left() > 5
        if stop:
            self.walk_stops += 1
            parts.append("A requirement that refers to another listing is met only once each page along that reference has been "
                         "opened itself (a result row does not count), including any page that refers further on. Not found yet: "
                         + " | ".join(lost[:3]) + ". Look it up by its seller, brand or wording, open its page, use the value it "
                         "states, and then check out.")
        return "".join(parts), stop

    # ---- live offers just before a checkout ----
    def offer_tool(self):
        return self.h.jobs.get("offer") or self.h.jobs.get("open")

    def refresh(self):
        """Open each basket line's offer again right before checkout; price and stock may move silently."""
        h = self.h
        tool = self.offer_tool()
        target = h.single_target(tool)
        since = max(h.step - 2, h.refused_at)
        stale = [i for i in self.basket_ids() if h.offer_seen.get(i, -99) < since and i not in h.gone] if tool and target else []
        if not stale or h.left() < len(stale) + 3:
            return ""
        changed = []
        for ident in stale:
            before = dict(h.seen.get(ident) or {})
            reply = fitted_call(h, tool, {target: ident})
            if reply is None:
                break
            shown = reply.get("observation")
            h.offer_seen[ident] = h.step
            h.observe(tool, shown)
            keys = _moved_keys(before, h.seen.get(ident) or {})
            if keys:
                changed.append(f"{ident} changed ({', '.join(keys[:4])}): {dump(clip(shown, 300))[:900]}")
        h.trace.append({"kind": "refresh", "ids": stale, "changed": len(changed)})
        h.epoch += 1 if changed else 0
        return "\n".join(changed)

    # ---- sold-out basket lines ----
    def swap_note(self):
        h = self.h
        lost = [i for i in self.basket_ids() if i not in h.gone and self.sold(i)]
        if not lost:
            return ""
        for ident in lost:
            h.gone[ident] = h.step
        h.shortlist = [i for i in h.shortlist if i not in lost]
        h.epoch += 1
        picks = self.swap(lost)
        text = "[auto] No longer sold: " + ", ".join(lost) + " (the store says it can no longer be purchased). That stays so; skip opening it from now on."
        if not picks:
            return text + " Take that line out of the basket and put in the best other fitting listing; search if none read so far fits."
        if h.left() <= 6:
            rest = ", ".join(picks[1:]) or "none"
            return (text + f" Few steps left: in this turn cart {picks[0]} at its requested count and drop the dead line "
                    f"(two calls), and send the checkout next. Other read listings that seem to qualify: {rest}.")
        if h.brief.get("pick") == "cheapest":
            return (text + f" Read listings that look like fits for that item, lowest price first: {', '.join(picks)}. The rule is "
                    f"the lowest price, so unless the whole fitting set is already known, search once below {picks[0]}'s price "
                    "(with the search's price ceiling, where it has one) before replacing the line with the cheapest fit.")
        return (text + f" Read listings that seem to qualify for that item: {', '.join(picks)}. In this turn drop the dead line "
                f"and cart {picks[0]} at its requested count; the checkout comes after.")

    def locate_sold(self):
        """A refusal mentioned a sell-out but no listing: with a single live line that line is taken as gone; with
        more, every line is read again and the fresh records tell."""
        ids = [i for i in self.basket_ids() if i not in self.h.gone]
        if not ids or any(self.sold(i) for i in ids):
            return ""
        if len(ids) == 1:
            self.h.unavailable.add(ids[0])
            return ""
        return self.refresh()

    def swap(self, lost):
        h = self.h
        pool = [] if h.cost > COST_EASE else self.records(LENS_SWAP, 25, lambda i: i in lost or i in h.gone or self.sold(i))
        if not pool:
            return []
        parts = [SWAP + self._brief_head(),
                 "UNAVAILABLE BASKET LINE:\n" + "\n".join(f"{i}: {self.record(i)}" for i in lost),
                 "BASKET NOW:\n" + self.basket_text(),
                 "OTHER LISTINGS SEEN (most recent first):\n" + "\n".join(pool)]
        got = h.ask_json("\n\n".join(parts), [DEEP_MODEL, FAST_MODEL], 1200)
        fits = self._swap_fits(got, lost)
        ranked = sorted(fits, key=lambda pair: pair[0]) if h.brief.get("pick") == "cheapest" else fits
        h.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return [ident for _, ident in ranked][:4]

    def _swap_fits(self, got, lost):
        """(stated price or infinity, id) for each proposed substitute that is known, live and not in the basket."""
        h = self.h
        fits = []
        for entry in items((got or {}).get("fits")):
            boxed = isinstance(entry, dict)
            ident = str(entry.get("id") if boxed else entry)
            price = entry.get("price") if boxed else None
            if ident in [i for _, i in fits] or ident not in h.seen or ident in lost or ident in h.gone:
                continue
            if ident in self.basket_ids():
                continue
            fits.append((float(price) if _is_number(price) else float("inf"), ident))
        return fits

    # ---- decline: keep going while there is still road, then review ----
    def _bare_items(self):
        """Requested items with no listing of their kind read in full yet."""
        h = self.h
        numbers = list(range(1, len(h.items) + 1))
        if not h.opened_facts:
            return [] if h.pages_read_ids else numbers
        slot = h.category_key()
        if not slot:
            return []
        kinds = set()
        for ident in h.opened_facts:
            label = str((h.seen.get(ident) or h.opened_facts.get(ident) or {}).get(slot) or "")
            if label.strip():
                kinds.add(stems(label))
        bare = []
        for number, item in enumerate(h.items, 1):
            want = stems(item.shelf or "")
            if want and not any(names_match(want, k) or names_match(k, want) for k in kinds):
                bare.append(number)
        return bare

    def _stay_reason(self):
        h = self.h
        live = {need for _, need in h.needs_now()}
        open_refs = [need for _, need in h.pointers if need in live and need not in h.pointed]
        if open_refs and "pointer" not in self.stays_used:
            return "pointer", ("A requirement takes its value from another listing that has not been opened yet: "
                               + " | ".join(open_refs[:3]) + ". Find that listing (search by its seller, brand or wording), "
                               "open its page and apply the value it states before deciding that nothing fits.")
        bare = self._bare_items()
        if bare and "kind" not in self.stays_used:
            names = "; ".join(f"item {n} ({h.items[n - 1].shelf or h.items[n - 1].name})" for n in bare[:4])
            return "kind", (f"No listing has been opened yet for {names}, and {h.left()} steps remain. Browse each of these kinds "
                            "before ending: search by its category and by plain words, page through the results and open the "
                            "closest listings. End without a purchase only after that.")
        held = self.basket_ids()
        lines = [i for i in held if not self.sold(i)]
        if lines and "basket" not in self.stays_used:
            fresh = self.key(held) not in self.asked
            self.order(None, COST_EASE)
            ok = self.verdict(held)
            if ok is False:
                if fresh:
                    self.asked.discard(self.key(held))
                return "", ""
            said = ("An independent review of these lines found no broken requirement. " if ok else "")
            return "basket", ("The basket still has " + ", ".join(lines) + ". Ending without a purchase discards it and "
                              "earns nothing if those lines fit. " + said + "Check out (correct a line first if one breaks a "
                              "requirement); end without a purchase only if you can name the requirement a line breaks and "
                              "no other listing seen meets it.")
        return "", ""

    def stay_note(self):
        """Refuses an early ending while steps and unexplored ground remain (each reason at most once)."""
        if not self.stay or len(self.stays_used) >= self.STAYS or self.h.left() <= self.ROOM:
            return ""
        reason, text = self._stay_reason()
        if not reason:
            return ""
        self.stays_used.add(reason)
        self.h.trace.append({"kind": "decline_stay", "reason": reason})
        return text

    def _decline_call(self):
        h = self.h
        pool = self.records(LENS_DECLINE, 30, self.sold)
        if not pool:
            return None
        prompt = AUDIT + self._brief_head() + "\n\nLISTINGS SEEN (most recent first):\n" + "\n".join(pool)
        got = h.ask_json(prompt, [DEEP_MODEL, FAST_MODEL], 1500)
        h.trace.append({"kind": "decline_review", "got": got})
        return got

    def _decline_fits(self, got):
        table = (got or {}).get("fits")
        if not isinstance(table, dict):
            return {}
        named = {}
        for number, ids in table.items():
            keep = [str(i) for i in items(ids) if str(i) in self.h.seen and not self.sold(str(i))][:4]
            if keep:
                named[str(number)] = keep
        return named

    def decline(self):
        h = self.h
        note = self.stay_note()
        if note:
            return note
        if self.declined_once or h.cost > COST_EASE:
            return ""
        self.declined_once = True
        got = self._decline_call()
        named = self._decline_fits(got)
        if not named:
            unread = [str(i) for i in items((got or {}).get("open"))]
            unread = [i for i in unread if i in h.seen and i not in h.pages_read_ids and not self.sold(i)]
            if unread and self.open_rows(unread):
                got = self._decline_call()
                named = self._decline_fits(got)
        if not named:
            return ""
        self.fits_by_item = named
        h.shortlist = [i for ids in named.values() for i in ids]
        why = got.get("reason")
        listed = "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items())
        return ("An independent review thinks these listings may fit: " + listed + (f" ({why})" if why else "")
                + ". Read them, order the one that really qualifies, and finish with no purchase only once all of them have failed.")

    def open_rows(self, rows):
        """Open listings seen only as search rows: all together through the side-by-side tool if it can, else two singly."""
        h = self.h
        steps = self._row_reads(rows)
        if not steps or h.left() < len(steps) + 5:
            return False
        count = len(h.opened_facts)
        for name, args in steps:
            faults = []
            shaped = fit_args(args, h.schemas.get(name), "args", faults)
            if faults:
                return False
            reply = h.call(name, shaped)
            if reply is None or h.finished:
                break
            h.observe(name, reply.get("observation"))
        h.trace.append({"kind": "decline_open", "rows": rows, "opened": len(h.opened_facts) - count})
        return len(h.opened_facts) > count

    def _row_reads(self, rows):
        h = self.h
        many, joiner = h.compare_target(), h.jobs.get("compare")
        if many and joiner and not h.compare_bare():
            return [(joiner, {many: rows[:h.compare_room()]})]
        single = h.facts_tool()
        target = h.single_target(single)
        return [(single, {target: i}) for i in rows[:2]] if single and target else []

    # ---- settle: buy the vouched lines instead of ending empty-handed ----
    def _settle_lines(self):
        h = self.h
        slot = h.category_key()
        kind = (lambda i: str((h.seen.get(i) or {}).get(slot) or "").strip().lower()) if slot else (lambda i: i)
        chosen = {}
        for number, ids in self.fits_by_item.items():
            index = whole(number)
            pick = next((i for i in ids if not self.sold(i)), None)
            if pick and index and 1 <= index <= len(h.items) and kind(pick) not in {kind(p) for p in chosen}:
                chosen[pick] = h.items[index - 1].qty
        if h.vouched not in chosen:
            return None
        taken = {kind(p) for p in chosen}
        others = [i for i in self.basket_ids() if i not in chosen]
        kept = {i: h.basket.get(i, 1) for i in others if not self.sold(i) and kind(i) not in taken}
        if len(chosen) + len({kind(i) for i in kept}) < len(h.items):
            return None
        return {**kept, **chosen}, [i for i in others if i not in kept]

    def settle(self):
        """The agent would end without buying although a check vouched for a reviewed listing: put the reviewed
        lines in the basket (one per item, keeping other items' lines) and check out, if a fresh check passes."""
        h = self.h
        cart, buy = h.jobs.get("cart"), h.jobs.get("buy")
        if self.settled or not h.vouched or self.sold(h.vouched) or not cart or not buy:
            return ""
        self.settled = True
        shaped = self._settle_lines()
        if shaped is None:
            return ""
        lines, drop = shaped
        calls = self.plan.calls(lines, drop)
        if not calls or len(calls) + 1 > h.left():
            return ""
        ids = list(lines)
        h.fill_facts(ids, len(calls) + 3)
        self.order(ids, COST_CAP)
        if not self.verdict(ids):
            return ""
        made = []
        for args in calls:
            faults = []
            args = fit_args(args, h.schemas.get(cart), "args", faults)
            if faults:
                return ""
            reply = h.call(cart, args)
            if reply is None or h.finished:
                return ""
            h.note_cart(cart, args)
            h.observe(cart, reply.get("observation"))
            made.append(dump(args))
        faults = []
        pay = fit_args(dict(h.brief.get("buy_args") or {}), h.schemas.get(buy), "args", faults)
        if faults:
            return ""
        said_back = self._pay(buy, pay, lines, drop)
        if not said_back:
            return ""
        return ("Not run (handled automatically): an independent check vouched for " + h.vouched + ", so the basket was set ("
                + "; ".join(made) + ") and the order placed. Store reply: " + " | then: ".join(said_back))

    def _pay(self, buy, args, lines, drop):
        h = self.h
        ids = list(lines)
        said_back = []
        for attempt in (0, 1):
            if attempt:
                self.refresh()
                self.order(ids, COST_CAP)
                blocked = (not self.verdict(ids) or h.left() < 1 or h.finished or any(self.sold(i) for i in lines))
                if blocked:
                    break
            go, said = h.duty_check("order")
            if said:
                said_back.append(said)
            if not go:
                break
            reply = h.call(buy, args)
            h.trace.append({"kind": "settled", "lines": lines, "drop": drop, "reply": clip(reply, 200)})
            if reply is None:
                break
            h.buys += 1
            if not reply.get("finished"):
                h.refused_at = h.step
            h.observe(buy, reply.get("observation"))
            said_back.append(h.show(reply.get("observation")))
            if h.finished or reply.get("finished"):
                break
        return said_back


NOT_THIS_SHAPE = object()


class CartPlan:
    """Builds, from the cart tool's schema, the calls after which the basket holds `lines` exactly and nothing in `drop`.

    Shapes tried in turn: a list-of-lines argument, then one line per call. Removing a line in the per-call shape uses,
    in turn: the tool's mode enum (second value removes), a quantity set to zero, or a negative quantity."""

    def __init__(self, host, review):
        self.h = host
        self.review = review

    @staticmethod
    def _is_text(spec):
        return isinstance(spec, dict) and "string" in kinds(spec) and "enum" not in spec

    @staticmethod
    def _is_int(spec):
        return isinstance(spec, dict) and "integer" in kinds(spec)

    @staticmethod
    def _is_choice(spec):
        return isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 2

    def calls(self, lines, drop):
        name = self.h.jobs.get("cart") or ""
        schema = self.h.schemas.get(name) or {}
        for shape in (self._list_shape, self._line_shape):
            got = shape(name, schema, lines, drop)
            if got is not NOT_THIS_SHAPE:
                return got
        return None

    def _list_shape(self, name, schema, lines, drop):
        for slot, spec in (schema.get("properties") or {}).items():
            if not (isinstance(spec, dict) and "array" in kinds(spec)):
                continue
            entry = spec.get("items")
            if not (isinstance(entry, dict) and isinstance(entry.get("properties"), dict)):
                continue
            inner = entry["properties"]
            id_field = next((k for k, s in inner.items() if self._is_text(s)), None)
            qty_field = next((k for k, s in inner.items() if self._is_int(s)), None)
            if not id_field:
                return None
            rows = []
            for ident, units in lines.items():
                row = {id_field: ident}
                if qty_field:
                    row[qty_field] = units
                rows.append(row)
            return [{slot: rows}]
        return NOT_THIS_SHAPE

    def _target(self, name, schema, texts):
        for tool, slot in self.h.id_args:
            if tool == name and slot in texts:
                return slot
        for slot in schema.get("required") or []:
            if field in texts:
                return field
        return texts[0] if len(texts) == 1 else None

    def _line_shape(self, name, schema, lines, drop):
        props = schema.get("properties") or {}
        texts = [k for k, s in props.items() if self._is_text(s)]
        target = self._target(name, schema, texts)
        qty = next((k for k, s in props.items() if self._is_int(s)), None)
        mode = next((k for k, s in props.items() if self._is_choice(s)), None)
        if not target:
            return None
        sets = self.h.cart_sets(name)
        out = []
        removers = (self._off_by_mode, self._off_by_zero, self._off_by_minus)
        for ident in drop:
            have = self.h.basket.get(ident, 1)
            made = None
            for remover in removers:
                made = remover(props, target, qty, mode, sets, ident, have)
                if made is not None:
                    break
            if made is None:
                return None
            out.append(made)
        present = self.review.basket_ids()
        for ident, want in lines.items():
            have = self.h.basket.get(ident, 0) if ident in present else 0
            if have == want:
                continue
            if have and not sets:
                return None
            call = {target: ident}
            if mode:
                call[mode] = props[mode]["enum"][0]
            if qty:
                call[qty] = want
            out.append(call)
        return out

    @staticmethod
    def _off_by_mode(props, target, qty, mode, sets, ident, have):
        if not mode:
            return None
        call = {target: ident, mode: props[mode]["enum"][1]}
        if qty:
            call[qty] = have
        return call

    @staticmethod
    def _off_by_zero(props, target, qty, mode, sets, ident, have):
        if mode or not (qty and sets):
            return None
        return {target: ident, qty: 0}

    @staticmethod
    def _off_by_minus(props, target, qty, mode, sets, ident, have):
        if mode or not qty or sets:
            return None
        floor = props[qty].get("minimum")
        if isinstance(floor, int) and floor <= -have:
            return {target: ident, qty: -have}
        return None


# ---- turn loop: call screening, the store call and what it teaches, the closing moves ----


PACING = True              # clock-aware loop and a turn cap that grows with the task's step limit
CLOCK_FLOOR_S = 45.0       # below this many seconds the model loop stops and the episode is closed
DUTY_FLOOR_S = 20.0        # below this the closing moves skip the request-step check and act at once
STAY_ON_DECLINE = True     # a no-purchase close is held back while a pointer, an unread kind or a basket line remains
TURN_HEADROOM = 10         # turns left unused under the step limit when the turn cap grows with it
EXTRA_TURNS = 10           # model turns allowed after a refused close
EXTRA_SPEND = 0.02
ASK_STALLS = 2             # interview rounds in a row that send nothing before asking is switched off

NO_FIT_LINE = "Sorry, I could not find a listing in this store that meets everything you asked for, so I am not placing an order."
CART_NOW = "\nBasket at the moment: "


@dataclass
class Verdict:
    """What a gate decides for one tool call: run it, stop it with a reply for the model, or run it changed."""
    kind: str = "run"
    text: str = ""
    args: dict | None = None
    note: str = ""
    lead: bool = False
    remark: bool = False

    @classmethod
    def go(cls):
        return cls()

    @classmethod
    def halt(cls, text):
        return cls(kind="stop", text=text)

    @classmethod
    def change(cls, args=None, note="", lead=False, remark=False):
        return cls(kind="rewrite", args=args, note=note, lead=lead, remark=remark)


@dataclass
class Call:
    """One tool call of a model turn, with what the gates learn about it on the way."""
    cid: str
    name: str
    raw: Any
    count: int
    merged: bool = False
    args: dict = field(default_factory=dict)
    note: str = ""
    side: str | None = None
    target: str | None = None
    ending: bool = False
    buying: bool = False
    room: bool = False
    deep: bool = False
    mark: str = ""
    readers: set = field(default_factory=set)
    reread: str = ""
    chain: str = ""


class Gates:
    """The ordered screens a tool call passes before it reaches the store. A screen that stops a call spends no step."""

    def __init__(self, desk):
        self.d = desk
        self.order = (
            self.merged, self.closed, self.talk_tool, self.unknown, self.narrowed, self.parsed, self.fitted,
            self.titles, self.side_cap, self.sold_target, self.absent_line, self.cart_exact, self.lone_end,
            self.end_reserve, self.repeat, self.refused_before, self.decline_again, self.cached, self.shown_one, self.shown_many, self.snapshot,
            self.buy_unchanged, self.buy_double, self.buy_wider, self.buy_refresh, self.buy_view, self.buy_swap, self.buy_pointers, self.buy_review,
            self.buy_reread, self.drop_review, self.drop_shortlist, self.drop_vouch,
            self.late_drop, self.early_drop, self.drop_settle, self.duty,
        )

    def judge(self, call):
        for screen in self.order:
            verdict = screen(call)
            if verdict is None or verdict.kind == "run":
                continue
            if verdict.kind == "stop":
                return verdict
            if verdict.args is not None:
                call.args = verdict.args
            if verdict.note:
                call.note = verdict.note + call.note if verdict.lead else call.note + verdict.note
            if verdict.remark:
                call.mark = call.name + dump(call.args)
        return Verdict.change(call.args, call.note)

    # ---- who may run at all ----
    def merged(self, c):
        if c.merged:
            return Verdict.halt("Already answered by the side-by-side result above: these listings were opened together in a single step.")
        return None

    def closed(self, c):
        return Verdict.halt("Not run: this episode has already ended.") if self.d.finished else None

    def talk_tool(self, c):
        if c.name == self.d.jobs.get("ask"):
            return Verdict.halt("Not run, no step spent: questions to the shopper are sent by the interviewing helper, not from this loop.")
        return None

    def unknown(self, c):
        if c.name in self.d.schemas:
            return None
        names = ", ".join(sorted(t["function"]["name"] for t in self.d.tools_now()))
        return Verdict.halt(f"Not run, no step spent: there is no tool by that name. Available now: {names}.")

    def narrowed(self, c):
        names = sorted(t["function"]["name"] for t in self.d.tools_now())
        if c.name in names:
            return None
        return Verdict.halt("Not run, no step spent: with the steps that remain, only these tools are open: " + ", ".join(names) + ".")

    # ---- arguments ----
    def parsed(self, c):
        if isinstance(c.raw, dict):
            return Verdict.change(c.raw)
        got = json_obj(c.raw) if (c.raw or "").strip() else {}
        if got is None:
            return Verdict.halt("Not run, no step spent: the arguments must form a single JSON object.")
        return Verdict.change(got)

    def fitted(self, c):
        faults = []
        shaped = fit_args(c.args, self.d.schemas[c.name], "args", faults)
        if faults:
            return Verdict.halt("Not run, no step spent: " + "; ".join(faults[:6]))
        return Verdict.change(shaped)

    def titles(self, c):
        fixed, problem = self.d.ids_from_titles(c.name, c.args)
        if problem:
            return Verdict.halt("Not run, no step spent: " + problem + ".")
        return Verdict.change(fixed)

    def side_cap(self, c):
        d = self.d
        c.side = d.side_arg() if c.name == d.jobs.get("compare") else None
        many = c.args.get(c.side) if c.side else None
        if not (isinstance(many, list) and d.side_bad and len(many) > d.side_room()):
            return None
        room = d.side_room()
        rest = ", ".join(map(str, many[room:]))
        note = f"\n[helper] Only the first {room} listings were sent: the store shows no more than {room} side by side. Still to view: {rest}."
        return Verdict.change({**c.args, c.side: many[:room]}, note)

    def sold_target(self, c):
        d = self.d
        c.target = d.single_target(c.name)
        if not c.target or c.name not in (d.jobs.get("open"), d.jobs.get("offer")):
            return None
        wanted = str(c.args.get(c.target))
        if wanted not in d.gone:
            return None
        return Verdict.halt(f"Not run, no step spent: {c.args[c.target]} went out of stock at step {d.gone[wanted]} and stays gone. "
                            "Swap its basket line for another listing that fits.")

    def absent_line(self, c):
        d = self.d
        if c.name != d.jobs.get("cart"):
            return None
        if not any((whole(v) or 0) < 0 for v in c.args.values() if not isinstance(v, str)):
            return None
        shown = dump(d.basket_view) if d.basket_seen >= d.basket_moved else ""
        missing = [v for v in c.args.values()
                   if isinstance(v, str) and v in d.seen and v not in d.basket and v not in shown and v not in d.basket_said]
        if not missing:
            return None
        return Verdict.halt(f"Not run, no step spent: {missing[0]} is not a basket line, so nothing can be taken off it." + CART_NOW + d.basket_text())

    def cart_exact(self, c):
        """A new line whose read attribute names another value than the asked one: held once per line, pointing to a read
        listing that states the value, or to a search for it when none does; the model may still confirm its choice."""
        d = self.d
        if c.name != d.jobs.get("cart"):
            return None
        lines = set(d.basket_ids())
        for ident in d.cart_adds(c.name, c.args):
            number = d.review.item_of(ident)
            off = {p: row for p, row in d.review.fit_table(ident, number).items() if row[0] in ("≠", "⊂")}
            mark = f"exact#{ident}#{d.epoch}"
            if ident in lines or not off or mark in d.held_marks:
                continue
            fits = d.review.exact_fits(number, lines | {ident}, d.review.room_for({ident}))
            d.held_marks.add(mark)
            read = "; ".join(f"{p} {k}={v[:50]} [{m}] against the asked {asked!r}" for p, (m, k, v, asked) in off.items())
            if not fits:
                return Verdict.halt(f"Held for now, no step spent. {ident} as read: {read}. No listing read so far states the "
                                    f"asked value: search for those words with the product's name, read what turns up and cart "
                                    f"the one that states it, or send this call again if {ident} really is the better fit."
                                    + CART_NOW + d.basket_text())
            shown = ", ".join(i if d.price_of(i) is None else f"{i} at {d.price_of(i):g}" for i in fits[:3])
            return Verdict.halt(f"Held for now, no step spent. {ident} as read: {read}. Listings already read whose attribute "
                                f"states the asked value itself: {shown}. Cart one of those at the requested count, or send this "
                                f"call again if {ident} really is the better fit." + CART_NOW + d.basket_text())
        return None

    # ---- finishing calls and step budget ----
    def lone_end(self, c):
        d = self.d
        c.ending = d.is_checkout(c.name, c.args) or d.is_decline(c.name, c.args)
        if c.ending and c.count > 1:
            return Verdict.halt("Not run, no step spent: a finishing call goes alone in its turn, after the other results have been read.")
        return None

    def end_reserve(self, c):
        d = self.d
        if c.ending or d.left() > d.keep_for_finish() or not (d.jobs.get("buy") or d.jobs.get("refuse")):
            return None
        return Verdict.halt("Not run, no step spent: the remaining steps are saved for the finish, either the checkout "
                            "(after the shopper approves the basket, when the request asks for that) or closing with no purchase.")

    # ---- calls whose answer is already known ----
    def repeat(self, c):
        d = self.d
        c.mark = c.name + dump(c.args)
        if c.ending or c.name == d.jobs.get("cart") or c.mark not in d.recent:
            return None
        return Verdict.halt("Not run, no step spent: the very same call was just made and nothing has moved since; its answer is above.")

    def decline_again(self, c):
        """A no-purchase close sent right after the store turned one down is held once, so the reply gets acted on."""
        d = self.d
        mark = f"decline#{d.step}"
        if not d.decline_reply or not d.is_decline(c.name, c.args) or d.decline_step != d.step or mark in d.held_marks:
            return None
        d.held_marks.add(mark)
        return Verdict.halt("Held for now, no step spent. The store did not take that close and its reply changes what is "
                            "wanted. Search again under the new limit and requirements, read what turns up, and close with "
                            "no purchase only if nothing fits even then.\n" + d.status())

    def refused_before(self, c):
        d = self.d
        kept = d.turned_down.get(c.mark)
        if not kept or c.ending or kept[1:] != (d.epoch, d.basket_mark()):
            return None
        fault = kept[0]
        return Verdict.halt(f"Not run, no step spent: this exact call was turned down before ({fault}) and nothing about it "
                            "has changed. Change the arguments or use another tool.")

    def cached(self, c):
        d = self.d
        c.readers = {d.jobs.get(job) for job in ("search", "open", "offer", "compare")} - {None}
        kept = d.replay_cache.get(c.mark)
        if c.name not in c.readers or not kept or kept[0] != d.epoch:
            return None
        return Verdict.halt(f"Not run, no step spent: step {kept[1]} made this call already and nothing changed after it. It returned:\n" + kept[2][:3000])

    def shown_one(self, c):
        d = self.d
        if not c.target or c.name not in c.readers or c.name == d.jobs.get("compare"):
            return None
        wanted = str(c.args.get(c.target))
        if not d.shown_already(c.name, wanted):
            return None
        return Verdict.halt(f"Not run, no step spent: {c.args[c.target]} already showed all this view has, and nothing changed since. "
                            "What is on record for it: " + d.record_of(wanted))

    def shown_many(self, c):
        d = self.d
        many = c.args.get(c.side) if c.side else None
        if not (isinstance(many, list) and many):
            return None
        fresh = [i for i in many if not d.shown_already(c.name, str(i))]
        if not fresh:
            listing = "\n".join(f"{i}: {d.record_of(str(i))}" for i in many)[:4000]
            return Verdict.halt("Not run, no step spent: every one of these listings already showed all this view has, "
                                "and nothing changed since. On record:\n" + listing)
        if len(fresh) == len(many):
            return None
        skipped = ", ".join(str(i) for i in many if i not in fresh)
        return Verdict.change({**c.args, c.side: fresh}, f"\n[helper] Skipped {skipped}: fully shown before and unchanged since.", remark=True)

    def snapshot(self, c):
        d = self.d
        c.room = c.ending and d.left() > 2
        c.buying = d.is_checkout(c.name, c.args)
        c.deep = d.left() > 5
        return None

    # ---- before a checkout ----
    def _buy_on(self, c):
        return c.room and c.buying

    def buy_unchanged(self, c):
        """A checkout sent again right after a refusal, with the basket and the task as they were, is held back once."""
        d = self.d
        mark = d.basket_mark() + f"#{d.epoch}"
        if not (BASKET_CHECKS and c.buying) or d.refused_mark != mark or mark in d.held_marks:
            return None
        d.held_marks.add(mark)
        return Verdict.halt("Not run, no step spent: the store refused the last checkout and neither the basket nor the task has "
                            "changed since. Read the refusal again, fix what it names (a line, a count, a price, the budget) "
                            "or re-read the basket lines, then check out." + CART_NOW + d.basket_text())

    def buy_double(self, c):
        """More basket lines of one kind than requested products of that kind: held back once per basket."""
        d = self.d
        if not (BASKET_CHECKS and self._buy_on(c)):
            return None
        mark = "double#" + d.basket_mark()
        extra = d.doubled_lines()
        if not extra or mark in d.held_marks:
            return None
        d.held_marks.add(mark)
        return Verdict.halt("Held for now, no step spent. The basket has more lines of one kind than the request asks for: "
                            + "; ".join(extra) + ". Keep one line per requested product (the best fit), take the others "
                            "out, then check out." + CART_NOW + d.basket_text())

    def buy_wider(self, c):
        if not self._buy_on(c):
            return None
        hint = self.d.look_wider()
        return Verdict.halt(hint + CART_NOW + self.d.basket_text()) if hint else None

    def buy_refresh(self, c):
        if self._buy_on(c):
            c.reread = self.d.refresh_lines()
        return None

    def buy_view(self, c):
        d = self.d
        viewer = d.jobs.get("cart_view")
        if not self._buy_on(c) or not d.basket_touched or d.basket_seen >= d.basket_moved or not viewer or d.left() <= 3:
            return None
        faults = []
        view_args = fit_args({}, d.schemas.get(viewer), "args", faults)
        reply = d.call(viewer, view_args) if not faults else None
        if reply is None:
            return None
        d.basket_view = reply.get("observation")
        d.basket_seen = d.step
        moved = d.observe(viewer, d.basket_view)
        if not moved:
            return None
        d.queued = []
        return Verdict.halt("Held for now, no step spent. The basket view showed a change: " + moved + CART_NOW + d.basket_text()
                            + "\nTake the change into account, repair the basket if it needs it, then check out.")

    def buy_swap(self, c):
        d = self.d
        if not self._buy_on(c):
            return None
        swap = d.replace_note()
        lost = [i for i in d.basket_ids() if i in d.gone] if d.held < 2 else []
        if not swap and not lost:
            return None
        if not swap:
            d.held += 1
        why = swap or f"{', '.join(lost)} in the basket is out of stock: replace that line before checking out."
        return Verdict.halt("Held for now, no step spent. " + why + CART_NOW + d.basket_text())

    def buy_pointers(self, c):
        if not self._buy_on(c):
            return None
        c.chain, wait = self.d.follow_pointers()
        return Verdict.halt("Held for now, no step spent. " + c.chain + CART_NOW + self.d.basket_text()) if wait else None

    def buy_review(self, c):
        d = self.d
        if not self._buy_on(c):
            return None
        read = d.open_facts(d.basket_ids(), d.keep_for_finish() + 3) + d.cheaper()
        found = d.review_order()
        if not found:
            return None
        opened = ("To judge the basket, the attributes of these listings were read first (one step each):\n" + "\n".join(read) + "\n") if read else ""
        return Verdict.halt("Held for now. " + c.chain + opened + found + CART_NOW + d.basket_text())

    def buy_reread(self, c):
        d = self.d
        if not self._buy_on(c) or not c.reread or d.cost <= COST_EASE:
            return None
        return Verdict.halt("Held for now, no step spent. The basket's listings were read again just now:\n" + c.reread
                            + "\nIf every line still has stock and fits the money limit, send the checkout again; if one does not, swap it out first.")

    # ---- before closing with no purchase ----
    def _drop_on(self, c):
        return c.room and not c.buying and c.deep

    def drop_review(self, c):
        if not self._drop_on(c):
            return None
        found = self.d.review_decline()
        return Verdict.halt("Held for now, no step spent. " + found) if found else None

    def drop_shortlist(self, c):
        d = self.d
        if not self._drop_on(c) or not d.shortlist or d.left() <= 4 or d.refuse_blocks != 0:
            return None
        d.refuse_blocks = 1
        d.trace.append({"kind": "decline_stopped", "shortlist": list(d.shortlist)})
        return Verdict.halt("Held for now, no step spent. The review turned up listings that look like fits: " + ", ".join(d.shortlist)
                            + ". Read any of them not yet opened, cart the strongest fit (obeying the pick rule) and send the checkout. "
                            "A no-purchase finish is right only if each of them clearly fails some requirement; then say which requirement in "
                            "your message and repeat the closing call.")

    def drop_vouch(self, c):
        d = self.d
        if not self._drop_on(c) or not d.shortlist or d.left() <= 4 or d.refuse_blocks != 1:
            return None
        d.refuse_blocks = 2
        first = d.shortlist[0]
        d.review_order([first], COST_CAP)
        if not d.review_verdict([first]) or first not in d.shortlist:
            return None
        d.vouched = first
        d.trace.append({"kind": "decline_stopped", "vouched": first})
        return Verdict.halt(f"Held for now, no step spent. A separate check of the records for {first} finds that it meets every requirement. "
                            "Put it in the basket at the asked quantity and check out; close with no purchase only when its record plainly breaks a requirement.")

    def late_drop(self, c):
        d = self.d
        if not d.is_decline(c.name, c.args) or d.left() > 5 or d.late_held:
            return None
        lines = [i for i in d.best_basket if not d.sold_out(i)]
        fresh = d.basket_view is not None and d.basket_seen >= d.basket_moved
        now = d.basket_ids()
        missing = [i for i in lines if i not in now or (i in d.dropped and not fresh)]
        if len(lines) < len(d.items) or len(missing) + 1 > d.left():
            return None
        d.late_held = True
        d.trace.append({"kind": "late_decline_stopped", "lines": lines, "missing": missing})
        redo = (f"Add {', '.join(missing)} back to the basket at the asked quantities, then check out." if missing
                else "Check out the basket as it stands.")
        return Verdict.halt("Held for now, no step spent. Closing with no purchase pays nothing when the store does have a fitting listing, "
                            "and being low on steps is no proof that it holds none. " + redo
                            + " Close with no purchase only if your searching showed that some requested product has no listing meeting "
                            "its requirements; then send the closing call again.")

    def early_drop(self, c):
        d = self.d
        if not d.is_decline(c.name, c.args) or d.shortlist or d.pushed or d.left() <= d.max_steps // 3:
            return None
        d.pushed = True
        return Verdict.halt(f"Held for now, no step spent. {d.left()} steps are still free, so try other routes before finishing empty-handed: "
                            "walk each requested product's category page after page using the continuation value the store returns, "
                            "query single words for the wanted values and for other names of the property, and open listings that might qualify. "
                            "Finish with no purchase only after that.")

    def drop_settle(self, c):
        d = self.d
        if not d.is_decline(c.name, c.args):
            return None
        done = d.settle()
        if done:
            return Verdict.halt(done + (" [the episode is over]" if d.finished else ""))
        return Verdict.change(d.hush(c.name, c.args))

    def duty(self, c):
        d = self.d
        if not c.ending or d.left() < 2:
            return None
        clear, said = d.duty_check("order" if d.is_checkout(c.name, c.args) else "end")
        if not clear:
            return Verdict.halt("Held for now. " + said + CART_NOW + d.basket_text())
        return Verdict.change(note="\n[helper] " + said, lead=True) if said else None


def _takes_off(props, args):
    """A cart call whose mode argument holds a value past the first of its enum (the first one adds)."""
    for key, value in args.items():
        choices = (props.get(key) or {}).get("enum")
        if isinstance(choices, list) and len(choices) >= 2 and value in choices[1:]:
            return True
    return False


def _object_rows(row_spec):
    return isinstance(row_spec, dict) and ("properties" in row_spec or row_spec.get("type") == "object")


def _next_count(mode, have, n):
    if mode == "off":
        return have - n if n else 0
    step = 1 if n is None else n
    return step if mode == "set" else have + step


BASKET_CHECKS = True   # hold a checkout once for doubled lines of one kind, or when nothing changed since a refusal
READ_FAULTS = True


def _code_like(value):
    """A short upper-case token with an underscore, the usual shape of a machine error code."""
    return isinstance(value, str) and "_" in value and re.fullmatch(r"[A-Z][A-Z0-9_]{2,40}", value) is not None


def _call_args(given):
    if isinstance(given, dict):
        return given
    parsed = json_obj(given) if isinstance(given, str) and given.strip() else {}
    return parsed if isinstance(parsed, dict) else {}


class Steering:
    """The model loop of an episode and everything between a model's tool call and the store. Mixed into Driver."""

    paced = PACING

    # ---- closing calls and the tools on offer ----
    def _fixed(self, key):
        fixed = self.brief.get(key)
        return fixed if isinstance(fixed, dict) else {}

    def is_checkout(self, name, args):
        return name == self.jobs.get("buy") and all(args.get(k) == v for k, v in self._fixed("buy_args").items())

    def is_decline(self, name, args):
        return name == self.jobs.get("refuse") and all(args.get(k) == v for k, v in self._fixed("refuse_args").items())

    def tools_now(self):
        """The tools the model may call this turn; near the end of the step budget only the closing ones (and the basket)."""
        jobs = self.jobs
        closers = {jobs.get("buy"), jobs.get("refuse")} - {None}
        talk = jobs.get("ask")
        open_tools = [t for t in self.tools if t["function"]["name"] != talk]
        allowed = self._narrowed(closers) if closers else None
        if allowed is None:
            return open_tools
        return [t for t in open_tools if t["function"]["name"] in allowed] or open_tools

    def _narrowed(self, closers):
        """The tool names still open when few steps are left, None while all are."""
        jobs, left, tail = self.jobs, self.left(), self.keep_for_finish()
        basket = {jobs.get("cart"), jobs.get("cart_view")}
        if left <= tail:
            return set(closers)
        if left <= tail + 3:
            return closers | basket
        if self.buys == 0 and left <= len(self.items) + 5:
            return closers | basket | {jobs.get("offer")}
        return None

    def hush(self, name, args):
        """Free-text fields of a no-purchase call get a plain apology when they name listings or the request wants silence."""
        schema = self.schemas.get(name) or {}
        needed = schema.get("required") or []
        quiet = self.brief.get("quiet_decline") is not False
        out = dict(args)
        for key, spec in (schema.get("properties") or {}).items():
            free_text = isinstance(spec, dict) and "enum" not in spec and "string" in kinds(spec)
            if not free_text or (out.get(key) is None and key not in needed):
                continue
            text = str(out.get(key) or "")
            if quiet or not text or any(ident in text for ident in self.seen):
                out[key] = NO_FIT_LINE[:spec.get("maxLength", 200)]
        return out

    def ids_from_titles(self, name, args):
        """A title given where a tool expects listing ids is swapped for the id of the single known listing whose text
        starts with it; when none or several do, the call is stopped with the reason and costs no step."""
        key = self._title_slot(name)
        if key is None or key not in args:
            return args, ""
        given = args[key]
        swapped = []
        for value in given if isinstance(given, list) else [given]:
            ident, problem = self._id_for(value)
            if problem:
                return args, problem
            swapped.append(ident)
        return {**args, key: swapped if isinstance(given, list) else swapped[0]}, ""

    def _title_slot(self, name):
        """The one free-text or list parameter through which a listing tool takes ids, once known to carry them."""
        jobs = self.jobs
        if not self.id_key or name not in {jobs.get(j) for j in ("cart", "open", "offer", "compare")} - {None}:
            return None
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, spec in props.items()
                if isinstance(spec, dict) and "enum" not in spec and ("string" in kinds(spec) or "array" in kinds(spec))]
        if len(free) != 1 or (name != jobs.get("cart") and (name, free[0]) not in self.id_args):
            return None
        return free[0]

    def _id_for(self, value):
        text = value.strip() if isinstance(value, str) else None
        if text in self.seen:
            return text, ""
        if text is None or len(text) < 8 or looks_id(value, True):
            return value, ""
        start = text.lower()
        hits = [ident for ident, row in self.seen.items() if self._titled(row, start)]
        if len(hits) == 1:
            return hits[0], ""
        if hits:
            return value, f"the text {value!r} matches more than one listing ({', '.join(hits[:6])}); give a single listing id"
        return value, f"{value!r} is no listing id; use the id shown in the result rows (the {self.id_key} field)"

    def _titled(self, row, start):
        """Whether some text field of a row (other than its id) begins with the given words."""
        return any(k != self.id_key and isinstance(v, str) and v.strip().lower().startswith(start) for k, v in row.items())

    # ---- the basket as the cart calls left it ----
    def cart_sets(self, name):
        """A cart tool with no add/remove choice whose count may be 0 sets a line's units; 0 takes the line out."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        for spec in props.values():
            if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 2:
                return False
        return any(isinstance(spec, dict) and spec.get("minimum") == 0 and "integer" in kinds(spec) for spec in props.values())

    @staticmethod
    def _units(values, fallback):
        return next((whole(v) for v in values if not isinstance(v, str) and whole(v) is not None), fallback)

    def note_cart(self, name, args):
        """Follow the basket through a cart call: a list of line objects replaces it, other calls add, set or take off."""
        self.basket_touched = True
        self.basket_moved = self.step
        props = (self.schemas.get(name) or {}).get("properties") or {}
        lines, replaced = self._cart_lines(props, args)
        if replaced:
            self.basket = {ident: n for ident, n in lines if n and n > 0}
            return
        mode = "off" if _takes_off(props, args) else ("set" if self.cart_sets(name) else "add")
        for ident, n in lines:
            count = _next_count(mode, self.basket.get(ident, 0), n)
            if count > 0:
                self.basket[ident] = count
            else:
                self.basket.pop(ident, None)

    def cart_adds(self, name, args):
        """The listings whose basket count a cart call would raise."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        lines, replaced = self._cart_lines(props, args)
        mode = "off" if _takes_off(props, args) else ("set" if self.cart_sets(name) else "add")
        out = []
        for ident, n in lines:
            have = self.basket.get(ident, 0)
            if ((n or 0) if replaced else _next_count(mode, have, n)) > have:
                out.append(ident)
        return out

    def _cart_lines(self, props, args):
        """(listing, units) pairs a cart call names, and whether its argument is a full list of line objects."""
        units = self._units(args.values(), None)
        lines = []
        replaced = False
        for key, value in args.items():
            if isinstance(value, str):
                lines += [(value, units)] if value in self.seen else []
            elif isinstance(value, list):
                replaced = replaced or (not value and _object_rows((props.get(key) or {}).get("items") or {}))
                for entry in value:
                    found, boxed = self._cart_entry(entry, units)
                    lines += found
                    replaced = replaced or boxed
        return lines, replaced

    def _cart_entry(self, entry, units):
        if isinstance(entry, dict):
            ident = next((v for v in entry.values() if isinstance(v, str) and v in self.seen), None)
            return ([(ident, self._units(entry.values(), 1))] if ident else []), True
        return ([(entry, units)] if isinstance(entry, str) and entry in self.seen else []), False

    def show(self, observation):
        text = dump(clip(observation, FIELD_CHARS))
        return text if len(text) <= OBS_CHARS else text[:OBS_CHARS] + "...(cut)"

    def show_listed(self, observation):
        """A listing result as the model reads it: a row already shown unchanged in this visit comes back short, since its
        full copy is further up, and a result of nothing but such rows says so."""
        told = self.__dict__.setdefault("rows_told", {})
        tally = {"new": 0, "again": 0}

        def row_view(row, ident):
            stamp = dump({k: v for k, v in row.items() if not isinstance(v, (str, list, dict))})
            if told.get(ident) == stamp:
                tally["again"] += 1
                return clip(row, ROW_AGAIN_CHARS)
            told[ident] = stamp
            tally["new"] += 1
            return clip(row, FIELD_CHARS)

        def walk(node, depth=0):
            if depth > 7:
                return "..."
            if isinstance(node, dict):
                return {name: walk(part, depth + 1) for name, part in node.items()}
            if isinstance(node, list):
                ids = [self.catalog.id_of(part, True) if isinstance(part, dict) else None for part in node]
                if ids and all(ids):
                    return [row_view(part, ident) for part, ident in zip(node, ids)]
                return [walk(part, depth + 1) for part in node]
            return clip(node, FIELD_CHARS)

        text = dump(walk(observation))
        text = text if len(text) <= OBS_CHARS else text[:OBS_CHARS] + "...(cut)"
        if tally["again"] and not tally["new"]:
            text += ("\n[helper] Every row above was shown earlier in this visit (short copies): rewording the same words "
                     "brings no others. Decide among the listings already shown, or read the next page.")
        return text

    # ---- side-by-side view ----
    def side_arg(self):
        schema = self.schemas.get(self.jobs.get("compare") or "") or {}
        props = schema.get("properties") or {}
        key = next((k for k, spec in props.items() if isinstance(spec, dict) and "array" in kinds(spec)), None)
        if not key or any(k != key for k in schema.get("required") or []):
            return None
        return key

    def side_room(self):
        """How many listings fit in one side-by-side call: the schema's maxItems, else what earlier calls showed works."""
        spec = (self.schemas.get(self.jobs.get("compare") or "") or {}).get("properties") or {}
        cap = (spec.get(self.side_arg() or "") or {}).get("maxItems")
        if isinstance(cap, int):
            return cap
        ok, bad = self.side_ok, self.side_bad
        return max(2, min(bad - 1, max(ok, 2))) if bad else max(4, ok)

    compare_target = side_arg
    compare_room = side_room

    def merge_opens(self, calls):
        """Single-listing opens requested in the same turn travel as one side-by-side call, spending one step in all."""
        opener, joiner = self.jobs.get("open"), self.jobs.get("compare")
        target, side = self.single_target(opener), self.side_arg()
        usable = opener and joiner and target and side and opener != joiner
        if not usable or self.compare_bare():
            return calls
        found = (self._single_open(n, call, opener, target) for n, call in enumerate(calls))
        picks = [pair for pair in found if pair][:self.side_room()]
        if len(picks) < 2:
            return calls
        ids = [ident for _, ident in picks]
        lead, folded = picks[0][0], dict(picks)
        joined = {"name": joiner, "arguments": dump({side: ids})}
        self.trace.append({"kind": "merged_opens", "ids": ids})

        def fold(n, call):
            if n == lead:
                return {"id": call["id"], "type": "function", "function": joined}
            return {"id": call["id"], "merged": True} if n in folded else call
        return [fold(n, call) for n, call in enumerate(calls)]

    def _single_open(self, n, call, opener, target):
        spec = call.get("function") or {}
        raw = spec.get("arguments")
        args = json_obj(raw) if isinstance(raw, str) else None
        if spec.get("name") != opener or not args or set(args) != {target}:
            return None
        ident = str(args.get(target) or "").strip()
        return (n, ident) if ident in self.seen and ident not in self.gone else None

    # ---- one model turn's calls ----
    def judge_calls(self, calls):
        gates = self.__dict__.get("_gates")
        if gates is None:
            gates = self.__dict__["_gates"] = Gates(self)
        calls = self.merge_opens(calls)
        out = []
        for entry in calls:
            spec = entry.get("function") or {}
            call = Call(cid=entry["id"], name=spec.get("name"), raw=spec.get("arguments"), count=len(calls), merged=bool(entry.get("merged")))
            verdict = gates.judge(call)
            out.append((call.cid, verdict.text if verdict.kind == "stop" else self.carry_out(call)))
        return out

    def carry_out(self, c):
        """Send the call to the store, learn from the result, and build the text the model reads back."""
        reply = self.call(c.name, c.args)
        if reply is None:
            return "The call did not go through (network or server trouble). It may be tried again."
        seen = reply.get("observation")
        fault = "" if c.ending else self.fault_in(seen)
        if fault:
            self.turned_down[c.mark] = (fault, self.epoch, self.basket_mark())
            self.recent = (self.recent + [c.mark])[-3:]
            said = f"\n[helper] The store turned this call down ({fault}); nothing changed, the basket included."
            return self.show(seen) + c.note + said + self._read_back(c, seen, reply, False)
        note = c.note + self._learn_side(c, seen)
        self.recent = (self.recent + [c.mark])[-3:]
        listed, paged = self._learn_result(c, seen)
        buying = self._count_checkout(c, reply)
        shown = self.show_listed(seen) if listed else self.show(seen)
        base = shown + paged + note + (self.ranked_here([seen] + self.paged_obs) if listed else "")
        text = base + self._read_back(c, seen, reply, buying)
        if self.is_decline(c.name, c.args) and not reply.get("finished"):
            self.decline_reply = "closing with no purchase was refused: " + base
            self.decline_step = self.step
        if c.name in c.readers:
            self.replay_cache[c.mark] = (self.epoch, self.step, base)
        return text + (" [the episode is over]" if reply.get("finished") else "")

    def fault_in(self, seen):
        """The error a result reports instead of an answer: an error field, a *_code or fault field holding a code,
        or a lone short upper-case code. Listing records and row lists never count."""
        if not READ_FAULTS or not isinstance(seen, dict) or self.ident(seen) or self.records_in(seen):
            return ""
        for key, value in seen.items():
            low = str(key).lower()
            if low in ("error", "errors") and value:
                return dump(value)[:160]
            coded = low.endswith("code") or "fault" in low
            if coded and (_code_like(value) or (whole(value) or 0) >= 400):
                return f"{key}={value}"
        texts = [v for v in seen.values() if isinstance(v, str)]
        if len(seen) <= 3 and len(texts) == 1 and _code_like(texts[0]):
            return texts[0]
        for key, value in seen.items():
            low = str(key).lower()
            said = value.strip() if isinstance(value, str) else ""
            if said and any(w in low for w in ("problem", "fault", "error")) and (" " in said or re.fullmatch(r"[A-Z]{1,3}\d{1,4}", said)):
                return f"{key}={said[:150]}"
        return ""

    def _learn_result(self, c, seen):
        """Fold a store result into the books; returns whether it was a listing search and any pages read after it."""
        name, args, jobs = c.name, c.args, self.jobs
        self._learn_id_args(name, args)
        self.take(seen)
        rows = self.records_in(seen) if name == jobs.get("compare") else []
        if rows:
            self.shows_facts[name] = self.shows_facts.get(name, False) or any(fact_block(r) for _, r in rows)
        if name in c.readers - {jobs.get("search")}:
            self.note_view(name, seen)
        listed = not c.ending and name not in {tool for job, tool in jobs.items() if job != "search"}
        paged = self.more_pages(name, args, seen) if listed else ""
        self._learn_views(name, seen)
        if name == jobs.get("cart"):
            self._after_cart(name, args, seen)
        if name == jobs.get("cart_view"):
            self.basket_view, self.basket_seen = seen, self.step
        return listed, paged

    def _count_checkout(self, c, reply):
        buying = self.is_checkout(c.name, c.args)
        if buying:
            self.buys += 1
            if not reply.get("finished"):
                self.refused_at = self.step
                self.refused_mark = self.basket_mark() + f"#{self.epoch}"
        return buying

    def doubled_lines(self):
        """Per kind of listing, the basket lines beyond the number of requested products of that kind."""
        field_name = self.category_key()
        lines = [i for i in self.basket_ids() if not self.sold_out(i)]
        if not field_name or len(lines) <= max(1, len(self.items)):
            return []
        by_kind = {}
        for ident in lines:
            kind = str((self.seen.get(ident) or {}).get(field_name) or "").strip()
            by_kind.setdefault(kind, []).append(ident)
        out = []
        for kind, ids in by_kind.items():
            wanted = sum(1 for it in self.items if it.shelf and names_match(stems(it.shelf), stems(kind)))
            if kind and len(ids) > max(1, wanted):
                out.append(f"{', '.join(ids)} are all {kind}")
        return out

    def _learn_side(self, c, seen):
        many = c.args.get(c.side) if c.side else None
        if not (isinstance(many, list) and len(many) > 1):
            return ""
        shown = dump(seen)
        if any(str(i) in shown for i in many):
            self.side_ok = max(self.side_ok, len(many))
            return ""
        self.side_bad = min(self.side_bad or 999, len(many))
        return f"\n[helper] The side-by-side view returned nothing for {len(many)} listings: send no more than {self.side_room()} in one call."

    def _learn_id_args(self, name, args):
        for key, value in args.items():
            values = value if isinstance(value, list) else [value]
            if values and all(isinstance(v, str) and v in self.seen for v in values):
                self.id_args.add((name, key))

    def _learn_views(self, name, seen):
        """A single-listing tool whose reply had no attributes gives up the attributes job to another such tool, and
        serves as the price-and-stock view if none is known yet."""
        jobs = self.jobs
        closing = (jobs.get("cart"), jobs.get("buy"), jobs.get("refuse"), jobs.get("ask"))
        single = self.single_target(name) and name not in closing
        listing = self.ident(seen) if single and isinstance(seen, dict) else None
        if not listing:
            return
        shows = self.shows_facts.get(name, False) or bool(fact_block(seen))
        self.shows_facts[name] = shows
        if not shows and name == jobs.get("open"):
            jobs["open"] = self._other_reader(name, closing + (jobs.get("search"),)) or name
        if not (shows or name == jobs.get("open") or jobs.get("offer")):
            jobs["offer"] = name
        if name == self.offer_tool():
            self.offer_seen[listing] = self.step

    def _other_reader(self, name, skip):
        return next((t for t in self.schemas
                     if t != name and t not in skip and self.single_target(t) and self.shows_facts.get(t) is not False), None)

    def _after_cart(self, name, args, seen):
        """Track the basket, lines a zero or negative count took off (unless the reply still lists them), and the
        fullest basket so far, which a late no-purchase close would otherwise throw away."""
        self.note_cart(name, args)
        self.basket_said = dump(seen)
        self.recent = []
        count = self._units(args.values(), None)
        named = [v for v in args.values() if isinstance(v, str) and v in self.seen] if count is not None else []
        for ident in named:
            if count > 0:
                self.dropped.discard(ident)
            elif not in_rows(seen, ident):
                self.dropped.add(ident)
        kept = [i for i in self.basket_ids() if not self.sold_out(i) and i not in self.dropped]
        if len(kept) >= len(self.best_basket):
            self.best_basket = kept

    def _read_back(self, c, seen, reply, buying):
        """Extras after the raw result: matched evidence, cart-count hints, notes read, sold-out follow-ups."""
        pieces = [self.proof(seen), self._zero_hint(c, seen)]
        self.note_sold = False
        pieces.append(self._note_line(seen))
        if buying and not reply.get("finished"):
            pieces.append(self._sold_line(seen))
        pieces.append(self.replace_note())
        pieces.append(self._queued_line())
        return "".join("\n" + piece for piece in pieces if piece)

    def _zero_hint(self, c, seen):
        if c.name != self.jobs.get("cart") or not any(whole(v) == 0 for v in c.args.values() if not isinstance(v, str)):
            return ""
        kept = any(isinstance(v, str) and v in self.seen and in_rows(seen, v) for v in c.args.values())
        return ("[helper] Sending 0 did not clear the line: this cart tool adds what it is given to the line's count, "
                "so take units away with a negative number.") if kept else ""

    def _note_line(self, seen):
        lines = self.loose_lines(seen)
        meaning = self.read_note(lines) if lines else ""
        return f"[note read] {meaning}. Carry this change into your picks before checking out." if meaning else ""

    def _sold_line(self, seen):
        said = spoken(seen)
        if not (self.note_sold or any(line in said for line in self.sold_lines)):
            return ""
        found = self.locate_sold()
        return "[basket lines read again after the refusal]\n" + found if found else ""

    def _queued_line(self):
        if not self.queued:
            return ""
        line = "[notes from the helper's own calls] " + " | ".join(self.queued)
        self.queued = []
        return line

    # ---- closing the episode ----
    def _duty_time(self):
        return not self.paced or self.clock.left_s() >= DUTY_FLOOR_S

    def _close_with(self, tool, args, stage):
        """Run the request-step check when there is room, then the closing call. False when nothing more can follow."""
        if self.left() >= 2 and self._duty_time():
            self.duty_check(stage)
        if self.finished or self.left() <= 0:
            return False, None
        reply = self.call(tool, args)
        self.trace.append({"kind": "closing_" + stage, "reply": clip(reply, 200)})
        return True, reply

    def wrap_up(self, again=False):
        """Finish the episode: a basket gets a checkout, else a no-purchase close. With `again`, a checkout the store
        turns down is handed back as text for the model to deal with, and no no-purchase close follows it."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        said = self._close_buying(again)
        return self._close_empty() if said is None else said

    def _fixed_args(self, tool, args):
        faults = []
        shaped = fit_args(args, self.schemas.get(tool), "args", faults)
        return None if faults else shaped

    def _close_buying(self, again):
        """The checkout at the close: None when it was not tried or may be followed by a no-purchase close."""
        buy = self.jobs.get("buy")
        fresh = self.basket_view is not None and self.basket_seen >= self.basket_moved
        holding = bool(self.basket_ids()) if fresh else bool(self.basket)
        if not buy or not (holding or (self.basket_touched and not fresh and self.left() >= 2)):
            return None
        args = self._fixed_args(buy, dict(self._fixed("buy_args")))
        if args is None:
            return None
        going, reply = self._close_with(buy, args, "order")
        if not going or self.finished or self.left() <= 0:
            return ""
        if reply is None or not again:
            return None
        self.buys += 1
        self.refused_at = self.step
        shown = reply.get("observation")
        self.observe(buy, shown)
        return "the checkout was refused: " + self.show(shown)

    def _close_empty(self):
        refuse = self.jobs.get("refuse")
        args = self._fixed_args(refuse, self.hush(refuse, dict(self._fixed("refuse_args")))) if refuse else None
        if args is None:
            return ""
        going, reply = self._close_with(refuse, args, "end")
        if not going or reply is None or self.finished:
            return ""
        shown = reply.get("observation")
        self.observe(refuse, shown)
        return "closing with no purchase was refused: " + self.show(shown)

    def shrink_history(self):
        """Over the size limit, the older half of the tool results is cut short, keeping the listing ids it named."""
        if sum(len(dump(m.get("content"))) for m in self.history) <= CHAT_CHARS:
            return
        results = [m for m in self.history if m.get("role") == "tool"]
        for message in results[:max(1, len(results) // 2)]:
            body = message.get("content") or ""
            if len(body) > 600:
                message["content"] = self._cut_result(body)

    def _cut_result(self, body):
        quoted = dict.fromkeys(re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', body))
        named = [i for i in quoted if i in self.seen][:20]
        return body[:400] + f"...(older result cut; listings named in it: {', '.join(named)})"

    # ---- the model loop ----
    def turn_cap(self):
        if not self.paced:
            return MAIN_TURNS
        return max(MAIN_TURNS, self.max_steps - TURN_HEADROOM)

    def call_cap(self):
        if not self.paced:
            return TURN_CAP
        return TURN_CAP + self.turn_cap() - MAIN_TURNS

    def time_ok(self):
        return not self.paced or self.clock.left_s() >= CLOCK_FLOOR_S

    def run(self):
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({"kind": "reading", "brief": getattr(self.brief, "raw", self.brief)})
        self.history = [{"role": "system", "content": GUIDE}, {"role": "user", "content": self._intro(self.view.get("opening"))}]
        self.set_call_cap(self.call_cap())
        self.spin(self.turn_cap(), COST_CAP)
        if not self.finished:
            self._second_try(self.decline_reply or self.wrap_up(again=True))
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.llm_calls})
        return self.trace

    def _intro(self, first):
        parts = ["WHAT THE SHOPPER ASKS FOR:\n" + self.request]
        if first is not None:
            self.take(first)
            parts.append("WHAT THE STORE SHOWED ON OPENING:\n" + self.show(first))
        parts.append("Begin with a search for each product inside its category and open a handful of listings in full, so the "
                     "interviewer learns which properties these listings carry.\n\n" + self.status())
        return "\n\n".join(parts)

    def _second_try(self, turned):
        """A close the store turned down gets a short extra loop to act on its reply, then a final close."""
        again = turned and not (self.finished or self.broke or self.no_credit) and self.left() > 1 and self.time_ok()
        if again:
            self.history.append({"role": "user", "content": "[helper] Closing the episode was tried and " + turned
                                 + "\nAct on what the store says, then finish.\n" + self.status()})
            self.spin(self.loop_turns + EXTRA_TURNS, COST_CAP + EXTRA_SPEND)
        if not self.finished:
            self.wrap_up()

    def _open_loop(self, turn_cap, spend_cap):
        return (not self.finished and not self.broke and self.left() > 0 and self.loop_turns < turn_cap
                and self.cost < spend_cap and self.time_ok())

    def _calls_of(self, message):
        """The tool calls of a reply in one plain shape, arguments as compact JSON text (an object, else {})."""
        out = []
        numbered = [(n, raw) for n, raw in enumerate(message.get("tool_calls") or []) if isinstance(raw, dict)]
        for number, raw in numbered:
            spec = raw.get("function") if isinstance(raw.get("function"), dict) else {}
            args = _call_args(spec.get("arguments"))
            out.append({"id": str(raw.get("id") or f"k{self.loop_turns}_{number}"), "type": "function",
                        "function": {"name": str(spec.get("name") or ""), "arguments": dump(args)}})
        return out

    def _ask_first(self, calls):
        binding = {self.jobs.get("cart"), self.jobs.get("buy"), self.jobs.get("refuse")} - {None}
        return (not self.asked and bool(self.jobs.get("ask")) and self.talk_free() > 0 and self.left() > 8
                and any(c["function"]["name"] in binding for c in calls))

    def _interview(self):
        """One interview round; rounds that move neither the step nor the message count are counted, and after
        ASK_STALLS of them in a row asking is marked done so the loop cannot circle without a model turn."""
        before = (self.step, self.talk_used)
        summary = self.ask_round()
        if (self.step, self.talk_used) != before:
            self.ask_stalls = 0
        else:
            self.ask_stalls = self.__dict__.get("ask_stalls", 0) + 1
            if self.ask_stalls >= ASK_STALLS:
                self.asked = True
        return summary

    def spin(self, turn_cap, spend_cap):
        quiet = [0]
        while self._open_loop(turn_cap, spend_cap):
            if self.ready_to_ask():
                self._after_interview(self._interview(), "")
                if STOP_AFTER_QUESTIONS and self.interviewer.done:
                    self._leave_after_questions()
                    return
            elif not self._model_turn(quiet):
                return

    def _leave_after_questions(self):
        """Ends the visit once the questions are done, reporting what the shopper made firm."""
        known = {it.name: list(it.musts) + list(it.told) for it in self.items}
        self.trace.append({"kind": "needs_known", "needs": known})
        tool = self.jobs.get("refuse")
        if tool:
            self.world.call(tool, dict(self._fixed("refuse_args") or {}))

    def _after_interview(self, summary, fallback):
        text = summary or fallback
        if text:
            self.history.append({"role": "user", "content": text + "\n" + self.status()})

    @staticmethod
    def _said(message):
        said = message.get("content")
        if isinstance(said, str):
            return said
        return dump(said) if said else ""

    def _model_turn(self, quiet):
        """One model turn and its calls; False when the loop should stop."""
        self.shrink_history()
        self.loop_turns += 1
        message = self.turn(self.history, [FAST_MODEL, BACKUP_MODEL], 3000, self.tools_now())
        if message is None:
            self.misses += 1
            return not (self.no_credit or self.llm_calls >= self.call_cap() or self.misses > 2)
        calls = self._calls_of(message)
        said = self._said(message)
        if not calls:
            quiet[0] += 1
            self.history += [{"role": "assistant", "content": said or "(no call)"},
                             {"role": "user", "content": "Pick one of the tools and call it to go on.\n" + self.status()}]
            return quiet[0] < 3
        quiet[0] = 0
        self.history.append({"role": "assistant", "content": said or None, "tool_calls": calls})
        if self._ask_first(calls):
            first = "Not run, no step spent: the shopper gets asked first."
            self.history += [{"role": "tool", "tool_call_id": c["id"], "content": first} for c in calls]
            self._after_interview(self._interview(), "The interviewer found nothing to ask.")
            return True
        self.trace.append({"kind": "turn", "text": said[:300], "calls": [c["function"] for c in calls]})
        replies = self.judge_calls(calls)
        tail = "\n" + self.status() if replies else ""
        self.history += [{"role": "tool", "tool_call_id": cid, "content": text + (tail if at == len(replies) - 1 else "")}
                         for at, (cid, text) in enumerate(replies)]
        return True


# ---- the episode: state books, helpers, and the names the helpers read them by ----

_HOMES = {
    "spend": {"cost": "cost", "no_credit": "no_credit", "broke": "broke", "misses": "misses", "waited": "waited",
              "llm_calls": "calls", "loop_turns": "loop_calls"},
    "talk": {"talk_cap": "cap", "talk_used": "used", "talk_kept": "kept", "duty_msgs": "duty_msgs", "duties": "duties",
             "duty_done": "duty_done", "record": "record", "record_steps": "record_steps", "heard": "heard",
             "notes": "notes"},
    "roles": {"jobs": "jobs", "shows_facts": "shows_facts", "id_args": "id_args", "side_ok": "compare_ok",
              "side_bad": "compare_bad"},
    "catalog": {name: name for name in ("seen", "opened_facts", "order_seen", "pages_read_ids", "unavailable", "id_key",
                                        "price_key", "stock_key", "tool_keys", "view_keys")},
    "books": {name: name for name in ("epoch", "basket", "basket_view", "basket_seen", "basket_moved", "basket_touched",
                                      "basket_said", "best_basket", "dropped", "offer_seen", "gone", "sold_lines",
                                      "note_sold", "held", "refused_at", "shortlist", "refuse_blocks", "pushed",
                                      "late_held", "vouched", "queued", "buys", "pointers", "pointed",
                                      "refused_mark", "held_marks")},
    "pages": {"replay_cache": "replay_cache", "recent": "recent", "turned_down": "turned_down"},
}
_ROUTE = {name: (home, field_name) for home, table in _HOMES.items() for name, field_name in table.items()}


class Driver(Steering):
    """One episode: the request, the state books, the model, the store and the clock, and the helpers working on them."""

    def __init__(self, problem, clock=None, model_link=None,
                 store_link=None):
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.view = env["policy_view"]
        self.request = str(self.view.get("query") or "")
        self.tools = [t for t in self.view.get("tools") or [] if self._usable_tool(t)]
        self.schemas = {t["function"]["name"]: self._params_of(t) for t in self.tools}
        self.max_steps = self._step_limit(self.view.get("max_steps"))
        self.trace = []
        self.brief = Brief()
        self.items = []
        self.budget = None
        self.history = []
        self.decline_reply = ""
        self.decline_step = -1
        self.clock = clock or Clock()
        self.spend = Spend()
        self.talk = Talk()
        self.roles = Roles()
        self.books = Books()
        self.pages = Pages()
        self.catalog = Catalog(self.schemas, gone=lambda ident: ident in self.books.gone,
                               epoch=lambda: self.books.epoch, wrapped=WRAPPED_RECORDS)
        self.shape = ResultShape(self.catalog, self.ask_json, self.trace.append)
        self.model = Model(self.spend, self.clock, self.trace, link=model_link)
        self.world = World(self.binding, self.spend, self.clock, self.trace,
                           finished=bool(self.view.get("finished")), link=store_link, after=self._after_store)
        self.desk = Board(self)
        self.legwork = Legwork(self)
        self.interviewer = Interviewer(self, two_waves=EARLY_ASK, skip_known=SKIP_KNOWN, resend_unclear=RESEND_UNCLEAR)
        self.gate = Gate(self)
        self.notes_reader = Notes(self)
        self.review = Review(self, stay=STAY_ON_DECLINE)

    # -- names that live in the books --
    def __getattr__(self, name):
        home = _ROUTE.get(name)
        if home is None:
            raise AttributeError(name)
        return getattr(self.__dict__[home[0]], home[1])

    def __setattr__(self, name, value):
        home = _ROUTE.get(name)
        if home is None:
            object.__setattr__(self, name, value)
        else:
            setattr(self.__dict__[home[0]], home[1], value)

    @staticmethod
    def _usable_tool(tool):
        spec = tool.get("function") if isinstance(tool, dict) else None
        return isinstance(spec, dict) and isinstance(spec.get("name"), str)

    @staticmethod
    def _params_of(tool):
        params = tool["function"].get("parameters")
        return params if isinstance(params, dict) else {"type": "object", "properties": {}}

    @staticmethod
    def _step_limit(raw):
        limit = whole(raw)
        if limit is None and isinstance(raw, float):
            limit = int(raw)
        return limit if limit and limit > 0 else 30

    @property
    def step(self):
        return self.world.step

    @step.setter
    def step(self, value):
        self.world.step = value

    @property
    def finished(self):
        return self.world.finished

    @finished.setter
    def finished(self, value):
        self.world.finished = value

    @property
    def asked(self):
        return self.talk.asked or self.interviewer.done

    @asked.setter
    def asked(self, value):
        self.talk.asked = value

    @property
    def paged_obs(self):
        return self.legwork.last_pages

    def _after_store(self, name, args, seen):
        """Remember the listings whose own page was shown (a search row does not count)."""
        if name not in self.page_tools():
            return
        if isinstance(seen, dict) and seen.get("error"):
            return
        for value in args.values():
            for entry in value if isinstance(value, list) else [value]:
                if isinstance(entry, str):
                    self.catalog.pages_read_ids.add(entry.strip())

    # -- reading the request --
    def read_request(self):
        brief = Intake(self.model, self.tools, self.schemas, self.request, self.trace).read(self.view)
        self.brief = brief
        self.jobs = dict(brief.jobs)
        self.items = brief.items
        self.pointers = brief.pointers
        self.budget = brief.budget
        self.talk_cap = brief.talk_cap
        self.talk_kept = brief.talk_kept
        self.max_steps = brief.steps
        self.duties = brief.duties

    # -- counters --
    def left(self):
        return self.max_steps - self.step

    def talk_free(self):
        return self.talk.free()

    def basket_mark(self):
        return self.books.mark()

    def page_tools(self):
        return self.roles.page_tools()

    def facts_tool(self):
        return self.roles.facts_tool(self.single_target)

    def compare_facts(self):
        return self.roles.compare_facts()

    def compare_bare(self):
        return self.roles.compare_bare(self.single_target)

    def duty_due(self, stage):
        return self.talk.due(stage, f"{self.basket_mark()}#{self.books.epoch}")

    def keep_for_finish(self):
        return self.talk.finish_steps(f"{self.basket_mark()}#{self.books.epoch}")

    def set_call_cap(self, cap):
        self.spend.CALL_CAP = cap

    # -- model and store --
    def turn(self, messages, tiers, max_tokens,
             tools=None):
        return self.model.turn(messages, tiers, max_tokens, tools)

    def ask_json(self, prompt, tiers, max_tokens=1800):
        return self.model.json(prompt, tiers, max_tokens)

    def call(self, name, args):
        return self.world.call(name, args)

    # -- listings --
    def take(self, observation):
        self.catalog.take(observation)

    def ident(self, node, in_list=False, depth=0):
        return self.catalog.id_of(node, in_list, depth)

    id_of = ident

    def records_in(self, observation):
        return self.catalog.records_in(observation)

    def note_view(self, tool, observation):
        self.catalog.note_view(tool, observation)

    def shown_already(self, tool, ident):
        return self.catalog.shown_already(tool, ident)

    def price_of(self, ident):
        return self.catalog.price_of(ident)

    def buyable(self, ident):
        return self.catalog.buyable(ident)

    def sold_out(self, ident):
        return self.catalog.sold_out(ident)

    def category_key(self):
        return self.catalog.category_key()

    def single_target(self, tool):
        return self.catalog.single_target(tool)

    def groups(self):
        return self.catalog.groups(self.items)

    def paging(self, tool, observation):
        return self.shape.paging(tool, observation)

    def price_sorted(self, observation):
        return self.shape.price_sorted(observation)

    # -- the board --
    def needs_now(self):
        return live_needs(self.items)

    def board(self):
        return self.desk.text()

    def status(self):
        return self.desk.status()

    # -- talking and notes --
    def ready_to_ask(self):
        return self.legwork.ready()

    def ask_round(self):
        return self.interviewer.run()

    def duty_check(self, stage):
        return self.gate.pass_(stage)

    def loose_lines(self, observation):
        return self.notes_reader.lines(observation)

    def read_note(self, lines):
        return self.notes_reader.read(lines)

    # -- own steps --
    def observe(self, tool, observation):
        return self.legwork.observe(tool, observation)

    def more_pages(self, tool, args, observation):
        return self.legwork.pages(tool, args, observation)

    def ranked_here(self, results):
        return self.legwork.ranked_here(results)

    def look_wider(self):
        return self.legwork.look_wider()

    def open_facts(self, ids, keep):
        return self.legwork.open_facts(ids, keep)

    fill_facts = open_facts

    def cheaper(self):
        return self.legwork.cheaper()

    # -- reviews --
    def review_order(self, ids=None, cap=COST_EASE):
        return self.review.order(ids, cap)

    def review_verdict(self, ids):
        return self.review.verdict(ids)

    def review_decline(self):
        return self.review.decline()

    def follow_pointers(self):
        return self.review.trail()

    def refresh_lines(self):
        return self.review.refresh()

    def replace_note(self):
        return self.review.swap_note()

    def locate_sold(self):
        return self.review.locate_sold()

    def settle(self):
        return self.review.settle()

    def basket_text(self):
        return self.review.basket_text()

    def basket_ids(self):
        return self.review.basket_ids()

    def proof(self, observation):
        return self.review.proof(observation)

    def record_of(self, ident):
        return self.review.record(ident)

    def offer_tool(self):
        return self.review.offer_tool()


def _crash(exc):
    return {"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}


def agent_main(problem_data):
    episode = None
    try:
        episode = Driver(problem_data, clock=Clock())
        return episode.run()
    except Exception as exc:
        if episode is None:
            return [_crash(exc)]
        episode.trace.append(_crash(exc))
    try:
        if not episode.finished:
            episode.wrap_up()
    except Exception as again:
        episode.trace.append(_crash(again))
    return episode.trace


__all__ = ["agent_main"]
