"""Nimbus: a shopping agent for one shopper inside an ORO store world, built on the public agent Halo (itself built on
the public agent Know Thyself).

A light model drives the store's own tools; code keeps the ledger of listings and requirements, plans the questions to
the shopper, and reviews the cart before an order and the evidence before an ending without a purchase. Kept from the
base: the 300 s clock, list questions with rotation and the late question before a checkout. New here:
- every store reply is read into records whatever its shape: rows written as "key: value | key: value" text, attribute
  blocks written as "k=v; k=v", JSON held inside a string; one listing's own page is shown without the short per-field
  clip, so text near the end of a long field still reaches the model and the reviews;
- LIST_WIDTH sets how many properties one message names: the list style of the base, or one property per
  message naming the product by the store's own category label (by its own name when two items share a label), with
  no repeat of a question the shopper turned back and no message left unsent while a property is left to ask;
- endings follow the store: after a refused ending without a purchase the agent buys the best basket it has instead of
  ending again; an ending without a purchase that no evidence review could check becomes an order of a complete basket;
- codes printed inside listing text are never sent as listing ids; store calls the agent makes on its own keep a step
  reserve for the finishing calls; spend stays inside a per-episode cap.
"""
from __future__ import annotations
import json
import random
import re
import time
_LOADED_AT = time.monotonic()
from typing import Any
from src.agent.proxy_client import ProxyClient
LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'
LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
CACHE_FROM = 4000
HEAVY_ROOM = 0.02
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3
TASK_SECONDS = 300.0
TIME_MARGIN = 24.0
CALL_FLOOR = 6.0
ASK_BY = 165.0
QUESTION_BY = 210.0
WIND_DOWN = 190.0
LAST_TURN = 255.0
EMPTY_ROOM = 12.0
LATE_BY = 190.0
LATE_STEPS = 8
SCARCE_MESSAGES = 2
# How many properties one message to the shopper names. 8 = the base's list style (shoppers that answer lists);
# 1 = one property per message (shoppers that turn lists back). Set once before upload; nothing changes it at run time.
LIST_WIDTH = 1
PAGE_CHARS = 1600
RECORD_CHARS = 700
STORE_DEADLINE = 276.0
END_DEADLINE = 291.0
STORE_WAIT = 30.0
FINISH_SLACK = 2
REVIEW_ROOM = 0.008
# The validator starts its 300 s before this module is loaded, so the episode clock starts at module load when the
# episode follows it closely (a fresh process per episode); otherwise (module loaded long before) at the episode.
LOAD_SLACK = 10.0
CAP_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
CAP_NOT = 'result|row|page|count|size|num|item|step'
SHELF_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MODE = 'field|filter|facet|brows|categor|list|structured'
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
FIELD_HEAD = re.compile('^\\s*([A-Za-z_][\\w .\\-/]{0,38}?):(?:\\s+|$)(.*)$', re.S)
PAIR_CUT = re.compile(';\\s*(?=[^;=]{1,48}=)')
TOKEN_ID = re.compile('(?=[^\\s]*\\d)(?=[^\\s]*[A-Za-z])[A-Za-z0-9][\\w.\\-:]{2,63}')
PLAIN_NUMBER = re.compile('-?(?:0|[1-9]\\d{0,11})(?:\\.\\d{1,6})?')


def _scalar(text: str) -> Any:
    """A field value as written in a text row: a plain number or a literal (true, false, none) becomes that value,
    anything else stays text."""
    plain = text.strip()
    if PLAIN_NUMBER.fullmatch(plain):
        return float(plain) if '.' in plain else int(plain)
    lowered = plain.lower()
    if lowered in ('true', 'false'):
        return lowered == 'true'
    if lowered in ('none', 'null'):
        return None
    return plain


def _pair_block(text: str) -> dict[str, Any] | None:
    """'k=v; k=v' (two or more pairs, nothing else) as a dict, else None."""
    plain = text.strip().rstrip(';').strip()
    if plain.count('=') < 2:
        return None
    block: dict[str, Any] = {}
    for part in PAIR_CUT.split(plain):
        key, sep, value = part.partition('=')
        key = key.strip()
        if not sep or not key or len(key) > 48 or key in block or '|' in key:
            return None
        block[key] = value.strip()
    return block if len(block) >= 2 else None


def _fields_by(text: str, joint: str) -> list[list[str]] | None:
    """The 'key: value' fields of a text cut at `joint`; a piece without its own 'key:' head belongs to the field
    before it. None when the text does not open with a field."""
    fields: list[list[str]] = []
    for part in text.split(joint):
        head = FIELD_HEAD.match(part)
        name = head.group(1).strip() if head else ''
        if head and name and all((name != f[0] for f in fields)):
            fields.append([name, head.group(2)])
        elif fields:
            fields[-1][1] += joint + part
        else:
            return None
    return fields


def _text_record(text: str) -> dict[str, Any] | None:
    """A listing written as text as a dict: 'key: value' fields joined by ' | ', by line breaks or by '; ' (the first
    joint that gives three or more fields), one value shaped like an identifier; 'k=v; k=v' values become dicts."""
    for joint in (' | ', '\n', '; '):
        if text.count(joint) < 2:
            continue
        fields = _fields_by(text.strip(), joint)
        if not fields or len(fields) < 3 or not any((TOKEN_ID.fullmatch(value.strip()) for _, value in fields)):
            continue
        record: dict[str, Any] = {}
        for name, value in fields:
            block = _pair_block(value) if joint != '; ' else None
            record[name] = block if block is not None else _scalar(value)
        return record
    return None


def _decode(node: Any, depth: int=0, field: bool=False) -> Any:
    """A store reply with its listings as records: JSON held in a string is parsed, and a text row standing alone (a
    list element, or the only value of a wrapper) becomes a dict. The text fields of a record stay text."""
    if depth > 8:
        return node
    if isinstance(node, str):
        plain = node.strip()
        if len(plain) >= 2 and plain[0] in '[{' and plain[-1] in ']}':
            try:
                inner = json.loads(plain)
            except ValueError:
                inner = None
            if isinstance(inner, (dict, list)):
                return _decode(inner, depth + 1)
        if field:
            return node
        record = _text_record(plain)
        return record if record is not None else node
    if isinstance(node, list):
        return [_decode(value, depth + 1) for value in node]
    if isinstance(node, dict):
        inside = len(node) >= 3
        return {key: _decode(value, depth + 1, inside) for key, value in node.items()}
    return node
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_short: dict[int, ProxyClient] = {}
_env_short: dict[tuple[int, int], ProxyClient] = {}
GUIDE = 'You are Nimbus, a meticulous buyer acting on behalf of one shopper inside a web shop; you act solely through the shop\'s tools. Grading is binary: the episode is worth 1 only if its finish is exactly right, and 0 otherwise. Exactly right means either an order containing each requested item in the quantity asked for, every item meeting all requirements in force at the moment you check out and the total staying within the budget; or, if the store genuinely has nothing that meets everything, an ending through the no-purchase tool carried out the way the request asks.\n\nTHE STORE AND ITS TOOLS\n- Each tool call uses up one of a limited number of steps, while thinking is free. This store has its own tool names, argument names and result shapes: read the description and schema of every tool, and pass argument names and allowed values exactly as written there.\n- Searching can be limited: a filter by category, a keyword mode that might read only the opening words of the query, pages of just a few rows, a token or offset that reaches the next page. If the answer depends on seeing every row, compare the total the store reports with the rows you have seen, and keep paging, passing back exactly the token or offset the store gave. Short, value-first keyword queries work best (the wanted value, then the product noun). When a filtered search comes back empty, usually either the value given to the filter or the filter parameter chosen is wrong: try again with the label spelled exactly as the result rows show it, or with a different filter parameter.\n- A search row is only a summary. Stores split a listing\'s record in different ways: one view may show its attributes (properties), another nothing but its offer and its description, while a side-by-side view can carry either kind. Once Nimbus has learned which tool shows attributes, the ledger names it. Every requirement on a property is to be judged from those attributes; read them, then, for each listing you might buy, and note that a single call covering several listings saves steps if it shows what you need. Never reopen a listing when nothing has changed since you last read it.\n- A part number or model code printed inside a listing\'s title or description is not its listing id: any call that takes a listing needs the id from the id field of the result rows. Such a code still serves as search words, and for matching one requested item to another when both pages print it.\n- You never write to the shopper yourself: Nimbus\'s interviewer puts every question to the shopper for you. Its answers appear in the conversation and go into the ledger.\n- Besides the purchase, the request may require steps on the way, such as an approval, a message, or something to check or read first; these appear in the ledger. Steps that fall due right before the order or the ending are performed by Nimbus itself once you call the checkout or ending tool: never hold back for them, and never end without a purchase on their account. Use the store\'s tools for the remaining ones (questions about the product go through Nimbus\'s interviewer).\n\nWHEN A LISTING FITS\n- A listing\'s kind of product is settled by its category label. Only a label matching the requested item as a whole counts; a neighbouring label never does.\n- A property requirement is satisfied once the listing\'s attribute for that property gives the value. The key may be spelled any way (spaces, underscores, camel case, prefixes), and a single entry of a list split by commas or slashes will do. Values are compared by their words: differences of case, singular versus plural, and plain synonyms do not matter, and a value that includes all the words of the required value counts ("dark olive" meets olive, "extra wide grip" meets wide). A value in different words fails: an abbreviation or code does not equal the word it stands for, and "olive green" is not "lime green". Convert units before comparing numbers; a limit includes its edge value; a range has to lie entirely within a limit.\n- An attribute decides when its key names the requirement\'s property, whether by that very name or a plain synonym: if its value says something else, the listing fails, whatever the title or other keys say. A key about some other property, even a closely related one (the strap\'s colour is not the case\'s colour), settles nothing about the requirement. After opened listings, Nimbus appends [evidence] lines naming the keys whose names match a requirement.\n- If no attribute of the listing covers that property, the wanted value counts when it is stated under another attribute (say, inside a list of styles or features) or in the title or description. A number of a different kind does not count (a weight cannot stand in for a capacity), and neither does a bare figure whose text leaves unsaid what it measures.\n- A requirement whose value is stated by the category label itself (a season or an audience inside the label) is met by the category.\n- Some requirements point at another listing rather than giving a value ("X equal to that of the listing <shop> sells with <property>", "X within N of what the page of ... shows"). Settle such a requirement before anything else: locate that listing (search by its category, brand, shop or words) and open its full page using the single-item or the side-by-side tool (its search row does not suffice); if that page sends you on to a further listing about the property, track that one down and open it in the same way; the last page supplies the value. An order is valid only when every page on that chain has been opened. If the wanted value is anything other than that listing\'s value, the rule bars only the listings that share that exact value, and it alone is never a reason to end without a purchase.\n- If a listing you are on the point of buying directs you to another page for more about itself, read that page before buying.\n- If one requested item has to fit another (say, an add-on for some appliance), choose the pair together, judging by the words of the listings themselves (a list of what it is compatible with, a model or series code printed on both pages). If none of the accessories meeting their own requirements fits the device you chose, switch to a different device, one meeting its own requirements that some such accessory does fit; do not end without a purchase while that switch has not been tried.\n- Exclusions and closed lists must be respected. Soft wishes ("ideally", "would be nice") only settle a tie between otherwise equal picks.\n- Nothing the store shows as sold out may be bought. The total, price times quantity added up over all lines, must not go above the budget; matching it exactly is fine.\n\nPICKING AMONG FITTING LISTINGS\n- For a cheapest request (lowest total, smallest price), the order has to contain the lowest-priced listing out of ALL fitting listings. Gather the complete fitting set (page through it: result rows often come in no price order, and Nimbus pages through a short result on its own), check the candidates from the lowest price upward, and settle on the earliest one that passes every check fully. If a search accepts a maximum price, make use of it: list the category below what your current best pick costs, so that only cheaper candidates remain to be checked.\n- For a most or least request on some measure, rank every fitting listing by that measure and pick whichever sits at the requested end; a listing that leaves the measure unstated cannot win.\n- With no such pick rule, any listing that satisfies everything will do; steps are not to be spent hunting for a better one.\n- Add each item to the cart the moment you have opened a listing meeting every requirement for it (under a cheapest, most or least pick: once the fitting set has been compared), then move on. A carted line often sells out shortly after it goes in, and carting early leaves steps for replacing it.\n- When several different items are requested, each needs its own cart line at its requested quantity.\n\nWHEN THINGS CHANGE\n- Things can change while you work, from the shopper or from the store: a new budget, an added or dropped requirement, a different price, a listing that sold out. Such changes come inside tool results, frequently arriving as a refusal at the moment you check out; a note counts no matter which tool\'s result carries it, and the ledger shows the requirements in force.\n- A refused checkout does not end the episode: read the refusal, apply every change it states (budget, requirements, a listing that is gone), fix the cart and check out again.\n- Since price and stock can change without notice, read a chosen listing\'s record again shortly before you check out.\n\nCLOSING THE EPISODE\n- Before checking out, make sure you have read the full page of each chosen listing in every single-listing view that can show its attributes, seller notes or references to other listings: such notes or pointers can hold a requirement\'s value or lead to another listing.\n- Make the cart hold exactly the chosen lines (take anything else out), view the cart once, then check out in a call by itself.\n- Ending without a purchase is right only after a genuine search has found no listing that meets everything. Being short of steps proves no such thing; in that case check out the best cart you have. Word that ending the way the request asks, and if a remark is needed, keep it brief and name no listing.\n- If the store refuses an attempt to end without a purchase, apply what the refusal says before anything else: when listings fit after that change, put one suitable listing per requested item into the cart and order; end without a purchase again only if nothing fits once the change is applied.\n- Each of your replies must call at least one tool. Keep the prose short: the facts in hand, followed by the next action.'
BRIEF = 'You turn a shopper\'s request into a working brief that a shopping agent will follow. Study the request and the store\'s tools, then reply with ONE JSON object, compact (no indentation) and with nothing around it, holding these keys:\n\n"items": a list holding one entry for every separate product the shopper wants to buy, each {"name": what sort of product it is, worded as in the request, "shelf": its category label in the store, copied exactly, when some tool lists such labels; failing that, the category the request itself places it in ("a stool in patio furniture" gives "patio furniture"); failing both, "", "quantity": how many units of that one listing (an integer; 1 if the request gives no number; a pair or set sold as a single listing counts as 1), "musts": the product\'s firm requirements as a list of "property: constraint" strings, each carrying the precise value, limit, exclusion or closed list}.\nRules for items and musts:\n- Use the request\'s own term as the property\'s name ("intended for a camping purpose" gives "purpose: camping", "an 8-inch handle" gives "handle length: 8 inch"); never put a neighbouring property in its place.\n- A must whose value has to be read from another listing is ONE must carrying that listing\'s whole description ("strap colour: unlike the strap colour of the watch Tamberlo sells with a steel case"). Whatever identifies that other listing (its shop, its kind, and any trailing "with <property> <value> ..." clause, however many pairs it names) stays inside that single must and is never a must of this product.\n- Musts never hold price, budget, stock or quantity.\n- The request\'s story (the person who will use it, the place it is going to stand, what went wrong earlier, the occasion that prompted it) counts as background rather than a must, except where it spells out a property\'s value or limit directly. Words that belong to what the product is called (say, a season or an audience that is part of its name) never become musts either: they go into "name".\n- A use or occasion that the request states as a property of the product itself ("shoes for office wear", "a shirt made for travel", "a hat, and it is for a graduation", "the occasion has to be formal") DOES count as a must on that property ("occasion: office"), except when the store\'s category label already carries it. Still background: one person\'s own event, the person or the setting the product is bought for (as in "to wear at her graduation" or "a gift for a neighbour"), and how it ought to sit, feel or look while in use.\n- Whatever the request names in a single phrase joined by "and" (a single "I want ..." with a single quantity, frequently a single store department\'s name) is ONE item; create separate items only if the request gives each product a quantity or requirements of its own.\n"budget": the shopper\'s ceiling for the whole spend, as a number in the unit the request uses, or null.\n"budget_unit": the name of that unit written out (such as "cents"), or null.\n"budget_scope": "total" for a limit on the entire order, "each" for a limit per item, null when no limit is given.\n"in_stock_only": true, except when the request openly allows items that are unavailable.\n"pick": "cheapest" if the request is after the lowest price or the least total outlay, "most" or "least" if it wants some measure pushed to its extreme, and "any" otherwise. Saying that money is tight does not set a pick rule.\n"pick_measure": for "most"/"least", the measure to maximise or minimise; "" for the other picks.\n"messages": the integer count of messages the agent may send the shopper, or null when the request does not say.\n"steps": the integer count of steps, moves or tool calls the request permits, or null.\n"lookups": those musts, copied exactly, whose value has to be fetched first from another listing the request refers to; [] when there are none.\n"rules": every instruction in the request about the way to work or the way to finish, each in 12 words at most.\n"wishes": things the shopper would like but does not insist on.\n"ask_first": points the request explicitly wants checked with the shopper before the choice is made ([] when there are none).\n"process": any steps the request demands of the agent beyond locating and buying the right items (for instance obtaining the shopper\'s approval, telling the shopper something or showing it to them, checking something with them, reading something beforehand, or using a specific tool), each as {"step": the action required, 15 words at most, "when": "choose" (before the products get chosen), "order" (immediately before checkout) or "end" (immediately before ending without a purchase), "messages": the number of messages to the shopper it needs (0 if it needs none)}; [] if the request asks for no step of this kind. Not steps: the item requirements, the budget, stock, the caps on messages and steps, and how to finish when nothing fits.\n"quiet_decline": true if, for the case where nothing fits, the request forbids suggesting or naming alternatives; false if it asks for alternatives; null if it is silent on the matter.\n"fits": [{"item": item number, "with": item number}, ...], one pair for each requested item that, per the request, has to work with, fit or suit ANOTHER of the requested items (whichever one the agent picks); [] if there is no such pair.\n"roles": an object that maps each job below to the exact name of the tool doing it, or to null when no tool does: "search", "open" (displays a single listing\'s attributes or properties; where two tools each show one listing, the one whose description says it shows attributes, properties or specifications), "offer" (displays a single listing\'s current offer, e.g. its price and stock, when this is a separate tool from "open"), "compare" (shows several listings in full), "ask" (messages the shopper), "cart" (modifies the cart), "cart_view" (displays the cart), "buy" (completes checkout), "refuse" (ends the episode with no purchase).\n"buy_args": the fixed arguments under which the "buy" tool completes a checkout (if the tool has a mode argument, that mode); {} otherwise.\n"refuse_args": the fixed arguments under which the "refuse" tool closes the episode without a purchase; {} otherwise.\n'
PLAN = 'You pick which questions a shopping agent will put to a shopper. Each shopper keeps a few firm preferences out of the written request, each on one property of one requested product. Only a question that names that exact property brings such a preference out; a property the shopper holds no view on is answered with "don\'t mind", and one the shopper never thinks about at all draws a puzzled reply, so every property you name uses up part of a message.\n\nYour input: the request; the items, each with what the request already settles for it; and a numbered list of the attribute keys found on opened listings in each item\'s category, together with the number of those listings that carry each key and a few example values. A preference left unwritten may concern a property that none of the listed keys names.\n\nOrder of choice:\n0. Anything the request explicitly wants confirmed with the shopper (listed in the ASK FIRST section when there is one), naming each one after the property it concerns.\n1. Hinted properties. Often the request\'s story hints at an unwritten preference while leaving its value out: a sentence about something the shopper is unable to judge or hands over to you, a person whose taste is decisive, how the product looks or the setting it goes in, the one who is going to use or wear it, or what it contains. For every such sentence, name the one property it hints at and the item it concerns, following the story\'s own order.\nUse a listed key of that item for a hinted property whenever one fits the hint (a hint about the wearer: the gender or audience key; a hint about looks: whichever listed colour, pattern or finish key most of its listings carry); a property carried by no listing of the item goes after that item\'s three most widely carried keys.\n2. After that, the listed keys in ranked order: favour what a buyer of such a product decides between first (often its type or style, shape or fit, size, capacity or dimensions, material, colour or pattern, whichever that product is chosen by) and keys that most of the item\'s listings carry; when five or more listings have been read, steer clear of keys showing up on merely one or two of them.\nSkip: any property that the request already fixes for that item (some requirement names it; but a key whose values only echo a required value, e.g. a pattern key that displays the required decoration, is NOT settled); another name for a property you already chose; a key each of whose values lists several purchase options together; identifiers; model names; warranty and similar logistics. Pack sizes, counts and per-pack quantities are ordinary properties, and a key tagged [not the stated ...] is a different property from that requirement, whatever words its name has in common with it. Rank each item separately: its hinted properties first, its listed keys after.\n"label" is the property\'s name in plain words: for a listed key, every word of the key in plain spelling ("trim colour" for trimColour, "cuff width" for cuff_width; never cut down to "colour" or "width"); just the name of the property, with no example values, no brackets and no second property.\nChoose at most COUNT per item, best first. Send back ONE JSON object, with no text outside it: {"ask": [{"item": item number, "label": "...", "hint": "the words of the story it springs from, 8 at most, or an empty string"}]}, with each item\'s picks in rank order'
HEAR = 'Some questions went from a shopping agent to a shopper, and the answers are below. Go through every answer, then send back ONE JSON object, with no text outside it:\n{"needs": [{"need": "property: constraint", "item": item number, "points": true when that value has to be taken off another listing, otherwise false}], "drop": [{"property": "a requirement the shopper is now giving up", "item": item number}], "free": [{"property": "...", "item": item number}], "crowded": [the numbers of those questions whose answer complains that the message bundled too many topics together, or asks for them to be put one at a time; not answers that merely say they did not understand it or want plainer words], "budget": a new limit for the whole order as a number, or null}\n- "needs": each firm preference an answer expresses, carrying its precise value or bound and unit ("at least 40 cm", "square", "not plastic"). Name the need after the property the answer itself names, even when that is not the property asked about; several answers about a single property add up to one need. Copy the value verbatim, exactly as phrased in the answer, with its nouns and its units kept ("at least 4 drawers", never just "at least 4"). The value covers only what a listing could itself state: starting at the wanted value or at its limit word and running to its end, minus the framing the answer wraps around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer may set the value by referring to another listing (a value the same as, unlike or near what some shop\'s product has, with that product identified through a handful of its properties): record this as ONE need, on the property named by the answer, with "points": true. For its value, copy the answer\'s own words exactly, starting at the comparison words and running to the end of that description, including whatever it says about a further listing that page may name; never cut it short or rephrase it. Properties describing that other product do not count as needs of this item.\n- "free": for every question to which the shopper replied that they do not mind, the property it asked about.\n- Homely or indirect phrasing that names a value still counts as stating it. Phrasing that refuses to haggle over a value (it will not be swapped, surrendered or altered) states that value as wanted; put "not X" only when the answer turns down X itself.'
NOTE = 'The note below turned up in a store reply while a shopping agent was working; its source may be the shopper or the shop. Send back ONE JSON object, with no text outside it:\n{"kind": "change" if the note alters the task (the budget, requirements, a price, what is available, or a refusal of the order), "error" if all it reports is that a call was badly formed or rejected, "other" in every remaining case, "budget": the new limit for the whole order, as a number in the unit of the request, or null, "add": [{"need": "property: constraint", "item": item number}], "drop": [{"need": "the requirement that no longer applies", "item": item number}], "gone": [ids of listings it calls unavailable], "sold_out": true if it says some chosen listing has gone, has sold out or is unavailable now, even when it names none, "price_moved": true when it says a price changed, "summary": "one brief sentence"}\nTreat a replaced requirement as a drop of the old one together with an add of its replacement. Report each change the note makes.'
CHECK = 'Right before a shopping agent checks out, you are the final review. From the records shown, decide whether this cart is exactly right.\nRules for reading the records: a requirement holds when the listing\'s attribute for its property gives the value (the key may be spelled any way; one entry of a list is enough; differences of case or of singular and plural, and plain synonyms, are fine; a value holding all the words of the required value passes, as "dark olive" passes olive; convert units; an edge value is within its limit; a range has to sit wholly inside). If an attribute\'s key names the requirement\'s own property, that attribute decides, and the listing fails when it says anything else. A key about some other property, even a closely related one (the strap\'s colour is not the case\'s colour), decides nothing. Where the listing lacks an attribute for the property, the wanted value written under a different attribute, in the title or in the description passes; a number of a different kind does not (a weight cannot stand in for a capacity), and neither does a lone figure whose text leaves unsaid what it measures. When the category label itself carries a requirement\'s value, the category satisfies it, and the category label has to match the requested item as a whole. Any rule asking to differ from another listing\'s value is met unless the records show that very value.\nEach requested item has to occupy a separate cart line holding the requested quantity; the cart may hold nothing else, no line may appear as sold out, and the sum over all lines of price times quantity has to fit the budget. When a listing has several records, the newest one counts.\nWhen the pick rule asks for the cheapest, the most or the least, look through the other listings shown as well: the cart is wrong if one of them clearly meets every requirement and beats the cart\'s choice under that rule.\nSend back ONE JSON object, with no text outside it: {"ok": true or false, "problems": ["one brief, specific problem", ...], "better": [ids of listings that ought to take a cart line\'s place, best first]}. Set ok to false solely when the records clearly show a problem, or when no record speaks to some requirement.'
READING = 'whenever an attribute\'s key names the requirement\'s own property, that attribute decides (spell the key any way, a single list entry suffices, synonyms and converted units are fine, a value holding all the words of the required value passes, as "dark olive" passes olive, an edge value is within its limit); a key about some other property, even a closely related one (the strap\'s colour is not the case\'s colour), decides nothing; if no attribute covers that property, the wanted value appearing under a different attribute, in the title or in the description passes (a lone figure that leaves unsaid what it measures passes nothing); a category label that itself carries a requirement\'s value satisfies it; the item and the category label have to match; a rule asking to differ from another listing\'s value is met unless the records show that very value; a sold-out listing or one beyond the budget is not a fit.'
AUDIT = 'Nothing seems to fit, so a shopping agent plans to end the episode without buying. Measure the listing records below against the requirements, applying these reading rules: ' + READING + '\nA record marked ROW is just a search-result row: nobody has read its page (the attributes), so where a property is missing from the row, the record can neither pass nor fail on it.\nSend back ONE JSON object, with no text outside it: {"fits": {"item number": [ids of listings apparently satisfying every requirement, cheapest first, 4 at most]}, "open": [ids of the ROW listings that are of a requested kind, within the budget, and whose page could prove a fit, best prospects first, 6 at most], "reason": "one sentence"}'
UNMIX = 'Firm requirements on a product were drawn out of a shopper\'s request. A few of them take their value from a different listing that the request points to, and the words identifying that listing (its kind, its shop, the properties given after it with "with ...") are sometimes wrongly taken for requirements on the product that is being bought. Going by how the request is worded, tell which of the OTHER REQUIREMENTS below merely describe that other listing. Send back ONE JSON object, with no text outside it: {"describe_other": [those requirements, copied exactly; empty if every one is a genuine requirement on the product]}'
TRAIL = 'Checkout is coming up for a shopping agent. Certain requirements do not state their value but borrow it from another listing, and that listing\'s page may pass the reader on to yet another listing for the same property. The order is valid only once every listing on each such chain has been opened in full by the agent. Starting from the description inside each requirement, follow its chain one link at a time through the words on the pages themselves, using the records below (OPENED: a page the agent has opened; ROW: seen just as a row in search results, its page unread).\nA record counts as a link only when it fits that link\'s entire description: shop or brand, product kind, AND every property value given in the description (e.g. "a 6 piece gift set" excludes that shop\'s listing that has 6 pieces but is no gift set). If no record fits all of it, the link has not been found yet, even if that shop also has other listings of the kind; for such a link, "find" carries a few search terms: the shop\'s name together with the description\'s values.\nList each chain\'s links in sequence, stopping at the first link that has not been found or whose page is not OPENED, because what it says is still unknown.\nSend back ONE JSON object, with no text outside it: {"chains": [{"need": an exact copy of the requirement, "links": [{"id": the listing id if one of the records below is this link, else "", "about": the way the requirement or the preceding page describes this listing, "find": "search words for a link not found yet (its shop and the values its description gives)"}], "done": true when the final link is an OPENED page which spells out the property\'s value itself (true as well when the requirement wants a value different from or unlike that one, since the requirement is judged against that page\'s value), otherwise false, "value": the property\'s value as written on that last page, or ""}]}'
GATE = 'The next move of a shopping agent is to STAGE. Along the way, the shopper\'s request requires the steps listed below. Using the numbered RECORD of the agent\'s actions so far, judge for every step due now whether it has been carried out for the cart as it now stands. Treat a step as done only if some RECORD line shows it happening; for a step that concerns the cart or the order, that line must also come after the cart\'s last change, and the reply in it must grant the step. Asking about a product\'s properties carries out no step.\nIf a step is still outstanding, name the single store call that performs it next: a tool from TOOLS plus its arguments. Messages to the shopper go out through the messaging tool as one short sentence or question in plain words, about that step and nothing else, mentioning no product property or value the step does not require. Give no call when all due steps are done, or when the remaining messages or steps cannot cover the missing one.\nSend back ONE JSON object, with no text outside it: {"steps": [{"step": the step as listed, "done": true or false, "lines": [numbers of the RECORD lines that show this step done]}], "call": {"tool": "...", "args": {...}} or null, "hold": "" or what a reply wants changed before the agent continues, "needs": [{"need": "property: constraint", "item": item number}, one for every firm requirement stated in such a reply]}'
SWAP = "A shopping agent's cart holds a line whose listing has just sold out. Look among the other listing records shown for listings that could replace that line, applying these reading rules: " + READING + '\nSend back ONE JSON object, with no text outside it: {"fits": [{"id": the listing\'s id, "price": the price the record shows, as a number}, one entry for each listing that seems to meet every requirement of the item on the sold-out line, 6 at most], "reason": "one sentence"}'
SHAPE = 'Below is what one store tool returned. Point out its parts, each by the exact key used in the result or in the tool\'s parameters. Send back ONE JSON object, with no text outside it: {"rows": key of the list that holds the listings, "count": key of the total number of matching listings (all matches, not just the rows shown), or null, "next": key whose value requests the page after this one, or null, "next_arg": name of the tool argument through which that value is passed, or null, "price": key inside a row for the listing\'s price, or null, "stock": row key that says if the listing is purchasable right now, or null}'
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5} ?\\d{3,6}'
PART_CODE = '[A-Z]{1,4}\\d{3,6}'

def _codes(text: Any) -> list[str]:
    """Model codes printed in a text, inner spaces folded, in lower case."""
    found: list[str] = []
    pattern = '\\b' + FIT_CODE + '\\b'
    for hit in re.finditer(pattern, str(text or '')):
        found.append(' '.join(hit.group(0).split()).lower())
    return found

def _fit_marks(text: Any) -> dict[str, Any]:
    """What a listing's own words say about codes: the part code it carries, the model codes it is also sold as, the
    codes it works with, other parts' fit lists it repeats, and a part code it refers the reader to."""
    body = str(text or '')
    marks: dict[str, Any] = {'own': '', 'codes': [], 'works': [], 'also': [], 'more': ''}
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', body, re.I)
    if own:
        marks['own'] = own.group(1).upper()
    aliases = ('\\bmodel\\s*(?:no\\.?|number|#)\\s*[:.]?\\s*(' + FIT_CODE + ')\\b', '\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)')
    for pattern in aliases:
        for hit in re.finditer(pattern, body, re.I):
            marks['codes'].extend(_codes(hit.group(1)))
    for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', body, re.I):
        start = hit.start()
        if re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', body[max(0, start - 40):start], re.I) is None:
            marks['works'].extend(_codes(hit.group(1)))
    for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', body, re.I):
        part, listed = hit.group(1, 2)
        marks['also'].append((part.upper(), _codes(listed)))
    pointer = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', body, re.I)
    if pointer:
        marks['more'] = pointer.group(1).upper()
    return marks

def _dump(value: Any) -> str:
    """Compact JSON text of a value, or its plain text when JSON cannot hold it."""
    try:
        text = json.dumps(value, separators=(',', ':'), ensure_ascii=False)
    except (ValueError, TypeError):
        text = str(value)
    return text

def _clip(value: Any, limit: int, depth: int=0) -> Any:
    """A copy of nested data with every text cut to `limit` characters and anything nested too deep elided."""
    if 7 < depth:
        return '...'
    deeper = depth + 1
    if isinstance(value, list):
        return [_clip(entry, limit, deeper) for entry in value]
    if isinstance(value, dict):
        return dict(((key, _clip(entry, limit, deeper)) for key, entry in value.items()))
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + '...'
    return value

def _json_obj(text: Any) -> dict[str, Any] | None:
    """The JSON object a model reply holds: the whole reply, the inside of a code fence, or the span from the first
    opening brace to the last closing one."""
    if isinstance(text, str):
        body = text.strip()
        fenced = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', body, re.S)
        return _json_obj_parse(fenced.group(1) if fenced else body)
    return None

def _words(key: str) -> str:
    """A key or label as lower-case words: camel case split, runs of separators turned into single spaces."""
    parts = re.split('[\\s_.\\-]+', re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key)))
    return ' '.join((part for part in parts if part)).lower()

def _snake(key: str) -> str:
    return '_'.join(_words(key).split(' '))

def _stem(text: str) -> frozenset[str]:
    """The words of a text with a plural -s dropped from words of four or more letters."""
    bare: set[str] = set()
    for word in _words(text).split():
        if word.endswith('s') and len(word) >= 4:
            word = word[:-1]
        bare.add(word)
    return frozenset(bare)

def _roots(text: str) -> set[str]:
    """Plural-folded words of three or more letters, without the few joining words."""
    kept: set[str] = set()
    for word in _words(text).split():
        if re.search('(?:ch|sh|ss|x|z)es$', word):
            word = word[:-2]
        elif word.endswith('s') and (not word.endswith('ss')) and len(word) > 3:
            word = word[:-1]
        if len(word) < 3 or word in ('and', 'for', 'the', 'with'):
            continue
        kept.add(word)
    return kept

def _spelled(text: str) -> frozenset[str]:
    """A name's words with a plural -s and a British -our ending folded, so that "colours" and "color" are equal."""
    folded: set[str] = set()
    for word in _words(text).split():
        if len(word) > 3 and word.endswith('s') and (not word.endswith('ss')):
            word = word[:-1]
        folded.add(re.sub('our$', 'or', word))
    return frozenset(folded)

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if prop and key:
        if prop <= key:
            return True
        return len(key) > 1 and key <= prop
    return False

def _kinds(schema: dict[str, Any]) -> list[str]:
    declared = schema.get('type')
    if not isinstance(declared, str):
        return list(filter(lambda entry: isinstance(entry, str), declared or []))
    return [declared]

def _fits_kind(value: Any, kind: str) -> bool:
    """Whether a value is of a JSON schema type (unknown type names accept anything)."""
    if kind == 'null':
        return value is None
    if kind == 'boolean':
        return isinstance(value, bool)
    if kind in ('integer', 'number'):
        wanted = int if kind == 'integer' else (int, float)
        return isinstance(value, wanted) and (not isinstance(value, bool))
    for name, python_type in (('string', str), ('array', list), ('object', dict)):
        if kind == name:
            return isinstance(value, python_type)
    return True

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Arguments fitted to a tool's JSON schema: plain type mix-ups and option case fixed, over-long text cut; what
    cannot be fixed is reported in `faults`."""
    if isinstance(schema, dict):
        kinds = _kinds(schema)
        mismatch = bool(kinds) and (not _shape_fits(value, kinds))
        if mismatch:
            value = _shape_coerce(value, kinds)
            if not _shape_fits(value, kinds):
                faults.append(f"{where} should be {'/'.join(kinds)}")
                return value
        value = _shape_enum(value, schema.get('enum'), where, faults)
        if isinstance(value, str):
            return _shape_text(value, schema, where, faults)
        if isinstance(value, dict):
            return _shape_object(value, schema, where, faults)
        if isinstance(value, list):
            return _shape_list(value, schema, where, faults)
        if not isinstance(value, bool) and isinstance(value, (int, float)):
            return _shape_number(value, schema, where, faults)
    return value

def _items(value: Any) -> list[Any]:
    """A list as it is, a non-empty text or object as a one-entry list, anything else as an empty list."""
    if not isinstance(value, list):
        single = isinstance(value, (str, dict)) and bool(value)
        return [value] if single else []
    return value

def _count(value: Any) -> int | None:
    """A whole number given as an int, a whole float, or digits (with an optional .0) in text; else None."""
    if isinstance(value, str):
        return int(float(value)) if re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value) else None
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, int) and (not isinstance(value, bool)):
        return value
    return None

def _id_like(value: Any, named: bool=False) -> bool:
    """A text shaped like a listing id: 3 to 64 characters without inner spaces; one under a key not named as an id
    also needs a digit."""
    if isinstance(value, str):
        core = value.strip()
        if 3 <= len(core) <= 64 and ' ' not in core:
            return named or re.search('\\d', value) is not None
    return False

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    """The key whose values look like distinct listing ids in every row: a key named as an id first, then the one
    with the longest values; None when no key qualifies."""
    chosen, chosen_rank = (None, None)
    for key in rows[0]:
        named = 'id' in str(key).lower()
        ids = _learn_id_values(rows, key, named)
        if ids is None or (len(rows) > 1 and len(set(ids)) < len(ids)):
            continue
        rank = (named, sum(map(len, ids)) / len(ids))
        if chosen_rank is None or rank > chosen_rank:
            chosen, chosen_rank = (key, rank)
    return chosen if chosen_rank is not None else None

def _in_rows(node: Any, ident: str, depth: int=0) -> bool:
    """Whether a row (a dict inside a list, at most six levels down) carries the id as one of its text values."""
    if depth > 5:
        return False
    if isinstance(node, dict):
        children = list(node.values())
    elif isinstance(node, list):
        for entry in node:
            if isinstance(entry, dict) and any((isinstance(x, str) and str(x).strip() == ident for x in entry.values())):
                return True
        children = node
    else:
        return False
    return any((_in_rows(child, ident, depth + 1) for child in children))

def _llm_within(seconds: float) -> ProxyClient:
    """The model client whose timeout fits the time left: the usual one, or one with a timeout in whole 5 s steps."""
    if seconds >= 50:
        return _llm
    limit = max(5, int(seconds) // 5 * 5)
    if limit not in _llm_short:
        _llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return _llm_short[limit]

def _fact_block(node: Any) -> dict[str, Any] | None:
    """The largest nested dict of plain values (two or more) inside a record: its attribute block."""
    if not isinstance(node, dict):
        return None
    found: dict[str, Any] | None = None
    for block in node.values():
        if not isinstance(block, dict) or len(block) < 2:
            continue
        plain: dict[str, Any] = {}
        for key, entry in block.items():
            if not isinstance(entry, bool) and isinstance(entry, (str, int, float)):
                plain[key] = entry
        if len(plain) > 1 and (found is None or len(plain) > len(found)):
            found = plain
    return found

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

def _json_obj_parse(body):
    """The dict a text parses to, whole or from its first opening to its last closing brace; else None."""
    try:
        parsed = json.loads(body)
    except ValueError:
        start, end = (body.find('{'), body.rfind('}'))
        if start < 0 or start >= end:
            return None
        try:
            parsed = json.loads(body[start:end + 1])
        except ValueError:
            return None
    return parsed if isinstance(parsed, dict) else None

def _shape_fits(value, kinds):
    return any((_fits_kind(value, kind) for kind in kinds))

def _shape_coerce(raw, kinds):
    """The value turned into the first listed type it plainly stands for (a whole float as an int, digits as a number,
    a number as text, 'true'/'false' as a flag, a single value as a list); unchanged when none applies."""
    for kind in kinds:
        if isinstance(raw, float) and raw.is_integer() and kind == 'integer':
            return int(raw)
        if isinstance(raw, str) and kind in ('integer', 'number') and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', raw):
            parsed = float(raw)
            return parsed if kind != 'integer' and (not parsed.is_integer()) else int(parsed)
        if kind == 'string' and (not isinstance(raw, bool)) and isinstance(raw, (int, float)):
            return str(raw)
        if kind == 'boolean' and isinstance(raw, str):
            flag = raw.strip().lower()
            if flag in ('true', 'false'):
                return flag == 'true'
        if kind == 'array' and raw is not None and (not isinstance(raw, list)):
            return [raw]
    return raw

def _shape_option(value, allowed):
    """A text value replaced by the allowed option it equals when case and outer spaces are ignored."""
    if isinstance(value, str):
        wanted = value.strip().lower()
        for option in allowed:
            if isinstance(option, str) and option.lower() == wanted:
                return option
    return value

def _shape_text(text, schema, where, faults):
    longest = schema.get('maxLength')
    if isinstance(longest, int) and len(text) > longest:
        text = text[:longest]
    shortest = schema.get('minLength')
    if isinstance(shortest, int) and len(text) < shortest:
        faults.append(f'{where} is too short')
    return text

def _shape_number(number, schema, where, faults):
    numeric = (int, float)
    floor, ceiling = (schema.get('minimum'), schema.get('maximum'))
    if isinstance(floor, numeric) and floor > number:
        faults.append(f'{where} is below {floor}')
    if isinstance(ceiling, numeric) and ceiling < number:
        faults.append(f'{where} is above {ceiling}')
    return number

def _shape_list(entries, schema, where, faults):
    most = schema.get('maxItems')
    if isinstance(most, int) and len(entries) > most:
        faults.append(f'{where} allows at most {most} entries')
    least = schema.get('minItems')
    if isinstance(least, int) and len(entries) < least:
        faults.append(f'{where} needs at least {least} entries')
    inner = schema.get('items')
    if not isinstance(inner, dict):
        return entries
    return [_shape(entry, inner, f'{where}[{index}]', faults) for index, entry in enumerate(entries)]

def _shape_object(fields, schema, where, faults):
    known = schema.get('properties')
    if not isinstance(known, dict):
        known = {}
    if schema.get('additionalProperties') is False:
        faults.extend((f'{where}.{name} is not an allowed argument' for name in fields if name not in known))
    faults.extend((f'{where}.{name} is required' for name in schema.get('required') or [] if name not in fields))
    shaped: dict[str, Any] = {}
    for key, entry in fields.items():
        shaped[key] = entry if key not in known else _shape(entry, known[key], f'{where}.{key}', faults)
    return shaped

def _shape_enum(value, options, where, faults):
    """A value checked against a list of allowed options (a case-only mismatch of text is fixed)."""
    if not isinstance(options, list) or value in options:
        return value
    value = _shape_option(value, options)
    if value not in options:
        faults.append(f'{where}: allowed values are {options}')
    return value

def _learn_id_values(rows, key, named):
    """Every row's value under `key`, stripped, when each one looks like a listing id; else None."""
    ids: list[str] = []
    for row in rows:
        entry = row.get(key)
        if not _id_like(entry, named):
            return None
        ids.append(entry.strip())
    return ids

class Nimbus:

    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.tools = [t for t in self.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tools}
        steps = _count(self.view.get('max_steps'))
        if steps is None and isinstance(self.view.get('max_steps'), float):
            steps = int(self.view['max_steps'])
        self.max_steps = steps if steps and steps > 0 else 30
        now = time.monotonic()
        self.started = _LOADED_AT if 0.0 <= now - _LOADED_AT <= LOAD_SLACK else now
        self.step = 0
        self.finished = bool(self.view.get('finished'))
        self.broke = False
        self.no_credit = False
        self.waited = 0
        self.misses = 0
        self.cost = 0.0
        self.calls = 0
        self.loop_calls = 0
        self.trace: list[dict[str, Any]] = []
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
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.seen_order: list[str] = []
        self.heard: set[str] = set()
        self.notes: list[str] = []
        self.version = 0
        self.cart_touched = False
        self.cart_at = -1
        self.cart_seen_at = -1
        self.cart_view: Any = None
        self.note_named = False
        self.cart_lines: dict[str, int] = {}
        self.checked: set[str] = set()
        self.audited = False
        self.review_ran = False
        self.reviews = 0
        self.decline_refused = False
        self.refused_end_at = -1
        self.recheck_done = False
        self.review_pending = False
        self.decline_cleared = False
        self.decline_holds = 0
        self.code_warned: set[str] = set()
        self.offer_seen: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.held = 0
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ''
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
        self.link_tried: set[str] = set()
        self.lane_misses: dict[int, int] = {}
        self.adopted: dict[str, int] = {}
        self.capped: set[tuple[str, float]] = set()
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.queued: list[str] = []
        self.buys = 0
        self.refusals = 0
        self.recent: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.opened: set[str] = set()
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.trails = 0
        self.trail_blocks = 0
        self.compare_ok = 0
        self.compare_bad = 0
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
        self.plan_order: list[int] = []
        self.lanes: dict[int, list[str]] = {}
        self.lane_found: dict[int, int] = {}
        self.lane_spent: dict[int, int] = {}
        self.lane_used: dict[int, set[str]] = {}
        self.late_asked = False
        self.one_product: set[int] = set()
        self.follow_up: list[tuple[int, list[str], bool]] = []
        self.listing = LIST_WIDTH > 1
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}

    def left(self) -> int:
        return self.max_steps - self.step

    def clock(self) -> float:
        return time.monotonic() - self.started

    def model_room(self) -> float:
        return TASK_SECONDS - TIME_MARGIN - self.clock()

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def facts_tool(self) -> str | None:
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get('open')
        return tool if self.facts_from.get(tool or '') is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        return self.facts_from.get(self.roles.get('compare') or '')

    def compare_bare(self) -> bool:
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))))

    def gate_due(self, stage: str) -> bool:
        return any((p['when'] == stage for p in self.process)) and (stage, self.cart_mark() + f'#{self.version}') not in self.gate_ok

    def finish_reserve(self) -> int:
        """Steps a store call made on the agent's own initiative must leave: one cart call per item, a look at the cart,
        the order itself and a message the order may need."""
        return self.tail() + len(self.items) + FINISH_SLACK

    def tail(self) -> int:
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS:
                return None
            if self.model_room() < CALL_FLOOR:
                self.trace.append({'kind': 'late_call_skipped', 'at': round(self.clock(), 1)})
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                if self.model_room() - WAITS[tries] * 1.2 < CALL_FLOOR:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get('content')
                if isinstance(content, str) and content:
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        result = _llm_within(self.model_room()).post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.cost += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return (message, 200)
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = _count(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800, fixed: int=0) -> dict[str, Any] | None:
        """One JSON answer from the first model that gives one. `fixed` marks a leading part of the prompt that is the
        same in every episode of a run (instructions and the store's tools): for the models that cache prompts it is
        sent as its own cached block, so concurrent episodes read it from the cache instead of paying for it again."""
        for model in models:
            content: Any = prompt
            if fixed >= CACHE_FROM and model.startswith('anthropic/'):
                content = [{'type': 'text', 'text': prompt[:fixed], 'cache_control': {'type': 'ephemeral'}}, {'type': 'text', 'text': prompt[fixed:]}]
            message = self.think([{'role': 'user', 'content': content}], [model], max_tokens)
            got = _json_obj((message or {}).get('content'))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def store_client(self, deadline: float) -> ProxyClient:
        """A store client whose timeout ends each attempt by `deadline` (at most STORE_WAIT seconds). The client's
        max_retries counts attempts: two while a minute or more is left, else one, so a slow or hanging store cannot
        push the episode past the validator's limit (act() adds its own bounded retries)."""
        room = deadline - self.clock()
        key = (int(max(2.0, min(STORE_WAIT, room))), 2 if room > 2 * STORE_WAIT + 2 else 1)
        if key not in _env_short:
            _env_short[key] = ProxyClient(timeout=key[0], max_retries=key[1], retry_delay=1)
        return _env_short[key]

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """One store action with step = last + 1. A 409 re-syncs the step (or marks the episode over), a 402 stops all
        store calls, 429 and server errors are retried after a short pause while time allows; no call starts within
        two seconds of its deadline (STORE_DEADLINE, or the later END_DEADLINE for the order or the ending)."""
        payload = dict(self.binding)
        payload['action'] = {'name': name, 'args': args}
        deadline = END_DEADLINE if name in (self.roles.get('buy'), self.roles.get('refuse')) else STORE_DEADLINE
        for attempt in range(4):
            if self.broke or deadline - self.clock() < 2.0:
                return None
            payload['step'] = self.step + 1
            answer = self.store_client(deadline).post_verbose('/environment/call', payload)
            if isinstance(answer.data, dict):
                return self.book_reply(name, args, dict(answer.data))
            failure = answer.error or {}
            code = failure.get('status')
            said = str(failure.get('body') or '')
            self.trace.append({'kind': 'env_fail', 'status': code, 'body': said[:200]})
            if code == 409:
                low = said.lower()
                if 'finish' in low or 'quarantin' in low or 'finaliz' in low:
                    self.finished = True
                    return None
                expected = re.search('expected (\\d+)', said)
                if expected:
                    self.step = int(expected.group(1)) - 1
                elif 'reuse' in low or 'different action' in low:
                    self.step += 1
                else:
                    return None
                continue
            if code in (400, 401, 402, 403, 404, 422):
                self.broke = self.broke or code == 402
                return None
            pause = (1.0 + attempt) * (0.7 + 0.6 * random.random())
            if code in (429, 502, 503, 504) or code is None:
                time.sleep(max(0.0, min(pause, deadline - self.clock() - 2.5)))
        return None

    def book_reply(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Book a store reply: the observation read into records, the step and the end flag taken over, the listings a
        page view showed marked as opened."""
        if 'observation' in data:
            data['observation'] = _decode(data['observation'])
        self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
        if data.get('finished'):
            self.finished = True
        seen = data.get('observation')
        if name in self.page_tools() and not (isinstance(seen, dict) and seen.get('error')):
            for value in args.values():
                for one in value if isinstance(value, list) else [value]:
                    if isinstance(one, str):
                        self.opened.add(one.strip())
        return data

    def read_request(self) -> None:
        """Read the request and the tools into the brief: roles, items (merged when two share a label and agree),
        fit pairs, lookups, budget, limits and process steps."""
        brief = self._read_request_brief(self._read_request_prompt())
        self.brief = brief
        listed = brief.get('roles')
        self.roles = {}
        if isinstance(listed, dict):
            self.roles = {job: tool for job, tool in listed.items() if isinstance(tool, str) and tool in self.schemas}
        brief.update({key: {} for key in ('buy_args', 'refuse_args') if not isinstance(brief.get(key), dict)})
        self._read_request_modes(brief)
        became = self._read_request_items(brief)
        self._read_request_links(brief, became)
        self.unmix()
        self._read_request_limits(brief)

    @staticmethod
    def shared_shelf(brief: dict[str, Any]) -> bool:
        """Whether two requested items carry the same store label."""
        seen: set[str] = set()
        for entry in _items(brief.get('items')):
            if not isinstance(entry, dict):
                continue
            label = _words(str(entry.get('shelf') or ''))
            if not label:
                continue
            if label in seen:
                return True
            seen.add(label)
        return False

    def unmix(self) -> None:
        """Requirements the brief split off a pointing requirement (words that only describe the other listing) are
        folded back into it, as the model judges from the request."""
        for number, item in enumerate(self.items, 1):
            pointers = [need for owner, need in self.refs if owner == number and need in item.musts]
            if not pointers:
                continue
            rest = [must for must in item.musts if must not in pointers]
            if not rest:
                continue
            moved = self._unmix_moved(item, pointers, rest)
            if moved:
                self._unmix_fold(number, item, pointers[0], moved)

    def described(self, number: int, need: str) -> bool:
        """Whether a requirement is the shopper's own description rather than a property value: four or more words
        whose property no attribute key of three or more opened listings of the item names."""
        pointed = {m for _, m in self.refs} | {str(x) for x in _items(self.brief.get('lookups'))}
        if need in pointed:
            return False
        prop, value = need.partition(':')[::2]
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4 or re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
            return False
        item = self.items[number - 1]
        value_stems = _stem(value)
        for other in self.items:
            if other is not item and any((len(w) >= 4 for w in _stem(other.name) & value_stems)):
                return False
        blocks = self._described_blocks(item)
        if len(blocks) < 3:
            return False
        prop_stems = _stem(prop)
        return all((_stem(key).isdisjoint(prop_stems) for key in {_words(k) for block in blocks for k in block}))

    def ledger(self) -> str:
        many = len(self.items) > 1
        lines = [self._ledger_item(number, item, many) for number, item in enumerate(self.items, 1)]
        lines += self._ledger_terms()
        source = self._ledger_source()
        if source:
            lines.append(source)
        return '\n'.join(('- ' + line for line in lines))

    def status(self) -> str:
        left = self.left()
        head = f'[Nimbus] step {self.step} of {self.max_steps} used, {left} left.' + self._status_pressure(left)
        if left > self.tail() + 1 and self.clock() >= WIND_DOWN:
            head += ' TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns; an episode that runs out of time earns nothing.'
        blocks = [head]
        pending = self._status_unresolved() if self.buys == 0 else []
        if pending:
            blocks.append('[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(pending[:3]))
        if self.leads:
            blocks.append('[review candidates] ' + ', '.join(self.leads))
        if self.bound and self.buys == 0:
            plan = self.fit_plan()
            if plan:
                blocks.append(plan)
        if self.notes:
            blocks.append('[changes so far] ' + ' | '.join(self.notes[-4:]))
        blocks.append('[ledger]\n' + self.ledger())
        return '\n'.join(blocks)

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        """The listing id a record carries: under the id key learned from result rows, else under a key whose values
        look like ids, for a record that stands in a list, at the top of a reply, or holds a nested block."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        known = node.get(self.id_key) if self.id_key else None
        if self.id_key and _id_like(known, True):
            return known.strip()
        standalone = in_list or depth == 0 or any((isinstance(v, dict) for v in node.values()))
        if not standalone:
            return None
        key = _learn_id([node])
        if not key:
            return None
        value = node[key].strip()
        return value if 'id' in str(key).lower() or value in self.rows else None

    def learn_rows(self, node: list[Any]) -> bool:
        """Whether a list holds listing rows (mostly records of three or more fields carrying an id); learns the id key
        from them when none is known yet or the known one does not fit them."""
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or 2 * len(rows) < len(node):
            return False
        if self.id_key and all((_id_like(r.get(self.id_key), True) for r in rows)):
            return True
        key = _learn_id(rows)
        if key:
            self.id_key = key
        return bool(key)

    def absorb(self, observation: Any) -> None:
        """Book every listing record a reply holds into the ledger: its row, and its full record when it carries an
        attribute block; a stock flag turning false (or back) marks it unavailable (or available again)."""
        self._absorb_node(observation, 0, False)

    def _absorb_node(self, node: Any, depth: int, in_list: bool) -> None:
        if depth > 6:
            return
        if isinstance(node, list):
            rows = self.learn_rows(node)
            for value in node:
                self._absorb_node(value, depth + 1, rows)
            return
        if not isinstance(node, dict):
            return
        ident = self.ident(node, in_list, depth)
        if not ident:
            for value in node.values():
                self._absorb_node(value, depth + 1, False)
            return
        self._absorb_record(ident, node)

    def _absorb_record(self, ident: str, node: dict[str, Any]) -> None:
        if ident not in self.rows:
            self.seen_order.append(ident)
            self.rows[ident] = {}
        held = self.rows[ident]
        if any((held.get(k) is True and v is False for k, v in node.items())):
            self.unavailable.add(ident)
        elif any((held.get(k) is False and v is True for k, v in node.items())):
            self.unavailable.discard(ident)
        held.update(node)
        if _fact_block(node):
            self.full[ident] = dict(held)
        elif ident in self.full:
            self.full[ident].update(node)

    def labels(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            for option in schema.get('enum') or []:
                if isinstance(option, str):
                    found.add(option.strip().lower())
            for value in schema.values():
                if isinstance(value, dict):
                    walk(value, depth + 1)
        for schema in self.schemas.values():
            walk(schema, 0)
        return found

    def category_key(self) -> str | None:
        """The row key that names each listing's category: a short text in at least 80% of the rows, preferring
        values the tools list as options, then few distinct values, wide presence and short values."""
        texts: dict[str, set[str]] = {}
        hits: dict[str, int] = {}
        pairs = [(key, value) for row in self.rows.values() for key, value in row.items() if self._category_key_short(value)]
        for key, value in pairs:
            if key in texts:
                texts[key].add(value)
                hits[key] += 1
            else:
                texts[key] = {value}
                hits[key] = 1
        floor = 0.8 * max(1, len(self.rows))
        options = self.labels()
        chosen, rank = (None, None)
        for key, values in texts.items():
            if hits[key] < floor or len(values) > 40:
                continue
            size = len(values)
            listed = sum((v.strip().lower() in options for v in values)) / size
            mean = sum(map(len, values)) / size
            candidate = (-listed, size, -hits[key], -mean)
            if rank is None or candidate < rank:
                chosen, rank = (key, candidate)
        return chosen if rank is not None else None

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        """Per category label of the requested items: the attribute keys of opened listings (look-alike keys joined),
        each with how many listings carry it and up to four example values."""
        wanted = self._key_table_wanted()
        groups = self._key_table_groups()
        owned = [label for label in groups if self._key_table_matches(label, wanted)]
        if owned:
            groups = {label: groups[label] for label in owned}
        table: list[tuple[str, str, int, list[str]]] = []
        for label in sorted(groups, key=lambda label: groups[label]['n'], reverse=True):
            table.extend((self._key_table_row(label, name, slot) for name, slot in self._key_table_keys(groups[label])[:14]))
        return table

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        """A mark for a key that shares a word with a requirement's property yet is another property: its values never
        state the required value's words."""
        number = self.owner_of(label)
        if not number:
            return ''
        shown = None
        for prop, _, value in (must.partition(':') for must in self.items[number - 1].musts):
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I) is not None:
                continue
            wanted = self._not_stated_words(value)
            if shown is None:
                shown = self._not_stated_words(' '.join(values))
            if prop.strip() and wanted and (not wanted <= shown) and _words(prop) != name and _stem(prop) & _stem(name):
                return f' [not the stated {prop.strip()}: another property]'
        return ''

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        """Whether two keys that share a word are one property: their values agree on the listings both carry (on
        every one when neither key's words hold the other's), or share a value when no listing carries both."""
        mine = set(name.split())
        theirs = set(other.split())
        if mine.isdisjoint(theirs):
            return False

        def close(first: str, second: str) -> bool:
            if first == second:
                return True
            words = [re.findall('[a-z0-9]+', text.lower()) for text in (first, second)]
            words.sort(key=len)
            fewer, more = words
            if not fewer:
                return False
            return len(more) <= 2 * len(fewer) + 2 and set(fewer).issubset(more)
        left, right = (spec['per'], slot['per'])
        shared = [ident for ident in left if ident in right and left[ident].strip() and right[ident].strip()]
        agreeing = [close(left[ident], right[ident]) for ident in shared].count(True)
        if not (mine.issubset(theirs) or theirs.issubset(mine)):
            return len(shared) > 1 and agreeing == len(shared)
        if not shared:
            common = set(spec['values']).intersection(slot['values'])
            common.discard('')
            return bool(common)
        return agreeing * 2 > len(shared)

    def item_groups(self) -> list[list[str]]:
        """Seen listing ids grouped by category label, the requested items' labels only when any match, largest
        groups first, one group per item."""
        cat_key = self.category_key()
        labels = [str(self.rows[ident].get(cat_key) or '') if cat_key else '' for ident in self.seen_order]
        by_label: dict[str, list[str]] = {}
        for label, ident in zip(labels, self.seen_order):
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(ident)
        wanted = self._key_table_wanted()
        pool = [ids for label, ids in by_label.items() if self._key_table_matches(label, wanted)]
        if not pool:
            pool = list(by_label.values())
        pool.sort(key=len, reverse=True)
        return pool[:max(1, len(self.items))]

    def single_target(self, tool: str | None) -> str | None:
        """The one text argument a single-listing tool takes, when it is the tool's only required argument."""
        schema = self.schemas.get(tool or '') or {}
        required = schema.get('required') or []
        texts = self._search_template_texts(schema.get('properties') or {})[0]
        named = [k for k in required if k in texts]
        target = named[0] if named else None
        if not target and len(texts) == 1:
            target = texts[0]
        if not (tool and target):
            return None
        return target if all((k == target for k in required)) else None

    def price_of(self, ident: str) -> float | None:
        if not self.price_key:
            return None
        raw = (self.rows.get(ident) or {}).get(self.price_key)
        if isinstance(raw, dict):
            raw = next(filter(self._price_of_number, raw.values()), None)
        elif isinstance(raw, str):
            raw = float(raw) if re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', raw) else None
        return float(raw) if self._price_of_number(raw) else None

    def buyable(self, ident: str) -> bool:
        if not self.sold_out(ident):
            stock = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
            return stock is not False and (not (self._price_of_number(stock) and stock <= 0))
        return False

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """The listing records a reply shows at its top level: the reply itself, the records of a list it holds, a
        record wrapped as one of its values, or the records of lists one level further down."""
        if isinstance(observation, list):
            return self._records_of_list(observation)
        if not isinstance(observation, dict):
            return []
        own = self.ident(observation)
        if own:
            return [(own, observation)]
        found: list[tuple[str, dict[str, Any]]] = []
        for value in observation.values():
            if isinstance(value, list):
                found.extend(self._records_of_list(value))
                continue
            if not isinstance(value, dict):
                continue
            wrapped = self.ident(value, True)
            if wrapped:
                found.append((wrapped, value))
            else:
                for inner in value.values():
                    if isinstance(inner, list):
                        found.extend(self._records_of_list(inner))
        return found

    def _records_of_list(self, values: list[Any]) -> list[tuple[str, dict[str, Any]]]:
        pairs = []
        for row in values:
            if isinstance(row, dict):
                key = self.ident(row, True)
                if key:
                    pairs.append((key, row))
        return pairs

    def note_view(self, name: str, observation: Any) -> None:
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and (keys <= have) and (ident not in self.gone)

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """How a tool's result is laid out (row list, total, next-page key and argument, price and stock keys), asked of
        the model once per tool and kept."""
        known = self.shapes.get(name, False)
        if known is not False:
            return known
        lists = [key for key, value in observation.items() if self._read_shape_table(value)] if isinstance(observation, dict) else []
        if not lists:
            return None
        self.shapes[name] = None
        schema = self.schemas.get(name) or {}
        answer = self.ask_json(self._read_shape_prompt(name, schema, observation, lists), [LIGHT, HEAVY], 2500) or {}
        rows = answer.get('rows')
        if not rows or rows not in lists:
            return None
        first = observation[rows][0]
        params = schema.get('properties') or {}
        shape: dict[str, Any] = {'rows': rows}
        for field, where in (('count', observation), ('next', observation), ('next_arg', params), ('price', first), ('stock', first)):
            key = answer.get(field)
            shape[field] = key if isinstance(key, str) and key in where else None
        total = observation[shape['count']] if shape['count'] else None
        if shape['count'] and (isinstance(total, bool) or not isinstance(total, int)):
            shape['count'] = None
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        self.price_key = self.price_key if self.price_key else shape['price']
        self.stock_key = self.stock_key if self.stock_key else shape['stock']
        self.shapes[name] = shape
        return self.shapes[name]

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        """(listings matched in all, rows shown, next-page argument, its value) for a result that can be paged."""
        shape = self.read_shape(name, observation)
        if not isinstance(observation, dict) or not shape:
            return None
        if not (shape['count'] and shape['next'] and shape['next_arg']):
            return None
        shown = observation.get(shape['rows'])
        total = observation.get(shape['count'])
        token = observation.get(shape['next'])
        if not isinstance(shown, list) or not shown:
            return None
        whole = isinstance(total, int) and (not isinstance(total, bool))
        usable = token is None or (isinstance(token, (str, int)) and (not isinstance(token, bool)))
        return (total, len(shown), shape['next_arg'], token) if whole and usable else None

    def price_sorted(self, observation: Any) -> bool:
        """Whether a result's rows (three or more, all priced) come cheapest first."""
        prices = [self.price_of(ident) for ident, _ in self.records_in(observation)]
        if len(prices) < 3 or None in prices:
            return False
        return all((prices[k] <= prices[k + 1] for k in range(len(prices) - 1)))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """With a cheapest, most or least pick, read the next pages of a short search result (at most two, inside the
        step reserve), or say how to page on."""
        self.paged_obs = []
        pick = self.brief.get('pick')
        if pick not in ('cheapest', 'most', 'least'):
            return ''
        plan = self.paging(name, observation)
        if not plan:
            return ''
        count, shown, param, value = plan
        if value is None or count <= shown or self._more_pages_later(args.get(param)):
            return ''
        if pick == 'cheapest' and self.price_sorted(observation):
            return ''
        pages = self._more_pages_count((count - 1) // shown, self.left() - self.tail())
        if not pages:
            return f"\n[Nimbus] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
        read, value = self._more_pages_read(name, args, param, value, pages)
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(read)})
        if not read:
            return ''
        text = '\n[Nimbus read the next pages of this search itself, one step a page]\n' + '\n'.join(read)
        if value is not None:
            text += f"\n[Nimbus] {count} listings matched and Nimbus has read {shown * (len(read) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest."
        return text

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A note when a search ignored the maximum price it was given and returned dearer rows."""
        if not self.price_key:
            return ''
        props, caps = self._cap_ignored_caps(name, args)
        if not caps:
            return ''
        cap = args[caps[0]]
        dearer = len([pair for pair in self.records_in(observation) if (self.price_of(pair[0]) or 0) > cap])
        if dearer == 0:
            return ''
        return f'\n[Nimbus] This search did not apply {caps[0]}={cap:g}: {dearer} of its rows cost more' + self._cap_ignored_hint(props, args, caps[0]) + '.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if not self.price_key or self.brief.get('pick') != 'cheapest':
            return ''
        priced: dict[str, float] = {}
        for seen in observations:
            for ident, _ in self.records_in(seen):
                price = self.price_of(ident)
                if price is not None and self.buyable(ident):
                    priced[ident] = price
        if len(priced) < 2:
            return ''
        order = sorted(((price, ident) for ident, price in priced.items()))
        return '\n[Nimbus] buyable rows here, cheapest first: ' + ', '.join((f'{ident} ({price:g})' for price, ident in order[:6]))

    def wider_hold(self) -> str:
        """Hold a cheapest-pick checkout once while steps allow and ask for searches under other product names."""
        if self.wider_held:
            return ''
        if self.brief.get('pick') != 'cheapest' or self.left() < 2 * len(self.items) + 14:
            return ''
        self.wider_held = True
        priced = []
        for ident in self.cart_ids():
            price = self.price_of(ident)
            if price is not None:
                priced.append(f'{ident} at {price:g}')
        listed = f" ({'; '.join(priced)})" if priced else ''
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + listed + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Open listings whose attributes were never read, one step each, while more than `keep` steps are left."""
        route = self._fill_facts_route()
        if route is None:
            return []
        texts: list[str] = []
        for ident in self._fill_facts_queue(ids):
            if self.finished or self.left() <= keep or self.facts_from.get(route[0]) is False:
                break
            page = self._fill_facts_open(*route, ident)
            if page is None:
                break
            texts.append(f'{ident}: ' + page[:1500])
        if texts:
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def show_links(self, ids: list[str], keep: int) -> None:
        """Run the part code a cart line sends the reader to through the search, once per code, and open the page it
        finds when steps allow; the result feeds the ledger, not the model. Multi-line carts only."""
        if len(self.items) < 2 and len(ids) < 2:
            return
        plan = self.value_template()
        if plan:
            for ident in ids:
                code = self._show_links_code(ident)
                if code and not self._show_links_follow(plan, ident, code, keep):
                    return

    def cap_template(self) -> tuple[str, dict[str, Any], str, str] | None:
        """The listing tool that takes a shelf and a maximum price, with the mode it needs: the mode an earlier listing by
        shelf used, else the one option naming a listing. None when no tool or mode is clear."""
        for tool, schema in self.schemas.items():
            props = (schema or {}).get('properties') or {}
            caps, shelves = self._cap_template_fields(props)
            if len(caps) == 1 and len(shelves) == 1:
                base = self._cap_template_modes(props, shelves[0])
                return None if base is None else (tool, base, shelves[0], caps[0])
        return None

    def list_below(self, cart: list[str]) -> None:
        plan = self.cap_template()
        cat_key = self.category_key()
        if not plan or not cat_key:
            return
        tool, base, shelf, cap = plan
        for line in cart:
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            if price is None or price < 2 or not label or (line, price) in self.capped:
                continue
            if self.left() <= max(self.tail() + 5, self.finish_reserve() + 3) or self.finished:
                return
            self.capped.add((line, price))
            faults: list[str] = []
            args = _shape({**base, shelf: label, cap: int(price) - 1 if float(price).is_integer() else int(price)}, self.schemas.get(tool), 'args', faults)
            if faults:
                return
            reply = self.act(tool, args)
            if reply is None:
                return
            self.observe(tool, reply.get('observation'))
            self.trace.append({'kind': 'list_below', 'line': line, 'price': price})
            page = self.paging(tool, reply.get('observation'))
            if page and page[3] is not None and page[1] < page[0] <= 2 * page[1] and self.left() > self.tail() + 5 and not self.price_sorted(reply.get('observation')):
                call = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                more = None if faults else self.act(tool, call)
                if more is not None:
                    self.observe(tool, more.get('observation'))

    def cheaper_rows(self) -> list[str]:
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.cart_mark() + f'#{self.version}'
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        self.list_below(cart)
        picks: list[str] = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:min(6, CHEAPER_OPENS * max(1, len(cart)))], max(self.tail() + 4, self.finish_reserve() + 2))

    def observe(self, name: str, observation: Any) -> str:
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ''
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def widen(self) -> None:
        """Before the interview, read the attributes of more listings of each requested kind: one side-by-side call when
        that tool shows attributes, else single opens."""
        tool = self.roles.get('compare')
        target = self.compare_target()
        if tool and target and (not self.compare_bare()):
            if self.left() > 12:
                self._widen_batch(tool, target)
            return
        self.widen_singly()

    def widen_singly(self) -> None:
        """Open unread listings one by one, round robin over the item groups: rows sharing more requirement words first
        (cheaper first on a cheapest pick), a first wave of four per group, then more while steps allow."""
        if self._fill_facts_route() is None:
            return
        values = ' '.join((need.split(':', 1)[-1].lower() for _, need in self.needs_now()))
        wanted = set(re.findall('[a-z0-9]{3,}', values))
        each = 4 if len(self.items) > 2 else WIDEN_EACH
        lanes: list[tuple[list[str], int]] = []
        for ids in self.item_groups():
            opened = len([ident for ident in ids if ident in self.full])
            lanes.append((self._widen_singly_lane(ids, wanted)[:max(0, each - opened)], max(0, 4 - opened)))
        first: list[str] = []
        later: list[str] = []
        for rank in range(max(4, each)):
            for lane, head in lanes:
                if rank >= len(lane):
                    continue
                if rank < head:
                    first.append(lane[rank])
                elif rank < each:
                    later.append(lane[rank])
        cut = max(5, 2 * len(lanes) + 2)
        opened = len(self.fill_facts(first[:min(cut, 8)], 22))
        rest = first[cut:] + later
        self.fill_facts(rest[:max(0, 10 - opened)], 30)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        if self.left() <= 8 or self.clock() > ASK_BY:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get('required') or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get('enum'), list) and spec['enum']:
                    args[key] = spec['enum'][0]
                else:
                    return None
        faults: list[str] = []
        args = _shape(args, schema, 'args', faults)
        return None if faults else args

    def interview(self) -> str:
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.widen()
        count = self.msg_limit - self.msg_used
        self.lanes = self.plan_lanes(max(count, LIST_WIDTH) if self.listing else count + (2 if LIST_WIDTH <= 1 else 0))
        self.lane_found = {n: 0 for n in self.lanes}
        self.lane_spent = {n: 0 for n in self.lanes}
        self.lane_used = {n: set() for n in self.lanes}
        self.follow_up = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.clock() < QUESTION_BY):
            batch = self.ask_next(told)
            if not batch:
                break
            sent += batch
        found_items = [n for n in self.lane_found if self.lane_found[n] > 0]
        searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "Nimbus's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if searched:
            summary += '\n' + searched
        if self.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def late_question(self) -> str:
        """Spend a message the order does not need on one more planned question just before a checkout.

        The interview keeps back every message the request reserves for a process step, at the order or at an ending
        without a purchase; once the agent is about to buy, the ones reserved for that ending are free, and with a tight
        message limit that question can be the one that names a hidden need. Returns the summary for the loop model
        when the answer adds a requirement, else ''."""
        if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit:
            return ''
        if any((p['when'] == 'order' and p['messages'] > 0 for p in self.process)):
            return ''
        if self.left() < LATE_STEPS or self.clock() > LATE_BY or not self.judges():
            return ''
        self.late_asked = True
        if not self.lanes:
            self.lanes = self.plan_lanes(LIST_WIDTH if self.listing else 1)
            self.lane_found = {n: 0 for n in self.lanes}
            self.lane_spent = {n: 0 for n in self.lanes}
            self.lane_used = {n: set() for n in self.lanes}
        told: list[str] = []
        budget = self.budget
        batch = self.ask_next(told, LATE_STEPS - 2)
        if not batch:
            self.trace.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
            return ''
        if self.budget != budget:
            told.append(f'budget now {self.budget}')
        number, question, answer = batch[0]
        self.trace.append({'kind': 'late_ask', 'q': question, 'a': answer[:200], 'told': told})
        if not told:
            return ''
        self.version += 1
        summary = f'Nimbus asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
        searched = self.value_search([number])
        return summary + ('\n' + searched if searched else '')

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        """The latest words search to copy: its tool, the fixed arguments to keep (modes and required ones that name no
        category) and its query field. None before a search with words."""
        tool = self.roles.get('search')
        if not tool:
            return None
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        texts, queries = self._search_template_texts(props)
        for args in reversed(self.searched):
            fields = self._search_template_fields(args, queries or texts, bool(queries))
            if not fields:
                continue
            field = fields[0]
            return (tool, self._search_template_fixed(args, props, schema, field), field)
        return None

    def value_words(self, number: int) -> list[tuple[str, bool]]:
        """Search words for each requirement value of the item a words search can match, each with whether an answer
        gave it: the value itself, or for an answer's lower numeric limit with a unit its number and
        unit, since such a hidden limit often sits on the value of a listing that fits."""
        item = self.items[number - 1]
        pointed = {m for _, m in self.refs}
        candidates = [need for owner, need in self.needs_now() if owner == number and ':' in need and need not in pointed]
        found: list[tuple[str, bool]] = []
        for need in candidates:
            value = need.partition(':')[2].strip()
            from_answer = need in item.told
            floor = re.search(FLOOR, value, re.I)
            if floor is not None:
                if from_answer:
                    found.append((floor.group(1).strip(), True))
            elif value and (not re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I)):
                found.append((value, from_answer))
        return found

    def value_template(self) -> tuple[str, dict[str, Any], str] | None:
        """The search a value search copies: search_template(), or, when every search so far listed a category without
        words, the tool's one query field with its one mode argument set to read it (the option naming words or a query,
        else the other option of a required two-option mode). None when the schema leaves a doubt."""
        tool = self.roles.get('search')
        plan = self.search_template()
        if plan is not None:
            return plan
        if not tool or not self.searched:
            return None
        schema = self.schemas.get(tool) or {}
        required = schema.get('required') or []
        props = schema.get('properties') or {}
        last = self.searched[-1]
        queries = self._search_template_texts(props)[1]
        if len(queries) != 1:
            return None
        choices = self._cap_template_options(props)
        modes = [pair for key, options in choices for pair in self._value_template_mode(key, options, last.get(key), key in required)]
        if len(modes) > 1 or (choices and (not modes)):
            return None
        fixed = {key: last[key] for key in required if key in last and key != queries[0]}
        return (tool, {**fixed, **dict(modes)}, queries[0])

    def value_search(self, numbers: list[int]) -> str:
        """Search the values the shopper's answers gave, per item, and open the first fitting-kind rows."""
        plan = self.value_template()
        if self.left() < 22 or not plan:
            return ''
        found = []
        left = 8
        for number in numbers:
            tries = self._value_search_tries(number)
            if left > 2 and tries:
                left = self._value_search_item(number, plan, tries, left, found)
        if not found:
            return ''
        clipped = [line[:200] for line in found]
        self.trace.append({'kind': 'value_search', 'seen': clipped})
        return "[Nimbus searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(found)

    def home_label(self, number: int) -> str:
        """The category label (in _words form) of the item's own listings: the request's shelf name when a listing
        seen carries it, else the label most of the item's opened listings (or, with none opened, its search rows)
        carry; '' when none is known."""
        cat_key = self.category_key()
        if not cat_key:
            return ''
        item = self.items[number - 1]
        preferred = [_words(name) for name in (item.shelf, item.name)]
        for pool in (self.full, self.rows):
            tally = self._home_label_tally(item, pool, cat_key)
            hit = next((label for label in preferred if label in tally), None)
            if hit is not None:
                return hit
            if tally:
                return max(tally, key=tally.get)
        return ''

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        scarce = self.msg_limit <= SCARCE_MESSAGES and (not self.listing)
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number] and (not self.told_about(number, label))]
            if left:
                return (number, left[:LIST_WIDTH if self.listing else 1])
        return (0, [])

    def told_about(self, number: int, label: str) -> bool:
        """Whether an answer already gave a need on this property of the item: the label's words are held by the told
        property's name ("colour" after "shade colour") or hold a told name of two or more words; colour and color,
        singular and plural alike. A one-word told name ("type") hides no longer label ("bulb type")."""
        mine = _spelled(label)
        return any((_names(mine, _spelled(t.split(':', 1)[0])) for t in self.items[number - 1].told))

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) >= 2:
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.item_said(number)}?"
        return self.question(number, labels[0], self.samples(number, labels[0]) if shown else None)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """The longest leading part of `labels` whose question the message tool's schema takes whole (the schema shaper
        cuts a text past a length cap, so a list too long drops labels from the end instead), with that question;
        ([], '') when not even one label fits."""
        for size in range(len(labels), 0, -1):
            text = self.ask_text(number, labels[:size], shown)
            args = self.message_args(text)
            if args and text in args.values():
                return (labels[:size], text)
        return ([], '')

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Send the next question (a pending follow-up first, else the next planned labels), read its answer into the
        items' needs (appending each new one to `told`) and settle the labels it named. Returns the sent batch, or []
        when nothing was sent."""
        for _ in range(max(LIST_WIDTH, 8)):
            self.follow_up = [f for f in self.follow_up if not self.told_about(f[0], f[1][0])]
            if self.follow_up:
                number, labels, shown = self.follow_up.pop(0)
            else:
                number, labels = self.next_ask(self.lanes, self.lane_found, self.lane_spent, self.lane_used)
                shown = False
            if not labels:
                return []
            fitted, text = self.fit_ask(number, labels, shown)
            if not fitted and shown:
                fitted, text = self.fit_ask(number, labels, False)
            if fitted:
                break
            self.lane_used.setdefault(number, set()).add(_words(labels[0]))
            self.trace.append({'kind': 'ask_too_long', 'item': number, 'label': labels[0]})
        else:
            return []
        labels = fitted
        batch = self.send([(number, labels[0], text)], floor)
        if not batch:
            return []
        self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
        had = [len(item.told) for item in self.items]
        heard = self.hear(batch, (LIGHT, HEAVY))
        if len(labels) > 1:
            heard.pop('free', None)
        self.take_heard(heard, told)
        new = [need for item, size in zip(self.items, had) for need in item.told[size:]]
        for n, (item, size) in enumerate(zip(self.items, had), 1):
            if len(item.told) > size:
                self.lane_found[n] = self.lane_found.get(n, 0) + 1
        crowded = any((_count(x) == 1 for x in _items(heard.get('crowded'))))
        self.settle(number, labels, shown, new, crowded)
        return batch

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        """Mark what an answer showed about the labels its question named.

        One label: it is asked; a reply asking for a narrower question gets the base's follow-up (one product of an
        item name joining several, the property's head word, or example values). A list: the shopper answers about
        the first property of a message it tracks, and its reply to a message naming none of its properties is the
        same as to a property it does not mind. So a disclosed need settles its label and every label before it, and
        the labels after it lead the item's next list; a reply that discloses nothing settles only the first label
        (a need there would have been answered) and the others come next in reverse order, so whatever the shopper
        does not mind no longer stands in front of them; after a second such reply in a row the labels not yet named lead, since the
        order would otherwise return to the first list's. A reply asking for one question at a time ends lists for
        the episode and the list's first label is asked alone."""
        used = self.lane_used.setdefault(number, set())
        if len(labels) == 1:
            used.add(_words(labels[0]))
            if new or not crowded:
                return
            if LIST_WIDTH <= 1:
                self.trace.append({'kind': 'turned_back', 'item': number, 'label': labels[0]})
                return
            if number not in self.one_product and self.name_parts(number):
                self.one_product.add(number)
                self.trace.append({'kind': 'one_product', 'item': number, 'said': self.item_said(number)})
                self.follow_up.insert(0, (number, labels, shown))
                return
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _words(head) not in used:
                self.follow_up.insert(0, (number, [head], False))
            elif not shown and len(self.samples(number, labels[0])) >= 2:
                self.follow_up.insert(0, (number, labels, True))
            return
        if crowded and not new:
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': number})
            self.follow_up.insert(0, (number, labels[:1], False))
            return
        hit = self.answered_label(labels, new) if new else -1
        ahead = labels[hit + 1:] if hit >= 0 else labels[:0:-1]
        used.update((_words(label) for label in labels[:max(hit, 0) + 1]))
        rest = [label for label in self.lanes.get(number) or [] if label not in labels]
        self.lane_misses[number] = 0 if hit >= 0 else self.lane_misses.get(number, 0) + 1
        self.lanes[number] = rest + ahead if self.lane_misses[number] and self.lane_misses[number] % 2 == 0 else ahead + rest
        self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        """Index of the earliest label a list's reply answered. Each new need maps to the label sharing most words with
        its property (colour and color alike), else to the label whose values on opened listings share most words with
        its value; -1 when no new need maps to a label (the reply is then settled like one that disclosed nothing)."""
        seen: dict[str, set[str]] = {}
        for _, key, _, values in self.key_table():
            seen.setdefault(_words(key), set()).update((w for v in values for w in _roots(v)))
        hits = []
        for need in new:
            prop, _, value = need.partition(':')
            scores = [len(_spelled(label) & _spelled(prop)) for label in labels]
            if max(scores, default=0) == 0:
                scores = [len(seen.get(_words(label), set()) & _roots(value)) for label in labels]
            if max(scores, default=0) > 0:
                hits.append(scores.index(max(scores)))
        return min(hits) if hits else -1

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.items[number - 1]
        settled = [_stem(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        if len(self.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)]

    def adopt_labels(self) -> None:
        cat_key = self.category_key()
        if len(self.items) < 2 or not cat_key:
            return
        labels = {str(record.get(cat_key) or '') for record in self.full.values()} - {''}
        strays = [label for label in labels if not self.owner_of(label)]
        orphans = [n for n in range(1, len(self.items) + 1) if not any((self.owner_of(label) == n and label not in self.adopted for label in labels))]
        if strays and len(orphans) == 1:
            self.adopted.update({label: orphans[0] for label in strays})
            self.trace.append({'kind': 'adopted', 'item': orphans[0], 'labels': strays})

    def label_scores(self, label: str) -> list[float]:
        """How well each requested item matches a category label: 3 for its shelf, 2 for its name, else the share of
        words the label has in common with them."""
        words = _words(label)
        long = _roots(label)
        scores = []
        for item in self.items:
            if words == _words(item.shelf or ''):
                scores.append(3.0)
            elif words == _words(item.name):
                scores.append(2.0)
            else:
                mine = _roots(item.shelf or '') | _roots(item.name)
                scores.append(len(long & mine) / max(1, len(long | mine)))
        return scores

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        if label in self.adopted:
            return self.adopted[label]
        best, top = (0, 0.0)
        for number, score in enumerate(self.label_scores(label), 1):
            if score > top:
                best, top = (number, score)
        return best

    def label_shared(self, label: str) -> bool:
        """Whether two or more requested items match a category label equally well (two items on one shelf)."""
        if len(self.items) < 2 or not label:
            return False
        scores = self.label_scores(label)
        top = max(scores)
        return top > 0 and scores.count(top) >= 2

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1)

    def records_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        self.adopt_labels()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing'))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + (' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in _items(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get('item') or 1) or 1), len(self.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            if all((_words(label) != _words(other) for other in lanes[number])):
                lanes[number].append(label)
            if str(pick.get('hint') or '').strip():
                hinted[number].add(_words(label))
                if number not in self.plan_order:
                    self.plan_order.append(number)
        for number in lanes:
            keys = self.lane_keys(number, table)
            read = max(1, self.records_of_item(number))
            own: dict[str, float] = {}
            for label, key, carried, _ in table:
                if self.owner_of(label) == number:
                    own[key] = max(own.get(key, 0.0), carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                if _words(label) in own:
                    return own[_words(label)]
                near = [s for k, s in own.items() if _names(_stem(label), _stem(k)) or _names(_stem(k), _stem(label))]
                return max(near) if near else 0.0
            hinted[number] = {_words(spelled(h)) for h in hinted[number]}
            lane = []
            for pick in lanes[number]:
                if all((_words(spelled(pick)) != _words(other) for other in lane)):
                    lane.append(spelled(pick))
            for key in [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and (not re.search(CATCH_ALL, k))][:GUARD_KEYS]:
                if any((_words(key) == _words(other) for other in lane[:3])):
                    continue
                weak = next((i for i, other in enumerate(lane[:3]) if _words(other) not in hinted[number] and share_of(other) + GUARD_GAP <= own.get(key, 0.0)), None)
                if weak is None and len(lane) >= 3:
                    continue
                placed = [other for other in lane if _words(other) != _words(key)]
                if weak is not None and len(self.items) > 1:
                    moved = placed[weak]
                    placed[weak] = key
                    placed.insert(min(3, len(placed)), moved)
                else:
                    placed.insert(min(len(placed), weak if weak is not None else len(placed)), key)
                if any((_words(x) in hinted[number] and all((_words(x) != _words(y) for y in placed[:3])) for x in lane[:3])):
                    continue
                lane = placed
            lanes[number] = lane
            for key in keys:
                if all((_words(key) != _words(other) for other in lanes[number])):
                    lanes[number].append(key)
            lanes[number] = [x for x in lanes[number] if not re.search(CATCH_ALL, _words(x))] + [x for x in lanes[number] if re.search(CATCH_ALL, _words(x))]
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def name_parts(self, number: int) -> list[str]:
        """The products an item name joins with "and" or commas when each is a plural noun, else []; a single product
        named with "and" gives []."""
        # Two products: "kettles and teapots". One product: "salt and pepper grinder", "pen and pencil cases", "his and
        # hers towels" (invented examples; none of them occurs in any recorded world).
        name = self.items[number - 1].name
        if not re.search('\\band\\b', name, re.I):
            return []
        parts = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I) if p.strip()]
        return parts if len(parts) >= 2 and all((re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts)) else []

    def shelf_name(self, number: int) -> str:
        """The category label the item's own listings carry, spelled as the store writes it (lower case): the request's
        shelf when a listing carries it, else the label most of the item's opened listings (or rows) carry; '' when no
        listing of the item has been seen."""
        cat_key = self.category_key()
        if not cat_key:
            return ''
        item = self.items[number - 1]
        for pool in (self.full, self.rows):
            counts: dict[str, int] = {}
            spelled: dict[str, str] = {}
            for record in pool.values():
                label = record.get(cat_key)
                if isinstance(label, str) and label.strip() and self.owns(item, label):
                    counts[_words(label)] = counts.get(_words(label), 0) + 1
                    spelled.setdefault(_words(label), ' '.join(label.split()))
            if not counts:
                continue
            for name in (item.shelf, item.name):
                if _words(name) in counts:
                    return spelled[_words(name)].lower()
            return spelled[max(counts, key=lambda label: (counts[label], -len(label)))].lower()
        return ''

    def item_said(self, number: int) -> str:
        """How a question names the item. One property per message: the store's category label of the item's
        listings, or the item's own name when another requested item matches that label as well. Lists: the item's
        own name, with a name that joins plural nouns by "and" said with "or"; after a one-at-a-time reply about such
        an item, the one joined product whose head noun most of the item's rows name outside their category label (the
        first when none does)."""
        item = self.items[number - 1]
        if LIST_WIDTH <= 1:
            label = self.shelf_name(number)
            return item.name if not label or self.label_shared(label) else label
        parts = self.name_parts(number)
        if not parts:
            return item.name
        if number not in self.one_product:
            return re.sub('\\band\\b', 'or', item.name, flags=re.I)
        cat_key = self.category_key()
        rows = [row for row in self.rows.values() if not cat_key or self.owns(item, str(row.get(cat_key) or ''))]
        named = [_roots(' '.join((str(v) for k, v in row.items() if k != cat_key and isinstance(v, str)))) for row in rows]
        hits = {p: sum((1 for words in named if _roots(p.split()[-1]) & words)) for p in parts}
        return max(parts, key=lambda p: (hits[p], -parts.index(p)))

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        text = f'What {label} would you like for the {self.item_said(number)}'
        return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def samples(self, number: int, label: str) -> list[str]:
        want = _words(label)
        settled = ' '.join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            if name == want:
                return [v for v in values if v and v not in settled and (len(v) <= 30)][:2]
        return []

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.finished or self.left() <= floor or self.msg_used >= self.msg_limit:
                break
            args = self.message_args(text)
            if not args:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            self.msg_used += 1
            answer = reply.get('observation')
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        several = len(self.items) > 1
        for entry in self._take_heard_dicts(heard, 'needs'):
            self._take_heard_need(entry, told, several)
        for entry in self._take_heard_dicts(heard, 'drop'):
            if str(entry.get('property') or '').strip():
                self._take_heard_drop(entry, told)
        for entry in self._take_heard_dicts(heard, 'free'):
            if entry.get('property'):
                self.items[self._take_heard_number(entry) - 1].free.append(str(entry['property']))
        budget = heard.get('budget')
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget = budget

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        due = [entry for entry in self.process if entry['when'] == stage]
        if not due or self.finished or not self.gate_due(stage):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        menu = self._gate_menu(ending)
        done_text = []
        pressed = False
        rounds = GATE_ROUNDS
        while rounds > 0:
            rounds -= 1
            models = [HEAVY, LIGHT] if pressed else [LIGHT, HEAVY]
            got = self.ask_json(self._gate_prompt(stage, due, menu), models, 3500)
            if got is None:
                break
            changes, hold = self._gate_heard(got, done_text)
            named = got.get('call')
            call = named if isinstance(named, dict) else None
            self._trail_log('gate', stage=stage, call=_clip(call, 200), steps=_clip(got.get('steps'), 120), hold=hold[:200], told=changes)
            if changes or hold:
                self.version += 1
                return (False, self._gate_changed(done_text, changes, hold))
            if call:
                if self._gate_call(call, ending, done_text):
                    continue
                break
            missing = self._gate_unproven(stage, due, got)
            if missing and self._gate_may_press(done_text, pressed):
                pressed = True
                self._gate_press(missing)
                continue
            self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
            break
        return (True, self._gate_done(done_text))

    def spoken(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        phrases = self._spoken_phrases(observation)
        if not phrases:
            return _dump(observation)[:400]
        return ' '.join(phrases)[:800]

    def loose_lines(self, observation: Any) -> list[str]:
        if not isinstance(observation, dict) or self.ident(observation):
            return []
        picked = []
        for value in observation.values():
            picked.extend(self._loose_lines_of(value))
        return [text for text in picked if text not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = '\n\n'.join([NOTE, 'ITEMS AND REQUIREMENTS NOW:\n' + self.ledger(), 'THE NOTE:\n' + '\n'.join(lines)])
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        gone = self._take_note_gone(got, lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        drops = self._take_note_entries(got.get('drop'))
        adds = self._take_note_entries(got.get('add'))
        changed = self._take_note_budget(got) + [self._take_note_drop(entry) for entry in drops] + [self._take_note_add(entry) for entry in adds]
        changed += ['unavailable: ' + ', '.join(gone)] if gone else []
        changed += ['a price changed'] if got.get('price_moved') is True else []
        summary = f"{got.get('summary') or ''}".strip()
        moved_budget = any(entry.startswith('budget now') for entry in changed)
        if self.brief.get('pick') == 'cheapest' and (drops or gone or self.note_sold or moved_budget):
            self.wider_held = False
            self.cheaper_marks = set()
        return self._take_note_record(summary, changed)

    def needs_now(self) -> list[tuple[int, str]]:
        found = []
        for position, item in enumerate(self.items, 1):
            found.extend((position, need) for need in self._needs_now_live(item))
        return found

    def evidence(self, observation: Any) -> str:
        found = self._evidence_ids(observation)
        needs = self.needs_now()
        if not found or not needs:
            return ''
        several = len(self.items) > 1
        lines = []
        for ident in found[:12]:
            line = self._evidence_line(ident, needs[:8], several)
            if line:
                lines.append(line)
        if not lines:
            return ''
        return "[evidence: attribute keys whose names include a requirement's property; a key about some other property (a different part or aspect) settles nothing, and a differently named key can still carry the same property] " + ' | '.join(lines)

    def cart_text(self) -> str:
        if not self._cart_ids_view_current():
            return "lines by Nimbus's record of the cart calls (listing: units): " + _dump(self.cart_lines)
        return _dump(_clip(self.cart_view, 300))[:3000]

    def held_units(self, ident: str) -> int:
        if not self._cart_ids_view_current():
            return self.cart_lines.get(ident, 1)
        waiting = [self.cart_view]
        while waiting:
            node = waiting.pop()
            if isinstance(node, list):
                waiting.extend(node)
                continue
            if not isinstance(node, dict):
                continue
            if any(isinstance(value, str) and value.strip() == ident for value in node.values()):
                numbers = self._held_units_numbers(node)
                if len(numbers) == 1:
                    return numbers[0]
            waiting.extend(value for value in node.values() if isinstance(value, (dict, list)))
        return self.cart_lines.get(ident, 1)

    def cart_ids(self) -> list[str]:
        if not self._cart_ids_view_current():
            return list(self.cart_lines)
        shown = _dump(self.cart_view)
        return [ident for ident in self.rows if ident in shown]

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, RECORD_CHARS))[:4000] if record else '(no record seen)'

    def others(self, skip: list[str], limit: int=30) -> str:
        newest = self.seen_order[::-1]
        if self.brief.get('pick') == 'cheapest':
            newest.sort(key=self._others_rank)
        shown = []
        for ident in newest:
            if ident not in skip and not self.sold_out(ident):
                shown.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 160))[:900])
                if len(shown) >= limit:
                    break
        return '\n'.join(shown)

    def fit_index(self) -> dict[str, Any]:
        marks = {}
        notes = {}
        for ident in self.seen_order:
            found = _fit_marks(self._fit_index_text(self.full.get(ident) or self.rows.get(ident) or {}))
            marks[ident] = found
            for part, codes in found['also']:
                if part not in notes:
                    notes[part] = set()
                notes[part].update(codes)
        return {'marks': marks, 'notes': notes}

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        result = set(mark.get('works') or [])
        own = mark.get('own')
        if own:
            result |= index['notes'].get(own, set())
        return result

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        if not cat_key:
            return []
        return [ident for ident in self.seen_order if self.owns(item, self._item_ids_label(ident, cat_key))]

    def fit_faults(self) -> list[str]:
        if not self.bound:
            return []
        index = self.fit_index()
        cart = self.cart_ids()
        faults = []
        for part, main in self.bound:
            faults.extend(self._fit_faults_pair(part, main, cart, index))
        return faults

    def fit_plan(self, short: bool=False) -> str:
        if not self.bound:
            return ''
        index = self.fit_index()
        lines = []
        for main in sorted({m for _, m in self.bound}):
            text = self._fit_plan_codes(main, index)
            if text:
                lines.append(f"codes the {self.items[main - 1].name} candidates' parts name -> {text}")
        if not lines:
            return ''
        joined = ' | '.join(lines)
        if short:
            return joined
        return "[fit plan: a part suits a device when the part's text names a model code of that device or an alias it is sold as; the codes cannot be searched, so browse the device's category and read the codes shown on its rows] " + joined

    def missing_items(self) -> list[int]:
        """When the cart has fewer lines than requested items: the item numbers it has no line for, leaving out items that
        no listing seen so far carries the label of."""
        cat_key = self.category_key()
        held = self._missing_items_held()
        if not cat_key or len(held) >= len(self.items):
            return []
        store_labels = [str(row.get(cat_key) or '') for row in self.rows.values()]
        cart_labels = [self._item_ids_label(ident, cat_key) for ident in held]
        out = []
        for position, item in enumerate(self.items, 1):
            if not any(self.owns(item, label) for label in store_labels):
                continue
            if not any(self.owns(item, label) for label in cart_labels):
                out.append(position)
        return out

    def states_needs(self, ident: str, number: int) -> bool:
        """True when the listing's attribute block spells out every requirement of the item verbatim; limits and exclusions
        are not interpreted here, so a requirement of that sort counts as unstated."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        for owner, need in self.needs_now():
            if owner == number and not self._states_needs_one(block, need):
                return False
        return True

    def fill_short_cart(self) -> bool:
        """Run before an order the agent forces: for every item the cart lacks, cart the lowest-priced opened listing whose
        attributes state that item's requirements. False when some item is still uncovered (the episode then declines
        instead of ordering a partial basket)."""
        cart = self.roles.get('cart') or ''
        paired = {number for pair in self.bound for number in pair}
        for number in self.missing_items():
            item = self.items[number - 1]
            pick = None if number in paired else self._fill_short_cart_pick(number, item)
            if not pick:
                return False
            wanted = {ident: self.cart_lines[ident] for ident in self.cart_ids() if ident in self.cart_lines}
            wanted[pick] = item.qty
            calls = self.cart_calls(wanted, [])
            if not calls or len(calls) + 1 > self.left() or not self._fill_short_cart_send(cart, calls):
                return False
            self._trail_log('filled_line', item=number, line=pick)
        return True

    def fuller_label(self, number: int, label: str) -> str:
        cat_key = self.category_key()
        item = self.items[number - 1]
        words = _roots(item.name)
        if len(words) < 2 or words <= _roots(label) or not cat_key:
            return ''
        if item.shelf and _roots(label) == _roots(item.shelf):
            return ''
        for seen in sorted({str(row.get(cat_key) or '') for row in self.rows.values()}):
            if seen and words <= _roots(seen):
                return seen
        return ''

    def line_faults(self) -> list[str]:
        self.adopt_labels()
        cat_key = self.category_key()
        ids = self._missing_items_held()
        if not ids or not cat_key:
            return []
        label = {ident: self._item_ids_label(ident, cat_key) for ident in ids}
        known = set()
        for position, item in enumerate(self.items, 1):
            if any(self.owns(item, str(row.get(cat_key) or '')) for row in self.rows.values()):
                known.add(position)
        owner = {ident: self._line_faults_owners(label[ident]) for ident in ids}
        faults = []
        for position, item in enumerate(self.items, 1):
            if position in known:
                faults.extend(self._line_faults_item(position, item, ids, label, owner))
        if len(known) == len(self.items):
            faults.extend(f"{ident} ({label[ident] or 'unknown kind'}) is no kind of product the shopper requested: remove it" for ident in ids if not owner[ident])
        faults.extend(f'{ident} is sold out' for ident in ids if self.sold_out(ident))
        faults.extend(self._line_faults_budget(ids))
        return faults

    def count_fixes(self) -> list[tuple[str, int, int]]:
        cat_key = self.category_key()
        if not cat_key:
            return []
        fixes = []
        for ident in self.cart_ids():
            have = self.held_units(ident) or 0
            owners = self._line_faults_owners(self._item_ids_label(ident, cat_key))
            if len(owners) != 1:
                continue
            item = self.items[owners[0] - 1]
            same_kind = sum(1 for other in self.cart_ids() if (self.cart_lines.get(other, 1) or 0) > 0 and self.owns(item, self._item_ids_label(other, cat_key)))
            if have > 0 and same_kind == 1 and have != item.qty and not self.sold_out(ident):
                fixes.append((ident, have, item.qty))
        return fixes

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        name, schema, props = self._cart_calls_schema()
        target, count, mode = self._cart_calls_fields(name, schema, props)
        if not (name and target and count):
            return []
        calls = self._fix_counts_plan(name, wrong, props, target, count, mode)
        if calls is None or len(calls) + 3 > self.left():
            return []
        done = []
        for raw in calls:
            line = self._fix_counts_send(name, schema, raw)
            if line is None:
                break
            done.append(line)
        self._trail_log('count_fix', calls=done)
        return done

    def check_key(self, ids: list[str]) -> str:
        return f"{'|'.join(sorted(ids))}#{self.version}"

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        lead = ids is not None
        chosen = ids if lead else self.cart_ids()
        key = self.check_key(chosen)
        if key in self.checked or not self.judges(limit) or not chosen:
            return ''
        self.checked.add(key)
        got = self._check_cart_verdict(self._check_cart_prompt(chosen, lead), lead, limit)
        self._trail_log('check', cart=chosen, got=got)
        approved = bool(got) and got.get('ok') is not False
        self.verdicts[key] = approved
        return '' if approved or not got else self._check_cart_advice(got, chosen)

    def trail(self) -> tuple[str, bool]:
        pending = self._trail_pending()
        tool = self.offer_tool()
        target = self.single_target(tool)
        opened_text, lost = [], []
        while pending and self.trails < (6 if self.left() > 15 else 4) and self.cost < SPEND_HARD:
            self.trails += 1
            reach, lost, gaps = self._trail_chains(self._trail_ask(pending), pending)
            pending = [need for need in pending if need not in self.ref_values]
            if not reach and gaps and tool and target and self.left() > 12 and self.link_searches < 3:
                reach, lost = self._trail_link(gaps[0], lost, opened_text)
            if not self._trail_can_open(reach, tool, target):
                break
            self._trail_open(tool, target, reach[:max(0, min(3, 6 - len(opened_text)))], opened_text)
        return self._trail_report(opened_text, lost)

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        text = about.lower()
        return [ident for ident in ids if ident not in self.opened and self._shop_rows_named(self.rows.get(ident) or {}, text)]

    def opened_shop(self, about: str) -> list[str]:
        text = about.lower()
        return [ident for ident in self.opened if self._shop_rows_named(self.rows.get(ident) or {}, text)]

    def find_link(self, about: str, words: str) -> list[str]:
        owners = self.shop_rows(about, list(self.rows))
        owners += self.opened_shop(about)
        named = self._find_link_shop_words(owners)
        source = (words or about).lower()
        values = {w for w in re.findall('[a-z0-9]+', source) if len(w) >= 3 or w.isdigit()}
        values -= named
        values -= {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        if values:
            shown = next((ident for ident in self.opened_shop(about) if self._find_link_shows(ident, values)), None)
            if shown is not None:
                self._trail_log('link_known', id=shown)
                return []
        known = self._find_link_search(about, words)
        kind = set(re.findall('[a-z0-9]{3,}', source))
        kind -= named
        kind -= {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}
        picked = self._find_link_of_kind(known, kind)
        if not picked:
            picked = self._find_link_of_kind(self.shop_rows(about, self.seen_order[::-1]), kind)
        return picked[:2]

    def offer_tool(self) -> str | None:
        offer = self.roles.get('offer')
        return offer if offer else self.roles.get('open')

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        stale = self._reread_offers_stale() if tool and target else []
        if not stale or self.left() < len(stale) + 3:
            return ''
        notes = []
        for ident in stale:
            note = self._reread_offers_one(tool, target, ident)
            if note is None:
                break
            notes.append(note)
        changes = [note for note in notes if note]
        self._trail_log('reread', ids=stale, changed=len(changes))
        if changes:
            self.version += 1
        return '\n'.join(changes)

    def sold_out(self, ident: str) -> bool:
        if ident in self.gone:
            return True
        return ident in self.unavailable

    def swap_hint(self) -> str:
        lost = [ident for ident in self.cart_ids() if ident in self.unavailable and ident not in self.gone]
        if not lost:
            return ''
        self.gone.update(dict.fromkeys(lost, self.step))
        self.leads = [ident for ident in self.leads if ident not in lost]
        self.version += 1
        picks = self.replacements(lost)
        head = f"[Nimbus] Gone from the store: {', '.join(lost)} can no longer be bought and stays unavailable, so skip opening it again."
        return head + self._swap_hint_advice(picks)

    def find_sold(self) -> str:
        held = [ident for ident in self.cart_ids() if ident not in self.gone]
        if not held or any(ident in self.unavailable for ident in held):
            return ''
        if len(held) > 1:
            return self.reread_offers()
        self.unavailable.add(held[0])
        return ''

    def replacements(self, lost: list[str]) -> list[str]:
        if not self.judges():
            return []
        records = self._replacements_records(lost)
        if not records:
            return []
        sold = '\n'.join(f'{ident}: {self.record_of(ident)}' for ident in lost)
        prompt = ''.join([SWAP, '\n\nREQUEST:\n', self.request, '\n\nREQUIREMENTS IN FORCE:\n', self.ledger(),
                          '\n\nSOLD-OUT CART LINE:\n', sold, '\n\nCART NOW:\n', self.cart_text(),
                          '\n\nOTHER LISTING RECORDS SEEN (newest first):\n', '\n'.join(records)])
        got = self.ask_json(prompt, self.judges()[::-1], 3500)
        fits = self._replacements_fits(got, lost)
        if self.brief.get('pick') == 'cheapest':
            ranked = [ident for _, ident in sorted(fits, key=lambda pair: pair[0])]
        else:
            ranked = [ident for _, ident in fits]
        self._trail_log('swap', lost=lost, fits=fits)
        return ranked[:4]

    def review_models(self) -> list[str]:
        """Models for the evidence review before an ending without a purchase: the usual judges, else the light model
        while spend is within REVIEW_ROOM of the hard cap (an ending nobody reviewed is the costlier mistake)."""
        judges = self.judges()
        if judges:
            return judges
        return [LIGHT] if self.cost <= SPEND_HARD + REVIEW_ROOM and not self.no_credit else []

    def audit(self) -> str:
        if self.audited or not self.review_models():
            return ''
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            rows = [str(i) for i in _items((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        self.review_pending = any((str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i)) for i in _items((got or {}).get('open'))))
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        for ident in reversed(self.seen_order):
            if self.sold_out(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            records.append('OPENED ' + _dump(_clip(record, 200))[:1200] if record else 'ROW ' + _dump(_clip(self.rows.get(ident), 120))[:500])
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        judges = self.review_models()
        got = self.ask_json(prompt, [LIGHT] if judges else [], 4000)
        if HEAVY in judges and not self.no_credit and not self.audit_fits(got):
            got = self.ask_json(prompt, [HEAVY], 4000) or got
        if isinstance(got, dict):
            self.review_ran = True
            self.reviews += 1
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {}
        for key, value in fits.items():
            named[str(key)] = self._audit_fits_ids(value)
        return {key: ids for key, ids in named.items() if ids}

    def open_rows(self, rows: list[str]) -> bool:
        plan = self._open_rows_plan(rows)
        if plan is None or self.left() < len(plan) + 5:
            return False
        known_before = len(self.full)
        for name, raw in plan:
            args = self._open_rows_shaped(self.schemas.get(name), raw)
            if args is None:
                return False
            if not self._open_rows_read(name, args):
                break
        gained = len(self.full) - known_before
        self._trail_log('audit_open', rows=rows, opened=gained)
        return gained > 0

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name, schema, props = self._cart_calls_schema()
        field = self._cart_calls_list_field(props)
        if field is not None:
            return self._cart_calls_whole(field, lines)
        target, count, mode = self._cart_calls_fields(name, schema, props)
        if not target:
            return None
        setting = self.cart_sets(name)
        calls = []
        for ident in drop:
            call = self._cart_calls_drop(props, target, count, mode, ident, setting)
            if call is None:
                return None
            calls.append(call)
        held = self.cart_ids()
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in held else 0
            if have == want:
                continue
            if have and not setting:
                return None
            call = {target: ident}
            if mode:
                call[mode] = props[mode]['enum'][0]
            if count:
                call[count] = want
            calls.append(call)
        return calls

    def complete_cart(self) -> str:
        cart, buy = self.roles.get('cart'), self.roles.get('buy')
        blocked = self.completed or not self.confirmed or self.sold_out(self.confirmed)
        if blocked or not (cart and buy):
            return ''
        self.completed = True
        cat_key = self.category_key()
        plan = self._complete_cart_plan(cat_key)
        if self.confirmed not in plan:
            return ''
        keep, drop = self._complete_cart_split(plan, cat_key)
        kinds_kept = {self._complete_cart_kind(ident, cat_key) for ident in keep}
        if len(plan) + len(kinds_kept) < len(self.items):
            return ''
        lines = dict(keep)
        lines.update(plan)
        calls = self.cart_calls(lines, drop)
        room = len(calls) + 3 if calls else 0
        if not calls or room - 2 > self.left():
            return ''
        self.fill_facts(list(lines), room)
        self.show_links(list(lines), room)
        if not self._complete_cart_confirmed(lines):
            return ''
        done = self._complete_cart_apply(cart, calls)
        args = None if done is None else self._open_rows_shaped(self.schemas.get(buy), dict(self.brief.get('buy_args') or {}))
        replies = [] if args is None else self._complete_cart_checkout(buy, args, lines, drop)
        if not replies:
            return ''
        return 'Not run (Nimbus acted instead): the independent check confirmed ' + self.confirmed + ', so Nimbus set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        return self._is_buy_match('buy', name, args)

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        return self._is_buy_match('refuse', name, args)

    def offered(self) -> list[dict[str, Any]]:
        ask = self.roles.get('ask')
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        keep = self._offered_keep(ends)
        tools = [tool for tool in self.tools if tool['function']['name'] != ask]
        if keep is None or not ends:
            return tools
        narrowed = [tool for tool in tools if tool['function']['name'] in keep]
        return narrowed or tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        out = dict(args)
        for key, spec in (schema.get('properties') or {}).items():
            if self._quiet_free_text(spec) and self._quiet_should_fill(out.get(key), key, schema):
                out[key] = 'Apologies: I could not find anything here that satisfies all of your requirements, so this ends without an order.'[:spec.get('maxLength', 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        field = self._resolve_ids_field(name)
        if field is None:
            return (args, '')
        out = dict(args)
        if field not in args:
            return (out, '')
        raw = args[field]
        several = isinstance(raw, list)
        fixed = [self._resolve_ids_one(value) for value in (raw if several else [raw])]
        problem = next((note for _, note in fixed if note), '')
        if problem:
            return (args, problem)
        out[field] = [value for value, _ in fixed] if several else fixed[0][0]
        return (out, '')

    def printed_code(self, name: str, args: dict[str, Any]) -> str:
        """A value sent to a listing tool that is shaped like an identifier, is no listing's id, and is printed inside
        the text of a listing seen (a part or model code in a title); '' when there is none or that code was already
        pointed out once (a second try goes through)."""
        if name not in {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.rows:
            return ''
        found = self.id_arguments(name, args)
        for value in found:
            if not value or value in self.rows or value in self.code_warned or len(value) > 40 or not TOKEN_ID.fullmatch(value):
                continue
            if self.id_shaped(value):
                continue
            pattern = re.compile('(?<![\\w-])' + re.escape(value) + '(?![\\w-])', re.I)
            for row in self.rows.values():
                if any((isinstance(v, str) and v.strip() != value and pattern.search(v) for v in row.values())):
                    self.code_warned.add(value)
                    return value
        return ''

    def id_arguments(self, name: str, args: dict[str, Any]) -> list[str]:
        """The texts a call passes where a listing id belongs: arguments that carried ids in earlier calls of this tool,
        the one target of a single-listing tool, the list of a side-by-side tool; for a cart tool not used yet, its
        required free-text or list arguments (inside a list of objects, their free-text fields)."""
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        keys = {key for tool, key in self.id_args if tool == name}
        for key in (self.single_target(name), self.compare_target() if name == self.roles.get('compare') else None):
            if key:
                keys.add(key)
        if not keys and name == self.roles.get('cart'):
            keys = {k for k in schema.get('required') or [] if isinstance(props.get(k), dict) and 'enum' not in props[k] and {'string', 'array'} & set(_kinds(props[k]))}
        texts: list[str] = []
        for key in keys:
            value = args.get(key)
            inner = ((props.get(key) or {}).get('items') or {}).get('properties') or {} if isinstance(props.get(key), dict) else {}
            for one in value if isinstance(value, list) else [value]:
                if isinstance(one, str):
                    texts.append(one.strip())
                elif isinstance(one, dict):
                    texts.extend((v.strip() for k, v in one.items() if isinstance(v, str) and 'enum' not in (inner.get(k) or {})))
        return texts

    def id_shaped(self, value: str) -> bool:
        """Whether a value has the shape of the listing ids seen so far: the same head (the letters and the separator
        that open every known id, when they share one) and one of their lengths."""
        known = list(self.rows)[:200]
        if not known:
            return False
        heads = {re.match('[A-Za-z]*[^A-Za-z0-9]?', ident).group() for ident in known}
        if len(heads) != 1:
            return False
        head = heads.pop()
        return value.startswith(head) and len(value) in {len(ident) for ident in known}

    def cart_sets(self, name: str) -> bool:
        specs = [spec for spec in (self.schemas.get(name, {}).get('properties') or {}).values() if isinstance(spec, dict)]
        for spec in specs:
            if isinstance(spec.get('enum'), list) and len(spec['enum']) >= 2:
                return False
        return any('integer' in _kinds(spec) and spec.get('minimum') == 0 for spec in specs)

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """When a removal of a line in the cart comes without a count, fill in the units held (capped by the schema), as the
        store would otherwise take off a single unit."""
        spec = self.schemas.get(name, {}).get('properties') or {}
        count = self._whole_removal_count(spec)
        removal = self._whole_removal_flag(spec, args)
        ident = next((value for value in args.values() if isinstance(value, str) and value in self.cart_lines), None)
        if not (count and removal and ident) or count in args:
            return args
        held = self.cart_lines[ident]
        out = dict(args)
        out[count] = min(held, spec[count].get('maximum', held))
        return out

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_at = self.step
        self.cart_touched = True
        spec = self.schemas.get(name, {}).get('properties') or {}
        removal = self._note_cart_removal(spec, args)
        whole, lines = self._note_cart_lines(spec, args, self._note_cart_loose(args.values()))
        if whole:
            self.cart_lines = dict((ident, units) for ident, units in lines if units and units > 0)
            return
        sets = self.cart_sets(name)
        counted = removal and any(isinstance(entry, dict) and 'integer' in _kinds(entry) for entry in spec.values())
        for ident, units in lines:
            have = self._note_cart_units(self.cart_lines.get(ident, 0), units, removal, sets, counted)
            if have <= 0:
                self.cart_lines.pop(ident, None)
                continue
            self.cart_lines[ident] = have

    def show(self, observation: Any) -> str:
        """A reply as the loop model reads it: each text field cut at SHOW_CHARS on a page of rows, at PAGE_CHARS on a
        page of one to three listings (their notes and pointers run long), the whole cut at OBS_LIMIT."""
        limit = SHOW_CHARS
        shown = len(self.records_in(observation))
        if 0 < shown <= 3 and len(_dump(observation)) <= 12 * PAGE_CHARS:
            limit = PAGE_CHARS
        text = _dump(_clip(observation, limit))
        return text if len(text) <= OBS_LIMIT else text[:OBS_LIMIT] + '...(cut)'

    def compare_target(self) -> str | None:
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        arrays = [key for key, spec in (schema.get('properties') or {}).items() if isinstance(spec, dict) and 'array' in _kinds(spec)]
        target = arrays[0] if arrays else None
        if not target:
            return None
        return target if all(key == target for key in schema.get('required') or []) else None

    def compare_room(self) -> int:
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        spec = (schema.get('properties') or {}).get(self.compare_target() or '') or {}
        limit = spec.get('maxItems')
        if isinstance(limit, int):
            return limit
        if not self.compare_bad:
            return max(4, self.compare_ok)
        return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener = self.roles.get('open')
        joiner = self.roles.get('compare')
        target = self.single_target(opener)
        many = self.compare_target()
        if not (opener and joiner and target and many) or opener == joiner or self.compare_bare():
            return calls
        picks = self._join_opens_picks(calls, opener, target)[:self.compare_room()]
        if len(picks) < 2:
            return calls
        ids = [ident for _, ident in picks]
        folded = {index for index, _ in picks}
        first = picks[0][0]
        merged = _dump({many: ids})
        out = []
        for index, call in enumerate(calls):
            if index not in folded:
                out.append(call)
            elif index == first:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': merged}})
            else:
                out.append({'id': call['id'], 'joined': True})
        self._trail_log('joined', ids=ids)
        return out

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Nimbus opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask'):
                out.append((cid, "Not run (no step used): Nimbus's interviewer handles every message to the shopper."))
                continue
            if name not in self.schemas:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _json_obj(raw) if (raw or '').strip() else {}
                if args is None:
                    out.append((cid, 'Not run (no step used): the arguments were not one JSON object.'))
                    continue
            faults: list[str] = []
            args = _shape(args, self.schemas[name], 'args', faults)
            if faults:
                out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6])))
                continue
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            code = self.printed_code(name, args)
            if code:
                self.trace.append({'kind': 'code_not_id', 'tool': name, 'code': code})
                out.append((cid, f'Not run (no step used): {code} is a code printed inside a listing\'s text, not a listing id, so the store would find nothing. Pass the listing id from the result rows' + (f' (their {self.id_key} field)' if self.id_key else '') + '; to find the listing a code names, search for the code\'s words.'))
                continue
            if name == self.roles.get('cart'):
                args = self.whole_removal(name, args)
            trimmed = ''
            many = self.compare_target() if name == self.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.compare_room()) and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Nimbus] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in (self.roles.get('open'), self.roles.get('offer')) and (str(args.get(target)) in self.gone):
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if name == self.roles.get('cart') and any(((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
                view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and (v not in self.cart_lines) and (v not in view) and (v not in self.cart_said)]
                if absent:
                    out.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.cart_text()))
                    continue
            ending = self.is_buy(name, args) or self.is_refuse(name, args)
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not ending and self.left() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            mark = name + _dump(args)
            if not ending and name != self.roles.get('cart') and (mark in self.recent):
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            readers = {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
            cached = self.read_cache.get(mark)
            if name in readers and cached and (cached[0] == self.version):
                out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            if target and name in readers and (name != self.roles.get('compare')) and self.shown_already(name, str(args.get(target))):
                out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + self.record_of(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.shown_already(name, str(i))]
                if not keep:
                    out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.record_of(str(i))}' for i in args[many]))[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Nimbus] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if self.is_refuse(name, args) and self.decline_refused:
                verdict, said = self.second_decline()
                if verdict == 'hold':
                    out.append((cid, said))
                    continue
                order = self.order_call() if verdict == 'order' else None
                if order is not None:
                    name, args = order
                    mark = name + _dump(args)
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + '\nCart now: ' + self.cart_text()))
                        continue
                    reread = self.reread_offers()
                    if self.note_sold:
                        self.find_sold()
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get('cart_view') and (self.left() > 3):
                        faults = []
                        view_args = _shape({}, self.schemas.get(self.roles['cart_view']), 'args', faults)
                        view = self.act(self.roles['cart_view'], view_args) if not faults else None
                        if view is not None:
                            self.cart_view = view.get('observation')
                            self.cart_seen_at = self.step
                            change = self.observe(self.roles['cart_view'], self.cart_view)
                            if change:
                                self.queued = []
                                out.append((cid, 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.cart_text() + '\nApply the change, fix the cart if needed, then check out.'))
                                continue
                    swap = self.swap_hint()
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.cart_text()))
                        continue
                    misfit = self.fit_faults() if self.fit_holds < 2 and self.left() > self.tail() + 3 else []
                    if misfit:
                        self.fit_holds += 1
                        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        out.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.cart_text()))
                        continue
                    faults = self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
                    fixed = []
                    if faults and all((' is held at ' in f for f in faults)):
                        fixed = self.fix_counts(self.count_fixes())
                        faults = self.line_faults() if fixed else faults
                        if fixed and (not faults):
                            self.line_holds += 1
                            out.append((cid, "Not run yet (no step used). Nimbus set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()))
                            continue
                    if faults:
                        self.line_holds += 1
                        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                        out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.cart_text()))
                        continue
                    lines = self.cart_ids()
                    read = self.fill_facts(lines, self.tail() + (1 if len(lines) >= 2 else 3))
                    self.show_links(lines, self.tail() + 1)
                    read += self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = "Nimbus read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Nimbus re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.left() > 5:
                    review = self.audit()
                    if review:
                        out.append((cid, 'Not run yet (no step used). ' + review))
                        continue
                    if self.leads and self.left() > 4 and (self.refuse_blocks == 0):
                        self.refuse_blocks = 1
                        self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
                        out.append((cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                        continue
                    if self.leads and self.left() > 4 and (self.refuse_blocks == 1):
                        self.refuse_blocks = 2
                        first = self.leads[0]
                        self.check_cart([first], SPEND_HARD)
                        if self.verdicts.get(self.check_key([first])) and first in self.leads:
                            self.confirmed = first
                            self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                            out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
            if self.is_refuse(name, args) and self.left() <= 5 and (not self.late_held):
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh))]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            if self.is_refuse(name, args) and (not self.leads) and (not self.pushed) and (self.left() > self.max_steps // 3):
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                order = self.unreviewed_order() if not self.decline_refused else None
                if order is not None:
                    name, args = order
                else:
                    args = self.quiet(name, args)
            if self.is_buy(name, args):
                late = self.late_question()
                if late:
                    out.append((cid, 'Not run yet. ' + late + '\nCart now: ' + self.cart_text()))
                    continue
            if ending and self.left() >= 2:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Nimbus] ' + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Nimbus] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all((isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))))
            self.absorb(observation)
            if name == self.roles.get('compare'):
                rows = self.records_in(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any((_fact_block(r) for _, r in rows))
            if name in readers - {self.roles.get('search')}:
                self.note_view(name, observation)
            plain = {tool for job, tool in self.roles.items() if job != 'search'}
            paged = self.more_pages(name, args, observation) if name not in plain and (not ending) else ''
            if self.single_target(name) and name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(observation))
                    if name == self.roles.get('open') and (not self.facts_from[name]):
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and (self.facts_from.get(t) is not False) and (t not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search')))), None)
                        if other:
                            self.roles['open'] = other
                    if not self.facts_from[name] and name != self.roles.get('open') and (not self.roles.get('offer')):
                        self.roles['offer'] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get('cart'):
                self.note_cart(name, args)
                self.cart_said = _dump(observation)
                self.recent = []
                counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None]
                for value in args.values():
                    if counts and isinstance(value, str) and (value in self.rows):
                        if counts[0] <= 0 and (not _in_rows(observation, value)):
                            self.dropped.add(value)
                        elif counts[0] > 0:
                            self.dropped.discard(value)
                lines = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
                if len(lines) >= len(self.built):
                    self.built = lines
            if name == self.roles.get('search') and self.records_in(observation):
                self.searched = (self.searched + [dict(args)])[-8:]
            if name == self.roles.get('cart_view'):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get('finished'):
                    self.refused_at = self.step
            refused_end = self.is_refuse(name, args) and (not reply.get('finished')) and (not self.finished)
            if refused_end:
                self.decline_refused = True
                self.refused_end_at = self.step
                self.trace.append({'kind': 'decline_refused', 'step': self.step})
            text = self.show(observation) + paged + trimmed
            if refused_end:
                text += '\n[Nimbus] The store did not accept ending without a purchase. Apply every change it states first; when listings fit now, cart one suitable listing per requested item and place the order. Ending that way again goes through only if a fresh review finds nothing that fits after the change.'
            if name not in plain and (not ending):
                text += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, args, observation)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                    text += '\n[Nimbus] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
            refused = self.is_buy(name, args) and (not reply.get('finished'))
            again = refused and any((line in self.spoken(observation) for line in self.sold_lines))
            loose = self.note_sold and (refused or (not self.note_named and (not self.is_buy(name, args))))
            if (loose or again) and self.cart_ids() and (not self.finished):
                seen = self.find_sold()
                if seen:
                    text += "\n[Nimbus re-read the cart's lines after the refusal]\n" + seen
            self.note_sold = False
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes seen on Nimbus's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def order_call(self) -> tuple[str, dict[str, Any]] | None:
        """The checkout call with its fixed arguments, or None when the store has none that fits its schema."""
        buy = self.roles.get('buy')
        if not buy or buy not in self.schemas:
            return None
        faults: list[str] = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
        return None if faults else (buy, args)

    def whole_basket(self) -> bool:
        """Whether the cart holds a buyable line for every requested item (by the category labels seen)."""
        lines = [i for i in self.cart_ids() if not self.sold_out(i) and (self.held_units(i) or 0) > 0]
        return bool(lines) and len(lines) >= len(self.items) and (not self.missing_items())

    def fresh_review(self) -> str:
        """Once after a refused ending without a purchase: review the records against the requirements as they now
        stand. 'fit' (leads named), 'nofit' (the review ran, named nothing and left no promising row unopened) or
        'unknown' (no review could run, or rows it asked for stayed closed)."""
        if self.recheck_done:
            return 'nofit' if self.decline_cleared else 'fit' if self.leads else 'unknown'
        self.recheck_done = True
        before = self.reviews
        self.audited = False
        self.leads, self.lead_items = [], {}
        self.audit()
        if self.leads:
            self.trace.append({'kind': 'decline_after_refusal', 'review': 'fits', 'leads': list(self.leads)})
            return 'fit'
        if self.reviews > before and not self.review_pending:
            self.decline_cleared = True
            self.trace.append({'kind': 'decline_after_refusal', 'review': 'no fit'})
            return 'nofit'
        return 'unknown'

    def second_decline(self) -> tuple[str, str]:
        """An ending without a purchase after the store refused one: ('decline', '') lets it go ahead, ('order', '')
        turns it into an order of the cart, ('hold', text) answers the call instead. The first such call reviews the
        listings against the requirements as they now stand (fresh_review): no fit lets the ending go ahead (the
        refusal may have left nothing affordable), a fit holds it with those listings. Later calls order the cart
        unless its last review failed, or fill an empty cart from the best candidates."""
        if not self.recheck_done:
            verdict = self.fresh_review()
            if verdict == 'nofit':
                return ('decline', '')
            if verdict == 'fit':
                return ('hold', 'Not run yet (no step used). The store refused ending without a purchase, and a review against the requirements as they stand now names listings that may fit: ' + ', '.join(self.leads) + '. Read their pages where needed, then put the strongest of them for each requested item in the cart at its quantity and order.')
        if self.decline_cleared:
            return ('decline', '')
        lines = [i for i in self.cart_ids() if not self.sold_out(i) and (self.held_units(i) or 0) > 0]
        if lines:
            if self.verdicts.get(self.check_key(lines)) is False and self.decline_holds < 2 and self.left() > 3:
                self.decline_holds += 1
                return ('hold', 'Not run (no step used). Ending without a purchase was refused before, and the last review of this cart found problems: fix those lines, then place the order.')
            failed = self.verdicts.get(self.check_key(lines)) is False
            if failed and self.leads and any((i not in self.leads for i in lines)) and self.left() > 3:
                built = self.build_basket()
                self.trace.append({'kind': 'decline_to_order', 'why': 'refused before, failed cart swapped for reviewed fits', 'lines': built})
                if built:
                    return ('order', '')
            self.trace.append({'kind': 'decline_to_order', 'why': 'refused before', 'lines': lines})
            return ('order', '')
        if self.decline_holds >= 2 or self.left() <= 2:
            built = self.build_basket()
            self.trace.append({'kind': 'decline_to_order', 'why': 'refused before, basket built', 'lines': built})
            return ('order', '') if built else ('decline', '')
        self.decline_holds += 1
        picks = self.basket_picks()
        named = '; '.join((f'item {n}: {i}' for n, i in picks.items()))
        return ('hold', 'Not run (no step used). The store already refused ending without a purchase in this episode, so it expects an order. Cart a suitable listing per requested item at its quantity, then complete the order.' + (f' Best candidates seen: {named}.' if named else ' Search for each item again if you have no candidate.'))

    def unreviewed_order(self) -> tuple[str, dict[str, Any]] | None:
        """A decline that no evidence review has checked, sent with a whole basket in the cart: review the records seen
        (no store step) and order the cart when the review names fits; order it outright when no review can run at
        all (spend exhausted); None lets the decline go ahead."""
        if self.review_ran or not self.whole_basket():
            return None
        if self.review_models():
            named = self.audit_fits(self.audit_call())
            listed = {i for ids in named.values() for i in ids}
            lines = [i for i in self.cart_ids() if not self.sold_out(i) and (self.held_units(i) or 0) > 0]
            if not named or any((i not in listed for i in lines)):
                return None
            why = 'review names the cart'
        else:
            why = 'no review could run'
        order = self.order_call()
        if order is not None:
            self.trace.append({'kind': 'decline_to_order', 'why': why, 'lines': self.cart_ids()})
        return order

    def basket_picks(self) -> dict[int, str]:
        """For each item, the best listing seen to buy when an order is due, among buyable listings of its kind whose
        price (times the quantity, for a total budget) fits the budget: one the evidence review named, else the
        cheapest opened one whose attributes state every requirement, else the cheapest opened one, else the cheapest
        row."""
        picks: dict[int, str] = {}
        for number in range(1, len(self.items) + 1):
            price = lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf')
            units = 1 if self.brief.get('budget_scope') == 'each' else self.items[number - 1].qty
            cap = self.budget if isinstance(self.budget, (int, float)) else None
            own = [i for i in self.item_ids(number) if self.buyable(i) and not (cap is not None and cap + 1e-09 < price(i) * units <= 10 * cap)]
            named = [i for key, ids in self.lead_items.items() for i in ids if re.sub('\\D', '', str(key)) == str(number) and i in own]
            stated = sorted((i for i in own if i in self.full and self.states_needs(i, number)), key=price)
            opened = sorted((i for i in own if i in self.full), key=price)
            pool = named or stated or opened or sorted(own, key=price)
            if pool:
                picks[number] = pool[0]
        return picks

    def build_basket(self) -> list[str]:
        """Cart the basket_picks() at the requested quantities (only after a refused ending without a purchase);
        returns the lines carted, [] when an item has no pick or a cart call fails."""
        cart = self.roles.get('cart')
        picks = self.basket_picks()
        if not cart or len(picks) < len(self.items):
            return []
        lines = {ident: self.items[number - 1].qty for number, ident in picks.items()}
        calls = self.cart_calls(lines, [i for i in self.cart_ids() if i not in lines])
        if not calls or len(calls) + 1 > self.left():
            return []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults or self.act(cart, args) is None:
                return []
            self.note_cart(cart, args)
        return list(lines)

    def can_refuse(self) -> bool:
        refuse = self.roles.get('refuse')
        if not refuse:
            return False
        faults: list[str] = []
        _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
        return not faults

    def finish(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if self.decline_refused and not self.recheck_done and self.left() >= 2 and self.model_room() > CALL_FLOOR + 4:
            self.fresh_review()
        if self.decline_refused and self.decline_cleared and not (holding and self.cart_at > self.refused_end_at):
            holding = False
            buy = None
        elif not holding and self.decline_refused and buy and self.left() >= 2:
            holding = bool(self.build_basket())
            self.trace.append({'kind': 'forced_basket', 'held': holding})
        whole = True
        if holding and len(self.items) >= 2:
            whole = self.fill_short_cart() or not self.can_refuse()
            if whole and len(self.cart_ids()) >= 2 and self.clock() < LAST_TURN + 10:
                self.fill_facts(self.cart_ids(), self.tail())
                if self.left() >= 3 and self.model_room() > 5:
                    self.show_links(self.cart_ids(), self.tail())
            if not whole:
                self.trace.append({'kind': 'short_cart_refused', 'lines': self.cart_ids()})
        if self.finished or self.left() <= 0:
            return ''
        if whole and buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            faults: list[str] = []
            args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
            if not faults:
                if self.left() >= 2:
                    self.gate('order')
                if self.finished or self.left() <= 0:
                    return ''
                reply = self.act(buy, args)
                tries = 0
                while reply is None and not (self.finished or self.broke) and END_DEADLINE - self.clock() > 4 and tries < 3:
                    time.sleep(max(0.0, min(2.0, END_DEADLINE - self.clock() - 3)))
                    tries += 1
                    reply = self.act(buy, args)
                self.trace.append({'kind': 'forced_buy', 'reply': _clip(reply, 200)})
                if reply is None and not self.finished:
                    return ''
                if self.finished or self.left() <= 0:
                    return ''
                if reply is not None and again:
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, reply.get('observation'))
                    return 'the checkout was refused: ' + self.show(reply.get('observation'))
                if reply is not None and (not self.finished) and self.left() >= 1:
                    again_reply = self.act(buy, args)
                    self.trace.append({'kind': 'forced_buy_again', 'reply': _clip(again_reply, 200)})
                    return ''
        if refuse:
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
            if not faults:
                if self.left() >= 2:
                    self.gate('end')
                if self.finished or self.left() <= 0:
                    return ''
                reply = self.act(refuse, args)
                self.trace.append({'kind': 'forced_refuse', 'reply': _clip(reply, 200)})
                if reply is not None and (not self.finished):
                    self.decline_refused = True
                    self.refused_end_at = self.step
                    self.observe(refuse, reply.get('observation'))
                    return 'ending without a purchase was refused: ' + self.show(reply.get('observation'))
        return ''

    def trim(self) -> None:
        size = sum((len(_dump(m.get('content'))) for m in self.chat))
        if size <= HISTORY_LIMIT:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            content = self.chat[index].get('content') or ''
            if len(content) > 600:
                ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                self.chat[index]['content'] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        """The whole episode: read the request, let the loop model work the store, then make sure it ends with the
        order or the ending tool (a refused finish gets a short second loop before the last forced finish)."""
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': self.opening_text()}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if self.finished:
            return self.closed()
        turned = self.finish(again=True)
        if turned and not (self.finished or self.broke or self.no_credit) and self.left() > 1:
            self.chat.append({'role': 'user', 'content': '[Nimbus] An attempt to end the episode came back: ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
            self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
        if not self.finished:
            self.finish()
        return self.closed()

    def opening_text(self) -> str:
        """The loop model's first message: the request, what the store shows before any call (read into the ledger
        too), the first move, and the status block."""
        parts = ["SHOPPER'S REQUEST:\n" + self.request]
        shown = self.view.get('opening')
        if shown is not None:
            shown = _decode(shown)
            self.absorb(shown)
            parts.append('WHAT THE STORE SHOWS FIRST:\n' + self.show(shown))
        parts.append("First search each requested item's category and open several of its listings in full: the interviewer plans its questions from the properties they show.")
        return '\n\n'.join(parts) + '\n\n' + self.status()

    def closed(self) -> list[dict[str, Any]]:
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.clock() < LAST_TURN or (not self.cart_ids() and self.model_room() > EMPTY_ROOM)):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get('tool_calls') or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                raw = fn.get('arguments')
                parsed = raw if isinstance(raw, dict) else _json_obj(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get('id') or f'k{self.loop_calls}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _dump(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            if not isinstance(content, str):
                content = _dump(content) if content else ''
            if not calls:
                idle += 1
                self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            if not self.asked and self.roles.get('ask') and (self.msgs_free() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.left() > 8) and (self.clock() <= ASK_BY):
                self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                for call in calls:
                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                summary = self.interview()
                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.status()})
                continue
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.run_calls(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.status()
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

    def _read_request_prompt(self):
        listed = []
        for tool in self.tools:
            spec = tool['function']
            params = _dump(spec.get('parameters') or {})
            listed.append(f"- {spec['name']}: {spec.get('description', '')} | parameters: {params}")
        text = BRIEF + '\nTOOLS:\n' + '\n'.join(listed) + '\n\nREQUEST:\n' + self.request
        opening = self.view.get('opening')
        if opening is not None:
            text += '\n\nOPENING NOTE FROM THE STORE:\n' + _dump(_clip(opening, 1500))
        return text

    @staticmethod
    def _read_request_whole(brief):
        """Whether a brief names items and a checkout tool."""
        roles = brief.get('roles')
        return bool(_items(brief.get('items'))) and isinstance(roles, dict) and bool(roles.get('buy'))

    def _read_request_brief(self, prompt):
        fixed = max(0, prompt.find('\n\nREQUEST:\n'))
        reply = self.ask_json(prompt, [LIGHT], 4000, fixed)
        first = reply if reply else {}
        if self.no_credit:
            return first
        if not self._read_request_whole(first):
            return self.ask_json(prompt, [HEAVY, LIGHT], 2500, fixed) or first
        if self.shared_shelf(first):
            second = self.ask_json(prompt, [HEAVY], 2500, fixed) or {}
            if self._read_request_whole(second):
                return second
        return first

    def _read_request_modes(self, brief):
        """One tool both checks out and ends without a purchase: learn which fixed arguments pick each mode, unless
        the brief already gives two different sets."""
        tool = self.roles.get('buy')
        if not tool or tool != self.roles.get('refuse'):
            return
        order_args, end_args = (brief['buy_args'], brief['refuse_args'])
        if order_args and end_args and (order_args != end_args):
            return
        question = f'The store tool {tool} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _dump(self.schemas.get(tool)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
        reply = self.ask_json(question, [LIGHT, HEAVY], 1500)
        modes = reply if reply else {}
        order_args = modes.get('buy_args')
        end_args = modes.get('refuse_args')
        if not (isinstance(order_args, dict) and isinstance(end_args, dict)) or order_args == end_args:
            return
        brief['buy_args'] = order_args
        brief['refuse_args'] = end_args

    @staticmethod
    def _read_request_item(entry):
        units = _count(entry.get('quantity'))
        if not units or units <= 0:
            units = 1
        name = str(entry.get('name') or 'the product')
        shelf = str(entry.get('shelf') or '')
        return Item(name, shelf, units, list(map(str, filter(None, _items(entry.get('musts'))))))

    @staticmethod
    def _read_request_clash(first, second):
        """Whether two items give one property different values."""
        for one in first.musts:
            prop_one, _, value_one = one.partition(':')
            for two in second.musts:
                prop_two, _, value_two = two.partition(':')
                if _words(prop_one) == _words(prop_two) and _words(value_one) != _words(value_two):
                    return True
        return False

    def _read_request_items(self, brief):
        """Add the brief's items, merge an item into an earlier one with the same store label and no clashing
        requirement, and return where each item number went."""
        self.items.extend(map(self._read_request_item, filter(lambda entry: isinstance(entry, dict), _items(brief.get('items')))))
        became = {}
        kept: list[Item] = []
        for old in range(1, len(self.items) + 1):
            item = self.items[old - 1]
            place = self._read_request_twin(kept, item)
            if place >= 0:
                self._read_request_fold(kept[place], item)
                became[old] = place + 1
            else:
                kept.append(item)
                became[old] = len(kept)
        self.items = kept or [Item('the requested product', '', 1, [])]
        return became

    def _read_request_fold(self, twin, item):
        """Merge a repeated item's requirements into the earlier item."""
        extra = [must for must in item.musts if must not in twin.musts]
        self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
        twin.musts.extend(extra)

    def _read_request_twin(self, kept, item):
        """Where in `kept` the earlier item this one repeats sits (same store label, no clashing requirement), or -1."""
        if not item.shelf.strip():
            return -1
        shelf = _words(item.shelf)
        for place, other in enumerate(kept):
            if _words(other.shelf) == shelf and (not self._read_request_clash(other, item)):
                return place
        return -1

    def _read_request_links(self, brief, became):
        for pair in filter(lambda entry: isinstance(entry, dict), _items(brief.get('fits'))):
            bond = tuple((became.get(_count(pair.get(side)) or 0) for side in ('item', 'with')))
            if all(bond) and bond[0] != bond[1] and all((1 <= side <= len(self.items) for side in bond)) and bond not in self.bound:
                self.bound.append(bond)
        for lookup in _items(brief.get('lookups')):
            entry = self._read_request_lookup(str(lookup).strip())
            if entry[1] and entry not in self.refs:
                self.refs.append(entry)

    def _read_request_lookup(self, text):
        """A lookup requirement as (item number, the item's own wording of it): matched whole, else by its property."""
        listed = [must for item in self.items for must in item.musts]
        if text not in listed:
            head = _words(text.partition(':')[0])
            alike = [must for must in listed if _words(must.partition(':')[0]) == head]
            if alike:
                text = alike[0]
        for owner, item in enumerate(self.items, 1):
            if text in item.musts:
                return (owner, text)
        return (1, text)

    @staticmethod
    def _read_request_money(raw):
        """The budget as a positive number, or None."""
        if isinstance(raw, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', raw):
            raw = float(raw)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not raw > 0:
            return None
        return raw

    @staticmethod
    def _read_request_step(entry):
        """A process step of the brief in the ledger's form, or None when it names no step."""
        if not isinstance(entry, dict):
            return None
        step = str(entry.get('step') or '').strip()
        if not step:
            return None
        moment = str(entry.get('when') or '').strip().lower()
        if moment not in ('choose', 'order', 'end'):
            moment = 'order'
        count = _count(entry.get('messages')) or 0
        return {'step': step, 'when': moment, 'messages': max(0, count)}

    def _read_request_limits(self, brief):
        money = self._read_request_money(brief.get('budget'))
        if money is not None:
            self.budget = money
        allowed = _count(brief.get('messages'))
        if allowed is None or allowed < 0:
            if self.roles.get('ask'):
                self.msg_limit = 3
        else:
            self.msg_limit = allowed
        cap = _count(brief.get('steps'))
        if (cap or 0) >= 10 and cap < self.max_steps:
            self.max_steps = cap
        self.process += [step for step in map(self._read_request_step, _items(brief.get('process'))) if step]
        if not self.roles.get('ask'):
            return
        talking = [entry['messages'] for entry in self.process if entry['messages'] and entry['when'] in ('order', 'end')]
        if talking:
            self.reserve = min(max(talking), self.msg_limit)
        else:
            self.reserve = 0

    def _unmix_moved(self, item, pointers, rest):
        parts = (UNMIX, '\n\nREQUEST:\n', self.request, f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n', '\n'.join(pointers), '\nOTHER REQUIREMENTS:\n', '\n'.join(rest))
        reply = self.ask_json(''.join(parts), [HEAVY, LIGHT], 1500)
        named = set()
        for entry in _items((reply or {}).get('describe_other')):
            named.add(str(entry).strip())
        return [must for must in rest if must in named]

    def _unmix_fold(self, number, item, pointer, moved):
        joined = f"{pointer} (that listing: {'; '.join(moved)})"
        self.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})
        item.musts = [joined if must == pointer else must for must in item.musts if must not in moved]
        renamed = []
        for owner, need in self.refs:
            renamed.append((owner, joined) if (owner, need) == (number, pointer) else (owner, need))
        self.refs = renamed

    def _described_blocks(self, item):
        """The attribute blocks of the opened listings that belong to the item (all opened ones without a label key)."""
        cat_key = self.category_key()
        found = []
        for record in self.full.values():
            if not cat_key or self.owns(item, str(record.get(cat_key) or '')):
                block = _fact_block(record)
                if block:
                    found.append(block)
        return found

    def _ledger_item(self, number, item, many):
        live = [must for must in (*item.musts, *item.told) if must not in item.dropped]
        own_words = list(filter(lambda must: self.described(number, must), live))
        needs = []
        for must in live:
            if must in own_words:
                continue
            if must in self.ref_values:
                must += f' (the page it points at gives: {self.ref_values[must]})'
            needs.append(must)
        title = f'item {number}: {item.name}' if many else f'item: {item.name}'
        if item.shelf:
            title += f' (store label: {item.shelf})'
        stated = '; '.join(needs) if needs else 'none beyond the kind of product'
        line = f'{title}, quantity {item.qty}; requirements: {stated}'
        if own_words:
            line += f"; described in the shopper's words, which no listing attribute states (decides nothing): {'; '.join(own_words)}"
        return line

    def _ledger_terms(self):
        brief = self.brief
        terms: list[str] = []
        if self.budget is not None:
            each = brief.get('budget_scope') == 'each'
            terms.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {('for each item' if each else 'in total')}".replace('  ', ' '))
        terms.append(self._ledger_pick(brief.get('pick')))
        if brief.get('in_stock_only', True) is not False:
            terms.append('in stock only')
        terms += [f'wish (tie-breaker only): {wish}' for wish in _items(brief.get('wishes'))]
        terms += [f'rule from the request: {rule}' for rule in _items(brief.get('rules'))]
        if brief.get('quiet_decline') is True:
            terms.append('if nothing fits: end without naming or suggesting any listing')
        moments = {'choose': 'before choosing', 'order': 'at the order (Nimbus does it when you check out)', 'end': 'at an ending without a purchase (Nimbus does it then)'}
        terms += [f"step the request asks for, {moments[entry['when']]}: {entry['step']}" for entry in self.process]
        return terms

    def _ledger_pick(self, pick):
        if pick in ('most', 'least'):
            return f"pick: the {pick} {self.brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' ')
        if pick == 'cheapest':
            return 'pick: the cheapest listing that fits (lowest price)'
        return 'pick: any listing that fits everything'

    def _ledger_source(self):
        facts = self.facts_tool()
        if not facts or (self.facts_from.get(facts) is not True and (not self.roles.get('offer'))):
            return ''
        pages = self.page_tools()
        bare = sorted((t for t in pages | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
        others = sorted((t for t in pages if t != facts))
        line = f'listing attributes come from {facts}'
        if bare:
            line += f"; {', '.join(bare)} show none"
        if others:
            line += f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(others)}"
        return line

    def _status_unresolved(self):
        """Requirements in force that take their value from another listing whose value is not known yet."""
        live = set()
        for _, need in self.needs_now():
            live.add(need)
        return [need for _, need in self.refs if need not in self.ref_values and need in live]

    def _status_pressure(self, left):
        if left <= self.tail() + 1:
            return ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        if left <= 6 or (self.buys == 0 and left <= len(self.items) + 5):
            return ' Few steps left: set the cart to the chosen lines and check out now.'
        if self.buys != 0:
            return ''
        if left <= 12:
            return ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        if not self.cart_touched and self.step >= self.max_steps // 2:
            return " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        return ''

    @staticmethod
    def _category_key_short(value):
        """A short text without a run of three digits: what a category label looks like."""
        return isinstance(value, str) and 0 < len(value) <= 40 and (not re.search('\\d{3}', value))

    @staticmethod
    def _key_table_row(label, name, slot):
        """(label, key, listings carrying it, up to four example values cut to 40 characters)."""
        examples = [text[:40] for text in slot['values'][:4]]
        return (label, name, slot['n'], examples)

    def _key_table_groups(self):
        """Opened listings in seen order, grouped by category label, with each attribute key's values per listing."""
        cat_key = self.category_key()
        groups = {}
        opened = [(ident, self.full[ident]) for ident in self.seen_order if self.full.get(ident)]
        for ident, record in opened:
            label = 'listings'
            if cat_key:
                label = str(record.get(cat_key) or 'listings')
            if label not in groups:
                groups[label] = {'n': 0, 'keys': {}}
            groups[label]['n'] += 1
            self._key_table_fill(groups[label]['keys'], ident, _fact_block(record) or {})
        return groups

    @staticmethod
    def _key_table_fill(keys, ident, block):
        """Add one listing's attribute values to its group, once per key name in _words form."""
        firsts: dict[str, Any] = {}
        for key, value in block.items():
            firsts.setdefault(_words(key), value)
        firsts.pop('', None)
        for name, value in firsts.items():
            entry = keys.get(name)
            if entry is None:
                entry = keys[name] = {'n': 0, 'values': [], 'per': {}}
            text = str(value).strip().lower()
            entry['per'][ident] = text
            entry['n'] += 1
            if text not in entry['values']:
                entry['values'].append(text)

    def _key_table_wanted(self):
        wanted = []
        for item in self.items:
            shelf = item.shelf or ''
            long_words = {w for w in _stem(shelf) | _stem(item.name) if len(w) >= 4}
            wanted.append((_words(shelf), _words(item.name), long_words))
        return wanted

    def _key_table_matches(self, label, wanted):
        """Whether a category label belongs to a requested item: by its words, a long shared word, or ownership."""
        plain = _words(label)
        long_words = {w for w in _stem(label) if len(w) >= 4}
        for shelf, name, words in wanted:
            if plain in (shelf, name) or long_words & words:
                return True
        return bool(self.owner_of(label))

    def _key_table_keys(self, group):
        """A group's useful keys, most carried first, each look-alike key joined into the first key it matches."""
        kept: list[tuple[str, dict[str, Any]]] = []
        ranked = sorted(group['keys'].items(), key=lambda pair: (-pair[1]['n'], len(pair[0])))
        for name, slot in ranked:
            if not slot['per'] or self._key_table_noise(list(slot['per'].values())):
                continue
            twin = self._key_table_twin(kept, name, slot)
            if twin is None:
                kept.append((name, slot))
            else:
                self._key_table_join(twin, slot)
        return sorted(kept, key=lambda pair: pair[1]['n'], reverse=True)

    def _key_table_twin(self, kept, name, slot):
        for other, spec in kept:
            if self.twins(other, spec, name, slot):
                return spec
        return None

    @staticmethod
    def _key_table_join(twin, slot):
        """Fold a look-alike key's values into the key it duplicates; the joined key counts every listing either had."""
        merged, examples = (twin['per'], twin['values'])
        for ident, text in slot['per'].items():
            if ident not in merged:
                merged[ident] = text
            if text not in examples:
                examples.append(text)
        twin['n'] = len(merged)

    def _key_table_noise(self, texts):
        """Whether a key's values are too long, mostly lists of options, or free text that never repeats a word."""
        count = len(texts)
        mean = sum(map(len, texts)) / count
        if mean > 40 or 2 * sum((' / ' in text for text in texts)) >= count:
            return True
        if count < 4 or len(set(texts)) != count or mean <= 12:
            return False
        return 2 * self._key_table_recurring(texts) < count

    @staticmethod
    def _key_table_recurring(texts):
        """How many values share a word of three or more letters with the other values."""
        count = 0
        for text in texts:
            rest = ' '.join((other for other in texts if other is not text))
            if set(re.findall('[a-z]{3,}', text)) & set(re.findall('[a-z]{3,}', rest)):
                count += 1
        return count

    @staticmethod
    def _not_stated_words(text):
        return set(re.findall('[a-z0-9]{3,}', text.lower()))

    @staticmethod
    def _price_of_number(value):
        return isinstance(value, (int, float)) and (not isinstance(value, bool))

    @staticmethod
    def _read_shape_table(value):
        return isinstance(value, list) and len(value) >= 2 and all((isinstance(row, dict) for row in value))

    @staticmethod
    def _read_shape_prompt(name, schema, observation, lists):
        """The layout question for a result, its row lists cut to their first three rows."""
        sample = dict(observation)
        for key in lists:
            sample[key] = observation[key][:3]
        shown = _dump(_clip(sample, 120))
        return f'{SHAPE}\n\nTOOL: {name} | parameters: {_dump(schema)}\n\nRESULT (first rows only):\n{shown[:4000]}'

    @staticmethod
    def _more_pages_later(given):
        """Whether the call already asked for a page past the first."""
        if isinstance(given, bool) or not isinstance(given, int):
            return bool(str(given or '').strip())
        return given not in (0, 1)

    def _more_pages_count(self, pages, room):
        """How many further pages to read on the agent's own: at most two, inside the page allowance and the step
        reserve; 0 when they cannot be read that way."""
        if pages > 2 and self.pages_read + 2 <= AUTO_PAGES and room >= 14:
            pages = 2
        allowed = pages <= 2 and self.pages_read + pages <= AUTO_PAGES and room - pages >= 12
        return pages if allowed else 0

    def _more_pages_read(self, name, args, param, token, pages):
        """Read up to `pages` further pages, one store call each; returns their texts and the token left unread."""
        read: list[str] = []
        schema = self.schemas.get(name)
        while token is not None and len(read) < pages:
            call = self._shape_call(schema, {**args, param: token})
            if call is None:
                break
            reply = self.act(name, call)
            if self.finished or reply is None:
                break
            page = reply.get('observation')
            self.pages_read += 1
            read.append(f'{param}={_dump(token)}: ' + self._more_pages_keep(name, call, page))
            following = self.paging(name, page)
            token = None if not following else following[3]
        return (read, token)

    def _more_pages_keep(self, name, call, page):
        """Take in a page read on the agent's own initiative, keep it for the read cache and return its text."""
        self.paged_obs.append(page)
        self.observe(name, page)
        text = self.show(page)
        key = name + _dump(call)
        self.read_cache[key] = (self.version, self.step, text)
        return text

    @staticmethod
    def _cap_ignored_cap(props, key, value):
        """Whether an argument is a maximum price the call set."""
        if key not in props or isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if re.search('max|most|under|below|limit|cap|upto|up_to', key, re.I) is None:
            return False
        spec = props.get(key)
        note = str(spec.get('description') or '') if isinstance(spec, dict) else ''
        return re.search('price|cost|amount|budget|spend|money', key + ' ' + note, re.I) is not None

    def _cap_ignored_caps(self, name, args):
        """The tool's argument schemas, and the maximum-price arguments of the call, lowest value first."""
        props = (self.schemas.get(name) or {}).get('properties') or {}
        return (props, sorted(filter(lambda key: self._cap_ignored_cap(props, key, args[key]), args), key=args.get))

    @staticmethod
    def _cap_ignored_hint(props, args, cap_key):
        swaps: list[str] = []
        for key, value in args.items():
            if not isinstance(value, str):
                continue
            options = (props.get(key) or {}).get('enum')
            if not isinstance(options, list):
                continue
            others = [option for option in options if option != value]
            if others:
                swaps.append(f'{key}={_dump(others[0])}')
        if swaps:
            return '; list the category with ' + ', '.join(swaps) + f' and the same {cap_key} to see only cheaper listings'
        return ''

    def _fill_facts_route(self):
        """(the tool that shows one listing's attributes, its id argument), or None when there is no such tool."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        return (tool, target) if tool and target else None

    def _fill_facts_queue(self, ids):
        """The ids seen as rows whose pages were never read, sold-out ones left out, fixed before any call is made."""
        return [ident for ident in ids if ident in self.rows and ident not in self.full and (not self.sold_out(ident))]

    def _fill_facts_open(self, tool, target, ident):
        """Open one listing's page and learn whether the tool shows attributes; returns the page text, or None when the
        call could not be made or got no reply."""
        call = self._shape_call(self.schemas.get(tool), {target: ident})
        reply = None if call is None else self.act(tool, call)
        if reply is None:
            return None
        page = reply.get('observation')
        self._fill_facts_seen(tool, page)
        if not self.facts_from.get(tool, False) and isinstance(page, dict) and self.ident(page):
            self.facts_from[tool] = bool(_fact_block(page))
        return self.show(page)

    def _fill_facts_seen(self, tool, page):
        """Take in a page the agent opened on its own and note which keys that tool showed for each listing."""
        self.observe(tool, page)
        self.note_view(tool, page)

    def _show_links_follow(self, plan, ident, code, keep):
        """Search one referred part code and open the listings that carry it; False when no further code may be run."""
        if self.finished or self.left() <= keep:
            return False
        self.link_tried.add(code)
        tool, base, field = plan
        call = self._shape_call(self.schemas.get(tool), {**base, field: code})
        reply = None if call is None else self.act(tool, call)
        if reply is None:
            return False
        self.observe(tool, reply.get('observation'))
        self.trace.append({'kind': 'show_link', 'line': ident, 'code': code})
        owners = [other for other, mark in self.fit_index()['marks'].items() if mark.get('own') == code]
        self.fill_facts(owners, max(keep, self.tail() + 6))
        return True

    def _show_links_code(self, ident):
        """The part code a listing refers the reader to, when it is new and no seen listing carries it; else ''."""
        marks = self.fit_index()['marks']
        mark = marks.get(ident) or {}
        code = mark.get('more')
        if code and code not in self.link_tried:
            carried = [mark.get('own') for mark in marks.values()]
            return '' if code in carried else code
        return ''

    @staticmethod
    def _cap_template_fields(props):
        caps: list[str] = []
        shelves: list[str] = []
        for key, spec in props.items():
            if not isinstance(spec, dict):
                continue
            kinds = set(_kinds(spec))
            if kinds & {'integer', 'number'} and re.search(CAP_ARG, _snake(key), re.I) and (not re.search(CAP_NOT, key, re.I)):
                caps.append(key)
            if 'string' in kinds and 'enum' not in spec and re.search(SHELF_ARG, _snake(key), re.I):
                shelves.append(key)
        return (caps, shelves)

    @staticmethod
    def _cap_template_options(props):
        """(argument, options) for each argument of a schema that takes one of two or more fixed options."""
        return [(key, spec['enum']) for key, spec in props.items() if isinstance(spec, dict) and isinstance(spec.get('enum'), list) and len(spec['enum']) >= 2]

    def _cap_template_modes(self, props, shelf):
        """Each mode argument set as an earlier listing by shelf set it, else to its one option that names a listing;
        None when a mode stays unclear."""
        base: dict[str, Any] = {}
        for key, options in self._cap_template_options(props):
            choice = self._cap_template_used(props, shelf, key)
            if choice is None:
                listing = [option for option in options if isinstance(option, str) and re.search(LIST_MODE, option, re.I)]
                if len(listing) != 1:
                    return None
                choice = listing[0]
            base[key] = choice
        return base

    def _cap_template_used(self, props, shelf, key):
        """The value of `key` in the latest search that listed a shelf without query words, else None."""
        for args in reversed(self.searched):
            if shelf in args and key in args and (not self._cap_template_worded(props, shelf, args)):
                return args[key]
        return None

    @staticmethod
    def _cap_template_worded(props, shelf, args):
        for name, value in args.items():
            if name == shelf:
                continue
            spec = props.get(name)
            if isinstance(spec, dict) and 'enum' in spec:
                continue
            if isinstance(value, str) and value.strip() and re.search(QUERY_WORDS, _snake(name), re.I):
                return True
        return False

    @staticmethod
    def _shape_call(schema, args):
        """Store call arguments fitted to a tool's schema; None if no fitting form exists."""
        problems = []
        fitted = _shape(args, schema, 'args', problems)
        if problems:
            return None
        return fitted

    def _widen_batch(self, tool, target):
        """One side-by-side call over unread listings of each group; single opens instead when it shows no attributes,
        a smaller call when it opened none."""
        before = len(self.full)
        schema = self.schemas.get(tool) or {}
        limit = schema['properties'][target].get('maxItems')
        picks = self._widen_picks(limit if isinstance(limit, int) else 4)
        call = self._shape_call(schema, {target: picks}) if len(picks) >= 2 else None
        reply = None if call is None else self.act(tool, call)
        if reply is None:
            return
        rows = self._widen_absorb(tool, reply.get('observation'))
        gained = len(self.full) - before
        self.trace.append({'kind': 'widen', 'ids': picks, 'opened': gained})
        if rows and self.facts_from.get(tool) is False:
            self.widen_singly()
        elif gained == 0 and len(picks) > 2 and self.left() > 12:
            self._widen_retry(tool, schema, {target: picks[:2]})

    def _widen_absorb(self, tool, page):
        """Take in a side-by-side page and learn whether that tool shows attributes; returns the records it held."""
        self._fill_facts_seen(tool, page)
        rows = self.records_in(page)
        if rows and (not self.facts_from.get(tool, False)):
            self.facts_from[tool] = any((_fact_block(record) for _, record in rows))
        return rows

    def _widen_retry(self, tool, schema, args):
        """One more side-by-side call with fewer listings, after a call that opened none."""
        call = self._shape_call(schema, args)
        if call is None:
            return
        reply = self.act(tool, call)
        if reply is not None:
            self.observe(tool, reply.get('observation'))

    def _widen_picks(self, limit):
        groups = self.item_groups()
        per_group = max(1, limit // (len(groups) or 1))
        unread = ([ident for ident in ids if ident not in self.full] for ids in groups)
        chosen = [ident for lane in unread for ident in lane[:per_group]]
        return chosen[:limit]

    def _widen_singly_price(self, ident):
        price = self.price_of(ident)
        return float('inf') if price is None else price

    def _widen_singly_overlap(self, ident, wanted):
        """How many requirement words of three or more characters a listing's row carries."""
        text = _dump(self.rows.get(ident) or {}).lower()
        return len(wanted & set(re.findall('[a-z0-9]{3,}', text)))

    def _widen_singly_lane(self, ids, wanted):
        """A group's unread buyable listings, those carrying more requirement words first (cheaper first among equals
        on a cheapest pick)."""
        pool = [ident for ident in ids if ident not in self.full and self.buyable(ident)]
        if self.brief.get('pick') == 'cheapest':
            pool = sorted(pool, key=self._widen_singly_price)
        return sorted(pool, key=lambda ident: self._widen_singly_overlap(ident, wanted), reverse=True)

    @staticmethod
    def _search_template_texts(props):
        """The free-text arguments of a tool, and those of them whose names say they take a query."""
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        return (texts, [k for k in texts if re.search(QUERY_WORDS, _snake(k), re.I)])

    def _search_template_fields(self, args, candidates, named):
        fields = [k for k in candidates if self._search_template_lettered(args.get(k))]
        if named:
            return fields
        cat_key = self.category_key()
        labels = set()
        if cat_key:
            for row in self.rows.values():
                labels.add(str(row.get(cat_key) or '').strip().lower())
            labels.discard('')
        return [k for k in fields if self._search_template_free(k, labels)]

    @staticmethod
    def _search_template_lettered(value):
        return isinstance(value, str) and re.search('[a-z]', value, re.I) is not None

    def _search_template_free(self, key, labels):
        """Whether a text argument held varied, multi-word text that was never a category label."""
        said = [args.get(key) for args in self.searched]
        if len({str(value) for value in said if value}) < 2:
            return False
        if all((len(str(value).split()) < 2 for value in said)):
            return False
        return all((str(value or '').strip().lower() not in labels for value in said))

    @staticmethod
    def _search_template_names(value, names):
        if _words(value) in names:
            return True
        own = {w for w in _stem(value) if len(w) >= 4}
        return any((own & {w for w in _stem(name) if len(w) >= 4} for name in names if name))

    def _search_template_fixed(self, args, props, schema, field):
        """The arguments of a past search to keep beside new words: required ones, and mode options that name no
        category, each mode set to the option most searches with words used."""
        required = schema.get('required') or []
        names = self._search_template_labels()
        kept: dict[str, Any] = {}
        moded: list[str] = []
        for key, value in args.items():
            if key == field:
                continue
            spec = props.get(key) or {}
            options = isinstance(spec.get('enum'), list)
            if key in required or (options and (not isinstance(value, str) or not self._search_template_names(value, names))):
                kept[key] = value
                if options:
                    moded.append(key)
        worded = [a for a in self.searched if self._search_template_lettered(a.get(field))]
        for key in moded:
            usual = self._search_template_usual(worded, key)
            if usual:
                kept[key] = usual[0]
        return kept

    def _search_template_labels(self):
        """Item names, store labels and seen category labels, in _words form."""
        names: set[str] = set()
        for item in self.items:
            names.add(_words(item.shelf or ''))
            names.add(_words(item.name))
        cat_key = self.category_key()
        if cat_key:
            names.update((_words(str(row.get(cat_key) or '')) for row in self.rows.values()))
        return names

    @staticmethod
    def _search_template_usual(worded, key):
        """[the value of `key` most searches used, first seen form], or [] when none set it."""
        counts: dict[str, int] = {}
        first_form: dict[str, Any] = {}
        for args in worded:
            if key not in args:
                continue
            text = _dump(args[key])
            if text not in counts:
                counts[text] = 0
                first_form[text] = args[key]
            counts[text] += 1
        if not counts:
            return []
        return [first_form[max(counts, key=counts.get)]]

    @staticmethod
    def _value_template_mode(key, options, used, needed):
        """The mode option that reads words: the one option naming a query, else, for a required two-option mode, the
        option the last search did not use; [] when neither is clear."""
        reads = [o for o in options if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)]
        if len(reads) == 1:
            return [(key, reads[0])]
        others = [o for o in options if o != used]
        if needed and len(options) == 2 and len(others) == 1:
            return [(key, others[0])]
        return []

    @staticmethod
    def _value_search_short(value):
        """A value a words search reads as it is: one word, or an amount with its unit."""
        return len(value.split()) == 1 or re.fullmatch(AMOUNT, value, re.I) is not None

    def _value_search_tries(self, number):
        """The word lists to search for an item: its values with one-word and amount values first; then the same
        without the last value the request stated (or with the item's name added to a single value)."""
        pairs = self.value_words(number)
        fresh = [value for value, told in pairs if told]
        if not fresh:
            return []
        lead = [value for value in fresh if self._value_search_short(value)]
        ordered = lead + [value for value, _ in pairs if value not in lead]
        if len(ordered) > 1:
            cut = 0
            for index, value in enumerate(ordered):
                if value not in fresh:
                    cut = index
            return [ordered, ordered[:cut] + ordered[cut + 1:]]
        if lead:
            return [[ordered[0], self.items[number - 1].name], ordered]
        return [ordered]

    def _value_search_item(self, number, plan, tries, calls_left, lines):
        """Run an item's searches until one finds rows of its kind, open the first unread ones, note what was seen in
        `lines`; returns the calls left."""
        tool, base, field = plan
        for words in filter(None, tries):
            if self.finished:
                break
            call = self._shape_call(self.schemas.get(tool), {**base, field: ' '.join(words)})
            if call is None:
                break
            calls_left -= 1
            reply = self.act(tool, call)
            if reply is None:
                break
            rows = self._value_search_rows(number, tool, reply.get('observation'))
            listed = ', '.join(rows[:6]) or 'no row of that kind'
            lines.append(f'{_dump(call.get(field))}: {listed}')
            if rows:
                unread = [ident for ident in rows if ident not in self.full]
                opened = self.fill_facts(unread[:min(3, calls_left)], self.tail() + 16)
                if opened:
                    joined = ' | '.join(opened)
                    lines.append('opened: ' + joined[:2400])
                return calls_left - len(opened)
        return calls_left

    def _value_search_rows(self, number, tool, observation):
        """The buyable rows of a value search that belong to the item and fit the budget, own label first."""
        self.observe(tool, observation)
        found = [pair[0] for pair in self.records_in(observation)]
        cap = self._value_search_cap(found)
        cat_key = self.category_key()
        item = self.items[number - 1]
        rows = [ident for ident in found if self._value_search_fits(ident, item, cat_key, cap)]
        home = self.home_label(number) if cat_key else ''
        if home:
            rows.sort(key=lambda ident: _words(self._value_search_label(ident, cat_key)) != home)
        return rows

    def _value_search_label(self, ident, cat_key):
        return str((self.rows.get(ident) or {}).get(cat_key) or '')

    def _value_search_fits(self, ident, item, cat_key, cap):
        """A row worth opening: buyable, of the item's own kind, and inside the price cap."""
        if not self.buyable(ident):
            return False
        if cat_key and (not self.owns(item, self._value_search_label(ident, cat_key))):
            return False
        return cap is None or (self.price_of(ident) or 0) <= cap

    def _value_search_cap(self, found):
        """The budget as a price cap for rows, or None: no numeric budget, or every row costs over ten times it (a
        budget in another unit)."""
        if not isinstance(self.budget, (int, float)):
            return None
        cap = self.budget
        prices = [self.price_of(ident) for ident in found]
        if prices and all((price is not None and price > 10 * cap for price in prices)):
            return None
        return cap

    def _home_label_tally(self, item, pool, cat_key):
        tally: dict[str, int] = {}
        for record in pool.values():
            label = str(record.get(cat_key) or '')
            if not label or not self.owns(item, label):
                continue
            plain = _words(label)
            tally[plain] = tally.get(plain, 0) + 1
        return tally

    def _trail_log(self, kind: str, **fields: Any) -> None:
        """Append one trace entry: the kind first, then the fields in the order given."""
        entry = {'kind': kind}
        entry.update(fields)
        self.trace.append(entry)

    def _take_heard_number(self, entry: dict[str, Any]) -> int:
        wanted = _count(entry.get('item')) or 1
        return min(len(self.items), max(wanted, 1))

    def _take_heard_need(self, entry: dict[str, Any], told: list[str], several: bool) -> None:
        need = str(entry.get('need') or '').strip()
        if not need:
            return
        slot = self._take_heard_number(entry)
        owner = self.items[slot - 1]
        if need not in owner.told:
            owner.told.append(need)
            told.append(f'item {slot}: {need}' if several else need)
        if entry.get('points') is True and (slot, need) not in self.refs:
            self.refs.append((slot, need))

    def _take_heard_drop(self, entry: dict[str, Any], told: list[str]) -> None:
        owner = self.items[self._take_heard_number(entry) - 1]
        head = _words(str(entry['property']).split(':')[0])
        matched = [live for live in self._needs_now_live(owner) if _words(live.split(':')[0]) == head]
        for need in matched:
            owner.dropped.append(need)
            told.append(f'dropped {need}')

    def _gate_menu(self, ending: set[str]) -> list[str]:
        menu = []
        for tool in self.tools:
            spec = tool['function']
            if spec['name'] in ending:
                continue
            menu.append(f"- {spec['name']}: {spec.get('description', '')} | parameters: {_dump(spec.get('parameters') or {})}")
        return menu

    def _gate_prompt(self, stage: str, due: list[dict[str, Any]], menu: list[str]) -> str:
        ordering = stage == 'order'
        record = '\n'.join(f'{position}. {line}' for position, line in enumerate(self.done_log, 1))
        notes = '\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else ''
        room = f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {'order' if ordering else 'ending'} itself takes one)"
        pieces = [GATE.replace('STAGE', 'place the order' if ordering else 'end without a purchase'),
                  '\n\nREQUEST:\n', self.request,
                  '\n\nSTEPS DUE NOW:\n', '\n'.join(f"- {entry['step']}" for entry in due),
                  '\n\nLEDGER:\n', self.ledger(),
                  '\n\nCART NOW: ', self.cart_text(),
                  '\n\nRECORD (messages and step calls so far):\n', record or '(nothing yet)',
                  notes, room,
                  '\n\nTOOLS:\n', '\n'.join(menu)]
        return ''.join(pieces)

    def _gate_unproven(self, stage: str, due: list[dict[str, Any]], got: dict[str, Any]) -> list[dict[str, Any]]:
        everything = range(1, len(self.done_log) + 1)
        if stage == 'order' and any(entry['messages'] > 0 for entry in due):
            steps = self.done_steps
            proven = {position for position in everything if position <= len(steps) and steps[position - 1] > self.cart_at}
        else:
            proven = set(everything)
        return [entry for entry in _items(got.get('steps')) if isinstance(entry, dict) and not self._gate_step_done(entry, proven)]

    def _gate_press(self, missing: list[dict[str, Any]]) -> None:
        self._trail_log('gate_unproven', steps=_clip(missing, 120))
        named = '; '.join(str(entry.get('step') or '') for entry in missing)[:200]
        self.done_log.append(f'(Nimbus: no line made since the cart last changed at step {self.cart_at} shows ' + named + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
        self.done_steps.append(-1)

    def _gate_follow(self, name: str, args: dict[str, Any], done_text: list[str]) -> bool:
        if name == self.roles.get('ask'):
            return self._gate_ask(args, done_text)
        reply = self.act(name, args)
        if reply is None:
            return False
        seen = reply.get('observation')
        self.absorb(seen)
        sent, page = _dump(args), self.show(seen)
        line = f'step {self.step}: {name} {sent} -> {page[:300]} (cart then: {self.cart_mark()})'
        self.done_log.append(line)
        self.done_steps.append(self.step)
        done_text.append(f'{name} {sent} -> {page[:400]}')
        return True

    def _spoken_phrases(self, observation: Any) -> list[str]:
        phrases = []
        waiting = [(observation, 0)]
        while waiting:
            node, depth = waiting.pop()
            if depth > 4:
                continue
            if isinstance(node, str):
                if len(node.split()) >= 2:
                    phrases.append(node)
                continue
            if isinstance(node, dict):
                children = list(node.values())
            elif isinstance(node, list):
                children = node
            else:
                continue
            waiting.extend((child, depth + 1) for child in reversed(children))
        return phrases

    def _loose_lines_of(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if len(value.split()) >= 5 else []
        if isinstance(value, list):
            if not value or not all(isinstance(entry, str) for entry in value):
                return []
            return [entry for entry in value if len(entry.split()) >= 5]
        if isinstance(value, dict) and not self.ident(value, False, 1):
            return [inner for inner in value.values() if isinstance(inner, str) and len(inner.split()) >= 6]
        return []

    def _take_note_gone(self, got: dict[str, Any], lines: list[str]) -> list[str]:
        gone = []
        for entry in _items(got.get('gone')):
            if isinstance(entry, (str, int)):
                gone.append(str(entry))
        known = [ident for ident in gone if ident in self.rows]
        self.unavailable.update(known)
        self.note_named = bool(known)
        self.note_sold = bool(gone) or got.get('sold_out') is True
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        return gone

    def _take_note_entries(self, raw: Any) -> list[dict[str, Any]]:
        return [entry for entry in _items(raw) if isinstance(entry, dict) and str(entry.get('need') or '').strip()]

    def _take_note_drop(self, entry: dict[str, Any]) -> str:
        owner = self.items[self._take_heard_number(entry) - 1]
        target = str(entry['need']).strip()
        live = self._needs_now_live(owner)
        if target in live:
            hit = [need for need in live if need == target]
        else:
            head = _words(target.split(':')[0])
            hit = [need for need in live if _words(need.split(':')[0]) == head][:1]
        owner.dropped.extend(hit or [target])
        return f'dropped {target}'

    def _take_note_add(self, entry: dict[str, Any]) -> str:
        owner = self.items[self._take_heard_number(entry) - 1]
        need = str(entry['need']).strip()
        if need in owner.dropped:
            owner.dropped.remove(need)
        if need not in owner.musts + owner.told:
            owner.told.append(need)
        return f'added {need}'

    def _needs_now_live(self, item: Any) -> list[str]:
        return [need for need in item.musts + item.told if need not in item.dropped]

    def _evidence_ids(self, observation: Any) -> list[str]:
        found = []
        waiting = [(observation, 0)]
        while waiting:
            node, depth = waiting.pop()
            if depth > 6:
                continue
            if isinstance(node, dict):
                ident = self.ident(node, False, depth)
                if ident and _fact_block(node) and ident not in found:
                    found.append(ident)
                    continue
                children = list(node.values())
            elif isinstance(node, list):
                children = node
            else:
                continue
            waiting.extend((child, depth + 1) for child in reversed(children))
        return found

    def _evidence_line(self, ident: str, needs: list[tuple[int, str]], several: bool) -> str:
        block = _fact_block(self.full.get(ident) or {}) or {}
        shown = []
        for position, need in needs:
            prop = need.split(':', 1)[0].strip()
            wanted = _stem(prop)
            keys = [f'{key}={str(value)[:50]}' for key, value in block.items() if _names(wanted, _stem(key))]
            if not keys:
                continue
            label = f'(item {position}) ' if several else ''
            shown.append(label + prop + ' -> ' + ', '.join(keys[:3]))
        return f'{ident}: ' + '; '.join(shown) if shown else ''

    def _cart_ids_view_current(self) -> bool:
        return self.cart_view is not None and self.cart_seen_at >= self.cart_at

    def _held_units_numbers(self, node: dict[str, Any]) -> list[int]:
        numbers = []
        for key, value in node.items():
            if isinstance(value, str):
                continue
            amount = _count(value)
            if amount is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _words(key)):
                numbers.append(amount)
        return numbers

    def _others_rank(self, ident: str) -> tuple[bool, float]:
        price = self.price_of(ident)
        return (ident not in self.full, float('inf') if price is None else price)

    def _fit_index_text(self, record: dict[str, Any]) -> str:
        plain = [str(value) for value in record.values() if isinstance(value, str)]
        pairs = []
        for block in record.values():
            if isinstance(block, dict):
                pairs.extend(f'{key} {value}' for key, value in block.items() if isinstance(value, str))
        return ' '.join(plain) + ' ' + '. '.join(pairs)

    def _item_ids_label(self, ident: str, cat_key: str) -> str:
        return str((self.rows.get(ident) or {}).get(cat_key) or '')

    def _fit_faults_pair(self, part: int, main: int, cart: list[str], index: dict[str, Any]) -> list[str]:
        if not cart:
            return []
        device_ids = self.item_ids(main)
        devices = [ident for ident in cart if ident in device_ids]
        part_ids = self.item_ids(part)
        held_parts = [ident for ident in cart if ident in part_ids]
        if len(devices) != 1 or not held_parts:
            return []
        device = devices[0]
        codes = set(index['marks'].get(device, {}).get('codes') or [])
        if not codes:
            return []
        out = []
        for line in held_parts:
            fits = self.fit_set(line, index)
            if not fits or fits & codes:
                continue
            good = [ident for ident in part_ids if self.fit_set(ident, index) & codes and not self.sold_out(ident) and ident not in cart][:4]
            if good:
                tail = f"; parts of that kind already seen that suit it: {', '.join(good)}"
            else:
                tail = '; no part of that kind seen so far suits it, so search for one or change the device: ' + self.fit_plan(short=True)
            out.append(f"{line} ({self.items[part - 1].name}) works with {', '.join(sorted(fits)[:4])}, not with {device} ({', '.join(sorted(codes))})" + tail)
        return out

    def _fit_plan_codes(self, main: int, index: dict[str, Any]) -> str:
        ranked = sorted(self._fit_plan_tally(main, index).items(), key=self._fit_plan_rank)[:4]
        pieces = []
        for code, per in ranked:
            owners = ', '.join(f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))
            seen = self._fit_plan_devices(main, code, index)
            where = f" (device seen: {', '.join(seen)})" if seen else ' (device not seen yet)'
            pieces.append(f'{code}: {owners}{where}')
        return '; '.join(pieces)

    def _missing_items_held(self) -> list[str]:
        return [ident for ident in self.cart_ids() if (self.held_units(ident) or 0) > 0]

    def _states_needs_words(self, text: str) -> set[str]:
        return set(re.sub('[,/;|]', ' ', _words(text)).split())

    def _states_needs_one(self, block: dict[str, Any], need: str) -> bool:
        prop, _, value = need.partition(':')
        words = self._states_needs_words(value)
        if not words or words & {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'}:
            return False
        name = _spelled(prop)
        return any(_names(name, _spelled(key)) and words <= self._states_needs_words(str(shown)) for key, shown in block.items())

    def _fill_short_cart_pick(self, number: int, item: Any) -> str | None:
        fitting = [ident for ident in self.item_ids(number) if ident in self.full and ident not in self.cart_lines and self.buyable(ident) and self.price_of(ident) is not None and self.states_needs(ident, number)]
        if not fitting:
            return None
        pick = sorted(fitting, key=lambda ident: self.price_of(ident) or 0.0)[0]
        if not isinstance(self.budget, (int, float)):
            return pick
        each = self.brief.get('budget_scope') == 'each'
        held = sum((self.price_of(ident) or 0.0) * (self.held_units(ident) or 0) for ident in self.cart_ids())
        cost = (self.price_of(pick) or 0.0) * (1 if each else item.qty)
        if cost + (0.0 if each else held) > self.budget + 1e-09:
            return None
        return pick

    def _fill_short_cart_send(self, cart: str, calls: list[dict[str, Any]]) -> bool:
        schema = self.schemas.get(cart)
        for raw in calls:
            args = self._open_rows_shaped(schema, raw)
            if args is None or self.act(cart, args) is None:
                return False
            self.note_cart(cart, args)
        return True

    def _line_faults_owners(self, label: str) -> list[int]:
        return [position for position, item in enumerate(self.items, 1) if self.owns(item, label)]

    def _line_faults_item(self, position: int, item: Any, ids: list[str], label: dict[str, str], owner: dict[str, list[int]]) -> list[str]:
        mine = [ident for ident in ids if position in owner[ident]]
        sole = [ident for ident in mine if owner[ident] == [position]]
        out = []
        if not mine:
            out.append(f'item {position} ({item.name}) has no line in the cart')
        elif len(sole) > 1:
            out.append(f"item {position} ({item.name}) has {len(mine)} lines ({', '.join(mine)}); keep one")
        for ident in mine:
            have = self.held_units(ident)
            fuller = self.fuller_label(position, label[ident])
            out += [f'{ident} is held at {have} unit(s); item {position} ({item.name}) wants {item.qty}'] if ident in sole and have != item.qty else []
            out += [f'{ident} is from {label[ident]}, yet the store also uses the label {fuller}, which spells out item {position} ({item.name}) fully: pick the line from that label'] if fuller else []
        return out

    def _line_faults_budget(self, ids: list[str]) -> list[str]:
        prices = {ident: self.price_of(ident) for ident in ids}
        if not isinstance(self.budget, (int, float)) or any(price is None for price in prices.values()):
            return []
        limit = self.budget
        if self.brief.get('budget_scope') == 'each':
            return [f'{ident} costs {prices[ident]:g}, over the {limit:g} limit for each item' for ident in ids if limit < prices[ident] <= 10 * limit]
        total = sum(prices[ident] * self.held_units(ident) for ident in ids)
        if limit + 1e-09 < total <= 10 * limit:
            return [f'the total is {total:g}, over the budget of {limit:g}']
        return []

    def _fix_counts_plan(self, name: str, wrong: list[tuple[str, int, int]], props: dict[str, Any], target: str, count: str, mode: str | None) -> list[dict[str, Any]] | None:
        entries = [(ident, have, want) for ident, have, want in wrong]
        if not entries:
            return []
        if self.cart_sets(name):
            return [{target: ident, count: want} for ident, have, want in entries]
        if not mode:
            return None
        options = list(map(str, props[mode]['enum']))
        off = options[1]
        for option in options:
            if re.search('remov|delet|drop|clear|take', option, re.I):
                off = option
                break
        on = options[0]
        for option in options:
            if option != off:
                on = option
                break
        planned = []
        for ident, have, want in entries:
            planned.append({target: ident, mode: off, count: have})
            planned.append({target: ident, mode: on, count: want})
        return planned

    def _check_cart_prompt(self, ids: list[str], lead: bool) -> str:
        parts = [CHECK, '\n\nREQUEST:\n', self.request, '\n\nREQUIREMENTS IN FORCE:\n', self.ledger()]
        if lead:
            quantity = '; '.join(f'item {position}: quantity {item.qty}' for position, item in enumerate(self.items, 1))
            parts += ['\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ', quantity, '):\n', ', '.join(ids)]
        else:
            parts += ['\n\nCART:\n', self.cart_text()]
        records = []
        for ident in ids:
            flag = 'SOLD OUT (the store says it can no longer be bought). ' if self.sold_out(ident) else ''
            records.append(f'{ident}: ' + flag + self.record_of(ident))
        parts += ["\n\nRECORDS OF THE CART'S LISTINGS:\n", '\n'.join(records)]
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            parts += ['\n\nOTHER LISTINGS SEEN (newest first):\n', self.others(ids, 15)]
        return ''.join(parts)

    def _check_cart_verdict(self, prompt: str, lead: bool, limit: float) -> dict[str, Any] | None:
        judges = self.judges(limit)
        got = None
        quick = len(judges) > 1 and not lead and not self.audited and self.brief.get('pick') not in ('cheapest', 'most', 'least')
        if quick:
            got = self.ask_json(prompt, [LIGHT], 3000)
            if got:
                second = got.get('ok') is False and HEAVY in self.judges(limit)
                judges = [HEAVY] if second else []
        if not judges or self.no_credit:
            return got
        return self.ask_json(prompt, judges, 3000) or got

    def _trail_records(self) -> list[str]:
        newest = self.seen_order[::-1]
        readable = [ident for ident in newest if ident in self.opened and (self.full.get(ident) or self.rows.get(ident))]
        records = ['OPENED ' + _dump(_clip(self.full.get(ident) or self.rows.get(ident), 300))[:1400] for ident in readable[:45]]
        skipped = set(readable)
        rest = [ident for ident in newest if ident not in skipped]
        records += ['ROW ' + _dump(_clip(self.rows.get(ident), 80))[:300] for ident in rest[:max(0, 60 - len(records))]]
        return records

    def _trail_ask(self, pending: list[str]) -> dict[str, Any]:
        records = self._trail_records()
        prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending) + '\n\nRECORDS (newest first):\n' + '\n'.join(records)
        models = [LIGHT, HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else [LIGHT]
        got = self.ask_json(prompt, models, 4500) or {}
        self._trail_log('trail', pending=pending, got=_clip(got, 150))
        return got

    def _trail_need(self, chain: dict[str, Any], pending: list[str]) -> str | None:
        named = str(chain.get('need') or '').strip()
        match = next((need for need in pending if need == named), None)
        if match:
            return match
        return pending[0] if len(pending) == 1 else None

    def _trail_settle(self, chain: dict[str, Any], need: str, links: list[dict[str, Any]], ids: list[str]) -> bool:
        if not links or not all(ident in self.opened for ident in ids):
            return False
        value = str(chain.get('value') or '').strip()
        if chain.get('done') is True and value:
            self.ref_values[need] = value
        elif not str(links[-1].get('find') or '').strip():
            self.ref_values[need] = value or f'as stated on {ids[-1]}'
        else:
            return False
        self.version += 1
        return True

    def _trail_chains(self, got: dict[str, Any], pending: list[str]) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        reach, lost, gaps = [], [], []
        chains = [chain for chain in _items(got.get('chains')) if isinstance(chain, dict)]
        for chain in chains:
            need = self._trail_need(chain, pending)
            if not need:
                continue
            links, ids = self._trail_links(chain)
            if self._trail_settle(chain, need, links, ids):
                continue
            closed = [ident for ident in ids if ident in self.rows and ident not in self.opened]
            if closed:
                reach += [ident for ident in closed if ident not in reach]
                continue
            about, words = self._trail_gap(links, ids)
            gaps.append((about, words))
            lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
        return reach, lost, gaps

    def _trail_open(self, tool: str, target: str, ids: list[str], opened_text: list[str]) -> None:
        schema = self.schemas.get(tool)
        for ident in ids:
            args = self._open_rows_shaped(schema, {target: ident})
            reply = None if args is None else self.act(tool, args)
            if reply is None or self.finished:
                return
            page = reply.get('observation')
            self.observe(tool, page)
            opened_text.append(f'{ident}: {self.show(page)[:1500]}')

    def _trail_report(self, opened_text: list[str], lost: list[str]) -> tuple[str, bool]:
        parts = []
        if opened_text:
            parts.append('[Nimbus opened the pages a requirement points at]\n' + '\n'.join(opened_text) + '\n')
        if self.ref_values:
            parts.append('[values taken from the pages requirements refer to] ' + '; '.join(f'{need} -> {value}' for need, value in self.ref_values.items()) + '\n')
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and self.left() > 5
        if hold:
            self.trail_blocks += 1
            parts.append('Checkout waits until each page a requirement refers to is open in full (a result row does not count), including any further listing such a page sends you to. Not found and opened yet: ' + ' | '.join(lost[:3]) + ". Search with the shop's name plus the values from its description (other listings of that kind from the same shop do not count), open that shop's row whose page carries those values, use the value it states, then check out.")
        return (''.join(parts), hold)

    def _shop_rows_names(self, row: dict[str, Any], floor: int=2) -> list[str]:
        names = []
        for key, value in row.items():
            if not isinstance(value, str):
                continue
            cleaned = value.strip()
            if floor < len(cleaned) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(key), re.I):
                names.append(cleaned)
        return names

    def _shop_rows_named(self, row: dict[str, Any], text: str) -> bool:
        return any(re.search('(?<![a-z0-9])' + re.escape(name.lower()) + '(?![a-z0-9])', text) for name in self._shop_rows_names(row))

    def _find_link_shop_words(self, owners: list[str]) -> set[str]:
        found = set()
        for ident in owners:
            for name in self._shop_rows_names(self.rows.get(ident) or {}, -1):
                found.update(re.findall('[a-z0-9]{3,}', name.lower()))
        return found

    def _find_link_shows(self, ident: str, values: set[str]) -> bool:
        waiting = [self.full.get(ident) or self.rows.get(ident) or {}]
        leaves = []
        while waiting:
            node = waiting.pop()
            if isinstance(node, dict):
                waiting.extend(node.values())
            elif isinstance(node, list):
                waiting.extend(node)
            else:
                leaves.append(str(node))
        text = ' '.join(leaves).lower()
        return all(re.search('(?<![a-z0-9])' + re.escape(value) + '(?![a-z0-9])', text) for value in values)

    def _find_link_run(self, plan: tuple[str, dict[str, Any], str], words: str) -> list[str] | None:
        """Ids of the rows one search for words returns; None when no search went out."""
        tool, base, field = plan
        request = dict(base)
        request[field] = words
        args = self._open_rows_shaped(self.schemas.get(tool), request)
        if args is None:
            return None
        self.link_searches += 1
        reply = self.act(tool, args)
        if reply is None:
            return None
        result = reply.get('observation')
        self.observe(tool, result)
        return [ident for ident, _ in self.records_in(result)]

    def _find_link_search(self, about: str, words: str) -> list[str]:
        plan = self.search_template()
        memo = ' '.join(words.lower().split())
        cached = self.link_seen.get(memo) if memo else None
        if cached is not None:
            self.link_searches += 1
            return self.shop_rows(about, cached)
        found = self._find_link_run(plan, words) if plan and words else None
        if found is None:
            return []
        self.link_seen[memo] = found
        known = self.shop_rows(about, found)
        self._trail_log('link_search', words=words[:80], rows=known[:4])
        return known

    def _find_link_of_kind(self, ids: list[str], kind: set[str]) -> list[str]:
        if not kind:
            return list(ids)
        return [ident for ident in ids if kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(ident) or {}).lower()))]

    def _reread_offers_one(self, tool: str, target: str, ident: str) -> str | None:
        args = self._open_rows_shaped(self.schemas.get(tool), {target: ident})
        if args is None:
            return None
        before = dict(self.rows.get(ident) or {})
        reply = self.act(tool, args)
        if reply is None or self.finished:
            return None
        page = reply.get('observation')
        self.offer_seen[ident] = self.step
        self.observe(tool, page)
        moved = self._reread_offers_moved(before, self.rows.get(ident) or {})
        if moved:
            return f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(page, 300))[:900]}"
        return ''

    def _swap_hint_advice(self, picks: list[str]) -> str:
        if not picks:
            return ' Remove that line and cart the strongest remaining candidate that fits; if nothing read so far fits, search for one.'
        if self.left() <= 6:
            return f" Steps are short: this turn, put {picks[0]} in at the quantity asked for and take the unavailable line out (two calls), then check out. Other read listings that look like fits: {', '.join(picks[1:]) or 'none'}."
        if self.brief.get('pick') == 'cheapest':
            return f" Read listings that look like fits for that item, lowest price first: {', '.join(picks)}. The shopper wants the lowest price, so unless the whole fitting set is already known, search once below {picks[0]}'s price (with the search's price cap when it offers one) before replacing the unavailable line with the cheapest fit."
        return f" Read listings that look like fits for that item: {', '.join(picks)}. This turn, take the unavailable line out of the cart and put {picks[0]} in at the quantity asked for, then check out."

    def _replacements_records(self, lost: list[str]) -> list[str]:
        records = []
        for ident in self.seen_order[::-1]:
            if ident in lost or self.sold_out(ident):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        return records

    def _replacements_fits(self, got: dict[str, Any] | None, lost: list[str]) -> list[tuple[float, str]]:
        fits = []
        taken = []
        for entry in _items((got or {}).get('fits')):
            if isinstance(entry, dict):
                ident, price = str(entry.get('id')), entry.get('price')
            else:
                ident, price = str(entry), None
            if ident not in self.rows or ident in lost or ident in self.gone or ident in self.cart_ids() or ident in taken:
                continue
            taken.append(ident)
            fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float('inf'), ident))
        return fits

    def _audit_fits_ids(self, value: Any) -> list[str]:
        ids = []
        for raw in _items(value):
            ident = str(raw)
            if ident in self.rows and not self.sold_out(ident):
                ids.append(ident)
        return ids[:4]

    def _open_rows_plan(self, rows: list[str]) -> list[tuple[str, dict[str, Any]]] | None:
        many = self.compare_target()
        joiner = self.roles.get('compare')
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            return [(joiner, {many: rows[:self.compare_room()]})]
        if tool and target:
            return [(tool, {target: ident}) for ident in rows[:2]]
        return None

    def _open_rows_shaped(self, schema: Any, raw: dict[str, Any]) -> dict[str, Any] | None:
        """raw fitted to a tool schema, or None when the schema check reports any fault."""
        problems = []
        fitted = _shape(raw, schema, 'args', problems)
        return None if problems else fitted

    def _cart_calls_fields(self, name: str, schema: dict[str, Any], props: dict[str, Any]) -> tuple[Any, Any, Any]:
        texts = [key for key, spec in props.items() if self._quiet_free_text(spec)]
        learned = [key for tool, key in self.id_args if tool == name and key in texts]
        target = learned[0] if learned else None
        if not target:
            required = [key for key in schema.get('required') or [] if key in texts]
            target = required[0] if required else None
        if not target:
            target = texts[0] if len(texts) == 1 else None
        return target, self._whole_removal_count(props), self._cart_calls_mode(props)

    def _cart_calls_list_field(self, props: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'array' not in _kinds(spec):
                continue
            shape = spec.get('items')
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                return key, shape['properties']
        return None

    def _cart_calls_drop(self, props: dict[str, Any], target: Any, count: Any, mode: Any, ident: str, setting: bool) -> dict[str, Any] | None:
        have = self.cart_lines.get(ident, 1)
        if mode:
            call = {target: ident, mode: props[mode]['enum'][1]}
            if count:
                call[count] = have
            return call
        if not count:
            return None
        if setting:
            return {target: ident, count: 0}
        floor = props[count].get('minimum')
        if isinstance(floor, int) and floor <= -have:
            return {target: ident, count: -have}
        return None

    def _complete_cart_kind(self, ident: str, cat_key: str | None) -> str:
        if not cat_key:
            return ident
        return self._item_ids_label(ident, cat_key).strip().lower()

    def _complete_cart_plan(self, cat_key: str | None) -> dict[str, int]:
        plan = {}
        for number, ids in self.lead_items.items():
            digits = re.search('\\d+', str(number))
            if digits:
                index = int(digits.group())
            else:
                index = 1 if len(self.items) == 1 else None
            pick = next((ident for ident in ids if not self.sold_out(ident)), None)
            if not pick or not index or not 1 <= index <= len(self.items):
                continue
            if self._complete_cart_kind(pick, cat_key) in {self._complete_cart_kind(other, cat_key) for other in plan}:
                continue
            plan[pick] = self.items[index - 1].qty
        return plan

    def _complete_cart_confirmed(self, lines: dict[str, int]) -> Any:
        self.check_cart(list(lines), SPEND_HARD)
        return self.verdicts.get(self.check_key(list(lines)))

    def _complete_cart_apply(self, cart: str, calls: list[dict[str, Any]]) -> list[str] | None:
        schema = self.schemas.get(cart)
        done = []
        for raw in calls:
            args = self._open_rows_shaped(schema, raw)
            reply = None if args is None else self.act(cart, args)
            if reply is None or self.finished:
                return None
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
            done.append(_dump(args))
        return done

    def _complete_cart_retry(self, lines: dict[str, int]) -> bool:
        self.reread_offers()
        if not self._complete_cart_confirmed(lines):
            return False
        return self.left() >= 1 and not self.finished and not any(self.sold_out(ident) for ident in lines)

    def _complete_cart_checkout(self, buy: str, args: dict[str, Any], lines: dict[str, int], drop: list[str]) -> list[str]:
        replies = []
        for attempt in range(2):
            if attempt and not self._complete_cart_retry(lines):
                break
            if not self._complete_cart_order(buy, args, lines, drop, replies):
                break
        return replies

    def _is_buy_match(self, job: str, name: str, args: dict[str, Any]) -> bool:
        if self.roles.get(job) != name:
            return False
        fixed = self.brief.get(job + '_args')
        if not isinstance(fixed, dict):
            return True
        return all(args.get(key) == value for key, value in fixed.items())

    def _offered_keep(self, ends: set[str]) -> set[Any] | None:
        left = self.left()
        tail = self.tail()
        if left <= tail:
            return set(ends)
        if left <= tail + 3:
            return ends | {self.roles.get('cart'), self.roles.get('cart_view')}
        if left <= len(self.items) + 5 and self.buys == 0:
            return ends | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
        return None

    def _quiet_free_text(self, spec: Any) -> bool:
        return isinstance(spec, dict) and 'string' in _kinds(spec) and 'enum' not in spec

    def _quiet_should_fill(self, value: Any, key: str, schema: dict[str, Any]) -> bool:
        if value is None and key not in (schema.get('required') or []):
            return False
        text = str(value or '')
        if not text:
            return True
        return any(ident in text for ident in self.rows) or self.brief.get('quiet_decline') is not False

    def _resolve_ids_free(self, spec: Any) -> bool:
        if not isinstance(spec, dict) or 'enum' in spec:
            return False
        kinds = _kinds(spec)
        return 'string' in kinds or 'array' in kinds

    def _resolve_ids_starts(self, row: dict[str, Any], low: str) -> bool:
        return any(key != self.id_key and isinstance(text, str) and text.strip().lower().startswith(low) for key, text in row.items())

    def _resolve_ids_one(self, value: Any) -> tuple[Any, str]:
        if not isinstance(value, str):
            return (value, '')
        stripped = value.strip()
        if stripped in self.rows:
            return (stripped, '')
        if _id_like(value, True) or len(stripped) < 8:
            return (value, '')
        low = stripped.lower()
        hits = [ident for ident, row in self.rows.items() if self._resolve_ids_starts(row, low)]
        if len(hits) == 1:
            return (hits[0], '')
        if hits:
            return (value, f"more than one listing carries the text {value!r} ({', '.join(hits[:6])}); send a single listing id")
        return (value, f'{value!r} is not a listing id; send the id shown in the result rows (their {self.id_key} field)')

    def _whole_removal_count(self, props: dict[str, Any]) -> Any:
        return next((key for key, spec in props.items() if isinstance(spec, dict) and 'integer' in _kinds(spec)), None)

    def _whole_removal_flag(self, spec: dict[str, Any], args: dict[str, Any]) -> bool:
        for key, value in args.items():
            options = (spec.get(key) or {}).get('enum')
            if isinstance(options, list) and value in options[1:]:
                return True
        return False

    def _note_cart_removal(self, spec: dict[str, Any], args: dict[str, Any]) -> bool:
        marks = []
        for key, value in args.items():
            options = (spec.get(key) or {}).get('enum')
            marks.append(isinstance(options, list) and len(options) >= 2 and value in options[1:])
        return any(marks)

    def _note_cart_entry(self, line: dict[str, Any]) -> tuple[str, Any] | None:
        ident = None
        for value in line.values():
            if isinstance(value, str) and value in self.rows:
                ident = value
                break
        if not ident:
            return None
        units = self._note_cart_loose(line.values())
        return (ident, 1 if units is None else units)

    def _note_cart_lines(self, spec: dict[str, Any], args: dict[str, Any], loose: Any) -> tuple[bool, list[tuple[str, Any]]]:
        lines = []
        whole = False
        for key, value in args.items():
            if isinstance(value, str):
                if value in self.rows:
                    lines.append((value, loose))
            elif isinstance(value, list):
                whole = self._note_cart_list(spec.get(key), value, loose, lines) or whole
        return whole, lines

    def _note_cart_units(self, have: int, units: Any, removal: bool, sets: bool, counted: bool) -> int:
        if removal:
            if counted:
                return have - (units or 1)
            return have - units if units else 0
        if sets:
            return 1 if units is None else units
        return have + (1 if units is None else units)

    def _join_opens_picks(self, calls: list[dict[str, Any]], opener: str, target: str) -> list[tuple[int, str]]:
        picks = []
        for index, call in enumerate(calls):
            fn = call.get('function') or {}
            if fn.get('name') != opener:
                continue
            raw = fn.get('arguments')
            args = _json_obj(raw) if isinstance(raw, str) else None
            if not args or set(args) != {target}:
                continue
            ident = str(args.get(target) or '').strip()
            if ident in self.rows and ident not in self.gone:
                picks.append((index, ident))
        return picks

    def _gate_heard(self, got: dict[str, Any], done_text: list[str]) -> tuple[list[str], str]:
        """What the step check's reply changes once something was done in this gate: new needs, and a hold."""
        if not done_text:
            return [], ''
        changes = []
        self.take_heard({'needs': got.get('needs')}, changes)
        return changes, str(got.get('hold') or '').strip()

    def _gate_done(self, done_text: list[str]) -> str:
        if not done_text:
            return ''
        return 'Before going on, Nimbus did the steps the request asks for:\n' + '\n'.join(done_text)

    def _gate_changed(self, done_text: list[str], changes: list[str], hold: str) -> str:
        lead = self._gate_done(done_text) + '\n' if done_text else ''
        said = changes + [hold] if hold else changes
        return lead + "The shopper's reply changes the task: " + '; '.join(said) + '. Fix the cart for it, then check out again.'

    def _gate_may_press(self, done_text: list[str], pressed: bool) -> bool:
        if done_text or pressed or not self.roles.get('ask'):
            return False
        return self.msg_used < self.msg_limit and self.left() >= 2

    def _gate_call(self, call: dict[str, Any], ending: set[str], done_text: list[str]) -> bool:
        """Run the call the step check named; False ends the gate's rounds."""
        name = str(call.get('tool') or '')
        if name not in self.schemas or name in ending or self.left() < 2:
            return False
        if name == self.roles.get('ask') and self.msg_used >= self.msg_limit:
            return False
        raw = call.get('args')
        args = self._open_rows_shaped(self.schemas[name], raw if isinstance(raw, dict) else {})
        return args is not None and self._gate_follow(name, args, done_text)

    def _gate_ask(self, args: dict[str, Any], done_text: list[str]) -> bool:
        text = ''
        for value in args.values():
            if isinstance(value, str):
                text = value
                break
        answered = self.send([(1, '', text)], 1)
        if answered:
            self.gate_msgs += 1
            done_text.append(f'Q: {text}\nA: {answered[0][2]}')
        return bool(answered)

    def _gate_step_done(self, entry: dict[str, Any], proven: set[int]) -> bool:
        if entry.get('done') is not True:
            return False
        return any((_count(line) or 0) in proven for line in _items(entry.get('lines')))

    def _check_cart_advice(self, got: dict[str, Any], ids: list[str]) -> str:
        self.leads = [ident for ident in self.leads if ident not in ids]
        problems = list(map(str, _items(got.get('problems'))))[:5] or ['see the requirements']
        better = []
        for entry in _items(got.get('better')):
            ident = str(entry)
            if ident in self.rows and not self.sold_out(ident):
                better.append(ident)
        advice = 'An independent review finds this cart not right yet: ' + '; '.join(problems)
        if better[:4]:
            advice += '. Better-looking listings: ' + ', '.join(better[:4])
        return advice + '. Correct the cart (or re-read the records to show the review is mistaken), then check out.'

    def _reread_offers_stale(self) -> list[str]:
        since = max(self.step - 2, self.refused_at)
        stale = []
        for ident in self.cart_ids():
            if ident in self.gone:
                continue
            if self.offer_seen.get(ident, -99) < since:
                stale.append(ident)
        return stale

    def _reread_offers_moved(self, before: dict[str, Any], after: dict[str, Any]) -> list[str]:
        moved = []
        for key, value in after.items():
            if isinstance(value, (dict, list)) or key not in before:
                continue
            if before[key] != value:
                moved.append(key)
        return moved

    def _note_cart_loose(self, values: Any) -> int | None:
        for value in values:
            if isinstance(value, str):
                continue
            amount = _count(value)
            if amount is not None:
                return amount
        return None

    def _note_cart_list(self, entry_spec: Any, value: list[Any], loose: Any, lines: list[tuple[str, Any]]) -> bool:
        shape = (entry_spec or {}).get('items') or {}
        whole = not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape)
        for line in value:
            if isinstance(line, dict):
                whole = True
                entry = self._note_cart_entry(line)
                if entry is not None:
                    lines.append(entry)
            elif isinstance(line, str) and line in self.rows:
                lines.append((line, loose))
        return whole

    def _trail_links(self, chain: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
        links, ids = [], []
        for link in _items(chain.get('links')):
            if isinstance(link, dict):
                links.append(link)
                ids.append(str(link.get('id') or '').strip())
        return links, ids

    def _trail_gap(self, links: list[dict[str, Any]], ids: list[str]) -> tuple[str, str]:
        gap = {}
        for link, ident in zip(links, ids):
            if ident not in self.rows:
                gap = link
                break
        about = str(gap.get('about') or '') or 'the listing it points at'
        return about, str(gap.get('find') or '').strip()

    def _trail_pending(self) -> list[str]:
        live = {need for _, need in self.needs_now()}
        return [need for _, need in self.refs if need in live and need not in self.ref_values]

    def _trail_link(self, gap: tuple[str, str], lost: list[str], opened_text: list[str]) -> tuple[list[str], list[str]]:
        reach = self.find_link(*gap)
        last = self.trace[-1] if self.trace else None
        if last is None or last.get('kind') != 'link_known':
            return reach, lost
        opened_text.append(': page already opened from the shop named in this chain, carrying the values named in its description; read the value off that page')
        return reach, lost[1:]

    def _trail_can_open(self, reach: list[str], tool: str | None, target: str | None) -> bool:
        if not reach or not tool or not target or self.finished:
            return False
        return self.left() >= len(reach) + max(4, self.finish_reserve())

    def _cart_calls_mode(self, props: dict[str, Any]) -> Any:
        for key, spec in props.items():
            if isinstance(spec, dict) and isinstance(spec.get('enum'), list) and len(spec['enum']) >= 2:
                return key
        return None

    def _cart_calls_whole(self, field: tuple[str, dict[str, Any]], lines: dict[str, int]) -> list[dict[str, Any]] | None:
        key, inner = field
        id_key = next((k for k, spec in inner.items() if self._quiet_free_text(spec)), None)
        count_key = self._whole_removal_count(inner)
        if not id_key:
            return None
        entries = []
        for ident, units in lines.items():
            entry = {id_key: ident}
            if count_key:
                entry[count_key] = units
            entries.append(entry)
        return [{key: entries}]

    def _complete_cart_split(self, plan: dict[str, int], cat_key: str | None) -> tuple[dict[str, int], list[str]]:
        planned = {self._complete_cart_kind(ident, cat_key) for ident in plan}
        keep, drop = {}, []
        for ident in self.cart_ids():
            if ident in plan:
                continue
            if self.sold_out(ident) or self._complete_cart_kind(ident, cat_key) in planned:
                drop.append(ident)
            else:
                keep[ident] = self.cart_lines.get(ident, 1)
        return keep, drop

    def _complete_cart_order(self, buy: str, args: dict[str, Any], lines: dict[str, int], drop: list[str], replies: list[str]) -> bool:
        """One checkout try; False when no further try should follow."""
        go, said = self.gate('order')
        if said:
            replies.append(said)
        if not go:
            return False
        reply = self.act(buy, args)
        self._trail_log('completed', lines=lines, drop=drop, reply=_clip(reply, 200))
        if reply is None:
            return False
        self.buys += 1
        ended = reply.get('finished')
        if not ended:
            self.refused_at = self.step
        seen = reply.get('observation')
        self.observe(buy, seen)
        replies.append(self.show(seen))
        return not (self.finished or ended)

    def _take_note_budget(self, got: dict[str, Any]) -> list[str]:
        budget = got.get('budget')
        if isinstance(budget, bool) or not isinstance(budget, (int, float)):
            return []
        if not (budget > 0 and budget != self.budget):
            return []
        self.budget = budget
        return [f'budget now {budget}']

    def _take_note_record(self, summary: str, changed: list[str]) -> str:
        if not changed and not summary:
            return ''
        detail = '; '.join(changed)
        self.version += 1
        self.recent = []
        self.notes.append(summary or detail)
        if not changed:
            return summary
        return f'{summary} ({detail})' if summary else f'({detail})'

    def _take_heard_dicts(self, heard: dict[str, Any], key: str) -> list[dict[str, Any]]:
        return [entry for entry in _items(heard.get(key)) if isinstance(entry, dict)]

    def _fit_plan_tally(self, main: int, index: dict[str, Any]) -> dict[str, dict[int, int]]:
        tally = {}
        parts = [p for p, m in self.bound if m == main]
        for part in parts:
            live = [ident for ident in self.item_ids(part) if not self.sold_out(ident)]
            for ident in live:
                for code in self.fit_set(ident, index):
                    per = tally.setdefault(code, {})
                    per[part] = per.get(part, 0) + 1
        return tally

    def _resolve_ids_field(self, name: str) -> str | None:
        tools = {self.roles.get(job) for job in ('cart', 'open', 'offer', 'compare')}
        tools.discard(None)
        if name not in tools or not self.id_key:
            return None
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [key for key, spec in props.items() if self._resolve_ids_free(spec)]
        if len(free) != 1:
            return None
        field = free[0]
        if name == self.roles.get('cart') or (name, field) in self.id_args:
            return field
        return None

    def _fix_counts_send(self, name: str, schema: dict[str, Any], raw: dict[str, Any]) -> str | None:
        args = self._open_rows_shaped(schema, raw)
        reply = None if args is None else self.act(name, args)
        if reply is None or self.finished:
            return None
        seen = reply.get('observation')
        self.note_cart(name, args)
        self.cart_said = _dump(seen)
        self.observe(name, seen)
        return f'{_dump(args)} -> {self.show(seen)[:300]}'

    def _open_rows_read(self, name: str, args: dict[str, Any]) -> bool:
        reply = self.act(name, args)
        if reply is None or self.finished:
            return False
        self.observe(name, reply.get('observation'))
        return True

    def _fit_plan_rank(self, entry: tuple[str, dict[int, int]]) -> tuple[int, int]:
        per = entry[1]
        return (-len(per), -sum(per.values()))

    def _fit_plan_devices(self, main: int, code: str, index: dict[str, Any]) -> list[str]:
        found = []
        for ident in self.item_ids(main):
            if code in (index['marks'].get(ident, {}).get('codes') or []):
                found.append(ident)
                if len(found) == 2:
                    break
        return found

    def _cart_calls_schema(self) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """The cart tool's name, its parameter schema and the schema's properties ('' and empty when there is none)."""
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        return name, schema, schema.get('properties') or {}

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Nimbus(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.trace
__all__ = ['agent_main']
