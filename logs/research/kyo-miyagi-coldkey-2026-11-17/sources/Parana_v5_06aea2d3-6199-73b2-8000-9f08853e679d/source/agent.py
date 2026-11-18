from __future__ import annotations
import json
import random
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient
WALKER = 'anthropic/claude-haiku-5.5'
THINKER = 'anthropic/claude-sonnet-5.5'
BACKUP = 'openai/gpt-6-luna'
TURN_CAP = 42
CALL_CAP = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
PAUSES = (4, 10, 20, 35)
PAUSE_BUDGET = 45
REPLY_CHARS = 450
OBS_CHARS = 15000
ROW_CHARS = 100
NAME_CHARS = 200
NARROW_SEARCH = 50
CHAT_CHARS = 150000
GATE_CALLS = 3
EXTRA_PAGES = 6
CHEAP_OPENS = 3
SWEEP_PAGES = 8
PLAN_MODE = 'llm'
CANNED_UNCLEAR = re.compile('not (sure what|follow)|did ?n.t (understand|catch|follow|land)|could not catch|simpler|plain(er)? (words|terms|english)|say (that|it) again|once more|another way|differently|rephrase|repeat that|confus|muddled|lost me|over my head|what (are|were) you asking|more simply|more slowly|baffl|none the wiser|not landing|come again|above my understanding|half the steps missing|asked me something|previous message|word it', re.I)
CANNED_FREE = re.compile('no (preference|opinion|special wish|say|strong feeling)|do ?n.t mind|not fussed|any(thing)? (works|will do|is fine|goes)|any option|whatever|up to you|you (choose|pick|decide)|choose for|pick (any|whichever)|happy with any|never (given|thought|bothered)|not given that a thought|either (is|way|will)|all the same|neither here nor there|does ?n.t matter|why would i mind|i am open|what would you suggest|any need to decide|can.t say i mind|alright with me|leave it (open|to you)|your call|no view', re.I)
CANNED_FIRM = re.compile("\\b(should|must|only|firm|resolute|insist|strictly|has to|have to|cannot|can't|matching|means|same as|at least|at most|no more than|other than|remain|keep|stay|settle for|nothing but|full stop)\\b", re.I)
SWEEP_KEEP = 10
CLOCK_SOFT = 185
CLOCK_HARD = 235
CLOCK_LAST = 260
ANSWER_TOOL = {'type': 'function', 'function': {'name': 'give_answer', 'description': 'Hand over the answer the instructions ask for.', 'parameters': {'type': 'object', 'properties': {'answer': {'type': 'object', 'description': 'The JSON object the instructions describe.'}}, 'required': ['answer']}}}
_model_link = ProxyClient(retry_delay=1, max_retries=1, timeout=50)
_store_link = ProxyClient(retry_delay=1, max_retries=3, timeout=95)

def flat(value: Any) -> str:
    try:
        held_1 = json.dumps(value, separators=(',', ':'), ensure_ascii=False)
        return held_1
    except (TypeError, ValueError):
        held_2 = str(value)
        return held_2

def cut(value: Any, limit: int, depth: int=0) -> Any:
    if 7 < depth:
        return '...'
    elif isinstance(value, str):
        if limit < len(value):
            held_3 = value[:limit] + '...'
            return held_3
        return value
    elif not isinstance(value, list):
        if isinstance(value, dict):
            held_5 = dict(((k, cut(v, limit, depth + 1)) for k, v in value.items()))
            return held_5
        return value
    else:
        held_4 = list((cut(v, limit, depth + 1) for v in value))
        return held_4

def as_json(text: Any) -> dict[str, Any] | None:
    if isinstance(text, str):
        body = text.strip()
        fenced = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', body, re.S)
        if fenced:
            body = fenced.group(1)
        for attempt in (body, body[body.find('{'):body.rfind('}') + 1] if '{' in body else ''):
            if attempt:
                try:
                    got = json.loads(attempt)
                except ValueError:
                    continue
                if isinstance(got, dict):
                    return got
                return None
            else:
                continue
        return None
    else:
        return None

def key_words(key: Any) -> str:
    held_6 = str(key)
    spaced = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', held_6)
    held_7 = re.sub('[\\s_.\\-]+', ' ', spaced).strip().lower()
    return held_7

def key_stem(text: Any) -> frozenset[str]:
    held_8 = frozenset((w[:-1] if 3 < len(w) and w.endswith('s') else w for w in key_words(text).split()))
    return held_8

def names_same(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not (not prop or not key):
        held_9 = key == prop or key >= prop or (2 <= len(key) and prop >= key)
        return held_9
    else:
        return False

def type_list(spec: dict[str, Any]) -> list[str]:
    kind = spec.get('type')
    if not isinstance(kind, str):
        held_11 = list((k for k in kind or [] if isinstance(k, str)))
        return held_11
    else:
        held_10 = [kind]
        return held_10
_TYPE_TEST = {'string': lambda v: isinstance(v, str), 'integer': lambda v: isinstance(v, int) and (not isinstance(v, bool)), 'number': lambda v: isinstance(v, (int, float)) and (not isinstance(v, bool)), 'boolean': lambda v: isinstance(v, bool), 'array': lambda v: isinstance(v, list), 'object': lambda v: isinstance(v, dict), 'null': lambda v: None is v}

def type_ok(value: Any, kind: str) -> bool:
    held_12 = _TYPE_TEST.get(kind, lambda v: True)(value)
    return held_12

def _convert(value: Any, kind: str) -> tuple[bool, Any]:
    if 'integer' == kind and isinstance(value, float) and value.is_integer():
        held_13 = (True, int(value))
        return held_13
    elif kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value):
        number = float(value)
        held_14 = (True, int(number) if 'integer' == kind or number.is_integer() else number)
        return held_14
    elif not ('string' == kind and isinstance(value, (int, float)) and (not isinstance(value, bool))):
        if 'boolean' == kind:
            if isinstance(value, str):
                if value.strip().lower() in ('true', 'false'):
                    held_16 = (True, value.strip().lower() == 'true')
                    return held_16
        if 'array' == kind:
            if None is not value:
                if not isinstance(value, list):
                    held_17 = (True, [value])
                    return held_17
        held_18 = (False, value)
        return held_18
    else:
        held_15 = (True, str(value))
        return held_15

def bend(value: Any, spec: Any, where: str, faults: list[str]) -> Any:
    if isinstance(spec, dict):
        kinds = type_list(spec)
        if kinds:
            if not any((type_ok(value, k) for k in kinds)):
                for kind in kinds:
                    done, value = _convert(value, kind)
                    if done:
                        break
                if not any((type_ok(value, k) for k in kinds)):
                    held_20 = '/'.join(kinds)
                    held_19 = '{} should be {}'.format(where, held_20)
                    faults.append(held_19)
                    return value
        allowed = spec.get('enum')
        if isinstance(allowed, list):
            if value not in allowed:
                if isinstance(value, str):
                    same = list((a for a in allowed if isinstance(a, str) and a.lower() == value.strip().lower()))
                    if same:
                        value = same[0]
                    else:
                        value = value
                if value not in allowed:
                    held_21 = '{}: allowed values are {}'.format(where, allowed)
                    faults.append(held_21)
        if isinstance(value, str):
            if isinstance(spec.get('maxLength'), int):
                if spec['maxLength'] < len(value):
                    value = value[:spec['maxLength']]
            if isinstance(spec.get('minLength'), int):
                if spec['minLength'] > len(value):
                    held_22 = '{} is too short'.format(where)
                    faults.append(held_22)
        if isinstance(value, (int, float)):
            if not isinstance(value, bool):
                low, high = (spec.get('minimum'), spec.get('maximum'))
                if isinstance(low, (int, float)):
                    if low > value:
                        held_23 = '{} is below {}'.format(where, low)
                        faults.append(held_23)
                if isinstance(high, (int, float)):
                    if high < value:
                        held_24 = '{} is above {}'.format(where, high)
                        faults.append(held_24)
        if isinstance(value, list):
            most, least = (spec.get('maxItems'), spec.get('minItems'))
            if isinstance(most, int):
                if most < len(value):
                    held_25 = '{} allows at most {} entries'.format(where, most)
                    faults.append(held_25)
            if isinstance(least, int):
                if least > len(value):
                    held_26 = '{} needs at least {} entries'.format(where, least)
                    faults.append(held_26)
            if isinstance(spec.get('items'), dict):
                value = list((bend(v, spec['items'], '{}[{}]'.format(where, i), faults) for i, v in enumerate(value)))
        if isinstance(value, dict):
            if isinstance(spec.get('properties'), dict):
                props = spec.get('properties')
            else:
                props = {}
            if spec.get('additionalProperties') is False:
                faults.extend(('{}.{} is not an allowed argument'.format(where, k) for k in value if k not in props))
            faults.extend(('{}.{} is required'.format(where, k) for k in spec.get('required') or [] if k not in value))
            value = dict(((k, bend(v, props[k], '{}.{}'.format(where, k), faults) if k in props else v) for k, v in value.items()))
        return value
    else:
        return value

def as_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        if isinstance(value, (str, dict)) and value:
            held_27 = [value]
            return held_27
        held_28 = []
        return held_28
    else:
        return value

def whole(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    elif isinstance(value, int):
        return value
    elif not (isinstance(value, float) and value.is_integer()):
        if isinstance(value, str):
            if re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
                held_30 = int(float(value))
                return held_30
        return None
    else:
        held_29 = int(value)
        return held_29

def looks_like_id(value: Any, keyed: bool=False) -> bool:
    if isinstance(value, str):
        text = value.strip()
        if not (not 3 <= len(text) <= 64 or ' ' in text):
            held_31 = keyed or bool(re.search('\\d', text))
            return held_31
        else:
            return False
    else:
        return False

def guess_id_key(rows: list[dict[str, Any]]) -> str | None:
    best: tuple[tuple[bool, float], str] | None = None
    for key in rows[0]:
        keyed = 'id' in str(key).lower()
        values = list((row.get(key) for row in rows))
        if all((looks_like_id(v, keyed) for v in values)):
            values = list((v.strip() for v in values))
            if not (1 < len(rows) and len(set(values)) < len(values)):
                rank = (keyed, sum(map(len, values)) / len(values))
                if None is best or best[0] < rank:
                    best = (rank, key)
            else:
                continue
        else:
            continue
    if best:
        held_32 = best[1]
        return held_32
    return None

def holds_id(node: Any, ident: str, depth: int=0) -> bool:
    if 5 < depth:
        return False
    elif isinstance(node, dict):
        held_33 = any((holds_id(v, ident, depth + 1) for v in node.values()))
        return held_33
    elif not isinstance(node, list):
        return False
    else:
        for v in node:
            if isinstance(v, dict):
                if ident in (str(x).strip() for x in v.values() if isinstance(x, str)):
                    return True
        held_34 = any((holds_id(v, ident, depth + 1) for v in node))
        return held_34

def attr_block(record: Any) -> dict[str, Any] | None:
    if isinstance(record, dict):
        best: dict[str, Any] | None = None
        for value in record.values():
            if not (not isinstance(value, dict) or 2 > len(value)):
                plain = dict(((k, v) for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))))
                if 2 <= len(plain):
                    if None is best or len(best) < len(plain):
                        best = plain
            else:
                continue
        return best
    else:
        return None
FIT_RULES = 'the attribute whose key names the property settles it (spelling of the key does not matter; one entry of a list is enough; plain synonyms and converted units count; a value holding the wanted value\'s words counts, as "dusty rose" holds rose; limits include their edge); a key about some other property, even a neighbouring one (a strap\'s colour is not the case\'s colour), settles nothing; when no key names the property, the wanted value written under another attribute, in the title or in the description counts (a bare number that does not say what it measures counts for nothing); a value the category label itself states is met by that category; the category label must be the requested kind; a rule of differing from another listing\'s value is met unless the records show that very value; a listing shown out of stock, or one that breaks the budget, does not fit.'
WALK = 'You are Kestrel, a buyer acting for one shopper in an online store. You act only by calling the store\'s tools. The episode scores 1 or 0, and only an exactly right finish scores: either an order holding every requested item at its requested quantity, each meeting every requirement in force at checkout, within the budget; or, when the store really has nothing that meets everything, an ending through the no-purchase tool done the way the request asks.\n\nTHE STORE\n- Each tool call is one step and the steps are counted; thinking is free. Every store names its tools, arguments and result fields its own way: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Searches can be narrow: a category filter, a keyword mode that may read only the first words, a few rows per page, a token or offset for the next page. When the answer depends on seeing every match, compare the reported total with the rows seen and page on with the exact token or offset the store gives. Keyword queries work best short and value-first: the wanted value, then the product noun. A filtered search with no rows usually means a wrong filter value or parameter: retry with the label exactly as result rows spell it, or with another filter.\n- Result rows are summaries. Stores split a listing\'s record in different ways: one view may show its attributes (its properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kestrel knows it. Judge every requirement on a property from those attributes, so read them for each listing you might buy; one call showing several listings saves steps when it shows what you need. Never reopen a listing when nothing has changed since you read it.\n- Kestrel\'s interviewer asks the shopper for you, so you never message the shopper yourself. The answers come back in the conversation and go into the ledger.\n- The request may also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kestrel does the ones due just before the order or the ending itself when you call the checkout or ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through the interviewer).\n\nWHAT FITS\n- The kind of product is set by the listing\'s category label: only the label matching the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one entry of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value holding the required value\'s words counts ("dusty rose" meets rose, "extra wide fit" meets wide). A value in other words does not count (an abbreviation or code is not the word it stands for, and "sea green" is not "teal green"). Convert units before comparing numbers; limits include the edge; a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails, whatever other keys or the title say. A key for another property, even a related one (a strap\'s colour is not the case\'s colour), decides nothing about it. When listings of this kind carry a key named plainly as the property, a key naming one part of the product does not stand in for it. Kestrel adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (a list of styles or features that includes it, for example) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), the category meets that requirement.\n- A requirement can point at another listing instead of giving a value ("the same X as the one <shop> sells with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page points to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule rules out only listings sharing that exact value, and it is never a reason on its own to end without a purchase.\n- When a listing you are about to buy refers you to another page about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the pair together from the listings\' own words (a list of what it fits, a model or series code both pages carry). When no accessory meeting its own requirements fits the device you picked, switch to another qualifying device that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") only break ties.\n- Never buy what the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, lowest price), the order must hold the lowest-priced listing among ALL fitting listings. Find the whole fitting set (page through it: rows often come in no price order, and Kestrel reads further pages of a short result itself), check candidates from the cheapest up, and take the first that fully fits. When a search takes a maximum price, use it: list the category below your current best pick\'s price, so only cheaper candidates remain.\n- When the request asks for the most or least of some measure, compare every fitting listing on it and take the extreme; a listing not stating the measure cannot win.\n- Otherwise any listing meeting everything is right: do not spend steps looking for a better one.\n- Cart each item as soon as you have opened a listing meeting every requirement for it (for a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- Several different items need one cart line each, at the requested quantity.\n\nCHANGES ON THE WAY\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a sold-out listing. They show up inside tool results, often as a refused checkout; a note counts whichever tool\'s result carries it, and the ledger lists the requirements in force. A refusal does not end the episode: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change without notice.\n\nFINISHING\n- Put exactly the chosen lines in the cart (take anything else out), look at the cart once, then check out with that call alone.\n- End without a purchase only after a real search shows no listing meets everything. Running short of steps shows nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
READ_REQUEST = 'You set up a shopping task for an agent. Read the shopper\'s request and the store\'s tools and answer with ONE JSON object (compact, no indentation, nothing else) holding these keys:\n\n"items": one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when some tool lists labels (copied exactly), else "", "quantity": units of that one listing (an integer, 1 when not stated; a pair or a set sold as one listing counts as 1), "musts": the firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word ("for a lakeside party" gives "occasion: lakeside", "a 30 cm blade" gives "blade length: 30 cm"); never swap it for a neighbouring property. A must whose value comes from another listing is ONE must holding the whole description of that listing ("lid colour: unlike the lid colour of the kettle Ovarno sells with a steel base"): the words describing that other listing (its shop, its kind, the "with <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity among the musts. The story around the request (who will use it, where it will stand, what went wrong before, the occasion) is context, not a must, unless it states a property\'s value or limit outright; words that are part of what the product is called (a season, an audience or a use named with the product) belong in "name", not in the musts. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often one store department\'s name) is ONE item; make separate items only when each product has its own quantity or its own requirements. A must that excludes: "anything bar X" / "anything other than X" (the shopper takes the rest) is "property: not X"; refusing everything except X ("I won\'t take anything other than X", "nothing but X will do") is "property: X".\n"budget": the most the shopper pays in total, as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null without a limit.\n"in_stock_only": true unless the request clearly accepts unavailable items.\n"pick": "cheapest" when the request asks for the lowest price or total, "most" or "least" when it asks for the extreme of a measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": the measure maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (an integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (an integer), or null.\n"lookups": the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": things the shopper would merely like.\n"ask_first": things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": the steps the request requires besides finding and buying the right items (the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": the messages to the shopper it takes (0 when none)}; [] when there is no such step. Requirements on the items, the budget, stock, the message and step limits and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"no_fit": what the request says to do when no listing meets every requirement: "order_closest" when it wants its nearest match bought anyway (the one missing the least, even when it also asks to be told), "no_purchase" when it asks to buy nothing, else "not said".\n"roles": the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; with two single-listing tools, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends the shopper a message), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
SPLIT_POINTER = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes misread as requirements on the product being bought. From the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Answer with ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when each one is a real requirement on the product]}'
PLAN_QUESTIONS = 'You pick the questions a shopping agent sends to a shopper. Shoppers keep a few firm preferences they did not write down, each about one property of one requested product. A question uncovers a preference only when it names that exact property (or a plain synonym), and each question is answered only about the first property it names, so every question spends itself on one property; a property the shopper does not care about gets "no preference" and the question is lost.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and sample values. An unwritten preference can sit on a property no listed key names.\n\nPick properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it concerns.\n1. Hinted properties. The request\'s story often points at an unwritten preference without its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the room it belongs in, the person meant to use or wear it, how it fastens, or what goes inside it. For each such sentence, name the one property it points at and the item it concerns, in the story\'s order: the person who will wear it points at an age group or a gender, but only for something worn or when a listed key carries age or gender (take whichever the listed keys carry; an age group when the story names children, teens or adults); for anything else, the person behind the story points at no property by itself, how it fastens at its closure, what goes inside a computer at the processor, its looks at colour, pattern or style (whichever the request leaves open). Use the listed key\'s name closest to the hint.\n2. Then the listed keys, best first: the properties a buyer of that kind of product decides on first (most often its size, capacity or dimensions, its type, its material or its colour, whichever that kind of product is chosen by; for an electronic device, its processor or chip as well, even when few listings show it), and keys most of the item\'s listings carry; a key seen on only one or two listings rarely holds a preference.\nSkip a property the request already settles for that item, a second name for one already picked, a key listing purchase variants (a mix of sizes, colours and pack counts), identifiers and model numbers, warranty, shipping and other logistics. With several items, the item the request settles least comes first: give every item one question, least settled first, before any item gets a second, and put hinted properties first within an item.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); just the property\'s name, no sample values, no brackets, no second property.\nPick up to COUNT, best first. Answer with ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}'
READ_ANSWERS = 'A shopping agent asked a shopper some questions. Read each answer and answer with ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer wants one question at a time, the questions taken in turn, or only the first part answered], "unclear": [numbers of the questions whose answer says the shopper did not follow it, or asks for it again, more simply or more gently], "budget": the new total limit as a number, or null}\n- "needs": each firm preference an answer states, with its exact value or limit and unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked; several answers about one property make one need. Copy the value as the answer words it, keeping its nouns and units ("at least 3 shelves", not "at least 3").\n- "Anything bar X" / "anything other than X" (the rest is fine) is the need "property: not X"; refusing everything except X ("I won\'t settle for anything other than X", "nothing but X will do", "can\'t do without X") is "property: X".\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparing words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- "crowded" is only for answers asking for fewer questions at once. Such an answer is about the asking, not the product: it states no need, even when its words (one by one, in turn, single, the first part) could also describe the property asked about. An answer that did not follow the question, asks to have it repeated, put more simply or more gently, is "unclear", never crowded. An answer saying the shopper has no preference, has never thought about it or will take anything ("not fussed", "never given it a thought", "any would do") is "free", never unclear.\n- Folksy or roundabout wording that names a value still states that value.'
READ_NOTE = 'While a shopping agent works, the store showed this note (from the shopper or the store). Answer with ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refused order), "error" when it only says a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even unnamed), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
REVIEW_CART = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: ' + FIT_RULES + ' When listings of this kind carry a key named plainly as the property, a key naming one part of the product does not stand in for it. Each requested item must be its own cart line at the requested quantity; nothing else may be in the cart, nothing shown out of stock, and price times quantity summed over all lines must be within the budget. A listing\'s newest record wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nAnswer with ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
REVIEW_DECLINE = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + FIT_RULES + '\nRecords marked ROW were only seen as a search result row: their page (the attributes) is unread, so they can be neither passed nor failed on a property the row does not show.\nAnswer with ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
FOLLOW_CHAIN = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from its own description, link by link, through the pages\' own words, using the records below (OPENED: the agent opened that page; ROW: seen only as a search result row, its page unread).\nRead each page\'s remarks about the property before taking a value from it:\n- A page whose remarks send the reader elsewhere for this property (to go by, look to or take it from another listing, or saying its own value is not what applies) does not state the value, even when its own attributes show one: the listing it describes is the next link. When that description fits several listings and names how to choose one (such as the highest or lowest of some property among those that match), the next link is the one that rule picks, and every listing fitting the description must be OPENED to apply it.\n- A rule on these pages that pairs values of this property ("pairs A with B", "A goes with B") turns the last page\'s value into its partner when the requirement asks for a value that works alongside, goes with or matches through that rule.\nAnswer with ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page stating the property\'s value itself, else false, "value": the value the requirement asks for after these rules (for a requirement that excludes a value, the excluded value), or ""}]}\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
STEP_GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. From the numbered RECORD of what the agent has done, decide whether each step due before this STAGE is done for the cart as it now stands. A step is done only when a RECORD line shows it done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions about a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nAnswer with ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers showing it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
FIND_SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + FIT_RULES + '\nAnswer with ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
WORLD_FACTS = 'You keep the notes a shopping agent works from. Every store differs: tools get new names, arguments new meanings, replies new shapes, and a tool\'s own description can be wrong. Read this store only from what you are shown: its tool list (function schemas) and the calls made so far with their replies. Note only what the replies show, and never repeat what a schema says unless a reply bears it out or contradicts it.\nAnswer with ONE JSON object and nothing else:\n{"search": {"rows_per_page": the most result rows one search reply showed, or null, "next_page": the argument that fetches more results and what goes in it, or null},\n"open_many_limit": the most listing ids one side-by-side call really takes, judged from the replies, or null. A schema limit can be wrong and a refusal can be worded misleadingly: a refusal that names no reason, after a call with more ids than a later call that worked, points at the count.\n"cart": {"amount_arg": the argument that sets how many units, or null, "amount_meaning": "units" (the number is the line\'s unit count), "position" (a line or slot position, not a count), "whole_cart_list" (one call sets the entire cart from a list, replacing what was there), "one_per_call" (each call adds one unit) or "unknown", "remove_how": how a listing is taken out of the cart, or null}. Judge the meaning from what the cart held after a change, not from the argument\'s name: a call that sent 0 and left 0 units means the number is the unit count.\n"surprises": [at most 5 notes, each one sentence of at most 20 words, on ways this store behaved differently from its tool descriptions]}'
NAME_PARTS = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Answer with ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, saying whether the listing can be bought now, or null}'

class Want:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name, self.shelf, self.qty, self.musts = (name, shelf, qty, musts)
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Wire:

    def model_turn(self, chat: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None, force: bool=False) -> dict[str, Any] | None:
        for model in models:
            if not CALL_CAP <= self.calls:
                self.calls = self.calls + 1
                tries = 0
                while True:
                    message, status = self.model_once(model, chat, max_tokens, tools, force)
                    if not None is not message:
                        held_35 = None is status
                        if not held_35:
                            held_35 = status in (402, 408, 425, 429)
                        if not held_35:
                            held_36 = isinstance(status, int)
                            if held_36:
                                held_36 = 500 <= status
                            held_35 = held_36
                        transient = held_35
                        if status in (None, 408):
                            limit = 1
                        else:
                            limit = len(PAUSES)
                        if not (not transient or limit <= tries or PAUSE_BUDGET < self.waited + PAUSES[tries]):
                            self.waited += PAUSES[tries]
                            time.sleep(PAUSES[tries] * random.uniform(0.8, 1.2))
                            tries = tries + 1
                        else:
                            break
                    else:
                        return message
                if 402 == status:
                    self.no_credit = True
                    return None
            else:
                return None
        return None

    def model_once(self, model: str, chat: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, force: bool=False) -> tuple[dict[str, Any] | None, Any]:
        sent = chat
        if model.startswith('anthropic/'):
            if 1 < len(chat):
                sent = list((dict(m) for m in chat))
                for spot in {0, len(sent) - 1}:
                    text = sent[spot].get('content')
                    if isinstance(text, str):
                        if text:
                            sent[spot]['content'] = [{'type': 'text', 'text': text, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body.update(tools=tools, tool_choice={'type': 'function', 'function': {'name': tools[0]['function']['name']}} if force else 'auto')
        result = _model_link.post_verbose('/inference/chat/completions', body)
        data = result.data
        if not (isinstance(data, dict) and data.get('choices')):
            held_40 = result.error
            if not held_40:
                held_40 = {}
            error = held_40
            status = error.get('status')
            if isinstance(data, dict):
                if isinstance(data.get('error'), dict):
                    status, error = (whole(data['error'].get('code')), {'body': data['error'].get('message')})
            self.trace.append({'kind': 'model_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
            held_41 = (None, status)
            return held_41
        else:
            spent = (data.get('usage') or {}).get('cost')
            if isinstance(spent, (int, float)):
                self.cost += float(spent)
            held_37 = data['choices'][0].get('message')
            if not held_37:
                held_37 = {}
            message = held_37
            if not (message.get('content') or message.get('tool_calls')):
                self.trace.append({'kind': 'model_empty', 'model': model})
                held_39 = (None, 0)
                return held_39
            else:
                held_38 = (message, 200)
                return held_38

    def model_json(self, prompt: str, models: list[str], max_tokens: int=1800, rescue: bool=True) -> dict[str, Any] | None:
        for model in models:
            held_42 = (self.model_turn([{'role': 'user', 'content': prompt}], [model], max_tokens) or {}).get('content')
            got = as_json(held_42)
            if None is not got:
                return got
            elif self.broke or self.no_credit or CALL_CAP <= self.calls:
                return None
        if rescue:
            held_43 = self.forced_json(prompt, models[0], max_tokens)
            return held_43
        return None

    def forced_json(self, prompt: str, model: str, max_tokens: int) -> dict[str, Any] | None:
        if not self.elapsed() > CLOCK_HARD:
            held_44 = self.model_turn([{'role': 'user', 'content': prompt}], [model], max_tokens, [ANSWER_TOOL], True)
            if not held_44:
                held_44 = {}
            message = held_44
            for call in message.get('tool_calls') or []:
                if isinstance(call, dict):
                    raw = (call.get('function') or {}).get('arguments')
                else:
                    raw = None
                if isinstance(raw, dict):
                    args = raw
                else:
                    args = as_json(raw)
                if isinstance((args or {}).get('answer'), dict):
                    self.trace.append({'kind': 'forced_answer'})
                    held_45 = args['answer']
                    return held_45
            return None
        else:
            return None

    def store(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding, step=self.step + 1, action={'name': name, 'args': args})
        for _ in range(4):
            result = _store_link.post_verbose('/environment/call', payload)
            if not isinstance(result.data, dict):
                held_47 = result.error
                if not held_47:
                    held_47 = {}
                error = held_47
                status, body = (error.get('status'), str(error.get('body') or ''))
                self.trace.append({'kind': 'store_fail', 'status': status, 'body': body[:200]})
                if 409 == status:
                    low = body.lower()
                    if not any((w in low for w in ('finish', 'quarantin', 'finaliz'))):
                        expected = re.search('expected (\\d+)', body)
                        if expected:
                            self.step = int(expected.group(1)) - 1
                        elif not ('reuse' in low or 'different action' in low):
                            return None
                        else:
                            self.step = self.step + 1
                        payload['step'] = self.step + 1
                        continue
                    else:
                        self.finished = True
                        return None
                elif status in (400, 401, 402, 403, 404, 422):
                    self.broke = self.broke or 402 == status
                    return None
            else:
                data = result.data
                if isinstance(data.get('step'), int):
                    self.step = data['step']
                else:
                    self.step = self.step + 1
                self.finished = self.finished or bool(data.get('finished'))
                seen = data.get('observation')
                held_46 = '[step {}] {} {}\n    -> {}'.format(self.step, name, flat(args)[:200], flat(cut(seen, 90))[:700])
                self.world_log.append(held_46)
                if name == self.roles.get('search'):
                    if not (isinstance(seen, dict) and seen.get('error')):
                        self.page_most = max(self.page_most, len(self.records_in(seen)))
                if name in self.page_tools():
                    if not (isinstance(seen, dict) and seen.get('error')):
                        for value in args.values():
                            self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                return data
        return None

    def read_request(self) -> None:
        tool_lines = list(('- {}: {} | parameters: '.format(t['function']['name'], t['function'].get('description', '')) + flat(t['function'].get('parameters') or {}) for t in self.tools))
        prompt = READ_REQUEST + '\nTOOLS:\n' + '\n'.join(tool_lines) + '\n\nREQUEST:\n' + self.request
        if self.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + flat(cut(self.view.get('opening'), 1500))
        held_48 = self.model_json(prompt, [THINKER, WALKER], 4000)
        if not held_48:
            held_48 = {}
        brief = held_48
        self.brief = brief
        if isinstance(brief.get('roles'), dict):
            roles = brief.get('roles')
        else:
            roles = {}
        self.roles = dict(((job, tool) for job, tool in roles.items() if isinstance(tool, str) and tool in self.schemas))
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        self.split_end_tool()
        for entry in as_list(brief.get('items')):
            if isinstance(entry, dict):
                held_49 = entry.get('quantity')
                qty = whole(held_49)
                musts = list((str(m) for m in as_list(entry.get('musts')) if m))
                held_51 = str(entry.get('name') or 'the product')
                held_52 = str(entry.get('shelf') or '')
                held_50 = Want(held_51, held_52, qty if qty and 0 < qty else 1, musts)
                self.wants.append(held_50)
        if not self.wants:
            self.wants = [Want('the requested product', '', 1, [])]
        held_54 = brief.get('lookups')
        held_53 = as_list(held_54)
        self.take_lookups(held_53)
        self.split_pointers()
        budget = brief.get('budget')
        if isinstance(budget, str):
            if re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
                budget = float(budget)
        if isinstance(budget, (int, float)):
            if not isinstance(budget, bool):
                if 0 < budget:
                    self.budget = budget
        held_55 = brief.get('messages')
        limit = whole(held_55)
        if None is not limit and 0 <= limit:
            self.msg_limit = limit
        else:
            self.msg_limit = 3 if self.roles.get('ask') else 0
        held_56 = brief.get('steps')
        steps = whole(held_56)
        if None is not steps:
            if 10 <= steps < self.max_steps:
                self.max_steps = steps
        for entry in as_list(brief.get('process')):
            if isinstance(entry, dict):
                if str(entry.get('step') or '').strip():
                    when = str(entry.get('when') or '').strip().lower()
                    self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, whole(entry.get('messages')) or 0)})
        if self.roles.get('ask'):
            talk = list((p['messages'] for p in self.process if 'order' == p['when'] and p['messages']))
            if talk:
                self.reserve = min(max(talk), self.msg_limit)
            else:
                self.reserve = 0

    def split_end_tool(self) -> None:
        brief, buy = (self.brief, self.roles.get('buy'))
        if not buy or buy != self.roles.get('refuse'):
            return
        elif not (brief['buy_args'] and brief['refuse_args'] and (brief['refuse_args'] != brief['buy_args'])):
            ask = 'The store tool {} either checks out or ends without a purchase, depending on its arguments. Its schema: '.format(buy) + flat(self.schemas.get(buy)) + '\nAnswer with ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase}'
            held_57 = self.model_json(ask, [WALKER, THINKER], 600)
            if not held_57:
                held_57 = {}
            got = held_57
            if isinstance(got.get('buy_args'), dict):
                if isinstance(got.get('refuse_args'), dict):
                    if got['refuse_args'] != got['buy_args']:
                        brief['buy_args'], brief['refuse_args'] = (got['buy_args'], got['refuse_args'])
        else:
            return

    def take_lookups(self, lookups: list[Any]) -> None:
        for raw in lookups:
            text = str(raw).strip()
            if not any((text in w.musts for w in self.wants)):
                prop = key_words(text.split(':')[0])
                text = next((m for w in self.wants for m in w.musts if key_words(m.split(':')[0]) == prop), text)
            number = next((n for n, w in enumerate(self.wants, 1) if text in w.musts), 1)
            if text:
                if (number, text) not in self.refs:
                    self.refs.append((number, text))

    def split_pointers(self) -> None:
        for number, want in enumerate(self.wants, 1):
            pointers = list((need for n, need in self.refs if number == n and need in want.musts))
            rest = list((m for m in want.musts if m not in pointers))
            if not (not pointers or not rest):
                prompt = SPLIT_POINTER + '\n\nREQUEST:\n' + self.request + '\n\nPRODUCT: {}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n'.format(want.name) + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
                named = set((str(x).strip() for x in as_list((self.model_json(prompt, [THINKER, WALKER], 500) or {}).get('describe_other'))))
                moved = list((m for m in rest if m in named))
                if moved:
                    joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
                    want.musts = list((joined if pointers[0] == m else m for m in want.musts if m not in moved))
                    self.refs = list(((n, joined if (number, pointers[0]) == (n, need) else need) for n, need in self.refs))
                    self.trace.append({'kind': 'split_pointer', 'item': number, 'moved': moved})
                else:
                    continue
            else:
                continue

class Books:

    def left(self) -> int:
        held_58 = self.max_steps - self.step
        return held_58

    def elapsed(self) -> float:
        held_59 = time.time() - self.started
        return held_59

    def msgs_free(self) -> int:
        held_60 = self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)
        return held_60

    def cart_mark(self) -> str:
        held_61 = flat(sorted(self.cart_lines.items()))
        return held_61

    def gate_due(self, stage: str) -> bool:
        held_62 = any((stage == p['when'] for p in self.process)) and (stage, self.cart_mark() + '#{}'.format(self.version)) not in self.gate_ok
        return held_62

    def tail(self) -> int:
        talk = any(('order' == p['when'] and 0 < p['messages'] for p in self.process))
        if talk and self.msg_limit > self.msg_used and self.gate_due('order'):
            return 2
        return 1

    def one_id_arg(self, tool: str | None) -> str | None:
        held_63 = self.schemas.get(tool or '')
        if not held_63:
            held_63 = {}
        spec = held_63
        held_64 = spec.get('properties')
        if not held_64:
            held_64 = {}
        props = held_64
        free_text = list((k for k, v in props.items() if isinstance(v, dict) and 'string' in type_list(v) and ('enum' not in v)))
        held_65 = spec.get('required')
        if not held_65:
            held_65 = []
        required = held_65
        held_66 = next((k for k in required if k in free_text), None)
        if not held_66:
            if 1 == len(free_text):
                held_66 = free_text[0]
            else:
                held_66 = None
        arg = held_66
        if not (not tool or not arg or any((arg != k for k in required))):
            return arg
        else:
            return None

    def page_tools(self) -> set[str]:
        held_67 = self.roles.get('offer')
        if not held_67:
            held_67 = self.roles.get('open')
        page = held_67
        held_68 = set((t for t in (page, self.roles.get('compare')) if t))
        return held_68

    def facts_tool(self) -> str | None:
        seen = next((t for t, shows in self.shows_facts.items() if shows and self.one_id_arg(t)), None)
        if not seen:
            tool = self.roles.get('open')
            if self.shows_facts.get(tool or '') is not False and self.one_id_arg(tool):
                return tool
            return None
        else:
            return seen

    def compare_shows(self) -> bool | None:
        held_69 = self.shows_facts.get(self.roles.get('compare') or '')
        return held_69

    def compare_useless(self) -> bool:
        seen = self.compare_shows()
        if not False is seen:
            held_70 = None is seen and (self.shows_facts.get(self.facts_tool() or '') is True or bool(self.roles.get('offer')))
            return held_70
        else:
            return True

    def id_of(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or 3 > len(node):
            return None
        elif self.id_key and looks_like_id(node.get(self.id_key), True):
            held_71 = node[self.id_key].strip()
            return held_71
        elif in_list or 0 == depth or any((isinstance(v, dict) for v in node.values())):
            key = guess_id_key([node])
            if key:
                if 'id' in str(key).lower() or node[key].strip() in self.rows:
                    held_72 = node[key].strip()
                    return held_72
            return None
        else:
            return None

    def rows_list(self, node: list[Any]) -> bool:
        rows = list((n for n in node if isinstance(n, dict) and 3 <= len(n)))
        if not rows or len(node) / 2 > len(rows):
            return False
        elif not (self.id_key and all((looks_like_id(r.get(self.id_key), True) for r in rows))):
            key = guess_id_key(rows)
            if key:
                self.id_key = key
            held_73 = bool(key)
            return held_73
        else:
            return True

    def take_in(self, observation: Any) -> None:

        def visit(node: Any, depth: int, in_list: bool=False) -> None:
            if 6 < depth:
                return
            elif isinstance(node, list):
                listed = self.rows_list(node)
                for value in node:
                    visit(value, depth + 1, listed)
                return
            elif isinstance(node, dict):
                ident = self.id_of(node, in_list, depth)
                if not ident:
                    for value in node.values():
                        visit(value, depth + 1)
                    return
                if ident not in self.rows:
                    self.seen_order.append(ident)
                    self.rows[ident] = {}
                old = self.rows[ident]
                if any((old.get(k) is True and False is v for k, v in node.items())):
                    self.unavailable.add(ident)
                elif any((old.get(k) is False and True is v for k, v in node.items())):
                    self.unavailable.discard(ident)
                old.update(node)
                if attr_block(node):
                    self.full[ident] = dict(old)
                elif ident in self.full:
                    self.full[ident].update(node)
            else:
                return
        visit(observation, 0)

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:

        def from_list(values: list[Any]) -> list[tuple[str, dict[str, Any]]]:
            held_74 = list(((i, r) for r in values if isinstance(r, dict) for i in [self.id_of(r, True)] if i))
            return held_74
        if isinstance(observation, list):
            held_75 = from_list(observation)
            return held_75
        elif isinstance(observation, dict):
            ident = self.id_of(observation)
            if not ident:
                found: list[tuple[str, dict[str, Any]]] = []
                for value in observation.values():
                    if isinstance(value, list):
                        found += from_list(value)
                    elif isinstance(value, dict):
                        for inner in value.values():
                            if isinstance(inner, list):
                                found += from_list(inner)
                return found
            else:
                held_77 = [(ident, observation)]
                return held_77
        else:
            held_76 = []
            return held_76

    def enum_labels(self) -> set[str]:
        found: set[str] = set()

        def visit(spec: Any, depth: int) -> None:
            if not (6 < depth or not isinstance(spec, dict)):
                found.update((o.strip().lower() for o in spec.get('enum') or [] if isinstance(o, str)))
                for value in spec.values():
                    if isinstance(value, dict):
                        visit(value, depth + 1)
            else:
                return
        for spec in self.schemas.values():
            visit(spec, 0)
        return found

    def category_field(self) -> str | None:
        values: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str):
                    if 0 < len(value) <= 40:
                        if not re.search('\\d{3}', value):
                            values.setdefault(key, set()).add(value)
                            present[key] = present.get(key, 0) + 1
        total, labels = (max(1, len(self.rows)), self.enum_labels())
        best = None
        for key, seen in values.items():
            if not (0.8 * total > present[key] or 40 < len(seen)):
                listed = sum((1 for v in seen if v.strip().lower() in labels)) / len(seen)
                rank = (-listed, len(seen), -present[key], -sum(map(len, seen)) / len(seen))
                if None is best or best[0] > rank:
                    best = (rank, key)
            else:
                continue
        if best:
            held_78 = best[1]
            return held_78
        return None

    def _matches_want(self, label: str) -> bool:
        for w in self.wants:
            long_words = set((x for x in key_stem(w.shelf or '') | key_stem(w.name) if 4 <= len(x)))
            if key_words(label) in (key_words(w.shelf or ''), key_words(w.name)) or set((x for x in key_stem(label) if 4 <= len(x))) & long_words:
                return True
        return False

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        cat = self.category_field()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if record:
                if cat:
                    label = str(record.get(cat) or 'listings')
                else:
                    label = 'listings'
                group = groups.setdefault(label, {'n': 0, 'keys': {}})
                group['n'] += 1
                done: set[str] = set()
                for key, value in (attr_block(record) or {}).items():
                    name = key_words(key)
                    if not (not name or name in done):
                        done.add(name)
                        slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                        slot['n'] += 1
                        text = str(value).strip().lower()
                        slot['per'][ident] = text
                        if text not in slot['values']:
                            slot['values'].append(text)
                    else:
                        continue
            else:
                continue
        mine = set((label for label in groups if self._matches_want(label)))
        if mine:
            groups = dict(((label, g) for label, g in groups.items() if label in mine))
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group['keys'].items(), key=lambda kv: -kv[1]['n']):
                held_79 = slot['per'].values()
                vals = list(held_79)
                if vals:
                    mean = sum(map(len, vals)) / len(vals)
                    variant_mix = sum((1 for v in vals if ' / ' in v)) * 2 >= len(vals)
                    held_80 = 4 <= len(vals)
                    if held_80:
                        held_80 = len(set(vals)) == len(vals)
                    if held_80:
                        held_80 = 12 < mean
                    all_distinct_long = held_80
                    twin = any((name.startswith(other + ' ') and spec['per'] == slot['per'] for other, spec in kept))
                    if not (40 < mean or variant_mix or all_distinct_long or twin):
                        kept.append((name, slot))
                    else:
                        continue
                else:
                    continue
            merged: list[dict[str, Any]] = []
            for name, slot in kept:
                held_81 = slot['per'].values()
                vs = set(held_81)
                home = None
                for m in merged:
                    held_82 = m['per'].values()
                    ws = set(held_82)
                    clash = any((m['per'].get(ident, value) != value for ident, value in slot['per'].items()))
                    if vs:
                        if ws:
                            if not clash:
                                if max(1, 0.5 * min(len(vs), len(ws))) <= len(vs & ws):
                                    if not (1 == min(len(vs), len(ws)) and (not set(name.split()) & set(m['names'][0].split()))):
                                        home = m
                                        break
                if not None is home:
                    home['names'].append(name)
                    home['each'].append(slot['n'])
                    for ident, value in slot['per'].items():
                        home['per'].setdefault(ident, value)
                    home['values'] += list((v for v in slot['values'] if v not in home['values']))
                else:
                    merged.append({'names': [name], 'each': [slot['n']], 'per': dict(slot['per']), 'values': list(slot['values'])})
                    continue
            top = max(list((len(m['per']) for m in merged)) or [1])
            merged.sort(key=lambda m: -(len(m['per']) / top) * min(max(len(set(m['per'].values())), 1), 3))
            merged.sort(key=lambda m: -sum(m['each']))
            for m in merged[:14]:
                name = max(zip(m['names'], m['each']), key=lambda p: (p[1], -len(p[0])))[0]
                self.key_aliases[label, name] = list((x for x in m['names'] if name != x))
                table.append((label, name, len(m['per']), list((v[:40] for v in m['values'][:4]))))
        return table

    def want_groups(self) -> list[list[str]]:
        cat = self.category_field()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            groups.setdefault(str(self.rows[ident].get(cat) or '') if cat else '', []).append(ident)
        mine = list((ids for label, ids in groups.items() if self._matches_want(label)))
        held_83 = sorted(mine or list(groups.values()), reverse=True, key=len)[:max(1, len(self.wants))]
        return held_83

    def price_of(self, ident: str) -> float | None:
        if self.price_key:
            value = (self.rows.get(ident) or {}).get(self.price_key)
        else:
            value = None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str):
            if re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
                value = float(value)
        if isinstance(value, (int, float)) and (not isinstance(value, bool)):
            held_84 = float(value)
            return held_84
        return None

    def buyable(self, ident: str) -> bool:
        if not self.sold_out(ident):
            if self.stock_key:
                value = (self.rows.get(ident) or {}).get(self.stock_key)
            else:
                value = None
            held_85 = not (False is value or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (0 >= value)))
            return held_85
        else:
            return False

    def sold_out(self, ident: str) -> bool:
        held_86 = ident in self.gone or ident in self.unavailable
        return held_86

    def note_view(self, name: str, observation: Any) -> None:
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.view_fields.setdefault(name, set()).update(keys)
            version, have = self.listing_fields.get(ident, (-1, set()))
            self.listing_fields[ident] = (self.version, (have if self.version == version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.listing_fields.get(ident, (-1, set()))
        keys = self.view_fields.get(name)
        held_87 = bool(keys) and self.version == version and (have >= keys) and (ident not in self.gone)
        return held_87

    def result_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.shapes:
            held_88 = self.shapes[name]
            return held_88
        elif isinstance(observation, dict):
            lists = list((k for k, v in observation.items() if isinstance(v, list) and 2 <= len(v) and all((isinstance(r, dict) for r in v))))
            if lists:
                self.shapes[name] = None
                sample = dict(((k, v[:3] if k in lists else v) for k, v in observation.items()))
                prompt = NAME_PARTS + '\n\nTOOL: ' + name + ' | parameters: ' + flat(self.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + flat(cut(sample, 120))[:4000]
                held_89 = self.model_json(prompt, [WALKER, THINKER], 1500)
                if not held_89:
                    held_89 = {}
                got = held_89
                if got.get('rows') in lists:
                    rows = got.get('rows')
                else:
                    rows = None
                if not rows:
                    return None
                first, params = (observation[rows][0], (self.schemas.get(name) or {}).get('properties') or {})
                pick = lambda key, where: key if isinstance(key, str) and key in where else None
                shape = {'rows': rows, 'count': pick(got.get('count'), observation), 'next': pick(got.get('next'), observation), 'next_arg': pick(got.get('next_arg'), params), 'price': pick(got.get('price'), first), 'stock': pick(got.get('stock'), first)}
                if shape['count']:
                    total = observation.get(shape['count'])
                else:
                    total = None
                if shape['count']:
                    if isinstance(total, bool) or not isinstance(total, int):
                        shape['count'] = None
                self.shapes[name] = shape
                self.price_key = self.price_key or shape['price']
                self.stock_key = self.stock_key or shape['stock']
                self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
                return shape
            else:
                return None
        else:
            return None

    def page_state(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.result_shape(name, observation)
        if not (not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict))):
            rows, count, value = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
            if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)):
                return None
            elif not (None is not value and (isinstance(value, bool) or not isinstance(value, (str, int)))):
                held_90 = (count, len(rows), shape['next_arg'], value)
                return held_90
            else:
                return None
        else:
            return None

    def in_price_order(self, observation: Any) -> bool:
        prices = list((self.price_of(i) for i, _ in self.records_in(observation)))
        held_91 = 3 <= len(prices) and None not in prices and all((b >= a for a, b in zip(prices, prices[1:])))
        return held_91

    def read_on(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        if pick in ('cheapest', 'most', 'least'):
            state = self.page_state(name, observation)
        else:
            state = None
        if state:
            count, shown, param, value = state
            given = args.get(param)
            if isinstance(given, int) and (not isinstance(given, bool)):
                paging_itself = given not in (None, 0, 1)
            else:
                paging_itself = bool(str(given or '').strip())
            if not (None is value or paging_itself or shown >= count or ('cheapest' == pick and self.in_price_order(observation))):
                pages = -(-(count - shown) // shown)
                if not (2 < pages or EXTRA_PAGES < self.pages_read + pages or self.left() - self.tail() - pages < 12):
                    texts = []
                    for _ in range(pages):
                        faults: list[str] = []
                        call = bend({**args, param: value}, self.schemas.get(name), 'args', faults)
                        if not faults:
                            reply = self.store(name, call)
                            if not (None is reply or self.finished):
                                self.pages_read = self.pages_read + 1
                                seen = reply.get('observation')
                                self.own_call_seen(name, seen)
                                self.note_search(name, call, seen)
                                self.paged_obs.append(seen)
                                text = self.show(seen)
                                self.read_cache[name + flat(call)] = (self.version, self.step, text)
                                held_94 = flat(value)
                                held_93 = '{}={}: {}'.format(param, held_94, text)
                                texts.append(held_93)
                                after = self.page_state(name, seen)
                                if after:
                                    value = after[3]
                                else:
                                    value = None
                                if None is value:
                                    break
                            else:
                                break
                        else:
                            break
                    self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
                    if texts:
                        held_95 = '\n[Kestrel read the rest of this search itself, one step a page]\n' + '\n'.join(texts)
                        return held_95
                    return ''
                else:
                    held_92 = "\n[Kestrel] This search matched {} listings and showed {}, in no price order. The pick weighs every fitting listing: search with narrower words (the requirement's value) or page on with {}={}.".format(count, shown, param, flat(value))
                    return held_92
            else:
                return ''
        else:
            return ''

    def cheapest_line(self, observations: list[Any]) -> str:
        if not (self.brief.get('pick') != 'cheapest' or not self.price_key):
            found = set((i for seen in observations for i, _ in self.records_in(seen)))
            rows = sorted(((self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None))
            if not 2 > len(rows):
                held_96 = '\n[Kestrel] buyable rows here, cheapest first: ' + ', '.join(('{} ({:g})'.format(i, p) for p, i in rows[:6]))
                return held_96
            else:
                return ''
        else:
            return ''

    def own_call_seen(self, name: str, observation: Any) -> str:
        self.take_in(observation)
        lines = self.loose_lines(observation)
        if lines:
            meaning = self.take_note(lines)
        else:
            meaning = ''
        if meaning:
            held_97 = '(seen on {}) {}'.format(name, meaning)
            self.queued.append(held_97)
        return meaning

    def read_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        arg = self.one_id_arg(tool)
        if not (not tool or not arg):
            texts = []
            for ident in list((i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i)))):
                if not (self.left() <= keep or self.finished or self.shows_facts.get(tool) is False):
                    faults: list[str] = []
                    args = bend({arg: ident}, self.schemas.get(tool), 'args', faults)
                    if not faults:
                        reply = self.store(tool, args)
                        if not None is reply:
                            seen = reply.get('observation')
                            self.own_call_seen(tool, seen)
                            self.note_view(tool, seen)
                            if isinstance(seen, dict):
                                if self.id_of(seen):
                                    self.shows_facts[tool] = self.shows_facts.get(tool, False) or bool(attr_block(seen))
                            texts.append('{}: '.format(ident) + self.show(seen)[:1500])
                        else:
                            break
                    else:
                        break
                else:
                    break
            if texts:
                self.trace.append({'kind': 'read_facts', 'n': len(texts)})
            return texts
        else:
            held_98 = []
            return held_98

    def cheaper_unread(self) -> list[str]:
        if not self.brief.get('pick') != 'cheapest':
            mark, cat = (self.cart_mark() + '#{}'.format(self.version), self.category_field())
            if not (mark in self.cheaper_marks or not cat):
                self.cheaper_marks.add(mark)
                cart, picks = (self.cart_ids(), [])
                kind_of = lambda i: str((self.rows.get(i) or {}).get(cat) or '').strip().lower()
                for line in cart:
                    price, kind = (self.price_of(line), kind_of(line))
                    if not (None is price or not kind):
                        pool = list((i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (kind_of(i) == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)))
                        pool.sort(key=lambda i: (-max((self.fit_score(i, w) for w in self.wants), default=0), self.price_of(i) or 0.0))
                        picks += list((i for i in pool[:2] if i not in picks))
                    else:
                        continue
                held_101 = self.read_facts(picks[:CHEAP_OPENS], self.tail() + 4)
                return held_101
            else:
                held_100 = []
                return held_100
        else:
            held_99 = []
            return held_99

    def look_wider(self) -> None:
        tool = self.roles.get('compare')
        held_102 = self.schemas.get(tool or '')
        if not held_102:
            held_102 = {}
        spec = held_102
        held_103 = spec.get('properties')
        if not held_103:
            held_103 = {}
        props = held_103
        arg = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in type_list(v)), None)
        if not tool or not arg or any((arg != k for k in spec.get('required') or [])) or self.compare_useless():
            self.look_wider_singly()
            return
        elif not self.left() <= 12:
            if isinstance(props[arg].get('maxItems'), int):
                cap = props[arg].get('maxItems')
            else:
                cap = 4
            groups = self.want_groups()
            share = max(1, cap // max(1, len(groups)))
            picks = list((i for ids in groups for i in list((x for x in ids if x not in self.full))[:share]))[:cap]
            if not 2 > len(picks):
                faults: list[str] = []
                args = bend({arg: picks}, spec, 'args', faults)
                if faults:
                    return
                before = len(self.full)
                reply = self.store(tool, args)
                if None is reply:
                    return
                seen = reply.get('observation')
                self.own_call_seen(tool, seen)
                self.note_view(tool, seen)
                rows = self.records_in(seen)
                if rows:
                    self.shows_facts[tool] = self.shows_facts.get(tool, False) or any((attr_block(r) for _, r in rows))
                self.trace.append({'kind': 'look_wider', 'ids': picks, 'opened': len(self.full) - before})
                if rows and self.shows_facts.get(tool) is False:
                    self.look_wider_singly()
                elif before == len(self.full):
                    if 2 < len(picks):
                        if self.left() > 12:
                            faults = []
                            args = bend({arg: picks[:2]}, spec, 'args', faults)
                            if not faults:
                                again = self.store(tool, args)
                                if None is not again:
                                    held_104 = again.get('observation')
                                    self.own_call_seen(tool, held_104)
            else:
                return
        else:
            return

    def look_wider_singly(self) -> None:
        tool = self.facts_tool()
        if not (not tool or not self.one_id_arg(tool)):
            lanes: list[list[str]] = []
            for ids in self.want_groups():
                opened = sum((1 for i in ids if i in self.full))
                pool = list((i for i in ids if i not in self.full and self.buyable(i)))
                if self.brief.get('pick') == 'cheapest':
                    pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
                lanes.append(pool[:max(0, min(2, 8 - opened))])
            queue = list((lane[k] for k in range(2) for lane in lanes if len(lane) > k))
            self.read_facts(queue[:5], 22)
        else:
            return

class Talk:

    def needs_now(self) -> list[tuple[int, str]]:
        held_105 = list(((n, need) for n, w in enumerate(self.wants, 1) for need in w.musts + w.told if need not in w.dropped))
        return held_105

    def order_closest(self) -> bool:
        held_106 = str(self.brief.get('no_fit') or '').strip().lower() == 'order_closest'
        return held_106

    def want_at(self, raw: Any) -> Want:
        held_107 = self.wants[min(max(1, whole(raw) or 1), len(self.wants)) - 1]
        return held_107

    def ledger(self) -> str:
        parts, many, brief = ([], 1 < len(self.wants), self.brief)
        for number, w in enumerate(self.wants, 1):
            needs = list((m + (' (the page it points at gives: {})'.format(self.ref_values[m]) if m in self.ref_values else '') for m in w.musts + w.told if m not in w.dropped))
            head = ('item {}: {}'.format(number, w.name) if many else 'item: {}'.format(w.name)) + (' (store label: {})'.format(w.shelf) if w.shelf else '')
            text = '{}, quantity {}; requirements: '.format(head, w.qty) + ('; '.join(needs) if needs else 'none beyond the kind of product')
            parts.append(text + ('; shopper does not mind: ' + ', '.join(w.free) if w.free else ''))
        if None is not self.budget:
            if brief.get('budget_scope') == 'each':
                scope = 'for each item'
            else:
                scope = 'in total'
            held_108 = ' '.join((x for x in ('budget: at most', str(self.budget), brief.get('budget_unit') or '', scope) if x))
            parts.append(held_108)
        pick = brief.get('pick')
        if 'cheapest' == pick:
            parts.append('pick: the cheapest listing that fits (lowest price)')
        elif not pick in ('most', 'least'):
            parts.append('pick: any listing that fits everything')
        else:
            held_109 = ' '.join((x for x in ('pick: the', pick, brief.get('pick_measure') or '', 'among fitting listings') if x))
            parts.append(held_109)
        if brief.get('in_stock_only', True) is not False:
            parts.append('in stock only')
        parts += list(('wish (tie-breaker only): {}'.format(w) for w in as_list(brief.get('wishes'))))
        parts += list(('rule from the request: {}'.format(r) for r in as_list(brief.get('rules'))))
        if self.order_closest():
            parts.append('if no listing meets everything: the request wants its nearest match bought anyway (the one missing the least, picked the way the request describes); an ending without a purchase cannot satisfy this request')
        if brief.get('quiet_decline') is True:
            parts.append('if nothing fits: end without naming or suggesting any listing')
        when_text = {'choose': 'before choosing', 'order': 'at the order (Kestrel does it when you check out)', 'end': 'at an ending without a purchase (Kestrel does it then)'}
        parts += list(('step the request asks for, {}: {}'.format(when_text[p['when']], p['step']) for p in self.process))
        facts = self.facts_tool()
        if facts:
            if self.shows_facts.get(facts) is True or self.roles.get('offer'):
                bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.shows_facts.get(t) is False))
                pages = sorted((t for t in self.page_tools() if facts != t))
                parts.append('listing attributes come from {}'.format(facts) + ('; {} show none'.format(', '.join(bare)) if bare else '') + ("; a listing's page (what a chain of pages or a page check means) is shown by {}".format(' or '.join(pages)) if pages else ''))
        held_110 = '\n'.join(('- ' + p for p in parts))
        return held_110

    def status(self) -> str:
        left = self.left()
        text = '[Kestrel] step {} of {} used, {} left.'.format(self.step, self.max_steps, left)
        if self.elapsed() > CLOCK_SOFT:
            text += ' TIME IS NEARLY UP: put the chosen lines in the cart at the requested quantities and check out now.'
        elif left <= self.tail() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif 6 >= left or (len(self.wants) + 5 >= left and 0 == self.buys):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif 0 == self.buys and 12 >= left:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif 0 == self.buys:
            if not self.cart_touched:
                if self.max_steps // 2 <= self.step:
                    text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        live = set((need for _, need in self.needs_now()))
        open_refs = list((need for _, need in self.refs if need in live and need not in self.ref_values))
        if open_refs:
            if 0 == self.buys:
                text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(open_refs[:3])
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        parts = {'progress by item': self.progress_text(), 'what this store has shown': self.world_text(), 'ledger': '\n[ledger]\n' + self.ledger()}
        self.status_calls = self.status_calls + 1
        held_111 = 1 == self.status_calls % 5
        if not held_111:
            held_111 = self.status_stale
        full = held_111
        self.status_stale = False
        same = []
        for key, body in parts.items():
            if full or self.status_sent.get(key) != body:
                text += body
                self.status_sent[key] = body
            elif body:
                same.append(key)
        if same:
            text += '\n[' + ', '.join(same) + ': unchanged since the last time they were shown above]'
        return text

    def progress_text(self) -> str:
        if not 2 > len(self.wants):
            cat = self.category_field()
            if not self.cart_fresh():
                cart = self.cart_lines
            else:
                cart = dict(((i, self.cart_lines.get(i, 1)) for i in self.cart_ids()))
            lines, todo = ([], [])
            for number, w in enumerate(self.wants, 1):
                ids = list((i for i in self.seen_order if cat and self._label_is(str(self.rows[i].get(cat) or ''), w)))
                opened = sum((1 for i in ids if i in self.full))
                held = list(((i, n) for i, n in cart.items() if cat and self._label_is(str((self.rows.get(i) or {}).get(cat) or ''), w)))
                if held:
                    state = ', '.join(('{} x{}'.format(i, n) for i, n in held))
                else:
                    state = 'nothing in the cart'
                if held:
                    if sum((n for _, n in held)) != w.qty:
                        state += ' (needs {})'.format(w.qty)
                held_112 = 'item {} {}: {} listings seen, {} opened, {}'.format(number, w.name, len(ids), opened, state)
                lines.append(held_112)
                if not held:
                    todo.append(number)
            if todo:
                share = ' About {} steps are left for each of them.'.format(self.left() // len(todo))
            else:
                share = ''
            if todo:
                ahead = ' Still to do: item{} {}, so give each its turn before polishing a done one.{}'.format('s' if 1 < len(todo) else '', ', '.join(map(str, todo)), share)
            else:
                ahead = ''
            held_113 = '\n[progress by item]\n' + '\n'.join(('- ' + x for x in lines)) + ahead
            return held_113
        else:
            return ''

    def refused(self, observation: Any) -> bool:
        if isinstance(observation, dict):
            held_114 = any((re.search('error|problem|fail|refus', str(k), re.I) and isinstance(v, str) and v.strip() for k, v in observation.items()))
            return held_114
        else:
            return False

    def world_due(self, name: str, seen: Any) -> str:
        if not (3 <= self.world_reads or SPEND_SOFT < self.cost or self.elapsed() > CLOCK_SOFT - 60):
            views = set((self.roles.get(r) for r in ('open', 'offer', 'compare'))) - {None}
            called = set((line.split(' ', 3)[2] for line in self.world_log if line.startswith('[step') and len(line.split(' ', 3)) > 2))
            reasons = []
            if self.roles.get('search') in called:
                if called & views:
                    reasons.append('views')
            held_115 = name == self.roles.get('cart_view')
            if not held_115:
                held_116 = name == self.roles.get('cart')
                if held_116:
                    held_116 = any((holds_id(seen, i) for i in self.cart_lines))
                held_115 = held_116
            shows_cart = held_115
            if self.cart_touched:
                if shows_cart:
                    reasons.append('cart')
            if self.refused(seen):
                held_117 = 'refusal{}'.format(self.step)
                reasons.append(held_117)
            fresh = list((r for r in reasons if r not in self.world_marks))
            if fresh:
                held_118 = fresh[0]
                return held_118
            return ''
        else:
            return ''

    def read_world(self, reason: str) -> None:
        self.world_marks.add(reason)
        self.world_reads = self.world_reads + 1
        tools = list(('- {}: {} | parameters: '.format(t['function']['name'], t['function'].get('description', '')) + flat(t['function'].get('parameters') or {}) for t in self.tools))
        log = self.world_log[:3] + self.world_log[3:][-22:]
        prompt = WORLD_FACTS + '\n\nTOOLS:\n' + '\n'.join(tools) + '\n\nCALLS SO FAR (oldest first, replies shortened):\n' + '\n'.join(log) + ('\n\nNOTES SO FAR: ' + flat(self.world) if self.world else '')
        got = self.model_json(prompt, [WALKER], 2500, rescue=False)
        if isinstance(got, dict):
            self.world = got
            self.trace.append({'kind': 'world', 'why': reason, 'facts': cut(got, 120)})
            held_119 = got.get('open_many_limit')
            limit = whole(held_119)
            if limit:
                if 1 <= limit:
                    self.world_compare = limit
        else:
            return

    def world_text(self) -> str:
        w = self.world
        if not (not w and (not self.page_most)):
            lines = []
            if isinstance(w.get('search'), dict):
                search = dict(w.get('search'))
            else:
                search = {}
            if self.page_most:
                search['rows_per_page'] = self.page_most
            if search.get('rows_per_page'):
                lines.append('search shows {} rows a page'.format(search['rows_per_page']) + ('; more: {}'.format(search['next_page']) if search.get('next_page') else ''))
                held_120 = search.get('rows_per_page')
                rows = whole(held_120)
                if rows:
                    if 5 >= rows:
                        held_121 = "with only {} rows a page, one page shows a sliver of the matches: before judging that nothing fits or picking the cheapest, read at least three pages of your best query (or narrow it with a requirement's value, or filter by category or price)".format(rows)
                        lines.append(held_121)
            if whole(w.get('open_many_limit')):
                held_123 = whole(w['open_many_limit'])
                held_122 = 'the side-by-side view takes at most {} listings a call'.format(held_123)
                lines.append(held_122)
            if isinstance(w.get('cart'), dict):
                cart = w.get('cart')
            else:
                cart = {}
            meaning = {'units': "is the line's unit count (0 removes the line)", 'position': 'is a line position, not a count', 'whole_cart_list': 'belongs to one list that replaces the whole cart', 'one_per_call': 'does not count: each call adds one unit'}
            if cart.get('amount_arg'):
                if cart.get('amount_meaning') in meaning:
                    held_124 = 'cart: {} {}'.format(cart['amount_arg'], meaning[cart['amount_meaning']])
                    lines.append(held_124)
            if cart.get('remove_how'):
                held_125 = 'to take a listing out of the cart: {}'.format(str(cart['remove_how'])[:160])
                lines.append(held_125)
            lines += list((str(s)[:200] for s in as_list(w.get('surprises'))[:5]))
            if lines:
                held_126 = '\n[what this store has shown so far]\n' + '\n'.join(('- ' + x for x in lines))
                return held_126
            return ''
        else:
            return ''

    def time_to_ask(self) -> bool:
        if not (self.asked or not self.roles.get('ask') or self.msgs_free() <= 0 or (self.left() <= 8)):
            held_127 = 3 * len(self.wants) + 1 <= len(self.full) or max(8, self.max_steps // 4) <= self.step
            return held_127
        else:
            return False

    def message_args(self, text: str) -> dict[str, Any] | None:
        held_128 = self.schemas.get(self.roles.get('ask') or '')
        if not held_128:
            held_128 = {}
        spec = held_128
        held_129 = spec.get('properties')
        if not held_129:
            held_129 = {}
        props = held_129
        free_text = list((k for k, v in props.items() if isinstance(v, dict) and 'string' in type_list(v) and ('enum' not in v)))
        held_130 = next((k for k in spec.get('required') or [] if k in free_text), None)
        if not held_130:
            if free_text:
                held_130 = free_text[0]
            else:
                held_130 = None
        arg = held_130
        if arg:
            args: dict[str, Any] = {arg: text}
            for key in spec.get('required') or []:
                if not key in args:
                    options = (props.get(key) or {}).get('enum')
                    if isinstance(options, list) and options:
                        args[key] = options[0]
                    else:
                        return None
                else:
                    continue
            faults: list[str] = []
            args = bend(args, spec, 'args', faults)
            if faults:
                return None
            return args
        else:
            return None

    def ask_name(self, number: int) -> str:
        name = self.wants[number - 1].name.strip()
        parts = list((p.strip() for p in re.split('\\s*(?:&|/|,)\\s*', name) if p.strip()))
        if not (2 > len(parts) or (len(parts[0].split()) == 1 and (not parts[0].lower().endswith('s')))):
            said = set((w[:-1] if 3 < len(w) and w.endswith('s') else w for w in re.findall('[a-z0-9]+', self.request.lower())))

            def fit(part: str) -> float:
                words = set((w[:-1] if 3 < len(w) and w.endswith('s') else w for w in re.findall('[a-z0-9]+', part.lower())))
                held_131 = len(words & said) / max(len(words), 1)
                return held_131
            best = max(parts, key=lambda p: (fit(p), -parts.index(p)))
            if name != best:
                self.trace.append({'kind': 'ask_name', 'from': name[:60], 'to': best[:40]})
            return best
        else:
            return name

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        held_132 = self.ask_name(number)
        text = 'What {} would you like for the {}'.format(label, held_132)
        held_133 = text + (', for example {} or {}?'.format(samples[0], samples[1]) if samples else '?')
        return held_133

    def samples(self, number: int, label: str) -> list[str]:
        settled = ' '.join(self.wants[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            if name == key_words(label):
                held_134 = list((v for v in values if v and v not in settled and (30 >= len(v))))[:2]
                return held_134
        held_135 = []
        return held_135

    def plan_picks(self, count: int) -> list[tuple[int, str]]:
        settled = list(('item {}: {} - settled by the request: '.format(n, w.name) + ('; '.join(w.musts) or 'nothing') for n, w in enumerate(self.wants, 1)))
        table_lines, last = ([], None)
        for index, (label, name, carried, values) in enumerate(self.key_table(), 1):
            if last != label:
                held_136 = 'category {!r}:'.format(label)
                table_lines.append(held_136)
                last = label
            held_137 = self.key_aliases.get((label, name))
            if not held_137:
                held_137 = []
            others = held_137
            table_lines.append('  {}. {} (on {})'.format(index, name, carried) + (' [also named: {}]'.format(', '.join(others[:4])) if others else '') + ': ' + ' | '.join(values))
        if 'code' == PLAN_MODE:
            if not as_list(self.brief.get('ask_first')):
                picks = self.table_picks(count + 3)
                if picks:
                    held_138 = self.cover_items(picks, count)
                    return held_138
        prompt = PLAN_QUESTIONS.replace('COUNT', str(count + 3)) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        if any(self.key_aliases.values()):
            prompt += '\n\nA key shown with other names is one property that several sellers name differently; count it once.'
        first = list((str(x) for x in as_list(self.brief.get('ask_first')) if str(x).strip()))
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(table_lines) or '(no full listing records yet)')
        held_139 = self.model_json(prompt, [THINKER, WALKER], 3000)
        if not held_139:
            held_139 = {}
        plan = held_139
        picks: list[tuple[int, str]] = []
        for entry in as_list(plan.get('ask')):
            if isinstance(entry, dict):
                held_140 = max(1, whole(entry.get('item') or 1) or 1)
                held_141 = len(self.wants)
                number = min(held_140, held_141)
                label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(entry.get('label') or ''))[0].strip().rstrip('?.')
                if label:
                    if len(label.split()) <= 5:
                        if all((key_words(label) != key_words(o) for n, o in picks if number == n)):
                            picks.append((number, label))
            else:
                continue
        self.trace.append({'kind': 'plan', 'ask': cut(plan.get('ask'), 120)})
        held_142 = self.cover_items(picks or self.table_picks(count + 3), count)
        return held_142

    def cover_items(self, picks: list[tuple[int, str]], count: int) -> list[tuple[int, str]]:
        n = len(self.wants)
        if not (2 > n or n > count):
            have = set((number for number, _ in picks))
            if n > len(have):
                table = self.table_picks(40)
                for number in range(1, n + 1):
                    if number not in have:
                        extra = next((p for p in table if number == p[0]), None)
                    else:
                        extra = None
                    if extra:
                        picks.append(extra)
            firsts, rest, seen = ([], [], set())
            for pick in picks:
                (rest if pick[0] in seen else firsts).append(pick)
                seen.add(pick[0])
            ordered = firsts + rest
            if picks != ordered:
                self.trace.append({'kind': 'cover_items', 'order': ordered[:8]})
            return ordered
        else:
            return picks

    def table_picks(self, limit: int) -> list[tuple[int, str]]:
        order = sorted(enumerate(self.wants, 1), key=lambda pair: len(pair[1].musts))
        lanes = []
        for number, w in order:
            rows = list((name for label, name, _, _ in self.key_table() if 1 == len(self.wants) or self._label_is(label, w)))
            said = ' '.join(w.musts).lower()
            aliases = dict(((name, [name] + (self.key_aliases.get((label, name)) or [])) for label, name, _, _ in self.key_table()))
            held_143 = list(((number, name) for name in rows if not any((a in said for a in aliases.get(name, [name])))))
            lanes.append(held_143)
        turns = list((lane[k] for k in range(max(map(len, lanes), default=0)) for lane in lanes if len(lane) > k))
        self.trace.append({'kind': 'table_plan', 'n': len(turns)})
        held_144 = turns[:limit]
        return held_144

    def _label_is(self, label: str, want: Want) -> bool:
        long_words = set((x for x in key_stem(want.shelf or '') | key_stem(want.name) if 4 <= len(x)))
        held_145 = key_words(label) in (key_words(want.shelf or ''), key_words(want.name)) or bool(set((x for x in key_stem(label) if 4 <= len(x))) & long_words)
        return held_145

    def interview(self) -> str:
        self.plan_tries = self.plan_tries + 1
        count = self.msgs_free()
        if not 0 >= count:
            if 1 == self.plan_tries:
                self.look_wider()
            queue = self.plan_picks(count)
            if queue:
                self.asked = True
                sent, told, retry = ([], [], [])
                retried: set[tuple[int, str]] = set()
                tried: set[tuple[int, str]] = set()
                while self.msgs_free() > 0 and self.elapsed() < CLOCK_SOFT:
                    if not queue:
                        queue = list((p for p in self.table_picks(40) if (p[0], key_words(p[1])) not in tried))
                        if queue:
                            self.trace.append({'kind': 'table_refill', 'n': len(queue)})
                        else:
                            break
                    number, label = queue.pop(0)
                    tried.add((number, key_words(label)))
                    text = self.question(number, label)
                    batch = self.send([(number, label, text)])
                    if batch:
                        sent += batch
                        heard = self.hear(batch, (WALKER, THINKER))
                        self.take_heard(heard, told)
                        revealed = set((min(max(1, whole(e.get('item')) or number), len(self.wants)) for e in as_list(heard.get('needs')) if isinstance(e, dict) and e.get('need')))
                        if revealed:
                            queue = list((q for q in queue if q[0] not in revealed)) + list((q for q in queue if q[0] in revealed))
                        if as_list(heard.get('crowded')) + as_list(heard.get('unclear')):
                            if (number, label) not in retried:
                                retried.add((number, label))
                                retry.append((number, label))
                    else:
                        break
                if sent:
                    if told:
                        self.version = self.version + 1
                    summary = "Kestrel's interviewer asked the shopper (each question used one step):\n" + '\n'.join(('Q: {}\nA: {}'.format(q, a) for _, q, a in sent))
                    summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
                    summary += '. Apply them to every pick (search again with the new values where needed).'
                    pointed = list((need for _, need in self.refs if any((need in t for t in told))))
                    if pointed:
                        summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
                    if self.queued:
                        summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
                        self.queued = []
                    return summary
                else:
                    return ''
            else:
                self.asked = 2 <= self.plan_tries
                return ''
        else:
            self.asked = True
            return ''

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool, sent = (self.roles['ask'], [])
        for number, _, text in questions:
            if not (self.finished or self.left() <= floor or self.msg_limit <= self.msg_used):
                args = self.message_args(text)
                if args:
                    reply = self.store(tool, args)
                else:
                    reply = None
                if not None is reply:
                    self.msg_used = self.msg_used + 1
                    answer = reply.get('observation')
                    words = self.spoken(answer)
                    sent.append((number, text, words))
                    held_146 = 'step {}: message to the shopper {!r} -> reply {!r} (cart then: {})'.format(self.step, text, words[:300], self.cart_mark())
                    self.done_log.append(held_146)
                    self.done_steps.append(self.step)
                    self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
                    if isinstance(answer, dict):
                        if answer.get('error'):
                            break
                else:
                    break
            else:
                break
        return sent

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(THINKER, WALKER)) -> dict[str, Any]:
        canned = self.canned_reply(sent)
        if not None is not canned:
            prompt = READ_ANSWERS + '\n\nITEMS:\n' + '\n'.join(('item {}: {}'.format(n, w.name) for n, w in enumerate(self.wants, 1)))
            prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join(('{}. (about item {}) Q: {}\nA: {}'.format(i, n, q, a) for i, (n, q, a) in enumerate(sent, 1)))
            held_148 = list(models)
            held_147 = self.model_json(prompt, held_148, 1500)
            if not held_147:
                held_147 = {}
            got = held_147
            held_149 = ' '.join((str(a) for _, _, a in sent))
            self.written_budget(got, held_149)
            return got
        else:
            return canned

    def written_budget(self, got: dict[str, Any], text: str) -> None:
        value = got.get('budget')
        if not (not isinstance(value, (int, float)) or isinstance(value, bool) or 0 >= value):
            written = set((float(n.replace(',', '')) for n in re.findall('\\d[\\d,]*(?:\\.\\d+)?', text)))
            if float(value) not in written:
                if 'cent' in text.lower():
                    if float(round(value * 100)) in written:
                        self.trace.append({'kind': 'budget_unit', 'read': value, 'kept': round(value * 100)})
                        got['budget'] = round(value * 100)
        else:
            return

    def canned_reply(self, sent: list[tuple[int, str, str]]) -> dict[str, Any] | None:
        if not 1 != len(sent):
            number, question, answer = sent[0]
            text = str(answer or '')
            if not (not text or re.search('\\d', text) or CANNED_FIRM.search(text)):
                low = text.lower()
                asked = re.search('what (.+?) would you like', question, re.I)
                names = set((key_words(name) for _, name, _, _ in self.key_table())) | ({key_words(asked.group(1))} if asked else set())
                if not any((w in low for name in names for w in name.split() if 4 <= len(w))):
                    if CANNED_UNCLEAR.search(text):
                        if not CANNED_FREE.search(text):
                            self.trace.append({'kind': 'canned', 'as': 'unclear', 'a': text[:80]})
                            held_150 = {'unclear': [1]}
                            return held_150
                    if CANNED_FREE.search(text):
                        if not CANNED_UNCLEAR.search(text):
                            self.trace.append({'kind': 'canned', 'as': 'free', 'a': text[:80]})
                            held_151 = {'free': [{'item': number, 'property': asked.group(1) if asked else ''}]}
                            return held_151
                    return None
                else:
                    return None
            else:
                return None
        else:
            return None

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        many = 1 < len(self.wants)
        for entry in as_list(heard.get('needs')):
            if not (not isinstance(entry, dict) or not str(entry.get('need') or '').strip()):
                held_152 = max(1, whole(entry.get('item')) or 1)
                held_153 = len(self.wants)
                number = min(held_152, held_153)
                need = str(entry['need']).strip()
                if need not in self.wants[number - 1].told:
                    self.wants[number - 1].told.append(need)
                    told.append('item {}: {}'.format(number, need) if many else need)
                if entry.get('points') is True:
                    if (number, need) not in self.refs:
                        self.refs.append((number, need))
            else:
                continue
        for entry in as_list(heard.get('drop')):
            if isinstance(entry, dict):
                if str(entry.get('property') or '').strip():
                    held_154 = entry.get('item')
                    w = self.want_at(held_154)
                    prop = key_words(str(entry['property']).split(':')[0])
                    hit = list((m for m in w.musts + w.told if m not in w.dropped and key_words(m.split(':')[0]) == prop))
                    w.dropped.extend(hit)
                    told.extend(('dropped {}'.format(m) for m in hit))
        for entry in as_list(heard.get('free')):
            if isinstance(entry, dict):
                if entry.get('property'):
                    self.want_at(entry.get('item')).free.append(str(entry['property']))
        budget = heard.get('budget')
        if isinstance(budget, (int, float)):
            if not isinstance(budget, bool):
                if 0 < budget:
                    self.budget = budget

    def spoken(self, observation: Any) -> str:
        if not isinstance(observation, str):
            found: list[str] = []

            def visit(node: Any, depth: int) -> None:
                if 4 < depth:
                    return
                elif isinstance(node, str) and len(node.split()) >= 2:
                    found.append(node)
                elif isinstance(node, dict):
                    for value in node.values():
                        visit(value, depth + 1)
                elif isinstance(node, list):
                    for value in node:
                        visit(value, depth + 1)
            visit(observation, 0)
            if found:
                held_155 = ' '.join(found)[:800]
                return held_155
            held_156 = flat(observation)[:400]
            return held_156
        else:
            return observation

    def loose_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict):
            if not self.id_of(observation):
                for value in observation.values():
                    if isinstance(value, str) and len(value.split()) >= 5:
                        lines.append(value)
                    elif isinstance(value, list) and value and all((isinstance(v, str) for v in value)):
                        lines += list((v for v in value if len(v.split()) >= 5))
                    elif isinstance(value, dict):
                        if not self.id_of(value, False, 1):
                            lines += list((v for v in value.values() if isinstance(v, str) and len(v.split()) >= 6))
        held_157 = list((line for line in lines if line not in self.heard))
        return held_157

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = READ_NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.ledger() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        held_158 = self.model_json(prompt, [WALKER, BACKUP], 1200)
        if not held_158:
            held_158 = {}
        got = held_158
        held_159 = ' '.join(lines)
        self.written_budget(got, held_159)
        gone = list((str(g) for g in as_list(got.get('gone')) if isinstance(g, (str, int))))
        self.unavailable.update((g for g in gone if g in self.rows))
        self.note_sold = bool(gone) or got.get('sold_out') is True
        if self.note_sold:
            if 1 == len(lines):
                self.sold_lines.update(lines)
        if not got.get('kind') not in (None, 'change'):
            changed = []
            budget = got.get('budget')
            if isinstance(budget, (int, float)):
                if not isinstance(budget, bool):
                    if 0 < budget:
                        if self.budget != budget:
                            self.budget = budget
                            held_160 = 'budget now {}'.format(budget)
                            changed.append(held_160)
            for entry in as_list(got.get('drop')):
                if not (not isinstance(entry, dict) or not str(entry.get('need') or '').strip()):
                    w, target = (self.want_at(entry.get('item')), str(entry['need']).strip())
                    live = list((m for m in w.musts + w.told if m not in w.dropped))
                    held_161 = list((m for m in live if target == m))
                    if not held_161:
                        held_161 = list((m for m in live if key_words(m.split(':')[0]) == key_words(target.split(':')[0])))[:1]
                    hit = held_161
                    w.dropped.extend(hit or [target])
                    held_162 = 'dropped {}'.format(target)
                    changed.append(held_162)
                else:
                    continue
            for entry in as_list(got.get('add')):
                if not (not isinstance(entry, dict) or not str(entry.get('need') or '').strip()):
                    w, need = (self.want_at(entry.get('item')), str(entry['need']).strip())
                    if need in w.dropped:
                        w.dropped.remove(need)
                    if need not in w.musts + w.told:
                        w.told.append(need)
                    held_163 = 'added {}'.format(need)
                    changed.append(held_163)
                else:
                    continue
            if gone:
                changed.append('unavailable: ' + ', '.join(gone))
            if got.get('price_moved') is True:
                changed.append('a price changed')
            summary = str(got.get('summary') or '').strip()
            if changed or summary:
                self.version = self.version + 1
                self.recent = []
                self.notes.append(summary or '; '.join(changed))
            held_164 = (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()
            return held_164
        else:
            return ''

    def evidence(self, observation: Any) -> str:
        ids: list[str] = []

        def visit(node: Any, depth: int) -> None:
            if 6 < depth:
                return
            elif isinstance(node, dict):
                ident = self.id_of(node, False, depth)
                if not (ident and attr_block(node)):
                    for value in node.values():
                        visit(value, depth + 1)
                else:
                    if ident not in ids:
                        ids.append(ident)
                    return
            elif isinstance(node, list):
                for value in node:
                    visit(value, depth + 1)
        visit(observation, 0)
        needs = self.needs_now()
        if not (not ids or not needs):
            many, lines = (1 < len(self.wants), [])
            for ident in ids[:12]:
                held_165 = attr_block(self.full.get(ident) or {})
                if not held_165:
                    held_165 = {}
                block = held_165
                parts = []
                for number, need in needs[:8]:
                    prop = need.split(':', 1)[0].strip()
                    hits = list(('{}={}'.format(k, str(v)[:50]) for k, v in block.items() if names_same(key_stem(prop), key_stem(k))))
                    if hits:
                        parts.append(('(item {}) '.format(number) if many else '') + '{} -> '.format(prop) + ', '.join(hits[:3]))
                if parts:
                    lines.append('{}: '.format(ident) + '; '.join(parts))
            if lines:
                held_166 = "[evidence: keys whose names hold a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)
                return held_166
            else:
                return ''
        else:
            return ''

class Close:

    def cart_fresh(self) -> bool:
        held_167 = None is not self.cart_view and self.cart_at <= self.cart_seen_at
        return held_167

    def cart_text(self) -> str:
        if not self.cart_fresh():
            held_169 = "lines by Kestrel's record of the cart calls (listing: units): " + flat(self.cart_lines)
            return held_169
        else:
            held_168 = flat(cut(self.cart_view, 300))[:3000]
            return held_168

    def cart_ids(self) -> list[str]:
        if not self.cart_fresh():
            held_171 = list(self.cart_lines)
            return held_171
        else:
            text = flat(self.cart_view)
            held_170 = list((i for i in self.rows if i in text))
            return held_170

    def record_of(self, ident: str) -> str:
        held_172 = self.full.get(ident)
        if not held_172:
            held_172 = self.rows.get(ident)
        record = held_172
        if record:
            held_173 = flat(cut(record, 400))[:3000]
            return held_173
        return '(no record seen)'

    def other_records(self, skip: list[str], limit: int=30) -> str:
        held_174 = reversed(self.seen_order)
        order = list(held_174)
        if self.brief.get('pick') == 'cheapest':
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float('inf')))
        lines = []
        for ident in order:
            if not (ident in skip or self.sold_out(ident)):
                lines.append(flat(cut(self.full.get(ident) or self.rows.get(ident), 160))[:900])
                if limit <= len(lines):
                    break
            else:
                continue
        held_175 = '\n'.join(lines)
        return held_175

    def verdict_key(self, ids: list[str]) -> str:
        held_176 = '|'.join(sorted(ids)) + '#{}'.format(self.version)
        return held_176

    def review_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        planned = None is not ids
        if planned:
            ids = ids
        else:
            ids = self.cart_ids()
        key = self.verdict_key(ids)
        if not (key in self.reviewed or limit < self.cost or (not ids)):
            self.reviewed.add(key)
            prompt = REVIEW_CART + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
            if not planned:
                prompt += '\n\nCART:\n' + self.cart_text()
            else:
                qty = '; '.join(('item {}: quantity {}'.format(n, w.qty) for n, w in enumerate(self.wants, 1)))
                prompt += '\n\nCART (planned, judge these listings as if carted at the requested quantities: {}):\n'.format(qty) + ', '.join(ids)
            prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join(('{}: '.format(i) + ('SOLD OUT (the store says it can no longer be bought). ' if self.sold_out(i) else '') + self.record_of(i) for i in ids))
            if self.brief.get('pick') in ('cheapest', 'most', 'least'):
                prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.other_records(ids, 15)
            got = self.model_json(prompt, [THINKER, WALKER], 1500)
            self.trace.append({'kind': 'review', 'cart': ids, 'got': got})
            self.verdicts[key] = bool(got) and got.get('ok') is not False
            if not (not got or got.get('ok') is not False):
                self.leads = list((i for i in self.leads if i not in ids))
                problems = list((str(p) for p in as_list(got.get('problems'))))[:5]
                better = list((str(b) for b in as_list(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))))[:4]
                text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
                if better:
                    text += '. Listings that look better: ' + ', '.join(better)
                held_177 = text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'
                return held_177
            else:
                return ''
        else:
            return ''

    def follow_chains(self) -> tuple[str, bool]:
        live = set((need for _, need in self.needs_now()))
        pending = list((need for _, need in self.refs if need in live and need not in self.ref_values))
        held_178 = self.roles.get('offer')
        if not held_178:
            held_178 = self.roles.get('open')
        tool = held_178
        arg = self.one_id_arg(tool)
        seen, lost = ([], [])
        while pending and 4 > self.trails and (SPEND_HARD > self.cost):
            self.trails = self.trails + 1
            records = []
            for ident in reversed(self.seen_order):
                if ident in self.opened:
                    record = self.full.get(ident) or self.rows.get(ident)
                else:
                    record = None
                records.append('OPENED ' + flat(cut(record, 300))[:1400] if record else 'ROW ' + flat(cut(self.rows.get(ident), 80))[:300])
                if 60 <= len(records):
                    break
            prompt = FOLLOW_CHAIN + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending) + '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            held_179 = self.model_json(prompt, [WALKER, THINKER], 1800)
            if not held_179:
                held_179 = {}
            got = held_179
            self.trace.append({'kind': 'chain', 'pending': pending, 'got': cut(got, 150)})
            reach, lost = ([], [])
            for chain in as_list(got.get('chains')):
                if isinstance(chain, dict):
                    held_180 = next((p for p in pending if p == str(chain.get('need') or '').strip()), None)
                    if not held_180:
                        if 1 == len(pending):
                            held_180 = pending[0]
                        else:
                            held_180 = None
                    need = held_180
                    if need:
                        links = list((x for x in as_list(chain.get('links')) if isinstance(x, dict)))
                        ids = list((str(x.get('id') or '').strip() for x in links))
                        value = str(chain.get('value') or '').strip()
                        if not (chain.get('done') is True and links and value and all((i in self.opened for i in ids))):
                            unopened = list((i for i in ids if i in self.rows and i not in self.opened))
                            if not unopened:
                                held_181 = next((str(x.get('about') or '') for x, i in zip(links, ids) if i not in self.rows), '')
                                if not held_181:
                                    held_181 = 'the listing it points at'
                                about = held_181
                                held_182 = '{} -> find {}'.format(need, about)
                                lost.append(held_182)
                            else:
                                reach += list((i for i in unopened if i not in reach))
                        else:
                            self.ref_values[need] = value
                            self.version = self.version + 1
                            continue
                    else:
                        continue
                else:
                    continue
            pending = list((p for p in pending if p not in self.ref_values))
            if not (not reach or not tool or (not arg) or (self.left() < len(reach) + 4) or self.finished):
                for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                    faults: list[str] = []
                    args = bend({arg: ident}, self.schemas.get(tool), 'args', faults)
                    if not faults:
                        reply = self.store(tool, args)
                        if not (None is reply or self.finished):
                            held_183 = reply.get('observation')
                            self.own_call_seen(tool, held_183)
                            seen.append('{}: '.format(ident) + self.show(reply.get('observation'))[:1500])
                        else:
                            break
                    else:
                        break
            else:
                break
        if seen:
            text = '[Kestrel opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        else:
            text = ''
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join(('{} -> {}'.format(k, v) for k, v in self.ref_values.items())) + '\n'
        held_184 = bool(lost)
        if held_184:
            held_184 = 2 > self.chain_holds
        if held_184:
            held_184 = self.left() > 5
        hold = held_184
        if hold:
            self.chain_holds = self.chain_holds + 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + '. Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.'
        held_185 = (text, hold)
        return held_185

    def offer_tool(self) -> str | None:
        held_186 = self.roles.get('offer') or self.roles.get('open')
        return held_186

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        arg = self.one_id_arg(tool)
        if not (not tool or not arg):
            since = max(self.step - 2, self.refused_at)
            stale = list((i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone))
            if not (not stale or self.left() < len(stale) + 3):
                lines = []
                for ident in stale:
                    faults: list[str] = []
                    args = bend({arg: ident}, self.schemas.get(tool), 'args', faults)
                    if not faults:
                        before = dict(self.rows.get(ident) or {})
                        reply = self.store(tool, args)
                        if not (None is reply or self.finished):
                            seen = reply.get('observation')
                            self.offer_seen[ident] = self.step
                            self.own_call_seen(tool, seen)
                            held_187 = self.rows.get(ident)
                            if not held_187:
                                held_187 = {}
                            after = held_187
                            moved = list((k for k, v in after.items() if k in before and v != before[k] and (not isinstance(v, (dict, list)))))
                            if moved:
                                held_189 = ', '.join(moved[:4])
                                held_188 = '{} changed ({}): {}'.format(ident, held_189, flat(cut(seen, 300))[:900])
                                lines.append(held_188)
                        else:
                            break
                    else:
                        break
                self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
                if lines:
                    self.version = self.version + 1
                held_190 = '\n'.join(lines)
                return held_190
            else:
                return ''
        else:
            return ''

    def swap_hint(self) -> str:
        lost = list((i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)))
        if lost:
            for ident in lost:
                self.gone[ident] = self.step
            self.leads = list((i for i in self.leads if i not in lost))
            self.version = self.version + 1
            picks = self.replacements(lost)
            held_191 = ', '.join(lost)
            text = '[Kestrel] Sold out now: {} (the store shows it cannot be bought). It will not come back, so do not open it again.'.format(held_191)
            if picks and self.left() <= 6:
                held_192 = text + ' Few steps are left: in this turn add {} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {}.'.format(picks[0], ', '.join(picks[1:]) or 'none')
                return held_192
            elif not (picks and self.brief.get('pick') == 'cheapest'):
                if picks:
                    held_194 = text + ' Listings already read that appear to fit that item instead: {}. In this turn take the sold-out line out of the cart and add {} at the requested quantity, then check out.'.format(', '.join(picks), picks[0])
                    return held_194
                held_195 = text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'
                return held_195
            else:
                held_193 = text + " Listings already read that appear to fit that item instead, cheapest first: {}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing.".format(', '.join(picks), picks[0])
                return held_193
        else:
            return ''

    def find_sold(self) -> str:
        ids = list((i for i in self.cart_ids() if i not in self.gone))
        if any((self.sold_out(i) for i in ids)):
            return ''
        elif not 1 == len(ids):
            if ids:
                held_196 = self.reread_offers()
                return held_196
            return ''
        else:
            self.unavailable.add(ids[0])
            return ''

    def replacements(self, lost: list[str]) -> list[str]:
        if not SPEND_SOFT < self.cost:
            records = []
            for ident in reversed(self.seen_order):
                if not (ident in lost or self.sold_out(ident)):
                    records.append(flat(cut(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
                    if 25 <= len(records):
                        break
                else:
                    continue
            if records:
                prompt = FIND_SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger() + '\n\nSOLD-OUT CART LINE:\n' + '\n'.join(('{}: {}'.format(i, self.record_of(i)) for i in lost)) + '\n\nCART NOW:\n' + self.cart_text() + '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
                got = self.model_json(prompt, [WALKER, THINKER], 3000)
                fits: list[tuple[float, str]] = []
                cart = self.cart_ids()
                for entry in as_list((got or {}).get('fits')):
                    ident = str(entry.get('id') if isinstance(entry, dict) else entry)
                    if isinstance(entry, dict):
                        price = entry.get('price')
                    else:
                        price = None
                    if ident in self.rows:
                        if ident not in lost:
                            if ident not in self.gone:
                                if ident not in cart:
                                    if ident not in list((i for _, i in fits)):
                                        fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
                if self.brief.get('pick') == 'cheapest':
                    ranked = list((i for _, i in sorted(fits)))
                else:
                    ranked = list((i for _, i in fits))
                self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
                held_199 = ranked[:4]
                return held_199
            else:
                held_198 = []
                return held_198
        else:
            held_197 = []
            return held_197

    def review_decline(self) -> str:
        if not (self.audited or SPEND_SOFT < self.cost):
            self.audited = True
            got = self.decline_review_call()
            named = self.named_fits(got)
            if not named:
                rows = list((str(i) for i in as_list((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))))
                if rows:
                    if self.open_rows(rows):
                        got = self.decline_review_call()
                        named = self.named_fits(got)
            if named:
                self.lead_items = named
                self.leads = list((i for ids in named.values() for i in ids))
                if (got or {}).get('reason'):
                    reason = ' ({})'.format(got.get('reason'))
                else:
                    reason = ''
                held_200 = 'A careful review thinks these listings may fit: ' + '; '.join(('item {}: {}'.format(k, ', '.join(v)) for k, v in named.items())) + reason + '. Open them and buy what truly fits; end without a purchase only if each one fails.'
                return held_200
            else:
                return ''
        else:
            return ''

    def decline_review_call(self) -> dict[str, Any] | None:
        records = []
        for ident in reversed(self.seen_order):
            if not self.sold_out(ident):
                record = self.full.get(ident)
                records.append('OPENED ' + flat(cut(record, 200))[:1200] if record else 'ROW ' + flat(cut(self.rows.get(ident), 120))[:500])
                if 30 <= len(records):
                    break
            else:
                continue
        if records:
            prompt = REVIEW_DECLINE + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger() + '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
            got = self.model_json(prompt, [THINKER, WALKER], 1500)
            self.trace.append({'kind': 'decline_review', 'got': got})
            return got
        else:
            return None

    def named_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if isinstance(fits, dict):
            named = dict(((str(k), list((str(i) for i in as_list(v) if str(i) in self.rows and (not self.sold_out(str(i)))))[:4]) for k, v in fits.items()))
            held_202 = dict(((k, v) for k, v in named.items() if v))
            return held_202
        else:
            held_201 = {}
            return held_201

    def open_rows(self, rows: list[str]) -> bool:
        many, joiner = (self.compare_arg(), self.roles.get('compare'))
        tool = self.facts_tool()
        arg = self.one_id_arg(tool)
        if many and joiner and (not self.compare_useless()):
            plan = [(joiner, {many: rows[:self.compare_room()]})]
        elif not (tool and arg):
            return False
        else:
            plan = list(((tool, {arg: i}) for i in rows[:2]))
        if not self.left() < len(plan) + 5:
            before = len(self.full)
            for name, args in plan:
                faults: list[str] = []
                held_203 = self.schemas.get(name)
                args = bend(args, held_203, 'args', faults)
                if not faults:
                    reply = self.store(name, args)
                    if not (None is reply or self.finished):
                        held_204 = reply.get('observation')
                        self.own_call_seen(name, held_204)
                    else:
                        break
                else:
                    return False
            self.trace.append({'kind': 'review_open', 'rows': rows, 'opened': len(self.full) - before})
            held_205 = before < len(self.full)
            return held_205
        else:
            return False

    def cart_sets_units(self, name: str) -> bool:
        held_206 = self.schemas.get(name, {}).get('properties')
        if not held_206:
            held_206 = {}
        props = held_206
        if not any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (2 <= len(s['enum'])) for s in props.values())):
            held_207 = any((isinstance(s, dict) and 'integer' in type_list(s) and (s.get('minimum') == 0) for s in props.values()))
            return held_207
        else:
            return False

    def cart_plan(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        held_208 = self.roles.get('cart')
        if not held_208:
            held_208 = ''
        name = held_208
        held_209 = self.schemas.get(name)
        if not held_209:
            held_209 = {}
        spec = held_209
        held_210 = spec.get('properties')
        if not held_210:
            held_210 = {}
        props = held_210
        for key, sub in props.items():
            if isinstance(sub, dict) and 'array' in type_list(sub):
                row = sub.get('items')
            else:
                row = None
            if isinstance(row, dict):
                if isinstance(row.get('properties'), dict):
                    inner = row['properties']
                    id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in type_list(s) and ('enum' not in s)), None)
                    qty_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in type_list(s)), None)
                    if id_key:
                        held_211 = [{key: list(({id_key: i, **({qty_key: n} if qty_key else {})} for i, n in lines.items()))}]
                        return held_211
                    else:
                        return None
        free_text = list((k for k, s in props.items() if isinstance(s, dict) and 'string' in type_list(s) and ('enum' not in s)))
        held_212 = next((k for t, k in self.id_args if name == t and k in free_text), None)
        if not held_212:
            held_212 = next((k for k in spec.get('required') or [] if k in free_text), None)
        if not held_212:
            if 1 == len(free_text):
                held_212 = free_text[0]
            else:
                held_212 = None
        target = held_212
        qty = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in type_list(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (2 <= len(s['enum']))), None)
        if target:
            sets = self.cart_sets_units(name)
            calls: list[dict[str, Any]] = []
            for ident in drop:
                have = self.cart_lines.get(ident, 1)
                if mode:
                    calls.append({target: ident, mode: props[mode]['enum'][1], **({qty: have} if qty else {})})
                elif qty and sets:
                    calls.append({target: ident, qty: 0})
                elif not (qty and isinstance(props[qty].get('minimum'), int) and (-have >= props[qty]['minimum'])):
                    return None
                else:
                    calls.append({target: ident, qty: -have})
            cart = self.cart_ids()
            for ident, want in lines.items():
                if ident in cart:
                    have = self.cart_lines.get(ident, 0)
                else:
                    have = 0
                if want == have:
                    continue
                elif not (have and (not sets)):
                    calls.append({target: ident, **({mode: props[mode]['enum'][0]} if mode else {}), **({qty: want} if qty else {})})
                else:
                    return None
            return calls
        else:
            return None

    def ladder_tool(self) -> tuple[str, str, str, dict[str, Any]] | None:
        skip = set((self.roles.get(r) for r in ('cart', 'cart_view', 'buy', 'refuse', 'ask')))
        for name, spec in self.schemas.items():
            if not name in skip:
                held_213 = spec.get('properties')
                if not held_213:
                    held_213 = {}
                props = held_213
                label = next((k for k, s in props.items() if isinstance(s, dict) and 'string' in type_list(s) and ('enum' not in s) and set(key_words(k).split()) & {'group', 'category', 'shelf', 'department', 'aisle', 'label'}), None)
                price = next((k for k, s in props.items() if isinstance(s, dict) and set(type_list(s)) & {'number', 'integer'} and set(key_words(k).split()) & {'price', 'cost'}), None)
                if not (not label or not price):
                    fixed: dict[str, Any] = {}
                    for k in spec.get('required') or []:
                        if not k in (label, price):
                            options = (props.get(k) or {}).get('enum')
                            if not (not isinstance(options, list) or not options):
                                filt = list((o for o in options if isinstance(o, str) and re.search('crit|filter|field|browse', o.lower())))
                                fixed[k] = (filt or options)[0]
                            else:
                                break
                        else:
                            continue
                    else:
                        held_214 = (name, label, price, fixed)
                        return held_214
                else:
                    continue
            else:
                continue
        return None

    def note_search(self, name: str, args: dict[str, Any], seen: Any) -> None:
        self.search_words |= set((w for v in args.values() if isinstance(v, str) for w in key_stem(re.sub('[^A-Za-z0-9 ]+', ' ', v))))
        got = self.next_page(name, args, seen)
        if got:
            param, value = got
            base = dict(((k, v) for k, v in args.items() if param != k))
            key = name + flat(base)
            trail = self.search_trail.get(key)
            if None is trail:
                trail = self.search_trail[key] = {'name': name, 'args': base, 'param': param, 'next': value, 'ids': set()}
            elif args.get(param) not in (None, '', 0, 1):
                trail['next'] = value
            trail['ids'].update((i for i, _ in self.records_in(seen)))
        else:
            return

    def next_page(self, name: str, args: dict[str, Any], seen: Any) -> tuple[str, Any] | None:
        state = self.page_state(name, seen)
        if state:
            held_215 = (state[2], state[3])
            return held_215
        elif isinstance(seen, dict):
            held_216 = (self.schemas.get(name) or {}).get('properties')
            if not held_216:
                held_216 = {}
            props = held_216
            held_217 = self.shapes.get(name)
            if not held_217:
                held_217 = {}
            shape = held_217
            held_218 = self.records_in(seen)
            rows = len(held_218)
            if shape.get('count'):
                total = whole(seen.get(shape['count']))
            else:
                total = None
            if None is total:
                total = next((whole(v) for k, v in seen.items() if set(key_words(k).split()) & {'found', 'total', 'count', 'hits', 'matches'} and whole(v) is not None), None)
            skip = next((k for k, spec in props.items() if isinstance(spec, dict) and set(type_list(spec)) & {'integer', 'number'} and set(key_words(k).split()) & {'skip', 'offset', 'start', 'from'}), None)
            if not skip:
                param = next((k for k, spec in props.items() if isinstance(spec, dict) and set(type_list(spec)) & {'integer', 'number'} and ('page' in key_words(k).split()) and (not set(key_words(k).split()) & {'size', 'per', 'limit', 'count'})), None)
                if not param:
                    return None
                held_224 = args.get(param)
                held_223 = whole(held_224)
                if not held_223:
                    held_223 = 1
                at = held_223
                last = next((whole(v) for k, v in seen.items() if key_words(k).replace(' ', '') in ('pages', 'totalpages', 'pagecount', 'lastpage')), None)
                held_226 = None is not last
                if held_226:
                    held_226 = last > at
                held_225 = held_226
                if not held_225:
                    held_227 = None is last
                    if held_227:
                        held_227 = None is not total
                    if held_227:
                        held_227 = 0 < rows
                    if held_227:
                        held_227 = total > at * rows
                    held_225 = held_227
                more = held_225
                held_228 = (param, at + 1 if more else None)
                return held_228
            else:
                held_220 = args.get(skip)
                held_219 = whole(held_220)
                if not held_219:
                    held_219 = 0
                at = held_219
                held_221 = (props.get(skip) or {}).get('maximum')
                top = whole(held_221)
                if rows and None is not total and (total > at + rows):
                    nxt = at + rows
                else:
                    nxt = None
                if None is not nxt:
                    if None is not top:
                        if top < nxt:
                            if top > at:
                                nxt = top
                            else:
                                nxt = None
                held_222 = (skip, nxt)
                return held_222
        else:
            return None

    def cheap_sweep(self) -> int:
        if not (self.brief.get('pick') != 'cheapest' or self.elapsed() > CLOCK_SOFT - 30):
            mark = self.cart_mark() + '#{}'.format(self.version)
            if not mark in self.sweep_marks:
                self.sweep_marks.add(mark)
                held_229 = self.cart_ids()
                cart = set(held_229)
                trails = list((t for t in self.search_trail.values() if None is not t['next'] and t['ids'] & cart))
                pages, before = (0, len(self.seen_order))
                for trail in trails:
                    while None is not trail['next'] and SWEEP_PAGES > pages and (self.left() - self.tail() > SWEEP_KEEP) and (self.elapsed() < CLOCK_SOFT - 20) and (not self.finished):
                        faults: list[str] = []
                        call = bend({**trail['args'], trail['param']: trail['next']}, self.schemas.get(trail['name']), 'args', faults)
                        if not faults:
                            reply = self.store(trail['name'], call)
                            if not (None is reply or self.finished):
                                pages = pages + 1
                                seen = reply.get('observation')
                                self.own_call_seen(trail['name'], seen)
                                used, shown = (trail['next'], set((i for i, _ in self.records_in(seen))))
                                got = self.next_page(trail['name'], call, seen)
                                trail['next'] = got[1] if got and used != got[1] and shown - trail['ids'] else None
                                trail['ids'].update(shown)
                            else:
                                trail['next'] = None
                                break
                        else:
                            trail['next'] = None
                            break
                if pages:
                    self.trace.append({'kind': 'cheap_sweep', 'pages': pages, 'new_rows': len(self.seen_order) - before})
                return pages
            else:
                return 0
        else:
            return 0

    def price_ladder(self) -> str:
        if not (self.brief.get('pick') != 'cheapest' or self.elapsed() > CLOCK_SOFT - 30 or self.left() < 10):
            found = self.ladder_tool()
            cat = self.category_field()
            if not (not found or not cat):
                tool, label_arg, price_arg, fixed = found
                mark = self.cart_mark() + '#{}'.format(self.version)
                if not mark in self.ladder_marks:
                    self.ladder_marks.add(mark)
                    lines, opened = ([], [])
                    for ident in self.cart_ids():
                        price, label = (self.price_of(ident), str((self.rows.get(ident) or {}).get(cat) or '').strip())
                        if not (None is price or not label):
                            cheaper: list[str] = []
                            top = price
                            for _ in range(2):
                                if not self.left() < 8:
                                    faults: list[str] = []
                                    args = bend({**fixed, label_arg: label, price_arg: top - 0.01}, self.schemas.get(tool), 'args', faults)
                                    if not faults:
                                        reply = self.store(tool, args)
                                        if not (None is reply or self.finished):
                                            seen = reply.get('observation')
                                            self.own_call_seen(tool, seen)
                                            rows = list((i for i, _ in self.records_in(seen) if i not in cheaper and self.buyable(i) and ((self.price_of(i) if self.price_of(i) is not None else top) < price)))
                                            if not rows:
                                                break
                                            cheaper += rows
                                            top = min((self.price_of(i) or top for i in rows))
                                        else:
                                            break
                                    else:
                                        break
                                else:
                                    break
                            if cheaper:
                                cheaper.sort(key=lambda i: self.price_of(i) or 0.0)
                                opened += self.read_facts(list((i for i in cheaper if i not in self.full))[:2], self.tail() + 5)
                                lines.append('{} ({:g}) has cheaper listings in its category {!r}: '.format(ident, price, label) + ', '.join(('{} ({:g})'.format(i, self.price_of(i)) for i in cheaper[:6])))
                        else:
                            continue
                    self.trace.append({'kind': 'price_ladder', 'lines': len(lines), 'opened': len(opened)})
                    if not lines:
                        return ''
                    held_230 = "[Kestrel] The pick is the cheapest, so Kestrel listed each cart line's category below its price:\n" + '\n'.join(lines) + ('\nPages read for the check:\n' + '\n'.join(opened) if opened else '') + '\nSwap in the cheapest one that meets every requirement (check its attributes), then check out.'
                    return held_230
                else:
                    return ''
            else:
                return ''
        else:
            return ''

    def want_lines(self, want: Want) -> list[str]:
        cat = self.category_field()
        ids = self.cart_ids()
        if not 1 == len(self.wants):
            held_231 = list((i for i in ids if cat and self._label_is(str((self.rows.get(i) or {}).get(cat) or ''), want)))
            return held_231
        else:
            return ids

    def fit_score(self, ident: str, want: Want) -> int:
        held_232 = self.full.get(ident)
        if not held_232:
            held_232 = self.rows.get(ident)
        if not held_232:
            held_232 = {}
        record = held_232
        held_233 = attr_block(record)
        if not held_233:
            held_233 = {}
        attrs = held_233
        text = flat(record).lower()
        score = 0
        for need in list((m for m in want.musts + want.told if m not in want.dropped)):
            prop, _, value = need.partition(':')
            if not re.search('\\b(not|other than|unlike|except|anything but|same as|near|within)\\b', value.lower()):
                words = list((w for w in re.findall('[a-z0-9.]+', value.lower()) if 2 < len(w) and w not in ('the', 'and', 'with', 'least', 'most')))
                if words:
                    said = list((str(v).lower() for k, v in attrs.items() if names_same(key_stem(prop), key_stem(k))))
                    if said:
                        score += 2 if any((all((w in s for w in words)) for s in said)) else -2
                    elif all((w in text for w in words)):
                        score = score + 1
                else:
                    continue
            else:
                continue
        return score

    def best_guess(self, want: Want, taken: set[str]) -> str | None:
        cat = self.category_field()
        pool = []
        for ident in self.seen_order:
            if ident in taken or not self.buyable(ident):
                continue
            elif not (1 < len(self.wants) and (not (cat and self._label_is(str((self.rows.get(ident) or {}).get(cat) or ''), want)))):
                price = self.price_of(ident)
                if not (None is not self.budget and None is not price and (self.budget < price * want.qty)):
                    pool.append(ident)
                else:
                    continue
            else:
                continue
        if pool:
            cheapest = self.brief.get('pick') == 'cheapest'
            rank = lambda i: (i in self.full, self.fit_score(i, want), i in self.leads, -(self.price_of(i) or 0.0) if cheapest else 0.0)
            held_234 = max(pool, key=rank)
            return held_234
        else:
            return None

    def safety_due(self) -> bool:
        if self.safety_done or self.finished or self.buys or (not self.roles.get('cart')) or (not self.roles.get('buy')):
            return False
        elif not str(self.brief.get('no_fit') or '').lower() == 'no_purchase':
            missing = list((w for w in self.wants if not self.want_lines(w)))
            if missing:
                held_235 = self.left() <= self.tail() + len(missing) + 2 or self.elapsed() > CLOCK_SOFT + 10
                return held_235
            else:
                return False
        else:
            return False

    def safety_cart(self) -> str:
        self.safety_done = True
        held_236 = self.cart_ids()
        taken = set(held_236)
        plan: dict[str, int] = dict(((i, self.cart_lines.get(i, 1)) for i in taken if not self.sold_out(i)))
        added = []
        for number, want in enumerate(self.wants, 1):
            if not self.want_lines(want):
                pick = self.best_guess(want, taken | set(plan))
                if pick:
                    plan[pick] = want.qty
                    held_237 = 'item {} {}: {} x{}'.format(number, want.name, pick, want.qty)
                    added.append(held_237)
            else:
                continue
        if added:
            calls = self.cart_plan(plan, [])
            if not (not calls or len(calls) + 1 > self.left()):
                cart = self.roles['cart']
                for args in calls:
                    faults: list[str] = []
                    held_238 = self.schemas.get(cart)
                    args = bend(args, held_238, 'args', faults)
                    if not faults:
                        reply = self.store(cart, args)
                        if not (None is reply or self.finished):
                            held_239 = reply.get('observation')
                            self.track_cart(cart, args, held_239)
                            held_240 = reply.get('observation')
                            self.own_call_seen(cart, held_240)
                        else:
                            break
                    else:
                        break
                self.trace.append({'kind': 'safety_cart', 'added': added, 'left': self.left(), 'secs': round(self.elapsed())})
                held_241 = '[Kestrel] The steps or the time are nearly out and these items had no cart line, so Kestrel carted the best-matching listing it has read for each: ' + '; '.join(added) + '. Ending without a purchase cannot satisfy this request: check out now (swap a line only if you know a better fit and the steps allow it).'
                return held_241
            else:
                return ''
        else:
            return ''

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched, self.cart_at = (True, self.step)
        held_242 = self.schemas.get(name, {}).get('properties')
        if not held_242:
            held_242 = {}
        props = held_242
        removing = any((isinstance((props.get(k) or {}).get('enum'), list) and 2 <= len(props[k]['enum']) and (v in props[k]['enum'][1:]) for k, v in args.items()))
        loose = next((whole(v) for v in args.values() if not isinstance(v, str) and whole(v) is not None), None)
        lines: list[tuple[str, int | None]] = []
        replaces = False
        for key, value in args.items():
            if isinstance(value, list):
                held_243 = (props.get(key) or {}).get('items')
                if not held_243:
                    held_243 = {}
                row = held_243
                if not value:
                    if isinstance(row, dict):
                        if row.get('type') == 'object' or 'properties' in row:
                            replaces = True
                for entry in value:
                    if isinstance(entry, dict):
                        replaces = True
                        ident = next((v for v in entry.values() if isinstance(v, str) and v in self.rows), None)
                        units = next((whole(v) for v in entry.values() if not isinstance(v, str) and whole(v) is not None), 1)
                        if ident:
                            lines.append((ident, units))
                    elif isinstance(entry, str):
                        if entry in self.rows:
                            lines.append((entry, loose))
            elif isinstance(value, str):
                if value in self.rows:
                    lines.append((value, loose))
        if not replaces:
            sets = self.cart_sets_units(name)
            for ident, units in lines:
                have = self.cart_lines.get(ident, 0)
                if removing:
                    if units:
                        have = have - units
                    else:
                        have = 0
                elif not sets:
                    have += units if None is not units else 1
                elif None is not units:
                    have = units
                else:
                    have = 1
                if not 0 < have:
                    self.cart_lines.pop(ident, None)
                else:
                    self.cart_lines[ident] = have
        else:
            self.cart_lines = dict(((i, n) for i, n in lines if n and 0 < n))
            return

    def finish_reviewed(self) -> str:
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if not (self.completed or not self.confirmed or self.sold_out(self.confirmed) or (not cart) or (not buy)):
            self.completed = True
            cat = self.category_field()
            if cat:
                kind = lambda i: str((self.rows.get(i) or {}).get(cat) or '').strip().lower()
            else:
                kind = lambda i: i
            plan: dict[str, int] = {}
            for number, ids in self.lead_items.items():
                index = whole(number)
                pick = next((i for i in ids if not self.sold_out(i)), None)
                if pick:
                    if index:
                        if 1 <= index <= len(self.wants):
                            if kind(pick) not in set((kind(p) for p in plan)):
                                plan[pick] = self.wants[index - 1].qty
            if not self.confirmed not in plan:
                now = self.cart_ids()
                planned_kinds = set((kind(p) for p in plan))
                keep = dict(((i, self.cart_lines.get(i, 1)) for i in now if i not in plan and (not self.sold_out(i)) and (kind(i) not in planned_kinds)))
                drop = list((i for i in now if i not in plan and i not in keep))
                if not len(plan) + len(set((kind(i) for i in keep))) < len(self.wants):
                    lines = {**keep, **plan}
                    calls = self.cart_plan(lines, drop)
                    if not calls or len(calls) + 1 > self.left():
                        return ''
                    held_244 = list(lines)
                    self.read_facts(held_244, len(calls) + 3)
                    held_245 = list(lines)
                    self.review_cart(held_245, SPEND_HARD)
                    if not self.verdicts.get(self.verdict_key(list(lines))):
                        return ''
                    done = []
                    for args in calls:
                        faults: list[str] = []
                        held_246 = self.schemas.get(cart)
                        args = bend(args, held_246, 'args', faults)
                        if not faults:
                            reply = self.store(cart, args)
                            if not (None is reply or self.finished):
                                self.note_cart(cart, args)
                                held_247 = reply.get('observation')
                                self.own_call_seen(cart, held_247)
                                held_248 = flat(args)
                                done.append(held_248)
                            else:
                                return ''
                        else:
                            return ''
                    faults = []
                    held_249 = dict(self.brief.get('buy_args') or {})
                    held_250 = self.schemas.get(buy)
                    args = bend(held_249, held_250, 'args', faults)
                    if faults:
                        return ''
                    replies = []
                    for attempt in range(2):
                        if attempt:
                            self.reread_offers()
                            held_251 = list(lines)
                            self.review_cart(held_251, SPEND_HARD)
                            if not self.verdicts.get(self.verdict_key(list(lines))) or self.left() < 1 or self.finished or any((self.sold_out(i) for i in lines)):
                                break
                        go, said = self.gate('order')
                        if said:
                            replies.append(said)
                        if go:
                            reply = self.store(buy, args)
                            self.trace.append({'kind': 'finish_reviewed', 'lines': lines, 'drop': drop, 'reply': cut(reply, 200)})
                            if not None is reply:
                                self.buys = self.buys + 1
                                if not reply.get('finished'):
                                    self.refused_at = self.step
                                held_252 = reply.get('observation')
                                self.own_call_seen(buy, held_252)
                                held_254 = reply.get('observation')
                                held_253 = self.show(held_254)
                                replies.append(held_253)
                                if self.finished or reply.get('finished'):
                                    break
                            else:
                                break
                        else:
                            break
                    if not replies:
                        return ''
                    held_255 = 'Not run (Kestrel acted instead): the independent check confirmed ' + self.confirmed + ', so Kestrel set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)
                    return held_255
                else:
                    return ''
            else:
                return ''
        else:
            return ''

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = list((p for p in self.process if stage == p['when']))
        if not (not steps or self.finished or (not self.gate_due(stage)) or (self.elapsed() > CLOCK_LAST)):
            ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
            tools = list(('- {}: {} | parameters: {}'.format(t['function']['name'], t['function'].get('description', ''), flat(t['function'].get('parameters') or {})) for t in self.tools if t['function']['name'] not in ending))
            said: list[str] = []
            nudged = False
            for _ in range(GATE_CALLS):
                prompt = STEP_GATE.replace('STAGE', 'place the order' if 'order' == stage else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join(('- {}'.format(p['step']) for p in steps)) + '\n\nLEDGER:\n' + self.ledger() + '\n\nCART NOW: ' + self.cart_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join(('{}. {}'.format(i, x) for i, x in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + '\n\nMESSAGES LEFT: {}   STEPS LEFT: {} '.format(max(0, self.msg_limit - self.msg_used), self.left()) + '(the {} itself takes one)'.format('order' if 'order' == stage else 'ending') + '\n\nTOOLS:\n' + '\n'.join(tools)
                got = self.model_json(prompt, [THINKER, WALKER] if nudged else [WALKER, THINKER], 2000)
                if not None is got:
                    told: list[str] = []
                    hold = ''
                    if said:
                        self.take_heard({'needs': got.get('needs')}, told)
                        hold = str(got.get('hold') or '').strip()
                    if isinstance(got.get('call'), dict):
                        call = got.get('call')
                    else:
                        call = None
                    self.trace.append({'kind': 'gate', 'stage': stage, 'call': cut(call, 200), 'steps': cut(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
                    if not (told or hold):
                        name = str((call or {}).get('tool') or '')
                        shown = list((e for e in as_list(got.get('steps')) if isinstance(e, dict)))
                        held_258 = 'order' == stage
                        if held_258:
                            held_258 = any((0 < p['messages'] for p in steps))
                        talk = held_258
                        proof = set((i for i in range(1, len(self.done_log) + 1) if not talk or (len(self.done_steps) >= i and self.done_steps[i - 1] > self.cart_at)))
                        unproven = list((e for e in shown if e.get('done') is not True or not any(((whole(n) or 0) in proof for n in as_list(e.get('lines'))))))
                        if not (not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_limit > self.msg_used) and (self.left() >= 2)):
                            if not call or name not in self.schemas or name in ending:
                                if not call:
                                    self.gate_ok.add((stage, self.cart_mark() + '#{}'.format(self.version)))
                                break
                            if self.left() < 2 or (name == self.roles.get('ask') and self.msg_limit <= self.msg_used):
                                break
                            faults: list[str] = []
                            args = bend(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
                            if faults:
                                break
                            if name == self.roles.get('ask'):
                                text = next((v for v in args.values() if isinstance(v, str)), '')
                                batch = self.send([(1, '', text)], 1)
                                if batch:
                                    self.gate_msgs = self.gate_msgs + 1
                                    held_259 = 'Q: {}\nA: {}'.format(text, batch[0][2])
                                    said.append(held_259)
                                    continue
                                else:
                                    break
                            reply = self.store(name, args)
                            if None is reply:
                                break
                            seen = reply.get('observation')
                            self.take_in(seen)
                            held_260 = 'step {}: {} {} -> {} (cart then: {})'.format(self.step, name, flat(args), self.show(seen)[:300], self.cart_mark())
                            self.done_log.append(held_260)
                            self.done_steps.append(self.step)
                            held_262 = flat(args)
                            held_261 = '{} {} -> {}'.format(name, held_262, self.show(seen)[:400])
                            said.append(held_261)
                        else:
                            nudged = True
                            self.trace.append({'kind': 'gate_unproven', 'steps': cut(unproven, 120)})
                            self.done_log.append('(Kestrel: no line made since the cart last changed at step {} shows '.format(self.cart_at) + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + ' done, and {} message(s) and {} steps are left: give the call that does it)'.format(self.msg_limit - self.msg_used, self.left()))
                            self.done_steps.append(-1)
                            continue
                    else:
                        self.version = self.version + 1
                        held_257 = (False, ('Before going on, Kestrel did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + "The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.')
                        return held_257
                else:
                    break
            held_263 = (True, 'Before going on, Kestrel did the steps the request asks for:\n' + '\n'.join(said) if said else '')
            return held_263
        else:
            held_256 = (True, '')
            return held_256

    def wider_first(self) -> str:
        if not (self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.wants)):
            self.wider_held = True
            lines = list(('{} at {:g}'.format(i, self.price_of(i)) for i in self.cart_ids() if self.price_of(i) is not None))
            held_264 = "Not run yet (no step used). {} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line".format(self.left()) + (' ({})'.format('; '.join(lines)) if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'
            return held_264
        else:
            return ''

class Moves:

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        if isinstance(self.brief.get('buy_args'), dict):
            fixed = self.brief.get('buy_args')
        else:
            fixed = {}
        held_265 = name == self.roles.get('buy') and all((args.get(k) == v for k, v in fixed.items()))
        return held_265

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        if isinstance(self.brief.get('refuse_args'), dict):
            fixed = self.brief.get('refuse_args')
        else:
            fixed = {}
        held_266 = name == self.roles.get('refuse') and all((args.get(k) == v for k, v in fixed.items()))
        return held_266

    def show(self, observation: Any) -> str:
        held_267 = cut(observation, REPLY_CHARS)
        text = flat(held_267)
        if OBS_CHARS < len(text):
            held_268 = text[:OBS_CHARS] + '...(cut)'
            return held_268
        return text

    def show_rows(self, observation: Any) -> str:

        def shorten(node: Any, depth: int=0) -> Any:
            if 6 < depth:
                return node
            elif isinstance(node, list):
                held_269 = list((shorten(v, depth + 1) for v in node))
                return held_269
            elif not isinstance(node, dict):
                return node
            elif not self.id_of(node, True, depth):
                held_271 = dict(((k, shorten(v, depth + 1)) for k, v in node.items()))
                return held_271
            else:
                held_270 = dict(((k, cut(v, NAME_CHARS if re.search('name|title', str(k), re.I) else ROW_CHARS)) for k, v in node.items()))
                return held_270
        held_272 = self.show(shorten(observation))
        return held_272

    def offered(self) -> list[dict[str, Any]]:
        left, ends = (self.left(), {self.roles.get('buy'), self.roles.get('refuse')} - {None})
        keep: set[str] | None = None
        if left <= self.tail():
            keep = set(ends)
        elif left <= self.tail() + 3 or self.elapsed() > CLOCK_SOFT:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view')}
        elif len(self.wants) + 5 >= left:
            if 0 == self.buys:
                keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
        tools = list((t for t in self.tools if t['function']['name'] != self.roles.get('ask')))
        if None is not keep:
            if ends:
                held_273 = list((t for t in tools if t['function']['name'] in keep))
                if not held_273:
                    held_273 = tools
                tools = held_273
        return tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        held_274 = self.schemas.get(name)
        if not held_274:
            held_274 = {}
        spec = held_274
        out = dict(args)
        for key, sub in (spec.get('properties') or {}).items():
            if not (not isinstance(sub, dict) or 'string' not in type_list(sub) or 'enum' in sub):
                value = out.get(key)
                if not (None is value and key not in (spec.get('required') or [])):
                    text = str(value or '')
                    if not text or any((i in text for i in self.rows)) or self.brief.get('quiet_decline') is not False:
                        out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:sub.get('maxLength', 200)]
                else:
                    continue
            else:
                continue
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if not (name not in set((self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare'))) - {None} or not self.id_key):
            held_276 = (self.schemas.get(name) or {}).get('properties')
            if not held_276:
                held_276 = {}
            props = held_276
            free = list((k for k, v in props.items() if isinstance(v, dict) and ('string' in type_list(v) or 'array' in type_list(v)) and ('enum' not in v)))
            if not (1 != len(free) or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args)):

                def fix(value: Any) -> tuple[Any, str]:
                    if isinstance(value, str) and value.strip() in self.rows:
                        held_278 = (value.strip(), '')
                        return held_278
                    elif not (not isinstance(value, str) or looks_like_id(value, True) or len(value.strip()) < 8):
                        low = value.strip().lower()
                        hits = list((i for i, row in self.rows.items() if any((self.id_key != k and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items()))))
                        if not 1 == len(hits):
                            if hits:
                                held_281 = (value, 'several listings carry the text {!r} ({}); pass one listing id'.format(value, ', '.join(hits[:6])))
                                return held_281
                            held_282 = (value, '{!r} is not a listing id; pass the id from the result rows (their {} field)'.format(value, self.id_key))
                            return held_282
                        else:
                            held_280 = (hits[0], '')
                            return held_280
                    else:
                        held_279 = (value, '')
                        return held_279
                out = dict(args)
                for key in free:
                    if not key not in args:
                        if isinstance(args[key], list):
                            fixed = list((fix(v) for v in args[key]))
                        else:
                            fixed = [fix(args[key])]
                        problem = next((p for _, p in fixed if p), '')
                        if not problem:
                            out[key] = list((v for v, _ in fixed)) if isinstance(args[key], list) else fixed[0][0]
                        else:
                            held_283 = (args, problem)
                            return held_283
                    else:
                        continue
                held_284 = (out, '')
                return held_284
            else:
                held_277 = (args, '')
                return held_277
        else:
            held_275 = (args, '')
            return held_275

    def compare_arg(self) -> str | None:
        held_285 = self.schemas.get(self.roles.get('compare') or '')
        if not held_285:
            held_285 = {}
        spec = held_285
        held_286 = spec.get('properties')
        if not held_286:
            held_286 = {}
        props = held_286
        arg = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in type_list(v)), None)
        if not (not arg or any((arg != k for k in spec.get('required') or []))):
            return arg
        else:
            return None

    def compare_room(self) -> int:
        held_287 = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.compare_arg() or '')
        if not held_287:
            held_287 = {}
        spec = held_287
        if isinstance(spec.get('maxItems'), int):
            schema_max = spec.get('maxItems')
        else:
            schema_max = None
        learned = self.world_compare
        if learned and (not self.compare_bad):
            if schema_max:
                held_288 = min(learned, schema_max)
                return held_288
            return learned
        elif not self.compare_bad:
            if schema_max:
                return schema_max
            held_291 = max(4, self.compare_ok)
            return held_291
        else:
            held_289 = min(self.compare_bad - 1, max(self.compare_ok, 2))
            room = max(2, held_289)
            if schema_max:
                held_290 = min(room, schema_max)
                return held_290
            return room

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner = (self.roles.get('open'), self.roles.get('compare'))
        arg, many = (self.one_id_arg(opener), self.compare_arg())
        if not (not opener or not joiner or (not arg) or (not many) or (joiner == opener) or self.compare_useless()):
            picks: list[tuple[int, str]] = []
            for index, call in enumerate(calls):
                held_292 = call.get('function')
                if not held_292:
                    held_292 = {}
                fn = held_292
                if isinstance(fn.get('arguments'), str):
                    args = as_json(fn.get('arguments'))
                else:
                    args = None
                ident = str((args or {}).get(arg) or '').strip()
                if fn.get('name') == opener:
                    if args:
                        if set(args) == {arg}:
                            if ident in self.rows:
                                if ident not in self.gone:
                                    picks.append((index, ident))
            picks = picks[:self.compare_room()]
            if not 2 > len(picks):
                first, joined = (picks[0][0], set((i for i, _ in picks)))
                out = []
                for index, call in enumerate(calls):
                    if first == index:
                        out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': flat({many: list((i for _, i in picks))})}})
                    elif not index in joined:
                        out.append(call)
                    else:
                        out.append({'id': call['id'], 'joined': True})
                self.trace.append({'kind': 'joined', 'ids': list((i for _, i in picks))})
                return out
            else:
                return calls
        else:
            return calls

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for call in self.join_opens(calls):
            cid = call['id']
            if not call.get('joined'):
                held_293 = len(calls)
                refused = self.precheck(call, held_293)
                if not isinstance(refused, str):
                    name, args, extra = refused
                    out.append((cid, self.carry_out(name, args, extra)))
                else:
                    out.append((cid, refused))
                    continue
            else:
                out.append((cid, 'Shown in the side-by-side result above: Kestrel opened these listings together in one step.'))
                continue
        return out

    def precheck(self, call: dict[str, Any], batch: int) -> str | tuple[str, dict[str, Any], str]:
        allowed = set((t['function']['name'] for t in self.offered()))
        held_294 = call.get('function')
        if not held_294:
            held_294 = {}
        fn = held_294
        name, raw = (fn.get('name'), fn.get('arguments'))
        if self.finished:
            return 'Not run: the episode is over.'
        elif name == self.roles.get('ask'):
            return "Not run (no step used): Kestrel's interviewer handles every message to the shopper."
        elif not name not in self.schemas:
            if name not in allowed:
                held_296 = 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'
                return held_296
            if isinstance(raw, dict):
                args = raw
            else:
                args = as_json(raw) if (raw or '').strip() else {}
            if None is args:
                return 'Not run (no step used): the arguments were not one JSON object.'
            faults: list[str] = []
            args = bend(args, self.schemas[name], 'args', faults)
            if faults:
                held_297 = 'Not run (no step used): ' + '; '.join(faults[:6])
                return held_297
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                held_298 = 'Not run (no step used): ' + wrong + '.'
                return held_298
            note = ''
            if name == self.roles.get('compare'):
                many = self.compare_arg()
            else:
                many = None
            if many:
                if isinstance(args.get(many), list):
                    if self.compare_bad or self.world_compare:
                        if len(args[many]) > self.compare_room():
                            room = self.compare_room()
                            held_300 = map(str, args[many][room:])
                            held_299 = ', '.join(held_300)
                            note = '\n[Kestrel] Sent the first {} listings only: this store shows at most {} side by side. Not shown yet: {}.'.format(room, room, held_299)
                            args = {**args, many: args[many][:room]}
            single = self.one_id_arg(name)
            if single:
                if name in (self.roles.get('open'), self.roles.get('offer')):
                    if str(args.get(single)) in self.gone:
                        held_301 = 'Not run (no step used): {} sold out at step {} and will not come back. Replace its cart line with another fitting listing.'.format(args[single], self.gone[str(args[single])])
                        return held_301
            if name == self.roles.get('cart'):
                if any(((whole(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
                    if self.cart_fresh():
                        view = flat(self.cart_view)
                    else:
                        view = ''
                    absent = list((v for v in args.values() if isinstance(v, str) and v in self.rows and (v not in self.cart_lines) and (v not in view) and (v not in self.cart_said)))
                    if absent:
                        held_302 = 'Not run (no step used): {} is not in the cart, so there is nothing to take off. Cart now: '.format(absent[0]) + self.cart_text()
                        return held_302
            held_303 = self.is_buy(name, args)
            if not held_303:
                held_303 = self.is_refuse(name, args)
            ending = held_303
            if ending:
                if 1 < batch:
                    return 'Not run (no step used): send the finishing call on its own, after reading your other results.'
            if not ending:
                if self.left() <= self.tail():
                    if self.roles.get('buy') or self.roles.get('refuse'):
                        return "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."
            mark = name + flat(args)
            if not ending:
                if name != self.roles.get('cart'):
                    if mark in self.recent:
                        return 'Not run (no step used): you just made this same call and nothing changed; its result is above.'
            readers = set((self.roles.get(r) for r in ('search', 'open', 'offer', 'compare'))) - {None}
            cached = self.read_cache.get(mark)
            held_304 = bool(single)
            if held_304:
                held_304 = name in readers
            if held_304:
                held_306 = args.get(single)
                held_305 = str(held_306)
                held_304 = self.line_stale(held_305)
            recheck = held_304
            if name in readers:
                if cached:
                    if self.version == cached[0]:
                        if not recheck:
                            held_307 = 'Not run (no step used): the same call at step {} returned this, and nothing has changed since:\n'.format(cached[1]) + cached[2][:3000]
                            return held_307
            if single:
                if name in readers:
                    if name != self.roles.get('compare'):
                        if not recheck:
                            if self.shown_already(name, str(args.get(single))):
                                held_308 = 'Not run (no step used): {} has shown everything this view gives since the last change. Its record: '.format(args[single]) + self.record_of(str(args[single]))
                                return held_308
            if many:
                if isinstance(args.get(many), list):
                    if args[many]:
                        fresh = list((i for i in args[many] if not self.shown_already(name, str(i))))
                        if not fresh:
                            held_309 = 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join(('{}: {}'.format(i, self.record_of(str(i))) for i in args[many]))[:4000]
                            return held_309
                        elif len(args[many]) > len(fresh):
                            note += '\n[Kestrel] Left out {}: shown in full already, nothing changed since.'.format(', '.join((str(i) for i in args[many] if i not in fresh)))
                            args = {**args, many: fresh}
            end_mark = ('order' if self.is_buy(name, args) else 'end') + '|' + self.cart_mark() + '#{}'.format(self.version)
            held_310 = ending
            if held_310:
                held_310 = end_mark in self.ending_advised
            heard = held_310
            if heard:
                self.trace.append({'kind': 'advice_overruled', 'on': end_mark.split('|')[0]})
            if self.is_buy(name, args) and (not heard or (3 > self.short_holds and self.left() > 3)):
                short = self.missing_items()
            else:
                short = ''
            if short:
                if heard:
                    self.short_holds = self.short_holds + 1
                    self.trace.append({'kind': 'short_order_held', 'n': self.short_holds})
                    held_311 = 'Not run yet (no step used). The cart still has ' + short + '; an order missing a requested item fails for certain. Cart a listing for each missing item, or end without a purchase if none fits' + ('. Sending this checkout once more runs it as it is.' if 3 <= self.short_holds else '.') + '\nCart now: ' + self.cart_text()
                    return held_311
            if short:
                self.short_holds = self.short_holds + 1
                held_312 = self.advise(end_mark, "Not run yet (no step used). Kestrel's record of the cart has " + short + '. An order missing a requested item cannot score. Cart a listing for it (the nearest match, if the request asks for one when nothing fits), or end without a purchase if no listing fits it and the request does not ask for the nearest match. If the cart does hold it, send the checkout again.\nCart now: ' + self.cart_text())
                return held_312
            if ending:
                if self.left() > 2:
                    if self.is_buy(name, args):
                        held = self.hold_buy(name, args, end_mark, heard)
                    elif heard:
                        held = ''
                    else:
                        held = self.hold_refuse(end_mark)
                    if held:
                        return held
            if self.is_refuse(name, args):
                if heard:
                    held = ''
                else:
                    held = self.hold_late_refuse() or self.push_before_refuse()
                if not held:
                    done = self.finish_reviewed()
                    if not done:
                        args = self.quiet(name, args)
                    else:
                        held_314 = done + (' [episode finished]' if self.finished else '')
                        return held_314
                else:
                    held_313 = self.advise(end_mark, held)
                    return held_313
            if ending:
                if self.left() >= 2:
                    go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                    if not go:
                        held_315 = 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()
                        return held_315
                    elif said:
                        note = '\n[Kestrel] ' + said + note
            held_316 = (name, args, note)
            return held_316
        else:
            held_295 = 'Not run (no step used): unknown tool. Tools: {}.'.format(', '.join(sorted(allowed)))
            return held_295

    def advise(self, mark: str, text: str) -> str:
        now = mark.split('|')[0] + '|' + self.cart_mark() + '#{}'.format(self.version)
        self.ending_advised.update({mark, now})
        self.trace.append({'kind': 'advice', 'on': mark.split('|')[0], 'text': text[:200]})
        held_317 = text + '\n[Kestrel] This is advice, not a rule. If you have checked it against the records and still disagree, send the same call again and say why in one sentence; it will then run.'
        return held_317

    def missing_items(self) -> str:
        have, want = (len(set(self.cart_ids())), len(self.wants))
        if not want <= have:
            held_318 = '{} line{} for {} requested item{}'.format(have, '' if 1 == have else 's', want, '' if 1 == want else 's')
            return held_318
        else:
            return ''

    def line_stale(self, ident: str) -> bool:
        held_319 = ident in self.cart_ids() and self.offer_seen.get(ident, -99) < self.cart_at
        return held_319

    def query_words(self, args: dict[str, Any]) -> list[str]:
        held_320 = (self.schemas.get(self.roles.get('search') or '') or {}).get('properties')
        if not held_320:
            held_320 = {}
        props = held_320
        if not any((isinstance(spec, dict) and 'enum' in spec and isinstance(args.get(k), str) and re.search('crit|filter|field|browse', args[k].lower()) for k, spec in props.items())):
            named = {'group', 'category', 'shelf', 'department', 'aisle', 'label', 'brand', 'maker', 'seller', 'vendor', 'token', 'cursor', 'page', 'id'}
            free = list((k for k, spec in props.items() if isinstance(spec, dict) and 'string' in type_list(spec) and ('enum' not in spec) and (not set(key_words(k).split()) & named)))
            held_322 = list((w for k in free if isinstance(args.get(k), str) for w in re.findall('[a-z0-9]+', args[k].lower()) if 2 < len(w)))
            return held_322
        else:
            held_321 = []
            return held_321

    def hidden_note(self, args: dict[str, Any], seen: Any) -> str:
        stem = lambda w: w[:max(4, len(w) - 2)]
        for ident, words in list(self.hidden_wait.items()):
            if ident in self.full:
                body = flat(self.full[ident]).lower()
                self.hidden_score[all((stem(w) in body for w in words))] += 1
                del self.hidden_wait[ident]
        words = self.query_words(args)
        if not (not words or (3 <= self.hidden_score[False] and self.hidden_score[True] < self.hidden_score[False])):
            held_323 = self.shapes.get(self.roles.get('search') or '')
            if not held_323:
                held_323 = {}
            shape = held_323
            if isinstance(seen, dict) and shape.get('count'):
                total = whole(seen.get(shape.get('count')))
            else:
                total = None
            if not (None is not total and NARROW_SEARCH < total):
                rows = []
                for ident, record in self.records_in(seen):
                    if ident in self.full or self.sold_out(ident):
                        continue
                    elif not all((stem(w) in flat(record).lower() for w in words)):
                        rows.append(ident)
                        self.hidden_wait[ident] = words
                if rows:
                    told, self.hidden_told = (self.hidden_told, True)
                    held_324 = '\n[Kestrel] ' + ', '.join(rows[:7]) + ' do not show every query word in their rows, so the store likely matched them on details a row leaves out (attributes). Open them before ruling them out by title.' + ('' if told else ' Naming two or three required values in one query narrows the rows to those holding them all, as far as the query length allows.')
                    return held_324
                else:
                    return ''
            else:
                return ''
        else:
            return ''

    def hold_buy(self, name: str, args: dict[str, Any], mark: str='', heard: bool=False) -> str:
        if heard:
            ladder = ''
        else:
            ladder = self.price_ladder()
        if not ladder:
            if heard:
                wider = ''
            else:
                wider = self.wider_first()
            if not wider:
                reread = self.reread_offers()
                if self.cart_touched:
                    if self.cart_at > self.cart_seen_at:
                        if self.roles.get('cart_view'):
                            if self.left() > 3:
                                faults: list[str] = []
                                view_args = bend({}, self.schemas.get(self.roles['cart_view']), 'args', faults)
                                if not faults:
                                    view = self.store(self.roles['cart_view'], view_args)
                                else:
                                    view = None
                                if None is not view:
                                    self.cart_view, self.cart_seen_at = (view.get('observation'), self.step)
                                    change = self.own_call_seen(self.roles['cart_view'], self.cart_view)
                                    if change:
                                        self.queued = []
                                        held_327 = 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.cart_text() + '\nApply the change, fix the cart if needed, then check out.'
                                        return held_327
                swap = self.swap_hint()
                if 2 > self.sold_holds:
                    stuck = list((i for i in self.cart_ids() if i in self.gone))
                else:
                    stuck = []
                if not (swap or stuck):
                    chain, hold = ('', False) if heard else self.follow_chains()
                    if hold:
                        held_329 = self.advise(mark, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.cart_text())
                        return held_329
                    if self.elapsed() > CLOCK_HARD - 15:
                        return ''
                    if not heard:
                        self.cheap_sweep()
                    if heard:
                        read = []
                    else:
                        read = self.read_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_unread()
                    if heard:
                        verdict = ''
                    else:
                        verdict = self.review_cart(limit=SPEND_HARD - 0.008)
                    if verdict:
                        if read:
                            seen = "Kestrel read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n'
                        else:
                            seen = ''
                        held_330 = self.advise(mark, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text())
                        return held_330
                    if reread:
                        if SPEND_SOFT < self.cost or heard:
                            held_331 = "Not run yet (no step used). Kestrel re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise fix the cart first.'
                            return held_331
                    return ''
                else:
                    self.sold_holds += 0 if swap else 1
                    held_328 = 'Not run yet (no step used). ' + (swap or '{} in the cart is sold out: replace that line first.'.format(', '.join(stuck))) + '\nCart now: ' + self.cart_text()
                    return held_328
            else:
                held_326 = self.advise(mark, wider + '\nCart now: ' + self.cart_text())
                return held_326
        else:
            held_325 = self.advise(mark, 'Not run yet. ' + ladder + '\nCart now: ' + self.cart_text())
            return held_325

    def hold_refuse(self, mark: str='') -> str:
        held = self.refuse_objection()
        if held:
            held_332 = self.advise(mark, held)
            return held_332
        return ''

    def refuse_objection(self) -> str:
        if self.order_closest() and 2 > self.nofit_holds and (self.left() > 2) and self.roles.get('cart'):
            self.nofit_holds = self.nofit_holds + 1
            self.trace.append({'kind': 'nofit_held'})
            return 'Not run yet (no step used). For the case where nothing meets every requirement, the request wants its nearest match bought anyway, so ending without a purchase cannot satisfy it. Put the nearest match for each item in the cart (picked the way the request describes, such as its price margin), then check out.'
        elif not (self.left() <= 5 or self.elapsed() > CLOCK_HARD - 15):
            unsearched = self.unsearched_items()
            if not (unsearched and self.left() > 6):
                review = self.review_decline()
                if review:
                    held_334 = 'Not run yet (no step used). ' + review
                    return held_334
                if self.leads:
                    if self.left() > 4:
                        if 0 == self.refuse_holds:
                            self.refuse_holds = 1
                            self.trace.append({'kind': 'refuse_held', 'leads': list(self.leads)})
                            held_335 = 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (at the requested quantity) and check out; end without a purchase only if each one fails a requirement.'
                            return held_335
                if self.leads:
                    if self.left() > 4:
                        if 1 == self.refuse_holds:
                            self.refuse_holds = 2
                            first = self.leads[0]
                            self.review_cart([first], SPEND_HARD)
                            if self.verdicts.get(self.verdict_key([first])):
                                if first in self.leads:
                                    self.confirmed = first
                                    self.trace.append({'kind': 'refuse_held', 'confirmed': first})
                                    held_336 = "Not run yet (no step used). An independent check of {}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out.".format(first)
                                    return held_336
                return ''
            else:
                self.trace.append({'kind': 'unsearched_held', 'items': unsearched})
                held_333 = 'Not run yet (no step used). No search so far has looked for ' + '; '.join(unsearched) + '. Search for each of them (short words: the product, then a required value) and look at what fits before ending without a purchase.'
                return held_333
        else:
            return ''

    def unsearched_items(self) -> list[str]:
        if self.search_words:
            out = []
            for w in self.wants:
                words = set((x for x in key_stem(re.sub('[^A-Za-z0-9 ]+', ' ', '{} {}'.format(w.name, w.shelf or ''))) if 4 <= len(x)))
                if words:
                    if not words & self.search_words:
                        out.append(w.name)
            return out
        else:
            held_337 = []
            return held_337

    def hold_late_refuse(self) -> str:
        if self.left() <= 5 or self.elapsed() > CLOCK_SOFT:
            if not self.built:
                if not self.late_held:
                    if not self.safety_done:
                        if str(self.brief.get('no_fit') or '').lower() != 'no_purchase':
                            if self.roles.get('buy'):
                                self.late_held = True
                                note = self.safety_cart()
                                if note:
                                    held_338 = 'Not run yet (no step used). ' + note
                                    return held_338
        if not (self.left() > 5 or self.late_held):
            lines = list((i for i in self.built if not self.sold_out(i) and i not in self.dropped))
            missing = list((i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not self.cart_fresh()))))
            if not (len(self.wants) > len(lines) or len(missing) + 1 > self.left()):
                self.late_held = True
                self.trace.append({'kind': 'late_refuse_held', 'lines': lines, 'missing': missing})
                if missing:
                    redo = 'Put {} back in the cart at the requested quantities, then check out.'.format(', '.join(missing))
                else:
                    redo = 'Check out the cart as it is.'
                held_339 = 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'
                return held_339
            else:
                return ''
        else:
            return ''

    def push_before_refuse(self) -> str:
        if not (self.leads or self.pushed or self.left() <= self.max_steps // 3):
            self.pushed = True
            held_340 = "Not run yet (no step used). {} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it, search the product's plain name and synonyms, and open the listings whose rows could fit. End without a purchase only when those searches show no listing that meets everything.".format(self.left())
            return held_340
        else:
            return ''

    def carry_out(self, name: str, args: dict[str, Any], note: str) -> str:
        reply = self.store(name, args)
        if not None is reply:
            seen = reply.get('observation')
            if name == self.roles.get('compare'):
                many = self.compare_arg()
            else:
                many = None
            if many:
                if isinstance(args.get(many), list):
                    if 1 < len(args[many]):
                        if any((str(i) in flat(seen) for i in args[many])):
                            self.compare_ok = max(self.compare_ok, len(args[many]))
                        else:
                            self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                            note += '\n[Kestrel] The side-by-side view showed nothing for {} listings: send at most {} per call.'.format(len(args[many]), self.compare_room())
            mark = name + flat(args)
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all((isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))))
            self.take_in(seen)
            if name == self.roles.get('compare'):
                rows = self.records_in(seen)
                if rows:
                    self.shows_facts[name] = self.shows_facts.get(name, False) or any((attr_block(r) for _, r in rows))
            readers = set((self.roles.get(r) for r in ('search', 'open', 'offer', 'compare'))) - {None}
            if name in readers - {self.roles.get('search')}:
                self.note_view(name, seen)
            held_341 = self.is_buy(name, args)
            if not held_341:
                held_341 = self.is_refuse(name, args)
            ending = held_341
            not_search = set((tool for job, tool in self.roles.items() if 'search' != job))
            if name not in not_search:
                if not ending:
                    self.note_search(name, args, seen)
            if name not in not_search and (not ending):
                paged = self.read_on(name, args, seen)
            else:
                paged = ''
            self.learn_views(name, seen)
            if name == self.roles.get('cart'):
                self.track_cart(name, args, seen)
            if name == self.roles.get('cart_view'):
                self.cart_view, self.cart_seen_at = (seen, self.step)
            if self.is_buy(name, args):
                self.buys = self.buys + 1
                if not reply.get('finished'):
                    self.refused_at = self.step
            if name == self.roles.get('search'):
                if not ending:
                    note += self.hidden_note(args, seen)
            text = (self.show_rows(seen) if name == self.roles.get('search') else self.show(seen)) + paged + note
            if name not in not_search:
                if not ending:
                    text += self.cheapest_line([seen] + self.paged_obs)
            if name == self.roles.get('search'):
                if not self.records_in(seen):
                    text += self.empty_search_note(args)
            plain = text
            proof = self.evidence(seen)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart'):
                if any((whole(v) == 0 for v in args.values() if not isinstance(v, str))):
                    if any((isinstance(v, str) and v in self.rows and holds_id(seen, v) for v in args.values())):
                        text += '\n[Kestrel] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            if name == self.roles.get('cart'):
                text += self.units_note(name, args)
            self.note_sold = False
            lines = self.loose_lines(seen)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
            if self.is_buy(name, args):
                if not reply.get('finished'):
                    if self.note_sold or any((x in self.spoken(seen) for x in self.sold_lines)):
                        found = self.find_sold()
                        if found:
                            text += "\n[Kestrel re-read the cart's lines after the refusal]\n" + found
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes seen on Kestrel's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, plain)
            why = self.world_due(name, seen)
            if why:
                self.read_world(why)
            held_342 = text + (' [episode finished]' if reply.get('finished') else '')
            return held_342
        else:
            return 'The call failed (network or server). You may try again.'

    def empty_search_note(self, args: dict[str, Any]) -> str:
        vocab: set[str] = set()
        for row in self.rows.values():
            for value in row.values():
                if isinstance(value, str):
                    if 160 >= len(value):
                        held_344 = value.lower()
                        held_343 = re.findall('[a-z0-9]+', held_344)
                        vocab.update(held_343)
        if not 30 > len(vocab):
            held_345 = (self.schemas.get(self.roles.get('search') or '') or {}).get('properties')
            if not held_345:
                held_345 = {}
            props = held_345
            free = list((k for k, spec in props.items() if isinstance(spec, dict) and 'string' in type_list(spec) and ('enum' not in spec)))
            words = list((w for k in free if isinstance(args.get(k), str) for w in re.findall('[a-z0-9]+', args[k].lower()) if 1 < len(w)))
            odd = list((w for w in dict.fromkeys(words) if w not in vocab))
            if odd:
                held_346 = '\n[Kestrel] No rows. Query words that no listing row seen so far uses: ' + ', '.join(odd[:5]) + ". Words like these (codes, terms from a page's details) rarely match this store's search: search with the product's plain name plus one value word that listing rows use, then check the details on the opened pages."
                return held_346
            else:
                return ''
        else:
            return ''

    def learn_views(self, name: str, seen: Any) -> None:
        skip = (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'))
        if not (not self.one_id_arg(name) or name in skip):
            if isinstance(seen, dict):
                listing = self.id_of(seen)
            else:
                listing = None
            if listing:
                self.shows_facts[name] = self.shows_facts.get(name, False) or bool(attr_block(seen))
                if name == self.roles.get('open'):
                    if not self.shows_facts[name]:
                        other = next((t for t in self.schemas if name != t and self.one_id_arg(t) and (self.shows_facts.get(t) is not False) and (t not in skip + (self.roles.get('search'),))), None)
                        if other:
                            self.roles['open'] = other
                if not self.shows_facts[name]:
                    if name != self.roles.get('open'):
                        if not self.roles.get('offer'):
                            self.roles['offer'] = name
                if name == self.offer_tool():
                    self.offer_seen[listing] = self.step
            else:
                return
        else:
            return

    def units_note(self, name: str, args: dict[str, Any]) -> str:
        if self.cart_sets_units(name):
            held_347 = (self.schemas.get(name) or {}).get('properties')
            if not held_347:
                held_347 = {}
            props = held_347
            arg = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in type_list(s)), None)
            ident = next((v for v in args.values() if isinstance(v, str) and v in self.rows), None)
            if arg:
                units = whole(args.get(arg))
            else:
                units = None
            held_348 = set((w.qty for w in self.wants))
            wanted = sorted(held_348)
            if not (not arg or not ident or None is units or (units in wanted)):
                if 1 == len(wanted):
                    need = '{}'.format(wanted[0])
                else:
                    need = ' or '.join(map(str, wanted))
                if not 0 == units:
                    held_350 = "\n[Kestrel] {} is the line's unit count, not a line position: {} now holds {} unit(s), and 0 removes a line. The request needs {} unit(s) of an item's listing, so send {}={} for it.".format(arg, ident, units, need, arg, need)
                    return held_350
                else:
                    held_349 = "\n[Kestrel] {}=0 took {} out of the cart: {} is the line's unit count, not a line position. To keep a listing in the cart, send {}={} for it.".format(arg, ident, arg, arg, need)
                    return held_349
            else:
                return ''
        else:
            return ''

    def track_cart(self, name: str, args: dict[str, Any], seen: Any) -> None:
        self.note_cart(name, args)
        self.cart_said = flat(seen)
        self.recent = []
        counts = list((whole(v) for v in args.values() if not isinstance(v, str) and whole(v) is not None))
        for value in args.values():
            if counts:
                if isinstance(value, str):
                    if value in self.rows:
                        if 0 >= counts[0] and (not holds_id(seen, value)):
                            self.dropped.add(value)
                        elif 0 < counts[0]:
                            self.dropped.discard(value)
        lines = list((i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped))
        if len(self.built) <= len(lines):
            self.built = lines

    def finish(self, again: bool=False) -> str:
        if not (self.finished or self.broke or self.left() <= 0):
            buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
            if self.safety_due() or (not self.safety_done and (not self.cart_lines) and (not self.buys) and buy and (self.left() >= 2) and (str(self.brief.get('no_fit') or '').lower() != 'no_purchase')):
                self.safety_cart()
            if self.cart_fresh():
                holding = bool(self.cart_ids())
            else:
                holding = bool(self.cart_lines)
            if buy:
                if holding or (self.cart_touched and (not self.cart_fresh()) and (self.left() >= 2)):
                    faults: list[str] = []
                    held_351 = dict(self.brief.get('buy_args') or {})
                    held_352 = self.schemas.get(buy)
                    args = bend(held_351, held_352, 'args', faults)
                    if not faults:
                        if self.left() >= 2:
                            self.gate('order')
                        if not (self.finished or self.left() <= 0):
                            reply = self.store(buy, args)
                            self.trace.append({'kind': 'forced_buy', 'reply': cut(reply, 200)})
                            if self.finished or self.left() <= 0:
                                return ''
                            elif None is not reply:
                                if again:
                                    self.buys = self.buys + 1
                                    self.refused_at = self.step
                                    held_353 = reply.get('observation')
                                    self.own_call_seen(buy, held_353)
                                    held_354 = 'the checkout was refused: ' + self.show(reply.get('observation'))
                                    return held_354
                        else:
                            return ''
            if refuse:
                faults = []
                held_357 = dict(self.brief.get('refuse_args') or {})
                held_355 = self.quiet(refuse, held_357)
                held_356 = self.schemas.get(refuse)
                args = bend(held_355, held_356, 'args', faults)
                if not faults:
                    if self.left() >= 2:
                        self.gate('end')
                    if not (self.finished or self.left() <= 0):
                        reply = self.store(refuse, args)
                        self.trace.append({'kind': 'forced_refuse', 'reply': cut(reply, 200)})
                        if None is not reply:
                            if not self.finished:
                                held_358 = reply.get('observation')
                                self.own_call_seen(refuse, held_358)
                                held_359 = 'ending without a purchase was refused: ' + self.show(reply.get('observation'))
                                return held_359
                    else:
                        return ''
            return ''
        else:
            return ''

    def trim(self) -> None:
        if not sum((len(flat(m.get('content'))) for m in self.chat)) <= CHAT_CHARS:
            replies = list((i for i, m in enumerate(self.chat) if m.get('role') == 'tool'))
            for index in replies[:max(1, len(replies) // 2)]:
                held_360 = self.chat[index].get('content')
                if not held_360:
                    held_360 = ''
                content = held_360
                if 600 < len(content):
                    ids = list((i for i in dict.fromkeys(re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)) if i in self.rows))[:20]
                    self.chat[index]['content'] = content[:400] + '...(older result shortened; listings in it: {})'.format(', '.join(ids))
                    self.status_stale = True
        else:
            return

    def walk(self, turn_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (turn_cap > self.turns) and (spend_cap > self.cost):
            if self.elapsed() > CLOCK_HARD:
                self.trace.append({'kind': 'clock', 'secs': round(self.elapsed())})
                break
            elif not (self.time_to_ask() and self.elapsed() < CLOCK_SOFT - 40):
                if self.safety_due():
                    note = self.safety_cart()
                    if note:
                        self.chat.append({'role': 'user', 'content': note + '\n' + self.status()})
                        continue
                self.trim()
                self.turns = self.turns + 1
                message = self.model_turn(self.chat, [WALKER, BACKUP], 3000, self.offered())
                if not None is message:
                    calls = []
                    for number, call in enumerate(message.get('tool_calls') or []):
                        if isinstance(call, dict):
                            if isinstance(call.get('function'), dict):
                                fn = call.get('function')
                            else:
                                fn = {}
                            raw = fn.get('arguments')
                            if isinstance(raw, dict):
                                parsed = raw
                            else:
                                parsed = as_json(raw) if isinstance(raw, str) and raw.strip() else {}
                            calls.append({'id': str(call.get('id') or 't{}_{}'.format(self.turns, number)), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': flat(parsed if isinstance(parsed, dict) else {})}})
                        else:
                            continue
                    content = message.get('content')
                    if isinstance(content, str):
                        content = content
                    else:
                        content = flat(content) if content else ''
                    if not calls:
                        idle = idle + 1
                        self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                        self.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.status()})
                        if not 3 <= idle:
                            continue
                        else:
                            break
                    idle = 0
                    committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
                    if not self.asked:
                        if self.roles.get('ask'):
                            if self.msgs_free() > 0:
                                if self.left() > 8:
                                    if any((c['function']['name'] in committing for c in calls)):
                                        self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                                        self.chat += list(({'role': 'tool', 'tool_call_id': c['id'], 'content': 'Not run (no step used): the shopper is asked first.'} for c in calls))
                                        summary = self.interview()
                                        self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.status()})
                                        continue
                    self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                    self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': list((c['function'] for c in calls))})
                    replies = self.run_calls(calls)
                    for index, (cid, text) in enumerate(replies):
                        self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text + ('\n' + self.status() if len(replies) - 1 == index else '')})
                else:
                    self.misses = self.misses + 1
                    if not (self.no_credit or CALL_CAP <= self.calls or 2 < self.misses):
                        continue
                    else:
                        break
            else:
                summary = self.interview()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue

    def run(self) -> list[dict[str, Any]]:
        if not self.finished:
            self.read_request()
            self.trace.append({'kind': 'brief', 'brief': self.brief})
            opening = self.view.get('opening')
            if None is not opening:
                self.take_in(opening)
            intro = "SHOPPER'S REQUEST:\n" + self.request
            if None is not opening:
                intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
            intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
            self.chat = [{'role': 'system', 'content': WALK}, {'role': 'user', 'content': intro}]
            self.walk(TURN_CAP, SPEND_HARD)
            if not self.finished:
                turned = self.finish(again=True)
                if turned:
                    if not self.finished:
                        if not self.broke:
                            if not self.no_credit:
                                if self.left() > 1:
                                    if self.elapsed() < CLOCK_HARD:
                                        self.chat.append({'role': 'user', 'content': '[Kestrel] Kestrel tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                                        self.walk(self.turns + 10, SPEND_HARD + 0.01)
                if not self.finished:
                    self.finish()
            self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
            held_362 = self.trace
            return held_362
        else:
            held_361 = self.trace
            return held_361

class Kestrel(Wire, Books, Talk, Close, Moves):

    def __init__(self, problem: dict[str, Any]):
        env = problem['environment']
        self.binding = dict(env['binding'])
        self.view = env['policy_view']
        self.request = str(self.view.get('query') or '')
        self.tools = list((t for t in self.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)))
        self.schemas = dict(((t['function']['name'], t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}}) for t in self.tools))
        held_363 = self.view.get('max_steps')
        steps = whole(held_363)
        if None is steps:
            if isinstance(self.view.get('max_steps'), float):
                steps = int(self.view['max_steps'])
        if steps and 0 < steps:
            self.max_steps = steps
        else:
            self.max_steps = 30
        self.step, self.turns, self.calls, self.misses = (0, 0, 0, 0)
        self.started = time.time()
        self.finished, self.broke, self.no_credit = (bool(self.view.get('finished')), False, False)
        self.waited, self.cost = (0, 0.0)
        self.trace: list[dict[str, Any]] = []
        self.chat: list[dict[str, Any]] = []
        self.brief: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.wants: list[Want] = []
        self.budget: Any = None
        self.process: list[dict[str, Any]] = []
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.msg_limit = self.msg_used = self.reserve = self.gate_msgs = 0
        self.asked = False
        self.done_log: list[str] = []
        self.done_steps: list[int] = []
        self.gate_ok: set[tuple[str, str]] = set()
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.seen_order: list[str] = []
        self.opened: set[str] = set()
        self.id_key: str | None = None
        self.id_args: set[tuple[str, str]] = set()
        self.shows_facts: dict[str, bool] = {}
        self.view_fields: dict[str, set[str]] = {}
        self.listing_fields: dict[str, tuple[int, set[str]]] = {}
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.pages_read = 0
        self.paged_obs: list[Any] = []
        self.compare_ok = self.compare_bad = 0
        self.heard: set[str] = set()
        self.notes: list[str] = []
        self.queued: list[str] = []
        self.version = 0
        self.recent: list[str] = []
        self.unavailable: set[str] = set()
        self.gone: dict[str, int] = {}
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.cart_touched = False
        self.cart_at = self.cart_seen_at = -1
        self.cart_view: Any = None
        self.cart_lines: dict[str, int] = {}
        self.cart_said = ''
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.offer_seen: dict[str, int] = {}
        self.hidden_wait: dict[str, list[str]] = {}
        self.hidden_score = {True: 0, False: 0}
        self.hidden_told = False
        self.buys = 0
        self.refused_at = -1
        self.reviewed: set[str] = set()
        self.verdicts: dict[str, bool] = {}
        self.audited = self.pushed = self.late_held = self.completed = self.wider_held = False
        self.ending_advised: set[str] = set()
        self.leads: list[str] = []
        self.lead_items: dict[str, list[str]] = {}
        self.confirmed: str | None = None
        self.refuse_holds = self.sold_holds = self.trails = self.chain_holds = 0
        self.cheaper_marks: set[str] = set()
        self.world_log: list[str] = []
        self.world: dict[str, Any] = {}
        self.world_reads = 0
        self.world_marks: set[str] = set()
        self.nofit_holds = 0
        self.plan_tries = 0
        self.safety_done = False
        self.ladder_marks: set[str] = set()
        self.short_holds = 0
        self.search_words: set[str] = set()
        self.search_trail: dict[str, dict[str, Any]] = {}
        self.sweep_marks: set[str] = set()
        self.key_aliases: dict[tuple[str, str], list[str]] = {}
        self.world_compare = 0
        self.page_most = 0
        self.status_sent: dict[str, str] = {}
        self.status_calls = 0
        self.status_stale = False

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Kestrel(problem_data)
    except Exception as exc:
        held_364 = [{'kind': 'crash', 'error': '{}: {}'.format(type(exc).__name__, exc)[:400]}]
        return held_364
    try:
        held_365 = episode.run()
        return held_365
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': '{}: {}'.format(type(exc).__name__, exc)[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': '{}: {}'.format(type(again).__name__, again)[:400]})
        held_366 = episode.trace
        return held_366
__all__ = ['agent_main']
