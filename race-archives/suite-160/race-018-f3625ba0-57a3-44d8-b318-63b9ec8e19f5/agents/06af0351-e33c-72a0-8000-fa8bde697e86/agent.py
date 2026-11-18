from __future__ import annotations
import json
import random
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient
LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'
LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.021
SPEND_HARD = 0.05
HEAVY_ROOM = 0.013
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
PAGE_CHARS = 2600
OBS_LIMIT = 12000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3
SWEEP_PAGES = 4
SWEEP_OPENS = 3
HARD_END = 280.0
FINISH_SECONDS = 36.0
MODEL_SECONDS = 10.0
LAST_CALL_SECONDS = 12.0
PLAN_MODELS = ('anthropic/claude-sonnet-5.5', 'anthropic/claude-haiku-5.5')
SPEND_CEILING = 0.054
CRITICAL_CEILING = 0.057
REPAIR_ROOM = 0.008
REFUSAL_ROOM = 3
LEAD_DEPTH = True
SET_APART = True
NO_WASTE = True
ASK_EARLY = True
EARLY_OPENED = (4, 2)
SKIP_PUZZLED = True
APART_AFTER_ROUND = True
ASK_BY_SECONDS = 170.0
ALIKE_NEAR_ONLY = True
PLAN_SKIP_CODES = True
PLAN_LIGHT_FROM_MSGS = 5
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=60, max_retries=3, retry_delay=1)
GUIDE = ('You act for Marlo, who shops for one person in an online store; your only way to change anything is a store tool call. An episode earns 1 or 0. It earns 1 when it ends right: either an order whose cart has each requested product once, in the requested number of units, every one meeting all requirements that hold at checkout, with the total within budget; or, when some requested product truly has no fitting listing in this store, the no-purchase ending made as the request describes.\n\n'
    'TOOLS AND STEPS\n'
    '- Calls are counted against a step limit; thinking is not. Read the schemas: every name and allowed value must be copied from them exactly.\n'
    '- Results may arrive a few rows at a time, behind a cursor or an offset, narrowed by category or by the opening words of a query. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). If a filtered search comes back empty, the filter was probably misspelled: reuse the label as the rows print it.\n'
    '- A row is a preview. Listings often have more than one view: attributes in one, the offer with seller notes and links to other listings in another, sometimes several listings side by side. Judge requirements from the view with the attributes, so open it for each listing you consider buying before you cart it; before an order Marlo also opens each view of every listing being bought and reports what they say. Never reopen an unchanged view.\n'
    '- Only identifiers taken from result rows are listing ids. Model numbers and bracketed part codes printed in titles or descriptions are search terms, not ids.\n'
    "- Messages to the shopper go through Marlo's interviewer only; its answers land in the ledger.\n"
    '- If the request wants something done along the way (approval, a message, a check), Marlo handles what is due at checkout or at the ending when you call that tool, so never stall or give up because of it; everything else you do with store tools.\n\n'
    'JUDGING A LISTING\n'
    '- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n'
    "- A requirement is met by the listing's own attribute for that property, read loosely for the spelling of the key and strictly for the meaning of the value: any list entry counts, plural, case and plain synonyms are fine, units convert, limits include the boundary, and a range has to sit entirely inside the limit. A code or abbreviation does not stand in for a word. A value holding the required value's words still matches, extra words around them included; other words name another value unless they are a plain synonym.\n"
    '- If the attribute for that very property says something different, the listing fails, whatever its title claims. An attribute about a neighbouring part or aspect (a lid colour when the body colour is asked) says nothing either way. [evidence] lines point at keys named like a requirement.\n'
    '- Without such an attribute, the wanted value may still appear in another attribute, the title, the description or the seller notes; a number of a different kind, or one that never says what it measures, does not count.\n'
    '- A category label that already carries a required value settles that requirement.\n'
    '- Some requirements borrow their value from another listing. Settle those first: find that listing, open its pages fully, follow any page that refers on to a further listing, and read the value off the last one; the order only counts if every page in that chain was opened. A rule to differ from that value rules out only listings sharing it.\n'
    '- Before buying a listing whose page refers to another page, read that other page.\n'
    '- Products that must work together are chosen as a pair from what both listings say (what one is made for, a model or series code both mention). If no suitable accessory works with the device you hold, change the device rather than give up.\n'
    '- Exclusions and closed lists bind; wishes only break ties. Unavailable listings cannot be bought. Price times units, summed over the cart, must stay within the budget; equal is allowed.\n\n'
    'PICKING\n'
    "- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Marlo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n"
    '- Most or least of a measure means comparing all fitting listings on it; a listing silent about the measure cannot win.\n'
    '- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n'
    '- Put a product in the cart as soon as its fitting listing is open (after the comparison, for cheapest, most or least), then go on. Stock can vanish after carting, and an early cart leaves time to swap in another. Each requested product gets its own cart line at its own count.\n\n'
    'WHEN THINGS CHANGE\n'
    '- Budgets, requirements, prices and stock may shift during the episode, and a checkout attempt is where such news usually arrives; notes count wherever they appear and the ledger shows what holds now. The episode survives a refusal: take in the change, rebuild the cart from candidates already judged, and try the checkout again.\n'
    '- When the store turns down a checkout or the no-purchase ending, it is saying an order is still expected, not that nothing fits: from then on, finish by buying.\n\n'
    'ENDING\n'
    '- Make the cart exactly the chosen lines, look at it once, then send the checkout as a call of its own.\n'
    '- End without buying only when your searching has shown that some requested product has no listing meeting everything. Being short of steps or time proves nothing like that; then check out the best cart. Word that ending as the request asks; any remark stays brief and names no listing.\n'
    '- Each reply must contain at least one tool call; independent calls (several listing views, say) go together in one reply to save time. Write little: the facts you have, then your next move.')
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = ('Task: pick the interview questions for a shopping assistant. Each question names exactly one property of one product. The customer has a handful of firm, unstated wishes; each concerns a single property of a single requested product, and the customer reveals one only when a question names that same property. Questions about properties the customer does not care about, or never considers, come back empty or baffled and are wasted.\n'
    'Material: the customer request; the products, each with what the request already pins down; for each product, the attribute keys present on its opened listings, the count of listings carrying each, and sample values. Hidden wishes may also concern properties no key mentions.\n'
    'Ordering, per product, strongest first:\n'
    '(a) whatever the request explicitly wants checked with the customer (see ASK FIRST), named by its property;\n'
    "(b) properties the story hints at without a value: a line about something left to the assistant or beyond the customer's judgement, about someone else whose taste has the last word, about the eventual user, about the place it is meant for, about the look it should have, or about what it will hold or carry inside; take such lines in the order the story gives them and give the single property each one points at. Name it with the product's listed key whose meaning covers what that line talks about, and when more than one listed key could, the one more of its listings carry. A line about the wearer or eventual user points at the product's gender or audience key; a line about how it looks points at the colour, pattern or finish key that most of its listings carry. A hinted property that none of the product's listed keys expresses comes only after that product's three most carried keys;\n"
    "(c) the remaining keys: first the ones a buyer of this kind of product settles before anything else when choosing among its listings, judged from what you know about buying such products, and among those the keys most of its listings carry; when five or more listings were opened, drop keys that only one or two of them carry.\n"
    'Exclude: properties the request pins down for that product (a key whose values merely echo a required value does not count as pinned), duplicates of a property already picked, keys marked [several values in one], identifiers, model names, warranty, delivery and service terms. Quantities and pack sizes do count as properties. A key sharing a word with a pinned requirement while describing a different part or aspect stays eligible.\n'
    '"label": the bare property name; for a key, every word of it spelled as the key spells it ("strap width" for strapWidth, not just "width"), without values, brackets or a second property.\n'
    'Up to COUNT picks per product. Output exactly one JSON object, nothing else: {"ask": [{"item": product number, "label": "...", "hint": "at most 8 words quoted from the request, or empty"}]} listing each product\'s picks in ranked order.')
HEAR = ('Below: questions an assistant put to a customer, each with the reply. Decode the replies into one JSON object (no other text) with exactly these fields:\n'
    '- needs: list of {"item": index of the product concerned, "need": "<property>: <exact value or limit>", "plain": true when that value could appear word for word on a listing / false for a bound, span, exclusion or comparison, "points": true when the value is to be looked up on some other listing / else false}\n'
    '- drop: list of {"item": index, "property": requirement the customer lets go of}\n'
    '- free: list of {"item": index, "property": property the customer says they have no preference on}\n'
    '- crowded: indexes of questions whose reply complains of being asked too much at once or wants topics taken singly (a reply that only shows the message was not understood goes under puzzled instead)\n'
    '- puzzled: indexes of questions whose reply shows the customer could not make sense of what was asked\n'
    '- budget: new overall spending cap as a number, else null\n'
    'How to fill needs: one entry per firm wish a reply voices, value or bound kept with its unit (examples of the form: "under 40 cm", "round", "not plastic"); name it after the property the reply mentions, which may not be the one asked; fold several replies on one property into one entry; quote the value in the reply\'s own terms, nouns and units kept ("2 drawers or fewer", never "2 or fewer").\n'
    "When a reply ties the value to another listing (matching, differing from or near what some shop's product shows, that product singled out by a few of its traits), it is still ONE entry: the property the reply names, points true, and as value the reply's verbatim wording from the comparison word through the description's last word, including anything about a listing that page may name; never cut or reword it. Traits that only single out the other product are not entries for this one.\n"
    'A value given casually or in passing is still a stated value.')
APART = ('The customer was asked "QUESTION" and objected as though it bundled several properties. Listed below are the attribute keys of that product. Pick the ONE key the customer may have understood as included in "LABEL", copying it exactly from the list, or "" if none fits. Output exactly one JSON object, nothing else: {"key": "..."}')
PROPOSE = ('The assistant may still ask the customer about the products below but has run out of properties to ask about. For each product, suggest up to 3 properties of that kind of product on which a customer could hold a firm wish, none of them already asked or pinned by the request. Bare property names, no values. Output exactly one JSON object, nothing else: {"names": [{"item": product number, "name": "..."}]}')
NOTE = ('A shopping agent met the passage below in the middle of its work; the shopper or the store wrote it. Return a single JSON object and no other text:\n'
    '{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\n'
    'A swapped requirement counts as a drop of the old plus an add of the new; leave out no change.')
READING = ('whichever key carries the requirement\'s own property decides it, spelled any way; one fitting entry of a list is enough; plain synonyms and unit conversions are fine, bounds include their edge, and a value holding the required words passes; a key about a different part or aspect decides nothing; with no such key, the value can come from another attribute, the title, the description or the seller notes, though a figure that never says what it measures counts for nothing; a category label that carries the value meets it; the category label must match the requested item as a whole; a rule to differ from another listing fails only on that exact value in the records; an unavailable listing, or one over budget, is never a fit.')
CHECK = ('This is the last look before a shopping agent pays. On the records below alone, decide whether the cart is right as it stands.\nRecord reading: ' + READING + '\n'
    'Right means: one line for each requested product at the count asked, nothing else in the cart, no line unavailable, and price times count added over the lines within budget. Where a listing has several records, trust the newest.\n'
    'With a cheapest, most or least rule, weigh the other listings shown too: a clear full fit that beats a cart line on that rule makes the cart wrong.\n'
    'Return a single JSON object and no other text: {"ok": true or false, "problems": ["short concrete problem", ...], "better": [ids that ought to replace a cart line, best first], "lines": {"<cart listing id>": true when it satisfies all of its product\'s requirements, false when some record shows a miss}}. Use ok=false only for a fault the records show outright or a requirement no record speaks to.')
AUDIT = ('A shopping agent is convinced the store has nothing suitable and wants to stop without buying. Put that to the test with the records below. Record reading: ' + READING + '\n'
    'A ROW record is only a search row whose attributes nobody opened; it neither passes nor fails on a property the row does not mention.\n'
    'Return a single JSON object and no other text: {"fits": {"item number": [ids that look like full fits, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose pages could settle it, likeliest first, at most 6], "reason": "one sentence"}')
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page, or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = ("A shopping agent's cart holds a listing that can no longer be bought. Search the other records for listings able to fill that line instead. Record reading: " + READING + '\n'
    'Return a single JSON object and no other text: {"fits": [{"id": listing id, "price": its price as a number, as recorded} for each listing that looks like a full fit for that line\'s product, at most 6], "reason": "one sentence"}')
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null, "ceiling": the tool parameter that sets the highest price of the listings returned, or null}'
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5} ?\\d{3,6}'


def _codes(text: Any) -> list[str]:
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]


def _marks_by_position(text: Any) -> dict[str, Any]:
    """Model and part codes by position only: a code inside brackets is the listing's own mark, the others are codes its text names."""
    text = str(text or '')
    own = [c for inner in re.findall('[\\[(]([^\\[\\]()]{1,40})[\\])]', text) for c in _codes(inner)]
    named = [c for c in _codes(re.sub('[\\[(][^\\[\\]()]{1,40}[\\])]', ' ', text)) if c not in own]
    return {'own': own, 'named': named}


def _bundled(values: list[str]) -> bool:
    # most shown values list several things at once (a comma or semicolon in a value of three or more words)
    many = [v for v in values if re.search('[,;]', v) and len(v.split()) >= 3]
    return bool(values) and len(many) * 2 >= len(values)


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [entry for entry in _items(value) if isinstance(entry, dict)]


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def _clip(value: Any, limit: int, depth: int=0) -> Any:
    if depth > 7:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    if isinstance(value, dict):
        return {k: _clip(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v, limit, depth + 1) for v in value]
    return value

def _shorten(value: Any, limit: int, depth: int=0) -> Any:
    """Texts longer than the limit keep their start and say how many characters were left out."""
    if depth > 7:
        return '...'
    if isinstance(value, (dict, list)):
        pairs = value.items() if isinstance(value, dict) else enumerate(value)
        kept = [(k, _shorten(v, limit, depth + 1)) for k, v in pairs]
        return dict(kept) if isinstance(value, dict) else [v for _, v in kept]
    hidden = len(value) - limit if isinstance(value, str) else 0
    return value[:limit] + f'...(+{hidden} chars)' if hidden > 0 else value


def _json_obj(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    fence = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', text, re.S)
    if fence:
        text = fence.group(1)
    try:
        got = json.loads(text)
        return got if isinstance(got, dict) else None
    except ValueError:
        pass
    left, right = (text.find('{'), text.rfind('}'))
    if 0 <= left < right:
        try:
            got = json.loads(text[left:right + 1])
            return got if isinstance(got, dict) else None
        except ValueError:
            return None
    return None

def _words(key: str) -> str:
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def _stem(text: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _words(text).split()))

def _roots(text: str) -> set[str]:
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def _spelled_alike(one: str, two: str) -> bool:
    # the same word written two ways: one letter more or less (colour / color), or two neighbouring letters swapped
    # (fibre / fiber); a changed letter (height / weight), a short word or another first letter keeps them apart
    if one == two:
        return True
    if min(len(one), len(two)) < 4 or one[:1] != two[:1]:
        return False
    if len(one) == len(two):
        spots = [k for k, (x, y) in enumerate(zip(one, two)) if x != y]
        return len(spots) == 2 and spots[1] - spots[0] == 1 and one[spots[0]] == two[spots[1]] and one[spots[1]] == two[spots[0]]
    short, longer = (one, two) if len(one) < len(two) else (two, one)
    if len(longer) - len(short) != 1:
        return False
    first = next((k for k in range(len(short)) if short[k] != longer[k]), len(short))
    return short[first:] == longer[first + 1:]


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]

def _fits_kind(value: Any, kind: str) -> bool:
    if kind == 'string':
        return isinstance(value, str)
    if kind == 'integer':
        return isinstance(value, int) and (not isinstance(value, bool))
    if kind == 'number':
        return isinstance(value, (int, float)) and (not isinstance(value, bool))
    if kind == 'boolean':
        return isinstance(value, bool)
    if kind == 'array':
        return isinstance(value, list)
    if kind == 'object':
        return isinstance(value, dict)
    if kind == 'null':
        return value is None
    return True

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    if kinds and (not any((_fits_kind(value, k) for k in kinds))):
        for kind in kinds:
            if kind == 'integer' and isinstance(value, float) and value.is_integer():
                value = int(value)
                break
            if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value):
                number = float(value)
                value = int(number) if kind == 'integer' or number.is_integer() else number
                break
            if kind == 'string' and isinstance(value, (int, float)) and (not isinstance(value, bool)):
                value = str(value)
                break
            if kind == 'boolean' and isinstance(value, str) and (value.strip().lower() in ('true', 'false')):
                value = value.strip().lower() == 'true'
                break
            if kind == 'array' and (not isinstance(value, list)) and (value is not None):
                value = [value]
                break
        if not any((_fits_kind(value, k) for k in kinds)):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get('enum')
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            low = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and o.lower() == low]
            if match:
                value = match[0]
        if value not in options:
            faults.append(f'{where}: allowed values are {options}')
    if isinstance(value, str) and isinstance(schema.get('maxLength'), int) and (len(value) > schema['maxLength']):
        value = value[:schema['maxLength']]
    if isinstance(value, str) and isinstance(schema.get('minLength'), int) and (len(value) < schema['minLength']):
        faults.append(f'{where} is too short')
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        if isinstance(schema.get('minimum'), (int, float)) and value < schema['minimum']:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get('maximum'), (int, float)) and value > schema['maximum']:
            faults.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems']:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems']:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get('items'), dict):
            value = [_shape(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for name in [n for n in value if n not in props]:
                faults.append(f'{where}.{name} is not an allowed argument')
        for name in schema.get('required') or []:
            if name not in value:
                faults.append(f'{where}.{name} is required')
        value = {k: _shape(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
    return value

def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []

def _count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        return int(float(value))
    return None

def _id_like(value: Any, named: bool=False) -> bool:
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip():
        return False
    return named or bool(re.search('\\d', value))

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all((_id_like(v, 'id' in str(key).lower()) for v in values)):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None

def _in_rows(node: Any, ident: str, depth: int=0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, list):
        return any((isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node)) or any((_in_rows(v, ident, depth + 1) for v in node))
    if isinstance(node, dict):
        return any((_in_rows(v, ident, depth + 1) for v in node.values()))
    return False

def _fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


def _plain_value(text: str) -> Any:
    word = text.strip()
    if word in ('True', 'true'):
        return True
    if word in ('False', 'false'):
        return False
    if word in ('None', 'null', ''):
        return None
    if re.fullmatch('-?\\d{1,15}', word):
        return int(word)
    if re.fullmatch('-?\\d{1,12}\\.\\d{1,6}', word):
        return float(word)
    return word


def _looks_like_key(text: str) -> bool:
    return 0 < len(text) <= 40 and bool(re.fullmatch('[A-Za-z_][A-Za-z0-9_ .\\-]*', text)) and len(text.split()) <= 4


def _field_break(text: str, mark: str) -> str | None:
    """The punctuation that separates fields in a one-line record, learned from the record itself: between two
    consecutive key marks it shows up about once, and it shows up between most of them."""
    spots = [m.start() for m in re.finditer(re.escape(mark), text)]
    gaps = [text[a + 1:b] for a, b in zip(spots, spots[1:])]
    if len(gaps) < 1:
        return None
    best = None
    for char in {c for gap in gaps for c in gap if not c.isalnum() and (c == '\n' or not c.isspace()) and c != mark and (c not in '"\'')}:
        holding = [g for g in gaps if char in g]
        share = len(holding) / len(gaps)
        if share < 0.6:
            continue
        per = sum((g.count(char) for g in holding)) / len(holding)
        score = (round(per, 1), -share)
        if best is None or score < best[0]:
            best = (score, char)
    return best[1] if best else None


def _text_record(text: str, mark: str=':', least: int=3, depth: int=0) -> dict[str, Any] | None:
    """Read a record written as text ('name: value | name: value ...') into a dict, with nested 'a=b; c=d' parts."""
    if depth > 2 or len(text) < 8 or text.count(mark) < least:
        return None
    char = _field_break(text, mark)
    if not char:
        return None
    fields: dict[str, Any] = {}
    last = None
    for part in text.split(char):
        head, found, tail = part.partition(mark)
        if found and _looks_like_key(head.strip()) and head.strip() not in fields:
            last = head.strip()
            fields[last] = tail.strip()
        elif last is None:
            return None
        else:
            fields[last] = str(fields[last]) + char + part
    if len(fields) < least:
        return None
    for key, value in list(fields.items()):
        inner = _text_record(value, '=', 2, depth + 1) if isinstance(value, str) and value.count('=') >= 2 else None
        fields[key] = inner if inner is not None else _plain_value(value) if isinstance(value, str) else value
    return fields


def _structured(node: Any, depth: int=0, row: bool=True) -> Any:
    """Observations as structure. Only where a whole record can stand (the observation itself, a list entry, or the
    value of a small wrapper object) is JSON held in a string parsed, and a text record read into a dict when it gives
    at least three named fields with an identifier or a nested block. Field values inside a record are left alone."""
    if depth > 7:
        return node
    if isinstance(node, str):
        if not row:
            return node
        bare = node.strip()
        wrapped = bare[:1] + bare[-1:]
        if wrapped in ('{}', '[]'):
            try:
                return _structured(json.loads(bare), depth + 1)
            except ValueError:
                pass
        record = None if len(bare) > 60000 else _text_record(bare)
        keyed = record is not None and any((isinstance(v, dict) or (isinstance(v, str) and _id_like(v)) for v in record.values()))
        return record if keyed else node
    if isinstance(node, list):
        return [_structured(entry, depth + 1) for entry in node]
    if not isinstance(node, dict):
        return node
    wrapper = len(node) < 3
    return dict(((key, _structured(inner, depth + 1, wrapper)) for key, inner in node.items()))


def _shared_start(texts: list[str]) -> str:
    """Longest opening every text shares (the opening shared by the alphabetically first and last is shared by all)."""
    if not texts:
        return ''
    first, last = sorted(texts)[::len(texts) - 1 or 1][:2] if len(texts) > 1 else (texts[0], texts[0])
    cut = next((at for at, (x, y) in enumerate(zip(first, last)) if x != y), min(len(first), len(last)))
    return first[:cut]

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Ladder:
    # the asking policy. One property a message, the product chosen in code (no reveal yet first, then fewest
    # questions, then planner order; with three or more messages the lead product gets a second try when its first
    # found nothing), the product called by its store label, nothing re-asked after an empty, puzzled or
    # no-preference reply, one look-alike rephrase after a turn-back once every product was asked, refills so that no
    # message stays unsent, and an early start once each product has a few opened listings.

    def __init__(self, host: Any):
        self.host = host
        self.plan_order: list[int] = []
        self.lanes: dict[int, list[str]] = {}
        self.tally: dict[int, list[int]] = {}
        self.closed: dict[int, set[str]] = {}
        self.sent_texts: set[str] = set()
        self.queue: list[tuple[int, str, str]] = []
        self.proposed = False
        self.unread: list[str] = []

    def ready(self) -> bool:
        h = self.host
        floor = EARLY_OPENED[0 if len(h.items) == 1 else 1]
        return min((h.opened_for(n) for n in range(1, len(h.items) + 1)), default=0) >= floor

    def wants_turn(self) -> bool:
        h = self.host
        if h.asked or 'ask' not in h.roles or h.msgs_free() < 1 or h.left() < 9:
            return False
        late = len(h.full) > 3 * len(h.items) or h.step >= max(8, h.max_steps // 4) or h.seconds_left() <= ASK_BY_SECONDS
        return late or (ASK_EARLY and self.ready())

    def pending(self) -> bool:
        return not self.host.asked

    def open_labels(self, number: int) -> list[str]:
        shut = self.closed.setdefault(number, set())
        return [x for x in self.lanes.get(number, []) if _words(x) not in shut and not any((self.same_property(number, x, y) for y in shut))]

    def same_property(self, number: int, one: str, two: str) -> bool:
        # two labels for one property of this product: the same words once the product's own name words are set
        # aside, each pair spelled alike (colour / color, lamp colour / colors)
        own = _roots(self.host.items[number - 1].name) | _roots(self.host.items[number - 1].shelf or '')
        core = lambda text: [w for w in sorted(_stem(text)) if w not in own] or sorted(_stem(text))
        a, b = core(one), core(two)
        return len(a) == len(b) and all((_spelled_alike(x, y) for x, y in zip(a, b)))

    def bare(self, label: str) -> str:
        # a short code leading several of the store's keys ('fa pattern', 'fa general styles') is left out only when the
        # store's own keys also show the property without it and the request never uses that code as a word
        words = _words(label).split()
        reader = getattr(self.host, 'store_keys', None)
        keys = reader() if reader else set()
        if len(words) < 2 or len(words[0]) > 3 or sum((1 for k in keys if k.split()[:1] == words[:1])) < 2 or words[0] in _words(getattr(self.host, 'request', '')).split():
            return label
        rest = ' '.join(words[1:])
        return rest if rest in keys else label

    def next_question(self, depth: bool) -> tuple[int, str, str]:
        if self.queue:
            return self.queue.pop(0)
        choices = [(n, self.open_labels(n)) for n in self.lanes]
        choices = [(n, left) for n, left in choices if left]
        if not choices:
            return (0, '', '')
        lead = self.plan_order[0] if self.plan_order else 1
        if depth and self.tally.get(lead, [0, 0]) == [1, 0]:
            again = [left for n, left in choices if n == lead]
            if again:
                return (lead, again[0][0], '')
        rank = {n: i for i, n in enumerate(self.plan_order)}
        number, left = min(choices, key=lambda c: (self.tally[c[0]][1] > 0, self.tally[c[0]][0], rank.get(c[0], len(rank) + c[0])))
        return (number, left[0], '')

    def wording(self, number: int, label: str, apart: str) -> str:
        name = self.host.shelf_word(number)
        return f'Leaving the {apart} aside, what {label} would you like for the {name}?' if apart else f'What {label} would you like for the {name}?'

    def planner_prompt(self, table: list[tuple[str, str, int, list[str]]], count: int) -> str:
        h = self.host
        parts = [PLAN.replace('COUNT', str(max(1, count))), 'REQUEST:\n' + h.request]
        parts.append('PRODUCTS:\n' + '\n'.join((f'product {n}: {item.name} - pinned by the request: ' + ('; '.join(item.musts) or 'nothing') for n, item in enumerate(h.items, 1))))
        wanted = [str(x) for x in _items(h.brief.get('ask_first')) if str(x).strip()]
        if wanted:
            parts.append('ASK FIRST (points the request says to confirm with the customer before choosing):\n' + '\n'.join(wanted))
        rows, shelf = [], None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != shelf:
                shelf = label
                rows.append(f'listings labelled {label!r}:')
            rows.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + (' [several values in one]' if _bundled(values) else ''))
        parts.append('KEYS ON OPENED LISTINGS:\n' + ('\n'.join(rows) or '(no listing opened yet)'))
        return '\n\n'.join(parts)

    def planner_picks(self, plan: dict[str, Any]) -> tuple[dict[int, list[str]], list[int]]:
        h = self.host
        picks: dict[int, list[str]] = {n: [] for n in range(1, len(h.items) + 1)}
        hinted: list[int] = []
        self.story_picks = {n: set() for n in picks}
        for pick in _dicts(plan.get('ask')):
            number = h.product_no(pick.get('item'))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            pinned = [_stem(m.split(':', 1)[0]) for m in h.items[number - 1].musts]
            lone = PLAN_SKIP_CODES and len(_words(label).split()) == 1 and re.fullmatch(NOT_PROPERTY, _words(label)) is not None and _words(label) != 'condition'
            if label and len(label.split()) <= 5 and _words(label) not in {_words(x) for x in picks[number]} and _stem(label) not in pinned and not lone:
                picks[number].append(label)
            if label and len(label.split()) <= 5 and str(pick.get('hint') or '').strip():
                self.story_picks[number].add(_words(label))
                if number not in hinted:
                    hinted.append(number)
        return picks, hinted

    def plan_lanes(self, count: int) -> None:
        h = self.host
        table = h.key_table()
        # five or more messages bill more shopper replies to the run: the planner then starts on the light model
        many = PLAN_LIGHT_FROM_MSGS and getattr(h, 'msg_limit', 0) >= PLAN_LIGHT_FROM_MSGS
        plan = h.ask_json(self.planner_prompt(table, count), sorted(PLAN_MODELS, key=lambda m: m != LIGHT) if many else list(PLAN_MODELS), 3500, None) or {}
        picks, hinted = self.planner_picks(plan)
        self.plan_order = hinted + [n for n in picks if n not in hinted]
        trim = lambda text: re.sub('(?<=[a-z]{3})s\\b', '', _words(text))
        for number, chosen in picks.items():
            keys = h.product_keys(number)
            pinned = [_stem(m.split(':', 1)[0]) for m in h.items[number - 1].musts]
            listed = {trim(k): k for k in keys}
            lane: list[str] = []
            cued = self.story_picks.get(number, set()) if hasattr(self, 'story_picks') else set()
            for label in chosen + [k for k in keys if all((not _names(p, _stem(k)) for p in pinned))]:
                raw = _words(label)
                label = self.bare(listed.get(trim(label), label))
                if raw in cued:
                    cued.add(_words(label))
                if all((_words(label) != _words(x) and not self.same_property(number, label, x) for x in lane)):
                    lane.append(label)
            heavy = {name for shelf, name, _, values in table if h.owner_of(shelf) == number and _bundled(values)}
            self.lanes[number] = self.keep_top(number, sorted(lane, key=lambda x: x in heavy), table)
            self.tally[number] = [0, 0]
            self.closed.setdefault(number, set())
        h.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'order': self.plan_order, 'lanes': {n: v[:8] for n, v in self.lanes.items()}})

    def keep_top(self, number: int, lane: list[str], table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        # the product's three most carried keys keep a place among its first three questions: an unhinted pick that far
        # fewer of its opened listings carry gives way (several products: it moves to the fourth place)
        h = self.host
        seen = max(1, h.opened_for(number))
        rows = sorted(((_words(self.bare(name)), carried / seen) for shelf, name, carried, _ in table if h.owner_of(shelf) == number), key=lambda pair: pair[1])
        share = dict(rows)
        marked = getattr(self, 'story_picks', {}).get(number, set())
        lane = list(lane)
        for top in sorted(share, key=share.get, reverse=True)[:3]:
            words = [_words(x) for x in lane]
            spot = None if top in words[:3] or top not in words else next((i for i, w in enumerate(words[:3]) if w not in marked and share.get(w, 0.0) + 0.34 <= share[top]), None)
            if spot is None:
                continue
            key = lane.pop(words.index(top))
            if len(h.items) == 1:
                lane.insert(spot, key)
            else:
                lane[spot], weak = key, lane[spot]
                lane.insert(min(3, len(lane)), weak)
        return lane

    def look_alike(self, number: int, label: str, asked: str) -> str:
        # one property name of this product (its listed keys and lane labels) the customer may have heard inside the
        # label: the only one sharing a word with it, else the light model's pick, taken only as an exact candidate
        h = self.host
        listed = [k for k in h.product_keys(number) + ([] if ALIKE_NEAR_ONLY else self.lanes.get(number, [])) if _words(k) != _words(label)]
        listed = [k for i, k in enumerate(listed) if all((_words(k) != _words(o) for o in listed[:i]))]
        near = [k for k in listed if _stem(k) & _stem(label)]
        if len(near) == 1 or not listed or (ALIKE_NEAR_ONLY and not near):
            return near[0] if near else ''
        if not h.room_for(MODEL_SECONDS + 4):
            return ''
        pool = (near or listed)[:24]
        got = h.ask_json(APART.replace('QUESTION', asked).replace('LABEL', label) + '\n\nKEYS:\n' + '\n'.join(('- ' + k for k in pool)), [LIGHT], 1500, 'low') or {}
        said = _words(str(got.get('key') or ''))
        return next((k for k in pool if said and _words(k) == said), '')

    def refill(self) -> bool:
        # lanes ran dry with messages left: the listed keys not yet asked; failing those, once, names the model proposes
        h = self.host
        grew = 0
        for number, lane in self.lanes.items():
            known = {_words(x) for x in lane} | self.closed[number]
            pinned = [_stem(m.split(':', 1)[0]) for m in h.items[number - 1].musts]
            fresh = []
            for k in h.product_keys(number):
                k = self.bare(k)
                if _words(k) not in known | {_words(x) for x in fresh} and all((not _names(p, _stem(k)) for p in pinned)) and not any((self.same_property(number, k, x) for x in lane)):
                    fresh.append(k)
            lane += fresh
            grew += len(fresh)
        if grew or self.proposed or not h.room_for(MODEL_SECONDS + 4):
            return grew > 0
        self.proposed = True
        about = [f'product {n}: {h.shelf_word(n)}; asked already: ' + (', '.join(sorted(self.closed.get(n, set()))) or 'nothing') + '; pinned by the request: ' + ('; '.join(item.musts) or 'nothing') for n, item in enumerate(h.items, 1)]
        got = h.ask_json(PROPOSE + '\n\nPRODUCTS:\n' + '\n'.join(about), [LIGHT, HEAVY], 1500, 'low') or {}
        for entry in _dicts(got.get('names')):
            number = h.product_no(entry.get('item'))
            name = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(entry.get('name') or ''))[0].strip().rstrip('?.')
            lane = self.lanes.get(number)
            pinned = [_stem(m.split(':', 1)[0]) for m in h.items[number - 1].musts]
            if lane is not None and name and len(name.split()) <= 4 and _words(name) not in self.closed[number] | {_words(x) for x in lane} and _stem(name) not in pinned:
                lane.append(name)
                grew += 1
        h.trace.append({'kind': 'propose', 'added': grew})
        return grew > 0

    def exchange(self, number: int, label: str, apart: str, told: list[str]) -> dict[str, Any] | None:
        # one question out, its reply read; None when nothing could be sent
        h = self.host
        text = self.wording(number, label, apart)
        if text in self.sent_texts:
            self.closed[number].add(_words(label))
            return {'skip': True}
        batch = h.send([(number, label, text)])
        if not batch:
            return None
        self.sent_texts.add(text)
        self.tally[number][0] += 1
        self.closed[number].add(_words(label))
        sizes = lambda: [len(item.told) for item in h.items]
        start = sizes()
        heard = h.hear(batch, (LIGHT, HEAVY))
        if heard.get('unread'):
            self.unread.append(batch[0][1])
        h.take_heard(heard, told)
        h.log_question(number, label, *batch[0][1:], heard)
        gained = [n + 1 for n, (a, b) in enumerate(zip(start, sizes())) if b > a]
        flags = {key: 1 in [_count(x) for x in _items(heard.get(key))] for key in ('crowded', 'puzzled')}
        return {'line': batch[0], 'heard': heard, 'gained': gained, 'turned': flags['crowded'] and not flags['puzzled']}

    def settle(self, number: int, label: str, apart: str, result: dict[str, Any]) -> None:
        # bookkeeping after a reply: reveals counted, no-preference properties closed, a rephrase queued after a turn-back
        h = self.host
        for n in result['gained']:
            self.tally.setdefault(n, [0, 0])[1] += 1
        if SKIP_PUZZLED:
            for entry in _dicts(result['heard'].get('free')):
                said = _words(str(entry.get('property') or '').split(':')[0])
                if said:
                    self.closed.setdefault(h.product_no(entry.get('item'), number), set()).add(said)
        if result['gained'] or not result['turned'] or apart or not SET_APART:
            return
        waiting = [n for n in self.lanes if self.tally[n][0] == 0 and self.open_labels(n)]
        if APART_AFTER_ROUND and waiting:
            h.trace.append({'kind': 'turned_back', 'label': label, 'apart': '', 'why': 'a product is still unasked'})
            return
        alike = self.look_alike(number, label, result['line'][1])
        if alike and self.wording(number, label, alike) not in self.sent_texts:
            self.queue.append((number, label, alike))
        h.trace.append({'kind': 'turned_back', 'label': label, 'apart': alike})

    def report(self, lines: list[tuple[int, str, str]], told: list[str], searched: str) -> str:
        h = self.host
        out = [f'[interview, {len(lines)} step(s)] questions Marlo sent and the replies:'] + [f'Q: {q}\nA: {a}' for _, q, a in lines]
        if told:
            out.append('New requirements: ' + '; '.join(told) + ' -- every pick must meet them; rerun searches with these values when that narrows things.')
        if self.unread:
            out.append('Marlo could not read the replies to: ' + ' | '.join((f'"{q}"' for q in self.unread)) + '. Read those replies above yourself: a property and value they state is a firm requirement for that product.')
        elif not told:
            out.append('New requirements: none.')
        borrowed = [need for _, need in h.refs if any((need in t for t in told))]
        if borrowed:
            out.append('Value held elsewhere: ' + ' | '.join(borrowed) + ' -- locate the listing described, read its page in full and keep following every listing named onward until one page spells the value out; a checkout before the whole chain is read will not count.')
        if searched:
            out.append(searched)
        if h.queued:
            out.append('[notes seen meanwhile] ' + ' | '.join(h.queued))
            h.queued = []
        return '\n'.join(out)

    def run(self) -> str:
        h = self.host
        h.asked = True
        if h.msgs_free() < 1:
            return ''
        if not self.ready() and h.seconds_left() > ASK_BY_SECONDS + 30:
            h.widen()
        budget = h.msgs_free()
        self.plan_lanes(budget)
        depth = LEAD_DEPTH and budget >= 3
        lines: list[tuple[int, str, str]] = []
        told: list[str] = []
        while h.msgs_free() > 0 and not h.finished and h.left() > 6 and h.room_for(MODEL_SECONDS + 4):
            number, label, apart = self.next_question(depth)
            if not label and not (NO_WASTE and self.refill()):
                break
            if not label:
                continue
            result = self.exchange(number, label, apart, told)
            if result is None:
                break
            if result.get('skip'):
                continue
            lines.append(result['line'])
            self.settle(number, label, apart, result)
        found = [n for n, (_, hits) in self.tally.items() if hits > 0]
        searched = h.value_search(found) if found else ''
        if told:
            h.version += 1
        return self.report(lines, told, searched)


class Marlo:

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
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.t_start = time.monotonic()
        # state added on top of the inherited buyer, each slot built fresh from its starting type: clock and finish
        # bookkeeping, the candidate board and listing views, price ceilings, model blocks, the interview's records
        fresh = {bool: ('wrapping', 'plain_think', 'silent'), int: ('late_skips', 'end_refusals', 'checkout_refusals', 'end_stops', 'req_stamp', 'code_hits', 'verdict_no'),
                 dict: ('ratings', 'views_seen', 'ceilings', 'question_log', 'model_checkout'), set: ('view_names', 'sweeps_done', 'blocked_models', 'plain_needs'),
                 list: ('recent_searches', 'ceiling_ways', 'price_unit'), type(None): ('review_state', 'ceiling_arg', 'repair_cap')}
        for start, names in fresh.items():
            for name in names:
                setattr(self, name, start())
        self.asker = Ladder(self)

    def seconds_left(self) -> float:
        return HARD_END - (time.monotonic() - self.t_start)

    def env_budget(self) -> tuple[int, int] | None:
        """Per-call timeout and retry count for a store call, shrinking with the clock; None once it is too late."""
        room = self.seconds_left()
        if room < 2:
            return None
        tries = 1 + (room > 50) + (room > 150)
        return (max(1, min(45, int(room // tries) - 1)), tries)

    def room_for(self, seconds: float) -> bool:
        """Is there clock for something taking this long? Exploring keeps the finishing reserve; finishing keeps enough
        for the last store call."""
        return self.seconds_left() - (LAST_CALL_SECONDS if self.wrapping else FINISH_SECONDS) >= seconds

    def left(self) -> int:
        return self.max_steps - self.step

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

    def tail(self) -> int:
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def closing_judges(self, limit: float) -> list[str]:
        # the last cart review may still go to the heavy model while one heavy call fits under the episode ceiling
        spare = SPEND_CEILING - self.cost
        if spare < 0.002:
            return []
        return [HEAVY, LIGHT] if self.cost <= limit and spare >= HEAVY_ROOM else [LIGHT]

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None, effort: str | None=None, critical: bool | int=False) -> dict[str, Any] | None:
        order = [m for m in models if m not in self.blocked_models]
        size = sum((len(_dump(m.get('content'))) for m in messages))
        if models and (not order) and SPARE not in self.blocked_models:
            order = [SPARE]
        for model in order:
            if self.calls >= ALL_CALLS:
                return None
            # spend ceilings: ordinary calls stop at SPEND_CEILING, the ones an order depends on (process steps, notes,
            # replies) run on to CRITICAL_CEILING, and the heavy model starts only when its estimated cost still fits
            ceiling = float('inf') if critical == 2 else CRITICAL_CEILING if critical else SPEND_CEILING
            if self.cost >= ceiling or (model == HEAVY and self.cost + self.heavy_estimate(size, max_tokens) > SPEND_CEILING):
                continue
            self.calls += 1
            tries = 0
            while True:
                if not self.room_for(MODEL_SECONDS):
                    self.late_skips += 1
                    if self.late_skips <= 3:
                        self.trace.append({'kind': 'clock_skip', 'left': round(self.seconds_left(), 1), 'closing': self.wrapping})
                    return None
                _llm.timeout = max(4, min(50, int(self.seconds_left() - (LAST_CALL_SECONDS if self.wrapping else FINISH_SECONDS)) - 1))
                bounded = effort if effort and model != HEAVY and (not self.plain_think) else None
                message, status = self.call_model(model, messages, max_tokens, tools, bounded)
                if message is not None:
                    return message
                if status in (400, 422) and bounded:
                    # turned down: when the same call goes through without the thinking setting, drop it for the episode
                    message, status = self.call_model(model, messages, max_tokens, tools, None)
                    if message is not None:
                        self.plain_think = True
                        self.trace.append({'kind': 'effort_refused', 'model': model})
                        return message
                if status == 403:
                    self.blocked_models.add(model)
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                pause = min(WAITS[tries] * (0.8 + 0.4 * random.random()), self.seconds_left() - FINISH_SECONDS - MODEL_SECONDS)
                if pause < 1:
                    break
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        if order and all((m in self.blocked_models for m in order)) and SPARE not in order and (SPARE not in self.blocked_models) and (not self.no_credit):
            return self.think(messages, [SPARE], max_tokens, tools, effort, critical)
        return None

    @staticmethod
    def heavy_estimate(size: int, max_tokens: int) -> float:
        # what one heavy-model call is expected to cost: its prompt (about 2.5 characters a token on these JSON-heavy
        # prompts) at $2/M plus a fifth of its token budget at $10/M (reasoning included)
        return (size / 2.5 + max_tokens) * 2e-06

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, effort: str | None=None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get('content')
                if isinstance(content, str) and content:
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if effort and model != HEAVY:
            # thinking is billed and counts against max_tokens: bounded thinking keeps the answer inside the budget
            body['reasoning'] = {'effort': effort}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        result = _llm.post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.cost += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            cut = data['choices'][0].get('finish_reason') == 'length'
            if (message.get('content') or message.get('tool_calls')) and not (cut and not tools and _json_obj(message.get('content')) is None):
                return (message, 200)
            self.silent = True
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0, 'cut': cut})
            return (None, 0)
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = _count(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800, effort: str | None=None, critical: bool | int=False) -> dict[str, Any] | None:
        for model in models:
            budget = max_tokens
            for attempt in range(2):
                self.silent = False
                message = self.think([{'role': 'user', 'content': prompt}], [model], budget, None, effort, critical)
                got = _json_obj((message or {}).get('content'))
                if got is not None:
                    return got
                if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                    return None
                # only a light-model reply that said nothing (its thinking used the room) earns the second try
                if attempt or model != LIGHT or not self.silent or not self.room_for(2 * MODEL_SECONDS):
                    break
                budget = min(2 * max_tokens, 12000)
                self.trace.append({'kind': 'silent_retry', 'tokens': budget})
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        for _ in range(4):
            budget = self.env_budget()
            if budget is None:
                self.trace.append({'kind': 'clock_stop', 'tool': name})
                return None
            _env.timeout, _env.max_retries = budget
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data['observation'] = _structured(data.get('observation'))
                self.mark_view(name, seen)
                if name in self.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            if status == 409:
                low = body.lower()
                if 'finish' in low or 'quarantin' in low or 'finaliz' in low:
                    self.finished = True
                    return None
                found = re.search('expected (\\d+)', body)
                if found:
                    self.step = int(found.group(1)) - 1
                    payload['step'] = self.step + 1
                    continue
                if 'reuse' in low or 'different action' in low:
                    self.step += 1
                    payload['step'] = self.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        return None

    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.request
        if self.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + _dump(_clip(self.view.get('opening'), 1500))
        brief = self.ask_json(prompt, [LIGHT], 6000, None) or {}
        if not self.no_credit and (not _items(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500, None) or brief
        elif not self.no_credit and self.shared_shelf(brief):
            again = self.ask_json(prompt, [HEAVY], 2500) or {}
            if _items(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                brief = again
        self.brief = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']):
            ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _dump(self.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
            if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and (fixed['buy_args'] != fixed['refuse_args']):
                brief['buy_args'], brief['refuse_args'] = (fixed['buy_args'], fixed['refuse_args'])
        for entry in _items(brief.get('items')):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get('quantity'))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get('musts')) if m]
            self.items.append(Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), count, musts))
        merged: list[Item] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.items, 1):
            twin = next((m for m in merged if item.shelf.strip() and _words(m.shelf) == _words(item.shelf) and (not any((_words(a.split(':')[0]) == _words(b.split(':')[0]) and _words(a.partition(':')[2]) != _words(b.partition(':')[2]) for a in m.musts for b in item.musts)))), None)
            if twin is None:
                merged.append(item)
                became[old] = len(merged)
            else:
                became[old] = merged.index(twin) + 1
                twin.musts += [m for m in item.musts if m not in twin.musts]
                self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
        self.items = merged
        if not self.items:
            self.items = [Item('the requested product', '', 1, [])]
        for pair in _items(brief.get('fits')):
            if isinstance(pair, dict):
                part, main = (became.get(_count(pair.get('item')) or 0), became.get(_count(pair.get('with')) or 0))
                if part and main and (part != main) and (1 <= part <= len(self.items)) and (1 <= main <= len(self.items)) and ((part, main) not in self.bound):
                    self.bound.append((part, main))
        for lookup in _items(brief.get('lookups')):
            text = str(lookup).strip()
            if not any((text in item.musts for item in self.items)):
                key = _words(text.split(':')[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(':')[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.unmix()
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            self.budget = budget
        limit = _count(brief.get('messages'))
        if limit is not None and limit >= 0:
            self.msg_limit = limit
        elif self.roles.get('ask'):
            self.msg_limit = 3
        steps = _count(brief.get('steps'))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in _items(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _count(entry.get('messages')) or 0)})
        if self.roles.get('ask'):
            talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    @staticmethod
    def shared_shelf(brief: dict[str, Any]) -> bool:
        shelves = [_words(str(e.get('shelf') or '')) for e in _items(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def unmix(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = UNMIX + '\n\nREQUEST:\n' + self.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.ask_json(prompt, [HEAVY, LIGHT], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get('describe_other'))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})

    def described(self, number: int, need: str) -> bool:
        if need in [m for _, m in self.refs] or need in [str(x) for x in _items(self.brief.get('lookups'))]:
            return False
        prop, _, value = need.partition(':')
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
            return False
        item = self.items[number - 1]
        if any(({w for w in _stem(other.name) if len(w) >= 4} & _stem(value) for other in self.items if other is not item)):
            return False
        cat_key = self.category_key()
        keys: set[str] = set()
        read = 0
        for record in self.full.values():
            if cat_key and (not self.owns(item, str(record.get(cat_key) or ''))):
                continue
            block = _fact_block(record) or {}
            if block:
                read += 1
                keys |= {_words(k) for k in block}
        return read >= 3 and (not any((_stem(k) & _stem(prop) for k in keys)))

    def ledger(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            live = [m for m in item.musts + item.told if m not in item.dropped]
            told = [m for m in live if self.described(number, m)]
            needs = [m + (f' (the page it points at gives: {self.ref_values[m]})' if m in self.ref_values else '') for m in live if m not in told]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            if item.shelf:
                head += f' (store label: {item.shelf})'
            head += f', quantity {item.qty}'
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if told:
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
            if item.free:
                text += '; shopper does not mind: ' + ', '.join(item.free)
            parts.append(text)
        brief = self.brief
        if self.budget is not None:
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        pick = brief.get('pick')
        if pick == 'cheapest':
            parts.append('pick: the cheapest listing that fits (lowest price)')
        elif pick in ('most', 'least'):
            parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        else:
            parts.append('pick: any listing that fits everything')
        if brief.get('in_stock_only', True) is not False:
            parts.append('in stock only')
        for wish in _items(brief.get('wishes')):
            parts.append(f'wish (tie-breaker only): {wish}')
        for rule in _items(brief.get('rules')):
            parts.append(f'rule from the request: {rule}')
        if brief.get('quiet_decline') is True:
            parts.append('if nothing fits: end without naming or suggesting any listing')
        for entry in self.process:
            when = {'choose': 'before choosing', 'order': 'at the order (Marlo does it when you check out)', 'end': 'at an ending without a purchase (Marlo does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.page_tools() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        left = self.left()
        text = f'[Marlo] step {self.step} of {self.max_steps} used, {left} left.'
        if left <= self.tail() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif self.buys == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2):
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if self.seconds_left() < FINISH_SECONDS + 3 * MODEL_SECONDS:
            text += ' Clock: little of this episode remains, so settle the cart and send the checkout within your next replies.'
        if self.refusal_seen():
            text += ' Earlier the store turned down ' + ('a checkout' if self.checkout_refusals else 'the no-purchase ending') + '; that tells you an order is wanted, so the episode closes with a checkout.'
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        plan = self.fit_plan() if self.bound and self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        board = self.board_line() if self.notes or self.refusal_seen() else ''
        if board:
            text += '\n' + board
        text += '\n[ledger]\n' + self.ledger()
        return text

    def store_keys(self) -> set[str]:
        # every attribute key the opened listings of this store carry, in plain words
        return {_words(k) for record in self.full.values() for k in (_fact_block(record) or {})}

    def product_keys(self, number: int) -> list[str]:
        """Interview service: the attribute keys seen on opened listings of one requested product, most carried
        first, without the do-not-ask keys."""
        table = self.key_table()
        rows = [(carried, key) for label, key, carried, _ in table if self.owner_of(label) == number]
        if not rows and len(self.items) == 1:
            rows = [(carried, key) for _, key, carried, _ in table]
        out: list[str] = []
        for _, key in sorted(rows, key=lambda r: -r[0]):
            if key not in out and (not re.search(NOT_PROPERTY, key)):
                out.append(key)
        return out

    def shelf_word(self, number: int) -> str:
        # interview service: the product by the store's own label (the one most of its opened listings carry, else the
        # label the request reading copied), unless another requested product goes by the same words
        item = self.items[number - 1]
        key = self.category_key()
        seen = [str(r.get(key) or '').strip() for r in self.full.values()] if key else []
        readable = lambda s: 0 < len(s) <= 60 and re.fullmatch("[A-Za-z][A-Za-z '&-]*", s) is not None
        mine = [s for s in seen if readable(s) and self.owner_of(s) == number and _roots(s) & (_roots(item.name) | _roots(item.shelf or ''))]
        shelf = (item.shelf or '').strip()
        fallback = shelf if readable(shelf) and _roots(shelf) & _roots(item.name) else ''
        best = max(sorted(set(mine)), key=mine.count) if mine else fallback
        taken = {_words(o.shelf or '') for o in self.items if o is not item} | {_words(o.name) for o in self.items if o is not item}
        if not best or _words(best) in taken:
            return item.name
        return ' '.join((w.lower() if w[:1].isupper() and w[1:] == w[1:].lower() else w for w in best.split()))

    def product_no(self, value: Any, fallback: int=1) -> int:
        # a product number named by a model reply, kept inside 1..len(items)
        number = _count(value)
        return min(len(self.items), number if number and number > 0 else fallback)

    def opened_for(self, number: int) -> int:
        """Interview service: opened listings (with attributes) of one requested product."""
        return self.records_of_item(number)

    def log_question(self, number: int, label: str, text: str, reply: str, heard: dict[str, Any]) -> dict[str, Any]:
        """Interview service: per-product question history (what was asked, the reply, how the reading call read it)."""
        mine = [e for e in _items(heard.get('needs')) + _items(heard.get('free')) if isinstance(e, dict) and min(max(1, _count(e.get('item')) or 1), len(self.items)) == number]
        entry = {'label': label, 'text': text, 'reply': str(reply)[:300], 'told': [str(e['need']) for e in mine if 'need' in e], 'free': any(('property' in e for e in mine))}
        entry.update({kind: bool(_items(heard.get(kind))) for kind in ('crowded', 'puzzled')})
        self.question_log.setdefault(number, []).append(entry)
        return entry

    def product_of(self, ident: str) -> int:
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
        return self.owner_of(label) if label else 0

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def rate(self, ident: str, fit: bool | None, why: str, item: int=0) -> None:
        """Record a model verdict on one listing under the requirements now in force (the candidate board)."""
        if ident not in self.rows:
            return
        slot = self.ratings.setdefault(ident, {'item': item or self.product_for(ident), 'fit': None, 'at': -1, 'why': ''})
        if item:
            slot['item'] = item
        self.verdict_no += 1
        slot['fit'], slot['at'], slot['why'], slot['n'] = (fit, self.req_stamp, why[:160], self.verdict_no)
        slot['whole'] = ident in self.full

    def ranked_for(self, number: int, loose: int=0) -> list[str]:
        """Candidates for one requested product, best first. Strict (0): judged to fit under the requirements in force.
        Loose (1): also judged under older requirements, then opened but never judged; (2) also rows never opened.
        Unavailable listings, listings judged unfit now and lines that alone break the budget never stand."""
        pool = set(self.ratings)
        if loose:
            pool |= {i for i in (self.rows if loose > 1 else self.full) if self.product_for(i) == number}
        cheapest = self.brief.get('pick') == 'cheapest'
        place = dict(zip(self.seen_order, range(len(self.seen_order))))
        kinds = {self.product_for(i) for i in self.rows}
        ranked = []
        for ident in pool:
            slot = self.ratings.get(ident) or {'item': self.product_for(ident), 'fit': None, 'at': -1}
            if slot['item'] != number or self.sold_out(ident) or not self.buyable(ident) or self.record_fails(ident, number, kinds):
                continue
            fresh = slot['at'] == self.req_stamp
            if slot['fit'] is False and (fresh or not loose):
                continue
            if not loose and (slot['fit'] is not True or not fresh):
                continue
            price = self.price_of(ident)
            if price is not None and isinstance(self.budget, (int, float)):
                spend = price if self.brief.get('budget_scope') == 'each' else price * self.items[number - 1].qty
                if self.budget + 1e-09 < spend <= 10 * self.budget:
                    continue
            rank = 0 if slot['fit'] is True and fresh else 1 if slot['fit'] is True else 2 if ident in self.full else 3
            order = slot.get('n', 0) if rank < 2 else len(self.ratings) + place.get(ident, len(place))
            ranked.append((rank, price if cheapest and price is not None else 0.0, order, ident))
        return [entry[-1] for entry in sorted(ranked)]

    def basket_on_file(self, loose: int=0) -> dict[str, int]:
        """One line per requested product: the cart's own line while it can still be bought; when loose, a line the
        buyer had carted before (its own earlier choice); else the best standing candidate. Empty when some product
        has none."""
        cart = [i for i in self.cart_ids() if not self.sold_out(i) and (self.held_units(i) or 0) > 0]
        earlier = [i for i in self.built if i not in cart and not self.sold_out(i)] if loose else []
        if not self.category_key() and len(self.items) > 1:
            return {i: max(1, self.held_units(i) or 1) for i in cart}
        plan: dict[str, int] = {}
        for number, item in enumerate(self.items, 1):
            held_here = [i for i in cart if self.product_for(i) == number] or [i for i in earlier if self.product_for(i) == number]
            line = next((i for i in held_here if not self.record_fails(i, number)), None)
            better = self.judged_instead(line) if line else ''
            pick = better or line or next(iter(self.ranked_for(number, loose)), None)
            if not pick or pick in plan:
                return {}
            plan[pick] = item.qty
        prices = [(self.price_of(i), n) for i, n in plan.items()]
        if isinstance(self.budget, (int, float)) and all((p is not None for p, _ in prices)):
            each = self.brief.get('budget_scope') == 'each'
            spent = max((p for p, _ in prices)) if each else sum((p * n for p, n in prices))
            if self.budget + 1e-09 < spent <= 10 * self.budget * max(1, len(plan)):
                return {}
        return plan

    def record_fails(self, ident: str, number: int, kinds: set[int] | None=None) -> str:
        # what the listing's own record holds against buying it for product `number` now ('' when nothing does): stock,
        # kind and budget always count; on attribute values a review that read the whole record under the requirements in
        # force and found the listing fitting outranks this check.
        if self.sold_out(ident) or not self.buyable(ident):
            return 'it cannot be bought now'
        kinds = kinds if kinds is not None else {self.product_for(i) for i in self.rows}
        if self.category_key() and number in kinds and self.product_for(ident) != number:
            return 'its category label names another kind of product'
        price = self.price_of(ident)
        if price is not None and isinstance(self.budget, (int, float)) and 0 < number <= len(self.items):
            spend = price if self.brief.get('budget_scope') == 'each' else price * self.items[number - 1].qty
            # past ten times the budget a price may just be in other units, unless the store shows prices within it
            if self.budget + 1e-09 < spend and (spend <= 10 * self.budget or self.prices_match_budget()):
                return f'it costs {spend:g}, over the budget of {self.budget:g}'
        verdict = self.ratings.get(ident) or {}
        if verdict.get('fit') is True and verdict.get('at') == self.req_stamp and verdict.get('whole'):
            return ''
        block = _fact_block(self.full.get(ident) or {}) or {}
        for owner, need in self.needs_now():
            prop, _, value = need.partition(':')
            wanted = _words(value).split()
            if owner != number or len(wanted) != 1 or not wanted[0].isalpha() or len(wanted[0]) < 3:
                continue
            said = [str(v).lower() for k, v in block.items() if _stem(k) == _stem(prop) and isinstance(v, (str, int, float))]
            core = next(iter(_roots(wanted[0])), wanted[0])
            if not said or any((core in text for text in said)):
                continue
            # only where the store itself writes that value under this key on some listing: a word it never uses for the
            # key may be the shopper's synonym for one of its own values
            used = (str(v).lower() for record in self.full.values() for k, v in (_fact_block(record) or {}).items() if _stem(k) == _stem(prop))
            if any((core in text for text in used)):
                return f'its {prop.strip()} reads {said[0][:40]!r}, not {wanted[0]!r}'
        return ''

    def prices_match_budget(self) -> bool:
        # some listing seen costs no more than the budget: the request's unit and the store's agree
        mark = (len(self.rows), self.budget)
        if self.price_unit[:1] != [mark]:
            self.price_unit[:] = [mark, any((0 < (self.price_of(i) or 0) <= self.budget + 1e-09 for i in self.rows))]
        return bool(self.price_unit[1])

    def judged_instead(self, ident: str) -> str:
        # for a line a review judged unfit under the requirements now in force: the best listing judged to fit them
        number, verdict = self.product_for(ident), self.ratings.get(ident) or {}
        unfit = bool(number) and verdict.get('fit') is False and verdict.get('at') == self.req_stamp
        return next((i for i in self.ranked_for(number) if i != ident), '') if unfit else ''

    def product_for(self, ident: str) -> int:
        # the requested product a listing stands for: its category label decides; a store without labels and a single
        # requested product make every listing seen a candidate for that product
        return self.product_of(ident) or int(len(self.items) == 1 and ident in self.rows and not self.category_key())

    def needs_repair(self) -> bool:
        return any((self.sold_out(i) or self.judged_instead(i) or self.record_fails(i, self.product_for(i)) for i in self.cart_ids()))

    def finish_reserve(self) -> int:
        # steps the finishing pipeline needs from here: the edits that set the best verified basket on file when the cart
        # is not that basket, then the checkout (or the no-purchase ending) and a message a due process step may take
        base = self.tail()
        cart = self.cart_ids()
        if self.roles.get('buy') and not self.buys and (cart or self.basket_on_file(1)):
            # a first checkout is often turned down (a sell-out): keep the steps to take the line out, add another, order
            base += REFUSAL_ROOM
        if not self.roles.get('buy') or (cart and not self.needs_repair()):
            return base
        basket = self.basket_on_file(1)
        extra = [i for i in cart if i not in basket]
        edits = self.cart_calls(basket, extra) if basket and (extra or any((i not in cart for i in basket))) else []
        return base + len(edits or [])

    def repairing(self) -> bool:
        # after a refused checkout: the cart still holds a line that cannot stand, or misses a requested product, and the
        # spend since the last refusal (or the SWAP after it) is inside the repair window
        if self.repair_cap is None or self.finished or self.cost >= max(self.repair_cap, CRITICAL_CEILING):
            return False
        cart = self.cart_ids()
        mapped = bool(self.category_key()) or len(self.items) == 1
        missing = mapped and any((not any((self.product_for(i) == n for i in cart if not self.sold_out(i))) for n in range(1, len(self.items) + 1)))
        if not (self.needs_repair() or missing or not cart):
            return False
        # nothing on file passes its record: searching on is the only road to an order
        return self.cost < (self.repair_cap if self.basket_on_file(1) else CRITICAL_CEILING)

    def repair_checkout(self, why: str) -> str:
        # replace the stale lines from what the reviews judged (then any opened listing, then any row of the
        # product's own kind within budget) and check out; '' when no such basket exists
        for loose in ((1, 2) if self.end_refusals else (1,)):
            done = self.checkout_on_file(why, loose)
            if done or self.finished:
                return done
        return ''

    def board_line(self) -> str:
        parts = []
        for number, item in enumerate(self.items, 1):
            ids = self.ranked_for(number)[:3]
            if ids:
                parts.append(f'item {number} ({item.name}): ' + ', '.join((f'{i} ({self.price_of(i):g})' if self.price_of(i) is not None else i for i in ids)))
        return '[reviews so far found these fitting, for the current requirements] ' + ' | '.join(parts) if parts else ''

    def repick_line(self) -> str:
        """After a change, what the cart should become, from candidates already judged (no step, no model call)."""
        cart = self.cart_ids()
        lines = []
        total = 0.0
        known = True
        for number, item in enumerate(self.items, 1):
            held_here = [i for i in cart if self.product_of(i) == number]
            live = [i for i in held_here if not self.sold_out(i)]
            slot = self.ratings.get(live[0]) if live else None
            unfit = bool(slot) and slot['fit'] is False and slot['at'] == self.req_stamp
            if live and (not unfit):
                price = self.price_of(live[0])
                known = known and price is not None
                total += (price or 0.0) * item.qty
                continue
            ready = [i for i in self.ranked_for(number) if i not in held_here]
            if ready:
                lines.append(f"item {number} ({item.name}): {'replace ' + held_here[0] if held_here else 'add'} -> {ready[0]}" + (f" (other judged fits: {', '.join(ready[1:3])})" if ready[1:] else ''))
        if known and isinstance(self.budget, (int, float)) and self.brief.get('budget_scope') != 'each' and (total > self.budget + 1e-09):
            cheaper = [f'item {n}: ' + ', '.join(self.ranked_for(n)[:3]) for n in range(1, len(self.items) + 1) if self.ranked_for(n)]
            lines.append(f'the cart costs {total:g}, over the budget of {self.budget:g}' + ('; judged fits, cheapest first: ' + ' | '.join(cheaper) if cheaper and self.brief.get('pick') == 'cheapest' else ''))
        return '[after this change, from listings already reviewed] ' + '; '.join(lines) if lines else ''

    def checkout_args(self, faults: list[str]) -> dict[str, Any]:
        """The checkout call's arguments: the fixed ones that choose checking out, plus a short plain note for any
        other required text argument."""
        buy = self.roles.get('buy')
        schema = self.schemas.get(buy or '') or {}
        props = schema.get('properties') or {}
        args = dict(self.brief.get('buy_args') or {})
        avoid = self.brief.get('refuse_args') if isinstance(self.brief.get('refuse_args'), dict) else {}
        for key, value in self.model_checkout.items():
            if key not in args and isinstance((props.get(key) or {}).get('enum'), list) and key in avoid and (avoid[key] != value):
                args[key] = value
        for key in schema.get('required') or []:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            if key not in args and 'string' in _kinds(spec) and ('enum' not in spec):
                args[key] = 'Placing the order for the chosen cart.'[:spec.get('maxLength', 200) if isinstance(spec.get('maxLength'), int) else 200]
        return _shape(args, schema, 'args', faults)

    def refusal_seen(self) -> bool:
        return bool(self.end_refusals or self.checkout_refusals)

    def count_refusal(self, name: str, args: dict[str, Any], reply: dict[str, Any]) -> None:
        """Book a finish the store turned down. Only a turned-down checkout clears the review marks (a new cart must be
        judged again); a turned-down ending keeps them, so the same review does not invite the same ending twice."""
        if self.finished or reply.get('finished'):
            return
        kind = 'refused_checkout' if self.is_buy(name, args) else 'refused_ending' if self.is_refuse(name, args) else ''
        if kind == 'refused_checkout':
            self.checkout_refusals += 1
            self.checked, self.cheaper_marks, self.sweeps_done = (set(), set(), set())
            self.repair_cap = self.cost + REPAIR_ROOM
        elif kind:
            self.end_refusals += 1
        if kind:
            self.trace.append({'kind': kind, 'n': self.checkout_refusals + self.end_refusals})

    def checkout_on_file(self, why: str, loose: int=0) -> str:
        """Check out the best basket on file instead of ending without a purchase, setting the cart first if needed."""
        basket = self.basket_on_file(loose)
        cart_tool, order_tool = (self.roles.get('cart'), self.roles.get('buy'))
        if not basket or not order_tool or self.finished or self.left() < 1:
            return ''
        holding = self.cart_ids()
        extra = [i for i in holding if i not in basket]
        changed = bool(extra) or any((i not in holding or (self.held_units(i) or 0) != units for i, units in basket.items()))
        edits = self.cart_calls(basket, extra) if changed else []
        if edits is None or (edits and (not cart_tool)) or self.left() < len(edits) + 1:
            return ''
        log: list[str] = []
        for edit in edits:
            problems: list[str] = []
            edit = _shape(edit, self.schemas.get(cart_tool), 'args', problems)
            answer = None if problems else self.act(cart_tool, edit)
            if answer is None or self.finished:
                return ''
            self.note_cart(cart_tool, edit)
            self.observe(cart_tool, answer.get('observation'))
            log.append(_dump(edit))
        problems = []
        order = self.checkout_args(problems)
        if problems:
            return ''
        self.read_all_views(list(basket), 2)
        if self.left() >= 2 and self.room_for(MODEL_SECONDS):
            go, said = self.gate('order')
            if not go:
                return 'Not run yet: ' + said
        answer = self.act(order_tool, order)
        self.trace.append({'kind': 'basket_checkout', 'why': why[:120], 'lines': basket, 'edits': log, 'answer': _clip(answer, 200)})
        if answer is None:
            return ''
        self.buys += 1
        self.refused_at = self.refused_at if answer.get('finished') else self.step
        self.count_refusal(order_tool, order, answer)
        self.observe(order_tool, answer.get('observation'))
        lines = ', '.join((f'{i} x{units}' for i, units in basket.items()))
        return f'Marlo placed the order itself ({why}); cart lines {lines}' + (f" via {'; '.join(log)}" if log else '') + '. The store answered: ' + self.show(answer.get('observation'))

    def listing_views(self) -> list[str]:
        tools = [t for t in (self.roles.get('open'), self.roles.get('offer')) if t and self.single_target(t)]
        return tools + sorted((t for t in self.view_names if t not in tools and self.single_target(t)))

    def mark_view(self, name: str, observation: Any) -> None:
        """Which single-listing view has shown which listing: the coverage ledger behind read_all_views()."""
        skip = {self.roles.get(r) for r in ('search', 'cart', 'cart_view', 'buy', 'refuse', 'ask', 'compare')}
        if name in skip or not self.single_target(name):
            return
        found = self.records_in(observation)
        if len(found) == 1:
            self.view_names.add(name)
            self.views_seen.setdefault(found[0][0], set()).add(name)

    def read_all_views(self, ids: list[str], keep: int) -> list[str]:
        """Open each listing about to be bought in every single-listing view the store has that has not shown it yet,
        so seller notes, pointers and long descriptions are read before the order, never guessed."""
        todo = [(ident, view) for ident in ids if ident in self.rows and (not self.sold_out(ident)) for view in self.listing_views() if view not in self.views_seen.get(ident, set())]
        pages: list[str] = []
        for ident, view in todo:
            if self.finished or self.left() <= keep + 1 or not self.room_for(4):
                break
            problems: list[str] = []
            call = _shape({self.single_target(view): ident}, self.schemas.get(view), 'args', problems)
            answer = None if problems else self.act(view, call)
            if answer is None:
                if problems:
                    continue
                break
            body = answer.get('observation')
            self.views_seen.setdefault(ident, set()).add(view)
            self.observe(view, body)
            self.note_view(view, body)
            self.offer_seen[ident] = self.step if view == self.offer_tool() else self.offer_seen.get(ident, -99)
            pages.append(f'{ident} in {view}: ' + self.show(body)[:PAGE_CHARS + 600])
        if pages:
            self.trace.append({'kind': 'cover', 'n': len(pages)})
        return pages

    def not_an_id(self, value: Any) -> bool:
        """True for a value passed as a listing id that cannot be one: the ids seen so far share a prefix or a length
        that it lacks (a model or part code copied from listing text)."""
        if not isinstance(value, str) or not _id_like(value) or value.strip() in self.rows or len(self.rows) < 3:
            return False
        ids = list(self.rows)[:120]
        text = value.strip()
        lead = _shared_start(ids)
        sizes = {len(i) for i in ids}
        return (len(lead) >= 2 and (not text.startswith(lead))) or (len(sizes) == 1 and len(text) not in sizes)

    def code_holders(self, code: str) -> list[str]:
        pattern = re.compile('(?<![A-Za-z0-9])' + re.escape(code.strip()) + '(?![A-Za-z0-9])', re.I)
        hits = [i for i in self.seen_order if pattern.search(_dump(self.rows.get(i) or {}))]
        own = [i for i in hits if code.strip().lower() in _marks_by_position(_dump(self.rows.get(i) or {}))['own']]
        return own if len(own) == 1 else hits

    def stop_codes(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Stop a call that passes a code from listing text as a listing id; map it when exactly one seen listing
        carries that code as its own mark."""
        tools = ({self.roles.get(r) for r in ('open', 'offer', 'compare', 'cart')} | self.view_names) - {None}
        if name not in tools or len(self.rows) < 3:
            return (args, '')
        out = dict(args)
        for key, value in args.items():
            values = value if isinstance(value, list) else [value]
            if not any((self.not_an_id(v) for v in values)):
                continue
            fixed = []
            for v in values:
                if not self.not_an_id(v):
                    fixed.append(v)
                    continue
                self.code_hits += 1
                carriers = self.code_holders(v)
                if len(carriers) == 1:
                    fixed.append(carriers[0])
                    self.trace.append({'kind': 'code_mapped', 'code': str(v)[:40], 'id': carriers[0]})
                    continue
                example = next(iter(self.rows))
                if carriers:
                    return (args, f"{v!r} is printed inside the text of {', '.join(carriers[:6])}; a listing is addressed by an identifier shaped like {example}, so pass the one you mean.")
                return (args, f'{v!r} is not shaped like a listing id here (they look like {example}); if it is a code from some listing text, search for it with {self.roles.get("search") or "the search tool"} first.')
            out[key] = fixed if isinstance(value, list) else fixed[0]
        return (out, '')

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_like(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any((isinstance(v, dict) for v in node.values()))
        if not (in_list or depth == 0 or nested):
            return None
        found = _learn_id([node])
        if found and ('id' in str(found).lower() or node[found].strip() in self.rows):
            return node[found].strip()
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_key and all((_id_like(r.get(self.id_key), True) for r in rows)):
            return True
        found = _learn_id(rows)
        if found:
            self.id_key = found
            return True
        return False

    def absorb(self, observation: Any) -> None:
        observation = _structured(observation)

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if ident:
                    if ident not in self.rows:
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
                    old = self.rows[ident]
                    if any((old.get(k) is True and v is False for k, v in node.items())):
                        self.unavailable.add(ident)
                    elif any((old.get(k) is False and v is True for k, v in node.items())):
                        self.unavailable.discard(ident)
                    self.rows[ident].update(node)
                    if _fact_block(node):
                        self.full[ident] = dict(self.rows[ident])
                    elif ident in self.full:
                        self.full[ident].update(node)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                rows = self.learn_rows(node)
                for value in node:
                    walk(value, depth + 1, rows)
        walk(observation, 0)

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
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and (not re.search('\\d{3}', value)):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        known = self.labels()
        best = None
        for key, values in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            listed = sum((1 for v in values if v.strip().lower() in known)) / len(values)
            mean = sum((len(v) for v in values)) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            block = _fact_block(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = _words(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][ident] = text
                if text not in slot['values']:
                    slot['values'].append(text)
        wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any((_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                values = list(slot['per'].values())
                if not values:
                    continue
                mean = sum((len(v) for v in values)) / len(values)
                if mean > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values):
                    continue
                recur = sum((len(set(re.findall('[a-z]{3,}', v)) & set(re.findall('[a-z]{3,}', ' '.join((o for o in values if o is not v))))) > 0 for v in values))
                if len(values) >= 4 and len(set(values)) == len(values) and (mean > 12) and (recur * 2 < len(values)):
                    continue
                twin = next((spec for other, spec in kept if self.twins(other, spec, name, slot)), None)
                if twin is not None:
                    for ident, text in slot['per'].items():
                        twin['per'].setdefault(ident, text)
                        if text not in twin['values']:
                            twin['values'].append(text)
                    twin['n'] = len(twin['per'])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for name, slot in kept[:14]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2))
        both = [i for i in spec['per'] if i in slot['per'] and spec['per'][i].strip() and slot['per'][i].strip()]
        same = sum((1 for i in both if agree(spec['per'][i], slot['per'][i])))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        if both:
            return same * 2 > len(both)
        return bool((set(spec['values']) & set(slot['values'])) - {''})

    def item_groups(self) -> list[list[str]]:
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or '') if cat_key else ''
            groups.setdefault(label, []).append(ident)
        wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any((_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.items))]

    def single_target(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            ident = self.ident(observation)
            if ident:
                return [(ident, observation)]
            found: list[tuple[str, dict[str, Any]]] = []
            for value in observation.values():
                if isinstance(value, list):
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.ident(r, True)] if i]
                elif isinstance(value, dict):
                    inner = self.ident(value, False, 1)
                    if inner:
                        found.append((inner, value))
                    else:
                        found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.ident(r, True)] if i]
            return found
        if isinstance(observation, list):
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.ident(r, True)] if i]
        return []

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
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(r, dict) for r in v))]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + _dump(self.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + _dump(_clip(sample, 120))[:4000]
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), props), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
        cap = named(got.get('ceiling'), props)
        spec = props.get(cap) if isinstance(props.get(cap), dict) else {}
        small = isinstance(spec.get('maximum'), (int, float)) and spec['maximum'] < 1000
        if cap and cap != shape['next_arg'] and set(_kinds(spec)) & {'integer', 'number'} and not small and name == self.roles.get('search'):
            self.ceiling_arg = self.ceiling_arg or cap
        if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
            shape['count'] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.read_shape(name, observation)
        if not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict)):
            return None
        rows, count, value = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return (count, len(rows), shape['next_arg'], value)

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        plan = self.paging(name, observation) if pick in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        if pick == 'cheapest' and self.price_sorted(observation):
            return ''
        pages = -(-(count - shown) // shown)
        if pages > 2 and self.pages_read + 2 <= AUTO_PAGES and (self.left() - self.tail() - 2 >= 12):
            pages = 2
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
            return f"\n[Marlo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = _shape({**args, param: value}, self.schemas.get(name), 'args', faults)
            if faults:
                break
            reply = self.act(name, call)
            if reply is None or self.finished:
                break
            self.pages_read += 1
            seen = reply.get('observation')
            self.observe(name, seen)
            self.paged_obs.append(seen)
            shown_text = self.show(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f'{param}={_dump(value)}: {shown_text}')
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        rest = f"\n[Marlo] {count} listings matched and Marlo has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Marlo read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        # a number argument seen capping prices in another way of searching (or named as the ceiling) left unapplied here
        if name != self.roles.get('search') or not self.price_key:
            return ''
        props = (self.schemas.get(name) or {}).get('properties') or {}
        way = self.way_of(name, args)
        for key, value in args.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0 or self.ceilings.get((key, way)) is not False:
                continue
            proven = [a for k, a, _ in self.ceiling_ways if k == key and self.ceilings.get((k, self.way_of(name, a))) is True]
            if not proven:
                continue
            over = [i for i, _ in self.records_in(observation) if (self.price_of(i) or 0) > value]
            if not over:
                continue
            if proven:
                modes = {k: v for k, v in proven[-1].items() if isinstance((props.get(k) or {}).get('enum'), list)}
                return f'\n[Marlo] This way of searching left {key}={value:g} unapplied: {len(over)} of its rows cost more. It did cap prices when searching with {_dump(modes)}.'
            others = {k: [o for o in props[k]['enum'] if o != v] for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
            hint = '; another setting of ' + ', '.join((f'{k} (such as {_dump(o[0])})' for k, o in others.items() if o)) + ' may apply it' if any(others.values()) else ''
            return f'\n[Marlo] This way of searching left {key}={value:g} unapplied: {len(over)} of its rows cost more{hint}.'
        return ''

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Marlo] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def way_of(self, tool: str, args: dict[str, Any]) -> str:
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        return _dump({k: v for k, v in sorted(args.items()) if isinstance((props.get(k) or {}).get('enum'), list)})

    def note_ceiling(self, tool: str, args: dict[str, Any], observation: Any) -> None:
        """A number argument proves to be a price ceiling for one way of searching (its fixed-choice arguments) when
        every row that search returns is priced at or under it; one row above it disproves it for that way."""
        if tool != self.roles.get('search') or not self.price_key:
            return
        found = [i for i, _ in self.records_in(observation)]
        prices = [self.price_of(i) for i in found]
        if len(prices) < 2 or None in prices:
            return
        modes = self.way_of(tool, args)
        for key, value in args.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                continue
            slot = (key, modes)
            if not all((p <= value for p in prices)) or max(prices) <= value / 20:
                self.ceilings[slot] = False
            elif self.ceilings.get(slot) is None:
                self.ceilings[slot] = True
                self.ceiling_ways.append((key, dict(args), found))

    def last_search_of(self, label: str) -> dict[str, Any] | None:
        cat_key = self.category_key()
        for args, ids in reversed(self.recent_searches):
            if cat_key and any((str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == label for i in ids)):
                return dict(args)
        return None

    def sweep_args(self, tool: str, line: str, label: str) -> tuple[dict[str, Any], str | None]:
        """Search arguments that list this line's kind below its price: a way of searching already proven to honour a
        price ceiling, pointed at this kind; else the buyer's own last search of this kind with a guessed ceiling."""
        cat_key = self.category_key()
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        page_arg = (self.shapes.get(tool) or {}).get('next_arg')
        shown = str((self.rows.get(line) or {}).get(cat_key) or '').strip()
        for key, args, ids in reversed(self.ceiling_ways):
            if self.ceilings.get((key, self.way_of(tool, args))) is not True:
                continue
            kinds = {str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() for i in ids} - {''}
            plan = {k: v for k, v in args.items() if k != page_arg and (isinstance((props.get(k) or {}).get('enum'), list) or k == key)}
            held = [k for k, v in args.items() if isinstance(v, str) and v.strip().lower() in kinds]
            if label in kinds or held:
                for k in held:
                    plan[k] = shown
                return (plan, key)
        base = self.last_search_of(label)
        if base is None:
            return ({}, None)
        plan = {k: v for k, v in base.items() if k != page_arg}
        nums = [k for k, v in props.items() if isinstance(v, dict) and set(_kinds(v)) & {'integer', 'number'} and k != page_arg and self.ceilings.get((k, self.way_of(tool, plan))) is not False]
        named = self.ceiling_arg if self.ceiling_arg in nums else None
        return (plan, named or (nums[0] if len(nums) == 1 else None))

    def cheaper_sweep(self) -> list[str]:
        """Cheapest pick: list each cart line's kind below its price (with a price ceiling the store has proven to
        honour, when there is one), read the pages that come back and open the cheapest unread rows of that kind,
        always keeping the steps the finish needs."""
        tool, cat_key = (self.roles.get('search'), self.category_key())
        if self.brief.get('pick') != 'cheapest' or not self.price_key or not tool or not cat_key:
            return []
        mark = self.cart_mark() + f'#{self.req_stamp}'
        if mark in self.sweeps_done:
            return []
        self.sweeps_done.add(mark)
        schema = self.schemas.get(tool) or {}
        keep = self.tail() + 2 * len(self.items) + 4
        notes: list[str] = []
        for line in self.cart_ids():
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not label:
                continue
            args, cap = self.sweep_args(tool, line, label)
            if not args:
                continue
            if cap:
                spec = (schema.get('properties') or {}).get(cap) or {}
                args[cap] = int(price) - 1 if 'integer' in _kinds(spec) or float(price).is_integer() else round(price - 0.01, 2)
            found: list[str] = []
            pages = 0
            while pages < SWEEP_PAGES and self.left() - 1 > keep and self.room_for(6) and (not self.finished):
                faults: list[str] = []
                call = _shape(args, schema, 'args', faults)
                if faults:
                    break
                reply = self.act(tool, call)
                pages += 1
                if reply is None:
                    break
                seen = reply.get('observation')
                self.observe(tool, seen)
                self.note_ceiling(tool, call, seen)
                rows = [i for i, _ in self.records_in(seen)]
                found += [i for i in rows if i != line and self.buyable(i) and str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == label and (self.price_of(i) if self.price_of(i) is not None else float('inf')) < price]
                plan = self.paging(tool, seen)
                if not plan or plan[3] is None or plan[0] <= plan[1] * pages:
                    break
                args = {**args, plan[2]: plan[3]}
            fresh = sorted({i for i in found if i not in self.full}, key=lambda i: self.price_of(i) or 0.0)[:SWEEP_OPENS]
            opened = self.fill_facts(fresh, keep)
            self.trace.append({'kind': 'sweep', 'line': line, 'cap': cap, 'pages': pages, 'cheaper': len(set(found)), 'opened': len(opened)})
            if found:
                notes.append(f"{line} costs {price:g}; buyable listings of its kind priced lower: " + ', '.join((f'{i} ({self.price_of(i):g})' for i in sorted(set(found), key=lambda i: self.price_of(i) or 0.0)[:6])))
            notes += opened
        return notes

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i))]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False or not self.room_for(3):
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if isinstance(seen, dict) and self.ident(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f'{ident}: ' + self.show(seen)[:1500])
        if texts:
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def cheaper_rows(self) -> list[str]:
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.cart_mark() + f'#{self.version}'
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        picks: list[str] = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

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
        tool = self.roles.get('compare')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])) or self.compare_bare():
            self.widen_singly()
            return
        if self.left() <= 12:
            return
        cap = props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4
        ranked = self.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = _shape({target: picks}, schema, 'args', faults)
        if faults:
            return
        before = len(self.full)
        reply = self.act(tool, args)
        if reply is not None:
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.note_view(tool, seen)
            rows = self.records_in(seen)
            if rows:
                self.facts_from[tool] = self.facts_from.get(tool, False) or any((_fact_block(r) for _, r in rows))
            self.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(self.full) - before})
            if rows and self.facts_from.get(tool) is False:
                self.widen_singly()
                return
            if len(self.full) == before and len(picks) > 2 and (self.left() > 12):
                faults = []
                args = _shape({target: picks[:2]}, schema, 'args', faults)
                if not faults:
                    reply = self.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get('observation'))

    def widen_singly(self) -> None:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        wanted = {w for _, need in self.needs_now() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())}
        each = WIDEN_EACH if len(self.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in self.item_groups():
            opened = sum((1 for i in ids if i in self.full))
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get('pick') == 'cheapest':
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
            pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))))
            lanes.append(pool[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.fill_facts(first[:cap][:8], 22)
        self.fill_facts((first[cap:] + more)[:max(0, 10 - len(read))], 30)


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


    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', k, re.I)]
        for args in reversed(self.searched):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
            if not named:
                labels = {str(r.get(self.category_key()) or '').strip().lower() for r in self.rows.values()} - {''} if self.category_key() else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.searched if a.get(k)}) >= 2 and any((len(str(a.get(k)).split()) >= 2 for a in self.searched)) and (not any((str(a.get(k) or '').strip().lower() in labels for a in self.searched)))]
            if fields and tool:
                field = fields[0]
                keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get('enum'), list) or k in (schema.get('required') or []))}
                cat_key = self.category_key()
                shelves = {_words(item.shelf or '') for item in self.items} | {_words(item.name) for item in self.items}
                shelves |= {_words(str(row.get(cat_key) or '')) for row in self.rows.values()} if cat_key else set()
                keep = {k: v for k, v in keep.items() if k in (schema.get('required') or []) or not isinstance(v, str) or (_words(v) not in shelves and (not any(({w for w in _stem(v) if len(w) >= 4} & {w for w in _stem(s) if len(w) >= 4} for s in shelves if s))))}
                worded = [a for a in self.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
                for k in list(keep):
                    if isinstance((props.get(k) or {}).get('enum'), list):
                        used = [_dump(a[k]) for a in worded if k in a]
                        if used:
                            best = max(used, key=used.count)
                            keep[k] = next((a[k] for a in worded if k in a and _dump(a[k]) == best))
                return (tool, keep, field)
        return None

    def store_says(self, value: str) -> bool:
        # does some opened listing show this value word for word inside an attribute value?
        low = ' '.join(value.lower().split())
        if len(low) < 2:
            return False
        pattern = re.compile('(?<![a-z0-9])' + re.escape(low) + '(?![a-z0-9])')
        return any((pattern.search(' '.join(str(v).lower().split())) for record in self.full.values() for v in (_fact_block(record) or {}).values()))

    def value_words(self, number: int) -> list[str]:
        # values worth a search: told values read as plain, or values an opened listing shows word for word
        borrowed = {m for _, m in self.refs}
        picks: list[str] = []
        for owner, need in self.needs_now():
            _, colon, said = need.partition(':')
            said = said.strip()
            if owner == number and colon and said and need not in borrowed and (need in self.plain_needs or self.store_says(said)):
                picks.append(said)
        return picks

    def value_search(self, numbers: list[int]) -> str:
        plan = self.search_template()
        if not plan or self.left() < 22 or not self.room_for(20):
            return ''
        tool, base, field = plan
        seen: list[str] = []
        budget = 8
        for number in numbers:
            values = self.value_words(number)
            item = self.items[number - 1]
            fresh = [v for v in values if any((':' in t and t.split(':', 1)[1].strip() == v for t in item.told))]
            if not fresh or budget <= 2:
                continue
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [values]
            for words in tries[:2]:
                if not words:
                    continue
                faults: list[str] = []
                args = _shape({**base, field: ' '.join(words)}, self.schemas.get(tool), 'args', faults)
                if faults or self.finished:
                    break
                reply = self.act(tool, args)
                budget -= 1
                if reply is None:
                    break
                observation = reply.get('observation')
                self.observe(tool, observation)
                cat_key = self.category_key()
                found = [i for i, _ in self.records_in(observation)]
                cap = self.budget if isinstance(self.budget, (int, float)) else None
                prices = [self.price_of(i) for i in found]
                if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                    cap = None
                rows = [i for i in found if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.price_of(i) or 0) <= cap)]
                seen.append(f'{_dump(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                if rows:
                    opened = self.fill_facts([i for i in rows if i not in self.full][:min(3, budget)], self.tail() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Marlo searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)




    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        words = _words(label)
        long = _roots(label)
        best, top = (0, 0.0)
        for number, item in enumerate(self.items, 1):
            if words == _words(item.shelf or ''):
                score = 3.0
            elif words == _words(item.name):
                score = 2.0
            else:
                mine = _roots(item.shelf or '') | _roots(item.name)
                score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                best, top = (number, score)
        return best

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1)

    def records_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))




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
        mark = (len(told), self.budget)
        self.absorb_heard(heard, told)
        if (len(told), self.budget) != mark:
            self.req_stamp += 1

    def absorb_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        for entry in _items(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
            need = str(entry['need']).strip()
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in self.refs:
                self.refs.append((number, need))
            if entry.get('plain') is True:
                self.plain_needs.add(need)
        for entry in _items(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
                gone = _words(str(entry['property']).split(':')[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
        for entry in _items(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry['property']))
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.budget = heard['budget']

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nPRODUCTS:\n' + '\n'.join((f'product {n}: {it.name}' + (f' (store label: {self.shelf_word(n)})' if self.shelf_word(n) != it.name else '') for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about product {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500, None, critical=True) or {'unread': True}

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished or (not self.gate_due(stage)):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.ledger() + '\n\nCART NOW: ' + self.cart_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500, None, critical=2)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if said:
                self.take_heard({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.version += 1
                return (False, ('Before going on, Marlo did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2):
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Marlo: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
                self.done_steps.append(-1)
                continue
            if not call or name not in self.schemas or name in ending:
                if not call:
                    self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
                break
            if self.left() < 2 or (name == self.roles.get('ask') and self.msg_used >= self.msg_limit):
                break
            faults: list[str] = []
            args = _shape(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
            if faults:
                break
            if name == self.roles.get('ask'):
                text = next((v for v in args.values() if isinstance(v, str)), '')
                batch = self.send([(1, '', text)], 1)
                if not batch:
                    break
                self.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = self.act(name, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.absorb(seen)
            self.done_log.append(f'step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            said.append(f'{name} {_dump(args)} -> {self.show(seen)[:400]}')
        return (True, 'Before going on, Marlo did the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def spoken(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, str) and len(node.split()) >= 2:
                found.append(node)
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        return ' '.join(found)[:800] if found else _dump(observation)[:400]

    def loose_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and (not self.ident(observation)):
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all((isinstance(v, str) for v in value)):
                    lines.extend((v for v in value if len(v.split()) >= 5))
                elif isinstance(value, dict) and (not self.ident(value, False, 1)):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.ledger() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500, None, critical=True) or {}
        gone = [str(g) for g in _items(got.get('gone')) if isinstance(g, (str, int))]
        self.unavailable.update((g for g in gone if g in self.rows))
        self.note_sold = bool(gone) or got.get('sold_out') is True
        self.note_named = any((g in self.rows for g in gone))
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = got.get('budget')
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (budget != self.budget):
            self.budget = budget
            changed.append(f'budget now {budget}')
        added: set[str] = set()
        drops = [e for e in _items(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        adds = [e for e in _items(got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        for entry in drops:
            item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
            target = str(entry['need']).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            if not hit:
                hit = [m for m in live if _words(m.split(':')[0]) == _words(target.split(':')[0])][:1]
            item.dropped.extend(hit or [target])
            changed.append(f'dropped {target}')
        for entry in adds:
            item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
            need = str(entry['need']).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            added.add(need)
            changed.append(f'added {need}')
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price changed')
        summary = str(got.get('summary') or '').strip()
        if self.brief.get('pick') == 'cheapest' and (drops or gone or self.note_sold or any((c.startswith('budget now') for c in changed))):
            self.wider_held = False
            self.cheaper_marks = set()
        if drops or adds or any((c.startswith('budget now') for c in changed)):
            self.req_stamp += 1
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or '; '.join(changed))
        again = self.repick_line() if changed else ''
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '') + (' ' + again if again else '')).strip()

    def needs_now(self) -> list[tuple[int, str]]:
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence(self, observation: Any) -> str:
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, False, depth)
                if ident and _fact_block(node) and (ident not in ids):
                    ids.append(ident)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        needs = self.needs_now()
        if not ids or not needs:
            return ''
        many = len(self.items) > 1
        lines = []
        for ident in ids[:12]:
            block = _fact_block(self.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if _names(_stem(prop), _stem(k))]
                tag = f'(item {number}) ' if many else ''
                if hits:
                    parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]))
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        if not lines:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)

    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by Marlo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def held_units(self, ident: str) -> int:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            stack: list[Any] = [self.cart_view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if ident in [v.strip() for v in node.values() if isinstance(v, str)]:
                        whole = [(k, _count(v)) for k, v in node.items() if not isinstance(v, (str, bool)) and _count(v) is not None and k != self.price_key]
                        counts = [c for k, c in whole if k == self.count_arg()]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.cart_lines.get(ident, 1)

    def count_arg(self) -> str | None:
        # the cart tool's single whole-number argument (the unit count), from its schema
        props = (self.schemas.get(self.roles.get('cart') or '') or {}).get('properties') or {}
        whole = [k for k, v in props.items() if isinstance(v, dict) and 'integer' in _kinds(v)]
        return whole[0] if len(whole) == 1 else None

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def record_of(self, ident: str, limit: int=400, room: int=3000) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_shorten(record, limit))[:room] if record else '(no record seen)'

    def others(self, skip: list[str], limit: int=30) -> str:
        lines = []
        order = list(reversed(self.seen_order))
        if self.brief.get('pick') == 'cheapest':
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float('inf')))
        for ident in order:
            if ident in skip or self.sold_out(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 160))[:900])
            if len(lines) >= limit:
                break
        return '\n'.join(lines)

    def fit_index(self) -> dict[str, Any]:
        """Codes on each seen listing by position: its own bracketed mark, and the other codes its text names."""
        marks: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident) or self.rows.get(ident) or {}
            text = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
            text += ' ' + '. '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            marks[ident] = _marks_by_position(text)
        return {'marks': marks}

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        return set(mark.get('named') or []) - set(mark.get('own') or [])

    def codes_on(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        return set(mark.get('own') or []) | set(mark.get('named') or [])

    def fit_faults(self) -> list[str]:
        """A part whose text names codes, none of them carried by the device in the cart: worth a second look."""
        if not self.bound:
            return []
        index = self.fit_index()
        cart = self.cart_ids()
        faults = []
        for part, main in self.bound:
            devices = [i for i in cart if i in self.item_ids(main)]
            parts = [i for i in cart if i in self.item_ids(part)]
            if len(devices) != 1 or not parts:
                continue
            codes = self.codes_on(devices[0], index)
            if not codes:
                continue
            for line in parts:
                named = self.fit_set(line, index)
                if named and (not named & codes):
                    good = [i for i in self.item_ids(part) if self.fit_set(i, index) & codes and (not self.sold_out(i)) and (i not in cart)][:4]
                    faults.append(f"{line} ({self.items[part - 1].name}) names {', '.join(sorted(named)[:4])} in its text, and none of these is on {devices[0]} ({', '.join(sorted(codes)[:4])})" + (f"; seen listings of that kind that name a code of {devices[0]}: {', '.join(good)}" if good else '; no seen listing of that kind names a code of that device, so look for one or reconsider the device: ' + self.fit_plan(short=True)))
        return faults

    def fit_plan(self, short: bool=False) -> str:
        if not self.bound:
            return ''
        index = self.fit_index()
        lines = []
        for main in sorted({m for _, m in self.bound}):
            parts = [p for p, m in self.bound if m == main]
            tally: dict[str, dict[int, int]] = {}
            for p in parts:
                for i in self.item_ids(p):
                    if self.sold_out(i):
                        continue
                    for code in self.fit_set(i, index):
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            devices = {code: [i for i in self.item_ids(main) if code in self.codes_on(i, index)][:2] for code, _ in ranked}
            text = '; '.join((f'{code}: named by ' + ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" (carried by {', '.join(devices[code])})" if devices[code] else ' (no seen device carries it yet)') for code, per in ranked))
            if text:
                lines.append(f"codes named by listings meant to go with the {self.items[main - 1].name} -> {text}")
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[pairing codes: an accessory and a device go together when the accessory's text names a code the device carries; read the device rows of its category for their codes, and judge each pairing from both listings' own words] " + ' | '.join(lines)

    def line_faults(self) -> list[str]:
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not ids or not cat_key:
            return []
        label = {i: str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids}
        known = {n for n, item in enumerate(self.items, 1) if any((self.owns(item, str(r.get(cat_key) or '')) for r in self.rows.values()))}
        owner = {i: [n for n, item in enumerate(self.items, 1) if self.owns(item, label[i])] for i in ids}
        faults: list[str] = []
        for n, item in enumerate(self.items, 1):
            if n not in known:
                continue
            lines = [i for i in ids if n in owner[i]]
            if not lines:
                faults.append(f'item {n} ({item.name}) has no line in the cart')
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.held_units(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
        if len(known) == len(self.items):
            for i in ids:
                if not owner[i]:
                    faults.append(f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
        for i in ids:
            if self.sold_out(i):
                faults.append(f'{i} is sold out')
        prices = {i: self.price_of(i) for i in ids}
        if isinstance(self.budget, (int, float)) and all((p is not None for p in prices.values())):
            if self.brief.get('budget_scope') == 'each':
                over = [i for i in ids if self.budget < prices[i] <= 10 * self.budget]
                faults += [f'{i} costs {prices[i]:g}, over the {self.budget:g} limit for each item' for i in over]
            else:
                total = sum((prices[i] * self.held_units(i) for i in ids))
                if self.budget + 1e-09 < total <= 10 * self.budget:
                    faults.append(f'the total is {total:g}, over the budget of {self.budget:g}')
        return faults

    def count_fixes(self) -> list[tuple[str, int, int]]:
        cat_key = self.category_key()
        if not cat_key:
            return []
        out = []
        for ident in self.cart_ids():
            have = self.held_units(ident) or 0
            owners = [n for n, item in enumerate(self.items, 1) if self.owns(item, str((self.rows.get(ident) or {}).get(cat_key) or ''))]
            mine = [i for i in self.cart_ids() if (self.cart_lines.get(i, 1) or 0) > 0 and len(owners) == 1 and self.owns(self.items[owners[0] - 1], str((self.rows.get(i) or {}).get(cat_key) or ''))]
            if have > 0 and len(owners) == 1 and (len(mine) == 1) and (have != self.items[owners[0] - 1].qty) and (not self.sold_out(ident)):
                out.append((ident, have, self.items[owners[0] - 1].qty))
        return out

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)), None)
        if not name or not target or (not count):
            return []
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if self.cart_sets(name):
                calls.append({target: ident, count: want})
            elif mode:
                options = [str(o) for o in props[mode]['enum']]
                off = next((o for o in options if re.search('remov|delet|drop|clear|take', o, re.I)), options[1])
                on = next((o for o in options if o != off), options[0])
                calls += [{target: ident, mode: off, count: have}, {target: ident, mode: on, count: want}]
            else:
                return []
        if len(calls) + 3 > self.left():
            return []
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, schema, 'args', faults)
            if faults:
                break
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            self.note_cart(name, args)
            self.cart_said = _dump(reply.get('observation'))
            self.observe(name, reply.get('observation'))
            done.append(_dump(args) + ' -> ' + self.show(reply.get('observation'))[:300])
        self.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    def check_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_CEILING - HEAVY_ROOM) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or not self.closing_judges(limit) or (not ids) or (not self.room_for(MODEL_SECONDS)):
            return ''
        self.checked.add(key)
        prompt = CHECK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1)))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS (every view read, merged):\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.sold_out(i) else '') + self.record_of(i, 1600, 5000) for i in ids))
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
        got = self.ask_json(prompt, self.closing_judges(limit), 3000)
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        said = got.get('lines') if isinstance(got, dict) and isinstance(got.get('lines'), dict) else {}
        for ident in ids:
            if isinstance(said.get(ident), bool):
                self.rate(ident, said[ident], 'final review')
            elif got and got.get('ok') is True:
                self.rate(ident, True, 'final review')
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get('problems'))][:5]
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        for ident in better:
            self.rate(ident, True, 'named by the final review')
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def trail(self) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < (6 if self.left() > 15 else 4) and (self.cost < SPEND_HARD) and self.room_for(MODEL_SECONDS):
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                record = self.full.get(ident) or self.rows.get(ident) if ident in self.opened else None
                if record and len(records) < 14:
                    records.append('OPENED ' + _dump(_shorten(record, 600))[:1600])
            for ident in reversed(self.seen_order):
                if len(records) >= 30:
                    break
                if ident not in self.opened or not (self.full.get(ident) or self.rows.get(ident)):
                    records.append('ROW ' + _dump(_clip(self.rows.get(ident), 80))[:260])
            prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.ask_json(prompt, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else []), 6000) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in _items(got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get('links')) if isinstance(l, dict)]
                ids = [str(l.get('id') or '').strip() for l in links]
                if chain.get('done') is True and links and all((i in self.opened for i in ids)) and str(chain.get('value') or '').strip():
                    self.ref_values[need] = str(chain['value']).strip()
                    self.version += 1
                    continue
                if links and all((i in self.opened for i in ids)) and (not str(links[-1].get('find') or '').strip()):
                    self.ref_values[need] = str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}'
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    gap = next((l for l, i in zip(links, ids) if i not in self.rows), {})
                    about = str(gap.get('about') or '') or 'the listing it points at'
                    words = str(gap.get('find') or '').strip()
                    lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and tool and target and (self.left() > 12) and (self.link_searches < 3):
                reach = self.find_link(*gaps[0])
                if self.trace and self.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
            if not reach or not tool or (not target) or (self.left() < len(reach) + 4) or self.finished:
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
                if faults:
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.observe(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.show(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[Marlo opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (text, hold)

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        low = about.lower()
        out = []
        for ident in ids:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if ident not in self.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def opened_shop(self, about: str) -> list[str]:
        low = about.lower()
        out = []
        for ident in self.opened:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def find_link(self, about: str, words: str) -> list[str]:
        known: list[str] = []
        named = {w for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        terms = self.plain_terms(words, about)
        values = terms - named
        for ident in self.opened_shop(about):
            stack, leaves = ([self.full.get(ident) or self.rows.get(ident) or {}], [])
            while stack:
                node = stack.pop()
                if isinstance(node, (dict, list)):
                    stack += list(node.values()) if isinstance(node, dict) else node
                else:
                    leaves.append(str(node))
            text = ' '.join(leaves).lower()
            if values and all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values)):
                self.trace.append({'kind': 'link_known', 'id': ident})
                return []
        plan = self.search_template()
        memo = ' '.join(words.lower().split())
        if memo and memo in self.link_seen:
            self.link_searches += 1
            known = self.shop_rows(about, self.link_seen[memo])
        elif plan and words:
            tool, base, field = plan
            faults: list[str] = []
            args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults)
            if not faults:
                self.link_searches += 1
                reply = self.act(tool, args)
                if reply is not None:
                    observation = reply.get('observation')
                    self.observe(tool, observation)
                    self.link_seen[memo] = [i for i, _ in self.records_in(observation)]
                    known = self.shop_rows(about, self.link_seen[memo])
                    self.trace.append({'kind': 'link_search', 'words': words[:80], 'rows': known[:4]})
        kind = {w for w in terms if len(w) >= 3} - named

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

    def plain_terms(self, words: str, about: str) -> set[str]:
        # the model's own search words when it gave some; else the description's words, less those most rows seen in
        # this episode carry (the episode's own filler); numbers of any length, other words of three letters up
        terms = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if w.isdigit() or len(w) > 2}
        texts = [_dump(r).lower() for r in list(self.rows.values())[:200]]
        if words.strip() or len(texts) < 4:
            return terms
        return {w for w in terms if sum((1 for t in texts if re.search('(?<![a-z0-9])' + re.escape(w) + '(?![a-z0-9])', t))) * 2 < len(texts)}

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if not stale or self.left() < len(stale) + 3:
            return ''
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            before = dict(self.rows.get(ident) or {})
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            observation = reply.get('observation')
            self.offer_seen[ident] = self.step
            self.observe(tool, observation)
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
        if lines:
            self.version += 1
        return '\n'.join(lines)

    def sold_out(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ''
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        number = self.product_of(lost[0])
        cart = self.cart_ids()
        ready = [i for i in self.ranked_for(number) if i not in lost and i not in cart] if number and len(lost) == 1 else []
        picks = ready[:4] if ready else self.replacements(lost)
        if self.repair_cap is not None or self.checkout_refusals:
            self.repair_cap = self.cost + REPAIR_ROOM
        self.trace.append({'kind': 'repick', 'lost': lost, 'from_board': bool(ready), 'picks': picks[:4]})
        text = f"[Marlo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get('pick') == 'cheapest':
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def find_sold(self) -> str:
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any((self.sold_out(i) for i in ids)):
            return ''
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ''
        return self.reread_offers() if ids else ''

    def replacements(self, lost: list[str]) -> list[str]:
        if not self.judges() or not self.room_for(MODEL_SECONDS):
            return []
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(_dump(_shorten(self.full.get(ident) or self.rows.get(ident), 300))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.record_of(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.cart_text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges(), 3500)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and (ident not in self.gone) and (ident not in self.cart_ids()) and (ident not in [i for _, i in fits]):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get('pick') == 'cheapest' else [i for _, i in fits]
        # a named fit whose own record says otherwise (another kind, over budget, its key for a requirement holding another
        # value) is no replacement: carting it only costs the steps of taking it out again
        number = self.product_for(lost[0]) if lost else 0
        held = [i for i in ranked if not (number and self.record_fails(i, number))]
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits, 'dropped': [i for i in ranked if i not in held]})
        ranked = held
        for ident in ranked:
            self.rate(ident, True, 'replacement review', self.product_of(lost[0]) if lost else 0)
        return ranked[:4]

    def audit(self) -> str:
        if self.audited:
            return ''
        if not self.judges() or not self.room_for(MODEL_SECONDS):
            self.review_state = False
            return ''
        self.audited = True
        got = self.audit_call()
        self.review_state = got is not None
        named = self.audit_fits(got)
        if not named:
            rows = [str(i) for i in _items((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ''
        for number, ids in named.items():
            for ident in ids:
                self.rate(ident, True, 'evidence review', _count(number) or 0)
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        for ident in reversed(self.seen_order):
            if self.sold_out(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            records.append('OPENED ' + _dump(_shorten(record, 600))[:1600] if record else 'ROW ' + _dump(_shorten(self.rows.get(ident), 120))[:500])
            if len(records) >= 26:
                break
        if not records:
            return None
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges(), 4000)
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and (not self.compare_bare()):
            plan = [(joiner, {many: rows[:self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:2]]
        else:
            return False
        if self.left() < len(plan) + 5:
            return False
        before = len(self.full)
        for name, args in plan:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(name), 'args', faults)
            if faults:
                return False
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            self.observe(name, reply.get('observation'))
        self.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.full) - before})
        return len(self.full) > before

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        for key, spec in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in _kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                inner = shape['properties']
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2)), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]['enum'][1], **({count: have} if count else {})})
            elif count and self.cart_sets(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get('minimum'), int) and (props[count]['minimum'] <= -have):
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if have and (not self.cart_sets(name)):
                return None
            calls.append({target: ident, **({mode: props[mode]['enum'][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def complete_cart(self) -> str:
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or (not cart) or (not buy):
            return ''
        self.completed = True
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ident
        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            index = _count(number)
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if pick and index and (1 <= index <= len(self.items)) and (kind(pick) not in {kind(p) for p in plan}):
                plan[pick] = self.items[index - 1].qty
        if self.confirmed not in plan:
            return ''
        now = self.cart_ids()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and (not self.sold_out(i)) and (kind(i) not in {kind(p) for p in plan})}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.items):
            return ''
        lines = {**keep, **plan}
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + 1 > self.left():
            return ''
        self.fill_facts(list(lines), len(calls) + 3)
        self.read_all_views(list(lines), len(calls) + 3)
        self.check_cart(list(lines), SPEND_HARD)
        if not self.verdicts.get(self.check_key(list(lines))):
            return ''
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                return ''
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ''
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
            done.append(_dump(args))
        faults = []
        args = self.checkout_args(faults)
        if faults:
            return ''
        replies = []
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                self.check_cart(list(lines), SPEND_HARD)
                if not self.verdicts.get(self.check_key(list(lines))) or self.left() < 1 or self.finished or any((self.sold_out(i) for i in lines)):
                    break
            go, said = self.gate('order')
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.act(buy, args)
            self.trace.append({'kind': 'completed', 'lines': lines, 'drop': drop, 'reply': _clip(reply, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
            self.count_refusal(buy, args, reply)
            self.observe(buy, reply.get('observation'))
            replies.append(self.show(reply.get('observation')))
            if self.finished or reply.get('finished'):
                break
        if not replies:
            return ''
        return 'Not run (Marlo acted instead): the independent check confirmed ' + self.confirmed + ', so Marlo set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('buy'):
            return False
        fixed = self.brief.get('buy_args') if isinstance(self.brief.get('buy_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('refuse'):
            return False
        fixed = self.brief.get('refuse_args') if isinstance(self.brief.get('refuse_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def offered(self) -> list[dict[str, Any]]:
        left = self.left()
        ask = self.roles.get('ask')
        keep: set[str] | None = None
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        if left <= self.tail():
            keep = set(ends)
        elif left <= self.tail() + 3:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view')}
        elif left <= len(self.items) + 5 and self.buys == 0:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
        tools = [t for t in self.tools if t['function']['name'] != ask]
        if self.refusal_seen() and self.roles.get('refuse') and (self.roles.get('refuse') != self.roles.get('buy')):
            tools = [t for t in tools if t['function']['name'] != self.roles.get('refuse')]
        if keep is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        return tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in _kinds(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get('required') or []):
                continue
            text = str(value or '')
            named = any((ident in text for ident in self.rows)) or self.brief.get('quiet_decline') is not False
            if named or not text:
                out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.id_key:
            return (args, '')
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _kinds(v) or 'array' in _kinds(v)) and ('enum' not in v)]
        if len(free) != 1 or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args):
            return (args, '')

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.rows:
                return (value.strip(), '')
            if not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8:
                return (value, '')
            low = value.strip().lower()
            hits = [i for i, row in self.rows.items() if any((k != self.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items()))]
            if len(hits) == 1:
                return (hits[0], '')
            if hits:
                return (value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id")
            return (value, f'{value!r} is not a listing id; pass the id from the result rows (their {self.id_key} field)')
        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for _, p in fixed if p), '')
            if problem:
                return (args, problem)
            out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return (out, '')

    def cart_sets(self, name: str) -> bool:
        props = self.schemas.get(name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in _kinds(s) and (s.get('minimum') == 0) for s in props.values()))

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get('properties') or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2 and (value in options[1:]):
                removal = True
        loose_count = next((_count(v) for v in args.values() if _count(v) is not None and (not isinstance(v, str))), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (spec.get(key) or {}).get('items') or {}
                if not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape):
                    whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                        count = next((_count(v) for v in line.values() if _count(v) is not None and (not isinstance(v, str))), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(line, str) and line in self.rows:
                        lines.append((line, loose_count))
            elif isinstance(value, str) and value in self.rows:
                lines.append((value, loose_count))
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_sets(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            if removal:
                have = have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def show(self, observation: Any) -> str:
        found = self.records_in(observation)
        limit = PAGE_CHARS if len(found) <= 1 else SHOW_CHARS
        text = _dump(_shorten(observation, limit))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT] + '...(result cut here; open single listings to read them whole)'
        return text

    def compare_target(self) -> str | None:
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def compare_room(self) -> int:
        spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.compare_target() or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = (self.roles.get('open'), self.roles.get('compare'), self.single_target(self.roles.get('open')))
        many = self.compare_target()
        if not opener or not joiner or (not target) or (not many) or (opener == joiner):
            return calls
        if self.compare_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get('function') or {}
            args = _json_obj(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            ident = str((args or {}).get(target) or '').strip()
            if fn.get('name') == opener and args and (set(args) == {target}) and (ident in self.rows) and (ident not in self.gone):
                picks.append((index, ident))
        picks = picks[:self.compare_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': _dump({many: [i for _, i in picks]})}})
            elif index in {i for i, _ in picks}:
                out.append({'id': call['id'], 'joined': True})
            else:
                out.append(call)
        self.trace.append({'kind': 'joined', 'ids': [i for _, i in picks]})
        return out

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Marlo opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask'):
                out.append((cid, "Not run (no step used): Marlo's interviewer handles every message to the shopper."))
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
            args, wrong = self.stop_codes(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong))
                continue
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            trimmed = ''
            many = self.compare_target() if name == self.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.compare_room()) and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Marlo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
                    trimmed += f"\n[Marlo] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if self.is_refuse(name, args) and (not self.is_buy(name, args)) and self.refusal_seen():
                self.end_stops += 1
                if self.end_stops == 1 and self.left() > 3:
                    self.trace.append({'kind': 'end_after_refusal_held'})
                    out.append((cid, 'Not run (no step used). Earlier in this episode the store turned down ' + ('a checkout' if self.checkout_refusals else 'the no-purchase ending') + ', which tells you the shopper expects an order; a no-purchase ending is off the table now. ' + (self.repick_line() or self.board_line() or 'Put the best fitting listing for each product in the cart.') + ' Adjust the cart to what the store said and send the checkout.\nCart now: ' + self.cart_text()))
                    continue
                done = self.checkout_on_file('a finish was turned down earlier, so an order is expected', loose=1)
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                self.trace.append({'kind': 'end_after_refusal_allowed', 'cart': self.cart_ids()[:4], 'items': len(self.items)})
            if ending and self.left() > 2:
                if self.is_buy(name, args):
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
                    stuck = [i for i in self.cart_ids() if i in self.gone]
                    placed = self.repair_checkout(f'after two reminders the cart still held {stuck[0]}, which sold out') if stuck else ''
                    if placed:
                        out.append((cid, placed + self.end_mark()))
                        continue
                    covered = self.read_all_views(self.cart_ids(), self.tail() + 2 * len(self.items) + 2)
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
                            out.append((cid, "Not run yet (no step used). Marlo set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()))
                            continue
                    if faults:
                        self.line_holds += 1
                        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                        out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.cart_text()))
                        continue
                    swept = self.cheaper_sweep()
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = "Pages Marlo opened for this review, a step apiece:\n" + '\n'.join(covered + swept + read) + '\n' if covered or swept or read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Marlo re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.left() > 5:
                    review = self.audit()
                    if review:
                        out.append((cid, 'Not run yet (no step used). ' + review))
                        continue
                    if self.review_state is False and (not self.leads):
                        done = self.checkout_on_file('the decline could not be reviewed, and the cart on hand is worth more than no order')
                        if done:
                            out.append((cid, done + (' [episode finished]' if self.finished else '')))
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
                args = self.quiet(name, args)
            if ending and self.left() >= 2:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Marlo] ' + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Marlo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
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
                self.recent_searches = (self.recent_searches + [(dict(args), [i for i, _ in self.records_in(observation)])])[-12:]
                self.note_ceiling(name, args, observation)
            if name == self.roles.get('cart_view'):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                self.model_checkout = dict(args)
                if not reply.get('finished'):
                    self.refused_at = self.step
            if ending:
                self.count_refusal(name, args, reply)
            text = self.show(observation) + paged + trimmed
            if name not in plain and (not ending):
                text += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, args, observation)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                    text += '\n[Marlo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
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
                    text += "\n[Marlo re-read the cart's lines after the refusal]\n" + seen
            self.note_sold = False
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes seen on Marlo's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def finish(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ''
        self.wrapping = True
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            if holding and self.left() >= 3 and self.needs_repair():
                mended = self.close_with_repair(again)
                if mended is not None:
                    return mended
            doomed = holding and refuse and not self.end_refusals and any((self.sold_out(i) or self.record_fails(i, self.product_for(i)) for i in self.cart_ids()))
            faults: list[str] = []
            args = self.checkout_args(faults) if not doomed else {}
            if not faults and not doomed:
                self.read_all_views(self.cart_ids(), 2 + (0 if self.buys else REFUSAL_ROOM))
                if self.left() >= 2 and self.room_for(MODEL_SECONDS):
                    self.gate('order')
                if self.finished or self.left() <= 0:
                    return ''
                reply = self.act(buy, args)
                self.trace.append({'kind': 'forced_buy', 'reply': _clip(reply, 200)})
                if self.finished or self.left() <= 0 or reply is None:
                    return ''
                self.buys += 1
                self.refused_at = self.step
                self.count_refusal(buy, args, reply)
                self.observe(buy, reply.get('observation'))
                return 'the checkout was refused: ' + self.show(reply.get('observation')) if again else ''
        elif buy and self.left() >= 2 and (self.refusal_seen() or self.review_state is not True or self.leads):
            owed = self.refusal_seen()
            done = self.checkout_on_file('time or budget is ending with an empty cart; ' + ('the store expects an order' if owed else 'a review judged these listings to fit'), loose=2 if self.end_refusals else 1 if owed else 0)
            if self.finished:
                return ''
            if done:
                return 'the checkout was refused: ' + done if again else ''
        if refuse and (not self.refusal_seen() or not self.basket_on_file(2 if self.end_refusals else 1)):
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
            if not faults:
                if self.left() >= 2 and self.room_for(MODEL_SECONDS):
                    self.gate('end')
                if self.finished or self.left() <= 0:
                    return ''
                reply = self.act(refuse, args)
                self.trace.append({'kind': 'forced_refuse', 'reply': _clip(reply, 200)})
                if reply is not None and (not self.finished):
                    self.count_refusal(refuse, args, reply)
                    self.observe(refuse, reply.get('observation'))
                    return 'ending without a purchase was refused: ' + self.show(reply.get('observation'))
        return self.last_word()

    def last_word(self) -> str:
        # nothing above ended the episode while a step remains: it ends now, an order of a cart that can stand, else the
        # no-purchase ending (an unfinished episode earns nothing either way)
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        if self.finished or self.broke or self.left() < 1 or not (buy or refuse):
            return ''
        cart = self.cart_ids()
        sound = bool(cart) and not any((self.sold_out(i) or self.record_fails(i, self.product_for(i)) for i in cart))
        ordering = bool(buy) and (sound or not refuse or (bool(cart) and bool(self.end_refusals)))
        name = buy if ordering else refuse
        faults: list[str] = []
        args = self.checkout_args(faults) if ordering else _shape(self.quiet(name, dict(self.brief.get('refuse_args') or {})), self.schemas.get(name), 'args', faults)
        if faults:
            return ''
        reply = self.act(name, args)
        self.trace.append({'kind': 'last_word', 'tool': name, 'sound_cart': sound, 'reply': _clip(reply, 200)})
        if reply is not None and not self.finished:
            self.count_refusal(name, args, reply)
        return ''

    def close_with_repair(self, again: bool) -> str | None:
        # the cart cannot go out as it is: Marlo sets it from the reviewed candidates and orders it; None when no basket
        # on file can replace the stale lines (the plain forced checkout follows)
        placed = self.repair_checkout('a cart line can no longer be bought as it is, and the episode is ending')
        if self.finished:
            return ''
        if not placed:
            return None
        return f'the repaired cart was turned down too: {placed}' if again else ''

    def end_mark(self) -> str:
        return ' [episode finished]' if self.finished else ''

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
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                self.chat.append({'role': 'user', 'content': '[Marlo] Marlo tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.wrapping = False
                self.work(self.loop_calls + 10, SPEND_HARD + 0.003)
            if not self.finished:
                self.finish()
            for _ in range(3):
                if self.finished or self.broke or self.no_credit or self.left() < 1 or self.seconds_left() < LAST_CALL_SECONDS:
                    break
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def interview(self) -> str:
        # the asking policy behind a guard: whatever goes wrong in it, the buyer keeps going without it
        try:
            return self.asker.run()
        except Exception as exc:
            self.asked = True
            self.trace.append({'kind': 'interview_error', 'error': f'{type(exc).__name__}: {exc}'[:300]})
            return ''

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap or self.repairing()) and self.room_for(MODEL_SECONDS):
            if self.asker.wants_turn():
                summary = self.interview()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue
            reserve = self.finish_reserve()
            if self.left() <= reserve and (self.cost >= spend_cap or self.refusal_seen() or reserve > self.tail()):
                self.trace.append({'kind': 'reserve_stop', 'left': self.left(), 'reserve': reserve})
                break
            self.trim()
            self.loop_calls += 1
            mending = self.repairing()
            if mending and self.cost >= spend_cap:
                self.trace.append({'kind': 'repair_turn', 'cost': round(self.cost, 5), 'until': round(self.repair_cap or 0, 5)})
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered(), None, mending)
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
            if self.asker.pending() and self.roles.get('ask') and (self.msgs_free() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.left() > 8) and self.room_for(MODEL_SECONDS + 4):
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

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Marlo(problem_data)
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
