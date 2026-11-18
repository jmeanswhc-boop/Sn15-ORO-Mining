import json
import random
import re
import time
from typing import Any, Dict, List, Optional

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

MODEL_CLIENT = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
ENV_CLIENT = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
SHORT_CLIENTS = {}

TURN_CAP = 42
CALL_CAP = 60
SOFT_SPEND = 0.055
HARD_SPEND = 0.078
HEAVY_ROOM = 0.02
BRIEF_DEADLINE = 120
OUTAGE_PAUSE = 10
OUTAGE_CUTOFF = 150
RETRY_WAITS = (4, 10, 20, 35)
WAIT_BUDGET = 45
EPISODE_LEN = 300
EPISODE_MARGIN = 20
CALL_FLOOR = 6
ASK_DEADLINE = 165
LATE_DEADLINE = 190
WIND_DOWN = 200
HUNT_DEADLINE = 200
HUNT_GRACE = 25
QUESTION_DEADLINE = 210
LAST_TURN = 255
EMPTY_CART_ROOM = 12
STORE_PAUSE = 0.25
OPENING_PAUSE = 0.5
ROW_CLIP = 450
TEXT_CLIP = 3000
OBS_CAP = 15000
HISTORY_CAP = 150000
PAGE_ATTR_CLIP = 700
PAGE_REST_CLIP = 300
TITLE_LIMIT = 200
KEY_TABLE_SIZE = 18
CRASH_RETRIES = 3
OPEN_MIN_STEPS = 20
OPEN_MIN_ITEMS = 2
OPEN_LATER_STEPS = 2
OPEN_ROWS_PER_ITEM = 6
OPEN_WIDEN_SHARE = 5
OPEN_ROWS = 5
SHORTER_RESERVE = 8
WIDEN_PER_ITEM = 6
AUTO_PAGES = 6
HUNT_CAP = 24
HUNT_RESERVE = 10
HUNT_ROUNDS = 2
HUNT_IDLE = 4
HUNT_YIELD = 5
HUNT_WHOLE = 49
HUNT_VALUE_LEN = 30
HUNT_COMMON = 0.6
FIT_WORDS = 4
FIT_OPENS = 4
FIT_PAGES = 2
CHEAPER_SHOWN = 4
CHEAPER_OPENED = 3
VALUE_ROOM = 12
VALUE_RESERVE = 8
PRICE_CHANCE = 0.05
WALK_RESERVE = 10
WALK_PER_LINE = 4
ASK_WAIT_TURNS = 2
SCARCE_MESSAGES = 2
LIST_ONE = 12
LIST_MANY = 8
RAMP_ONE = (1, 3)
RAMP_MANY = (3, 3)
MIXED_LABELS = 3
LOOKALIKE_KEYS = 2
QUIET_WORD_LEN = 4
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
GATE_ROUNDS = 3
TRAIL_ROOM = 15
EARLY_ROUNDS = 3
CHEAP_SETS = 2
SERIES_HUNTED = 3
SERIES_ROWS = 3
SERIES_BREADTH = 12
RECORD_MIN = 3
NOTE_MIN = 5
FIELD_SHARE = 0.15
PAIR_SHARE = 0.7
BLOCK_NAME = 60
BLOCK_VALUE = 40
TAGGED_LIMIT = 20000

PAIR_MARKS = (": ", " = ", "=", " -> ", " - ")
FIELD_MARKS = (" | ", "\n", " || ", " ; ", "; ", "\t", " // ", " · ", " • ", "|", ";", ", ")

V1_MSG = re.compile(r"(?:^|_)(?:texts?|body|messages?|notes?|contents?|questions?|msg|say)(?:_|$)", re.I)
V2_NONMSG = re.compile(r"(?:^|_)(?:ids?|type|format|code|lang|language|kind)$", re.I)
V3_OPEN_SKIP = re.compile(r"\b(?:not|no|other\s+than|unlike|at\s+most|at\s+least|under|over|below|above|less|more|between|within|up\s+to|or|same|listing)\b", re.I)
V4_OUT_WHOLE = re.compile(r"(?:no|none|false|0|n)", re.I)
V4_OUT_ANY = re.compile(r"out[ _-]?of[ _-]?stock|sold[ _-]?out|unavailable|not[ _-]?available", re.I)
V5_HOW = re.compile(r"\b(?:one|each|single|first|once|several|many|much|all|lot|too|list|separate\w*|split|short\w*|time|turn|questions?|ask\w*)\b", re.I)
V6_CAP = re.compile(r"(?:^|_)(?:cap|max|maximum|ceiling|under|up_to|upto)(?:_|$)", re.I)
V7_NONPRICE = re.compile(r"result|row|page|count|size|num|item|step", re.I)
V8_PRICE = re.compile(r"price|cost|budget|spend|money|amount", re.I)
V9_LOWER = re.compile(r"(?:^|_)(?:min|minimum|low|lowest|least|from|start|offset|floor)(?:_|$)", re.I)
V10_SHELF = re.compile(r"section|department|categor|label|shelf|aisle|(?:^|(?<!age)_)group(?:_(?:name|label))?$", re.I)
V11_MODE = re.compile(r"field|filter|facet|brows|categor|list|structured", re.I)
V12_AMOUNT = r"\d[\d.,]*\s*[A-Za-z\"']{1,12}"
V12_WHOLE = re.compile(V12_AMOUNT)
V13_FLOOR = re.compile(r"\b(?:at\s+least|no\s+less\s+than|no\s+fewer\s+than|min|minimum)\b\s+(" + V12_AMOUNT + r")\s*$", re.I)
V14_QUERY = re.compile(r"(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword", re.I)
V15_NONPROP = re.compile(r"^\W*(?:options?|variants?|models?|model\W+numbers?|part\W+numbers?|item\W+(?:numbers?|ids?|codes?)|(?:\w+\W+)?(?:ids?|skus?|asin|upc|ean|barcodes?)|codes?|warrant\w*(?:\W+\w+)?|url|links?|images?|photos?)\W*$", re.I)
V15_MODELS_OK = re.compile(r"\w[ _]models?\b")
V16_KINDWORD = re.compile(r"\b(?:types?|styles?|kinds?|shapes?)\b")
V16_CATCHALL = re.compile(r"\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|extras?|includ\w*|accessor\w*|misc\w*)\b")
V17_FIT = r"[A-Z][A-Za-z]{0,5}[ -]?(?:\d{3,6}|\d{1,5}[xX]{1,3})"
V17_RE = re.compile(r"\b" + V17_FIT + r"\b")
V18_PART = r"[A-Za-z]{1,5}\d{0,3}-?\d{3,7}"
V19_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_ ./&'()\-]{0,38}")
V20_BULLET = re.compile(r"^[ \t]*(?:[-*•·]|\d{1,3}[.)])[ \t]+", re.M)
V21_PAIRKEY = re.compile(r"name|key|label|attr|prop|field|spec|title|^k$", re.I)
V22_ID = re.compile(r"(?i:(?:^|[ _-])(?:id|idx|sku|pid|ref|asin|ident\w*)(?=$|[ _-]))|^[Ii][Dd][A-Z0-9]|Ident|(?i:(?<![A-Za-z])(?!(?:paid|valid|grid|solid|rigid|fluid|liquid|hybrid|rapid|void|acid|humid|android|said|laid)$)[A-Za-z]*id$)")
V23_FAIL = re.compile(r"error|fail|(?<!de)fault|trouble(?!shoot)|denied|refus|invalid", re.I)
V24_PLACEHOLDER = re.compile(r"(?:none|null|n/a|unknown)", re.I)

YEAR_CODE = re.compile(r"[A-Z][a-z]* (?:19|20)\d\d")
OWN_CODE = re.compile(r"[\[(]\s*(?:part\s*)?(" + V18_PART + r")\s*[\])]", re.I)
MODEL_CODE = re.compile(r"\bmodel\s*(?:(?:no\.|no|number|#)\s*)?[:.]?\s*(?!(?:no|number)\b)(" + V17_FIT + r")\b", re.I)
ALSO_SOLD = re.compile(r"\b(?:also\s+sold\s+as|also\s+known\s+as|sold\s+also\s+as)([^.;]*)", re.I)
WORKS_WITH = re.compile(r"(?<!not )(?<!never )\b(?:works\s+with|compatible\s+with|fits)\s*:?\s+([^.;]*)", re.I)
EXCEPT_CUT = re.compile(r"\b(?:except|excluding|but\s+not)\b", re.I)
PART_ALSO_TAIL = re.compile(r"part\s+" + V18_PART + r"\s+also\s*$", re.I)
ALSO_FITS = re.compile(r"\bpart\s+(" + V18_PART + r")\s+also\s+fits:?\s*([^.;]*)", re.I)
POINTER = re.compile(r"\b(?:more|details|see|read)\s+(?:on|at|under|about)?\s*(?:the\s+)?part\s+(" + V18_PART + r")", re.I)
OWN_SERIES = re.compile(r"\b([A-Za-z]{1,5}\d{1,3})-\d{2,7}\b")
SERIES_EXCL = re.compile(r"\b(?:except|excluding|but\s+not|not\s+for|not\s+compatible\s+with|does\s+not\s+fit|does\s+not\s+take)\b[^.;]*", re.I)
NAMED_SERIES = re.compile(r"\b[A-Za-z]{1,5}\d{1,3}\b")


def find_codes(text):
    out = []
    for m in V17_RE.finditer(text or ""):
        c = m.group(0)
        if YEAR_CODE.fullmatch(c):
            continue
        n = re.sub(r"[ -]+", " ", c).strip().lower()
        if n not in out:
            out.append(n)
    return out


def code_covers(part_code, device_code):
    p = re.sub(r"[ -]", "", str(part_code)).lower()
    d = re.sub(r"[ -]", "", str(device_code)).lower()
    m = re.search(r"(\d)(x+)$", p)
    if m:
        head = p[:m.start(2)]
        return d.startswith(head) and len(d) == len(p)
    return p == d


def fit_marks(text):
    text = text or ""
    own = ""
    m = OWN_CODE.search(text)
    if m:
        own = m.group(1).upper()
    models = []
    for m in MODEL_CODE.finditer(text):
        for c in find_codes(m.group(1)):
            if c not in models:
                models.append(c)
    for m in ALSO_SOLD.finditer(text):
        for c in find_codes(m.group(1)):
            if c not in models:
                models.append(c)
    works = []
    barred = []
    for m in WORKS_WITH.finditer(text):
        before = text[max(0, m.start() - 40):m.start()]
        if PART_ALSO_TAIL.search(before):
            continue
        span = m.group(1)
        cut = EXCEPT_CUT.search(span)
        head = span[:cut.start()] if cut else span
        tail = span[cut.start():] if cut else ""
        for c in find_codes(head):
            if c not in works:
                works.append(c)
        for c in find_codes(tail):
            if c not in barred:
                barred.append(c)
    notes = []
    for m in ALSO_FITS.finditer(text):
        notes.append((m.group(1).upper(), find_codes(m.group(2))))
    pointer = ""
    m = POINTER.search(text)
    if m:
        pointer = m.group(1).upper()
    own_series = sorted(set(x.upper() for x in OWN_SERIES.findall(text)))
    rest = OWN_SERIES.sub(" ", text)
    rest = SERIES_EXCL.sub(" ", rest)
    named = sorted(set(x.upper() for x in NAMED_SERIES.findall(rest)))
    return {"own": own, "models": models, "works": works, "barred": barred, "notes": notes,
            "pointer": pointer, "series": own_series, "named": named}


GUIDE = """You are Cairn, a careful purchaser who serves one single shopper inside a web shop. Your only way to act is to call one of the shop's tools. Grading is all or nothing: a score of 1 requires a finish that is exactly correct, and anything else scores 0. A correct purchase must hold every requested item, each one in exactly the quantity asked, each one satisfying every requirement still in force at the moment checkout happens (changes that arrived later count), and it must cost no more than the budget. The single other correct finish: when the shop truly holds nothing that satisfies everything, you must instead close the episode through the ending (no-purchase) tool, carried out the way the request specifies.

HOW THE SHOP OPERATES
- Every tool call spends one step and the supply of steps is finite; thinking is free. Every shop has its own tool names, parameter names and result layouts, so you must study the schema and the description of every tool and pass parameter names and permitted values exactly as they are given. Never guess a parameter name or a value.
- A search may be restricted: it may filter by category, a keyword mode may read only the first words of a query, a page may hold only a handful of rows, and reaching the next page may need a token or an offset. Set the total the shop reports against the number of rows you have seen; if the answer hinges on seeing every row, fetch the further pages, passing back unchanged whatever token or offset came with the result. Keep keyword queries brief and lead with the value: the wanted value first, then the product noun. A shop's search may match attribute values and seller names as well as titles, and may demand that every query word be present, so a handful of required values together with the noun can return almost the entire fitting set. A search by a part number or a model code can come back empty; a single attempt shows whether this shop's search finds codes at all. When a filtered search returns zero rows, the usual cause is a wrong filter value or a wrong filter parameter: try again with the label copied letter for letter as the rows of a result show it, or switch to another filter parameter.
- Rows in search results are summaries only. Shops split a listing's record across views differently: one view may expose the attributes (also called properties), another only the offer and a description, and either kind may appear in the side-by-side compare view. Each property requirement is judged on those attributes, so you must inspect the attributes of every listing you might purchase. A single call that shows several listings at once saves steps, provided it shows what is required. Never reopen a listing whose state has not changed since you read it.
- Asking the shopper is the job of Cairn's interviewer: you never send the shopper a message yourself. Replies appear in your conversation and the ledger records them.
- A request may also demand intermediate steps: obtaining an approval, sending a message, verifying or reading something beforehand. Each appears in the ledger. Those that fall due right ahead of the order or of a no-purchase ending are carried out by Cairn itself at the moment you call the checkout or the ending tool, so never hold off waiting for them and never decline on their account. The remaining steps you perform yourself with the shop's tools; questions about products are the exception and go through the interviewer.

WHAT COUNTS AS FITTING
- The kind of product is settled by the listing's category label: only a label that matches the requested item in its entirety qualifies; an adjacent label never does.
- A property requirement is satisfied when the attribute the listing has for that property gives the value. How the key is written does not matter: spacing, underscore separators, prefixes or camel-cased joins. When the attribute is a list split by commas or slashes, one matching element is enough. Matching works word by word; differences of letter case, of singular and plural, and ordinary synonyms are tolerated. Containment counts: a value that includes all the words of the required value meets it despite extra words (a colour with a shade word in front of it satisfies the plain colour; a fit word with an intensifier before it and a noun after it satisfies the plain fit word). A value made of other words fails: an abbreviation or a code is not the word it abbreviates, and two colour names that share one word but differ in the other are different values (dark green is not forest green). Numbers: convert units first, then compare; a limit includes its boundary; a value given as a range must sit entirely within the limit.
- An attribute keyed by the requirement's property (that exact name or an ordinary synonym of it) is decisive: if it states another value, the listing fails, whatever the title or any other key says. Exception: when a key prefixes the property name with the item's name (for a bag, a key meaning the bag's shape), its value counts even if the plain property key states otherwise. A key about another property, even a closely related one, settles nothing for this requirement (the colour of a handle tells nothing about the colour of the body). After listings are opened, Cairn appends evidence lines, marked [evidence], naming for each requirement the attribute keys whose name matches it.
- When a listing has no attribute at all for the property, the value sought still counts when it appears under some other attribute (a styles list or a features list that contains it), or in its description or title. It does not count when the figure is a different kind of quantity (weight is no substitute for capacity), or when a bare number appears with no text saying what it measures.
- When the requirement's value is written into the category label (a season or a target audience inside the label), the category satisfies it. Conversely, a listing filed under the label that was asked for counts as a product of the requested kind, even when its description or title mentions something different; such a listing is judged solely on its requirements, its price and its stock.
- A requirement may refer to another listing instead of giving its value, for example "the same X as the item <shop> sells with <property>..." or "within N units of the value shown on that listing's page". Resolve such a requirement first: locate the referenced listing (search on its words, brand, shop or category) and open its whole page with the single-item tool or the side-by-side compare tool; a row from a search does not suffice. If the opened page sends you on to one more listing that holds the property, locate and open that listing by the same method, and so on: the final page in the chain supplies the value. If the requirement wants whatever goes with the referenced value (its partner or counterpart) and that page says what its own value is paired with, that paired value becomes the requirement. An order is valid only if every page along the chain was opened. If the requirement accepts anything other than the referenced listing's value, it removes only the listings sharing exactly that value, and that rule alone must never cause a decline.
- If a listing you intend to buy now points to a further page with more detail about itself, check that page before the purchase.
- If one requested item has to suit another, as an accessory suits a device, pick both jointly, based on the listings' own wording: a compatibility list, or a series or model code that appears on both pages. If no accessory that meets its own requirements is compatible with the chosen device, change to a different device that satisfies its own requirements and that one of those accessories does fit. Never decline while that device switch has not been tried.
- Exclusions must be honoured, and so must closed lists. Wishes (softly worded preferences such as "ideally" or "it would be nice") serve only to break ties.
- Sold-out listings must never be bought. The total (each line's price times its quantity, added across all lines) must not go over the budget; landing exactly on it is allowed.

CHOOSING
- If the request wants the cheapest choice (the smallest price, the lowest total), the order must contain the listing with the lowest price out of ALL fitting listings. Establish the complete fitting set by paging through it (rows often arrive unsorted by price; Cairn fetches further pages of short result lists itself), examine the candidates from the cheapest upward, and pick the earliest one that fits fully. If a search tool accepts a maximum price, use it: list that category with the cap set below what your present best pick costs, so that only the cheaper ones still need checking.
- If the request wants the greatest or smallest amount of a measure, compare every fitting listing on that measure and take the extreme one; a listing silent on the measure is never the winner.
- Otherwise any listing that satisfies everything is correct: do not burn steps looking for a better one.
- Put each item in the cart the moment an opened listing satisfies all of that item's requirements (for a price-ranked or measure-ranked pick: only after the fitting set has been compared), then go on to the next item. A shop often sells out a line shortly after it enters the cart, and carting early keeps steps in hand for a replacement.
- When several distinct items are requested, each needs its own cart line, every line at that item's requested quantity.

CHANGES DURING THE WORK
- Both the shopper and the shop may change the task: a fresh budget, an added or removed requirement, a new price, a listing gone out of stock. Such changes show up inside tool results, often on a checkout attempt as a refusal. A note is valid whichever tool's result carries it. The ledger shows the requirements in force now. A refusal does not end the episode: apply all changes, correct the cart and try checkout again.
- Prices and stock may also shift without any note. When you call checkout, Cairn re-checks the cart lines and views the cart itself, so do not spend a turn on either.

FINISHING
- Make the cart contain only the chosen lines, exactly those, removing everything else, then send checkout as a call by itself, never together with other calls in one turn.
- A no-purchase ending is allowed only once a genuine search has found that no listing satisfies everything. Being low on steps proves no such thing: in that situation check out with the best cart you hold. The ending follows the request's own wording; when a remark has to go with it, keep the remark brief and name no listing.
- Every reply must include one or more tool calls. Keep text brief: the known facts, then the next action."""


BRIEF_PROMPT = """You set up a buying job that an agent will carry out. You receive the shopper's text and the list of the store's tools. Answer with exactly ONE JSON object, compact and unindented, with no text around it, holding these keys:

"items": a list with one entry for each distinct product to be bought. Each entry is an object with four fields: "name": what sort of product it is, worded as the request words it; "shelf": if some tool exposes the store's category labels, the matching label copied exactly; else a category the request gives the product (a product wanted "in" a named department takes that department's name); else ""; "quantity": the number of units to buy of that single listing, an integer, 1 when no number is stated; pairs and sets that come as one listing are quantity 1; "musts": the list of the product's firm requirements, each written as property: constraint, stating the precise value, bound, excluded value or permitted set of values.
Name the property inside a must with the very word the request uses for it: wording that the product is meant for some kind of occasion gives the occasion property with that kind as its value; a measure placed before a part of the product (a twelve-inch blade) gives that part's length, with number and unit. Never trade the property for a neighbouring one.
A reference requirement, whose value has to come from a different listing, is written as ONE must that carries the other listing's full description, for example "colour: different from the colour of the backpack that ShopName sells with a laptop sleeve". All the words describing the other listing (its shop, its product kind, any trailing "with" clause of properties and values) stay inside that single must; none of them ever becomes a must of this product.
Price, budget, stock and quantity never go into musts. Narrative around the purchase (the intended user, the place the product goes, earlier trouble, the occasion that prompted the purchase) is context and never a must, except where it explicitly gives some property's value or bound. Words that are part of the product's own name, such as a season or a target group mentioned with the product, stay in "name" and never become musts.
Exception: when the request treats a purpose or an event as a property of the product, it IS a requirement on that very property (for example "occasion: business"), unless the store's category label already names it. This covers something said to be for a kind of wear, something said to be made for a kind of event, a product followed by a separate clause saying it is for an event, and a plain statement that the occasion must be a given value. These stay context: one specific person's own event (wearing it at their own wedding), the recipient or place the purchase is meant for (a present for a co-worker), and how it should sit, feel or look when used.
When one phrase names products joined by "and" (a single "I want" clause with a single quantity, often the title of a single store department), that is ONE item. Split into separate items only if every product gets its own quantity or its own requirements.

"budget": the shopper's ceiling for the whole spend as a number, in whatever unit the request uses; null if absent.
"budget_unit": the unit written out (for example "cents"); null if absent.
"budget_scope": "total" if the cap applies to the whole order, "each" if it applies per item; null if there is no cap.
"in_stock_only": true by default; false only when the request plainly allows unavailable listings.
"pick": "cheapest" when the request wants the lowest price or the least total cost; "most" or "least" when it wants the maximum or minimum of some measure; "any" otherwise. Saying money is short does not make a pick rule.
"pick_measure": for "most" or "least", the quantity pushed to its extreme; else "".
"messages": the number of shopper messages the request allows the agent, an integer; null if unstated.
"steps": the request's allowance of steps (it may call them moves or tool calls), an integer; else null.
"lookups": exact copies of the musts whose value must first be fetched from a listing the request refers to; [] if none.
"rules": every instruction in the request about the working method or the way to finish, each at most 12 words.
"wishes": nice-to-haves, preferences the shopper voices without requiring them.
"ask_first": whatever the request explicitly wants confirmed with the shopper before choosing; [] if none.
"process": the duties the request imposes beyond finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, confirming a point with the shopper, reading something before acting, using a specific tool). Each entry is an object: "step": the duty in at most 15 words; "when": exactly one of "choose" (before any product is chosen), "order" (just before checking out), "end" (just before finishing with no purchase); "messages": the number of shopper messages the duty uses, 0 if none. [] if there is no such duty. Item requirements, the budget, stock, the message and step limits, and what to do if nothing fits are NOT duties.
"quiet_decline": true if the request forbids suggesting or naming alternatives when nothing fits; false if it wants alternatives; null if it says nothing.
"fits": one entry for every requested item that, by the request, has to work with, fit or suit some OTHER requested item (that is, whichever listing the agent picks for that other item), as {"item": the dependent item's number, "with": the other item's number}; [] if none.
"roles": an object mapping each of these nine jobs to the exact tool name, or null when no tool does it: "search"; "open" (shows the properties or attributes of a single listing; of two tools that each show a single listing, take the one whose description offers specifications, properties or attributes); "offer" (shows one listing's live offer such as price and stock, only if a tool other than the "open" one does that); "compare" (shows several listings completely); "ask" (messages the shopper); "cart" (changes the cart); "cart_view" (shows the cart); "buy" (performs the checkout); "refuse" (finishes with no purchase).
"search_words": {"tool": the tool that takes free-text words to search listings (if one tool browses a category and another takes words, choose the word-taking one), "field": the parameter of that tool that receives the words, "args": constant arguments needed to switch it to word search (a mode tool needs its mode value here), else {}}.
"buy_args": constant arguments the buy tool needs to perform the checkout (a mode tool needs its mode value here); else {}.
"refuse_args": constant arguments the refuse tool needs to finish with no purchase; else {}."""

PLAN_PROMPT = """You pick the questions an agent will put to the person it shops for. Every shopper keeps a small number of firm, unwritten preferences, and each one is about a single property of a single requested product. Asking reveals a preference only if the question names exactly that property. Where the shopper has no opinion the answer is "don't mind"; where the property never crossed the shopper's mind the answer is puzzled. So each property put in a message spends a share of that message.

You receive the request text; the items, each with the requirements the request already fixes; and a numbered table of the attribute keys found on opened listings in each item's category, giving per key how many of those listings carry it and some sample values. Careful: the unwritten preference can concern a property that none of the listed keys covers.

Choose in this order:
0. Anything the request wants checked with the shopper (under ASK FIRST, if present), each named after the property involved.
1. Hinted properties. A request's story often hints at an unwritten preference while leaving its value unsaid: a sentence about a matter the shopper cannot judge or hands over to the agent; about a person whose taste is decisive; about how the product looks or the surroundings it belongs to; about the intended wearer or user; about its contents. For each such sentence give the single property it hints at, with its item, in the story's own order. Whenever a key listed for the same item matches the hint, use that key: a hint about the wearer maps to the gender or audience key; a hint about the product's looks maps to whichever colour, finish or pattern key is carried by the largest share of that item's listings. A hinted property that no listing of this item carries ranks below that item's three most widely carried keys, so never above fourth.
2. Then the listed keys in rank order. Prefer the properties shoppers for that sort of product decide on first (commonly the type or style, the form or fit, the dimensions, size or capacity, the material, the pattern or colour: whichever ones that sort of product is chosen by), and keys present on a majority of that item's listings. Once at least five listings have been read, avoid keys found on only one or two listings.

Skip, for each item: any property the request has already settled for that item (a requirement names it); but a key whose values merely echo a required value is not settled and may still be asked (a pattern key whose values show the decoration the request demands). Also skip a further name for a property already chosen; a key in which each value bundles several purchase options together; identifiers; model names; warranty and any other logistics. Do not skip pack sizes, per-pack quantities and counts: they are ordinary properties. A key marked [not the stated ...] is a different property from the requirement it mentions, however many words the two names share.

Rank every item separately: its hinted properties before its listed keys.

"label" is the plain name of the property. From a listed key take every word of the key in plain spelling: a key whose words run together with a capital at each join, or are joined by underscores, is written as separate lower-case words, and it is never cut down to its last word (the label for a lid's colour is "lid colour", never just "colour"; for a rim's finish it is "rim finish", never only "finish"). Just the name: no sample values, no bracketed text, never a second property.

Pick at most COUNT properties per item, strongest first. Answer with ONE JSON object only: {"ask": [{"item": item number, "label": "...", "hint": "at most eight words from the story that led to this pick, or empty"}]}. Within each item, list the picks in rank order."""

REPLY_PROMPT = """An agent has put some questions to a shopper. Read every answer. Answer with ONE JSON object only, nothing besides it, with five keys:
"needs": a list of {"item": item number, "need": "property: constraint", "points": true if the value has to be taken from a different listing, else false};
"drop": a list of {"item": item number, "property": a requirement the shopper has now given up};
"free": a list of {"item": item number, "property": a property the shopper does not mind about};
"crowded": a list of the question numbers whose answer objects that the message covered too many subjects, or wants the subjects asked one at a time (not answers that only say the question was not understood or ask for plainer wording);
"budget": a revised overall spending ceiling as a number, else null.

"needs" covers every firm preference any answer expresses, with its precise value or bound and its unit (for example "length: at least 60 cm", "shape: round", "material: not plastic"). Name the need after the property the answer itself mentions, even when that differs from the property the question was about. Several answers about one property give one need. Copy the value word for word from the answer, nouns and units included (a minimum number of shelves keeps the noun: "shelves: at least 4 shelves", not only the number). Keep the value to what a listing might state: it starts at the value wanted (or at the bound word before it), ends where the value ends, and leaves out the answer's framing (from "matching navy" keep "navy"; from "it's settled, oval" keep "oval"; from "walnut and nothing else will do" keep "walnut"; from "equal to at least 2 litres" keep "at least 2 litres").
An answer may supply the value by reference to another listing: a value like, unlike or close to the value on a product from some shop, the product being identified by a few of its properties. That makes ONE single need for the property the answer names, with "points": true. Its value is the exact text of the answer from the comparison word through to where that description ends, together with anything said about a listing that page might name. Never shorten or reword it. Properties that describe the other product are never needs of this item.
"free": for each question, the property the shopper declared indifference to.
Folksy or indirect phrasing that still names a value does state that value. Phrasing that refuses to negotiate a value (it is not to be traded away, given up or changed) states that value as wanted. Write a need as "not" a value only if the answer turns that very value down."""


NOTE_PROMPT = """While an agent was working, the store displayed the notice below; its author may be the store itself or the shopper. Reply with one JSON object, nothing besides it, with these keys in this order:
"kind": "change" if the notice alters the job at hand (a budget, requirements, some price, availability, or refusing the order); "error" if it only says some call was badly formed or turned down; "other" in every other case;
"budget": the new limit on total spend as a number in the unit the request used, else null;
"add": new requirements, a list of {"item": item number, "need": "property: constraint"};
"drop": requirements no longer wanted, a list of {"item": item number, "need": the wording of the withdrawn requirement};
"gone": the ids of listings the notice calls unavailable;
"sold_out": true if the notice says a picked listing has vanished, sold out or can no longer be had, even when it names no id;
"price_moved": true if it reports some price moving;
"summary": one brief sentence.
A requirement replaced by another must appear twice: as a drop of the old wording and as an add of the new one. Report every change the notice contains, not only the first."""

CART_PROMPT = """You are the final gate ahead of checkout. From the records below, decide whether this cart is exactly right.
Reading rules, for a requirement R on property P with value V: R holds when the listing's field for P states V; the key K may be spelled any way, a single member of a list is enough, letter case, singular or plural and simple synonyms are fine, a value that includes the words of V counts (a shade word before the wanted colour satisfies that colour), units are converted, a limit includes its own boundary, and a range has to fall wholly inside. A field whose key names P itself decides: if it states something other than V, the listing fails; except that a value filed under a key that puts the item's own kind before P (on a backpack, a key for the backpack's colour) still counts even though the bare key for P disagrees. A key for another property, however closely related, decides nothing (a strap's colour says nothing about the main body's colour). Where no field for P exists at all, V stated under some other field, or in the title or description, is enough; but not a figure of a different sort (a weight figure is not a capacity), and not a lone figure whose text never says what it measures. If the category label itself states R, being in that category satisfies R. The category label has to match the whole requested item, not just part of it. A requirement to differ from another listing's value is taken as met unless the records show that very value.
Cart rules: every requested item must have its own cart line with the quantity asked; the cart must hold nothing beyond those lines; no line may be shown as unavailable or out of stock; and the sum of each line's price times its quantity must be no higher than the budget.
When there are several records of one listing, the newest governs.
If other listings are shown, weigh them too: if one of them plainly satisfies every requirement and beats the cart's choice under the request's pick rule (cheapest, most or least), this cart is wrong.
Reply with one JSON object only, nothing besides it: {"ok": true or false, "problems": [short concrete faults], "better": [ids of listings to put in place of some cart line, strongest first]}.
Set "ok" to false only for a fault the records plainly show, or when some requirement is addressed by no record."""

READING_RULES = """for a requirement R on property P with value V: a field of the listing settles P whenever its key K names P (K may be spelled any way, one member of a list is enough, synonyms are fine, units may be converted, a value that includes the words of V is fine, as a shade word before the wanted colour satisfies that colour, and a limit includes its own boundary); a value filed under a key that puts the item's own kind before P (on a backpack, a key for the backpack's colour) still counts even though the bare key for P disagrees; a key for another property, however closely related, decides nothing (a strap's colour says nothing about the main body's colour); with no field naming P, V stated under some other field, or in the title or description, is enough, but a lone figure that never says what it measures is worth nothing; if the category label states R, that category meets R; the category label has to match the item; a requirement to differ from another listing's value is taken as met unless the records show that very value; a listing that is out of stock, or that goes over the budget, is not a fit."""

AUDIT_PROMPT = ("""The agent means to stop without buying, believing nothing in the store fits. Check the records shown below against the requirements, using these reading rules: """
                + READING_RULES + """
A ROW record comes only from a search row: nobody has read its page (its fields), so on any property the row itself does not show, it may be judged neither a pass nor a fail.
Reply with one JSON object only: {"fits": {"item number": [ids that seem to satisfy all requirements, lowest price first, 4 at most]}, "open": [ids of ROW listings of a requested kind, within budget, whose page might prove a fit, likeliest first, 6 at most], "reason": "one sentence"}.""")

UNMIX_PROMPT = """Someone turned a shopping request into firm requirements for one product. Some requirements take their value from a different listing the request refers to, and the words that pick out that listing (the kind of product it is, the shop it comes from, the properties a "with" phrase after it names) are sometimes mistaken for requirements on the product being bought. Going by the request's own phrasing, say which entries of the list headed OTHER REQUIREMENTS below do nothing but describe the referenced listing. Reply with one JSON object only: {"describe_other": [those requirements, each copied word for word]}, or an empty list when all of them are genuine requirements of the purchase."""

CHAIN_PROMPT = """The agent is near checkout. Some requirements take their value from a second listing instead of stating it, and one listing's page may send the reader on to yet another listing for that same property. An order only counts once the agent has fully opened every listing along every such chain.
Trace the chain of every requirement, starting from its own description, link after link, through what the pages themselves say, using the records after this text. OPENED means the agent opened that page; ROW means the page is unread and the listing is known only from a search row or through its fields.
Reply with one JSON object only: {"chains": [{"need": the requirement copied word for word, "links": [{"id": that listing's id if some record is this link, else "", "about": how this link is described, by the requirement or by the page before it, "find": only when the link is not yet located, search words made of its shop plus the values the description gives}], "done": true if the final link is OPENED and its page itself states the property's value, true as well when the requirement wants a value that differs from that one or is unlike it, because the value on that page is the yardstick, "value": the property's value as the final page states it, or, if the requirement wants whatever pairs with that value and the page names its partner, that partner, else ""}]}.
A record counts as a link only if it fits the link's entire description: shop or brand, product kind AND every property value the description lists (a description of a multipack holding some count excludes a listing from that shop with the same count of another unit and no multipack). If no record fits all of it, the link is unlocated, even when that shop carries other listings of the same kind. For an unlocated link, "find" gives a few search words: the shop's name plus the values in the description.
Give the links of every chain in order and end the list at the first link that is unlocated or has no OPENED page, because what that page says is not known yet."""

GATE_PROMPT = """The agent will shortly STAGE. The request wants the steps listed below performed on the way. Using the RECORD (the numbered lines recording the agent's actions), decide for each step owed before this STAGE whether it has been completed for the cart as it is now.
A step is complete only if some RECORD line records its performance; for a step about the cart or the order, that same line must also come after the latest cart change, and the reply recorded on that same line must grant it. Questions about a product's properties never perform a step.
For a step still missing, name the single store call that performs it now: pick a tool from the TOOLS list and supply its arguments. A shopper message uses the messaging tool and is one brief sentence or question about that step alone, plainly worded, mentioning no product property or value beyond what the step needs.
Give no call if every owed step is complete, or if the remaining messages or steps cannot achieve the missing one.
Reply with one JSON object only: {"steps": [{"step": the step, "done": true or false, "lines": [numbers of the RECORD lines that prove it]}], "call": {"tool": name, "args": {...}} or null, "hold": a change some reply asks for before the agent goes on, else "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}."""

SWAP_PROMPT = ("""A listing in the agent's cart sold out a moment ago. Pick substitutes for it from the other records shown below, using these reading rules: """
               + READING_RULES + """
Reply with one JSON object only: {"fits": [{"id": a listing id, "price": its price as a number, the way the record states it}, covering every listing that seems to satisfy all requirements of the lost line's item, 6 at most], "reason": "one sentence"}.""")

SHAPE_PROMPT = """A store tool produced the output shown below. Label its parts using the exact keys found in that result and in the tool's parameters. Reply with one JSON object only: {"rows": the key holding the array of listings, "count": the key holding the total number of matches (not the number displayed) or null, "next": the key holding the value that requests the next page or null, "next_arg": the tool parameter that accepts that value or null, "price": the key, within a row, of the listing's price or null, "stock": the key, within a row, telling whether it can be bought right now or null}."""


BOOL_WORD = re.compile(r"true|false", re.I)
EMPTY_WORD = re.compile(r"none|null|", re.I)
MONEY_RE = re.compile(r"[$€£]?\s?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
LEAD_ZERO = re.compile(r"0\d")
LOOSE_PAIR = re.compile(r"['\"]?([A-Za-z][A-Za-z0-9_ ./\-]{0,38})['\"]?\s*[:=]\s*['\"]?([^,'\"}\]]*)")
BLOCK_CUT = re.compile(r";\s|;$|\s\|\s|\n")
NV_PART = re.compile(r"[^=:]{1,40}=[^=]{0,200}")
RULE_LINE = re.compile(r"[\s|:+=\-]*")
NUM_HEAD = re.compile(r"[\d.,$]+")
BLANK_CUT = re.compile(r"\n\s*\n")
TAG_ANY = re.compile(r"<[A-Za-z/][^<>]{0,40}>")
TAG_INLINE = re.compile(r"</?(?:b|i|em|strong|span|a|u|small)\b[^<>]{0,40}>", re.I)
TAG_CELL = re.compile(r"</t[dh]>\s*<t[dh][^<>]{0,40}>", re.I)
TAG_RUN = re.compile(r"\s*(?:<[^<>]{1,40}>\s*)+")
COL_GAP = re.compile(r" {2,}|\t")
COMMA_CUT = re.compile(r"(?<!\d),|,(?!\d)")
GLUE_RE = re.compile(r"(\d)\s+([a-z]{1,3})(?![a-z0-9])")
LD_RUN = re.compile(r"[a-z0-9]+")
TOK1 = re.compile(r"[a-z0-9]{3,}")
TOK2 = re.compile(r"(?<![a-z0-9])(?:\d+[a-z]|\d*[a-z][a-z0-9]+)(?![a-z0-9])")
TOK3 = re.compile(r"\d+")
FENCE = re.compile(r"```[A-Za-z]*\s*(.*?)\s*```", re.S)
CAPSPLIT = re.compile(r"([a-z0-9])([A-Z])")
SEPRUN = re.compile(r"[\s_.\-]+")
ES_END = re.compile(r"(?:ch|sh|ss|x|z)es$")
STOP_WORD = re.compile(r"and|for|the|with")
OUR_END = re.compile(r"our$")
SIGNED_DEC = re.compile(r"\s*-?\d+(?:\.\d+)?\s*")
COUNT_TEXT = re.compile(r"\s*(\d+)(?:\.0*)?\s*")
HAS_DIGIT = re.compile(r"\d")
FILLER_WORD = "question"
FIRST_SPELLING = {}


def is_failure(v):
    if not isinstance(v, dict) or not (1 <= len(v) <= 3):
        return False
    for k, x in v.items():
        if V23_FAIL.search(str(k)) and x:
            return True
    if len(v) == 1:
        x = list(v.values())[0]
        if isinstance(x, str) and 1 <= len(x) <= 40 and " " not in x:
            return True
    return False


def plain(text):
    s = str(text).strip()
    if BOOL_WORD.fullmatch(s):
        return s.lower() == "true"
    if EMPTY_WORD.fullmatch(s):
        return None
    m = MONEY_RE.fullmatch(s)
    if m and not LEAD_ZERO.match(m.group(1)):
        num = m.group(1).replace(",", "")
        if "." in num:
            return float(num)
        return int(num)
    return s


def block_value(text):
    p = plain(text)
    if isinstance(p, bool) or (p is None and str(text).strip() != ""):
        return str(text).strip()
    return p


def pair_under(text, mark):
    i = text.find(mark)
    bare = mark.strip()
    if i >= 0:
        name, val = text[:i], text[i + len(mark):]
    elif bare and text.endswith(bare):
        name, val = text[:-len(bare)], ""
    else:
        return None
    name = V20_BULLET.sub("", name, count=1).strip()
    if not V19_NAME.fullmatch(name) or len(name.split()) > 5:
        return None
    return (name, val.strip())


def split_fields(text, sep, mark):
    parts = [p for p in text.split(sep) if p.strip()]
    if not parts or pair_under(parts[0].strip(), mark) is None:
        return [], 0
    fields = []
    named = 0
    heading = None
    for raw in parts:
        st = raw.strip()
        if sep == "\n" and heading is not None and (V20_BULLET.match(raw) or raw[:1] in (" ", "\t")):
            cur = fields[heading][1]
            fields[heading][1] = cur + ("\n" if cur else "") + st
            continue
        heading = None
        if sep == "\n" and st.endswith(":") and V19_NAME.fullmatch(st[:-1].strip()):
            fields.append([st[:-1].strip(), ""])
            heading = len(fields) - 1
            named += 1
            continue
        pr = pair_under(st, mark)
        if pr is not None:
            fields.append([pr[0], pr[1]])
            named += 1
        elif fields:
            fields[-1][1] = fields[-1][1] + sep + st
    return fields, named


def attr_block_text(text):
    s = str(text).strip()
    if len(s) < 5:
        return None
    if s[0] in "{[" and s[-1] in "}]":
        try:
            p = json.loads(s)
            return p if isinstance(p, dict) else None
        except Exception:
            found = {}
            for m in LOOSE_PAIR.finditer(s):
                found[m.group(1).strip()] = plain(m.group(2))
            return found if len(found) >= 2 else None
    if "=" in s:
        parts = [p for p in BLOCK_CUT.split(s) if p and p.strip()]
        block = {}
        last = None
        for part in parts:
            name, eq, rest = part.partition("=")
            nm = name.strip()
            if eq and 1 <= len(nm) <= BLOCK_NAME and nm not in block:
                block[nm] = block_value(rest)
                last = nm
            elif last is not None:
                v = block[last]
                block[last] = ("None" if v is None else str(v)) + "; " + part.strip()
            else:
                return None
        if len(block) >= 2:
            return block
        if len(block) == 1:
            nm = list(block)[0]
            if ":" not in nm and len(nm) <= 30 and len(s) <= 90:
                return block
        return None
    lines = [l for l in s.split("\n") if l.strip()]
    if len(lines) >= 2 and all(V20_BULLET.match(l) for l in lines):
        prs = [pair_under(l.strip(), ": ") for l in lines]
        if all(prs):
            return {n: plain(v) for n, v in prs}
    for sep in (", ", "; "):
        if sep in s and ": " in s:
            parts = [p.strip() for p in s.split(sep)]
            prs = [pair_under(p, ": ") for p in parts]
            if len(parts) >= 2 and all(prs) and all("." not in n for n, v in prs) and all(len(v) <= BLOCK_VALUE for n, v in prs):
                return {n: plain(v) for n, v in prs}
    return None


def record_text(text):
    best = None
    for sep in FIELD_MARKS:
        if sep not in text:
            continue
        pieces = len([p for p in text.split(sep) if p.strip()])
        share = PAIR_SHARE if sep == ", " else FIELD_SHARE
        for mark in PAIR_MARKS:
            fields, named = split_fields(text, sep, mark)
            if named < RECORD_MIN or named < share * pieces:
                continue
            if len(set(f[0] for f in fields)) * 2 <= len(fields):
                continue
            if best is None or named > best[1]:
                best = (fields, named)
        if best is not None:
            break
    if best is None:
        return None
    obj = {}
    for name, val in best[0]:
        if name in obj:
            continue
        blk = attr_block_text(val)
        if blk is not None:
            obj[name] = blk
        elif V22_ID.search(name):
            obj[name] = val.strip()
        else:
            obj[name] = plain(val)
    return obj


def table_text(text):
    lines = []
    for l in text.split("\n"):
        if not l.strip() or RULE_LINE.fullmatch(l):
            continue
        lines.append(l.strip().strip("|").strip())
    if len(lines) < 2:
        return None
    for sep in ("|", "\t", ";", ","):
        rows = [[c.strip() for c in l.split(sep)] for l in lines]
        heads = rows[0]
        if len(heads) < 3 or any(len(r) != len(heads) for r in rows):
            continue
        if not all(V19_NAME.fullmatch(c) for c in heads):
            continue
        if any(NUM_HEAD.fullmatch(c) for c in heads):
            continue
        out = []
        for r in rows[1:]:
            d = {}
            for h, c in zip(heads, r):
                d[h] = plain(c)
            out.append(d)
        return out
    return None


def records_from_text(text):
    s = str(text).strip()
    if len(s) < 12:
        return None
    if s[0] in "{[":
        try:
            parsed = json.loads(s)
            return restore(parsed, 0, False, True)
        except Exception:
            pass
    for parts in (BLANK_CUT.split(s), None):
        if parts is None:
            parts = s.split("\n")
        if len(parts) >= 2:
            recs = [record_text(p) for p in parts]
            readable = [r for r in recs if r]
            if len(readable) >= 2 and len(readable) * 5 >= len(parts) * 4:
                firsts = set(list(r.keys())[0] for r in readable)
                if len(firsts) == 1:
                    return readable
    rec = record_text(s)
    if rec is None and "\n" in s:
        found = table_text(s)
    else:
        found = rec
    if not found and len(s) <= TAGGED_LIMIT and TAG_ANY.search(s):
        t = TAG_INLINE.sub("", s)
        t = TAG_CELL.sub(": ", t)
        t = TAG_RUN.sub("\n", t).strip()
        if t != s and len(t) >= 12:
            return records_from_text(t)
    if not found and "\n" in s:
        lines = [COL_GAP.sub(": ", l.strip(), count=1) for l in s.split("\n") if l.strip()]
        j = "\n".join(lines)
        if j != s:
            return record_text(j)
    return found


def block_text(t):
    if len(t) < 5 or "=" not in t or not ("; " in t or " | " in t or "\n" in t):
        return t
    parts = [p for p in BLOCK_CUT.split(t.strip()) if p and p.strip()]
    if len(parts) < 2 or not all(NV_PART.fullmatch(p.strip()) for p in parts):
        return t
    r = attr_block_text(t)
    return t if r is None else r


def attr_list(lst, allowed):
    if len(lst) < 2:
        return None
    first = lst[0]
    if isinstance(first, dict):
        keys = list(first.keys())
        if len(keys) != 2 or not V21_PAIRKEY.search(str(keys[0])):
            return None
        out = {}
        for el in lst:
            if not isinstance(el, dict) or len(el) != 2 or set(el.keys()) != set(keys):
                return None
            n, v = el.get(keys[0]), el.get(keys[1])
            if not isinstance(n, str) or not (1 <= len(n) <= BLOCK_NAME) or isinstance(v, (dict, list)):
                return None
            out[n.strip()] = v
        return out
    if isinstance(first, list):
        out = {}
        for el in lst:
            if not isinstance(el, list) or len(el) != 2:
                return None
            n, v = el[0], el[1]
            if not isinstance(n, str) or not (1 <= len(n) <= BLOCK_NAME) or isinstance(v, (dict, list)):
                return None
            out[n.strip()] = v
        return out
    if allowed and all(isinstance(e, str) for e in lst):
        out = {}
        for e in lst:
            s = e.strip()
            pr = pair_under(s, ": ") or pair_under(s, "=")
            if pr is None or len(pr[1]) > BLOCK_VALUE or len(pr[0].split()) > 3:
                return None
            out[pr[0]] = plain(pr[1])
        return out
    return None


def holds_id(obj):
    return isinstance(obj, dict) and any(V22_ID.search(str(k)) for k in obj)


def id_like(v, named=False):
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not (3 <= len(s) <= 64) or " " in s:
        return False
    return named or bool(HAS_DIGIT.search(s))


def column_rows(obj):
    heads = [k for k, v in obj.items() if isinstance(v, list) and len(v) >= 2
             and all(isinstance(x, str) and V19_NAME.fullmatch(x) for x in v)]
    for hk in heads:
        hv = obj[hk]
        for k, v in obj.items():
            if k == hk:
                continue
            if isinstance(v, list) and v and all(isinstance(r, list) and len(r) == len(hv) for r in v):
                out = {}
                for k2, v2 in obj.items():
                    if k2 == hk:
                        continue
                    if k2 == k:
                        out[k2] = [dict(zip(hv, r)) for r in v]
                    else:
                        out[k2] = v2
                return out
    return obj


def id_texts(obj):
    if not any(isinstance(v, str) and " " in v for v in obj.values()):
        return obj
    out = {}
    for k, v in obj.items():
        if isinstance(v, int) and not isinstance(v, bool) and V22_ID.search(str(k)) and abs(v) >= 100:
            out[k] = str(v)
        else:
            out[k] = v
    return out


def keyed_rows(obj):
    if len(obj) < 2:
        return None
    vals = list(obj.values())
    if not all(isinstance(v, dict) and len(v) >= 3 for v in vals):
        return None
    if not all(HAS_DIGIT.search(str(k)) and " " not in str(k) for k in obj):
        return None
    shared = set(vals[0].keys())
    for v in vals[1:]:
        shared &= set(v.keys())
    if len(shared) < 3:
        return None
    out = []
    for k, v in obj.items():
        if holds_id(v):
            out.append(v)
        else:
            row = {"id": str(k)}
            row.update(v)
            out.append(row)
    return out


def flat_block(obj):
    if not any(isinstance(v, dict) for v in obj.values()):
        return obj
    out = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            for ik, iv in v.items():
                if isinstance(iv, (dict, list)):
                    continue
                out[str(k) + " " + str(ik)] = iv
        else:
            out[k] = v
    return out


def _flat_ok(v, outer_len):
    if not isinstance(v, dict) or not v or outer_len < 3 or holds_id(v):
        return False
    inner = list(v.values())
    if not any(isinstance(x, dict) for x in inner) or any(isinstance(x, list) for x in inner):
        return False
    f = inner[0]
    if isinstance(f, dict):
        return bool(f)
    return isinstance(f, (str, int, float))


def field_rule(v, depth, inrec, page):
    r = restore(v, depth + 1, inrec, page)
    if isinstance(v, str) and isinstance(r, dict) and len(r) < NOTE_MIN and not holds_id(r)\
            and not any(id_like(x) for x in r.values()):
        return v
    return r


def restore(value, depth=0, inrec=False, page=True):
    if depth > 7:
        return value
    if isinstance(value, str):
        r = records_from_text(value)
        return value if r is None else r
    if isinstance(value, list):
        al = attr_list(value, inrec and page)
        if al is not None:
            return al
        return [field_rule(e, depth, False, page) for e in value]
    if isinstance(value, dict):
        first_text = bool(value) and isinstance(list(value.values())[0], str)
        obj = id_texts(column_rows(value))
        carded = len(obj) >= RECORD_MIN or holds_id(obj)
        obj = dict(obj)
        for k, v in list(obj.items()):
            if isinstance(v, dict):
                kr = keyed_rows(v)
                if kr is not None:
                    obj[k] = kr
        listing = holds_id(obj) or any(id_like(v) for v in obj.values())
        new = {}
        for k, v in obj.items():
            if not isinstance(v, str) or len(obj) <= 2:
                new[k] = field_rule(v, depth, carded, page)
            else:
                new[k] = block_text(v) if listing else v
        if depth <= 3:
            for k, v in list(new.items()):
                if _flat_ok(v, len(new)):
                    new[k] = flat_block(v)
        if len(new) == 1 and first_text:
            only = list(new.values())[0]
            if isinstance(only, dict) and len(only) >= RECORD_MIN:
                return only
        return new
    return value


def join_values(held, new):
    got = set(p.strip().lower() for p in COMMA_CUT.split(held))
    extras = []
    for p in COMMA_CUT.split(new):
        ps = p.strip()
        if ps and ps.lower() not in got:
            got.add(ps.lower())
            extras.append(ps)
    if not extras:
        return held
    joined = held + (", " if held.strip() else "") + ", ".join(extras)
    FIRST_SPELLING[joined] = FIRST_SPELLING.get(held, held)
    return joined


def fold_twins(v, depth=0):
    if depth > 5:
        return v
    if isinstance(v, list):
        return [fold_twins(e, depth + 1) for e in v]
    if not isinstance(v, dict):
        return v
    out = {}
    seen = {}
    for k, val in v.items():
        nk = word_form(k)
        fv = fold_twins(val, depth + 1)
        if nk not in seen:
            seen[nk] = k
            out[k] = fv
            continue
        kk = seen[nk]
        if fv == out[kk]:
            continue
        if isinstance(fv, str) and isinstance(out[kk], str) and not V22_ID.search(nk):
            out[kk] = join_values(out[kk], fv)
            continue
        out[k] = fv
    if len(out) < 2 and len(v) >= 2:
        return {k: fold_twins(val, depth + 1) for k, val in v.items()}
    return out


def dump(v):
    try:
        return json.dumps(v, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(v)


def clip(v, limit, depth=0):
    if depth > 7:
        return "..."
    if isinstance(v, str):
        return v[:limit] + "..." if len(v) > limit else v
    if isinstance(v, dict):
        return {k: clip(x, limit, depth + 1) for k, x in v.items()}
    if isinstance(v, list):
        return [clip(x, limit, depth + 1) for x in v]
    return v


def json_obj(text):
    if not isinstance(text, str):
        return None
    s = text.strip()
    m = FENCE.fullmatch(s)
    if m:
        s = m.group(1)
    try:
        p = json.loads(s)
        return p if isinstance(p, dict) else None
    except Exception:
        pass
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            p = json.loads(s[i:j + 1])
            return p if isinstance(p, dict) else None
        except Exception:
            return None
    return None


def word_form(name):
    s = CAPSPLIT.sub(r"\1 \2", str(name))
    s = SEPRUN.sub(" ", s)
    return s.strip().lower()


def under_form(name):
    return word_form(name).replace(" ", "_")


def stems(text):
    return set(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in word_form(text).split())


def roots(text):
    out = set()
    for w in word_form(text).split():
        if ES_END.search(w):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if len(w) >= 3 and not STOP_WORD.fullmatch(w):
            out.add(w)
    return out


def spelled(text):
    out = set()
    for w in word_form(text).split():
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(OUR_END.sub("or", w))
    return out


def names_match(prop, key):
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def glued(text):
    return GLUE_RE.sub(r"\1\2", str(text).lower())


def text_words(text):
    t = str(text).lower()
    return set(LD_RUN.findall(t)) | set(LD_RUN.findall(glued(t)))


def value_tokens(v):
    g = glued(str(v))
    for rx in (TOK1, TOK2, TOK3):
        f = rx.findall(g)
        if f:
            return f
    return []


def kinds(schema):
    if not isinstance(schema, dict):
        return []
    t = schema.get("type")
    if isinstance(t, str):
        return [t]
    if isinstance(t, list):
        return [x for x in t if isinstance(x, str)]
    return []


def kind_fit(v, k):
    if k == "string":
        return isinstance(v, str)
    if k == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if k == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if k == "boolean":
        return isinstance(v, bool)
    if k == "array":
        return isinstance(v, list)
    if k == "object":
        return isinstance(v, dict)
    if k == "null":
        return False
    return True


def shape(value, schema, path, faults):
    if not isinstance(schema, dict):
        return value
    ks = kinds(schema)
    if ks and not any(kind_fit(value, k) for k in ks):
        for k in ks:
            if k == "integer" and isinstance(value, float) and value == int(value):
                value = int(value)
                break
            if k in ("integer", "number") and isinstance(value, str) and SIGNED_DEC.fullmatch(value):
                num = float(value.strip())
                if k == "integer" or num == int(num):
                    value = int(num)
                else:
                    value = num
                break
            if k == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
                break
            if k == "boolean" and isinstance(value, str) and BOOL_WORD.fullmatch(value.strip()):
                value = value.strip().lower() == "true"
                break
            if k == "array" and value is not None and not isinstance(value, list):
                value = [value]
                break
        if not any(kind_fit(value, k) for k in ks):
            faults.append("%s should be %s" % (path, "/".join(ks)))
            return value
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and value not in enum:
        if isinstance(value, str):
            low = value.strip().lower()
            for opt in enum:
                if isinstance(opt, str) and opt.lower() == low:
                    value = opt
                    break
        if value not in enum:
            faults.append("%s must be one of %s" % (path, dump(enum)))
    if isinstance(value, str):
        mx = schema.get("maxLength")
        if isinstance(mx, int) and not isinstance(mx, bool) and len(value) > mx:
            value = value[:mx]
        mn = schema.get("minLength")
        if isinstance(mn, int) and not isinstance(mn, bool) and len(value) < mn:
            faults.append("%s is too short" % path)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if isinstance(lo, (int, float)) and not isinstance(lo, bool) and value < lo:
            faults.append("%s is below the minimum %s" % (path, lo))
        if isinstance(hi, (int, float)) and not isinstance(hi, bool) and value > hi:
            faults.append("%s exceeds the maximum %s" % (path, hi))
    if isinstance(value, list):
        mx, mn = schema.get("maxItems"), schema.get("minItems")
        if isinstance(mx, int) and len(value) > mx:
            faults.append("%s allows at most %s entries" % (path, mx))
        if isinstance(mn, int) and len(value) < mn:
            faults.append("%s needs at least %s entries" % (path, mn))
        it = schema.get("items")
        if isinstance(it, dict):
            value = [shape(e, it, "%s[%d]" % (path, i), faults) for i, e in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get("properties")
        props = props if isinstance(props, dict) else {}
        if schema.get("additionalProperties") is False:
            for n in value:
                if n not in props:
                    faults.append("%s.%s is not allowed by the schema" % (path, n))
        req = schema.get("required")
        if isinstance(req, list):
            for n in req:
                if n not in value:
                    faults.append("%s.%s is required" % (path, n))
        out = {}
        for n, x in value.items():
            if n in props:
                out[n] = shape(x, props[n], "%s.%s" % (path, n), faults)
            else:
                out[n] = x
        value = out
    return value


def items_of(v):
    if isinstance(v, list):
        return v
    if (isinstance(v, str) and v) or (isinstance(v, dict) and v):
        return [v]
    return []


def count_of(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v == int(v):
        return int(v)
    if isinstance(v, str):
        m = COUNT_TEXT.fullmatch(v)
        if m:
            return int(m.group(1))
    return None


def learn_id(rows):
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return None
    best, best_score = None, None
    for k in list(rows[0].keys()):
        vals = [r.get(k) for r in rows]
        named = bool(V22_ID.search(str(k)))
        if not all(id_like(x, named) for x in vals):
            continue
        st = [x.strip() for x in vals]
        if len(rows) >= 2 and len(set(st)) < len(st):
            continue
        score = (named, sum(len(x) for x in st) / float(len(st)))
        if best_score is None or score > best_score:
            best, best_score = k, score
    return best


def row_present(lid, struct):
    lid = str(lid)

    def walk(x, depth, inlist):
        if depth > 5:
            return False
        if isinstance(x, dict):
            if inlist and any(isinstance(v, str) and v.strip() == lid for v in x.values()):
                return True
            return any(walk(v, depth + 1, False) for v in x.values())
        if isinstance(x, list):
            return any(walk(e, depth + 1, True) for e in x)
        return False
    return walk(struct, 0, False)


def client_for(room):
    if room >= 50:
        return MODEL_CLIENT
    t = max(5, int(room // 5) * 5)
    c = SHORT_CLIENTS.get(t)
    if c is None:
        c = ProxyClient(timeout=t, max_retries=1, retry_delay=1)
        SHORT_CLIENTS[t] = c
    return c


def attr_of(rec):
    if not isinstance(rec, dict):
        return None
    best, bestn = None, 1
    for v in rec.values():
        if isinstance(v, dict) and len(v) >= 2:
            n = sum(1 for x in v.values() if isinstance(x, (str, int, float)) and not isinstance(x, bool))
            if n >= 2 and n > bestn:
                best, bestn = v, n
    if best is None:
        return None
    return {k: x for k, x in best.items() if isinstance(x, (str, int, float)) and not isinstance(x, bool)}


class Trace(list):
    def __init__(self, t0):
        list.__init__(self)
        self.t0 = t0

    def append(self, entry):
        if isinstance(entry, dict) and "t" not in entry:
            entry = dict(entry)
            entry["t"] = round(time.monotonic() - self.t0, 1)
        list.append(self, entry)


def filler(schema):
    if not isinstance(schema, dict):
        return False, None
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return True, enum[0]
    if "const" in schema:
        return True, schema["const"]
    if "default" in schema:
        return True, schema["default"]
    ks = kinds(schema)
    if "integer" in ks or "number" in ks:
        mn = schema.get("minimum")
        return True, (mn if isinstance(mn, (int, float)) and not isinstance(mn, bool) else 1)
    if "boolean" in ks:
        return True, False
    if "string" in ks:
        mx = schema.get("maxLength")
        return True, FILLER_WORD[:(mx if isinstance(mx, int) and mx else 8)]
    return False, None


def strings_of(x):
    out = []
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, dict):
        for v in x.values():
            out.extend(strings_of(v))
    elif isinstance(x, list):
        for v in x:
            out.extend(strings_of(v))
    return out


def says(x, text):
    return any(s == text for s in strings_of(x))


class Item(object):
    def __init__(self, name, shelf, quantity, musts):
        self.name = name or "the product"
        self.shelf = shelf or ""
        self.quantity = quantity if isinstance(quantity, int) and quantity > 0 else 1
        self.musts = list(musts or [])
        self.told = []
        self.dropped = []
        self.free = []

    def live(self):
        return [r for r in self.musts + self.told if r not in self.dropped]


END_409 = re.compile(r"finish|quarantin|finaliz")
RESYNC_409 = re.compile(r"expected (\d+)")
REUSE_409 = re.compile(r"reuse|different action")
POINTER_WORDS = re.compile(r"\b(?:listing|page|shown|other\s+than|unlike|same\s+as)\b", re.I)
WORD_RUN = re.compile(r"[a-z0-9]+")
PLAIN_BUDGET = re.compile(r"\s*\d+(?:\.\d+)?\s*")
ANTHROPIC_ID = re.compile(r"anthropic/")


class Duty(object):
    def __init__(self, text, stage, messages):
        self.text = text
        self.stage = stage
        self.messages = messages


class E1(object):
    def __init__(self, problem_data):
        env = problem_data["environment"]
        self.binding = dict(env["binding"])
        self.pv = env["policy_view"]
        q = self.pv.get("query")
        self.request = q if isinstance(q, str) else ""
        self.tools = []
        self.schemas = {}
        self.descs = {}
        for t in self.pv.get("tools") or []:
            if not isinstance(t, dict) or not isinstance(t.get("function"), dict):
                continue
            fn = t["function"]
            if not isinstance(fn.get("name"), str):
                continue
            ps = fn.get("parameters")
            if not isinstance(ps, dict):
                ps = {"type": "object", "properties": {}}
            self.tools.append(t)
            self.schemas[fn["name"]] = ps
            self.descs[fn["name"]] = fn.get("description") or ""
        ms = self.pv.get("max_steps")
        b = count_of(ms)
        if b is None and isinstance(ms, float):
            b = int(ms)
        self.budget_steps = b if b and b > 0 else 30
        self.t0 = time.monotonic()
        self.steps = 0
        self.trace = Trace(self.t0)
        self.trace.t0 = self.t0
        self.finished = bool(self.pv.get("finished"))
        self.store_broke = False
        self.no_credit = False
        self.waited = 0
        self.misses = 0
        self.spend = 0.0
        self.model_calls = 0
        self.turns = 0
        self.gap = STORE_PAUSE
        self.store_secs = 0.0
        self.store_calls = 0
        self.last_call = 0.0
        self.convo = []
        self.brief = {}
        self.roles = {}
        self.items = []
        self.budget = None
        self.msg_limit = 0
        self.msgs_used = 0
        self.duties = []
        self.msg_reserve = 0
        self.gate_msgs = 0
        self.gate_passes = set()
        self.done_log = []
        self.done_steps = []
        self.asked = False
        self.question_log = []
        self.heard = set()
        self.change_notes = []
        self.version = 0
        self.queued = []
        self.rows = {}
        self.full = {}
        self.seen = []
        self.opened = set()
        self.views = {}
        self.facts = {}
        self.bare = {}
        self.factless = set()
        self.id_key = None
        self.row_keys = set()
        self.tool_keys = {}
        self.view_keys = {}
        self.shapes = {}
        self.price_key = None
        self.stock_key = None
        self.read_cache = {}
        self.pages_read = 0
        self.paged = []
        self.mem_results = {}
        self.mem_args = {}
        self.page_arg = {}
        self.page_queries = {}
        self.searched = []
        self.value_rows = set()
        self.compare_ok = 0
        self.compare_bad = 0
        self.code_tries = []
        self.codes_dead = False
        self.cart_touched = False
        self.cart_changed_at = -1
        self.cart_seen_at = -1
        self.cart_view = None
        self.cart = {}
        self.cart_said = ""
        self.id_args = set()
        self.built = []
        self.dropped_ids = set()
        self.checkouts = 0
        self.refused_at = -1
        self.refusals = 0
        self.recent = []
        self.sold_holds = 0
        self.line_holds = 0
        self.fit_holds = 0
        self.checked = set()
        self.verdicts = {}
        self.audited = False
        self.leads = []
        self.lead_items = {}
        self.decline_blocks = 0
        self.pushed = False
        self.late_held = False
        self.confirmed = None
        self.completed = False
        self.offer_seen = {}
        self.gone = {}
        self.unavailable = set()
        self.sold_lines = set()
        self.note_sold = False
        self.note_named = False
        self.refs = []
        self.ref_values = {}
        self.chain_rounds = 0
        self.chain_holds = 0
        self.chain_room = None
        self.decline_chain_done = False
        self.link_tried = set()
        self.link_searches = 0
        self.link_seen = {}
        self.fit_pairs = []
        self.fit_cache = None
        self.owners_seen = {}
        self.adopted = {}
        self.plan_order = []
        self.lanes = {}
        self.lane_found = {}
        self.lane_spent = {}
        self.lane_used = {}
        self.lane_misses = {}
        self.late_asked = False
        self.one_product = set()
        self.list_asks = 0
        self.apart = {}
        self.follow_ups = []
        self.listing = True
        self.reply_reads = 0
        self.capped = set()
        self.cheaper_marks = set()
        self.wider_held = False
        self.hunts = 0
        self.hunt_mark = ""
        self.hunt_tried = {}
        self.hunt_stems = {}
        self.hunt_nonoun = set()
        self.fit_hunted = False
        self.lookups = []
        self.planned = False
        self.questions_sent = 0
        self.lookalike_done = False
        self.title_key = None
        self.cat_key = None
        self.cat_cache = None
        self.shape_asked = set()
        self.notes_shown = set()

    def steps_left(self):
        return self.budget_steps - self.steps

    def clock(self):
        return time.monotonic() - self.t0

    def room(self):
        return EPISODE_LEN - EPISODE_MARGIN - self.clock()

    def rushed(self):
        pace = self.store_secs / self.store_calls if self.store_calls else 1.0
        return self.room() <= 2 * max(pace, 1.0)

    def free_msgs(self):
        return self.msg_limit - self.msgs_used - max(0, self.msg_reserve - self.gate_msgs)

    def cart_mark(self):
        return dump(sorted([[k, v] for k, v in self.cart.items()], key=lambda p: str(p[0])))

    def pass_mark(self):
        return self.cart_mark() + "#" + str(self.version)

    def gate_due(self, stage):
        return any(d.stage == stage for d in self.duties) and (stage, self.pass_mark()) not in self.gate_passes

    def reserve(self):
        if any(d.stage == "order" and d.messages > 0 for d in self.duties) and self.msgs_used < self.msg_limit\
                and self.gate_due("order"):
            return 2
        return 1

    def judges(self, limit=SOFT_SPEND):
        if self.spend <= limit and self.spend + HEAVY_ROOM <= HARD_SPEND:
            return [HEAVY, LIGHT]
        if self.spend <= HARD_SPEND:
            return [LIGHT]
        return []

    def role(self, name):
        return self.roles.get(name)

    def page_tools(self):
        out = []
        if self.role("offer"):
            out.append(self.role("offer"))
        elif self.role("open"):
            out.append(self.role("open"))
        if self.role("compare"):
            out.append(self.role("compare"))
        return out

    def facts_tool(self):
        for t, v in self.facts.items():
            if v is True and self.single_target(t):
                return t
        o = self.role("open")
        if o and self.facts.get(o) is not False and self.single_target(o):
            return o
        return None

    def compare_facts(self):
        c = self.role("compare")
        return self.facts.get(c) if c else None

    def compare_bare(self):
        cf = self.compare_facts()
        if cf is False:
            return True
        return cf is None and (self.facts.get(self.facts_tool()) is True or bool(self.role("offer")))

    def map_row_ids(self, tool, args):
        if tool not in [self.role(r) for r in ("open", "offer", "compare", "cart") if self.role(r)]:
            return args
        if not self.rows or not self.get_title_key() or not isinstance(args, dict):
            return args
        props = (self.schemas.get(tool) or {}).get("properties") or {}

        def swap(name, val):
            if not isinstance(val, str):
                return val
            s = val.strip()
            if s in self.rows or not (3 <= len(s) <= 30) or not HAS_DIGIT.search(s):
                return val
            sch = props.get(name) if isinstance(props, dict) else None
            if isinstance(sch, dict) and (sch.get("enum") or (isinstance(sch.get("items"), dict) and sch["items"].get("enum"))):
                return val
            rx = re.compile(r"(?<![a-z0-9])" + re.escape(s.lower()) + r"(?![a-z0-9])")
            hits = []
            for lid, row in self.rows.items():
                title = row.get(self.title_key) if isinstance(row, dict) else None
                if isinstance(title, str) and rx.search(title.lower()):
                    hits.append(lid)
            if len(hits) == 1:
                self.trace.append({"kind": "row_id", "from": s, "to": hits[0]})
                return hits[0]
            return val
        out = {}
        for k, v in args.items():
            if isinstance(v, list):
                out[k] = [swap(k, e) for e in v]
            else:
                out[k] = swap(k, v)
        return out

    def think(self, messages, models, cap, tools=None):
        for m in models:
            if self.model_calls >= CALL_CAP:
                return None
            if self.room() < CALL_FLOOR:
                self.trace.append({"kind": "late_call_skipped", "model": m})
                return None
            self.model_calls += 1
            tries = 0
            status = None
            while True:
                msg, status = self.single_call(m, messages, cap, tools)
                if msg is not None:
                    return msg
                retryable = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                most = 1 if status in (None, 408) else len(RETRY_WAITS)
                if not retryable or tries >= most:
                    break
                wait = RETRY_WAITS[tries]
                if self.waited + wait > WAIT_BUDGET or self.room() - 1.2 * wait < CALL_FLOOR:
                    break
                time.sleep(wait * random.uniform(0.8, 1.2))
                self.waited += wait
                tries += 1
            if status == 402:
                self.no_credit = True
                return None
        return None

    def single_call(self, model, messages, cap, tools=None):
        msgs = messages
        if ANTHROPIC_ID.match(model) and len(messages) >= 2:
            msgs = list(messages)
            for i in (0, len(msgs) - 1):
                c = msgs[i].get("content")
                if isinstance(c, str) and c:
                    mm = dict(msgs[i])
                    mm["content"] = [{"type": "text", "text": c, "cache_control": {"type": "ephemeral"}}]
                    msgs[i] = mm
        body = {"model": model, "messages": msgs, "max_tokens": cap, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        try:
            res = client_for(self.room()).post_verbose("/inference/chat/completions", body)
            data, err = res.data, res.error
        except Exception as e:
            data, err = None, {"status": None, "body": "%s: %s" % (type(e).__name__, e)}
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            cost = usage.get("cost") if isinstance(usage, dict) else None
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                self.spend += cost
            msg = (data["choices"][0] or {}).get("message") or {}
            if msg.get("content") or msg.get("tool_calls"):
                return msg, 200
            self.trace.append({"kind": "empty_reply", "model": model})
            return None, 0
        status = (err or {}).get("status")
        body_text = (err or {}).get("body")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            status = count_of(data["error"].get("code"))
            body_text = data["error"].get("message")
        self.trace.append({"kind": "model_error", "model": model, "status": status, "body": str(body_text)[:200]})
        return None, status

    def ask_json(self, prompt, models, cap):
        for m in models:
            msg = self.think([{"role": "user", "content": prompt}], [m], cap)
            obj = json_obj(msg.get("content")) if msg else None
            if obj is not None:
                return obj
            if self.store_broke or self.no_credit or self.model_calls >= CALL_CAP:
                return None
        return None

    def env_call(self, tool, args):
        env = dict(self.binding)
        env["step"] = self.steps + 1
        env["action"] = {"name": tool, "args": args}
        now = time.monotonic()
        if now - self.last_call < self.gap:
            time.sleep(self.gap - (now - self.last_call))
        self.last_call = time.monotonic()
        for _ in range(4):
            t = time.monotonic()
            try:
                res = ENV_CLIENT.post_verbose("/environment/call", env)
                data, err = res.data, res.error
            except Exception as e:
                data, err = None, {"status": None, "body": "%s: %s" % (type(e).__name__, e)}
            self.store_secs += time.monotonic() - t
            self.store_calls += 1
            if isinstance(data, dict):
                st = data.get("step")
                if isinstance(st, int) and not isinstance(st, bool):
                    self.steps = st
                else:
                    self.steps += 1
                if data.get("finished") is True:
                    self.finished = True
                if tool != self.role("ask"):
                    data["observation"] = self.unspell(tool, data.get("observation"))
                obs = data.get("observation")
                self.note_read(tool, args, obs)
                if tool == self.facts_tool() and isinstance(obs, dict) and not is_failure(obs):
                    ident = self.identity(obs)
                    if ident and attr_of(obs) is None:
                        self.factless.add((tool, ident))
                if tool in self.page_tools() and not (isinstance(obs, dict) and obs.get("error"))\
                        and not is_failure(obs) and isinstance(args, dict):
                    cj = None
                    for v in args.values():
                        vals = []
                        if isinstance(v, str):
                            vals = [v.strip()]
                        elif isinstance(v, list):
                            if cj is None:
                                cj = dump(obs)
                            vals = [e.strip() for e in v if isinstance(e, str) and e.strip() in cj]
                        for x in vals:
                            self.opened.add(x)
                            self.views.setdefault(x, set()).add(tool)
                return data
            status = (err or {}).get("status")
            body = str((err or {}).get("body"))
            self.trace.append({"kind": "env_error", "tool": tool, "status": status, "body": body[:200]})
            if status == 409:
                low = body.lower()
                if END_409.search(low):
                    self.finished = True
                    return None
                m = RESYNC_409.search(body)
                if m:
                    n = int(m.group(1))
                    self.steps = n - 1
                    env["step"] = n
                    continue
                if REUSE_409.search(low):
                    self.steps += 1
                    env["step"] = self.steps + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.store_broke = True
                return None
        return None

    def unspell(self, tool, obs):
        views = [self.role(r) for r in ("open", "offer", "compare") if self.role(r)]
        page = tool in views or tool != self.role("search")
        r = fold_twins(restore(obs, 0, False, page))
        if tool not in views:
            return r
        for lid, row in self.rows.items():
            if lid not in self.opened and isinstance(row, dict):
                self.row_keys |= set(row.keys())
        if not self.row_keys:
            return r

        def gather(rec):
            if not isinstance(rec, dict) or attr_of(rec) is not None or not self.identity(rec, True):
                return rec
            extras = {}
            for k, v in rec.items():
                if k not in self.row_keys and isinstance(v, (str, int, float)) and not isinstance(v, bool):
                    extras[k] = v
            if len(extras) < RECORD_MIN:
                return rec
            out = {k: v for k, v in rec.items() if k not in extras}
            out["attributes"] = extras
            return out
        if isinstance(r, dict) and self.identity(r):
            return gather(r)
        if isinstance(r, dict):
            return {k: ([gather(e) for e in v] if isinstance(v, list) else v) for k, v in r.items()}
        return r

    def read_request(self):
        lines = []
        for t in self.tools:
            fn = t["function"]
            lines.append("- %s: %s | parameters: %s" % (fn["name"], fn.get("description") or "", dump(fn.get("parameters") or {})))
        prompt = BRIEF_PROMPT + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.pv.get("opening") is not None:
            prompt += "\n\nTHE STORE'S OPENING NOTE:\n" + dump(clip(self.pv.get("opening"), 1500))
        b = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit:
            roles = b.get("roles")
            if not b.get("items") or not isinstance(roles, dict) or not roles.get("buy"):
                b2 = self.ask_json(prompt, [HEAVY, LIGHT], 2500)
                if b2:
                    b = b2
            else:
                shelves = [word_form(e.get("shelf")) for e in items_of(b.get("items")) if isinstance(e, dict)
                           and e.get("shelf") and word_form(e.get("shelf"))]
                if len(shelves) != len(set(shelves)):
                    b2 = self.ask_json(prompt, [HEAVY], 2500)
                    if b2 and b2.get("items") and isinstance(b2.get("roles"), dict) and b2["roles"].get("buy"):
                        b = b2
        while not b and not self.no_credit and self.model_calls < CALL_CAP and self.clock() < BRIEF_DEADLINE:
            time.sleep(OUTAGE_PAUSE)
            b = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        if b:
            self.waited = 0
        roles = b.get("roles") if isinstance(b.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        for k in ("buy_args", "refuse_args"):
            if not isinstance(b.get(k), dict):
                b[k] = {}
        self.brief = b
        buy, ref = self.role("buy"), self.role("refuse")
        if buy and buy == ref and not (b["buy_args"] and b["refuse_args"] and b["buy_args"] != b["refuse_args"]):
            p = ("The store tool %s can either complete the purchase or close the episode with no purchase, depending on "
                 "the arguments it is given. Its parameter schema: %s\nReply with a single JSON object only: "
                 "{\"buy_args\": {the constant arguments that make it buy}, \"refuse_args\": {the constant arguments that "
                 "make it decline}}. Give only the arguments that select the mode." % (buy, dump(self.schemas.get(buy))))
            ans = self.ask_json(p, [LIGHT, HEAVY], 1500)
            if ans and isinstance(ans.get("buy_args"), dict) and isinstance(ans.get("refuse_args"), dict)\
                    and ans["buy_args"] != ans["refuse_args"]:
                b["buy_args"], b["refuse_args"] = ans["buy_args"], ans["refuse_args"]
        raw = []
        for e in items_of(b.get("items")):
            if not isinstance(e, dict):
                continue
            q = count_of(e.get("quantity"))
            musts = [str(x) for x in items_of(e.get("musts")) if x]
            name = e.get("name")
            raw.append(Item(str(name) if name is not None else "the product", str(e.get("shelf") or ""),
                            q if q and q > 0 else 1, musts))
        merged, newnum = [], {}
        for i, it in enumerate(raw, 1):
            target = None
            if it.shelf.strip():
                for j, m in enumerate(merged, 1):
                    if m.quantity == it.quantity and word_form(m.shelf) == word_form(it.shelf) and not self._conflict(m, it):
                        target = j
                        break
            if target is None:
                merged.append(it)
                newnum[i] = len(merged)
            else:
                m = merged[target - 1]
                for r in it.musts:
                    if r not in m.musts:
                        m.musts.append(r)
                newnum[i] = target
                self.trace.append({"kind": "merged_item", "item": i, "into": target})
        if not merged:
            merged = [Item("the requested product", "", 1, [])]
        self.items = merged
        for f in items_of(b.get("fits")):
            if not isinstance(f, dict):
                continue
            a, w = count_of(f.get("item")), count_of(f.get("with"))
            if a in newnum and w in newnum:
                pa, pw = newnum[a], newnum[w]
                if pa != pw and 1 <= pa <= len(self.items) and 1 <= pw <= len(self.items) and (pa, pw) not in self.fit_pairs:
                    self.fit_pairs.append((pa, pw))
        self.lookups = [str(x).strip() for x in items_of(b.get("lookups")) if isinstance(x, (str, int, float))]
        for lk in self.lookups:
            if not any(lk in it.musts for it in self.items):
                lp = word_form(lk.partition(":")[0])
                for it in self.items:
                    hit = [m for m in it.musts if word_form(m.partition(":")[0]) == lp]
                    if hit:
                        lk = hit[0]
                        break
            num = 1
            for i, it in enumerate(self.items, 1):
                if lk in it.musts:
                    num = i
                    break
            if lk and (num, lk) not in self.refs:
                self.refs.append((num, lk))
        self.unmix()
        bud = b.get("budget")
        if isinstance(bud, str) and PLAIN_BUDGET.fullmatch(bud):
            bud = float(bud)
        if isinstance(bud, (int, float)) and not isinstance(bud, bool) and bud > 0:
            self.budget = bud
        n = count_of(b.get("messages"))
        if n is not None and n >= 0:
            self.msg_limit = n
        elif self.role("ask"):
            self.msg_limit = 3
        s = count_of(b.get("steps"))
        if s is not None and s >= 10 and s < self.budget_steps:
            self.budget_steps = s
        for d in items_of(b.get("process")):
            if not isinstance(d, dict) or not str(d.get("step") or "").strip():
                continue
            st = str(d.get("when") or "").strip().lower()
            if st not in ("choose", "order", "end"):
                st = "order"
            mc = count_of(d.get("messages"))
            self.duties.append(Duty(str(d.get("step")).strip(), st, max(0, mc or 0)))
        if self.role("ask"):
            counts = [min(d.messages, self.msg_limit) for d in self.duties if d.stage in ("order", "end") and d.messages > 0]
            self.msg_reserve = max(counts) if counts else 0
        self.trace.append({"kind": "brief", "brief": b})

    def _conflict(self, a, b):
        for x in a.musts:
            for y in b.musts:
                if word_form(x.partition(":")[0]) == word_form(y.partition(":")[0]) and\
                        word_form(x.partition(":")[2]) != word_form(y.partition(":")[2]):
                    return True
        return False

    def unmix(self):
        for i, it in enumerate(self.items, 1):
            pointers = [r for (n, r) in self.refs if n == i and r in it.musts]
            rest = [m for m in it.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (UNMIX_PROMPT + "\n\nREQUEST:\n" + self.request + "\n\nPRODUCT: " + it.name
                      + "\nTHESE REQUIREMENTS TAKE THEIR VALUE FROM A DIFFERENT LISTING:\n" + "\n".join(pointers)
                      + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            ans = self.ask_json(prompt, [HEAVY, LIGHT], 1500) or {}
            texts = [x.strip() for x in items_of(ans.get("describe_other")) if isinstance(x, str)]
            moved = [r for r in rest if r in texts]
            if not moved:
                continue
            it.musts = [m for m in it.musts if m not in moved]
            first = pointers[0]
            new = first + " (that listing: " + "; ".join(moved) + ")"
            it.musts = [new if m == first else m for m in it.musts]
            self.refs = [(n, new) if (n == i and r == first) else (n, r) for (n, r) in self.refs]
            self.trace.append({"kind": "unmixed", "item": i, "moved": moved})

    def record_label(self, lid):
        ck = self.get_cat_key()
        if not ck:
            return None
        for src in (self.full.get(lid), self.rows.get(lid)):
            if isinstance(src, dict) and src.get(ck) is not None:
                return src.get(ck)
        return None

    def described(self, num, req):
        if any(req == r for _, r in self.refs) or req in self.lookups:
            return False
        prop, sep, val = req.partition(":")
        if not sep or not prop.strip():
            return False
        if len(WORD_RUN.findall(val.lower())) < 4:
            return False
        if POINTER_WORDS.search(val):
            return False
        vs = stems(val)
        for j, other in enumerate(self.items, 1):
            if j != num and vs & set(w for w in stems(other.name) if len(w) >= 4):
                return False
        it = self.items[num - 1]
        blocks = []
        ck = self.get_cat_key()
        for lid, rec in self.full.items():
            ab = attr_of(rec)
            if ab is None:
                continue
            if ck:
                lab = self.record_label(lid)
                if lab is None or not (self.label_owner(lab) == num or word_form(lab) == word_form(it.shelf)):
                    continue
            blocks.append(ab)
        if len(blocks) < 3:
            return False
        ps = stems(prop)
        for ab in blocks:
            for k in ab:
                if stems(k) & ps:
                    return False
        return True

    def live_of(self, num):
        return self.items[num - 1].live()

    def ledger(self):
        lines = []
        many = len(self.items) > 1
        for i, it in enumerate(self.items, 1):
            head = ("item %d: %s" % (i, it.name)) if many else ("item: %s" % it.name)
            if it.shelf:
                head += " (store label: %s)" % it.shelf
            head += ", quantity %d" % it.quantity
            deciding, desc = [], []
            for r in it.live():
                if self.described(i, r):
                    desc.append(r)
                elif r in self.ref_values:
                    deciding.append("%s (value read on the referenced page: %s)" % (r, self.ref_values[r]))
                else:
                    deciding.append(r)
            if deciding:
                head += "; requirements: " + "; ".join(deciding)
            else:
                head += "; requirements: nothing besides the product type"
            if desc:
                head += ("; in the shopper's own phrasing only, recorded by no attribute of any listing, so they "
                         "settle nothing: " + "; ".join(desc))
            lines.append(head)
        b = self.brief
        if self.budget is not None:
            unit = b.get("budget_unit")
            unit = str(unit) if unit else ""
            s = "budget: at most %s %s %s" % (self.budget, unit, "for each item" if b.get("budget_scope") == "each" else "in total")
            lines.append(s.replace("  ", " ", 1))
        pick = b.get("pick")
        if pick == "cheapest":
            lines.append("pick: the lowest-priced matching listing (lowest price)")
        elif pick in ("most", "least"):
            lines.append("pick: the %s %s among the matching listings" % (pick, b.get("pick_measure") or ""))
        else:
            lines.append("pick: any listing meeting every requirement")
        if b.get("in_stock_only") is not False:
            lines.append("in stock only")
        for w in items_of(b.get("wishes")):
            lines.append("wish (only breaks ties): %s" % w)
        for r in items_of(b.get("rules")):
            lines.append("rule the request gives: %s" % r)
        if b.get("quiet_decline") is True:
            lines.append("if nothing fits, the episode ends without naming or proposing any listing")
        for d in self.duties:
            if d.stage == "choose":
                when = "before choosing"
            elif d.stage == "end":
                when = "at a decline; Cairn carries it out at that moment"
            else:
                when = "at the purchase; Cairn carries it out when you order"
            lines.append("step the request requires (%s): %s" % (when, d.text))
        ft = self.facts_tool()
        if ft and (self.facts.get(ft) is True or self.role("offer")):
            s = "listing attributes come from %s" % ft
            pool = list(self.page_tools())
            if self.role("offer") and self.role("offer") not in pool:
                pool.append(self.role("offer"))
            falses = sorted(set(t for t in pool if self.facts.get(t) is False))
            if falses:
                s += "; " + ", ".join(falses) + " show none"
            others = sorted(set(t for t in self.page_tools() if t != ft))
            if others:
                s += "; " + " or ".join(others) + " display a listing's page, the page meant whenever a chain is followed or a page is checked"
            lines.append(s)
        return "\n".join("- " + l for l in lines)

    def status(self):
        S, M, L = self.steps, self.budget_steps, self.steps_left()
        R = self.reserve()
        head = "[Cairn] step %d of %d used, %d left." % (S, M, L)
        if L <= R + 1:
            head += (" FINAL STEPS: order now if every item already has a carted line; decline only when searching has "
                     "proved that some item lacks any matching listing, and a shortage of steps proves no such thing.")
        elif L <= 6 or (L <= len(self.items) + 5 and self.checkouts == 0):
            head += " Few steps remain: make the cart hold exactly the picked lines and order at once."
        elif self.checkouts == 0 and L <= 12:
            head += " Steps are getting scarce: decide the picks, put them in the cart and order within a few more steps."
        elif self.checkouts == 0 and not self.cart_touched and S >= M // 2:
            head += (" Half the step budget is spent with nothing in the cart: put the matching listing of every item "
                     "in it now, because the store may yet sell a listing out.")
        if self.clock() >= WIND_DOWN and L > R + 1:
            head += (" TIME IS ALMOST OVER: make the cart hold the picked lines and order within two more turns, since "
                     "an episode cut off by the clock scores zero.")
        out = [head]
        live = set()
        for it in self.items:
            live |= set(it.live())
        pend = [r for (n, r) in self.refs if r in live and r not in self.ref_values]
        if pend and self.checkouts == 0:
            out.append("[settle these first: visit the listing each one points at, and also every listing to which that "
                       "listing's page directs onward (only those, not every listing it mentions), since the final page "
                       "holds the value] " + " | ".join(pend[:3]))
        if self.leads:
            out.append("[review candidates] " + ", ".join(self.leads))
        if self.fit_pairs and self.checkouts == 0:
            fp = self.fit_plan()
            if fp:
                out.append(fp)
        if self.change_notes:
            out.append("[changes so far] " + " | ".join(self.change_notes[-4:]))
        out.append("[ledger]\n" + self.ledger())
        return "\n".join(out)


E2_COMPARE = re.compile(r"\b(?:not|no|other\s+than|unlike|under|less|below|up\s+to|at\s+most|more|above|over|at\s+least|between|within|or)\b", re.I)
E2_PRICED = re.compile(r"price|money|cost|amount|spend|budget", re.I)
E2_CAPWORD = re.compile(r"max|most|under|below|limit|cap|upto|up_to", re.I)
THREE_DIGITS = re.compile(r"\d{3}")
SPACED_SLASH = re.compile(r" / ")
LETTER_RUN = re.compile(r"[a-z]{3,}")
ALNUM3 = re.compile(r"[a-z0-9]{3,}")
ALNUM = re.compile(r"[a-z0-9]+")
PRICE_TEXT = re.compile(r"\s*[^\d\s]{0,3}\s?(\d[\d,]*(?:\.\d+)?)\s*[A-Za-z]{0,6}\.?\s*")


def gnum(v):
    try:
        return "%g" % float(v)
    except Exception:
        return str(v)


class KeySlot(object):
    def __init__(self, wf):
        self.wf = wf
        self.count = 0
        self.per = {}
        self.distinct = []


class E2(object):
    def identity(self, obj, inrows=False, depth=0):
        if not isinstance(obj, dict) or len(obj) < 3:
            return None
        if self.id_key and id_like(obj.get(self.id_key), True):
            return obj[self.id_key].strip()
        if not (inrows or depth == 0 or any(isinstance(v, dict) for v in obj.values())):
            return None
        k = learn_id([obj])
        if k:
            v = obj[k].strip()
            if "id" in str(k).lower() or v in self.rows:
                return v
        return None

    def is_row_list(self, lst):
        cands = [e for e in lst if isinstance(e, dict) and len(e) >= 3]
        if not cands or len(cands) * 2 < len(lst):
            return False
        if self.id_key and all(id_like(c.get(self.id_key), True) for c in cands):
            return True
        k = learn_id(cands)
        if k:
            self.id_key = k
            return True
        return False

    def absorb(self, obs):
        self._absorb(obs, 0, False)

    def _absorb(self, x, depth, inrows):
        if depth > 6:
            return
        if isinstance(x, dict):
            lid = self.identity(x, inrows, depth)
            if lid:
                if lid not in self.rows:
                    self.seen.append(lid)
                    self.rows[lid] = {}
                row = self.rows[lid]
                went_false = any(row.get(k) is True and v is False for k, v in x.items())
                went_true = any(row.get(k) is False and v is True for k, v in x.items())
                if went_false:
                    self.unavailable.add(lid)
                elif went_true:
                    self.unavailable.discard(lid)
                row.update(x)
                if attr_of(x) is not None:
                    self.full[lid] = dict(row)
                elif lid in self.full:
                    self.full[lid].update(x)
                self.cat_cache = None
                return
            for v in x.values():
                self._absorb(v, depth + 1, False)
        elif isinstance(x, list):
            rl = self.is_row_list(x)
            for e in x:
                self._absorb(e, depth + 1, rl)

    def enum_labels(self):
        out = set()

        def walk(s, d):
            if d > 6 or not isinstance(s, dict):
                return
            e = s.get("enum")
            if isinstance(e, list):
                for o in e:
                    if isinstance(o, str):
                        out.add(o.strip().lower())
            for v in s.values():
                if isinstance(v, dict):
                    walk(v, d + 1)
        for sch in self.schemas.values():
            walk(sch, 0)
        return out

    def get_cat_key(self):
        if self.cat_cache is not None:
            return self.cat_cache[0]
        distinct, present, lens, order = {}, {}, {}, []
        for row in self.rows.values():
            for k, v in row.items():
                if isinstance(v, str) and 1 <= len(v) <= 40 and not THREE_DIGITS.search(v):
                    if k not in distinct:
                        distinct[k] = []
                        present[k] = 0
                        lens[k] = []
                        order.append(k)
                    if v not in distinct[k]:
                        distinct[k].append(v)
                    present[k] += 1
                    lens[k].append(len(v))
        total = max(1, len(self.rows))
        labels = self.enum_labels()
        best, best_score = None, None
        for k in order:
            if present[k] < 0.8 * total or len(distinct[k]) > 40:
                continue
            share = sum(1 for v in distinct[k] if v.strip().lower() in labels) / float(len(distinct[k]))
            score = (share, -len(distinct[k]), present[k], sum(lens[k]) / float(len(lens[k])))
            if best_score is None or score > best_score:
                best, best_score = k, score
        self.cat_cache = (best,)
        return best

    def label_matches(self, label):
        lf = word_form(label)
        ls = set(w for w in stems(label) if len(w) >= 4)
        for it in self.items:
            if lf == word_form(it.shelf) or lf == word_form(it.name):
                return True
            s = set(w for w in (stems(it.shelf) | stems(it.name)) if len(w) >= 4)
            if ls & s:
                return True
        return bool(self.label_owner(label))

    def key_table(self):
        ck = self.get_cat_key()
        groups, gorder = {}, []
        for lid in self.seen:
            rec = self.full.get(lid)
            if rec is None:
                continue
            blk = attr_of(rec) or {}
            lab = str(rec.get(ck)) if ck and rec.get(ck) is not None else "listings"
            if lab not in groups:
                groups[lab] = {"n": 0, "slots": {}, "order": []}
                gorder.append(lab)
            g = groups[lab]
            g["n"] += 1
            seenwf = set()
            for k, v in blk.items():
                wf = word_form(k)
                if not wf or wf in seenwf:
                    continue
                seenwf.add(wf)
                if wf not in g["slots"]:
                    g["slots"][wf] = KeySlot(wf)
                    g["order"].append(wf)
                slot = g["slots"][wf]
                slot.count += 1
                shown = str(v).strip().lower()
                first = FIRST_SPELLING.get(v, v) if isinstance(v, str) else v
                measured = str(first).strip().lower()
                slot.per[lid] = (shown, measured)
                if shown not in slot.distinct:
                    slot.distinct.append(shown)
        labs = [l for l in gorder if self.label_matches(l)] or list(gorder)
        labs.sort(key=lambda l: -groups[l]["n"])
        out = []
        for lab in labs:
            g = groups[lab]
            slots = [g["slots"][wf] for wf in g["order"]]
            slots.sort(key=lambda s: (-s.count, len(s.wf)))
            kept = []
            for slot in slots:
                vals = [m for (s, m) in slot.per.values()]
                if not vals:
                    continue
                mean = sum(len(v) for v in vals) / float(len(vals))
                if mean > 40 or sum(1 for v in vals if SPACED_SLASH.search(v)) * 2 >= len(vals):
                    continue
                if len(vals) >= 4 and len(set(vals)) == len(vals) and mean > 12:
                    sharing = 0
                    for i, v in enumerate(vals):
                        rest = set()
                        for j, w in enumerate(vals):
                            if j != i:
                                rest |= set(LETTER_RUN.findall(w))
                        if set(LETTER_RUN.findall(v)) & rest:
                            sharing += 1
                    if sharing * 2 < len(vals):
                        continue
                merged = False
                for kk in kept:
                    if self.twin_keys(kk, slot):
                        for lid, sv in slot.per.items():
                            if lid not in kk.per:
                                kk.per[lid] = sv
                        for s in slot.distinct:
                            if s not in kk.distinct:
                                kk.distinct.append(s)
                        kk.count = len(kk.per)
                        merged = True
                        break
                if not merged:
                    kept.append(slot)
            kept.sort(key=lambda s: -s.count)
            out.append((lab, g["n"], kept[:KEY_TABLE_SIZE]))
        return out

    def twin_keys(self, a, b):
        wa, wb = set(a.wf.split()), set(b.wf.split())
        if not wa or not wb or not (wa & wb):
            return False

        def agree(x, y):
            if x == y:
                return True
            xw, yw = ALNUM.findall(x), ALNUM.findall(y)
            if len(xw) <= len(yw):
                sh, lo = xw, yw
            else:
                sh, lo = yw, xw
            if not sh:
                return False
            return all(w in set(lo) for w in sh) and len(lo) <= 2 * len(sh) + 2
        shared = [lid for lid in a.per if lid in b.per and a.per[lid][1].strip() and b.per[lid][1].strip()]
        agreeing = sum(1 for lid in shared if agree(a.per[lid][1], b.per[lid][1]))
        if not (wa <= wb or wb <= wa):
            return len(shared) >= 2 and agreeing == len(shared)
        if shared:
            return agreeing * 2 > len(shared)
        return bool(set(x for x in a.distinct if x) & set(x for x in b.distinct if x))

    def not_stated(self, label, wf, shown):
        owner = self.label_owner(label)
        if not owner:
            return ""
        sw = set(ALNUM3.findall(" ".join(shown).lower()))
        for r in self.items[owner - 1].live():
            prop, _, val = r.partition(":")
            if E2_COMPARE.search(val):
                continue
            vw = ALNUM3.findall(val.lower())
            if prop.strip() and word_form(prop) != wf and (stems(prop) & stems(wf)) and vw and not all(w in sw for w in vw):
                return " [not the stated %s: another property]" % prop.strip()
        return ""

    def item_groups(self):
        ck = self.get_cat_key()
        groups, order = {}, []
        for lid in self.seen:
            row = self.rows.get(lid) or {}
            lab = str(row.get(ck)) if ck and row.get(ck) is not None else ""
            if lab not in groups:
                groups[lab] = []
                order.append(lab)
            groups[lab].append(lid)
        pool = [l for l in order if self.label_matches(l)] or list(order)
        pool.sort(key=lambda l: -len(groups[l]))
        return [(l, groups[l]) for l in pool[:max(1, len(self.items))]]

    def single_target(self, tool):
        sch = self.schemas.get(tool)
        if not isinstance(sch, dict):
            return None
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        texts = [p for p, s in props.items() if isinstance(s, dict) and "string" in kinds(s) and not s.get("enum")]
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        target = None
        for r in req:
            if r in texts:
                target = r
                break
        if target is None and len(texts) == 1:
            target = texts[0]
        if target is None:
            return None
        if any(r != target for r in req):
            return None
        return target

    def price_of(self, lid):
        if not self.price_key:
            return None
        row = self.rows.get(lid) or {}
        v = row.get(self.price_key)
        if isinstance(v, dict):
            for x in v.values():
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    return float(x)
            return None
        if isinstance(v, str):
            m = PRICE_TEXT.fullmatch(v)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except Exception:
                    return None
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return None

    def sold_out(self, lid):
        return lid in self.unavailable or lid in self.gone

    def buyable(self, lid):
        if self.sold_out(lid):
            return False
        if self.stock_key:
            v = (self.rows.get(lid) or {}).get(self.stock_key)
            if isinstance(v, str):
                s = v.strip()
                if V4_OUT_WHOLE.fullmatch(s) or V4_OUT_ANY.search(s):
                    return False
            if v is False:
                return False
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v <= 0:
                return False
        return True

    def listing_records(self, obs):
        out = []
        if isinstance(obs, dict):
            lid = self.identity(obs, False, 0)
            if lid:
                return [(lid, obs)]
            for v in obs.values():
                if isinstance(v, list):
                    for e in v:
                        i = self.identity(e, True, 2) if isinstance(e, dict) else None
                        if i:
                            out.append((i, e))
                elif isinstance(v, dict):
                    i = self.identity(v, False, 1)
                    if i:
                        out.append((i, v))
                    else:
                        for vv in v.values():
                            if isinstance(vv, list):
                                for e in vv:
                                    j = self.identity(e, True, 3) if isinstance(e, dict) else None
                                    if j:
                                        out.append((j, e))
        elif isinstance(obs, list):
            for e in obs:
                i = self.identity(e, True, 1) if isinstance(e, dict) else None
                if i:
                    out.append((i, e))
        return out

    def record_view_keys(self, tool, obs):
        for lid, rec in self.listing_records(obs):
            keys = set(rec.keys())
            self.tool_keys.setdefault(tool, set()).update(keys)
            vk = self.view_keys.get(lid)
            if vk and vk[0] == self.version:
                vk[1].update(keys)
            else:
                self.view_keys[lid] = [self.version, set(keys)]

    def shown_in_full(self, tool, lid):
        tk = self.tool_keys.get(tool)
        vk = self.view_keys.get(lid)
        return bool(tk) and bool(vk) and vk[0] == self.version and tk <= vk[1] and lid not in self.gone

    def shape_of(self, tool, obs):
        if tool in self.shapes:
            return self.shapes[tool]
        if not isinstance(obs, dict):
            return None
        cands = [k for k, v in obs.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(e, dict) for e in v)]
        if not cands:
            return None
        self.shapes[tool] = None
        sample = {k: (v[:3] if k in cands else v) for k, v in obs.items()}
        prompt = (SHAPE_PROMPT + "\n\nTOOL: %s | %s\n\nRESULT (shortened to its first rows):\n" % (tool, dump(self.schemas.get(tool)))
                  + dump(clip(sample, 120))[:4000])
        ans = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        rk = ans.get("rows")
        if not isinstance(rk, str) or rk not in cands:
            return None
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        first = obs[rk][0]
        sh = {"rows": rk, "count": None, "next": None, "next_arg": None, "price": None, "stock": None}
        c = ans.get("count")
        if isinstance(c, str) and c in obs and isinstance(obs[c], int) and not isinstance(obs[c], bool):
            sh["count"] = c
        nx = ans.get("next")
        if isinstance(nx, str) and nx in obs:
            sh["next"] = nx
        na = ans.get("next_arg")
        if isinstance(na, str) and isinstance(props, dict) and na in props:
            sh["next_arg"] = na
        for f in ("price", "stock"):
            v = ans.get(f)
            if isinstance(v, str) and v in first:
                sh[f] = v
        self.shapes[tool] = sh
        if sh["price"] and not self.price_key:
            self.price_key = sh["price"]
        if sh["stock"] and not self.stock_key:
            self.stock_key = sh["stock"]
        self.trace.append({"kind": "shape", "tool": tool, "shape": sh})
        return sh

    def paging_plan(self, tool, obs):
        sh = self.shape_of(tool, obs)
        if not sh or not sh.get("count") or not sh.get("next") or not sh.get("next_arg") or not isinstance(obs, dict):
            return None
        rows = obs.get(sh["rows"])
        total = obs.get(sh["count"])
        nv = obs.get(sh["next"])
        if not isinstance(rows, list) or not rows or not isinstance(total, int) or isinstance(total, bool):
            return None
        if nv is not None and (isinstance(nv, bool) or not isinstance(nv, (str, int))):
            return None
        return (total, len(rows), sh["next_arg"], nv)

    def price_sorted(self, obs):
        recs = self.listing_records(obs)
        prices = [self.price_of(lid) for lid, _ in recs]
        if len(prices) < 3 or any(p is None for p in prices):
            return False
        return all(prices[i] <= prices[i + 1] for i in range(len(prices) - 1))

    def auto_pages(self, tool, args, obs):
        self.paged = []
        pick = self.brief.get("pick")
        if pick not in ("most", "least", "cheapest"):
            return ""
        plan = self.paging_plan(tool, obs)
        if not plan:
            return ""
        total, shown, narg, nval = plan
        sent = args.get(narg)
        if isinstance(sent, bool):
            later = sent
        elif isinstance(sent, int):
            later = sent not in (0, 1)
        elif isinstance(sent, float):
            later = True
        else:
            later = bool(sent)
        if nval is None or later or total <= shown:
            return ""
        if pick == "cheapest" and self.price_sorted(obs):
            return ""
        need = -(-(total - shown) // shown)
        R = self.reserve()
        if need > 2 and self.pages_read + 2 <= AUTO_PAGES and self.steps_left() - R - 2 >= 12:
            need = 2
        if need > 2 or self.pages_read + need > AUTO_PAGES or self.steps_left() - R - need < 12:
            return ("[Cairn] This search matched %d listings but displayed only %d, and not sorted by price; the pick "
                    "weighs every listing that fits. Search again with tighter words, those words being the value the "
                    "requirement names, or request the next page by sending %s set to %s."
                    % (total, shown, narg, dump(nval)))
        lines, read = [], 0
        for _ in range(need):
            a = dict(args)
            a[narg] = nval
            f = []
            a = shape(a, self.schemas.get(tool), "args", f)
            if f:
                break
            r = self.env_call(tool, a)
            if r is None or self.finished:
                break
            self.pages_read += 1
            read += 1
            o = r.get("observation")
            self.observe(tool, a, o)
            self.paged.append(o)
            rend = self.display(o)
            self.read_cache[tool + dump(a)] = (self.version, self.steps, rend)
            lines.append("%s=%s: %s" % (narg, dump(nval), rend))
            p2 = self.paging_plan(tool, o)
            if not p2 or p2[3] is None:
                nval = None
                break
            nval = p2[3]
        self.trace.append({"kind": "auto_pages", "read": read})
        if not read:
            return ""
        out = "[Cairn fetched the following pages of this search by itself, one step per page]\n" + "\n".join(lines)
        if nval is not None:
            out += ("\n[Cairn] The search matched %d listings and Cairn has now read %d rows. Make the search tighter, "
                    "using the value the requirement names, or keep requesting pages, so that the remaining listings "
                    "come into view." % (total, shown * (read + 1)))
        return out

    def cap_ignored(self, tool, args, obs):
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        if not isinstance(props, dict) or not self.price_key:
            return ""
        ranked = self.cap_args(tool)

        def priced(k):
            d = props.get(k, {}).get("description", "") if isinstance(props.get(k), dict) else ""
            return bool(E2_PRICED.search(str(k) + " " + str(d)))
        caps = []
        for k, v in args.items():
            if k not in props or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if (E2_CAPWORD.search(k) and priced(k)) or (len(ranked) == 1 and ranked[0] == k):
                caps.append((v, k))
        if not caps:
            return ""
        cap, cname = min(caps, key=lambda p: p[0])
        over = [lid for lid, _ in self.listing_records(obs) if (self.price_of(lid) or 0) > cap]
        if not over:
            return ""
        others = []
        for k, v in args.items():
            sch = props.get(k)
            if isinstance(v, str) and isinstance(sch, dict) and isinstance(sch.get("enum"), list):
                opts = [o for o in sch["enum"] if o != v]
                if opts:
                    others.append("%s=%s" % (k, dump(opts[0])))
        out = "[Cairn] This search did not honour %s at %s: %d of its rows are priced above it." % (cname, gnum(cap), len(over))
        if others:
            out += (" Browse the category with %s, keeping the same %s, so that only the cheaper listings show."
                    % (", ".join(others), cname))
        return out

    def cheapest_here(self, obs):
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        ids = []
        for o in [obs] + list(self.paged):
            for lid, _ in self.listing_records(o):
                if lid not in ids:
                    ids.append(lid)
        keep = [lid for lid in ids if self.buyable(lid) and self.price_of(lid) is not None]
        if len(keep) < 2:
            return ""
        keep.sort(key=lambda x: (self.price_of(x), x))
        return "[Cairn] Buyable rows here from the lowest price up: " + ", ".join(
            "%s (%s)" % (x, gnum(self.price_of(x))) for x in keep[:6])

    def wider_hold(self):
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.steps_left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = ["%s at %s" % (x, gnum(self.price_of(x))) for x in self.cart_ids() if self.price_of(x) is not None]
        out = ("[Cairn] This order call was not executed yet and spent no step. %d steps remain and the pick is "
               "cheapest. A search may match just a portion of a listing's text, so searching by a requirement's value "
               "surfaces just part of the listings that fit, and rows also arrive unordered by price. Ahead of any "
               "checkout, for each item, run a few extra searches under different product names: the product's bare "
               "one-word name, its synonyms, and words lifted from titles of fitting listings already seen. Consider only "
               "rows that can be bought and cost less than that item's current cart line" % self.steps_left())
        if lines:
            out += " [" + "; ".join(lines) + "]"
        out += (". Then check the attribute records of the lowest-priced of those rows, replace the cart line with any "
                "one that fits, and then check out.")
        return out

    def set_facts(self, tool, val):
        self.facts[tool] = val

    def note_facts(self, tool, obs):
        if attr_of(obs) is not None:
            self.set_facts(tool, True)
        elif self.facts.get(tool) is not True:
            self.bare[tool] = self.bare.get(tool, 0) + 1
            if self.bare[tool] >= 2:
                self.set_facts(tool, False)

    def lists_things(self, tool):
        st = self.single_target(tool)
        if st and (V10_SHELF.search(under_form(st)) or V14_QUERY.search(under_form(st))):
            return True
        return tool in self.shapes


LOW4 = re.compile(r"[a-z]{4,}")
NAME_WORD = re.compile(r"[a-z][a-z'\-]{2,}")


class E3(object):
    def soft(self, name, fallback, fn, *args):
        try:
            return fn(*args)
        except Exception as e:
            self.trace.append({"kind": "soft_crash", "method": name, "error": ("%s: %s" % (type(e).__name__, e))[:300]})
            return [] if fallback is list else fallback

    def fill_facts(self, ids, reserve):
        ft = self.facts_tool()
        tgt = self.single_target(ft) if ft else None
        cmp = self.role("compare")
        if (not ft or not tgt) and self.compare_facts() is not True:
            return []
        wanted = [x for x in ids if x in self.rows and x not in self.full and not self.sold_out(x)]
        texts, unread = [], []
        ctgt = self.compare_target(cmp) if cmp else None
        while (cmp and ctgt and self.compare_facts() is True and len(wanted) >= 2 and self.steps_left() > reserve
               and not self.finished and not self.rushed()):
            room = self.compare_room()
            batch, wanted = wanted[:room], wanted[room:]
            f = []
            a = shape({ctgt: batch}, self.schemas.get(cmp), "args", f)
            if f:
                wanted = batch + wanted
                break
            r = self.env_call(cmp, a)
            if r is None:
                wanted = batch + wanted
                break
            o = r.get("observation")
            self.observe(cmp, a, o)
            self.record_view_keys(cmp, o)
            recs = self.listing_records(o)
            for lid, rec in recs:
                texts.append("%s: %s" % (lid, self.display(rec)[:1500]))
            if recs:
                self.compare_ok = max(self.compare_ok, len(recs))
                unread += [x for x in batch if x not in self.full]
                continue
            self.compare_bad = min(self.compare_bad or 999, len(batch))
            wanted = batch + wanted
            if self.compare_room() >= len(batch):
                break
        if not ft or not tgt:
            if texts:
                self.trace.append({"kind": "fill_facts", "texts": len(texts)})
            return texts
        order = unread + [x for x in wanted if x not in unread]
        for lid in order:
            if self.steps_left() <= reserve or self.finished or self.facts.get(ft) is False or self.rushed():
                break
            if (ft, lid) in self.factless:
                continue
            f = []
            a = shape({tgt: lid}, self.schemas.get(ft), "args", f)
            if f:
                break
            r = self.env_call(ft, a)
            if r is None:
                break
            o = r.get("observation")
            self.observe(ft, a, o)
            self.record_view_keys(ft, o)
            if isinstance(o, dict) and self.identity(o):
                self.note_facts(ft, o)
            texts.append("%s: %s" % (lid, self.display(o)[:1500]))
        if texts:
            self.trace.append({"kind": "fill_facts", "texts": len(texts)})
        return texts

    def show_links(self, ids, reserve):
        tpl = self.value_template()
        if not tpl or (len(self.items) < 2 and len(ids) < 2):
            return
        tool, fixed, field = tpl
        for lid in ids:
            fi = self.fit_index()
            code = (fi.get(lid) or {}).get("pointer")
            if not code or code in self.link_tried or any(m.get("own") == code for m in fi.values()):
                continue
            if self.steps_left() <= reserve or self.finished or self.rushed():
                return
            self.link_tried.add(code)
            a = dict(fixed)
            a[field] = code
            f = []
            a = shape(a, self.schemas.get(tool), "args", f)
            if f:
                return
            r = self.env_call(tool, a)
            if r is None:
                return
            self.observe(tool, a, r.get("observation"))
            self.trace.append({"kind": "link", "code": code})
            fi = self.fit_index()
            hits = [x for x, m in fi.items() if m.get("own") == code]
            self.fill_facts(hits, max(reserve, self.reserve() + 6))

    def cap_args(self, tool):
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        if not isinstance(props, dict):
            return []
        cands = [k for k, s in props.items() if isinstance(s, dict) and ("integer" in kinds(s) or "number" in kinds(s))
                 and not V7_NONPRICE.search(k) and not V9_LOWER.search(under_form(k))]
        ev = self.ceiling_evidence(tool)
        proved = [k for k in cands if any(v == "holds" and a == k for (a, m), v in ev.items())]
        maxn = [k for k in cands if V6_CAP.search(under_form(k))]
        money = [k for k in cands if V8_PRICE.search(under_form(k))]
        both = [k for k in maxn if k in money]
        return proved or both or money or maxn or []

    def label_of(self, lid):
        ck = self.get_cat_key()
        if not ck:
            return None
        v = (self.rows.get(lid) or {}).get(ck)
        return v if v is not None else None

    def ceiling_evidence(self, tool):
        ck = self.get_cat_key()
        pools = {}
        for lid in self.seen:
            p = self.price_of(lid)
            if p is None:
                continue
            lab = str(self.label_of(lid)).strip().lower() if ck else ""
            pools.setdefault(lab, []).append(p)
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        if not isinstance(props, dict):
            return {}
        enum_args = [k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list)]
        ev = {}
        for cm in list(self.mem_args):
            t, args = self.mem_args[cm]
            if t != tool:
                continue
            recs = self.listing_records(self.mem_results.get(cm))
            if not recs or any(self.price_of(lid) is None for lid, _ in recs):
                continue
            mode = tuple(sorted(((k, args[k]) for k in enum_args if k in args and isinstance(args[k], (str, int, float, bool))),
                                key=lambda p: p[0]))
            prices = [self.price_of(lid) for lid, _ in recs]
            dearest = max(prices)
            pool = []
            if ck:
                labs = set(str(self.label_of(lid)).strip().lower() for lid, _ in recs)
                for lab in labs:
                    pool += pools.get(lab, [])
            for k, v in args.items():
                if k not in props or not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                key = (k, mode)
                if dearest > v:
                    ev[key] = "fails"
                elif ev.get(key) != "fails" and pool:
                    share = sum(1 for p in pool if p <= v) / float(len(pool))
                    if share ** len(recs) < PRICE_CHANCE:
                        ev[key] = "holds"
        return ev

    def capped_template(self):
        for t in self.tools:
            tn = t["function"]["name"]
            props = (self.schemas.get(tn) or {}).get("properties") or {}
            if not isinstance(props, dict):
                continue
            caps = self.cap_args(tn)
            shelves = [k for k, s in props.items() if isinstance(s, dict) and "string" in kinds(s) and not s.get("enum")
                       and V10_SHELF.search(under_form(k))]
            if len(caps) != 1 or len(shelves) != 1:
                continue
            cap, shelf = caps[0], shelves[0]
            ev = self.ceiling_evidence(tn)
            learned = [(dict(m), v) for (a, m), v in ev.items() if a == cap]
            enum_args = [k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2]
            req = (self.schemas.get(tn) or {}).get("required") or []
            single = len(enum_args) == 1 and enum_args[0] in req

            def worded(sa):
                return any(isinstance(v, str) and v.strip() and V14_QUERY.search(under_form(k))
                           for k, v in sa.items() if k != shelf and k not in enum_args)
            chosen = {}
            for k in enum_args:
                opts = props[k]["enum"]
                val = None
                for m, vd in learned:
                    if vd == "holds" and k in m:
                        val = m[k]
                        break
                if val is None:
                    for sa in reversed(self.searched):
                        if shelf in sa and k in sa and not worded(sa):
                            val = sa[k]
                            break
                if val is None:
                    lm = [o for o in opts if isinstance(o, str) and V11_MODE.search(o)]
                    if len(lm) == 1:
                        val = lm[0]
                if val is None and single:
                    wv = [sa[k] for sa in self.searched if k in sa and worded(sa)]
                    never = [o for o in opts if o not in wv]
                    if wv and len(never) == 1:
                        val = never[0]
                if single:
                    spoiled = [m[k] for m, vd in learned if vd == "fails" and k in m]
                    if spoiled and (val is None or val in spoiled):
                        rest = [o for o in opts if o not in spoiled]
                        val = rest[0] if len(rest) == 1 else None
                if val is None:
                    return None
                chosen[k] = val
            return (tn, chosen, shelf, cap)
        return None

    def list_below(self, lines):
        tpl = self.capped_template()
        ck = self.get_cat_key()
        if not tpl or not ck:
            return
        tool, opts, shelf, capk = tpl
        for lid in lines:
            p = self.price_of(lid)
            lab = self.label_of(lid)
            if p is None or p < 2 or not lab or (lid, p) in self.capped:
                continue
            self.capped.add((lid, p))
            ceil = int(p) - 1 if p == int(p) else int(p)
            used = set()
            for i in range(WALK_PER_LINE):
                need = self.reserve() + (7 if i == 0 else WALK_RESERVE)
                if self.steps_left() <= need or self.finished or self.rushed():
                    return
                if ceil < 1 or ceil in used:
                    break
                used.add(ceil)
                a = dict(opts)
                a[shelf] = lab
                a[capk] = ceil
                f = []
                a = shape(a, self.schemas.get(tool), "args", f)
                if f:
                    return
                r = self.env_call(tool, a)
                if r is None:
                    return
                o = r.get("observation")
                self.observe(tool, a, o)
                recs = self.listing_records(o)
                self.trace.append({"kind": "list_below", "line": lid, "price": p, "ceiling": ceil, "rows": len(recs)})
                plan = self.paging_plan(tool, o)
                if (not recs or any((self.price_of(x) or 0) > ceil for x, _ in recs) or not plan or plan[0] <= plan[1]
                        or self.price_sorted(o)):
                    break
                if plan[0] <= 2 * plan[1]:
                    if plan[3] is not None:
                        a2 = dict(a)
                        a2[plan[2]] = plan[3]
                        f = []
                        a2 = shape(a2, self.schemas.get(tool), "args", f)
                        if not f and self.steps_left() > self.reserve() + 7 and not self.rushed():
                            r2 = self.env_call(tool, a2)
                            if r2 is not None:
                                self.observe(tool, a2, r2.get("observation"))
                    break
                known = [self.price_of(x) for x, _ in recs if self.price_of(x) is not None]
                if not known:
                    break
                m = min(known)
                ceil = int(m) - 1 if m == int(m) else int(m)

    def cheaper_pool(self, lid):
        p = self.price_of(lid)
        lab = str(self.label_of(lid) or "").strip().lower()
        ids = self.cart_ids()
        pool = [x for x in self.seen if x not in ids and x not in self.full and self.buyable(x)
                and str(self.label_of(x) or "").strip().lower() == lab and self.price_of(x) is not None
                and self.price_of(x) < p]
        pool.sort(key=lambda x: self.price_of(x))
        return pool

    def cheaper_rows(self):
        if self.brief.get("pick") != "cheapest":
            return []
        mark = self.pass_mark()
        if not self.get_cat_key() or mark in self.cheaper_marks:
            return []
        self.cheaper_marks.add(mark)
        self.list_below(self.cart_ids())
        room = self.compare_room() if self.compare_facts() is True else 0
        ids = self.cart_ids()
        picks = []
        for lid in ids:
            if self.price_of(lid) is None or not self.label_of(lid):
                continue
            pool = self.cheaper_pool(lid)
            vr = [x for x in pool if x in self.value_rows]
            if vr:
                pool.remove(vr[0])
                pool.insert(0, vr[0])
            n, added = room or 2, 0
            for x in pool:
                if added >= n:
                    break
                if x not in picks:
                    picks.append(x)
                    added += 1
        cap = room * len(ids) if room else min(6, max(1, 3 * len(ids)))
        return self.fill_facts(picks[:cap], self.reserve() + 6)

    def below_cart_line(self):
        return self.soft("below_cart_line", "", self._below_cart_line)

    def _below_cart_line(self):
        if not self.get_cat_key():
            return ""
        entries = []
        for lid in self.cart_ids():
            if self.price_of(lid) is None or not self.label_of(lid):
                continue
            for x in self.cheaper_pool(lid)[:CHEAPER_SHOWN]:
                entries.append("%s at %s" % (x, gnum(self.price_of(x))))
        if not entries:
            return ""
        return "Not read yet and priced below a line of the cart: " + ", ".join(entries) + "."

    def get_title_key(self):
        ck = self.get_cat_key()
        stats, order = {}, []
        for row in self.rows.values():
            for k, v in row.items():
                if k == ck or not isinstance(v, str) or len(v) > TITLE_LIMIT:
                    continue
                if k not in stats:
                    stats[k] = [0, 0, 0, set()]
                    order.append(k)
                s = stats[k]
                s[0] += len(v.split())
                s[1] += 1
                s[2] += len(v)
                s[3].add(v)
        total = len(self.rows)
        keep = [k for k in order if stats[k][1] * 5 >= total * 4 and stats[k][0] >= 2 * stats[k][1]
                and len(stats[k][3]) * 2 >= stats[k][1]]
        if not keep:
            self.title_key = None
            return None
        most = max(len(stats[k][3]) for k in keep)
        keep = [k for k in keep if len(stats[k][3]) * 10 >= most * 9]
        keep.sort(key=lambda k: (stats[k][2] / float(stats[k][1]), str(k)))
        self.title_key = keep[0]
        return keep[0]

    def hunt_words(self, num):
        it = self.items[num - 1]
        ck = self.get_cat_key()
        tk = self.get_title_key()
        labels, names = [], []
        if ck:
            for lid in self.seen:
                lab = self.label_of(lid)
                if lab is not None and self.owns_label(str(lab), num):
                    labels.append(str(lab))
                    if tk:
                        tv = (self.rows.get(lid) or {}).get(tk)
                        if isinstance(tv, str):
                            names.append(tv)
            for lid in self.seen:
                if lid in self.full:
                    lab = self.label_of(lid)
                    if lab is not None and self.owns_label(str(lab), num):
                        blk = attr_of(self.full[lid]) or {}
                        names.append(" ".join(v for v in blk.values() if isinstance(v, str) and len(v) <= HUNT_VALUE_LEN))
        other = set()
        for j, o in enumerate(self.items, 1):
            if j != num:
                other |= set(LOW4.findall((o.shelf + " " + o.name).lower()))
        if labels:
            cnt, first = {}, []
            for l in labels:
                if l not in cnt:
                    cnt[l] = 0
                    first.append(l)
                cnt[l] += 1
            home = max(first, key=lambda l: cnt[l]).lower()
        else:
            home = (it.shelf or it.name).lower()
        cands = [w for w in LOW4.findall(home) if w not in other]
        noun = ""
        for w in cands:
            if len(w) > len(noun):
                noun = w
        if num in self.hunt_nonoun:
            noun = ""
        if not names:
            return noun, []
        counts = {}
        for n in names:
            for w in set(NAME_WORD.findall(n.lower())):
                counts[w] = counts.get(w, 0) + 1
        own = roots(it.shelf) | roots(it.name)
        ceil = max(2, int(HUNT_COMMON * len(names)))
        words = [w for w, c in counts.items() if 2 <= c <= ceil and w != noun and roots(w) and not (roots(w) & own)]
        words.sort(key=lambda w: (-counts[w], w))
        return noun, words

    def item_rows(self, num):
        if not self.get_cat_key():
            return len(self.rows)
        return sum(1 for lid in self.seen if self.label_of(lid) is not None and self.owns_label(str(self.label_of(lid)), num))

    def cheaper_hunt(self):
        return self.soft("cheaper_hunt", "", self._cheaper_hunt)

    def _cheaper_hunt(self):
        mark = dump([[it.live() for it in self.items], self.budget, sorted(self.gone)])
        if self.brief.get("pick") != "cheapest" or self.hunts >= HUNT_ROUNDS or mark == self.hunt_mark:
            return ""
        tpl = self.value_template()
        if not tpl or not self.price_key or self.clock() > HUNT_DEADLINE:
            return ""
        room = min(HUNT_CAP, self.steps_left() - self.reserve() - HUNT_RESERVE - 2 * len(self.items))
        if room < 3:
            return ""
        self.hunts += 1
        self.hunt_mark = mark
        share = max(3, room // len(self.items))
        spent = gain = 0
        for i in range(1, len(self.items) + 1):
            if spent >= room or self.finished:
                break
            s, g = self.item_hunt(i, tpl, min(share, room - spent))
            spent += s
            gain += g
        self.trace.append({"kind": "cheaper_hunt", "steps": spent, "new": gain, "cart": self.cart_mark()})
        if spent:
            self.wider_held = True
            return ("Cairn itself ran %d extra searches chasing a lower price, one step each, and they brought up %d new rows."
                    % (spent, gain))
        return ""

    def item_hunt(self, num, tpl, room):
        tool, fixed, field = tpl
        tried = self.hunt_tried_for(num)
        noun, words = self.hunt_words(num)
        vals = [v for v, flag in self.value_words(num)]
        st = {"spent": 0, "gain": 0, "idle": 0, "refused": False}

        def run(q):
            tried.add(q.lower())
            a = dict(fixed)
            a[field] = q
            f = []
            a = shape(a, self.schemas.get(tool), "args", f)
            if f:
                st["refused"] = True
                return False, False
            if st["spent"] >= room or self.finished:
                return False, False
            holds = bool(vals) and all(v.lower() in q.lower() for v in vals)
            visited, found = set(), False
            while not self.finished and self.clock() <= HUNT_DEADLINE + HUNT_GRACE:
                cm = self.canon(tool, a)
                if cm in visited:
                    break
                visited.add(cm)
                if cm in self.mem_results:
                    o = self.mem_results[cm]
                    new = HUNT_YIELD
                else:
                    if st["spent"] >= room:
                        break
                    before = self.item_rows(num)
                    r = self.env_call(tool, a)
                    if r is None:
                        break
                    st["spent"] += 1
                    o = r.get("observation")
                    self.observe(tool, a, o)
                    new = self.item_rows(num) - before
                    st["gain"] += new
                    st["idle"] = 0 if new else st["idle"] + 1
                recs = self.listing_records(o)
                if recs:
                    found = True
                    if holds:
                        self.value_rows |= set(x for x, _ in recs)
                plan = self.paging_plan(tool, o)
                if not plan or plan[3] is None or (new < HUNT_YIELD and plan[0] > HUNT_WHOLE):
                    break
                a2 = dict(a)
                a2[plan[2]] = plan[3]
                f = []
                a2 = shape(a2, self.schemas.get(tool), "args", f)
                if f:
                    break
                a = a2
            return True, found
        key = (num, tuple(vals))
        if key not in self.hunt_stems:
            tries = []
            if vals:
                tries += [(" ".join(vals) + " " + noun).strip(), " ".join(vals).strip()]
            if len(vals) >= 2:
                tries.append((" ".join(vals[:-1]) + " " + noun).strip())
            if noun:
                tries.append(noun.strip())
            seen_t, stem = [], ""
            for q in tries:
                if q in seen_t:
                    continue
                seen_t.append(q)
                issued, found = run(q)
                if not issued:
                    return st["spent"], st["gain"]
                if found:
                    stem = q
                    break
                if noun and q == noun.strip():
                    self.hunt_nonoun.add(num)
                    noun = ""
            self.hunt_stems[key] = stem
        stem = self.hunt_stems[key]
        while (st["spent"] < room and st["idle"] < HUNT_IDLE and not st["refused"] and not self.finished
               and self.clock() <= HUNT_DEADLINE + HUNT_GRACE):
            noun, words = self.hunt_words(num)
            sw = set(stem.lower().split())
            cand = [w for w in words if (w + " " + stem).strip().lower() not in tried and w not in sw]
            if not cand:
                break
            issued, found = run((cand[0] + " " + stem).strip())
            if not issued:
                break
        return st["spent"], st["gain"]

    def hunt_tried_for(self, num):
        if not isinstance(self.hunt_tried, dict):
            self.hunt_tried = {}
        return self.hunt_tried.setdefault(num, set())

    def fit_hunt(self):
        return self.soft("fit_hunt", list, self._fit_hunt)

    def _fit_hunt(self):
        tpl = self.value_template()
        if self.fit_hunted or self.clock() > HUNT_DEADLINE or not tpl:
            return []
        res = self.reserve() + len(self.items) + 1
        spare = self.steps_left() - res
        if spare < 3:
            return []
        self.fit_hunted = True
        covered = set()
        for lid, u in self.held_units().items():
            if u >= 1 and self.label_of(lid) is not None:
                covered.add(str(self.label_of(lid)))
        hunted = [i for i in range(1, len(self.items) + 1) if not any(self.owns_label(l, i) for l in covered)]
        if not hunted:
            hunted = list(range(1, len(self.items) + 1))
        wanted = {}
        for i in hunted:
            toks = []
            for v, flag in self.value_words(i):
                toks += value_tokens(v)
            wanted[i] = toks
        walked = self.unread_walk(self.role("search"), wanted, min(FIT_PAGES, spare // 2))
        room = min(HUNT_CAP, self.steps_left() - self.reserve() - HUNT_RESERVE)
        tool, fixed, field = tpl
        spent = 0
        if room >= 3:
            for i in hunted:
                vals = [v for v, flag in self.value_words(i)]
                noun, words = self.hunt_words(i)
                qs = []
                for s in self.plan_series(i)[:SERIES_HUNTED]:
                    qs.append(("%s %s %s" % (s, vals[0] if vals else "", noun)).strip())
                for v in vals:
                    qs.append((v + " " + noun).strip())
                if noun:
                    for v in vals:
                        qs.append(v.strip())
                for w in words[:FIT_WORDS]:
                    qs.append((w + " " + noun).strip())
                if noun:
                    qs.append(noun.strip())
                uniq = []
                for q in qs:
                    if q and q not in uniq:
                        uniq.append(q)
                share = max(2, room // len(hunted))
                used = 0
                for q in uniq:
                    if spent >= room or used >= share or self.finished or self.clock() > HUNT_DEADLINE + HUNT_GRACE:
                        break
                    a = dict(fixed)
                    a[field] = q
                    f = []
                    a = shape(a, self.schemas.get(tool), "args", f)
                    if f:
                        continue
                    r = self.env_call(tool, a)
                    if r is None:
                        return []
                    spent += 1
                    used += 1
                    self.observe(tool, a, r.get("observation"))
        ck = self.get_cat_key()
        picks = []
        for i in hunted:
            toks = set(wanted[i])
            pool = []
            for x in self.seen:
                if x in self.full or not self.buyable(x):
                    continue
                if ck and not (self.label_of(x) is not None and self.owns_label(str(self.label_of(x)), i)):
                    continue
                if isinstance(self.budget, (int, float)) and (self.price_of(x) or 0) > self.budget:
                    continue
                pool.append(x)
            pool.sort(key=lambda x: (-len(toks & text_words(dump(self.rows.get(x)))),
                                     self.price_of(x) if self.price_of(x) is not None else float("inf")))
            for x in pool[:FIT_OPENS]:
                if x not in picks:
                    picks.append(x)
        texts = []
        for k in range(0, len(picks), 4):
            if self.clock() > HUNT_DEADLINE + HUNT_GRACE:
                break
            texts += self.fill_facts(picks[k:k + 4], res)
        self.trace.append({"kind": "fit_hunt", "pages": walked, "searches": spent, "opened": len(texts)})
        return texts

    def unread_walk(self, tool, wanted, room):
        if not tool or not wanted:
            return 0
        spent = 0
        share = max(1, room // len(wanted))
        visited = set()
        for limit in (share, room):
            for i, toks in wanted.items():
                stop = min(room, spent + limit)
                tokset = set(toks)
                for cm in list(self.mem_args):
                    t, args = self.mem_args[cm]
                    if t != tool:
                        continue
                    txt = " ".join(str(v).lower() for v in args.values() if isinstance(v, str))
                    if not (text_words(txt) & tokset):
                        continue
                    o = self.mem_results.get(cm)
                    owners = [self.label_owner(self.label_of(x)) for x, _ in self.listing_records(o)]
                    if owners and all(owners) and i not in owners:
                        continue
                    a = args
                    while spent < stop and not self.finished:
                        plan = self.paging_plan(tool, o)
                        if not plan or plan[3] is None:
                            break
                        a2 = dict(a)
                        a2[plan[2]] = plan[3]
                        f = []
                        a2 = shape(a2, self.schemas.get(tool), "args", f)
                        if f:
                            break
                        cm2 = self.canon(tool, a2)
                        if cm2 in visited:
                            break
                        visited.add(cm2)
                        if cm2 in self.mem_results:
                            o = self.mem_results[cm2]
                            a = a2
                            continue
                        r = self.env_call(tool, a2)
                        if r is None:
                            return spent
                        spent += 1
                        o = r.get("observation")
                        self.observe(tool, a2, o)
                        a = a2
        return spent


class E4(object):
    def canon(self, tool, args):
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        if not isinstance(props, dict):
            props = {}
        kept = {}
        for k in sorted(args or {}):
            v = args[k]
            p = props.get(k) if isinstance(props.get(k), dict) else {}
            if isinstance(v, int) and not isinstance(v, bool):
                if ("minimum" in p and v == p["minimum"]) or ("default" in p and v == p["default"]):
                    continue
                if v == 1 and self.page_arg.get(tool) == k and "minimum" not in p and "default" not in p:
                    continue
            if isinstance(v, str):
                v = " ".join(v.lower().split())
            kept[k] = v
        return tool + dump(kept)

    def note_read(self, tool, args, obs):
        skip = [self.role("offer") or self.role("open")] + [self.role(r) for r in ("compare", "ask", "cart", "cart_view", "buy", "refuse")]
        if tool in [s for s in skip if s]:
            return
        self.new_query(tool, args or {})
        if isinstance(obs, (dict, list)) and not is_failure(obs):
            cm = self.canon(tool, args or {})
            self.mem_results.pop(cm, None)
            self.mem_args.pop(cm, None)
            self.mem_results[cm] = obs
            self.mem_args[cm] = (tool, dict(args or {}))

    def new_query(self, tool, args):
        kind = dump({k: v for k, v in args.items() if not (isinstance(v, int) and not isinstance(v, bool))})
        mem = self.page_queries.setdefault(tool, set())
        new = kind not in mem
        mem.add(kind)
        return new

    def fresh_search(self, tool, args):
        return self.new_query(tool, args)

    def low_value(self, tool, arg):
        p = ((self.schemas.get(tool) or {}).get("properties") or {}).get(arg) or {}
        mn = p.get("minimum") if isinstance(p, dict) else None
        return mn if isinstance(mn, int) and not isinstance(mn, bool) else 1

    def first_page(self, tool, args):
        pa = self.page_arg.get(tool)
        if pa and isinstance(args.get(pa), int) and not isinstance(args.get(pa), bool):
            a = dict(args)
            a[pa] = self.low_value(tool, pa)
            return a
        return args

    def last_page_first(self, tool, args, obs):
        return self.soft("last_page_first", "", self._last_page_first, tool, args, obs)

    def _last_page_first(self, tool, args, obs):
        if tool in self.page_arg or not isinstance(obs, dict):
            return ""
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        cands = []
        for k, v in args.items():
            if isinstance(v, int) and not isinstance(v, bool) and k in props:
                mn = props[k].get("minimum") if isinstance(props[k], dict) else None
                floor = mn if isinstance(mn, int) and not isinstance(mn, bool) else 1
                if v > floor:
                    cands.append(k)
        if len(cands) != 1 or self.steps_left() <= self.reserve() + 4:
            return ""
        sh = self.shape_of(tool, obs)
        if not sh or not sh.get("count") or not sh.get("next") or obs.get(sh["next"]) is not None:
            return ""
        rows, total = obs.get(sh["rows"]), obs.get(sh["count"])
        if not isinstance(rows, list) or not isinstance(total, int) or isinstance(total, bool) or total <= len(rows):
            return ""
        cand = cands[0]
        low = self.low_value(tool, cand)
        a = dict(args)
        a[cand] = low
        f = []
        a = shape(a, self.schemas.get(tool), "args", f)
        if f:
            return ""
        r = self.env_call(tool, a)
        if r is None:
            return ""
        o = r.get("observation")
        self.observe(tool, a, o)
        rows2 = o.get(sh["rows"]) if isinstance(o, dict) else None
        if not isinstance(rows2, list) or len(rows2) < len(rows):
            return ""
        ids1 = set(x for x, _ in self.listing_records(obs))
        if any(x in ids1 for x, _ in self.listing_records(o)):
            return ""
        self.page_arg[tool] = cand
        return ("[Cairn] %s is this search's page number, and the call just made displayed the final page. Page %s of "
                "that same search:\n%s" % (cand, low, self.display(o)))

    def names_code(self, tool, args):
        return self.soft("names_code", False, self._names_code, tool, args)

    def _names_code(self, tool, args):
        said = []
        for v in (args or {}).values():
            if isinstance(v, str):
                said += find_codes(v)
        if not said:
            return False
        known = set()
        for m in self.fit_index().values():
            known |= set(m.get("models") or []) | set(m.get("works") or [])
        return any(c in known for c in said)

    def shorter_search(self, tool, args, obs):
        return self.soft("shorter_search", "", self._shorter_search, tool, args, obs)

    def _shorter_search(self, tool, args, obs):
        tpl = self.value_template()
        if not tpl or tpl[0] != tool:
            return ""
        if self.steps_left() <= self.reserve() + SHORTER_RESERVE or self.finished or not isinstance(obs, dict):
            return ""
        field = tpl[2]
        q = args.get(field)
        if not isinstance(q, str) or len(q.split()) < 3 or self.listing_records(obs):
            return ""
        for v in obs.values():
            if isinstance(v, (list, str)) and v:
                return ""
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                return ""
        words = [w for w in q.split() if not find_codes(w)]
        short = " ".join(words[:2])
        if not short:
            return ""
        a = {k: v for k, v in args.items() if not (isinstance(v, int) and not isinstance(v, bool))}
        a[field] = short
        f = []
        a = shape(a, self.schemas.get(tool), "args", f)
        if f:
            return ""
        r = self.env_call(tool, a)
        if r is None:
            return ""
        o = r.get("observation")
        self.observe(tool, a, o)
        return ("[Cairn] No listing carries every one of those words; the identical search with the short words '%s' "
                "gives:\n%s" % (short, self.display(o)))

    def opening_moves(self):
        return self.soft("opening_moves", "", self._opening_moves)

    def _opening_moves(self):
        tpl = self.value_template()
        if not tpl or self.budget_steps < OPEN_MIN_STEPS or len(self.items) < OPEN_MIN_ITEMS:
            return ""
        self.gap = OPENING_PAUSE
        stop = self.budget_steps - self.reserve() - self.free_msgs() - OPEN_LATER_STEPS * len(self.items) - HUNT_RESERVE // 2
        tool, fixed, field = tpl
        lines = []
        reftexts = set(r for _, r in self.refs)
        for i, it in enumerate(self.items, 1):
            vals = []
            for m in it.musts:
                if ":" not in m or m in reftexts:
                    continue
                v = m.partition(":")[2].strip()
                if v and len(v.split()) <= 3 and not V3_OPEN_SKIP.search(v):
                    vals.append(v)
            ws = it.name.split()
            noun = it.name if len(ws) <= 3 else " ".join(ws[-2:])
            tries = []
            if vals:
                tries.append(" ".join(vals[:2] + [noun]))
            if noun not in tries:
                tries.append(noun)
            for q in tries:
                a = dict(fixed)
                a[field] = q
                f = []
                a = shape(a, self.schemas.get(tool), "args", f)
                if f or self.finished or self.steps >= stop:
                    break
                r = self.env_call(tool, a)
                if r is None:
                    return "\n".join(lines)
                o = r.get("observation")
                self.observe(tool, a, o)
                if self.listing_records(o):
                    self.searched = (self.searched + [dict(a)])[-8:]
                    lines.append("%s: %s" % (q, self.display(o)))
                    plan = self.paging_plan(tool, o)
                    if plan and plan[3] is not None and self.item_rows(i) < OPEN_ROWS_PER_ITEM and not self.finished\
                            and self.steps < stop:
                        a2 = dict(a)
                        a2[plan[2]] = plan[3]
                        f = []
                        a2 = shape(a2, self.schemas.get(tool), "args", f)
                        if not f:
                            r2 = self.env_call(tool, a2)
                            if r2 is not None:
                                self.observe(tool, a2, r2.get("observation"))
                                lines.append("next page: %s" % self.display(r2.get("observation")))
                    if self.item_rows(i) >= OPEN_ROWS_PER_ITEM:
                        break
                else:
                    if is_failure(o):
                        lines.append("%s: the store rejected the call: %s" % (q, self.display(o)))
                    else:
                        lines.append("%s: no rows" % q)
        if self.rows and self.role("ask") and self.free_msgs() > 0:
            found = [i for i in range(1, len(self.items) + 1) if self.item_rows(i) >= 1]
            start = self.steps
            while self.steps - start < self.budget_steps // OPEN_WIDEN_SHARE and self.steps < stop and not self.finished:
                if all(self.item_full(i) >= OPEN_ROWS_PER_ITEM for i in found):
                    break
                before = len(self.full)
                self.widen()
                if len(self.full) <= before:
                    break
        self.gap = STORE_PAUSE
        for lid in list(self.full):
            lines.append("read in full: " + self.page_text(lid))
        ev = self.evidence_ids(list(self.full))
        if ev:
            lines.append(ev)
        return "\n".join(lines)

    def page_text(self, lid):
        rec = self.full.get(lid) or self.rows.get(lid) or {}
        blk = attr_of(rec) or {}
        a = dump(clip(blk, 120))[:PAGE_ATTR_CLIP]
        rest = {k: v for k, v in rec.items() if not isinstance(v, (dict, list))}
        b = dump(clip(rest, 200))[:PAGE_REST_CLIP]
        return "%s: attributes %s | %s" % (lid, a, b)

    def observe(self, tool, args, obs):
        self.absorb(obs)
        lines = self.loose_lines(obs)
        if not lines:
            return ""
        m = self.read_note(lines)
        if m:
            self.queued.append("(seen on %s) %s" % (tool, m))
        return m or ""

    def compare_list_arg(self, tool):
        sch = self.schemas.get(tool) or {}
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        for k, s in props.items():
            if isinstance(s, dict) and "array" in kinds(s):
                return k
        return None

    def widen(self):
        cmp = self.role("compare")
        tgt = self.compare_list_arg(cmp) if cmp else None
        req = (self.schemas.get(cmp) or {}).get("required") or [] if cmp else []
        if not cmp or not tgt or any(r != tgt for r in req) or self.compare_bare():
            return self.widen_singly()
        if self.steps_left() <= 12:
            return
        p = ((self.schemas.get(cmp) or {}).get("properties") or {}).get(tgt) or {}
        mx = p.get("maxItems") if isinstance(p.get("maxItems"), int) else 4
        cap = min(mx, self.compare_room())
        groups = self.item_groups()
        groups = sorted(groups, key=lambda g: sum(1 for x in g[1] if x in self.full))
        lanes = [[x for x in ids if x not in self.full] for lab, ids in groups]
        picks = []
        for r in range(cap):
            for lane in lanes:
                if r < len(lane):
                    picks.append(lane[r])
        picks = picks[:cap]
        if len(picks) < 2:
            self.fill_facts(picks, 12)
            return
        f = []
        a = shape({tgt: picks}, self.schemas.get(cmp), "args", f)
        if f:
            return
        before = len(self.full)
        r = self.env_call(cmp, a)
        recs = []
        if r is not None:
            o = r.get("observation")
            self.observe(cmp, a, o)
            self.record_view_keys(cmp, o)
            recs = self.listing_records(o)
            if recs:
                self.set_facts(cmp, self.facts.get(cmp) is True or any(attr_of(x) is not None for _, x in recs))
                self.compare_ok = max(self.compare_ok, len(recs))
            else:
                self.compare_bad = min(self.compare_bad or 999, len(picks))
        if recs and self.facts.get(cmp) is False:
            return self.widen_singly()
        if len(self.full) <= before and len(picks) > 2 and self.steps_left() > 12:
            f = []
            a2 = shape({tgt: picks[:2]}, self.schemas.get(cmp), "args", f)
            if not f:
                r2 = self.env_call(cmp, a2)
                if r2 is not None:
                    o2 = r2.get("observation")
                    self.observe(cmp, a2, o2)
                    self.compare_ok = max(self.compare_ok, len(self.listing_records(o2)))

    def widen_singly(self):
        ft = self.facts_tool()
        tgt = self.single_target(ft) if ft else None
        if not ft or not tgt:
            return
        wanted = set()
        for it in self.items:
            for r in it.live():
                v = r.partition(":")[2] if ":" in r else r
                wanted |= set(ALNUM3.findall(v.lower()))
        lane_size = WIDEN_PER_ITEM if len(self.items) <= 2 else 4
        lanes, heads = [], []
        for lab, ids in self.item_groups():
            opened = sum(1 for x in ids if x in self.full)
            pool = [x for x in ids if x not in self.full and self.buyable(x)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda x: (self.price_of(x) is None, self.price_of(x) or 0))
            pool.sort(key=lambda x: -len(wanted & set(ALNUM3.findall(dump(self.rows.get(x)).lower()))))
            n = lane_size - opened
            lanes.append(pool[:n] if n > 0 else [])
            heads.append(max(0, 4 - opened))
        first, more = [], []
        for r in range(4):
            for li, lane in enumerate(lanes):
                if r < heads[li] and r < len(lane):
                    first.append(lane[r])
        for r in range(lane_size):
            for li, lane in enumerate(lanes):
                if r >= heads[li] and r < len(lane):
                    more.append(lane[r])
        cut = max(5, 2 * len(lanes) + 2)
        got = self.fill_facts(first[:min(cut, 8)], 22)
        second = (first[cut:] + more)[:max(0, 10 - len(got))]
        if second:
            self.fill_facts(second, 30)

    def interview_ready(self):
        if self.asked or not self.role("ask") or self.free_msgs() <= 0:
            return False
        if self.steps_left() <= 8 or self.clock() > ASK_DEADLINE:
            return False
        if self.turns < ASK_WAIT_TURNS and any(self.item_full(i) == 0 for i in range(1, len(self.items) + 1)):
            return False
        if len(self.full) >= 3 * len(self.items) + 1:
            return True
        return self.steps >= max(8, self.budget_steps // 4)


VALUE_EXCL = re.compile(r"\b(?:not|no|other\s+than|unlike|alongside|listing|page|shown|at\s+most|at\s+least|under|over|below|above|less|more|between|within|up\s+to|or)\b", re.I)
HAS_LETTER = re.compile(r"[A-Za-z]")
MEASURE_VAL = re.compile(r"\s*[\d.,/x \-]+[a-z]{0,6}\s*")
LABEL_CUT = re.compile(r"\s(?:and|or)\s|/|,|;|\(")
PLAIN_S = re.compile(r"(?<=[a-z]{3})s$")
AND_WORD = re.compile(r"\band\b", re.I)
AND_CUT = re.compile(r"\s*(?:,|\band\b)\s*", re.I)
PLURAL_END = re.compile(r"[a-z]{2}[a-rtv-z]s$", re.I)
Q_FILLER = re.compile(r"\b(?:and|for|like|me|or|please|tell|the|what|which|would|you)\b", re.I)
ANY_DIGIT = re.compile(r"\d")


def plain_form(text):
    out = []
    for w in word_form(text).split():
        w = PLAIN_S.sub("", w)
        out.append(OUR_END.sub("or", w))
    return " ".join(out)


def catch_all(key):
    k = word_form(key)
    return bool(V16_CATCHALL.search(k)) and not V16_KINDWORD.search(k)


def quote(text):
    return repr(str(text))


class E5(object):
    def msg_args(self, text):
        tool = self.role("ask")
        sch = self.schemas.get(tool) if tool else None
        if not isinstance(sch, dict):
            sch = {}
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        textf = [k for k, s in props.items() if isinstance(s, dict) and "string" in kinds(s) and "enum" not in s and "const" not in s]
        pool = [r for r in req if r in textf] or textf
        args = {}
        if pool:
            best, bkey = None, None
            for k in pool:
                uf = under_form(k)
                ml = props[k].get("maxLength")
                ml = ml if isinstance(ml, int) and not isinstance(ml, bool) else 1000000
                key = (not V2_NONMSG.search(uf), bool(V1_MSG.search(uf)), ml)
                if bkey is None or key > bkey:
                    best, bkey = k, key
            args[best] = text
        else:
            for k, s in props.items():
                if not isinstance(s, dict):
                    continue
                if "array" in kinds(s):
                    args[k] = [text]
                    break
                sp = s.get("properties")
                if isinstance(sp, dict):
                    strs = [kk for kk, ss in sp.items() if isinstance(ss, dict) and "string" in kinds(ss)]
                    if strs:
                        args[k] = {strs[0]: text}
                        break
        if not args:
            return None
        for r in req:
            if r not in args:
                ok, v = filler(props.get(r))
                if not ok:
                    return None
                args[r] = v
        f = []
        out = shape(args, sch, "args", f)
        if f:
            return None
        return out

    def interview(self):
        self.asked = True
        if self.free_msgs() <= 0:
            return ""
        self.widen()
        count = self.msg_limit - self.msgs_used
        if self.listing and self.list_now() > 1 and (len(self.items) > 1 or self.msg_limit > SCARCE_MESSAGES):
            count = max(count, 8)
        self.plan_lanes(count)
        self.reset_lanes()
        self.follow_ups = []
        sent_all, told = [], []
        while self.free_msgs() > 0 and not self.finished and self.steps_left() > 6 and self.clock() < QUESTION_DEADLINE:
            sent = None
            g = self.surplus_group()
            if g:
                sent = self.group_ask(g, 6, told)
            if not sent:
                sent = self.ask_next(6, told)
            if not sent:
                break
            sent_all += sent
        found = [n for n in range(1, len(self.items) + 1) if self.lane_found.get(n, 0) > 0]
        early = self.early_trail()
        vs = self.value_search(found) if found else ""
        if told:
            self.version += 1
        out = ["[Cairn's interviewer questioned the shopper as shown, one step per question]"]
        for n, q, a in sent_all:
            out.append("Q: " + q)
            out.append("A: " + a)
        out.append("Firm requirements the answers added: %s. They must be applied to every pick, with a fresh search on "
                   "the new values wherever one is needed." % ("; ".join(told) if told else "none"))
        rs = self.ref_sentence(told, True)
        if rs:
            out.append(rs)
        if early:
            out.append(early)
        if vs:
            out.append(vs)
        if self.queued:
            out.append("[notes seen in the meantime] " + " | ".join(self.queued))
            self.queued = []
        return "\n".join(out)

    def ref_sentence(self, told, full_clause):
        names = []
        for n, r in self.refs:
            if any(r in t for t in told) and r not in names:
                names.append(r)
        if not names:
            return ""
        s = ("These needs get their value from a different listing: %s. Locate that listing, read its whole page, and "
             "then open every further listing to which that page directs onward (only those, not every listing it "
             "mentions); the final page along this chain supplies the value" % " | ".join(names))
        if full_clause:
            s += "; an order is valid only once every one of them has been opened"
        return s + "."

    def reset_lanes(self):
        for n in range(1, len(self.items) + 1):
            self.lane_found[n] = 0
            self.lane_spent[n] = 0
            self.lane_used[n] = set()

    def late_question(self):
        if self.late_asked or not self.role("ask") or self.msgs_used >= self.msg_limit:
            return ""
        if any(d.stage == "order" and d.messages >= 1 for d in self.duties):
            return ""
        if self.steps_left() < 8 or self.clock() > LATE_DEADLINE or not self.judges():
            return ""
        self.late_asked = True
        if not self.lanes:
            self.plan_lanes(self.list_cap() if self.listing else 1)
            self.reset_lanes()
        b0 = self.budget
        told = []
        sent = self.ask_next(6, told)
        if not sent:
            self.trace.append({"kind": "late_question", "sent": False})
            return ""
        if self.budget != b0:
            told.append("budget now %s" % self.budget)
        if not told:
            return ""
        self.version += 1
        out = "[Cairn put a further question to the shopper ahead of the checkout, at a cost of one step]"
        for n, q, a in sent:
            out += "\nQ: " + q + "\nA: " + a
        out += ("\nNew firm requirements: " + "; ".join(told) + "\nThe checkout did not happen. Test every cart line "
                "against these requirements (re-searching on those new values where necessary), repair the cart, and "
                "after that check out.")
        rs = self.ref_sentence(told, False)
        if rs:
            out += "\n" + rs
        vs = self.value_search([sent[0][0]])
        if vs:
            out += "\n" + vs
        return out

    def shelf_like(self, v):
        wv = word_form(v)
        refs = [it.shelf for it in self.items] + [it.name for it in self.items]
        ck = self.get_cat_key()
        if ck:
            for row in self.rows.values():
                lab = row.get(ck)
                if isinstance(lab, str) and lab not in refs:
                    refs.append(lab)
        vs = set(w for w in stems(v) if len(w) >= 4)
        for r in refs:
            if not r:
                continue
            if word_form(r) == wv:
                return True
            if vs & set(w for w in stems(r) if len(w) >= 4):
                return True
        return False

    def search_template(self):
        tool = self.role("search")
        if not tool:
            return None
        sch = self.schemas.get(tool) or {}
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        textf = [k for k, s in props.items() if isinstance(s, dict) and "string" in kinds(s) and not s.get("enum")]
        named = [k for k in textf if V14_QUERY.search(under_form(k))]
        ck = self.get_cat_key()
        labels = set()
        if ck:
            for row in self.rows.values():
                if isinstance(row.get(ck), str):
                    labels.add(row[ck].strip().lower())
        for sa in reversed(self.searched):
            cands = [k for k in (named or textf) if isinstance(sa.get(k), str) and HAS_LETTER.search(sa[k])]
            if not named:
                keep = []
                for k in cands:
                    vals = [s.get(k) for s in self.searched if k in s]
                    if len(set(dump(v) for v in vals)) < 2:
                        continue
                    if not any(isinstance(v, str) and len(v.split()) >= 2 for v in vals):
                        continue
                    if any(isinstance(v, str) and v.strip().lower() in labels for v in vals):
                        continue
                    keep.append(k)
                cands = keep
            if not cands:
                continue
            field = cands[0]
            kept = {}
            for k, v in sa.items():
                if k == field:
                    continue
                p = props.get(k)
                if isinstance(p, dict) and ("enum" in p or k in req):
                    kept[k] = v
            for k in list(kept):
                if isinstance(kept[k], str) and k not in req and self.shelf_like(kept[k]):
                    del kept[k]
            for k in list(kept):
                p = props.get(k)
                if not (isinstance(p, dict) and "enum" in p):
                    continue
                vals = [dump(s[k]) for s in self.searched if k in s and isinstance(s.get(field), str) and HAS_LETTER.search(s[field])]
                if vals:
                    cnt, order = {}, []
                    for v in vals:
                        if v not in cnt:
                            cnt[v] = 0
                            order.append(v)
                        cnt[v] += 1
                    kept[k] = json.loads(max(order, key=lambda v: cnt[v]))
            return (tool, kept, field)
        return None

    def value_words(self, num):
        out = []
        reftexts = set(r for _, r in self.refs)
        it = self.items[num - 1]
        for r in it.live():
            if r in reftexts or ":" not in r:
                continue
            v = r.partition(":")[2].strip()
            told = r in it.told
            m = V13_FLOOR.search(v)
            if m:
                if told:
                    out.append((m.group(1).strip(), True))
                continue
            if not v or VALUE_EXCL.search(v):
                continue
            out.append((v, told))
        return out

    def value_template(self):
        t = self.search_template()
        if t:
            return t
        stool = self.role("search")
        if not stool:
            return None
        sw = self.brief.get("search_words")
        if isinstance(sw, dict):
            tool = sw.get("tool") if sw.get("tool") in self.schemas else stool
            field = sw.get("field")
            fixed = sw.get("args") if isinstance(sw.get("args"), dict) else {}
            props = (self.schemas.get(tool) or {}).get("properties") or {}
            p = props.get(field) if isinstance(props, dict) and isinstance(field, str) else None
            if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum"):
                a = dict(fixed)
                a[field] = "x"
                f = []
                a = shape(a, self.schemas.get(tool), "args", f)
                if not f:
                    a.pop(field, None)
                    return (tool, a, field)
        sch = self.schemas.get(stool) or {}
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        wf = [k for k, s in props.items() if isinstance(s, dict) and "string" in kinds(s) and not s.get("enum")
              and V14_QUERY.search(under_form(k))]
        last = self.searched[-1] if self.searched else {}
        optsets = [k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2]
        modes = {}
        for k in optsets:
            opts = props[k]["enum"]
            m = [o for o in opts if isinstance(o, str) and V14_QUERY.search(o)]
            if len(m) == 1:
                modes[k] = m[0]
            elif k in req and len(opts) == 2 and k in last:
                diff = [o for o in opts if o != last[k]]
                if len(diff) == 1:
                    modes[k] = diff[0]
        if len(wf) != 1 or len(modes) > 1 or (optsets and not modes):
            return None
        base = {k: last[k] for k in req if k in last and k != wf[0]}
        base.update(modes)
        return (stool, base, wf[0])

    def value_search(self, nums):
        tpl = self.value_template()
        if not tpl or self.steps_left() < self.reserve() + VALUE_ROOM:
            return ""
        allow = min(VALUE_RESERVE, self.steps_left() - self.reserve() - VALUE_RESERVE)
        tool, base, field = tpl
        lines = []
        ck = self.get_cat_key()
        for i in nums:
            vw = self.value_words(i)
            if not [v for v, f in vw if f] or allow <= 2:
                continue
            lead = [(v, f) for v, f in vw if f and (len(v.split()) == 1 or V12_WHOLE.fullmatch(v.strip()))]
            order = lead + [p for p in vw if p not in lead]
            drop = 0
            for k, (v, f) in enumerate(order):
                if not f:
                    drop = k
            phrases = [v for v, f in order]
            if len(phrases) >= 2:
                tries = [phrases, phrases[:drop] + phrases[drop + 1:]]
            elif order[0] in lead:
                tries = [[phrases[0], self.items[i - 1].name], [phrases[0]]]
            else:
                tries = [[phrases[0]]]
            for words in tries[:2]:
                a = dict(base)
                a[field] = " ".join(" ".join(words).split())
                f = []
                a = shape(a, self.schemas.get(tool), "args", f)
                if f or self.finished:
                    break
                r = self.env_call(tool, a)
                allow -= 1
                if r is None:
                    break
                o = r.get("observation")
                self.observe(tool, a, o)
                recs = self.listing_records(o)
                kept = []
                for lid, _ in recs:
                    if not self.buyable(lid):
                        continue
                    if ck:
                        lab = self.label_of(lid)
                        if lab is None or not self.owns_label(str(lab), i):
                            continue
                    kept.append(lid)
                if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool):
                    prices = [self.price_of(x) for x, _ in recs]
                    ignore = bool(prices) and all(p is not None for p in prices) and all(p > 10 * self.budget for p in prices)
                    if not ignore:
                        kept = [x for x in kept if self.price_of(x) is None or self.price_of(x) <= self.budget]
                home = self.home_label(i)
                if ck and home:
                    kept = [x for x in kept if word_form(self.label_of(x) or "") == home] +\
                           [x for x in kept if word_form(self.label_of(x) or "") != home]
                lines.append("%s: %s" % (dump(a.get(field)), ", ".join(kept[:6]) if kept else "no row of this item's kind came back"))
                if kept:
                    n = min(3, allow)
                    picks = [x for x in kept if x not in self.full][:max(0, n)]
                    texts = self.fill_facts(picks, self.reserve() + 8) if picks else []
                    allow -= len(texts)
                    if texts:
                        lines.append(("opened: " + " | ".join(texts))[:2400])
                    break
        if not lines:
            return ""
        self.trace.append({"kind": "value_search", "lines": len(lines)})
        return ("[Cairn searched all the requirement values of each such item at once and read the leading rows of the "
                "right kind]\n" + "\n".join(lines))

    def home_label(self, num):
        ck = self.get_cat_key()
        if not ck:
            return ""
        it = self.items[num - 1]
        target = word_form(it.shelf) or word_form(it.name)
        for pool in (self.full, self.rows):
            cnt, order = {}, []
            for lid, rec in pool.items():
                lab = rec.get(ck) if isinstance(rec, dict) else None
                if lab is None or not self.owns_label(str(lab), num):
                    continue
                wl = word_form(lab)
                if wl not in cnt:
                    cnt[wl] = 0
                    order.append(wl)
                cnt[wl] += 1
            if target in cnt:
                return target
            if order:
                return max(order, key=lambda l: cnt[l])
        return ""

    def surplus_group(self):
        free = self.free_msgs()
        n = len(self.items)
        if not self.listing or self.msg_limit > SCARCE_MESSAGES or n <= self.msg_limit or free < 1:
            return []
        order = sorted(range(1, n + 1), key=lambda x: self.plan_order.index(x) if x in self.plan_order else len(self.plan_order) + x)
        open_ = [x for x in order if self.lane_found.get(x, 0) == 0 and self.remaining(x)]
        if len(open_) <= max(1, free):
            return []
        return open_[:-(-len(open_) // free)]

    def group_items(self):
        ordered = [n for n in self.plan_order if self.lanes.get(n)]
        ordered += [n for n in range(1, len(self.items) + 1) if n not in ordered and self.lanes.get(n)]
        fm = self.free_msgs()
        if fm <= 0 or len(ordered) <= fm:
            return []
        groups, rem, g = [], list(ordered), fm
        while rem and g:
            size = -(-len(rem) // g)
            groups.append(rem[:size])
            rem = rem[size:]
            g -= 1
        return groups

    def list_cap(self):
        return 10 if len(self.items) > 1 else 8

    def list_now(self):
        if self.msg_limit <= SCARCE_MESSAGES and len(self.items) == 1:
            return 5
        return 8

    def told_about(self, num, label):
        ls = spelled(label)
        for t in self.items[num - 1].told:
            if names_match(ls, spelled(t.partition(":")[0])):
                return True
        return False

    def remaining(self, num):
        used = self.lane_used.setdefault(num, set())
        return [l for l in self.lanes.get(num, []) if word_form(l) not in used and not self.told_about(num, l)]

    def group_text(self, parts):
        clauses = []
        for n, labs in parts:
            if len(labs) >= 2:
                piece = ", ".join(labs[:-1]) + " or " + labs[-1]
            else:
                piece = labs[0]
            clauses.append("what %s for the %s" % (piece, self.asked_name(n)))
        t = ", and ".join(clauses)
        t = "W" + t[1:]
        i = t.find("for the")
        if i >= 0:
            t = t[:i] + "would you like " + t[i:]
        return t + "?"

    def group_ask(self, group, floor, told):
        cap, now = self.list_cap(), self.list_now()
        gs = len(group)
        share = max(1, now // gs)
        parts = [(n, self.remaining(n)[:share]) for n in group if self.remaining(n)]
        if not parts:
            return None
        if len(parts) == 1:
            n, labs = parts[0]
            fl, text = self.fit_ask(n, labs, False)
            if not fl:
                return None
            sent = self.send([(n, fl[0], text)], floor)
            if sent:
                self.finish_ask(n, fl, False, sent, told)
            return sent
        found = None
        while parts and not found:
            for c in range(share, 0, -1):
                cut = [(n, l[:c]) for n, l in parts]
                text = self.group_text(cut)
                a = self.msg_args(text)
                if a is not None and says(a, text):
                    found = (cut, text)
                    break
            if not found:
                parts = parts[:-1]
        if not found:
            return None
        cut, text = found
        sent = self.send([(cut[0][0], cut[0][1][0], text)], floor)
        if not sent:
            return None
        for n, _ in cut:
            self.lane_spent[n] = self.lane_spent.get(n, 0) + 1
        self.questions_sent += 1
        before = {i: list(it.told) for i, it in enumerate(self.items, 1)}
        reading = self.read_answers([(0, sent[0][1], sent[0][2])], [LIGHT, HEAVY])
        reading.pop("free", None)
        self.take_reading(reading, told)
        answered = []
        for n, labs in cut:
            gained = [t for t in self.items[n - 1].told if t not in before[n]]
            if gained:
                answered.append(n)
                self.lane_found[n] = self.lane_found.get(n, 0) + 1
                self.settle(n, labs, False, gained, False)
        crowded = any(count_of(x) == 1 for x in items_of(reading.get("crowded")))
        if crowded:
            self.listing = False
        if crowded and not answered:
            self.listing = False
            self.trace.append({"kind": "lists_off", "by": "group"})
            return sent
        if not answered:
            self.lane_used.setdefault(cut[0][0], set()).add(word_form(cut[0][1][0]))
        self.trace.append({"kind": "group_ask", "items": [n for n, _ in cut], "answered": answered})
        return sent

    def finish_ask(self, num, labels, shown, sent, told):
        self.lane_spent[num] = self.lane_spent.get(num, 0) + 1
        self.questions_sent += 1
        before = {i: list(it.told) for i, it in enumerate(self.items, 1)}
        reading = self.read_answers(sent, [LIGHT, HEAVY])
        if len(labels) > 1:
            reading.pop("free", None)
        self.take_reading(reading, told)
        new = []
        for i, it in enumerate(self.items, 1):
            gained = [t for t in it.told if t not in before[i]]
            if gained:
                self.lane_found[i] = self.lane_found.get(i, 0) + 1
                new += gained
        crowded = any(count_of(x) == 1 for x in items_of(reading.get("crowded")))
        self.settle(num, labels, shown, new, crowded)

    def next_ask(self):
        scarce = self.msg_limit <= SCARCE_MESSAGES and not self.listing
        n_items = len(self.items)

        def key(n):
            pr = self.plan_order.index(n) if n in self.plan_order else 1000 + n
            return (self.lane_found.get(n, 0) > 0, 0 if scarce else self.lane_spent.get(n, 0), pr, n)
        for n in sorted(range(1, n_items + 1), key=key):
            rem = self.remaining(n)
            if rem:
                return n, (rem[:self.list_now()] if self.listing else rem[:1])
        return None

    def lookalikes(self, num, label):
        keyroots, order = {}, []
        for lid in self.item_ids(num):
            if lid in self.full:
                blk = attr_of(self.full[lid]) or {}
                for k, v in blk.items():
                    wf = word_form(k)
                    if wf not in keyroots:
                        keyroots[wf] = set()
                        order.append(wf)
                    keyroots[wf] |= roots(str(v))
        cnt = {}
        for wf in order:
            for w in set(wf.split()):
                cnt[w] = cnt.get(w, 0) + 1
        lw = word_form(label)
        plainw = [w for w in lw.split() if cnt.get(w, 0) <= 2]
        own = keyroots.get(lw, set())
        scored = []
        for idx, wf in enumerate(order):
            if not wf or wf == lw:
                continue
            shared = [w for w in plainw if w in wf.split()]
            common = own & keyroots[wf]
            if shared or common:
                scored.append((3 * len(shared) + min(3, len(common)), idx, wf))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [s[2] for s in scored[:LOOKALIKE_KEYS]]

    def examples(self, num, label):
        wf = word_form(label)
        musts = " ".join(self.items[num - 1].musts).lower()
        for lab, n, kept in self.key_table():
            for slot in kept:
                if slot.wf == wf:
                    return [v for v in slot.distinct[:4] if v and len(v) <= 30 and v not in musts][:2]
        return []

    def question_text(self, num, labels, shown):
        asked = self.asked_name(num)
        if len(labels) >= 2:
            piece = ", ".join(labels[:-1]) + " or " + labels[-1]
            return "What %s would you like for the %s?" % (piece, asked)
        label = labels[0]
        la = (self.apart.get(label) or {}).get(num)
        if la:
            s = "Apart from the %s" % la[0]
            if len(la) > 1:
                s += " and the %s" % la[1]
            return s + ", which %s would you like for the %s?" % (label, asked)
        if shown:
            ex = self.examples(num, label)
            return "What %s would you like for the %s, for example %s or %s?" % (label, asked, ex[0], ex[1])
        return "What %s would you like for the %s?" % (label, asked)

    def fit_ask(self, num, labels, shown):
        for c in range(len(labels), 0, -1):
            text = self.question_text(num, labels[:c], shown)
            a = self.msg_args(text)
            if a is not None and says(a, text):
                return labels[:c], text
        return [], None

    def ask_next(self, floor, told):
        pick = None
        for _ in range(LIST_ONE):
            self.follow_ups = [f for f in self.follow_ups if not self.told_about(f[0], f[1][0])]
            if self.follow_ups:
                n, labels, shown = self.follow_ups.pop(0)
            else:
                na = self.next_ask()
                if not na:
                    return None
                n, labels = na
                shown = False
            if not labels:
                return None
            fl, text = self.fit_ask(n, labels, shown)
            if not fl and shown:
                fl, text = self.fit_ask(n, labels, False)
            if fl:
                pick = (n, fl, text, shown)
                break
            self.lane_used.setdefault(n, set()).add(word_form(labels[0]))
            self.trace.append({"kind": "question_too_long", "item": n, "label": labels[0]})
        if not pick:
            return None
        n, fl, text, shown = pick
        sent = self.send([(n, fl[0], text)], floor)
        if not sent:
            return None
        self.finish_ask(n, fl, shown, sent, told)
        return sent


class E6(object):
    def settle(self, num, labels, shown, new, crowded):
        used = self.lane_used.setdefault(num, set())
        if len(labels) == 1:
            used.add(word_form(labels[0]))
            if crowded and self.listing:
                self.listing = False
                self.trace.append({"kind": "lists_off", "by": "single"})
            if new or not crowded:
                return
            if num not in self.one_product and self.name_parts(num):
                self.one_product.add(num)
                self.follow_ups.insert(0, (num, [labels[0]], shown))
                self.trace.append({"kind": "one_product", "item": num})
                return
            if not any(self.apart.values()):
                keys = self.lookalikes(num, labels[0])
                if keys:
                    self.apart.setdefault(labels[0], {})[num] = keys
                    self.follow_ups.insert(0, (num, [labels[0]], False))
                    self.trace.append({"kind": "set_apart", "label": labels[0], "keys": keys})
            return
        if crowded:
            self.listing = False
        if crowded and not new:
            self.listing = False
            self.follow_ups.insert(0, (num, [labels[0]], False))
            self.trace.append({"kind": "lists_off", "by": "list"})
            return
        hit = self.answered_pos(labels, new) if new else None
        if hit is not None:
            ahead, usedl = labels[hit + 1:], labels[:hit + 1]
        else:
            ahead, usedl = list(reversed(labels[1:])), labels[:1]
        for l in usedl:
            used.add(word_form(l))
        rest = [l for l in self.lanes.get(num, []) if l not in labels]
        if hit is not None:
            self.lane_misses[num] = 0
        else:
            self.lane_misses[num] = self.lane_misses.get(num, 0) + 1
        m = self.lane_misses[num]
        if m and m % 2 == 0:
            mid = (len(ahead) - 1) / 2.0
            ro = sorted(range(len(ahead)), key=lambda i: (abs(i - mid), i))
            self.lanes[num] = rest + [ahead[i] for i in ro]
        else:
            self.lanes[num] = ahead + rest
        self.trace.append({"kind": "list_settled", "item": num, "hit": hit})

    def answered_pos(self, labels, needs):
        pooled = {}
        for lab, n, kept in self.key_table():
            for slot in kept:
                pooled.setdefault(slot.wf, set()).update(roots(" ".join(slot.distinct[:4])))
        best = None
        for need in needs:
            if ":" in need:
                prop, _, val = need.partition(":")
            else:
                prop, val = need, ""
            ps = spelled(prop)
            scores = [len(spelled(l) & ps) for l in labels]
            if not labels or max(scores) == 0:
                vr = roots(val)
                scores = [len(pooled.get(word_form(l), set()) & vr) for l in labels]
            if labels and max(scores) > 0:
                pos = scores.index(max(scores))
                best = pos if best is None else min(best, pos)
        return best

    def ranked_keys(self, num, table):
        settled = [stems(m.partition(":")[0]) for m in self.items[num - 1].musts]
        rows = [(lab, slot) for lab, n, kept in table for slot in kept if self.label_owner(lab) == num]
        if len(self.items) == 1 and not rows:
            rows = [(lab, slot) for lab, n, kept in table for slot in kept]
        meas = {}
        for lab, n, kept in table:
            for slot in kept:
                vals = slot.distinct[:4]
                meas[slot.wf] = sum(1 for v in vals if MEASURE_VAL.fullmatch(v)) * 2 > len(vals)
        rows.sort(key=lambda r: (-r[1].count, meas.get(r[1].wf, False), -len(r[1].distinct[:4])))
        keys = [slot.wf for lab, slot in rows if not any(names_match(s, stems(slot.wf)) for s in settled)]
        prop, demoted = [], []
        for k in keys:
            if not V15_NONPROP.search(k) or V15_MODELS_OK.search(k):
                prop.append(k)
            else:
                demoted.append(k)
        return [k for k in prop if not catch_all(k)] + [k for k in prop if catch_all(k)] + demoted

    def adopt_strays(self):
        ck = self.get_cat_key()
        if len(self.items) < 2 or not ck:
            return
        self.part_shelves()
        labels = []
        for rec in self.full.values():
            v = rec.get(ck)
            if v is not None and str(v) and str(v) not in labels:
                labels.append(str(v))
        strays = [l for l in labels if not self.label_owner(l)]
        orphans = [i for i in range(1, len(self.items) + 1) if not any(self.label_owner(l, False) == i for l in labels)]
        if strays and len(orphans) == 1:
            for l in strays:
                self.adopted[l] = orphans[0]

    def label_owner(self, label, adopt=True):
        if label is None:
            return 0
        lab = str(label)
        if not lab:
            return 0
        if adopt and lab in self.adopted:
            return self.adopted[lab]
        key = (lab, len(self.items))
        if adopt and key in self.owners_seen:
            return self.owners_seen[key]
        wl = word_form(lab)
        lr = roots(lab)
        best, bs = 0, 0
        for i, it in enumerate(self.items, 1):
            if wl == word_form(it.shelf):
                s = 3
            elif wl == word_form(it.name):
                s = 2
            else:
                ir = roots(it.shelf) | roots(it.name)
                s = len(lr & ir) / float(max(1, len(lr | ir)))
            if s > bs:
                best, bs = i, s
        if adopt:
            self.owners_seen[key] = best
        return best

    def owns_label(self, label, num):
        if label is None or not str(label) or not (1 <= num <= len(self.items)):
            return False
        return self.label_owner(label) == num or word_form(label) == word_form(self.items[num - 1].shelf)

    def item_full(self, num):
        ck = self.get_cat_key()
        if not ck:
            return 0
        n = 0
        for lid, rec in self.full.items():
            lab = rec.get(ck)
            if lab is not None and self.owns_label(str(lab), num) and attr_of(rec) is not None:
                n += 1
        return n

    def plan_lanes(self, count):
        self.adopt_strays()
        table = self.key_table()
        lines = []
        for i, it in enumerate(self.items, 1):
            lines.append("item %d: %s - already settled by the request: %s" % (i, it.name, "; ".join(it.musts) if it.musts else "nothing"))
        prompt = PLAN_PROMPT.replace("COUNT", str(max(1, count))) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(lines)
        af = [str(x) for x in items_of(self.brief.get("ask_first")) if str(x).strip()]
        if af:
            prompt += "\n\nASK FIRST (the request wants the shopper asked about these):\n" + "\n".join(af)
        listing, num, prev = [], 0, None
        for lab, n, kept in table:
            for slot in kept:
                num += 1
                if lab != prev:
                    listing.append("category %s:" % quote(lab))
                    prev = lab
                line = "  %d. %s (on %d): %s" % (num, slot.wf, slot.count, " | ".join(v[:40] for v in slot.distinct[:4]))
                line += self.not_stated(lab, slot.wf, slot.distinct[:4])
                if catch_all(slot.wf):
                    line += " [lumps several features together; ask last]"
                listing.append(line)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) if listing else "(no listing has been read in full yet)")
        ans = self.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}
        N = len(self.items)
        picks = {n: [] for n in range(1, N + 1)}
        hinted = {n: set() for n in range(1, N + 1)}
        for p in items_of(ans.get("ask")):
            if not isinstance(p, dict):
                continue
            n = count_of(p.get("item")) or 1
            n = min(max(n, 1), N)
            lab = LABEL_CUT.split(str(p.get("label") or ""))[0].strip().rstrip("?.").strip()
            if not lab or len(lab.split()) > 5:
                continue
            if word_form(lab) not in [word_form(x) for x in picks[n]]:
                picks[n].append(lab)
            if str(p.get("hint") or "").strip():
                hinted[n].add(word_form(lab))
                if n not in self.plan_order:
                    self.plan_order.append(n)
        for n in range(1, N + 1):
            keys = self.ranked_keys(n, table)
            rc = max(1, self.item_full(n))
            own = {}
            for lab, cnt, kept in table:
                if self.label_owner(lab) == n:
                    for slot in kept:
                        own[slot.wf] = max(own.get(slot.wf, 0), slot.count / float(rc))

            def store(l):
                pf = plain_form(l)
                for k in own:
                    if plain_form(k) == pf:
                        return k
                return l

            def share(l):
                wf = word_form(l)
                if wf in own:
                    return own[wf]
                b, ls = 0, stems(l)
                for k, s in own.items():
                    ks = stems(k)
                    if names_match(ls, ks) or names_match(ks, ls):
                        b = max(b, s)
                return b
            hint_set = set(word_form(store(h)) for h in hinted[n])
            lane = []
            for l in picks[n]:
                s = store(l)
                if word_form(s) not in [word_form(x) for x in lane]:
                    lane.append(s)
            gk = [k for k in keys if not catch_all(k) and own.get(k, 0) >= GUARD_SHARE][:GUARD_KEYS]
            for k in gk:
                if word_form(k) in [word_form(x) for x in lane[:3]]:
                    continue
                ks = own.get(k, 0)
                weak = None
                for l in lane[:3]:
                    if word_form(l) not in hint_set and share(l) + GUARD_GAP <= ks:
                        weak = l
                        break
                if weak is None and len(lane) >= 3:
                    continue
                new = [x for x in lane if word_form(x) != word_form(k)]
                if weak is not None:
                    wi = new.index(weak)
                    if N > 1:
                        new[wi] = k
                        if len(new) >= 3:
                            new.insert(3, weak)
                        else:
                            new.append(weak)
                    else:
                        new.insert(wi, k)
                else:
                    new.append(k)
                top_new = [word_form(y) for y in new[:3]]
                if any(word_form(x) in hint_set and word_form(x) not in top_new for x in lane[:3]):
                    continue
                lane = new
            for k in keys:
                if word_form(k) not in [word_form(x) for x in lane]:
                    lane.append(k)
            lane = [l for l in lane if not catch_all(l)] + [l for l in lane if catch_all(l)]
            self.lanes[n] = lane
        self.planned = True
        self.trace.append({"kind": "plan", "lanes": {str(k): v[:12] for k, v in self.lanes.items()}})

    def name_parts(self, num):
        name = self.items[num - 1].name
        if not AND_WORD.search(name):
            return []
        pieces = [p.strip() for p in AND_CUT.split(name) if p and p.strip()]
        if len(pieces) >= 2 and all(PLURAL_END.search(p) for p in pieces):
            return pieces
        return []

    def asked_name(self, num):
        name = self.items[num - 1].name
        parts = self.name_parts(num)
        if not parts:
            return name
        if num not in self.one_product:
            return AND_WORD.sub("or", name)
        ck = self.get_cat_key()
        rsets = []
        for lid in self.seen:
            row = self.rows.get(lid) or {}
            if ck and not self.owns_label(row.get(ck), num):
                continue
            rsets.append(roots(" ".join(v for k, v in row.items() if k != ck and isinstance(v, str))))
        best, bs = parts[0], -1
        for p in parts:
            pr = roots(p.split()[-1]) if p.split() else set()
            sc = sum(1 for rs in rsets if pr and pr <= rs)
            if sc > bs:
                best, bs = p, sc
        return best

    def send(self, qs, floor=6):
        out = []
        for num, label, text in qs:
            if self.finished or self.steps_left() <= floor or self.msgs_used >= self.msg_limit:
                break
            a = self.msg_args(text)
            if a is None:
                break
            r = self.env_call(self.role("ask"), a)
            if r is None:
                break
            self.msgs_used += 1
            obs = r.get("observation")
            ans = self.spoken(obs)
            out.append((num, text, ans))
            self.done_log.append("step %d: shopper message sent %s -> reply %s (cart then: %s)"
                                 % (self.steps, quote(text), quote(ans[:300]), self.cart_mark()))
            self.done_steps.append(self.steps)
            self.trace.append({"kind": "message", "item": num, "label": label, "q": text, "a": ans[:300]})
            if isinstance(obs, dict) and "error" in obs:
                break
        return out

    def item_num(self, v):
        c = count_of(v)
        if not c:
            c = 1
        return min(max(c, 1), len(self.items))

    def take_reading(self, reading, lines):
        many = len(self.items) > 1
        for e in items_of(reading.get("needs")):
            if not isinstance(e, dict) or not str(e.get("need") or "").strip():
                continue
            n = self.item_num(e.get("item"))
            t = str(e.get("need")).strip()
            it = self.items[n - 1]
            if t not in it.told:
                it.told.append(t)
                lines.append(("item %d: %s" % (n, t)) if many else t)
            if e.get("points") is True and (n, t) not in self.refs:
                self.refs.append((n, t))
        for e in items_of(reading.get("drop")):
            if not isinstance(e, dict) or not str(e.get("property") or "").strip():
                continue
            n = self.item_num(e.get("item"))
            it = self.items[n - 1]
            prop = word_form(str(e.get("property")).partition(":")[0])
            for r in it.musts + it.told:
                if r not in it.dropped and word_form(r.partition(":")[0]) == prop:
                    it.dropped.append(r)
                    lines.append("dropped " + r)
        for e in items_of(reading.get("free")):
            if isinstance(e, dict) and e.get("property"):
                self.items[self.item_num(e.get("item")) - 1].free.append(str(e.get("property")))
        b = reading.get("budget")
        if isinstance(b, (int, float)) and not isinstance(b, bool) and b > 0:
            self.budget = b

    def read_answers(self, sent, models=None):
        if self.quiet(sent):
            self.trace.append({"kind": "quiet_answer"})
            return {}
        self.reply_reads += 1
        il = "\n".join("item %d: %s" % (i, it.name) for i, it in enumerate(self.items, 1))
        blocks = []
        for k, (n, q, a) in enumerate(sent, 1):
            tag = ("about item %d" % n) if n else "about several items"
            blocks.append("%d. (%s) Q: %s\nA: %s" % (k, tag, q, a))
        prompt = REPLY_PROMPT + "\n\nITEMS:\n" + il + "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(blocks)
        return self.ask_json(prompt, models or [HEAVY, LIGHT], 2500) or {}

    def quiet(self, sent):
        return self.soft("quiet", False, self._quiet, sent)

    def _quiet(self, sent):
        if self.reply_reads == 0:
            return False
        known = set()
        for it in self.items:
            known |= spelled(it.name) | spelled(it.shelf)
            for r in it.musts + it.told:
                known |= spelled(" ".join(ALNUM.findall(r.partition(":")[0].lower())))
        for lab, n, kept in self.key_table():
            known |= spelled(lab)
            for slot in kept:
                known |= spelled(slot.wf)
        for rec in self.full.values():
            blk = attr_of(rec) or {}
            for v in blk.values():
                known |= spelled(" ".join(w for w in ALNUM.findall(str(v).lower()) if len(w) >= QUIET_WORD_LEN))
        for n, q, a in sent:
            qw = spelled(" ".join(ALNUM.findall(Q_FILLER.sub(" ", q).lower())))
            aw = spelled(" ".join(ALNUM.findall(str(a).lower())))
            if ANY_DIGIT.search(str(a)) or (aw & known) or (aw & qw) or V5_HOW.search(str(a)):
                return False
        return True


COUNT_KEY = re.compile(r"\b(?:qty|quantity|count|units?)\b")
LIMIT_WORD = re.compile(r"\b(?:not|no|least|most|under|over|min|max|minimum|maximum|unlike|different)\b")
REMOVE_OPT = re.compile(r"remov|delet|drop|clear|take", re.I)
VAL_SPLIT = re.compile(r"[,/;|]")
COMPARATIVE = re.compile(r"cheapest|most|least")


class E7(object):
    def gate(self, stage):
        due = [d for d in self.duties if d.stage == stage]
        if not due or self.finished or not self.gate_due(stage):
            return False, ""
        excl = [self.role("buy"), self.role("refuse")]
        tl = []
        for t in self.tools:
            fn = t["function"]
            if fn["name"] in excl:
                continue
            tl.append("- %s: %s | parameters: %s" % (fn["name"], fn.get("description") or "", dump(fn.get("parameters") or {})))
        phrase = "place the order" if stage == "order" else "end without buying anything"
        said, nudged = [], False
        talk = stage == "order" and any(d.messages >= 1 for d in due)
        for _ in range(GATE_ROUNDS):
            rec = "\n".join("%d. %s" % (i + 1, l) for i, l in enumerate(self.done_log)) or "(nothing recorded yet)"
            p = (GATE_PROMPT.replace("STAGE", phrase) + "\n\nREQUEST:\n" + self.request + "\n\nSTEPS DUE NOW:\n"
                 + "\n".join("- " + d.text for d in due) + "\n\nLEDGER:\n" + self.ledger() + "\n\nCART: " + self.cart_text()
                 + "\n\nRECORD (shopper messages and store actions so far):\n" + rec)
            if self.change_notes:
                p += "\n\nSTORE NOTES SEEN:\n" + "\n".join(self.change_notes[-4:])
            p += ("\n\nMessages left: %d   steps left: %d (the %s itself uses one step)"
                  % (max(0, self.msg_limit - self.msgs_used), self.steps_left(), "order" if stage == "order" else "ending"))
            p += "\n\nTOOLS:\n" + "\n".join(tl)
            ans = self.ask_json(p, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if ans is None:
                break
            if said:
                lines = []
                self.take_reading({"needs": ans.get("needs")}, lines)
                hold = str(ans.get("hold") or "").strip()
                if lines or hold:
                    self.version += 1
                    txt = ""
                    if said:
                        txt = "Cairn carried out the request's own steps before continuing:\n" + "\n".join(said) + "\n"
                    txt += ("The shopper's answer alters the task: " + "; ".join(lines + ([hold] if hold else []))
                            + ". Repair the cart accordingly and check out once more.")
                    return True, txt
            call = ans.get("call")
            if not isinstance(call, dict):
                proof = set()
                for i, st in enumerate(self.done_steps):
                    if not talk or st > self.cart_changed_at:
                        proof.add(i + 1)
                unproven = []
                for s in items_of(ans.get("steps")):
                    if not isinstance(s, dict):
                        continue
                    cited = [count_of(x) for x in items_of(s.get("lines")) if not isinstance(x, bool)]
                    if s.get("done") is not True or not any(c in proof for c in cited if c is not None):
                        unproven.append(s)
                if (unproven and not said and not nudged and self.role("ask") and self.msgs_used < self.msg_limit
                        and self.steps_left() >= 2):
                    nudged = True
                    texts = "; ".join(str(s.get("step") or "") for s in unproven)[:200]
                    self.done_log.append("(Cairn: none of the record lines written after the cart's latest change at step %d "
                                         "proves: %s. Messages left: %d, steps left: %d. Name the action that performs it.)"
                                         % (self.cart_changed_at, texts, self.msg_limit - self.msgs_used, self.steps_left()))
                    self.done_steps.append(-1)
                    continue
                self.gate_passes.add((stage, self.pass_mark()))
                break
            tool = call.get("tool")
            if tool not in self.schemas or tool in excl:
                break
            if self.steps_left() < 2:
                break
            if tool == self.role("ask") and self.msgs_used >= self.msg_limit:
                break
            f = []
            a = shape(call.get("args") if isinstance(call.get("args"), dict) else {}, self.schemas.get(tool), "args", f)
            if f:
                break
            if tool == self.role("ask"):
                strs = strings_of(a)
                if not strs:
                    break
                q = max(strs, key=len)
                sent = self.send([(1, "", q)], 1)
                if not sent:
                    break
                self.gate_msgs += 1
                said.append("Q: %s\nA: %s" % (q, sent[0][2]))
                continue
            r = self.env_call(tool, a)
            if r is None:
                break
            o = r.get("observation")
            self.absorb(o)
            shown = self.display(o)
            self.done_log.append("step %d: %s %s -> %s (cart then: %s)" % (self.steps, tool, dump(a), shown[:300], self.cart_mark()))
            self.done_steps.append(self.steps)
            said.append("%s %s -> %s" % (tool, dump(a), shown[:400]))
        if said:
            return False, "Cairn carried out the request's own steps before continuing:\n" + "\n".join(said)
        return False, ""

    def spoken(self, obs):
        if isinstance(obs, str):
            return obs
        out = []

        def walk(x, d):
            if d > 4:
                return
            if isinstance(x, str):
                if len(x.split()) >= 2:
                    out.append(x)
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v, d + 1)
            elif isinstance(x, list):
                for v in x:
                    walk(v, d + 1)
        walk(obs, 0)
        if out:
            return " ".join(out)[:800]
        return dump(obs)[:400]

    def loose_lines(self, obs):
        if not isinstance(obs, dict) or self.identity(obs, False, 0):
            return []
        lines = []
        for v in obs.values():
            if isinstance(v, str):
                if len(v.split()) >= 5:
                    lines.append(v)
            elif isinstance(v, list) and v and all(isinstance(s, str) for s in v):
                lines += [s for s in v if len(s.split()) >= 5]
            elif isinstance(v, dict) and not self.identity(v, False, 1):
                lines += [s for s in v.values() if isinstance(s, str) and len(s.split()) >= 6]
        out = []
        for l in lines:
            if l not in self.heard and l not in out:
                out.append(l)
        return out

    def read_note(self, lines):
        for l in lines:
            self.heard.add(l)
        p = NOTE_PROMPT + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger() + "\n\nNOTICE:\n" + "\n".join(lines)
        ans = self.ask_json(p, [LIGHT, SPARE], 2500) or {}
        gone = [str(x) for x in items_of(ans.get("gone")) if isinstance(x, str) or (isinstance(x, int) and not isinstance(x, bool))]
        known = [g for g in gone if g in self.rows]
        for g in known:
            self.unavailable.add(g)
        if gone or ans.get("sold_out") is True:
            self.note_sold = True
            self.note_named = bool(known)
            if len(lines) == 1:
                self.sold_lines.add(lines[0])
        kind = ans.get("kind")
        if kind is not None and kind != "change":
            return ""
        changes = []
        budget_changed = dropped_any = False
        b = ans.get("budget")
        if isinstance(b, (int, float)) and not isinstance(b, bool) and b > 0 and b != self.budget:
            self.budget = b
            changes.append("budget now %s" % gnum(b))
            budget_changed = True
        for e in items_of(ans.get("drop")):
            if not isinstance(e, dict) or not str(e.get("need") or "").strip():
                continue
            it = self.items[self.item_num(e.get("item")) - 1]
            need = str(e.get("need")).strip()
            live = it.live()
            m = need if need in live else None
            if m is None:
                np = word_form(need.partition(":")[0])
                for r in live:
                    if word_form(r.partition(":")[0]) == np:
                        m = r
                        break
            it.dropped.append(m or need)
            changes.append("dropped " + need)
            dropped_any = True
        for e in items_of(ans.get("add")):
            if not isinstance(e, dict) or not str(e.get("need") or "").strip():
                continue
            it = self.items[self.item_num(e.get("item")) - 1]
            need = str(e.get("need")).strip()
            while need in it.dropped:
                it.dropped.remove(need)
            if need not in it.musts and need not in it.told:
                it.told.append(need)
            changes.append("added " + need)
        if gone:
            changes.append("unavailable: " + ", ".join(gone))
        if ans.get("price_moved") is True:
            changes.append("a price moved")
        if self.brief.get("pick") == "cheapest" and (dropped_any or gone or ans.get("sold_out") is True or budget_changed):
            self.wider_held = False
            self.cheaper_marks = set()
        summary = str(ans.get("summary") or "").strip()
        if changes or summary:
            self.version += 1
            self.recent = []
            self.change_notes.append(summary or "; ".join(changes))
        out = summary
        if changes:
            out += " (" + "; ".join(changes) + ")"
        return out.strip()

    def live_pairs(self):
        return [(i, r) for i, it in enumerate(self.items, 1) for r in it.live()]

    def evidence(self, obs):
        ids = []

        def walk(x, d, inlist):
            if d > 6:
                return
            if isinstance(x, dict):
                lid = self.identity(x, inlist, d)
                if lid and attr_of(x) is not None:
                    if lid not in ids:
                        ids.append(lid)
                    return
                for v in x.values():
                    walk(v, d + 1, False)
            elif isinstance(x, list):
                for v in x:
                    walk(v, d + 1, True)
        walk(obs, 0, False)
        return self.evidence_ids(ids)

    def evidence_ids(self, ids):
        live = self.live_pairs()
        if not ids or not live:
            return ""
        many = len(self.items) > 1
        lines = []
        for lid in ids[:12]:
            blk = attr_of(self.full.get(lid)) or {}
            parts = []
            for i, r in live[:8]:
                prop = r.partition(":")[0].strip()
                ps = stems(prop)
                hits = ["%s=%s" % (k, str(v)[:80]) for k, v in blk.items() if names_match(ps, stems(k))][:3]
                if hits:
                    parts.append("%s%s -> %s" % (("item %d " % i) if many else "", prop, ", ".join(hits)))
            if parts:
                lines.append("%s: %s" % (lid, "; ".join(parts)))
        if not lines:
            return ""
        return ("[evidence: each listed key carries some requirement's property inside its name; a key about another "
                "property (another component or aspect) settles nothing either way; a differently named key might still "
                "hold that same property] " + " | ".join(lines))

    def fresh_view(self):
        return self.cart_view is not None and self.cart_seen_at >= self.cart_changed_at

    def cart_text(self):
        if self.fresh_view():
            return dump(clip(self.cart_view, 300))[:3000]
        return "(from Cairn's cart record, listing: units) " + dump(self.cart)

    def units_held(self, lid):
        if self.fresh_view():
            stack = [self.cart_view]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    if lid in [v.strip() for v in x.values() if isinstance(v, str)]:
                        cands = [count_of(v) for k, v in x.items() if not isinstance(v, str) and count_of(v) is not None
                                 and COUNT_KEY.search(word_form(k))]
                        if len(cands) == 1:
                            return cands[0]
                    for v in x.values():
                        if isinstance(v, (dict, list)):
                            stack.append(v)
                elif isinstance(x, list):
                    stack.extend(x)
        return self.cart.get(lid, 1)

    def held_units(self):
        return {x: self.units_held(x) for x in self.cart_ids()}

    def cart_ids(self):
        if self.fresh_view():
            vt = dump(self.cart_view)
            return [x for x in self.seen if x in vt]
        return list(self.cart.keys())

    def record_text(self, lid):
        rec = self.full.get(lid) or self.rows.get(lid)
        if rec is None:
            return "(no record seen)"
        return dump(clip(rec, 400))[:3000]

    def other_listings(self, skip, limit=30):
        ids = list(reversed(self.seen))
        if self.brief.get("pick") == "cheapest":
            ids.sort(key=lambda x: (x not in self.full, self.price_of(x) is None, self.price_of(x) or 0))
        out = []
        for x in ids:
            if x in skip or self.sold_out(x):
                continue
            rec = self.full.get(x) or self.rows.get(x)
            out.append(dump(clip(rec, 160))[:900])
            if len(out) >= limit:
                break
        return out

    def fit_index(self):
        key = (len(self.seen), len(self.full), sum(len(v) for v in self.views.values()))
        if self.fit_cache and self.fit_cache[0] == key:
            return self.fit_cache[1]
        idx, notes = {}, {}
        for lid in self.seen:
            rec = self.full.get(lid) or self.rows.get(lid)
            if not isinstance(rec, dict):
                continue
            text = " ".join(v for v in rec.values() if isinstance(v, str))
            subs = []
            for v in rec.values():
                if isinstance(v, dict):
                    subs += ["%s %s" % (k, s) for k, s in v.items() if isinstance(s, str)]
            text += " " + ". ".join(subs)
            m = fit_marks(text)
            idx[lid] = m
            for part, codes in m["notes"]:
                notes.setdefault(part, []).extend(codes)
        for lid, m in idx.items():
            fs = list(m["works"])
            if m["own"]:
                fs += [c for c in notes.get(m["own"], []) if c not in fs]
            m["fitset"] = fs
        self.fit_cache = (key, idx)
        return idx

    def item_ids(self, num):
        if not self.get_cat_key():
            return []
        return [x for x in self.seen if self.owns_label(self.label_of(x), num)]

    def series_system(self, part, main):
        fi = self.fit_index()
        for carrier, namer in ((part, main), (main, part)):
            cl = self.item_ids(carrier)
            tagged = [x for x in cl if fi.get(x, {}).get("series")]
            if len(tagged) < SERIES_ROWS or len(tagged) * 2 < len(cl):
                continue
            fams = set()
            for x in tagged:
                fams |= set(fi[x]["series"])
            if len(fams) > SERIES_BREADTH:
                continue
            if any(set(fi.get(y, {}).get("named") or []) & fams for y in self.item_ids(namer)):
                return (carrier, namer, fams)
        return None

    def link_holds(self, carried, naming, fams):
        fi = self.fit_index()
        a = set(fi.get(carried, {}).get("series") or []) & fams
        b = set(fi.get(naming, {}).get("named") or []) & fams
        if a and b:
            return bool(a & b)
        return None

    def link_faults(self, part, main, device, part_lines):
        sysm = self.series_system(part, main)
        if not sysm:
            return []
        carrier, namer, fams = sysm
        fi = self.fit_index()
        out = []
        for pl in part_lines:
            carried, naming = (pl, device) if carrier == part else (device, pl)
            if self.link_holds(carried, naming, fams) is not False:
                continue
            wanted = set(fi[naming]["named"]) & fams
            alts = [x for x in self.item_ids(carrier) if set(fi.get(x, {}).get("series") or []) & wanted
                    and not self.sold_out(x) and x != carried][:4]
            txt = "%s (series %s) against %s (takes %s)" % (carried, ", ".join(sorted(fi[carried]["series"])), naming,
                                                           ", ".join(sorted(wanted)))
            if alts:
                txt += "; seen listings carrying that series: " + ", ".join(alts)
            else:
                txt += ("; search for that series together with the item's own words, or take a different %s, one with a "
                        "series carried by the wanted lines" % self.items[main - 1].name)
            out.append(txt)
        return out

    def plan_series(self, num):
        fi = self.fit_index()
        ids = self.cart_ids()
        cnt, held, order = {}, [], []
        found = False
        for p, m in self.fit_pairs:
            if num not in (p, m):
                continue
            sysm = self.series_system(p, m)
            if not sysm:
                continue
            found = True
            fams = sysm[2]
            for it in (p, m):
                for x in self.item_ids(it):
                    mk = fi.get(x) or {}
                    for s in (set(mk.get("series") or []) | set(mk.get("named") or [])) & fams:
                        if s not in cnt:
                            cnt[s] = 0
                            order.append(s)
                        cnt[s] += 1
                        if x in ids and s not in held:
                            held.append(s)
        if not found:
            return []
        rest = sorted([s for s in order if s not in held], key=lambda s: (-cnt[s], s))
        return (held + rest)[:SERIES_HUNTED]

    def covers(self, lid, dcodes):
        fi = self.fit_index()
        m = fi.get(lid) or {}
        fs, ex = m.get("fitset") or [], m.get("barred") or []
        return any(code_covers(c, d) for c in fs for d in dcodes) and not any(code_covers(c, d) for c in ex for d in dcodes)

    def fit_faults(self):
        if not self.fit_pairs:
            return []
        fi = self.fit_index()
        ids = self.cart_ids()
        out = []
        for part, main in self.fit_pairs:
            dev = [x for x in ids if self.owns_label(self.label_of(x), main)]
            parts = [x for x in ids if self.owns_label(self.label_of(x), part)]
            if len(dev) != 1 or not parts:
                continue
            lf = self.link_faults(part, main, dev[0], parts)
            if lf:
                out += lf
                continue
            dcodes = (fi.get(dev[0]) or {}).get("models") or []
            if not dcodes:
                continue
            for pl in parts:
                fs = (fi.get(pl) or {}).get("fitset") or []
                if not fs or self.covers(pl, dcodes):
                    continue
                alts = [x for x in self.item_ids(part) if self.covers(x, dcodes) and not self.sold_out(x) and x not in ids][:4]
                txt = "%s (%s) is meant for %s, against %s (%s)" % (pl, self.items[part - 1].name, ", ".join(sorted(fs)[:4]),
                                                                   dev[0], ", ".join(sorted(dcodes)))
                if alts:
                    txt += "; compatible parts of the same sort: " + ", ".join(alts)
                else:
                    txt += ("; none of the seen parts of this sort suits the device. Hunt for a part that suits it, or "
                            "change the device: " + self.fit_plan(True))
                out.append(txt)
        return out

    def fit_plan(self, short=False):
        if not self.fit_pairs:
            return ""
        fi = self.fit_index()
        lines = []
        for main in sorted(set(m for p, m in self.fit_pairs)):
            parts = [p for p, m in self.fit_pairs if m == main]
            tally, order = {}, []
            for p in parts:
                for x in self.item_ids(p):
                    if self.sold_out(x):
                        continue
                    for c in set((fi.get(x) or {}).get("fitset") or []):
                        if c not in tally:
                            tally[c] = {}
                            order.append(c)
                        tally[c][p] = tally[c].get(p, 0) + 1
                sysm = self.series_system(p, main)
                if sysm and sysm[0] == p:
                    for x in self.item_ids(p):
                        if self.sold_out(x):
                            continue
                        for s in set((fi.get(x) or {}).get("series") or []) & sysm[2]:
                            if s not in tally:
                                tally[s] = {}
                                order.append(s)
                            tally[s][p] = tally[s].get(p, 0) + 1
            ranked = sorted(order, key=lambda c: (-len(tally[c]), -sum(tally[c].values())))[:6]
            segs = []
            for c in ranked:
                devs = []
                for x in self.item_ids(main):
                    mk = fi.get(x) or {}
                    if c in (mk.get("named") or []) or any(code_covers(c, d) for d in (mk.get("models") or [])):
                        devs.append(x)
                    if len(devs) >= 2:
                        break
                counts = ", ".join("%d %s" % (tally[c][p], self.items[p - 1].name) for p in parts if p in tally[c])
                segs.append("%s: %s (%s)" % (c, counts, ("devices seen: " + ", ".join(devs)) if devs else "no device has turned up yet"))
            if segs:
                lines.append("codes the parts name for the %s candidates -> %s" % (self.items[main - 1].name, "; ".join(segs)))
            if self.brief.get("pick") == "cheapest":
                cs = self.soft("cheapest_sets", "", self.cheapest_sets, main, parts)
                if cs:
                    lines.append(cs)
        if not lines:
            return ""
        if short:
            return " | ".join(lines)
        return ("[fit plan: a part suits a device if the part's wording mentions the device's model number or an "
                "alternative trade name of it, or if the device's wording mentions the series the part's code belongs to; "
                "try a single query combining one code (or one series) with the item's own terms, and if that returns "
                "nothing, browse the category the device belongs to and read the codes off its rows] " + "\n".join(lines))

    def cheapest_sets(self, main, parts):
        fi = self.fit_index()
        shelves, systems = {}, {}
        for p in parts:
            systems[p] = self.series_system(p, main)
            sh = [x for x in self.item_ids(p) if self.price_of(x) is not None and self.buyable(x)]
            sh.sort(key=lambda x: (self.price_of(x), x))
            shelves[p] = sh
        sets = []
        for d in self.item_ids(main):
            m = fi.get(d) or {}
            if not (m.get("models") or m.get("named")) or self.price_of(d) is None or not self.buyable(d):
                continue
            total = self.price_of(d) * self.items[main - 1].quantity
            ids, ok = [d], True
            for p in parts:
                fit = None
                for x in shelves[p]:
                    s = systems[p]
                    if self.covers(x, m.get("models") or []) or (s and s[0] == p and self.link_holds(x, d, s[2]) is True):
                        fit = x
                        break
                if not fit:
                    ok = False
                    break
                total += self.price_of(fit) * self.items[p - 1].quantity
                ids.append(fit)
            if ok:
                sets.append((total, ids))
        sets.sort(key=lambda s: s[0])
        if not sets:
            return ""
        return ("cheapest complete sets seen (requirements still unverified): "
                + "; ".join("+".join(ids) + "=" + gnum(t) for t, ids in sets[:CHEAP_SETS]))

    def missing_items(self):
        if not self.get_cat_key():
            return []
        lines = [x for x in self.cart_ids() if self.units_held(x) > 0]
        if len(lines) >= len(self.items):
            return []
        out = []
        for i in range(1, len(self.items) + 1):
            if any(self.owns_label(self.label_of(x), i) for x in self.seen) and\
                    not any(self.owns_label(self.label_of(x), i) for x in lines):
                out.append(i)
        return out

    def states_all(self, lid, num):
        blk = attr_of(self.full.get(lid)) or {}
        for r in self.items[num - 1].live():
            prop, _, val = r.partition(":")
            vw = set(VAL_SPLIT.sub(" ", word_form(val)).split())
            if not vw or LIMIT_WORD.search(" ".join(vw)):
                return False
            ok = False
            for k, v in blk.items():
                if names_match(spelled(prop), spelled(k)) and vw <= set(VAL_SPLIT.sub(" ", word_form(str(v))).split()):
                    ok = True
                    break
            if not ok:
                return False
        return True

    def fill_short_cart(self):
        bound = set()
        for p, m in self.fit_pairs:
            bound.add(p)
            bound.add(m)
        picks = []
        for i in self.missing_items():
            it = self.items[i - 1]
            cands = [x for x in self.item_ids(i) if x in self.full and x not in self.cart and self.buyable(x)
                     and self.price_of(x) is not None and self.states_all(x, i)]
            cands.sort(key=lambda x: self.price_of(x))
            pick = cands[0] if cands and i not in bound else None
            if pick and isinstance(self.budget, (int, float)):
                each = self.brief.get("budget_scope") == "each"
                held = sum((self.price_of(x) or 0) * self.units_held(x) for x in self.cart_ids())
                cost = self.price_of(pick) * (1 if each else it.quantity)
                if cost + (0 if each else held) - self.budget > 1e-9:
                    pick = None
            if not pick:
                return False
            picks.append((pick, it.quantity))
        if not picks:
            return True
        target = {x: self.cart[x] for x in self.cart_ids() if x in self.cart}
        for p, q in picks:
            target[p] = q
        calls = self.cart_calls(target)
        if not calls or len(calls) > self.steps_left() - 1:
            return False
        for tool, a in calls:
            f = []
            a = shape(a, self.schemas.get(tool), "args", f)
            if f:
                return False
            r = self.env_call(tool, a)
            if r is None:
                return False
            self.note_cart(tool, a, r.get("observation"))
        self.trace.append({"kind": "short_cart_filled", "picks": picks})
        return True

    def part_shelves(self):
        ck = self.get_cat_key()
        if not ck:
            return
        labels = sorted(set(str(r.get(ck)).strip() for r in self.rows.values() if r.get(ck) is not None and str(r.get(ck)).strip()))
        moved = False
        for i, it in enumerate(self.items, 1):
            nr = roots(it.name)
            if not nr or not it.shelf:
                continue
            others = [o for j, o in enumerate(self.items, 1) if j != i]
            if not any(word_form(o.shelf) == word_form(it.shelf) for o in others):
                continue
            if any(roots(o.name) == nr for o in others):
                continue
            own = [l for l in labels if roots(l) == nr and word_form(l) != word_form(it.shelf)]
            if len(own) == 1 and not any(word_form(o.shelf) == word_form(own[0]) for o in self.items):
                it.shelf = own[0]
                self.trace.append({"kind": "shelf_parted", "item": i, "shelf": own[0]})
                moved = True
        if moved:
            self.owners_seen = {}
            self.fit_cache = None

    def fuller_label(self, num, lab):
        it = self.items[num - 1]
        nr = roots(it.name)
        lr = roots(lab or "")
        ck = self.get_cat_key()
        if len(nr) < 2 or nr <= lr or not ck or (it.shelf and lr == roots(it.shelf)):
            return None
        labels = sorted(set(str(r.get(ck)) for r in self.rows.values() if r.get(ck) is not None and str(r.get(ck))))
        for l in labels:
            if nr <= roots(l):
                return l
        return None

    def named_label(self, num, lab):
        it = self.items[num - 1]
        if not it.shelf or not self.get_cat_key() or word_form(lab or "") == word_form(it.shelf):
            return None
        for x in self.seen:
            v = self.label_of(x)
            if v is not None and word_form(v) == word_form(it.shelf):
                return v
        return None

    def line_faults(self):
        self.adopt_strays()
        ck = self.get_cat_key()
        lines = [x for x in self.cart_ids() if self.units_held(x) > 0]
        if not lines or not ck:
            return []
        N = len(self.items)
        owners = {x: [i for i in range(1, N + 1) if self.owns_label(self.label_of(x), i)] for x in lines}
        known = [i for i in range(1, N + 1) if any(self.owns_label(self.label_of(x), i) for x in self.seen)]
        out = []
        for i in known:
            it = self.items[i - 1]
            mine = [x for x in lines if i in owners[x]]
            if not mine:
                out.append(("missing", "item %d (%s) lacks a cart line" % (i, it.name)))
                continue
            alone = [x for x in mine if owners[x] == [i]]
            if len(alone) > 1:
                out.append(("several", "item %d (%s) has %d lines (%s); keep one" % (i, it.name, len(mine), ", ".join(mine))))
            for x in mine:
                if owners[x] == [i]:
                    u = self.units_held(x)
                    if u != it.quantity:
                        out.append(("count", "%s is held at %s unit%s; item %d (%s) wants %d"
                                    % (x, u, "" if u == 1 else "s", i, it.name, it.quantity)))
                lab = self.label_of(x)
                alt = self.fuller_label(i, lab) or self.named_label(i, lab)
                if alt:
                    out.append(("label", "%s comes from %s while the store also has %s, which names item %d (%s) "
                                "completely; source this line from %s instead" % (x, lab, alt, i, it.name, alt)))
        if len(known) == N:
            for x in lines:
                if not owners[x]:
                    out.append(("kind", "%s is of kind %s, which nobody requested; remove it" % (x, self.label_of(x) or "(unknown kind)")))
        for x in lines:
            if self.sold_out(x):
                out.append(("sold", "%s is sold out" % x))
        b = self.budget
        if isinstance(b, (int, float)) and not isinstance(b, bool) and all(self.price_of(x) is not None for x in lines):
            if self.brief.get("budget_scope") == "each":
                for x in lines:
                    p = self.price_of(x)
                    if b < p <= 10 * b:
                        out.append(("budget", "%s costs %s, above the per-item limit %s" % (x, gnum(p), gnum(b))))
            else:
                total = sum(self.price_of(x) * self.units_held(x) for x in lines)
                if total - b > 1e-9 and total <= 10 * b:
                    out.append(("budget", "the total %s is above the budget %s" % (gnum(total), gnum(b))))
        return out

    def count_fixes(self, faults=None):
        if not self.get_cat_key():
            return []
        N = len(self.items)
        fixes = []
        ids = self.cart_ids()
        for x in ids:
            u = self.units_held(x) or 0
            ow = [i for i in range(1, N + 1) if self.owns_label(self.label_of(x), i)]
            if u <= 0 or len(ow) != 1:
                continue
            i = ow[0]
            n_lines = sum(1 for y in ids if self.cart.get(y, 1) > 0 and self.owns_label(self.label_of(y), i))
            want = self.items[i - 1].quantity
            if n_lines != 1 or u == want or self.sold_out(x):
                continue
            fixes.append((x, u, want))
        tool = self.role("cart")
        sch = self.schemas.get(tool) if tool else None
        if not isinstance(sch, dict) or not fixes:
            return []
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        textp = [k for k, s in props.items() if isinstance(s, dict) and "string" in kinds(s) and not s.get("enum")]
        idp = None
        for t, k in self.id_args:
            if t == tool and k in textp:
                idp = k
                break
        if idp is None:
            idp = next((r for r in req if r in textp), None)
        if idp is None and len(textp) == 1:
            idp = textp[0]
        cntp = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in kinds(s)), None)
        modep = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not idp or not cntp:
            return []
        calls = []
        sets = self.cart_sets_counts()
        for x, u, w in fixes:
            if sets:
                calls.append({idp: x, cntp: w})
            elif modep:
                opts = props[modep]["enum"]
                rem = next((o for o in opts if REMOVE_OPT.search(str(o))), opts[1])
                add = next((o for o in opts if o != rem), opts[0])
                calls.append({idp: x, modep: rem, cntp: u})
                calls.append({idp: x, modep: add, cntp: w})
            else:
                return []
        if len(calls) + 3 > self.steps_left():
            return []
        done = []
        for a in calls:
            f = []
            a = shape(a, sch, "args", f)
            if f:
                break
            r = self.env_call(tool, a)
            if r is None:
                break
            o = r.get("observation")
            self.note_cart(tool, a, o)
            self.cart_said = dump(o)
            self.observe(tool, a, o)
            done.append("%s -> %s" % (dump(a), self.display(o)[:300]))
            if self.finished:
                break
        self.trace.append({"kind": "count_fixes", "done": len(done)})
        return done

    def cart_check(self, planned=None, limit=SOFT_SPEND):
        ids = list(planned) if planned else self.cart_ids()
        key = "|".join(sorted(ids)) + "#" + str(self.version)
        if key in self.checked or not self.judges(limit) or not ids:
            return ""
        self.checked.add(key)
        pick = self.brief.get("pick")
        comparative = pick in ("cheapest", "most", "least")
        p = CART_PROMPT + "\n\nREQUEST:\n" + self.request + "\n\nBINDING REQUIREMENTS:\n" + self.ledger()
        if planned:
            p += ("\n\nCART (only planned: judge it as though these listings sat in the cart at the asked quantities): "
                  + "; ".join("item %d: quantity %d" % (i, it.quantity) for i, it in enumerate(self.items, 1))
                  + "\n" + ", ".join(ids))
        else:
            p += "\n\nCART: " + self.cart_text()
        recs = []
        for x in ids:
            so = "SOLD OUT (the store reports it cannot be purchased any more) " if self.sold_out(x) else ""
            recs.append("%s: %s%s" % (x, so, self.record_text(x)))
        p += "\n\nRECORDS OF THE CARTED LISTINGS:\n" + "\n".join(recs)
        if comparative:
            p += "\n\nOTHER LISTINGS SEEN (newest first):\n" + "\n".join(self.other_listings(ids, 15))
        judges = self.judges(limit)
        ans = None
        if len(judges) == 2 and not planned and not self.audited and not comparative:
            ans = self.ask_json(p, [LIGHT], 3000)
            if ans is not None:
                judges = [HEAVY] if (ans.get("ok") is False and HEAVY in self.judges(limit)) else []
        if judges and not self.no_credit:
            a2 = self.ask_json(p, judges, 3000)
            if a2 is not None:
                ans = a2
        self.trace.append({"kind": "cart_check", "ids": ids, "ok": None if ans is None else ans.get("ok")})
        self.verdicts[key] = ans is not None and ans.get("ok") is not False
        if ans is None or ans.get("ok") is not False:
            return ""
        self.leads = [x for x in self.leads if x not in ids]
        probs = [str(q) for q in items_of(ans.get("problems"))][:5]
        better = [str(b) for b in items_of(ans.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        out = "A careful review finds the cart still wrong: " + ("; ".join(probs) if probs else "see the requirements in the ledger") + "."
        if better:
            out += " Candidates that seem preferable: " + ", ".join(better) + "."
        return out + " Repair the cart, or else demonstrate that the review erred by reading the records afresh, and then check out."

    def planned_ok(self, ids, limit=HARD_SPEND):
        self.cart_check(planned=ids, limit=limit)
        return self.verdicts.get("|".join(sorted(ids)) + "#" + str(self.version)) is True


LINK_STOP1 = re.compile(r"the|and|with|from|that|comes|listing|page|its|for|of")
LINK_STOP2 = re.compile(r"the|and|with|from|that|comes|listing|page|its|for")
DIGIT_RUN = re.compile(r"\d+")
STOCK_DECLINE = "I am sorry: no product in the shop satisfies all of your requirements, so no purchase is made."


def token_in(name, text):
    return re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", text) is not None


class E8(object):
    def page_read(self, lid):
        if lid not in self.opened:
            return False
        off = self.role("offer")
        if off and off != self.role("open") and self.single_target(off):
            return off in self.views.get(lid, set())
        return True

    def pending_refs(self):
        out = []
        for n, r in self.refs:
            if 1 <= n <= len(self.items) and r in self.items[n - 1].live() and r not in self.ref_values and r not in out:
                out.append(r)
        return out

    def ref_trail(self, limit=None):
        pending = self.pending_refs()
        op = self.role("offer") or self.role("open")
        tgt = self.single_target(op) if op else None
        seen_lines, lost = [], []
        while (pending and self.chain_rounds < min(6 if self.steps_left() > 15 else 4, self.chain_room or 6)
               and self.spend < HARD_SPEND and (limit is None or self.clock() <= limit)):
            self.chain_rounds += 1
            recs, nopen = [], 0
            for lid in reversed(self.seen):
                rec = self.full.get(lid) or self.rows.get(lid)
                if rec is not None and self.page_read(lid) and nopen < 45:
                    recs.append("OPENED " + dump(clip(rec, 300))[:1400])
                    nopen += 1
            for lid in reversed(self.seen):
                if len(recs) >= 60:
                    break
                rec = self.full.get(lid) or self.rows.get(lid)
                if rec is not None and self.page_read(lid):
                    continue
                if lid in self.full:
                    recs.append("ROW " + dump(clip(self.full[lid], 120))[:700])
                elif lid in self.rows:
                    recs.append("ROW " + dump(clip(self.rows[lid], 80))[:300])
            p = (CHAIN_PROMPT + "\n\nREQUIREMENTS WHOSE VALUE SITS ON ANOTHER LISTING:\n" + "\n".join(pending)
                 + "\n\nRECORDS (newest first):\n" + "\n".join(recs))
            models = [LIGHT] + ([HEAVY] if self.spend + HEAVY_ROOM <= HARD_SPEND else [])
            ans = self.ask_json(p, models, 4500) or {}
            reach, lost, gaps = [], [], []
            for ch in items_of(ans.get("chains")):
                if not isinstance(ch, dict):
                    continue
                need = str(ch.get("need") or "").strip()
                req = need if need in pending else (pending[0] if len(pending) == 1 else None)
                if not req:
                    continue
                links = [l for l in items_of(ch.get("links")) if isinstance(l, dict)]
                lids = [str(l.get("id") or "").strip() for l in links]
                allread = bool(links) and all(self.page_read(x) for x in lids)
                val = str(ch.get("value") or "").strip()
                if ch.get("done") is True and allread and val:
                    self.ref_values[req] = val
                    self.version += 1
                    continue
                if allread and not str(links[-1].get("find") or "").strip():
                    self.ref_values[req] = val or ("whatever the listing %s shows" % lids[-1])
                    self.version += 1
                    continue
                unread = [x for x in lids if x in self.rows and not self.page_read(x)]
                if unread:
                    for x in unread:
                        if x not in reach:
                            reach.append(x)
                    continue
                ll = {}
                for l, x in zip(links, lids):
                    if x not in self.rows:
                        ll = l
                        break
                desc = str(ll.get("about") or "").strip() or "the listing this requirement leads to"
                words = str(ll.get("find") or "").strip()
                lost.append("%s -> find %s%s" % (req, desc, (" (search: %s)" % words) if words else ""))
                gaps.append((desc, words))
            pending = [r for r in pending if r not in self.ref_values]
            if not reach and gaps and op and tgt and self.steps_left() > 12 and self.link_searches < 3:
                ids, known = self.link_finder(gaps[0][0], gaps[0][1])
                reach = ids
                if known:
                    seen_lines.append("%s: a page, already open, of the shop the chain names; it shows the values listed in "
                                      "the description. Take the value off that page." % known)
                    lost = lost[1:]
            if not reach or not op or not tgt or self.steps_left() < len(reach) + 4 or self.finished:
                break
            n = max(0, min(3, 6 - len(seen_lines)))
            for lid in reach[:n]:
                f = []
                a = shape({tgt: lid}, self.schemas.get(op), "args", f)
                if f or self.rushed():
                    break
                r = self.env_call(op, a)
                if r is None:
                    break
                o = r.get("observation")
                self.observe(op, a, o)
                seen_lines.append("%s: %s" % (lid, self.display(o)[:1500]))
                if self.finished:
                    break
        parts = []
        if seen_lines:
            parts.append("[Cairn opened these pages, which the requirements lead to]\n" + "\n".join(seen_lines))
        if self.ref_values:
            parts.append("[values taken from those pages] " + "; ".join("%s -> %s" % (k, v) for k, v in self.ref_values.items()))
        hold = False
        if lost and self.chain_holds < (3 if self.steps_left() > 15 else 2) and self.steps_left() > 5:
            self.chain_holds += 1
            hold = True
            parts.append("The purchase is valid only once you have read completely every page a requirement leads to (a "
                         "result row will not do), including each page that leads on to yet another listing. Not yet "
                         "located or read: " + " | ".join(lost[:3]) + ". Search using the shop's name plus the values in "
                         "the description (another product of that kind sold by the same shop is the wrong one), open "
                         "whichever row from that shop has a page showing those values, use the value printed on that "
                         "page, and then order.")
        return "\n".join(parts).strip(), hold

    def early_trail(self):
        if self.steps_left() < TRAIL_ROOM or self.clock() > HUNT_DEADLINE or not self.pending_refs():
            return ""
        save = (self.chain_rounds, self.chain_holds)
        self.chain_rounds = 0
        self.chain_room = EARLY_ROUNDS
        try:
            text, hold = self.ref_trail(HUNT_DEADLINE)
        finally:
            self.chain_rounds, self.chain_holds = save
            self.chain_room = None
        return text

    def decline_trail(self):
        if self.decline_chain_done:
            return ""
        if self.steps_left() < TRAIL_ROOM or self.clock() > HUNT_DEADLINE or not self.pending_refs():
            return ""
        self.decline_chain_done = True
        text, hold = self.ref_trail(HUNT_DEADLINE)
        if text:
            return text + " Run searches for those values at once and purchase whatever matches; decline only if nothing matches."
        return ""

    def nonname_keys(self):
        return set(k for k in (self.id_key, self.get_cat_key(), self.get_title_key()) if k)

    def name_cands(self, row, nk):
        out = []
        for k, v in (row or {}).items():
            if k in nk or not isinstance(v, str):
                continue
            s = v.strip()
            if 2 < len(s) <= 40 and not V24_PLACEHOLDER.fullmatch(s.lower()):
                out.append(s)
        return out

    def shop_rows(self, desc, ids, nk):
        d = desc.lower()
        return [x for x in ids if x not in self.opened and x in self.rows
                and any(token_in(c.lower(), d) for c in self.name_cands(self.rows[x], nk))]

    def opened_shops(self, desc, nk):
        d = desc.lower()
        return [x for x in self.seen if x in self.opened and any(token_in(c.lower(), d) for c in self.name_cands(self.rows.get(x), nk))]

    def link_finder(self, desc, words):
        nk = self.nonname_keys()
        namew = set()
        for x in self.shop_rows(desc, list(self.rows), nk) + self.opened_shops(desc, nk):
            for k, v in (self.rows.get(x) or {}).items():
                if k in nk or not isinstance(v, str):
                    continue
                if len(v.strip()) <= 40 and not V24_PLACEHOLDER.fullmatch(v.strip().lower()):
                    namew |= set(ALNUM3.findall(v.lower()))
        src = words or desc
        valw = [w for w in ALNUM.findall(src.lower()) if (len(w) >= 3 or w.isdigit()) and w not in namew
                and not LINK_STOP1.fullmatch(w)]
        for x in self.opened_shops(desc, nk):
            rec = self.full.get(x) or self.rows.get(x)
            leaf = " ".join(str(s) for s in self.leaves(rec)).lower()
            if valw and all(token_in(w, leaf) for w in valw):
                return [], x
        tpl = self.value_template()
        key = " ".join(words.lower().split())
        cands = []
        if key in self.link_seen:
            cands = self.shop_rows(desc, self.link_seen[key], nk)
        elif tpl and key:
            qs = [words]
            two = " ".join(words.split()[:2])
            if two != words:
                qs.append(two)
            for q in qs:
                a = dict(tpl[1])
                a[tpl[2]] = q
                f = []
                a = shape(a, self.schemas.get(tpl[0]), "args", f)
                if f:
                    break
                self.link_searches += 1
                r = self.env_call(tpl[0], a)
                if r is None:
                    break
                o = r.get("observation")
                self.observe(tpl[0], a, o)
                ids = [x for x, _ in self.listing_records(o)]
                self.link_seen[key] = ids
                cands = self.shop_rows(desc, ids, nk)
                if ids:
                    break
        kindw = set(w for w in ALNUM3.findall(src.lower()) if w not in namew and not LINK_STOP2.fullmatch(w))

        def of_kind(x):
            if not kindw:
                return True
            return bool(set(ALNUM3.findall(dump(self.rows.get(x)).lower())) & kindw)
        res = [x for x in cands if of_kind(x)][:2]
        if not res:
            res = [x for x in self.shop_rows(desc, list(reversed(self.seen)), nk) if of_kind(x)][:2]
        return res, None

    def leaves(self, x):
        out = []
        if isinstance(x, dict):
            for v in x.values():
                out += self.leaves(v)
        elif isinstance(x, list):
            for v in x:
                out += self.leaves(v)
        elif x is not None:
            out.append(x)
        return out

    def reread_offers(self):
        op = self.role("offer") or self.role("open")
        tgt = self.single_target(op) if op else None
        if not op or not tgt:
            return ""
        cutoff = max(self.steps - 2, self.refused_at)
        stale = [x for x in self.cart_ids() if x not in self.gone and self.offer_seen.get(x, -99) < cutoff]
        if not stale or self.steps_left() < len(stale) + 5:
            return ""
        lines = []
        for x in stale:
            f = []
            a = shape({tgt: x}, self.schemas.get(op), "args", f)
            if f or self.rushed():
                break
            before = dict(self.rows.get(x) or {})
            r = self.env_call(op, a)
            if r is None or self.finished:
                break
            self.offer_seen[x] = self.steps
            o = r.get("observation")
            self.observe(op, a, o)
            after = self.rows.get(x) or {}
            changed = [k for k in before if k in after and after[k] != before[k] and not isinstance(after[k], (dict, list))]
            if changed:
                lines.append("%s changed (%s): %s" % (x, ", ".join(str(k) for k in changed[:4]), dump(clip(o, 300))[:900]))
        if lines:
            self.version += 1
        return "\n".join(lines)

    def swap_hint(self):
        newly = [x for x in self.cart_ids() if x not in self.gone and self.sold_out(x)]
        if not newly:
            return ""
        for x in newly:
            self.gone[x] = self.steps
        self.leads = [l for l in self.leads if l not in newly]
        self.version += 1
        reps = self.replacements(newly)
        ids = ", ".join(newly)
        out = ("[Cairn] %s sold out from now on: the store shows it cannot be purchased any more. It is gone for good and "
               "must not be opened again. " % ids)
        if reps and self.steps_left() <= 6:
            out += ("Steps are scarce: within this very turn put %s in the cart in the quantity asked and remove the line "
                    "that sold out (two calls), and then order. Other already-read listings that seem to match: %s."
                    % (reps[0], ", ".join(reps[1:]) or "none"))
        elif reps and self.brief.get("pick") == "cheapest":
            out += ("Already-read listings that seem to match that item, lowest price first: %s. Since the request wants "
                    "the lowest price, unless you have already seen every matching listing, you must make one search capped "
                    "below the price of %s (through the search's price ceiling argument, if it has one) before replacing "
                    "the line that sold out with the lowest-priced matching listing." % (", ".join(reps), reps[0]))
        elif reps:
            out += ("Already-read listings that seem to match that item: %s. Within this turn remove the line that sold out, "
                    "put %s in the cart in the quantity asked, and order after that." % (", ".join(reps), reps[0]))
        else:
            out += "Remove that line and put in the best other matching listing, searching when none of those already read matches."
        return out

    def sold_out_finder(self):
        lines = [x for x in self.cart_ids() if x not in self.gone]
        if any(self.sold_out(x) for x in lines):
            return ""
        if len(lines) == 1:
            self.unavailable.add(lines[0])
            return ""
        if len(lines) >= 2:
            return self.reread_offers()
        return ""

    def replacements(self, lost):
        js = self.judges()
        if not js:
            return []
        recs = []
        for x in reversed(self.seen):
            if x in lost or x in self.gone or self.sold_out(x):
                continue
            rec = self.full.get(x) or self.rows.get(x)
            recs.append(dump(clip(rec, 200))[:1200])
            if len(recs) >= 25:
                break
        if not recs:
            return []
        p = (SWAP_PROMPT + "\n\nREQUEST:\n" + self.request + "\n\nLIVE REQUIREMENTS:\n" + self.ledger()
             + "\n\nTHE LINE THAT SOLD OUT:\n" + "\n".join("%s: %s" % (x, self.record_text(x)) for x in lost)
             + "\n\nCURRENT CART: " + self.cart_text() + "\n\nREMAINING RECORDS SEEN (latest first):\n" + "\n".join(recs))
        ans = self.ask_json(p, list(reversed(js)), 3500) or {}
        now = self.cart_ids()
        kept = []
        fits = ans.get("fits")
        for e in (fits if isinstance(fits, list) else ([fits] if fits is not None else [])):
            lid = str(e.get("id")) if isinstance(e, dict) else str(e)
            pr = e.get("price") if isinstance(e, dict) else None
            pr = float(pr) if isinstance(pr, (int, float)) and not isinstance(pr, bool) else float("inf")
            if lid in self.rows and lid not in lost and lid not in self.gone and lid not in now and lid not in [k for k, _ in kept]:
                kept.append((lid, pr))
        if self.brief.get("pick") == "cheapest":
            kept.sort(key=lambda k: k[1])
        return [k for k, _ in kept[:4]]

    def audit(self):
        if self.audited or not self.judges():
            return ""
        self.audited = True
        ans = self.audit_call()
        fits = self.read_fits(ans)
        if not fits:
            op = (ans or {}).get("open")
            op = op if isinstance(op, list) else ([op] if op is not None else [])
            opn = [str(x) for x in op if str(x) in self.rows and str(x) not in self.opened and not self.sold_out(str(x))]
            if opn and self.open_for_audit(opn):
                ans = self.audit_call()
                fits = self.read_fits(ans)
        if not fits:
            return ""
        self.lead_items = fits
        self.leads = []
        for ids in fits.values():
            for x in ids:
                if x not in self.leads:
                    self.leads.append(x)
        reason = str((ans or {}).get("reason") or "").strip()
        out = "The review believes the following listings could match: " + "; ".join(
            "item %s: %s" % (k, ", ".join(v)) for k, v in fits.items())
        if reason:
            out += " (%s)" % reason
        return out + ". Open these listings and purchase what really matches; decline only when every one of them fails."

    def audit_call(self):
        js = self.judges()
        ids = list(reversed(self.seen))
        ids.sort(key=lambda x: x not in self.full)
        lines = []
        for x in ids:
            if self.sold_out(x):
                continue
            if x in self.full:
                lines.append("OPENED " + dump(clip(self.full[x], 200))[:1200])
            else:
                lines.append("ROW " + dump(clip(self.rows.get(x), 120))[:500])
            if len(lines) >= 30:
                break
        if not lines:
            return None
        p = (AUDIT_PROMPT + "\n\nREQUEST:\n" + self.request + "\n\nLIVE REQUIREMENTS:\n" + self.ledger()
             + "\n\nRECORDS SEEN (latest first):\n" + "\n".join(lines))
        ans = None
        if js:
            ans = self.ask_json(p, [LIGHT], 4000)
            if HEAVY in js and not self.no_credit and not self.read_fits(ans):
                a2 = self.ask_json(p, [HEAVY], 4000)
                if a2 is not None:
                    ans = a2
        return ans

    def read_fits(self, ans):
        if not isinstance(ans, dict) or not isinstance(ans.get("fits"), dict):
            return {}
        out = {}
        for k, v in ans["fits"].items():
            vals = v if isinstance(v, list) else [v]
            ids = [str(x) for x in vals if str(x) in self.rows and not self.sold_out(str(x))][:4]
            if ids:
                out[str(k)] = ids
        return out

    def open_for_audit(self, ids):
        cmp = self.role("compare")
        ct = self.compare_target(cmp) if cmp else None
        if cmp and ct and self.compare_facts() is not False:
            plan = [(cmp, {ct: ids[:self.compare_room()]})]
        else:
            ft = self.facts_tool()
            t = self.single_target(ft) if ft else None
            if not (ft and t):
                return False
            plan = [(ft, {t: x}) for x in ids[:2]]
        ph = {}
        for i in range(len(self.items)):
            ph["#%d" % i] = 1
        cs = self.cart_calls(ph)
        cart_steps = len(cs) if cs is not None else len(self.items)
        if self.steps_left() < len(plan) + max(5, self.reserve() + cart_steps):
            return False
        before = len(self.full)
        for tool, a in plan:
            f = []
            a = shape(a, self.schemas.get(tool), "args", f)
            if f:
                return False
            r = self.env_call(tool, a)
            if r is None or self.finished:
                break
            self.observe(tool, a, r.get("observation"))
        return len(self.full) > before

    def cart_props(self):
        tool = self.role("cart")
        sch = self.schemas.get(tool) if tool else None
        if not isinstance(sch, dict):
            return tool, {}, []
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        return tool, props, req

    def cart_sets_counts(self):
        tool, props, req = self.cart_props()
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in props.values()):
            return False
        for s in props.values():
            if isinstance(s, dict) and "integer" in kinds(s):
                mn = s.get("minimum")
                if isinstance(mn, (int, float)) and not isinstance(mn, bool) and mn == 0:
                    return True
        return False

    def cart_calls(self, want, drop=()):
        tool, props, req = self.cart_props()
        if not tool:
            return None
        for k, s in props.items():
            if isinstance(s, dict) and "array" in kinds(s) and isinstance(s.get("items"), dict)\
                    and isinstance(s["items"].get("properties"), dict):
                ep = s["items"]["properties"]
                idk = next((kk for kk, ss in ep.items() if isinstance(ss, dict) and "string" in kinds(ss) and not ss.get("enum")), None)
                cnk = next((kk for kk, ss in ep.items() if isinstance(ss, dict) and "integer" in kinds(ss)), None)
                if not idk:
                    return None
                els = []
                for x, u in want.items():
                    e = {idk: x}
                    if cnk:
                        e[cnk] = u
                    els.append(e)
                return [(tool, {k: els})]
        textp = [k for k, s in props.items() if isinstance(s, dict) and "string" in kinds(s) and not s.get("enum")]
        target = None
        for t, k in self.id_args:
            if t == tool and k in textp:
                target = k
                break
        if target is None:
            target = next((r for r in req if r in textp), None)
        if target is None and len(textp) == 1:
            target = textp[0]
        if not target:
            return None
        cnt = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        sets = self.cart_sets_counts()
        calls = []
        for x in drop:
            held = self.cart.get(x, 1)
            if mode:
                a = {target: x, mode: props[mode]["enum"][1]}
                if cnt:
                    a[cnt] = held
            elif cnt and sets:
                a = {target: x, cnt: 0}
            elif cnt and isinstance(props[cnt].get("minimum"), int) and props[cnt]["minimum"] <= -held:
                a = {target: x, cnt: -held}
            else:
                return None
            calls.append((tool, a))
        now = self.cart_ids()
        for x, u in want.items():
            held = self.cart.get(x, 0) if x in now else 0
            if held == u:
                continue
            if held > 0 and not sets:
                return None
            a = {target: x}
            if mode:
                a[mode] = props[mode]["enum"][0]
            if cnt:
                a[cnt] = u
            calls.append((tool, a))
        return calls

    def whole_removal(self, args):
        tool, props, req = self.cart_props()
        cnt = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in kinds(s)), None)
        removal = any(isinstance(props.get(k), dict) and isinstance(props[k].get("enum"), list)
                      and v in props[k]["enum"][1:] for k, v in args.items())
        line = next((v for v in args.values() if isinstance(v, str) and v in self.cart), None)
        if cnt and removal and line and cnt not in args:
            held = self.cart[line]
            mx = props[cnt].get("maximum")
            a = dict(args)
            a[cnt] = min(held, mx) if isinstance(mx, (int, float)) and not isinstance(mx, bool) else held
            return a
        return args

    def note_cart(self, tool, args, obs):
        self.cart_touched = True
        self.cart_changed_at = self.steps
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        if not isinstance(props, dict):
            props = {}
        args = args or {}
        removal = any(isinstance(props.get(k), dict) and isinstance(props[k].get("enum"), list) and len(props[k]["enum"]) >= 2
                      and v in props[k]["enum"][1:] for k, v in args.items())
        loose = None
        for v in args.values():
            if not isinstance(v, (str, bool)) and isinstance(v, (int, float)) and count_of(v) is not None:
                loose = count_of(v)
                break
        lines, whole = [], False
        for k, v in args.items():
            if isinstance(v, list):
                es = props.get(k, {}).get("items") if isinstance(props.get(k), dict) else None
                objel = isinstance(es, dict) and ("object" in kinds(es) or isinstance(es.get("properties"), dict))
                if not v and objel:
                    whole = True
                for e in v:
                    if isinstance(e, dict):
                        whole = True
                        lid = next((s for s in e.values() if isinstance(s, str) and s in self.rows), None)
                        n = next((count_of(s) for s in e.values() if not isinstance(s, (str, bool)) and count_of(s) is not None), None)
                        if lid:
                            lines.append((lid, n if n is not None else 1))
                    elif isinstance(e, str) and e in self.rows:
                        lines.append((e, loose))
            elif isinstance(v, str) and v in self.rows:
                lines.append((v, loose))
        if whole:
            self.cart = {lid: n for lid, n in lines if n and n > 0}
            return
        has_int = any(isinstance(s, dict) and "integer" in kinds(s) for s in props.values())
        sets = self.cart_sets_counts()
        for lid, n in lines:
            held = self.cart.get(lid, 0)
            if removal:
                if has_int:
                    held -= (n or 1)
                else:
                    held = held - n if n is not None else 0
            elif sets:
                held = n if n is not None else 1
            else:
                held += n if n is not None else 1
            if held > 0:
                self.cart[lid] = held
            else:
                self.cart.pop(lid, None)

    def kind_of(self, x):
        if self.get_cat_key():
            return str(self.label_of(x) or "").strip().lower()
        return x

    def complete_cart(self):
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or not self.role("cart") or not self.role("buy"):
            return ""
        self.completed = True
        N = len(self.items)
        planned, pkinds = {}, []
        for key, ids in self.lead_items.items():
            m = DIGIT_RUN.search(str(key))
            n = int(m.group()) if m else (1 if N == 1 else None)
            pick = next((x for x in ids if not self.sold_out(x)), None)
            if pick and n and 1 <= n <= N and self.kind_of(pick) not in pkinds:
                planned[pick] = self.items[n - 1].quantity
                pkinds.append(self.kind_of(pick))
        if self.confirmed not in planned:
            return ""
        keep, drop = {}, []
        for x in self.cart_ids():
            if x in planned:
                continue
            if not self.sold_out(x) and self.kind_of(x) not in pkinds:
                keep[x] = self.cart.get(x, 1)
            else:
                drop.append(x)
        if len(planned) + len(set(self.kind_of(x) for x in keep)) < N:
            return ""
        target = dict(keep)
        target.update(planned)
        calls = self.cart_calls(target, drop)
        if not calls or self.steps_left() < len(calls) + 1:
            return ""
        ids = list(target)
        self.fill_facts(ids, len(calls) + 3)
        self.show_links(ids, len(calls) + 3)
        if not self.planned_ok(ids, HARD_SPEND):
            return ""
        sent = []
        for tool, a in calls:
            f = []
            a = shape(a, self.schemas.get(tool), "args", f)
            if f:
                return ""
            r = self.env_call(tool, a)
            if r is None or self.finished:
                return ""
            self.note_cart(tool, a, r.get("observation"))
            self.observe(tool, a, r.get("observation"))
            sent.append(dump(a))
        buy = self.role("buy")
        f = []
        ba = shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", f)
        if f:
            return ""
        kept = []
        for attempt in range(2):
            if attempt == 1:
                self.reread_offers()
                if not self.planned_ok(ids, HARD_SPEND) or self.steps_left() <= 0 or self.finished\
                        or any(self.sold_out(x) for x in ids):
                    break
            stop, gt = self.gate("order")
            if gt:
                kept.append(gt)
            if stop:
                break
            r = self.env_call(buy, ba)
            if r is None:
                break
            self.checkouts += 1
            if not self.finished:
                self.refused_at = self.steps
            o = r.get("observation")
            self.observe(buy, ba, o)
            kept.append(self.display(o))
            if self.finished:
                break
        if not kept:
            return ""
        return ("[Cairn] The decline was not sent because Cairn acted in its place: the independent check passed %s, so "
                "Cairn changed the cart (%s) and placed the order. " % (self.confirmed, "; ".join(sent)) + " [then] ".join(kept))

    def is_order(self, name, args):
        buy = self.role("buy")
        fixed = self.brief.get("buy_args") if isinstance(self.brief.get("buy_args"), dict) else {}
        return bool(buy) and name == buy and all(args.get(k) == v for k, v in fixed.items())

    def is_decline(self, name, args):
        ref = self.role("refuse")
        fixed = self.brief.get("refuse_args") if isinstance(self.brief.get("refuse_args"), dict) else {}
        return bool(ref) and name == ref and all(args.get(k) == v for k, v in fixed.items())

    def offered_tools(self):
        ending = [self.role(r) for r in ("buy", "refuse") if self.role(r)]
        L, R = self.steps_left(), self.reserve()
        allowed = None
        if ending:
            if L <= R:
                allowed = ending
            elif L <= R + 3:
                allowed = ending + [self.role("cart"), self.role("cart_view")]
            elif L <= len(self.items) + 5 and self.checkouts == 0:
                allowed = ending + [self.role("cart"), self.role("cart_view"), self.role("offer")]
        out = [t for t in self.tools if allowed is None or t["function"]["name"] in allowed]
        if not out:
            out = list(self.tools)
        ask = self.role("ask")
        if ask and self.msg_args("x" * 200) is not None:
            out = [t for t in out if t["function"]["name"] != ask]
        return out

    def quiet_fill(self, tool, args):
        sch = self.schemas.get(tool) or {}
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        a = dict(args or {})
        qd = self.brief.get("quiet_decline")
        for k, s in props.items():
            if not isinstance(s, dict) or "string" not in kinds(s) or s.get("enum"):
                continue
            if k not in a and k not in req:
                continue
            v = a.get(k)
            txt = v if isinstance(v, str) else ""
            if not txt or any(x in txt for x in self.rows) or qd is not False:
                mx = s.get("maxLength")
                mx = mx if isinstance(mx, int) and not isinstance(mx, bool) else 200
                a[k] = STOCK_DECLINE[:mx]
        return a

    def resolve_ids(self, tool, args):
        if tool not in [self.role(r) for r in ("cart", "open", "offer", "compare") if self.role(r)] or not self.id_key:
            return args, ""
        sch = self.schemas.get(tool) or {}
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        free = [k for k, s in props.items() if isinstance(s, dict) and ("string" in kinds(s) or "array" in kinds(s)) and not s.get("enum")]
        if len(free) != 1:
            return args, ""
        fk = free[0]
        if tool != self.role("cart") and (tool, fk) not in self.id_args:
            return args, ""
        problem = []

        def m(v):
            if not isinstance(v, str):
                return v
            s = v.strip()
            if s in self.rows:
                return s
            if id_like(s, True) or len(s) < 8:
                return v
            low = s.lower()
            hits = [lid for lid, row in self.rows.items()
                    if any(isinstance(fv, str) and k != self.id_key and fv.lower().startswith(low) for k, fv in row.items())]
            if len(hits) == 1:
                return hits[0]
            if not problem:
                if hits:
                    problem.append("More than one listing holds the text %s (%s); give a single listing id"
                                   % (quote(s), ", ".join(hits[:6])))
                else:
                    problem.append("%s is no listing id; give the id as the result rows show it, in their %s field"
                                   % (quote(s), self.id_key))
            return v
        if fk not in args:
            return args, ""
        v = args[fk]
        nv = [m(e) for e in v] if isinstance(v, list) else m(v)
        if problem:
            return args, problem[0]
        a = dict(args)
        a[fk] = nv
        return a, ""

    def display(self, obs):
        lim = ROW_CLIP if self.listing_records(obs) else TEXT_CLIP
        t = dump(clip(obs, lim))
        if len(t) > OBS_CAP:
            t = t[:OBS_CAP] + " ...(cut)"
        return t

    def compare_target(self, tool):
        if not tool:
            return None
        sch = self.schemas.get(tool) or {}
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        t = self.compare_list_arg(tool)
        if not t or any(r != t for r in req):
            return None
        return t

    def compare_room(self):
        cmp = self.role("compare")
        tgt = self.compare_target(cmp) if cmp else None
        if tgt:
            p = ((self.schemas.get(cmp) or {}).get("properties") or {}).get(tgt) or {}
            mx = p.get("maxItems")
            if isinstance(mx, int) and not isinstance(mx, bool):
                return mx
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def join_opens(self, calls):
        op, cmp = self.role("open"), self.role("compare")
        t = self.single_target(op) if op else None
        ct = self.compare_target(cmp) if cmp else None
        if not (op and t and cmp and ct and op != cmp and self.compare_facts() is not False):
            return calls
        picks = []
        for idx, c in enumerate(calls):
            fn = c["function"]
            if fn["name"] != op or not isinstance(fn.get("arguments"), str):
                continue
            obj = json_obj(fn["arguments"])
            if not isinstance(obj, dict) or set(obj.keys()) != set([t]):
                continue
            v = obj[t]
            if isinstance(v, str) and v.strip() in self.rows and v.strip() not in self.gone:
                picks.append((idx, v.strip()))
        picks = picks[:self.compare_room()]
        if len(picks) < 2:
            return calls
        out = list(calls)
        first = picks[0][0]
        out[first] = {"id": calls[first]["id"], "type": "function",
                      "function": {"name": cmp, "arguments": dump({ct: [p[1] for p in picks]})}}
        for idx, _ in picks[1:]:
            out[idx] = JOINED
        return out


QUOTED_ID = re.compile(r"\"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})\"")
SKIP = "[Cairn] This call was not executed and spent no step."
HOLD = "[Cairn] This call was not executed yet and spent no step."
BARE = "[Cairn] This call was not executed yet."
JOINED = "__joined__"


class E9(object):
    def footer(self):
        return "\nCart now: " + self.cart_text()

    def run(self):
        if self.finished:
            return self.trace
        self.read_request()
        op = self.pv.get("opening")
        shown_op = None
        if op is not None:
            shown_op = restore(op)
            self.absorb(shown_op)
        report = ""
        if not self.finished:
            report = self.opening_moves()
        self.gap = STORE_PAUSE
        msg = "SHOPPER'S REQUEST:\n" + self.request
        if op is not None:
            msg += "\n\nTHE STORE'S STARTING DISPLAY:\n" + self.display(shown_op)
        if report:
            msg += ("\n\nCairn has already searched for every item and read part of each item's listings, one step per "
                    "read:\n" + report + "\nJudge these listings by the requirements, and search more only where "
                    "something is still lacking.\n\n" + self.status())
        else:
            msg += ("\n\nBegin with a search for each requested item, sought within the category it belongs to, and "
                    "read several listings completely, so that the interviewer can tell which properties such listings "
                    "show.\n\n" + self.status())
        self.convo = [{"role": "system", "content": GUIDE}, {"role": "user", "content": msg}]
        self.work_loop(TURN_CAP, HARD_SPEND)
        if not self.finished:
            txt = self.finish_by_code(True)
            if txt and not self.finished and not self.store_broke and not self.no_credit and self.steps_left() > 1:
                self.convo.append({"role": "user", "content": "[Cairn] Cairn attempted to finish, and " + txt
                                   + "\nFollow the store's answer, then finish.\n" + self.status()})
                self.work_loop(self.turns + 10, HARD_SPEND + 0.01)
        if not self.finished:
            self.finish_by_code(False)
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.steps,
                           "spend": round(self.spend, 5), "calls": self.model_calls})
        return self.trace

    def work_loop(self, cap, spend_cap):
        for _ in range(CRASH_RETRIES):
            try:
                self.turn_loop(cap, spend_cap)
                return
            except Exception as e:
                self.trace.append({"kind": "crash", "error": ("%s: %s" % (type(e).__name__, e))[:400]})
                self.answer_open_calls()

    def answer_open_calls(self):
        idx = None
        for i in range(len(self.convo) - 1, -1, -1):
            if self.convo[i].get("tool_calls"):
                idx = i
                break
        if idx is None:
            return
        done = set(m.get("tool_call_id") for m in self.convo[idx + 1:] if m.get("role") == "tool")
        for c in self.convo[idx]["tool_calls"]:
            if c.get("id") not in done:
                self.convo.append({"role": "tool", "tool_call_id": c.get("id"),
                                   "content": "[Cairn] This call possibly did not execute, because something failed "
                                              "along the way. Inspect the present state before repeating it."})
        self.convo.append({"role": "user", "content": self.status()})

    def turn_loop(self, cap, spend_cap):
        idle = 0
        while (not self.finished and not self.store_broke and self.steps_left() >= 1 and self.turns < cap
               and self.spend < spend_cap
               and (self.clock() < LAST_TURN or (not self.cart_ids() and self.room() > EMPTY_CART_ROOM))):
            if self.interview_ready():
                s = self.interview()
                if s:
                    self.convo.append({"role": "user", "content": s + "\n" + self.status()})
                continue
            self.trim()
            self.turns += 1
            msg = self.think(self.convo, [LIGHT, SPARE], 3000, self.offered_tools())
            if msg is None:
                self.misses += 1
                if self.no_credit or self.model_calls >= CALL_CAP:
                    break
                if self.misses > 2:
                    if self.clock() >= OUTAGE_CUTOFF:
                        break
                    time.sleep(OUTAGE_PAUSE)
                continue
            calls = []
            for i, tc in enumerate(msg.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                a = fn.get("arguments")
                if isinstance(a, dict):
                    args = a
                elif isinstance(a, str) and a.strip():
                    args = json_obj(a) or {}
                else:
                    args = {}
                calls.append({"id": tc.get("id") or "k%d_%d" % (self.turns, i), "type": "function",
                              "function": {"name": str(fn.get("name") or ""), "arguments": dump(args)}})
            text = msg.get("content")
            if not isinstance(text, str):
                text = dump(text) if text is not None else ""
            if not calls:
                idle += 1
                self.convo.append({"role": "assistant", "content": text or "(no call)"})
                self.convo.append({"role": "user", "content": "[Cairn] Use one of the tools so the work goes on.\n" + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            commit = [self.role(r) for r in ("cart", "buy", "refuse") if self.role(r)]
            if (not self.asked and self.role("ask") and self.free_msgs() > 0 and self.steps_left() > 8
                    and self.clock() <= ASK_DEADLINE and any(c["function"]["name"] in commit for c in calls)):
                self.convo.append({"role": "assistant", "content": text or None, "tool_calls": calls})
                for c in calls:
                    self.convo.append({"role": "tool", "tool_call_id": c["id"],
                                       "content": SKIP + " Cairn's interviewer questions the shopper first."})
                s = self.interview()
                self.convo.append({"role": "user", "content": (s or "[Cairn] The interviewer had no question to put "
                                   "to the shopper.") + "\n" + self.status()})
                continue
            self.convo.append({"role": "assistant", "content": text or None, "tool_calls": calls})
            self.trace.append({"kind": "turn", "text": text[:300], "calls": [c["function"] for c in calls]})
            replies = self.dispatch(calls)
            for i, c in enumerate(calls):
                r = replies[i] if i < len(replies) else SKIP
                if i == len(calls) - 1:
                    r = r + "\n" + self.status()
                self.convo.append({"role": "tool", "tool_call_id": c["id"], "content": r})

    def trim(self):
        total = sum(len(dump(m.get("content"))) for m in self.convo)
        if total <= HISTORY_CAP:
            return
        tms = [m for m in self.convo if m.get("role") == "tool"]
        n = max(1, len(tms) // 2)
        for m in tms[:n]:
            c = m.get("content")
            if isinstance(c, str) and len(c) > 600:
                ids = []
                for tok in QUOTED_ID.findall(c):
                    if tok in self.rows and tok not in ids:
                        ids.append(tok)
                m["content"] = c[:400] + "... (an older result was shortened; listing ids it held: " + ", ".join(ids[:20]) + ")"

    def search_like(self, name):
        plain_roles = [self.role(r) for r in ("open", "offer", "compare", "ask", "cart", "cart_view", "buy", "refuse")]
        return name not in [p for p in plain_roles if p]

    def dispatch(self, calls):
        joined = self.join_opens(calls)
        n = len(joined)
        replies = []
        for c in joined:
            replies.append(self.dispatch_one(c, n))
        return replies

    def dispatch_one(self, c, n):
        allowed = [t["function"]["name"] for t in self.offered_tools()]
        if c == JOINED:
            return ("[Cairn] This listing is in the comparison result given earlier in this turn: Cairn read them "
                    "jointly in a single step.")
        if self.finished:
            return "[Cairn] This call was not executed: the episode has ended."
        name = c["function"]["name"]
        if name == self.role("ask") and self.msg_args("x" * 200) is not None:
            return SKIP + " Cairn's interviewer sends all shopper messages."
        if name not in self.schemas:
            return SKIP + " Unknown tool. Allowed: " + ", ".join(sorted(allowed))
        if name not in allowed:
            return SKIP + " Too few steps remain for this tool; use one of these instead: " + ", ".join(sorted(allowed))
        a = c["function"].get("arguments")
        if isinstance(a, dict):
            args = a
        elif isinstance(a, str) and a.strip():
            args = json_obj(a)
            if args is None:
                return SKIP + " The argument text did not form exactly one JSON object."
        else:
            args = {}
        faults = []
        args = shape(args, self.schemas[name], "args", faults)
        if faults:
            return SKIP + " " + "; ".join(faults[:6])
        args, problem = self.resolve_ids(name, args)
        if problem:
            return SKIP + " " + problem + "."
        cart_tool = self.role("cart")
        if name == cart_tool:
            args = self.whole_removal(args)
        tail = []
        if name == self.role("compare"):
            tgt = self.compare_target(name)
            room = self.compare_room()
            if tgt and isinstance(args.get(tgt), list) and len(args[tgt]) > room and self.compare_bad:
                left = args[tgt][room:]
                args = dict(args)
                args[tgt] = args[tgt][:room]
                tail.append("[Cairn] Only the first %d listings went out, as this store's comparison takes no more than "
                            "%d; not sent yet: %s" % (room, room, ", ".join(str(x) for x in left)))
        st = self.single_target(name)
        if st and name in (self.role("open"), self.role("offer")) and isinstance(args.get(st), str)\
                and args[st].strip() in self.gone:
            return SKIP + (" This listing's sell-out was recorded at step %d and it will not return. Put a different "
                           "listing that fits in place of its cart line." % self.gone[args[st].strip()])
        if name == cart_tool and any(not isinstance(v, str) and isinstance(count_of(v), int) and count_of(v) < 0
                                     for v in args.values()):
            vt = dump(self.cart_view) if self.fresh_view() else ""
            for v in args.values():
                if isinstance(v, str) and v.strip() in self.rows:
                    lid = v.strip()
                    if lid not in self.cart and lid not in vt and lid not in self.cart_said:
                        return SKIP + " Listing %s is not held in the cart, so no units can come off it." % lid + self.footer()
        is_order = self.is_order(name, args)
        is_decline = self.is_decline(name, args)
        finishing = is_order or is_decline
        if finishing and n > 1:
            return SKIP + " A finishing call must go alone, once the results of the other calls have been read."
        if not finishing and self.steps_left() <= self.reserve() and (self.role("buy") or self.role("refuse")):
            extra = ""
            if self.reserve() > 1:
                extra = " (with the shopper approving the basket first)"
            return SKIP + " The last few steps are reserved for the order%s or for a decline." % extra
        fresh = False
        if self.search_like(name):
            fresh = self.fresh_search(name, args)
            if fresh:
                a2 = self.first_page(name, args)
                if a2 != args:
                    tail.append("[Cairn] This being a new search, Cairn sent it for page one; you had sent %s." % dump(args))
                    args = a2
        args = self.map_row_ids(name, args)
        if name == self.role("search") and self.codes_dead and self.names_code(name, args):
            return SKIP + (" Code searches in this store returned nothing: search with words and attribute values "
                           "instead, browse the category listing, and take the codes from the rows shown.")
        mark = name + dump(args)
        if not finishing and name != cart_tool and mark in self.recent:
            return SKIP + " This exact call was just made, with no change since; the earlier result stands."
        readers = [self.role(r) for r in ("search", "open", "offer", "compare") if self.role(r)]
        reader = name in readers
        if reader and mark in self.read_cache and self.read_cache[mark][0] == self.version:
            ver, stp, txt = self.read_cache[mark]
            return SKIP + " The identical call at step %d gave the result below, with no change since then.\n%s" % (stp, txt[:3000])
        if reader and name != self.role("compare") and st:
            lid = args.get(st)
            lid = lid.strip() if isinstance(lid, str) else lid
            if lid is not None and (self.shown_in_full(name, lid) or (name, lid) in self.factless):
                return SKIP + " This view has nothing new to show for that listing; its record: " + self.record_text(lid)
        if name == self.role("compare"):
            tgt = self.compare_target(name)
            lst = args.get(tgt) if tgt else None
            if isinstance(lst, list) and lst:
                keep = [x for x in lst if not self.shown_in_full(name, x)]
                if not keep:
                    body = "\n".join("%s: %s" % (x, self.record_text(x)) for x in lst)
                    return SKIP + " These views have nothing new to show for those listings; their records:\n" + body[:4000]
                if len(keep) < len(lst):
                    tail.append("[Cairn] Left out as already shown in full with no change since: "
                                + ", ".join(str(x) for x in lst if x not in keep))
                    args = dict(args)
                    args[tgt] = keep
                    mark = name + dump(args)
        hunt_s = None
        trail_text = ""
        changes = ""
        if is_order and self.steps_left() > 2:
            hunt_s = self.cheaper_hunt()
            w = self.wider_hold()
            if w:
                return w + self.footer()
            changes = self.reread_offers()
            if self.note_sold:
                self.sold_out_finder()
            cv = self.role("cart_view")
            if self.cart_changed_at >= 0 and not self.fresh_view() and cv and self.steps_left() > 5 and not self.rushed():
                f = []
                a0 = shape({}, self.schemas.get(cv), "args", f)
                if not f:
                    r = self.env_call(cv, a0)
                    if r is not None:
                        self.cart_view = r.get("observation")
                        self.cart_seen_at = self.steps
                        ch = self.observe(cv, a0, r.get("observation"))
                        if ch:
                            self.queued = []
                            return (HOLD + " Viewing the cart revealed a change: " + ch + self.footer()
                                    + "\nApply it, repair the cart where necessary, then order.")
            hint = self.swap_hint()
            gone_lines = []
            if self.sold_holds < 2:
                gone_lines = [x for x in self.cart_ids() if x in self.gone]
            if hint or gone_lines:
                if not hint:
                    self.sold_holds += 1
                    body = ("These carted lines are sold out: %s. Replace that carted line first." % ", ".join(gone_lines))
                else:
                    body = hint
                return HOLD + " " + body + self.footer()
            trail_text, hold = self.ref_trail()
            if hold:
                return HOLD + " " + trail_text + self.footer()
            if self.fit_holds < 2 and self.steps_left() > self.reserve() + 3:
                ff = self.fit_faults()
                if ff:
                    self.fit_holds += 1
                    self.trace.append({"kind": "fit_hold", "faults": ff[:4]})
                    return (HOLD + " A carted part is incompatible with the carted device: " + "; ".join(ff[:3])
                            + ". Replace the part, or else the device, and then order." + self.footer())
            if self.line_holds < 2 and self.steps_left() > self.reserve() + 2:
                lf = self.line_faults()
                fixes = []
                if lf and all(k == "count" for k, t in lf):
                    fixes = self.count_fixes(lf)
                    if fixes:
                        lf = self.line_faults()
                if fixes and not lf:
                    self.line_holds += 1
                    return (HOLD + " Cairn adjusted the unit counts itself: " + "; ".join(fixes)
                            + ". If the store rejected any of those fixes, correct those lines. Then order." + self.footer())
                if lf:
                    self.line_holds += 1
                    self.trace.append({"kind": "line_hold", "faults": [t for k, t in lf][:6]})
                    return (HOLD + " The cart and the request do not match yet: " + "; ".join(t for k, t in lf[:6])
                            + ". Correct those lines with the means the cart tool offers for adding and removing units, "
                              "then order." + self.footer())
            ids = self.cart_ids()
            keep = self.reserve() + 1 if len(ids) >= 2 else self.reserve() + 3
            fetched = "\n".join(self.fill_facts(ids, keep))
            self.show_links(ids, self.reserve() + 1)
            self.cheaper_rows()
            verdict = self.cart_check()
            if verdict:
                out = BARE + " "
                if trail_text:
                    out += trail_text + " "
                if hunt_s:
                    out += hunt_s + " " + self.below_cart_line() + "\n"
                if fetched:
                    out += ("Cairn fetched the attribute records of the listings below for this review, at one step "
                            "apiece:\n" + fetched + "\n")
                return out + verdict + self.footer()
            if changes and self.spend > SOFT_SPEND:
                return (HOLD + " Cairn has just fetched the cart's listings again: " + changes + " Order again once all "
                        "lines are still in stock and priced within budget; otherwise swap that line before ordering.")
        if is_decline and self.steps_left() > 5:
            t = self.decline_trail()
            if t:
                return HOLD + " " + t
            self.fit_hunt()
            t = self.audit()
            if t:
                return HOLD + " " + t
            if self.leads and self.steps_left() > 4 and self.decline_blocks == 0:
                self.decline_blocks = 1
                self.trace.append({"kind": "lead_hold", "leads": list(self.leads)})
                return (HOLD + " The review named listings that seem to fit: %s. Open them where necessary, cart whichever "
                        "fits best while obeying the pick rule, then order. Decline only if all of them plainly fail some "
                        "requirement, and in that case state in your own reply text (not a shopper message) which one "
                        "fails and send the decline once more." % ", ".join(self.leads))
            if self.leads and self.steps_left() > 4 and self.decline_blocks == 1:
                self.decline_blocks = 2
                first = self.leads[0]
                ok = self.planned_ok([first], HARD_SPEND)
                if ok and first in self.leads:
                    self.confirmed = first
                    self.trace.append({"kind": "confirmed", "id": first})
                    return (HOLD + " A second, separate check of the records of %s finds that it satisfies all "
                            "requirements: cart it in the quantity asked and order. Decline only when its record "
                            "plainly breaks a requirement." % first)
        if is_decline and self.steps_left() <= 5 and not self.late_held:
            built = [x for x in self.built if not self.sold_out(x)]
            ids = self.cart_ids()
            missing = [x for x in built if x not in ids]
            if not self.fresh_view():
                missing += [x for x in built if x in self.dropped_ids and x not in missing]
            if len(built) >= len(self.items) and len(missing) + 1 <= self.steps_left():
                self.late_held = True
                self.trace.append({"kind": "late_decline_hold", "missing": missing})
                out = (HOLD + " A decline scores zero whenever a fitting listing exists, and running low on steps is no "
                       "evidence about whether one exists. ")
                if missing:
                    out += "Cart these lines again in the quantities asked and order: %s. " % ", ".join(missing)
                else:
                    out += "Order the cart unchanged. "
                return out + ("Decline only if the searching proved that, for one of the requested items, no listing "
                              "satisfies its requirements, and in that case send the decline once more.")
        if is_decline and not self.leads and self.steps_left() > self.budget_steps // 3 and not self.pushed:
            self.pushed = True
            return (HOLD + " %d steps remain. Try untried kinds of search before giving up: list every requested item's "
                    "category and page onward through that listing using the page token or offset value the store "
                    "returns; try one-word queries of values and alternative names of the property; open any listing "
                    "that could fit. Decline only after all that." % self.steps_left())
        if is_decline:
            if self.confirmed:
                t = self.complete_cart()
                if t:
                    return t + (" [episode finished]" if self.finished else "")
            args = self.quiet_fill(name, args)
        if is_order:
            s = self.late_question()
            if s:
                return BARE + " " + s + self.footer()
        if finishing and self.steps_left() >= 2:
            stop, gt = self.gate("order" if is_order else "end")
            if stop:
                return BARE + " " + gt + self.footer()
            if gt:
                tail.insert(0, "[Cairn] " + gt)
        res = self.env_call(name, args)
        if res is None:
            return "[Cairn] The call broke down through a network or server problem; you may retry it."
        obs = res.get("observation")
        if name == self.role("compare"):
            tgt = self.compare_target(name)
            lst = args.get(tgt) if tgt else None
            if isinstance(lst, list) and len(lst) > 1:
                sj = dump(obs)
                if not any(str(x) in sj for x in lst):
                    self.compare_bad = min(self.compare_bad or 999, len(lst))
                    tail.append("[Cairn] The comparison came back empty for %d listings. Send no more than %d per call."
                                % (len(lst), self.compare_room()))
                else:
                    self.compare_ok = max(self.compare_ok, len(lst))
        self.recent = (self.recent + [mark])[-3:]
        for k, v in args.items():
            if (isinstance(v, str) and v.strip() in self.rows) or\
                    (isinstance(v, list) and v and any(isinstance(e, str) and e.strip() in self.rows for e in v)):
                self.id_args.add((name, k))
        self.absorb(obs)
        if name == self.role("compare"):
            recs = self.listing_records(obs)
            if recs:
                self.set_facts(name, self.facts.get(name) is True or any(attr_of(r) is not None for r in recs))
        if reader and name != self.role("search"):
            self.record_view_keys(name, obs)
        page_txt = ""
        if self.search_like(name) and not finishing:
            parts = [self.auto_pages(name, args, obs)]
            if fresh:
                parts.append(self.last_page_first(name, args, obs))
            parts.append(self.shorter_search(name, args, obs))
            page_txt = "\n".join(p for p in parts if p)
        nonview = [self.role(r) for r in ("cart", "buy", "refuse", "ask") if self.role(r)]
        if st and name not in nonview:
            lid = self.identity(obs) if isinstance(obs, dict) else None
            if lid:
                self.note_facts(name, obs)
                if name == self.role("open") and self.facts.get(name) is False:
                    excl = nonview + [self.role("search")]
                    for t in self.tools:
                        tn = t["function"]["name"]
                        if tn == name or tn in excl or not self.single_target(tn) or self.facts.get(tn) is False:
                            continue
                        if self.lists_things(tn):
                            continue
                        self.roles["open"] = tn
                        break
                if self.facts.get(name) is False and name != self.role("open") and not self.role("offer"):
                    self.roles["offer"] = name
                if name == (self.role("offer") or self.role("open")):
                    self.offer_seen[lid] = self.steps
        cnt = None
        if name == cart_tool:
            self.note_cart(name, args, obs)
            self.cart_said = obs if isinstance(obs, str) else dump(obs)
            self.recent = []
            counts = [count_of(v) for v in args.values() if not isinstance(v, str) and count_of(v) is not None]
            if counts:
                cnt = counts[0]
                for v in args.values():
                    if isinstance(v, str) and v.strip() in self.rows:
                        lid = v.strip()
                        if cnt <= 0 and not row_present(lid, obs):
                            self.dropped_ids.add(lid)
                        elif cnt > 0:
                            self.dropped_ids.discard(lid)
            live = [x for x in self.cart_ids() if not self.sold_out(x) and x not in self.dropped_ids]
            if len(live) >= len(self.built):
                self.built = live
        got_rows = bool(self.listing_records(obs))
        if name == self.role("search") and self.names_code(name, args):
            self.code_tries.append(got_rows)
            if len(self.code_tries) >= 2 and not any(self.code_tries):
                self.codes_dead = True
        if name == self.role("search") and got_rows:
            self.searched = (self.searched + [dict(args)])[-8:]
        if name == self.role("cart_view"):
            self.cart_view = obs
            self.cart_seen_at = self.steps
        if is_order:
            self.checkouts += 1
            if not self.finished:
                self.refused_at = self.steps
        text = self.display(obs)
        if page_txt:
            text += "\n" + page_txt
        if tail:
            text += "\n" + "\n".join(tail)
        if self.search_like(name) and not finishing:
            ch = self.cheapest_here(obs)
            if ch:
                text += "\n" + ch
            ci = self.cap_ignored(name, args, obs)
            if ci:
                text += "\n" + ci
        cacheable = text
        ev = self.evidence(obs)
        if ev:
            text += "\n" + ev
        if name == cart_tool and cnt == 0:
            for v in args.values():
                if isinstance(v, str) and row_present(v.strip(), obs):
                    text += ("\n[Cairn] A zero count did not remove the line: the tool adds the number given to the line, "
                             "and removing units takes a negative number.")
                    break
        self.note_sold = False
        lines = self.loose_lines(obs)
        if lines:
            meaning = self.read_note(lines)
            if meaning:
                text += "\n[note read] " + meaning + " Carry this change into the picks before ordering."
        refused = is_order and not self.finished
        spoken = self.spoken(obs)
        again = refused and any(l in spoken for l in self.sold_lines)
        loose = self.note_sold and (refused or (not self.note_named and not is_order))
        if (loose or again) and self.cart_ids() and not self.finished:
            t = self.sold_out_finder()
            if t:
                text += "\n[Cairn fetched the cart lines again following the rejection]\n" + t
        self.note_sold = False
        hint = self.swap_hint()
        if hint:
            text += "\n" + hint
        if name == cart_tool and self.fit_pairs and not self.finished:
            ff = self.fit_faults()
            if ff:
                text += "\n[Cairn] These lines are not yet compatible with each other: " + "; ".join(ff)
        if self.queued:
            text += "\n[store notes seen during Cairn's own calls] " + " | ".join(self.queued)
            self.queued = []
        if reader:
            self.read_cache[mark] = (self.version, self.steps, cacheable)
        if self.finished:
            text += " [episode finished]"
        return text

    def can_decline(self):
        ref = self.role("refuse")
        if not ref:
            return False
        f = []
        shape(self.quiet_fill(ref, dict(self.brief.get("refuse_args") or {})), self.schemas.get(ref), "args", f)
        return not f

    def finish_by_code(self, again=False):
        if self.finished or self.store_broke or self.steps_left() <= 0:
            return ""
        holding = bool(self.cart_ids())
        whole = True
        if holding and len(self.items) >= 2:
            covered = self.fill_short_cart()
            whole = covered or not self.can_decline()
        ids = self.cart_ids()
        if whole and len(ids) >= 2:
            self.fill_facts(ids, self.reserve())
            if self.steps_left() >= 3 and self.room() > 5:
                self.show_links(ids, self.reserve())
        if not whole:
            self.trace.append({"kind": "short_cart_decline"})
        if self.finished or self.steps_left() <= 0:
            return ""
        buy = self.role("buy")
        if whole and buy and (holding or (self.cart_changed_at >= 0 and not self.fresh_view() and self.steps_left() >= 2)):
            f = []
            a = shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", f)
            if not f:
                if self.steps_left() >= 2:
                    self.gate("order")
                if self.finished or self.steps_left() <= 0:
                    return ""
                res = self.env_call(buy, a)
                self.trace.append({"kind": "forced_order", "answered": res is not None})
                if self.finished or self.steps_left() <= 0:
                    return ""
                if again and res is not None:
                    self.checkouts += 1
                    self.refused_at = self.steps
                    self.observe(buy, a, res.get("observation"))
                    return "the order was rejected: " + self.display(res.get("observation"))
        ref = self.role("refuse")
        if ref:
            f = []
            a = shape(self.quiet_fill(ref, dict(self.brief.get("refuse_args") or {})), self.schemas.get(ref), "args", f)
            if not f:
                if self.steps_left() >= 2:
                    self.gate("end")
                if self.finished or self.steps_left() <= 0:
                    return ""
                res = self.env_call(ref, a)
                self.trace.append({"kind": "forced_decline", "answered": res is not None})
                if res is not None and not self.finished:
                    self.observe(ref, a, res.get("observation"))
                    return "the decline was rejected: " + self.display(res.get("observation"))
        return ""


def agent_main(problem_data):
    try:
        ep = Cairn(problem_data)
    except Exception as e:
        return [{"kind": "crash", "error": ("%s: %s" % (type(e).__name__, e))[:400]}]
    try:
        return list(ep.run())
    except Exception as e:
        ep.trace.append({"kind": "crash", "error": ("%s: %s" % (type(e).__name__, e))[:400]})
        if not ep.finished:
            try:
                ep.finish_by_code(False)
            except Exception as e2:
                ep.trace.append({"kind": "crash", "error": ("%s: %s" % (type(e2).__name__, e2))[:400]})
        return list(ep.trace)


class Cairn(E1, E2, E3, E4, E5, E6, E7, E8, E9):
    pass


__all__ = ["agent_main"]
