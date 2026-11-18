import json
import random
import re
import time
from typing import Any, Dict, List, Optional

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

TURN_CAP = 42
DIRECTOR_ROOM = 8000
CALL_CAP = 60
SOFT_SPEND = 0.055
HARD_SPEND = 0.078
HEAVY_ROOM = 0.02
BRIEF_DEADLINE = 120.0
OUTAGE_PAUSE = 10.0
OUTAGE_CUTOFF = 150.0

RETRY_WAITS = (4, 10, 20, 35)
WAIT_BUDGET = 45.0

EPISODE_LEN = 300.0
EPISODE_MARGIN = 20.0
CALL_FLOOR = 6.0
INTERVIEW_DEADLINE = 165.0
LATE_Q_DEADLINE = 190.0
WIND_DOWN = 200.0
HUNT_DEADLINE = 200.0
HUNT_GRACE = 25.0
QUESTION_DEADLINE = 210.0
LAST_TURN = 255.0
EMPTY_CART_ROOM = 12.0

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
HUNT_STEP_CAP = 24
HUNT_RESERVE = 10
CHEAPER_ROUNDS = 2
HUNT_IDLE = 4
HUNT_PAGE_YIELD = 5
HUNT_WHOLE = 49
HUNT_VALUE_LEN = 30
HUNT_COMMON = 0.6
FIT_HUNT_WORDS = 4
FIT_HUNT_OPENS = 4
FIT_HUNT_PAGES = 2
CHEAPER_SHOWN = 4
CHEAPER_OPENED = 3
VALUE_ROOM = 12
VALUE_RESERVE = 8
PRICE_LIMIT_CHANCE = 0.05
WALK_RESERVE = 10
WALK_PER_LINE = 4

TURNS_BEFORE_ASK = 2
SCARCE_MSGS = 2
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
EARLY_TRAIL_ROUNDS = 6
CHAIN_SEARCHES = 6
EARLY_OPENS = 12
LOW_EFFORT_CLOCK = 90.0
CLOSEST_CLOCK = 215.0
BIG_STEPS = 60
BIG_OPENS = 2
BIG_SPARE = 20
PINNED_SERIES = 2
SERIES_PAGES = 6
VS_SWEEPS = 2
VS_PAGES = 2
SWEEP_MIN = 7
SWEEP_MAX = 48
SWEEP_KEEP = 6
LAST_PAGE_ROOM = 6
END_READS = 2
END_ROOM = 14
DEAR_SHARE = 2.0 / 3.0
DEAR_ROOM = 12
SOLD_PEERS = 2
LABEL_SHARE = 0.4
HUNT_CHOICES = 3
CHEAPEST_SETS = 2
SERIES_HUNTED = 3
SERIES_ROWS = 3
SERIES_BREADTH = 12
REC_MIN_FIELDS = 3
NOTE_MIN_FIELDS = 5
FIELD_SHARE = 0.15
PAIR_SHARE = 0.7
BLOCK_NAME_LEN = 60
BLOCK_VALUE_LEN = 40
TAGGED_LIMIT = 20000

PAIR_MARKS = (": ", " = ", "=", " -> ", " - ")
FIELD_MARKS = (" | ", "\n", " || ", " ; ", "; ", "\t", " // ", " · ", " • ", "|", ";", ", ")

_MODEL_CLIENT = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_SHORT_CLIENTS: Dict[int, Any] = {}
_ENV_CLIENT = ProxyClient(timeout=95, max_retries=3, retry_delay=1)

RX_MSG_ARG = re.compile(r"(?:^|_)(?:text|texts|body|message|messages|note|notes|content|contents|question|questions|msg|say)(?:_|$)", re.I)
RX_NON_MSG_END = re.compile(r"(?:^|_)(?:id|ids|type|format|code|lang|language|kind)$", re.I)
RX_OPEN_EXCLUDE = re.compile(r"\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or|same|listing)\b", re.I)
RX_STOCK_WHOLE = re.compile(r"^(?:no|none|false|0|n)$", re.I)
RX_STOCK_PART = re.compile(r"out[ _-]?of[ _-]?stock|sold[ _-]?out|unavailable|not[ _-]?available", re.I)
RX_HOW_ASK = re.compile(r"\b(?:one|each|single|first|once|several|many|much|all|lot|too|list|separate\w*|split|short\w*|time|turn|question|questions|ask\w*)\b", re.I)
RX_CAP_ARG = re.compile(r"(?:^|_)(?:cap|max|maximum|ceiling|under|up_to|upto)(?:_|$)", re.I)
RX_NON_PRICE = re.compile(r"result|row|page|count|size|num|item|step", re.I)
RX_PRICE_NAME = re.compile(r"price|cost|budget|spend|money|amount", re.I)
RX_LOW_ARG = re.compile(r"(?:^|_)(?:min|minimum|low|lowest|least|from|start|offset|floor)(?:_|$)", re.I)
RX_SHELF_ARG = re.compile(r"section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_(?:name|label))?$", re.I)
RX_LIST_MODE = re.compile(r"field|filter|facet|brows|categor|list|structured", re.I)
RX_AMOUNT = re.compile(r"\d[\d.,]*\s*[A-Za-z\"']{1,12}")
RX_FLOOR_PHRASE = re.compile(r"\b(?:at least|no less than|no fewer than|min|minimum)\b\s+(\d[\d.,]*\s*[A-Za-z\"']{1,12})\s*$", re.I)
RX_QUERY_ARG = re.compile(r"(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword", re.I)
RX_NON_PROP = re.compile(r"\b(?:option|options|variant|variants|model|models|sku|id|ids|code|codes|asin|upc|ean|barcode|barcodes|warrant\w*|video|url|link|links|image|images|photo|photos|condition)\b")
RX_MODELS_KEY = re.compile(r"[a-z0-9]+[ _]models?\b")
RX_KIND_WORD = re.compile(r"\b(?:type|types|style|styles|kind|kinds|shape|shapes)\b")
RX_CATCHALL = re.compile(r"\b(?:feature|features|function|functions|highlight|highlights|spec|specs|specification|specifications|description|descriptions|extra|extras|includ\w*|accessor\w*|misc\w*)\b")
RX_FIT_CODE = re.compile(r"\b[A-Z][A-Za-z]{0,5}[ -]?(?:\d{3,6}|\d{1,5}[xX]{1,3})\b")
PART_CODE = r"[A-Za-z]{1,5}\d{0,3}-?\d{3,7}"
RX_FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_ ./&'()\-]{0,38}")
RX_BULLET = re.compile(r"^ *(?:[-*•·]|\d{1,3}[.)]) +")
RX_PAIR_KEY = re.compile(r"name|key|label|attr|prop|field|spec|title|^k$", re.I)
RX_ID_FIELD = re.compile(r"(?i:(?:^|[ _-])(?:id|idx|sku|pid|ref|asin|ident\w*)(?=$|[ _-]))|^[iI][dD][A-Z0-9]|Ident|(?i:(?<![a-z])(?!(?:paid|valid|grid|solid|rigid|fluid|liquid|hybrid|rapid|void|acid|humid|android|said|laid)$)[a-z]*id$)")
RX_FAIL_KEY = re.compile(r"error|fail|(?<!de)fault|trouble(?!shoot)|denied|refus|invalid", re.I)
RX_UNIT_VALUE = re.compile(r"^\d[\d.]*\s*[a-z]{2,4}$", re.I)
RX_MONEY_FIELD = re.compile(r"price|cost|fee|amount|charge", re.I)
RX_PLACEHOLDER = re.compile(r"^(?:none|null|n/a|unknown)$", re.I)

RX_SEASON_YEAR = re.compile(r"^[A-Z][a-z]* (?:19|20)\d\d$")
RX_GAP = re.compile(r"[ -]+")
RX_OWN_CODE = re.compile(r"[\[(]\s*(?:part\s*)?(" + PART_CODE + r")\s*[\])]", re.I)
RX_MODEL_MARK = re.compile(r"\bmodel\s*(?:no\.|no|number|#)?\s*[:.]?\s*(?!(?:no|number)\b)([A-Z][A-Za-z]{0,5}[ -]?(?:\d{3,6}|\d{1,5}[xX]{1,3}))\b", re.I)
RX_ALSO_SOLD = re.compile(r"(?:also sold as|also known as|sold also as)([^.;]*)", re.I)
RX_WORKS_WITH = re.compile(r"(?<!not )(?<!never )\b(?:works with|compatible with|fits)\s*:?\s+([^.;]*)", re.I)
RX_EXCEPT_CUT = re.compile(r"\b(?:except|excluding|but not)\b", re.I)
RX_PART_ALSO = re.compile(r"part\s+" + PART_CODE + r"\s+also\s*$", re.I)
RX_ALSO_FITS = re.compile(r"\bpart\s*(" + PART_CODE + r")\s*also fits:?([^.;]*)", re.I)
RX_POINTER = re.compile(r"\b(?:more|details|see|read)\s+(?:on|at|under|about)?\s*(?:the\s+)?part\s+(" + PART_CODE + r")", re.I)
RX_OWN_SERIES = re.compile(r"\b([A-Za-z]{1,5}\d{1,3})-\d{2,7}\b")
RX_SERIES_EXCL = re.compile(r"\b(?:except|excluding|but not|not for|not compatible with|does not fit|does not take)\b[^.;]*", re.I)
RX_NAMED_SERIES = re.compile(r"\b[A-Za-z]{1,5}\d{1,3}\b")


def find_codes(text):
    out = []
    for m in RX_FIT_CODE.finditer(text or ""):
        c = m.group(0)
        if RX_SEASON_YEAR.match(c):
            continue
        out.append(RX_GAP.sub(" ", c).strip().lower())
    return out


def code_covers(part_code, device_code):
    p = re.sub(r"[ -]", "", str(part_code or "")).lower()
    d = re.sub(r"[ -]", "", str(device_code or "")).lower()
    m = re.search(r"(?<=\d)x+$", p)
    if m:
        head = p[:m.start()]
        return d.startswith(head) and len(d) == len(p)
    return p == d


def fit_marks(text):
    text = text or ""
    own = ""
    m = RX_OWN_CODE.search(text)
    if m:
        own = m.group(1).upper()
    models = []
    for m in RX_MODEL_MARK.finditer(text):
        models.extend(find_codes(m.group(1)))
    for m in RX_ALSO_SOLD.finditer(text):
        models.extend(find_codes(m.group(1)))
    works = []
    barred = []
    for m in RX_WORKS_WITH.finditer(text):
        before = text[max(0, m.start() - 40):m.start()]
        if RX_PART_ALSO.search(before):
            continue
        seg = m.group(1)
        cut = RX_EXCEPT_CUT.search(seg)
        if cut:
            works.extend(find_codes(seg[:cut.start()]))
            barred.extend(find_codes(seg[cut.start():]))
        else:
            works.extend(find_codes(seg))
    notes = []
    for m in RX_ALSO_FITS.finditer(text):
        notes.append((m.group(1).upper(), find_codes(m.group(2))))
    pointer = ""
    m = RX_POINTER.search(text)
    if m:
        pointer = m.group(1).upper()
    own_series = sorted(set(x.upper() for x in RX_OWN_SERIES.findall(text)))
    rest = RX_OWN_SERIES.sub(" ", text)
    rest = RX_SERIES_EXCL.sub(" ", rest)
    named = sorted(set(x.upper() for x in RX_NAMED_SERIES.findall(rest)))
    return {"own": own, "models": models, "works": works, "barred": barred, "notes": notes,
            "pointer": pointer, "own_series": own_series, "named_series": named}


GUIDE = (
    "You are Cairn, a careful purchaser who serves one single shopper inside a web shop. Your sole means of acting is a "
    "call to one of the store's tools. Grading is all-or-nothing: a score of 1 requires a finish that is exactly correct, "
    "and anything else scores 0. A correct purchase MUST hold all requested items, every one in the quantity asked, every "
    "one satisfying each requirement still in force when checkout happens (requirements that changed during the work count "
    "as changed), and it MUST cost no more than the budget. The single alternative correct finish: when the shop truly "
    "holds nothing that satisfies everything, the episode MUST instead close through the ending (no-purchase) tool, "
    "performed in the manner the request specifies.\n\n"
    "HOW THE STORE OPERATES\n"
    "- Every call to a tool spends one step; the supply of steps is finite, and reasoning is free. Every store has its own "
    "tool names, parameter names and result layouts, so you MUST study the schema and the description of every tool and "
    "pass parameter names and permitted values exactly as given. Never guess a parameter name or a value.\n"
    "- A search can be restricted: it may filter by category; a keyword mode may read only the opening words of a query; "
    "pages may hold only a handful of rows; reaching the next page may need a token or an offset. Set the total the store "
    "reports against the number of rows you have seen; if the answer hinges on seeing every row, fetch further pages, "
    "passing back unchanged whatever token or offset came with the result. Keep keyword queries brief and lead with the "
    "value: the wanted value first, then the product noun. A store's search may match attribute values and seller names "
    "as well as titles, and may demand that every query word be present, so a handful of required values together with "
    "the noun can return almost the entire fitting set. A search by a part number or a model code can come back empty; "
    "a single attempt shows whether this store's search finds codes at all. When a filtered search returns zero rows, the "
    "usual cause is a wrong filter value or a wrong filter parameter: try again with the label copied letter for letter as "
    "the rows of a result show it, or switch to another filter parameter.\n"
    "- Rows in search results are summaries only. Shops divide a listing's record across views differently: one view may "
    "expose the attribute block (attributes and properties are the same thing here), another just the offer and a "
    "description, and either kind may appear in the side-by-side compare view. Once Cairn has learned which tool exposes "
    "attributes, the ledger says so. Each property requirement is judged on those attributes, so you MUST inspect the "
    "attributes of every listing you might purchase. A single call that displays several listings at once saves steps, "
    "provided it displays what is required. Never reopen a listing whose state has not changed since you read it.\n"
    "- Asking the shopper is the job of Cairn's interviewer: you never send the shopper a message yourself. The replies "
    "appear in this conversation, and the ledger records them.\n"
    "- A request may also demand intermediate steps: obtaining an approval, sending a message, verifying or reading "
    "something beforehand. Each appears in the ledger. Those that fall due right ahead of the order or of a no-purchase "
    "ending are carried out by Cairn itself, at the moment you call the checkout or the ending tool. So never hold off "
    "waiting for them, and never decline on their account. The remaining steps (those not due at the finish) you perform "
    "yourself with the shop's tools; questions about products are the exception and go through Cairn's interviewer.\n\n"
    "WHAT COUNTS AS FITTING\n"
    "- The kind of product is settled by the listing's category label: only a label that matches the requested item in "
    "its entirety qualifies; an adjacent label never does.\n"
    "- A property requirement is satisfied when the attribute the listing has for that property gives the value. How the "
    "key is written does not matter: spacing, underscore separators, prefixes or camel-cased joins. When the attribute is "
    "a list split by commas or slashes, a single matching element suffices. Matching works word by word; differences of "
    "letter case, of singular versus plural, and ordinary synonyms are tolerated. Containment counts: a value that "
    "includes all the words of the required value meets it despite extra words (\"deep navy blue\" satisfies \"navy "
    "blue\"; \"very relaxed fit\" satisfies \"relaxed\"). A value made of other words fails: an abbreviation or a code "
    "does not equal the word it abbreviates, and \"navy blue\" is not \"royal blue\". Numbers: convert the units first, "
    "then compare; a limit includes its boundary; a value given as a range must sit entirely within the limit.\n"
    "- An attribute keyed by the requirement's property (that exact name or an ordinary synonym of it) is decisive: if it "
    "states another value, the listing fails, whatever the title or any other key says. Exception: when a key prefixes "
    "the property name with the item's name (for a bag, a key meaning bag shape), its value counts even if the plain "
    "property key states otherwise. A key about another property, even a closely related one, settles nothing for this "
    "requirement: the colour of a handle tells nothing about the colour of the body. After listings are opened, Cairn "
    "appends [evidence] lines naming, for each requirement, the attribute keys whose name matches it.\n"
    "- When a listing lacks any attribute for the property, the value sought still counts when it appears under some "
    "other attribute (such as a styles list or a features list that contains it), or in its description or title. It "
    "does not count when the figure is a different kind of quantity (a weight is no substitute for a capacity), or when "
    "a bare number appears with no text saying what it measures.\n"
    "- When the requirement's value is written into the category label (a season or a target audience inside the "
    "label), the category satisfies it. Conversely, a listing filed under the label that was asked for counts as a "
    "product of the requested kind even when its description or title mentions something different; such a listing is "
    "judged solely on its requirements, its price and its stock.\n"
    "- A requirement may refer to another listing instead of giving its value, for example \"the same X as the item "
    "<shop> sells with <property>...\" or \"within N units of the value shown on that listing's page\". Resolve such a "
    "requirement first: locate the referenced listing (search on its words, brand, shop or category) and open its whole "
    "page with the single-item tool or the compare tool; a row from a search does not suffice. If the opened page sends "
    "you onward to one more listing that holds the property, locate and open that listing too by the same method, and so "
    "on: the final page in the chain supplies the value. If the requirement wants whatever goes with the referenced value "
    "(its partner or its counterpart), and that page says what its own value is paired with, then that paired value "
    "becomes the requirement. An order is valid only if every page along the chain was opened. If the requirement "
    "accepts anything other than the value of the referenced listing, it removes only the listings sharing exactly that "
    "value, and that rule by itself must never cause a decline.\n"
    "- If a listing you intend to buy now points to a further page with more detail about itself, check that page before "
    "the purchase.\n"
    "- If one requested item has to suit another, as an accessory suits a device, pick both jointly, based on the "
    "listings' own wording: a compatibility list, or a series or model code that appears on both pages. If no accessory "
    "meeting its own requirements is compatible with the chosen device, change to a different device that satisfies its "
    "own requirements and that one of those accessories does fit. Never decline while that device switch has not been "
    "tried.\n"
    "- Exclusions MUST be honoured, and so must closed lists. Wishes (softly worded preferences such as \"ideally\" or "
    "\"nice to have\") serve only to break ties.\n"
    "- Sold-out listings must never be bought. The total (each line's price times its quantity, added across all lines) "
    "must not go over the budget; landing exactly on it is allowed.\n\n"
    "CHOOSING\n"
    "- If the request wants the cheapest choice (the smallest price or the lowest total), the order MUST contain the "
    "listing with the lowest price out of ALL fitting listings. Establish the complete fitting set by paging through it "
    "(rows often arrive unsorted by price; Cairn fetches further pages of short result lists itself), examine the "
    "candidates from the cheapest upward, and pick the earliest one that fits fully. If a search tool accepts a maximum "
    "price, use it: list that category with the cap set below what its present best pick costs, so that only the "
    "cheaper ones remain to be checked.\n"
    "- If the request wants the greatest or the smallest amount of a measure, compare every fitting listing on that "
    "measure and take the extreme one; a listing silent on the measure is never the winner.\n"
    "- Otherwise any listing satisfying everything is correct; do not burn steps looking for a better one.\n"
    "- Put each item in the cart the moment an opened listing satisfies all of that item's requirements (for a "
    "price-ranked or measure-ranked pick: only after the fitting set has been compared), then go on to the next item. A "
    "store frequently sells out a line shortly after it enters the cart, and carting early keeps steps in hand for a "
    "replacement.\n"
    "- When several distinct items are requested, each needs its own cart line, every line at that item's requested "
    "quantity.\n\n"
    "CHANGES DURING THE WORK\n"
    "- Both the shopper and the store may alter the task: a fresh budget, an added or a removed requirement, a new price, "
    "a listing gone out of stock. Such changes show up within tool results, often on a checkout attempt as a refusal. Any "
    "note is valid whichever tool's result carries it. The ledger shows the requirements currently in force. A refusal "
    "does not end the episode: apply all changes, correct the cart and try checkout again.\n"
    "- Prices and stock may also shift without any note. When you call checkout, Cairn re-checks the cart lines and views "
    "the cart itself, so do not spend a turn on either.\n\n"
    "FINISHING\n"
    "- Make the cart contain only the chosen lines, exactly those, removing everything else; then send checkout as a call "
    "by itself, never together with other calls in one turn.\n"
    "- A no-purchase ending is allowed only once a genuine search has found that no listing satisfies everything. Being "
    "low on steps proves no such thing: in that situation, check out with the best cart you hold. The ending follows the "
    "request's own wording; when a remark has to accompany it, keep the remark brief and name no listing.\n"
    "- Each reply MUST include one or more tool calls. Keep your text brief: the known facts, then the next action."
)


BRIEF_PROMPT = (
    "You set up a buying job that an agent will carry out. You get the shopper's text and the list of the store's tools. "
    "Reply with exactly ONE JSON object, compact and unindented, with no text around it, holding these keys:\n"
    "items: a list with one entry for each distinct product to be bought. Each entry is an object with four fields: "
    "name (what sort of product it is, worded as the request words it); shelf (if some tool exposes the store's category "
    "labels, the matching label copied exactly; else a category the request gives the product, e.g. a product wanted "
    "\"in\" a named department takes that department's name; else empty text); quantity (the count of units to buy of "
    "that single listing, an integer, 1 if no number is stated; pairs and sets that come as one listing are quantity 1); "
    "musts (a list of the product's firm requirements, each written as property: constraint, stating the precise value, "
    "bound, excluded value or permitted set of values).\n"
    "Name the property inside a must with the very word the request used for it: \"for a formal occasion\" gives "
    "\"occasion: formal\"; \"a 12-inch blade\" gives \"blade length: 12 inch\". Never trade the property for a "
    "neighbouring one.\n"
    "A requirement whose value has to come from a different listing is written as ONE must that carries the other "
    "listing's full description, e.g. \"color: different from the color of the lamp that ShopX sells with a brass "
    "base\". All the words describing the other listing (its shop, its product kind, and any trailing \"with\" clause of "
    "properties and values) stay inside that single must; none of them ever becomes a must of this product, and "
    "whatever words the request uses to relate the wanted value to the referenced value are copied as they stand: never "
    "reduce them to \"same as\".\n"
    "Price, budget, stock and quantity never go into musts. Narrative around the purchase (the intended user, the place "
    "the product goes, earlier trouble, the occasion that prompted the purchase) is context and never a must, except "
    "where it explicitly gives some property's value or bound. Words forming part of the product's own name, such as a "
    "season or a target group mentioned alongside the product, stay in name and never become musts.\n"
    "Exception that does create a must: when the request treats a purpose or event as a property of the product, it IS "
    "a requirement on that very property (e.g. \"occasion: business\"), unless the store's category label already names "
    "it. This covers: something said to be for a kind of wear (\"for business wear\"); something said to be made for a "
    "kind of event; a product followed by a separate clause stating it is for an event; a plain statement that the "
    "occasion must equal a value. In contrast these stay context: one specific person's own event (wearing it at my own "
    "wedding), the recipient or place the purchase is meant for (a present for a co-worker), and its intended sit, feel "
    "or appearance when used.\n"
    "When one phrase names products linked by \"and\" (a single \"I want\" clause with a single quantity, often the "
    "title of a single department of the store), that is ONE item. Split into separate items only if every product gets "
    "a separate quantity or separate requirements.\n"
    "budget: the shopper's ceiling for the whole spend, as a number in whatever unit the request uses; null if absent.\n"
    "budget_unit: the unit, written out (e.g. cents); null if absent.\n"
    "budget_scope: \"total\" if the cap applies to the full order, \"each\" if it applies per item, null if there is no "
    "cap.\n"
    "in_stock_only: true by default; false only when the request plainly allows unavailable listings.\n"
    "pick: \"cheapest\" if the request wants the minimum price or the least total cost; \"most\" or \"least\" if it wants "
    "the maximum or minimum of some measure; \"any\" otherwise. Saying money is short does not make a pick rule.\n"
    "pick_measure: for most or least, the quantity pushed to its extreme; else empty text.\n"
    "messages: the number of shopper messages the request permits the agent, an integer; null if unstated.\n"
    "steps: the request's allowance of steps (it may call them moves or tool calls), an integer; else null.\n"
    "lookups: exact copies of those musts whose value must first be fetched from a listing the request refers to; empty "
    "list if none.\n"
    "rules: every instruction in the request on working method or on the way to finish, each at most 12 words.\n"
    "wishes: nice-to-haves the shopper voices without requiring them.\n"
    "ask_first: whatever the request explicitly wants confirmed with the shopper before choosing; empty list if none.\n"
    "process: duties the request imposes on top of locating and buying the right items (getting the shopper's approval, "
    "telling or showing the shopper something, confirming a point with the shopper, reading something before acting, "
    "using a specific tool). Each entry is an object: step (the duty, at most 15 words), when (exactly one of \"choose\" "
    "= before any product is chosen, \"order\" = just before checking out, \"end\" = just before finishing with no "
    "purchase), messages (the number of shopper messages the duty consumes, 0 if none). Empty list if none. Item "
    "requirements, budget, stock, the message and step limits, and what to do if nothing fits are NOT duties.\n"
    "quiet_decline: true if the request forbids suggesting or naming alternatives when nothing fits; false if it wants "
    "alternatives; null if silent.\n"
    "fits: one entry for every requested item that, by the request, has to work with, fit or suit some OTHER requested "
    "item (that is, whichever listing the agent picks for that other item): an object {\"item\": number of the dependent "
    "item, \"with\": number of the other item}. Empty list if none.\n"
    "roles: an object mapping each of these jobs to the exact tool name, or null where no tool does it: search; open "
    "(shows the properties or attributes of one listing; if two tools each show one listing, pick the one whose "
    "description offers specifications, properties or attributes); offer (shows one listing's live offer such as price "
    "and stock, only if a tool other than the open one does that); compare (shows several listings in full); ask "
    "(messages the shopper); cart (changes the cart); cart_view (shows the cart); buy (performs the checkout); refuse "
    "(finishes with no purchase).\n"
    "search_words: an object {\"tool\": the tool that takes free-text words to search listings (if one tool browses a "
    "category while another accepts words, choose the word-taking one), \"field\": the parameter of that tool that "
    "receives the words, \"args\": constant arguments needed to switch it to word search (a mode tool needs its mode "
    "value here), else {}}.\n"
    "buy_args: constant arguments the buy tool needs to perform the checkout (a mode tool needs its mode value here); "
    "else {}.\n"
    "refuse_args: constant arguments the refuse tool needs to finish with no purchase; else {}."
)

PLAN_PROMPT = (
    "You pick which questions an agent will put to the person it shops for. Each shopper keeps a small number of firm, "
    "unwritten preferences, and every such preference is about a single property of a single requested product. Asking "
    "reveals a preference only if the question names precisely that property. Where the shopper holds no opinion the "
    "answer is \"don't mind\"; where the property never crossed the shopper's mind the answer is puzzled. So each "
    "property put in a message spends a share of that message.\n"
    "You receive the request text; the items, each with the requirements the request already fixes; and a numbered "
    "table of the attribute keys found on opened listings in every item's category, giving per key the number of those "
    "listings that carry it and sample values. Caution: the unwritten preference can concern a property none of the "
    "listed keys covers.\n"
    "Choose in this order:\n"
    "0. Anything the request wants checked with the shopper (under ASK FIRST, if present), each named after the property "
    "involved.\n"
    "1. Hinted properties. A request's story often hints at an unwritten preference while leaving its value unsaid: a "
    "sentence about a matter the shopper cannot judge or leaves to the agent; about a person whose taste is decisive; "
    "about how the product looks or the surroundings it belongs to; about the intended wearer or user; about its "
    "contents. For each such sentence give the single property it hints at, with its item, in the story's own order. "
    "Whenever a key listed for the same item matches the hint, use that key: a hint about the wearer maps to the gender "
    "or audience key; a hint about the product's looks maps to whichever colour, finish or pattern key is carried by the "
    "largest share of that item's listings. A hinted property absent from every listing of this item ranks below that "
    "item's three most widely carried keys, so never above fourth.\n"
    "2. Then the listed keys in rank order. Prefer the properties shoppers for that sort of product decide on earliest "
    "(commonly the type or style, the form or fit, the dimensions, size or capacity, the material, the pattern or "
    "colour: whichever ones that sort of product is chosen by), and keys present on a majority of that item's listings. "
    "Once at least five listings have been read, steer clear of keys found on only one or two listings.\n"
    "Skip, per item: any property the request has settled for that item, i.e. a requirement already names it (a key "
    "whose values merely echo a required value is NOT settled and stays askable, e.g. a pattern key whose values show "
    "the decoration the request demands); a further name for an already chosen property; a key in which each value "
    "bundles several purchase options together; identifiers; model names; warranty and any other logistics. Do not "
    "skip pack sizes, per-pack quantities and counts: they are ordinary properties. A key marked [not the stated ...] is "
    "a different property from the requirement it mentions, however many words the two names share.\n"
    "Rank every item separately: its hinted properties before its listed keys.\n"
    "label holds the plain name of the property. From a listed key take every word of the key in plain spelling: words "
    "run together with a capital at each join, or joined by underscores, become separate lower-case words, and never cut "
    "the key down to its last word (the label for a lid's colour is \"lid colour\", never just \"colour\"; for a rim's "
    "finish it is \"rim finish\", never only \"finish\"). Just the name: no sample values, no bracketed text, never a "
    "second property.\n"
    "Pick at most COUNT properties per item, strongest first. Answer with ONE JSON object only: {\"ask\": [{\"item\": "
    "item number, \"label\": text, \"hint\": at most eight words from the story that led to the pick, or \"\"}, ...]}, "
    "the picks of each item in rank order."
)

REPLY_PROMPT = (
    "An agent has put some questions to a shopper. Read every answer and reply with ONE JSON object only, nothing "
    "besides it, with five keys:\n"
    "needs: a list of {\"item\": item number, \"need\": \"property: constraint\", \"points\": true if the value has "
    "to be taken from a different listing, else false};\n"
    "drop: a list of {\"item\": item number, \"property\": a requirement the shopper now gives up};\n"
    "free: a list of {\"item\": item number, \"property\": a property the shopper does not mind about};\n"
    "crowded: a list of question numbers whose answer objects that the message covered too many subjects, or wants the "
    "subjects asked one at a time (not answers that merely say the question was not understood or ask for plainer "
    "wording);\n"
    "budget: a revised overall ceiling as a number, else null.\n"
    "needs covers each firm preference any answer expresses, with its precise value or bound and its unit (e.g. "
    "\"length: at least 30 cm\", \"shape: round\", \"material: not plastic\"). Name the need after the property the "
    "answer itself mentions, even when that differs from the property the question was about; several answers on one "
    "property give one need. Copy the value word for word from the answer, nouns and units included (\"at least 4 "
    "shelves\", not \"at least 4\"). Restrict the value to what a listing might state: start at the value wanted (or at "
    "the bound word before it), end where the value ends, and leave out the answer's framing (\"matching walnut\" gives "
    "\"walnut\"; \"it's settled, oak\" gives \"oak\"; \"brass, nothing else will do\" gives \"brass\"; \"equal to at "
    "most 2 kg\" gives \"at most 2 kg\").\n"
    "An answer may give the value by reference to some other listing: a value like, unlike or close to the value on a "
    "product from some shop, identified by a handful of its properties. That makes ONE single need for the property the "
    "answer names, with points true. Its value is the exact text of the answer from the comparison word through to "
    "where that description ends, together with anything said about a listing that page might name. Never shorten or "
    "reword it. Properties describing the other product are never needs of this item.\n"
    "free: per question, the property the shopper declared indifference to.\n"
    "Folksy or indirect phrasing that still names a value does state that value. Phrasing that refuses to negotiate a "
    "value (it is not to be traded away, given up or changed) states the value as wanted; write a need as \"not\" a "
    "value only if the answer turns that very value down."
)


NOTE_PROMPT = (
    "While an agent was working, the store displayed the notice below; its author may be the store itself or the "
    "shopper. Reply with one JSON object, nothing besides it, with these keys: \"kind\": \"change\" if the notice alters "
    "the job at hand (a budget, requirements, some price, availability, or refusing the order), \"error\" if it merely "
    "says some call was badly formed or turned down, \"other\" in all remaining cases; \"budget\": the new limit on the "
    "total spend as a number in the unit the request used, or null; \"add\": a list of new requirements, each "
    "{\"item\": item number, \"need\": \"property: constraint\"}; \"drop\": a list of requirements no longer wanted, each "
    "{\"item\": item number, \"need\": the wording of the withdrawn requirement}; \"gone\": a list of ids of listings the "
    "notice calls unavailable; \"sold_out\": true if the notice says a picked listing has vanished, sold out or is no "
    "longer obtainable, even when no id is named; \"price_moved\": true if it reports some price moving; \"summary\": "
    "one brief sentence. A requirement swapped for another must appear twice: as a drop of the old wording and as an add "
    "of the new. Report every change the notice contains."
)

CHECK_PROMPT = (
    "You are the final gate ahead of checkout. From the records below, decide whether this cart is precisely right. "
    "The requirement list in force (under its heading) already includes every later change made by the shopper and "
    "overrides what the request text says; a requirement carrying the marker DROPPED is void, and stays void even where "
    "the request text goes on stating it. "
    "Reading rules, for a requirement R on property P with value V: R holds when the listing's field for P states V; "
    "the key may be spelled any way, a single list member suffices, letter case, singular or plural and simple synonyms "
    "are fine, a value that includes V's words counts (\"dark forest green\" satisfies \"forest green\"), units are "
    "converted, a limit includes its own boundary, and a range has to fall wholly inside. A field whose key names P "
    "itself decides: if it states something other than V, the listing fails; except that a value filed under a key that "
    "prefixes P with the item's own kind (on a backpack, \"backpack color\") still counts even though the bare key for P "
    "disagrees. A key for another property, however related, decides nothing (a strap's colour says nothing about the "
    "main body's colour). Where no field for P exists at all, V stated under some other field, or in the title or "
    "description, is enough; but not a figure of a different sort (a weight is no capacity), and not a lone figure "
    "whose surrounding text never says what it measures. When R is a bound expressed with a unit (a measured limit), the "
    "only attribute that can settle it is one whose name has some word in common with P: if the figure sits under an "
    "attribute for some other dimension, or solely in the title, R stays unknown, and so the listing is to be judged as "
    "failing R. An attribute whose name is P in plural form and whose value is a list enumerates the variants the seller "
    "offers; such a list does not give this listing's value whenever the title names some other member of it. If R is "
    "stated by the category label itself, being in that "
    "category satisfies R. The category label has to match the whole requested item, not just part of it. A requirement "
    "to differ from another listing's value is taken as met, except where the records reveal that very value.\n"
    "Cart rules: every requested item must have its own cart line carrying the quantity asked; the cart must hold "
    "nothing beyond those lines; no line may be shown as unavailable or out of stock; and each line's price multiplied "
    "by its quantity, added up, must give a total no higher than the budget. When there are several records of one "
    "listing, the newest governs.\n"
    "If the pick rule is cheapest, most or least, also weigh the other listings provided: should one of them plainly "
    "satisfy all requirements while beating the cart's choice under that rule, this cart is wrong.\n"
    "Reply with one JSON object, nothing besides it: {\"ok\": true or false, \"problems\": [short, concrete faults], "
    "\"better\": [ids of listings to put in place of some cart line, strongest first]}. Set ok to false only for a fault "
    "the records plainly show, or when some requirement is addressed by no record."
)

READ_RULES = (
    "for a requirement R on property P with value V: a field of the listing settles P whenever its key names P (the key "
    "may be spelled any way, one list member suffices, synonyms are fine, units may be converted, a value that includes "
    "V's words is fine, as \"dark forest green\" for \"forest green\", and a limit includes its own boundary); a value "
    "filed under a key that prefixes P with the item's own kind (on a backpack, \"backpack color\") still counts even "
    "though the bare key for P disagrees; a key for another property, however related, decides nothing (a strap's colour "
    "says nothing about the main body's colour); with no field naming P, V stated under some other field, or in the "
    "title or description, is enough, but a lone figure that never says what it measures is worth nothing; when R is a "
    "bound expressed with a unit, only an attribute whose name has some word in common with P settles it, and a figure "
    "under an attribute for some other dimension, or solely in the title, leaves R unknown, so the listing fails R; an "
    "attribute whose name is P in plural form and whose value is a list enumerates the variants the seller offers and "
    "does not give this listing's value whenever the title names some other member of it; if the "
    "category label states R, that category meets R; the category label has to match the item; a requirement to differ "
    "from another listing's value is taken as met, except where the records reveal that very value; a listing that is "
    "out of stock, or that goes over the budget, is not a fit; the binding requirement list overrides the request text, and "
    "any requirement flagged DROPPED is void."
)

AUDIT_PROMPT = (
    "An agent means to stop without buying, believing that nothing in the store fits. Check the records shown below "
    "against the requirements, using these reading rules: " + READ_RULES + "\n"
    "Records marked ROW come only from a search row: nobody has read their page, so on any property the row itself does "
    "not show, such a record may be judged neither a pass nor a fail.\n"
    "Reply with one JSON object only: {\"fits\": {item number: [ids that seem to satisfy all requirements, lowest "
    "price first, 4 at most]}, \"open\": [ids of ROW listings of a requested kind, within budget, whose page might "
    "prove a fit, likeliest first, 6 at most], \"reason\": one sentence}."
)

UNMIX_PROMPT = (
    "Someone turned a shopping request into firm requirements for one product. Some requirements take their value from "
    "a different listing that the request refers to, and the words that pick out that listing (the kind it is, the shop "
    "it comes from, the properties a \"with\" phrase after it names) are sometimes mistaken for requirements on the "
    "product being bought. Going by the request's own phrasing, say which entries of the list headed OTHER REQUIREMENTS "
    "below do nothing but describe the referenced listing. Reply with one JSON object only: {\"describe_other\": [such "
    "requirements, each copied word for word]}, an empty list when all are genuine requirements for the purchase."
)

CHAIN_PROMPT = (
    "An agent is near checkout. Some requirements draw their value off a second listing instead of stating it, and one "
    "listing's page may refer the reader onward to yet another listing for that same property. An order only counts "
    "once the agent has fully opened every listing along every such chain. Trace the chain of every requirement, "
    "starting at its own description, link after link, through what the pages themselves say, using the records placed "
    "after this text. OPENED means that page was opened by the agent; ROW means the page is unread and the listing is "
    "known only from a search row or through its fields.\n"
    "Reply with one JSON object only: {\"chains\": [{\"need\": the requirement copied word for word, \"links\": [{\"id\": "
    "the identifier of the record that is this link (the record's own identifier, not any code its title happens to "
    "print), else \"\", \"about\": how this listing is described, by the requirement or by the preceding page, \"values\": "
    "the values that description attributes to this listing, in a handful of words, with no property names and without "
    "the shop's name, \"find\": only where the link is not yet located, search words formed from the shop name plus the "
    "described values, leaving out property names and the product kind}], \"done\": true if the final link is OPENED, "
    "its page itself states the property's value, and its own wording does not point the reader onward to yet another "
    "listing about that property (a page that hands the value over to some other listing does not qualify as the final "
    "link, even if its attribute block shows a value); true as well when the requirement wants some value that differs "
    "from that one or is unlike it, because the value on that page is the yardstick; otherwise false, \"value\": the "
    "value that final page gives for the property; but when the requirement itself, or the original request phrasing "
    "behind it, wants a separate value that partners the page's value (a partner or counterpart value rather than the "
    "page's own) and the page also says what its own value is paired with, that partner value; phrasing that wants "
    "whatever value the pointed page states, or wants to match it, means that very value, even if a verb about going "
    "together is used; else \"\"}]}, one entry per requirement.\n"
    "A record counts as a link only if it fits the link's entire description: shop or brand, product kind AND each "
    "property value the description lists (a description of a 6-pack of cans excludes a listing from that shop with 6 "
    "bottles and no pack). If no record fits all of it, the link is unlocated, even when that shop carries further "
    "listings of the same kind. For an unlocated link, find is a short query: the shop name plus the described values, "
    "with property names and the product kind omitted (describing one shop's kettle by a lid material of glass and the "
    "colour red gives three words: the shop's name, glass, red): the store's word search requires every query word to "
    "appear in a listing, while listings often lack the property's name or the word for the product kind. Give the "
    "links of every chain in sequence and end the list at the earliest link that is unlocated or lacks an OPENED page, "
    "because what that page says is not yet known."
)

GATE_PROMPT = (
    "The agent will shortly STAGE. The request wants the steps listed below performed on the way. Using the RECORD (the "
    "numbered lines recording the agent's actions), decide for each step owed ahead of this STAGE whether it has been "
    "completed for the cart in its current state. A step is complete only if some RECORD line records its performance; "
    "for a step concerning the cart or the order, that same line must also come later than the latest cart change, and "
    "the reply recorded on that same line must grant it. Questions about a product's properties never perform a step. "
    "For a step still missing, name the single store call that performs it now, picking a tool from the TOOLS list and "
    "supplying its arguments. Shopper messages use the messaging tool and consist of one brief sentence or question on "
    "that step alone, plainly worded, mentioning no product property or value beyond what the step needs. Give no call "
    "if all owed steps are complete, or if the remaining messages or steps cannot achieve the missing one.\n"
    "Reply with one JSON object only: {\"steps\": [{\"step\": the step, \"done\": true or false, \"lines\": [numbers of "
    "the RECORD lines that prove it]}], \"call\": {\"tool\": name, \"args\": {...}} or null, \"hold\": a change that some "
    "reply asks for before the agent goes on, else \"\", \"needs\": [{\"item\": item number, \"need\": \"property: "
    "constraint\"}, one per firm requirement such a reply states]}."
)

SWAP_PROMPT = (
    "A listing in the agent's cart sold out a moment ago. Pick substitutes for it out of the remaining records shown "
    "below, using these reading rules: " + READ_RULES + "\n"
    "Reply with one JSON object only: {\"fits\": [{\"id\": a listing id, \"price\": its price as a number, the way the "
    "record states it}], covering every listing that seems to satisfy all requirements of the lost line's item, 6 at "
    "most, \"reason\": one sentence}."
)

SHAPE_PROMPT = (
    "A store tool produced the output shown below. Label its parts using the exact keys found in that result and in the "
    "tool's parameters. Reply with one JSON object only: {\"rows\": the key where the array of listings sits, \"count\": "
    "the key where the total number of matches sits (not the number displayed) or null, \"next\": the key where the "
    "value that requests the following page sits or null, \"next_arg\": the tool parameter that accepts that value or "
    "null, \"price\": within a row, the key of the listing's price or null, \"stock\": within a row, the key telling "
    "whether it can be bought right now or null}."
)


RX_TRUE_FALSE = re.compile(r"^(?:true|false)$", re.I)
RX_EMPTY_WORD = re.compile(r"^(?:none|null|)$", re.I)
RX_MONEY = re.compile(r"^[$€£]?\s?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)$")
RX_LEAD_ZERO = re.compile(r"^0\d")
RX_LOOSE_PAIR = re.compile(r"['\"]?([A-Za-z][A-Za-z0-9_ ./\-]{0,38})['\"]?\s*[:=]\s*['\"]?([^,'\"}\]]*)")
RX_BLOCK_CUT = re.compile(r";\s|;$|\s\|\s|\n")
RX_BLOCK_PART = re.compile(r"[^=:]{1,40}=[^=]{0,200}")
RX_RULE_LINE = re.compile(r"^[\s|:+=\-]*$")
RX_NUM_HEAD = re.compile(r"^[\d.,$]+$")
RX_BLANK_CUT = re.compile(r"\n\s*\n")
RX_TAG = re.compile(r"<[A-Za-z/][^<>]{0,40}>")
RX_INLINE_TAG = re.compile(r"</?(?:b|i|em|strong|span|a|u|small)\b[^<>]{0,40}>", re.I)
RX_CELL_TAG = re.compile(r"</t[dh]>\s*<t[dh][^<>]{0,40}>", re.I)
RX_TAG_RUN = re.compile(r"\s*(?:<[^<>]{1,40}>\s*)+")
RX_COL_GAP = re.compile(r" {2,}|\t")
RX_COMMA_CUT = re.compile(r",(?!\d)|(?<!\d),")
RX_FENCE = re.compile(r"^```[A-Za-z]*\s*(.*?)\s*```$", re.S)
RX_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
RX_SEP_RUN = re.compile(r"[\s_.\-]+")
RX_ES_END = re.compile(r"(?:ch|sh|ss|x|z)es$")
RX_STOP_WORD = re.compile(r"^(?:and|for|the|with)$")
RX_OUR_END = re.compile(r"our$")
RX_GLUE = re.compile(r"(\d)\s+([a-z]{1,3})(?![a-z0-9])")
RX_ALNUM = re.compile(r"[a-z0-9]+")
RX_TOK3 = re.compile(r"[a-z0-9]{3,}")
RX_TOK_SHORT = re.compile(r"\d+[a-z]|\d*[a-z][a-z0-9]+")
RX_DIGITS = re.compile(r"\d+")
RX_SIGNED = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")
RX_COUNT_TEXT = re.compile(r"^\s*(\d+)(?:\.0*)?\s*$")
RX_HAS_DIGIT = re.compile(r"\d")
RX_SPACE = re.compile(r"\s")
FILL_WORD = "question"
NOFILL = object()
FIRST_SPELLING: Dict[str, str] = {}


def is_failure(v):
    if not isinstance(v, dict) or not (1 <= len(v) <= 3):
        return False
    for k, x in v.items():
        if RX_FAIL_KEY.search(str(k)) and x:
            return True
    if len(v) == 1:
        x = list(v.values())[0]
        if isinstance(x, str) and 1 <= len(x) <= 40 and " " not in x:
            return True
    return False


def plain(text):
    s = str(text).strip()
    if RX_TRUE_FALSE.match(s):
        return s.lower() == "true"
    if RX_EMPTY_WORD.match(s):
        return None
    m = RX_MONEY.match(s)
    if m and not RX_LEAD_ZERO.match(m.group(1)):
        n = m.group(1).replace(",", "")
        if "." in n:
            return float(n)
        return int(n)
    return s


def block_value(text):
    v = plain(text)
    s = str(text).strip()
    if isinstance(v, bool) or (v is None and s):
        return s
    return v


def pair_under(text, mark):
    i = text.find(mark)
    if i >= 0:
        name, value = text[:i], text[i + len(mark):]
    else:
        m2 = mark.strip()
        if m2 and text.endswith(m2):
            name, value = text[:-len(m2)], ""
        else:
            return None
    name = RX_BULLET.sub("", name, count=1).strip()
    if not RX_FIELD_NAME.fullmatch(name) or len(name.split()) > 5:
        return None
    return name, value.strip()


def split_fields(text, sep, mark):
    parts = [p for p in text.split(sep) if p.strip()]
    if not parts or pair_under(parts[0].strip(), mark) is None:
        return [], 0
    fields = []
    named = 0
    heading = None
    for raw in parts:
        s = raw.strip()
        if sep == "\n" and heading is not None and (RX_BULLET.match(raw) or raw[:1] in (" ", "\t")):
            f = fields[heading]
            f[1] = f[1] + "\n" + s if f[1] else s
            continue
        heading = None
        if sep == "\n" and s.endswith(":") and RX_FIELD_NAME.fullmatch(s[:-1].strip()):
            fields.append([s[:-1].strip(), ""])
            heading = len(fields) - 1
            named += 1
            continue
        p = pair_under(s, mark)
        if p is not None:
            fields.append([p[0], p[1]])
            named += 1
        elif fields:
            fields[-1][1] = fields[-1][1] + sep + s
    return fields, named


def attr_block(text):
    s = str(text).strip()
    if len(s) < 5:
        return None
    if s[0] in "{[" and s[-1] in "}]":
        try:
            v = json.loads(s)
        except Exception:
            out = {}
            for m in RX_LOOSE_PAIR.finditer(s):
                out[m.group(1).strip()] = plain(m.group(2).strip())
            return out if len(out) >= 2 else None
        return v if isinstance(v, dict) else None
    if "=" in s:
        parts = [p for p in RX_BLOCK_CUT.split(s) if p.strip()]
        block = {}
        last = None
        first_name = None
        for p in parts:
            i = p.find("=")
            name = p[:i].strip() if i >= 0 else ""
            if i >= 0 and 1 <= len(name) <= BLOCK_NAME_LEN and name not in block:
                block[name] = block_value(p[i + 1:])
                last = name
                if first_name is None:
                    first_name = name
            elif last is not None:
                block[last] = "%s; %s" % (block[last], p.strip())
            else:
                return None
        if len(block) >= 2:
            return block
        if len(block) == 1 and ":" not in first_name and len(first_name) <= 30 and len(s) <= 90:
            return block
        return None
    lines = [ln for ln in s.split("\n") if ln.strip()]
    if len(lines) >= 2 and all(RX_BULLET.match(ln) for ln in lines):
        out = {}
        ok = True
        for ln in lines:
            p = pair_under(ln.strip(), ": ")
            if p is None:
                ok = False
                break
            out[p[0]] = plain(p[1])
        if ok:
            return out
    for sep in (", ", "; "):
        if sep in s and ": " in s:
            parts = [p.strip() for p in s.split(sep)]
            pairs = [pair_under(p, ": ") for p in parts]
            if len(parts) >= 2 and all(pairs) and not any("." in p[0] for p in pairs) and all(len(p[1]) <= BLOCK_VALUE_LEN for p in pairs):
                return {p[0]: plain(p[1]) for p in pairs}
    return None


def record_text(text):
    best = None
    best_named = -1
    for sep in FIELD_MARKS:
        if sep not in text:
            continue
        pieces = len([p for p in text.split(sep) if p.strip()])
        share = PAIR_SHARE if sep == ", " else FIELD_SHARE
        for mark in PAIR_MARKS:
            fields, named = split_fields(text, sep, mark)
            if named < REC_MIN_FIELDS or named < share * pieces:
                continue
            if len(set(f[0] for f in fields)) * 2 <= len(fields):
                continue
            if named > best_named:
                best, best_named = fields, named
        if best is not None:
            break
    if best is None:
        return None
    obj = {}
    for name, val in best:
        if name in obj:
            continue
        b = attr_block(val)
        if b is not None:
            obj[name] = b
        elif RX_ID_FIELD.search(name):
            obj[name] = val.strip()
        else:
            obj[name] = plain(val)
    return obj


def table_text(text):
    lines = []
    for ln in text.split("\n"):
        if not ln.strip() or RX_RULE_LINE.match(ln):
            continue
        lines.append(ln.strip().strip("|").strip())
    if len(lines) < 2:
        return None
    for sep in ("|", "\t", ";", ","):
        rows = [[c.strip() for c in ln.split(sep)] for ln in lines]
        head = rows[0]
        if len(head) < 3 or any(len(r) != len(head) for r in rows):
            continue
        if not all(RX_FIELD_NAME.fullmatch(c) for c in head):
            continue
        if any(RX_NUM_HEAD.match(c) for c in head):
            continue
        out = []
        for r in rows[1:]:
            o = {}
            for i, c in enumerate(r):
                o[head[i]] = plain(c)
            out.append(o)
        return out
    return None


def records_from_text(text):
    s = str(text).strip()
    if len(s) < 12:
        return None
    if s[0] in "{[":
        try:
            v = json.loads(s)
        except Exception:
            pass
        else:
            return restore(v, 0, False, True)
    for cut in (RX_BLANK_CUT.split(s), s.split("\n")):
        if len(cut) >= 2:
            recs = [record_text(p.strip()) for p in cut]
            ok = [r for r in recs if r is not None]
            if len(ok) >= 2 and len(ok) * 5 >= len(cut) * 4 and len(set(next(iter(r)) for r in ok)) == 1:
                return ok
    if "\n" in s:
        found = record_text(s)
        if found is None:
            found = table_text(s)
    else:
        found = record_text(s)
    if found is not None:
        return found
    if len(s) <= TAGGED_LIMIT and RX_TAG.search(s):
        t = RX_INLINE_TAG.sub("", s)
        t = RX_CELL_TAG.sub(": ", t)
        t = RX_TAG_RUN.sub("\n", t).strip()
        if t != s and len(t) >= 12:
            return records_from_text(t)
    if "\n" in s:
        t = "\n".join(RX_COL_GAP.sub(": ", ln.strip(), count=1) for ln in s.split("\n") if ln.strip())
        if t != s:
            return record_text(t)
    return found


def block_text(text):
    if len(text) < 5 or "=" not in text or not ("; " in text or " | " in text or "\n" in text):
        return text
    parts = [p for p in RX_BLOCK_CUT.split(text.strip()) if p.strip()]
    if len(parts) < 2 or not all(RX_BLOCK_PART.fullmatch(p.strip()) for p in parts):
        return text
    b = attr_block(text)
    return text if b is None else b


def attr_list(lst, allowed):
    if len(lst) < 2:
        return None
    first = lst[0]
    if isinstance(first, dict):
        keys = list(first.keys())
        if len(keys) != 2 or not RX_PAIR_KEY.search(str(keys[0])):
            return None
        out = {}
        for el in lst:
            if not isinstance(el, dict) or set(el.keys()) != set(keys):
                return None
            n, v = el[keys[0]], el[keys[1]]
            if not isinstance(n, str) or not (1 <= len(n) <= BLOCK_NAME_LEN) or isinstance(v, (dict, list)):
                return None
            out[n.strip()] = v
        return out
    if isinstance(first, list):
        out = {}
        for el in lst:
            if not isinstance(el, list) or len(el) != 2:
                return None
            n, v = el
            if not isinstance(n, str) or not (1 <= len(n) <= BLOCK_NAME_LEN) or isinstance(v, (dict, list)):
                return None
            out[n.strip()] = v
        return out
    if allowed and all(isinstance(x, str) for x in lst):
        out = {}
        for x in lst:
            p = pair_under(x.strip(), ": ")
            if p is None:
                p = pair_under(x.strip(), "=")
            if p is None or len(p[1]) > BLOCK_VALUE_LEN or len(p[0].split()) > 3:
                return None
            out[p[0]] = plain(p[1])
        return out
    return None


def column_rows(obj):
    heads = [k for k, v in obj.items() if isinstance(v, list) and len(v) >= 2
             and all(isinstance(x, str) and RX_FIELD_NAME.fullmatch(x) for x in v)]
    for hk in heads:
        names = obj[hk]
        for k, v in obj.items():
            if k == hk:
                continue
            if isinstance(v, list) and v and all(isinstance(r, list) and len(r) == len(names) for r in v):
                out = {}
                for kk, vv in obj.items():
                    if kk == hk:
                        continue
                    out[kk] = [dict(zip(names, r)) for r in v] if kk == k else vv
                return out
    return obj


def id_texts(obj):
    if not any(isinstance(v, str) and " " in v for v in obj.values()):
        return obj
    out = {}
    for k, v in obj.items():
        if isinstance(v, int) and not isinstance(v, bool) and abs(v) >= 100 and RX_ID_FIELD.search(str(k)):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def holds_id(obj):
    return isinstance(obj, dict) and any(RX_ID_FIELD.search(str(k)) for k in obj)


def keyed_rows(obj):
    vals = list(obj.values())
    if len(vals) < 2 or not all(isinstance(v, dict) and len(v) >= 3 for v in vals):
        return None
    if not all(RX_HAS_DIGIT.search(str(k)) and " " not in str(k) for k in obj):
        return None
    shared = set(vals[0])
    for v in vals[1:]:
        shared &= set(v)
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
                out["%s %s" % (k, ik)] = iv
        else:
            out[k] = v
    return out


def id_like(v, named=False):
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not (3 <= len(s) <= 64) or RX_SPACE.search(s):
        return False
    return named or bool(RX_HAS_DIGIT.search(s))


def field_rule(x, depth, inside, page):
    r = restore(x, depth + 1, inside, page)
    if isinstance(x, str) and isinstance(r, dict) and len(r) < 5 and not holds_id(r) and not any(id_like(y) for y in r.values()):
        return x
    return r


def _flattenable(inner, outer_len):
    if outer_len < 3 or not inner or holds_id(inner):
        return False
    vals = list(inner.values())
    if not any(isinstance(x, dict) for x in vals) or any(isinstance(x, list) for x in vals):
        return False
    f = vals[0]
    if isinstance(f, dict):
        return bool(f)
    return isinstance(f, (str, int, float))


def restore(v, depth=0, inside=False, page=True):
    if depth > 7:
        return v
    if isinstance(v, str):
        r = records_from_text(v)
        return v if r is None else r
    if isinstance(v, list):
        a = attr_list(v, inside and page)
        if a is not None:
            return a
        return [field_rule(x, depth, False, page) for x in v]
    if isinstance(v, dict):
        first_text = bool(v) and isinstance(list(v.values())[0], str)
        o = id_texts(column_rows(v))
        carded = len(o) >= 3 or holds_id(o)
        o2 = {}
        for k, x in o.items():
            kr = keyed_rows(x) if isinstance(x, dict) else None
            o2[k] = kr if kr is not None else x
        o = o2
        listing = holds_id(o) or any(id_like(x) for x in o.values())
        out = {}
        for k, x in o.items():
            if not isinstance(x, str) or len(o) <= 2:
                out[k] = field_rule(x, depth, carded, page)
            else:
                out[k] = block_text(x) if listing else x
        if depth <= 3:
            for k in list(out.keys()):
                x = out[k]
                if isinstance(x, dict) and _flattenable(x, len(out)):
                    out[k] = flat_block(x)
        if len(out) == 1 and first_text:
            inner = list(out.values())[0]
            if isinstance(inner, dict) and len(inner) >= 3:
                return inner
        return out
    return v


def join_values(held, new):
    hp = set(p.strip().lower() for p in RX_COMMA_CUT.split(held))
    extras = []
    for p in RX_COMMA_CUT.split(new):
        q = p.strip().lower()
        if q and q not in hp:
            hp.add(q)
            extras.append(p.strip())
    if not extras:
        return held
    joined = held + (", " if held.strip() else "") + ", ".join(extras)
    FIRST_SPELLING[joined] = FIRST_SPELLING.get(held, held)
    return joined


def fold(v, depth=0):
    if depth > 5:
        return v
    if isinstance(v, list):
        return [fold(x, depth + 1) for x in v]
    if not isinstance(v, dict):
        return v
    out = {}
    seen = {}
    for k, x in v.items():
        fx = fold(x, depth + 1)
        n = word_form(k)
        if n not in seen:
            seen[n] = k
            out[k] = fx
            continue
        kk = seen[n]
        if fx == out[kk]:
            continue
        if isinstance(fx, str) and isinstance(out[kk], str) and not RX_ID_FIELD.search(n):
            out[kk] = join_values(out[kk], fx)
        else:
            out[k] = fx
    if len(out) < 2 and len(v) >= 2:
        return {k: fold(x, depth + 1) for k, x in v.items()}
    return out


def dump(v):
    try:
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
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


def rec_dump(v, cl, limit):
    t = dump(clip(v, cl))
    if len(t) <= limit or not isinstance(v, dict):
        return t[:limit]
    best = None
    bl = -1
    for k, x in v.items():
        if isinstance(x, dict) and len(x) >= 2:
            n = len(dump(clip(x, cl)))
            if n > bl:
                best, bl = k, n
    if best is None:
        return t[:limit]
    keys = list(v.keys())
    after = keys[keys.index(best) + 1:]
    if not any(isinstance(v[k], str) and v[k].strip() for k in after):
        return t[:limit]
    blk = list(v[best].items())
    for keep in range(len(blk) - 1, -1, -1):
        nb = dict(blk[:keep])
        nb["..."] = "%d more entries cut" % (len(blk) - keep)
        r = dict((k, nb if k == best else x) for k, x in v.items())
        d = dump(clip(r, cl))
        if len(d) <= limit:
            return d
    return t[:limit]


RX_LETTERS_ONLY = re.compile(r"^[^\W\d_]+$")
RX_KIN_VOWEL = re.compile(r"[aeiouy]")
RX_SPLIT_PARTS = re.compile(r"[,/;|]")
RX_MONEY_TEXT = re.compile(r"^\s*[^\d\s]{0,3}\s?(\d[\d,]*(?:\.\d+)?)\s*[A-Za-z]{0,6}\.?\s*$")


def kin(a, b):
    if not RX_LETTERS_ONLY.match(a or "") or not RX_LETTERS_ONLY.match(b or "") or a == b:
        return a == b
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    if n < 4:
        return False
    for tail in (a[n:], b[n:]):
        if tail and not (2 <= len(tail) <= 4 and RX_KIN_VOWEL.search(tail)):
            return False
    return True


def fold_words(text):
    out = []
    for w in RX_ALNUM.findall(str(text).lower()):
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def split_run(value, text):
    want = fold_words(value)
    if len(want) < 2:
        return False
    run = spread = False
    for part in RX_SPLIT_PARTS.split(str(text)):
        f = fold_words(part)
        for i in range(len(f) - len(want) + 1):
            if f[i:i + len(want)] == want:
                run = True
        j = 0
        for w in f:
            if j < len(want) and w == want[j]:
                j += 1
        if j == len(want):
            spread = True
    return spread and not run


def money(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for x in v.values():
            if _num(x):
                return float(x)
        return None
    if not isinstance(v, str):
        return None
    m = RX_MONEY_TEXT.match(v)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def json_obj(text):
    if not isinstance(text, str):
        return None
    s = text.strip()
    m = RX_FENCE.match(s)
    if m:
        s = m.group(1)
    try:
        v = json.loads(s)
    except Exception:
        i, j = s.find("{"), s.rfind("}")
        if 0 <= i < j:
            try:
                v = json.loads(s[i:j + 1])
            except Exception:
                return None
            return v if isinstance(v, dict) else None
        return None
    return v if isinstance(v, dict) else None


def word_form(name):
    s = RX_CAMEL.sub(" ", str(name))
    return RX_SEP_RUN.sub(" ", s).strip().lower()


def under_form(name):
    return word_form(name).replace(" ", "_")


def stems(name):
    return set(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in word_form(name).split())


def roots(name):
    out = set()
    for w in word_form(name).split():
        if RX_ES_END.search(w):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if len(w) >= 3 and not RX_STOP_WORD.match(w):
            out.add(w)
    return out


def spelled(name):
    out = set()
    for w in word_form(name).split():
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(RX_OUR_END.sub("or", w))
    return out


def names_match(p, k):
    if not p or not k:
        return False
    return p == k or p <= k or (len(k) >= 2 and k <= p)


def glued(text):
    return RX_GLUE.sub(r"\1\2", str(text).lower())


def text_words(text):
    t = str(text).lower()
    return set(RX_ALNUM.findall(t)) | set(RX_ALNUM.findall(glued(t)))


def value_tokens(value):
    g = glued(value)
    for rx in (RX_TOK3, RX_TOK_SHORT, RX_DIGITS):
        t = rx.findall(g)
        if t:
            return t
    return []


def kinds(schema):
    t = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(t, str):
        return [t]
    if isinstance(t, list):
        return [x for x in t if isinstance(x, str)]
    return []


def kind_fit(k, v):
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


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def shape(v, schema, path, faults):
    if not isinstance(schema, dict):
        return v
    ks = kinds(schema)
    if ks and not any(kind_fit(k, v) for k in ks):
        for k in ks:
            if k == "integer" and isinstance(v, float) and v == v and v not in (float("inf"), float("-inf")) and v == int(v):
                v = int(v)
                break
            if (k == "integer" or k == "number") and isinstance(v, str) and RX_SIGNED.match(v):
                t = v.strip()
                if "." in t:
                    f = float(t)
                    v = int(f) if (k == "integer" or f == int(f)) else f
                else:
                    v = int(t)
                break
            if k == "string" and _num(v):
                v = str(v)
                break
            if k == "boolean" and isinstance(v, str) and RX_TRUE_FALSE.match(v.strip()):
                v = v.strip().lower() == "true"
                break
            if k == "array" and not isinstance(v, list) and v is not None:
                v = [v]
                break
        if not any(kind_fit(k, v) for k in ks):
            faults.append("%s should be %s" % (path, "/".join(ks)))
            return v
    en = schema.get("enum")
    if isinstance(en, list) and en and v not in en:
        if isinstance(v, str):
            low = v.strip().lower()
            for o in en:
                if isinstance(o, str) and o.lower() == low:
                    v = o
                    break
        if v not in en:
            faults.append("%s must be one of %s" % (path, dump(en)))
    ml = schema.get("maxLength")
    if isinstance(v, str) and _num(ml) and len(v) > ml:
        cut = v[:int(ml)]
        if (not v[int(ml)].isspace() and cut and not cut[-1].isspace() and " " in cut.strip()):
            cut = cut[:cut.rfind(" ")].rstrip()
        v = cut
    mn = schema.get("minLength")
    if isinstance(v, str) and _num(mn) and len(v) < mn:
        faults.append("%s is too short" % path)
    if _num(v):
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if _num(lo) and v < lo:
            faults.append("%s is below the minimum %s" % (path, lo))
        if _num(hi) and v > hi:
            faults.append("%s exceeds the maximum %s" % (path, hi))
    if isinstance(v, list):
        mx, mi = schema.get("maxItems"), schema.get("minItems")
        if _num(mx) and len(v) > mx:
            faults.append("%s allows at most %s entries" % (path, mx))
        if _num(mi) and len(v) < mi:
            faults.append("%s needs at least %s entries" % (path, mi))
        it = schema.get("items")
        if isinstance(it, dict):
            v = [shape(x, it, "%s[%d]" % (path, i), faults) for i, x in enumerate(v)]
    if isinstance(v, dict):
        props = schema.get("properties")
        props = props if isinstance(props, dict) else {}
        if schema.get("additionalProperties") is False:
            for k in v:
                if k not in props:
                    faults.append("%s.%s is not allowed by the schema" % (path, k))
        req = schema.get("required")
        if isinstance(req, list):
            for k in req:
                if isinstance(k, str) and k not in v:
                    faults.append("%s.%s is required" % (path, k))
        out = {}
        for k, x in v.items():
            out[k] = shape(x, props[k], "%s.%s" % (path, k), faults) if k in props else x
        v = out
    return v


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
    if isinstance(v, float):
        if v == v and v not in (float("inf"), float("-inf")) and v == int(v):
            return int(v)
        return None
    if isinstance(v, str):
        m = RX_COUNT_TEXT.match(v)
        return int(m.group(1)) if m else None
    return None


def learn_id(rows):
    if not rows or not isinstance(rows[0], dict):
        return None
    best = None
    bs = None
    for k in rows[0]:
        named = bool(RX_ID_FIELD.search(str(k)))
        vals = [r.get(k) if isinstance(r, dict) else None for r in rows]
        if not all(id_like(x, named) for x in vals):
            continue
        ss = [x.strip() for x in vals]
        if len(rows) >= 2 and len(set(ss)) < len(ss):
            continue
        sc = (named, sum(len(x) for x in ss) / float(len(ss)))
        if bs is None or sc > bs:
            best, bs = k, sc
    return best


def in_rows(id_, v, depth=0, inlist=False):
    if depth > 5:
        return False
    if isinstance(v, list):
        return any(in_rows(id_, x, depth + 1, True) for x in v)
    if isinstance(v, dict):
        if inlist and any(isinstance(x, str) and x.strip() == id_ for x in v.values()):
            return True
        return any(in_rows(id_, x, depth + 1, False) for x in v.values() if isinstance(x, (list, dict)))
    return False


def model_client(room):
    if room >= 50:
        return _MODEL_CLIENT
    t = max(5, int(room // 5) * 5)
    c = _SHORT_CLIENTS.get(t)
    if c is None:
        c = _SHORT_CLIENTS[t] = ProxyClient(timeout=t, max_retries=1, retry_delay=1)
    return c


def attrs_of(rec):
    if not isinstance(rec, dict):
        return None
    best = None
    for v in rec.values():
        if isinstance(v, dict) and len(v) >= 2:
            sc = {k: x for k, x in v.items() if isinstance(x, (str, int, float)) and not isinstance(x, bool)}
            if len(sc) >= 2 and (best is None or len(sc) > len(best)):
                best = sc
    return best


def soft(fallback):
    def deco(fn):
        def run(self, *a, **kw):
            try:
                return fn(self, *a, **kw)
            except Exception as e:
                self.trace.append({"kind": "soft_crash", "method": fn.__name__,
                                   "error": ("%s: %s" % (type(e).__name__, e))[:300]})
                return fallback() if callable(fallback) else fallback
        run.__name__ = fn.__name__
        return run
    return deco


class Trace(list):
    def __init__(self, start):
        list.__init__(self)
        self.start = start

    def append(self, entry):
        if isinstance(entry, dict) and "t" not in entry:
            entry = dict(entry)
            entry["t"] = round(time.monotonic() - self.start, 1)
        list.append(self, entry)


def filler(schema):
    if not isinstance(schema, dict):
        return NOFILL
    en = schema.get("enum")
    if isinstance(en, list) and en:
        return en[0]
    if "const" in schema:
        return schema["const"]
    if "default" in schema:
        return schema["default"]
    ks = kinds(schema)
    if "integer" in ks or "number" in ks:
        m = schema.get("minimum")
        return m if _num(m) else 1
    if "boolean" in ks:
        return False
    if "string" in ks:
        ml = schema.get("maxLength")
        ml = int(ml) if _num(ml) and ml else 8
        return FILL_WORD[:ml]
    return NOFILL


def strings_of(v, out=None):
    if out is None:
        out = []
    if isinstance(v, str):
        out.append(v)
    elif isinstance(v, dict):
        for x in v.values():
            strings_of(x, out)
    elif isinstance(v, list):
        for x in v:
            strings_of(x, out)
    return out


def says(v, text):
    return text in strings_of(v)


class Item(object):
    def __init__(self, name, shelf, qty, musts):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = list(musts)
        self.told = []
        self.dropped = []
        self.free = []

    def live(self):
        return [r for r in self.musts + self.told if r not in self.dropped]


RX_409_END = re.compile(r"finish|quarantin|finaliz")
RX_EXPECTED = re.compile(r"expected (\d+)")
RX_REUSE = re.compile(r"reuse|different action")
RX_POINTER_WORDS = re.compile(r"\b(?:listing|page|shown|other than|unlike|same as)\b", re.I)
RX_PLAIN_NUM = re.compile(r"^\s*\d+(?:\.\d+)?\s*$")
RX_TITLE_EDGE = r"(?<![a-z0-9])%s(?![a-z0-9])"
RX_STAGE = re.compile(r"^(?:choose|order|end)$")
RX_ANTHROPIC = re.compile(r"^anthropic/")
RX_SENT_BREAK = re.compile(r"(?<=[.!?;])\s+|\n+")
RX_FIGURE = re.compile(r"\d+(?:\.\d+)?")


class E1(object):
    def __init__(self, problem):
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.pv = env["policy_view"]
        q = self.pv.get("query")
        self.request = str(q) if q else ""
        self.tools = []
        self.schemas = {}
        for t in self.pv.get("tools") or []:
            if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str):
                f = t["function"]
                self.tools.append(t)
                p = f.get("parameters")
                self.schemas[f["name"]] = p if isinstance(p, dict) else {"type": "object", "properties": {}}
        ms = self.pv.get("max_steps")
        n = count_of(ms)
        if n is None and isinstance(ms, float) and ms == ms and ms not in (float("inf"), float("-inf")):
            n = int(ms)
        self.budget_steps = n if n and n > 0 else 30
        self.start = time.monotonic()
        self.step = 0
        self.trace = Trace(self.start)
        self.finished = bool(self.pv.get("finished"))
        self.store_broke = False
        self.no_credit = False
        self.waited = 0.0
        self.misses = 0
        self.spend = 0.0
        self.model_calls = 0
        self.turns = 0
        self.gap = STORE_PAUSE
        self.store_secs = 0.0
        self.store_calls = 0
        self.last_start = 0.0
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
        self.qlog = []
        self.heard = set()
        self.change_notes = []
        self.version = 0
        self.queued = []
        self.lookups = []
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
        self.remembered = {}
        self.remembered_args = {}
        self.page_arg = {}
        self.page_queries = {}
        self.searched = []
        self.value_rows = set()
        self.cmp_ok = 0
        self.cmp_fail = 0
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
        self.unavail = set()
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
        self.followups = []
        self.listing_mode = True
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
        self.cheaper_holds = 0
        self.dear_paged = set()
        self.cheaper_named = set()
        self.check_better = {}
        self.swap_agreed = False
        self.weak_holds = 0
        self.proofs = {}
        self.tools_reopened = False
        self.item_origins = {}
        self.swept = False
        self.series_swept = set()
        self.set_holds = 0
        self.link_pages = {}
        self.trail_notes = []
        self.ever_carted = {}
        self.effort_ok = True
        self.last_args = {}
        self.carted_ids = set()
        self.units_known = False

    def steps_left(self):
        return self.budget_steps - self.step

    def clock(self):
        return time.monotonic() - self.start

    def room(self):
        return EPISODE_LEN - EPISODE_MARGIN - self.clock()

    def rushed(self):
        pace = self.store_secs / self.store_calls if self.store_calls else 1.0
        return self.room() <= 2 * max(pace, 1.0)

    def msgs_free(self):
        return self.msg_limit - self.msgs_used - max(0, self.msg_reserve - self.gate_msgs)

    def cart_mark(self):
        return dump(sorted([[k, v] for k, v in self.cart.items()], key=lambda p: str(p[0])))

    def gate_due(self, stage):
        if not any(d["stage"] == stage for d in self.duties):
            return False
        return (stage, self.cart_mark() + "#" + str(self.version)) not in self.gate_passes

    def reserve(self):
        if (any(d["stage"] == "order" and d["msgs"] > 0 for d in self.duties) and self.msgs_used < self.msg_limit
                and self.gate_due("order")):
            return 2
        return 1

    def judges(self, limit=SOFT_SPEND):
        if self.spend <= limit and self.spend + HEAVY_ROOM <= HARD_SPEND:
            return [HEAVY, LIGHT]
        if self.spend <= HARD_SPEND:
            return [LIGHT]
        return []

    def pick(self):
        p = self.brief.get("pick")
        if p == "cheapest" or p == "most" or p == "least":
            return p
        return "any"

    def page_tools(self):
        out = []
        o = self.roles.get("offer") or self.roles.get("open")
        if o:
            out.append(o)
        c = self.roles.get("compare")
        if c:
            out.append(c)
        return out

    def facts_tool(self):
        for t, v in self.facts.items():
            if v is True and self.single_target(t):
                return t
        o = self.roles.get("open")
        if o and self.facts.get(o) is not False and self.single_target(o):
            return o
        return None

    def compare_facts(self):
        c = self.roles.get("compare")
        return self.facts.get(c) if c else None

    def compare_bare(self):
        cf = self.compare_facts()
        if cf is False:
            return True
        ft = self.facts_tool()
        return cf is None and ((ft is not None and self.facts.get(ft) is True) or bool(self.roles.get("offer")))

    def row_ids(self, tool, args):
        if tool not in (self.roles.get("open"), self.roles.get("offer"), self.roles.get("compare"), self.roles.get("cart")):
            return args
        tk = self.title_key()
        if not self.rows or not tk:
            return args
        props = (self.schemas.get(tool) or {}).get("properties") or {}

        def fix(k, v):
            if not isinstance(v, str):
                return v
            s = v.strip()
            if s in self.rows or not (3 <= len(s) <= 30) or not RX_HAS_DIGIT.search(s):
                return v
            sch = props.get(k) if isinstance(props, dict) else None
            if isinstance(sch, dict) and sch.get("enum"):
                return v
            rx = re.compile(RX_TITLE_EDGE % re.escape(s.lower()))
            hits = [rid for rid in self.seen if rx.search(str(self.rows[rid].get(tk, "")).lower())]
            if len(hits) == 1:
                self.trace.append({"kind": "row_id", "from": s, "to": hits[0]})
                return hits[0]
            return v
        out = {}
        for k, v in args.items():
            out[k] = [fix(k, x) for x in v] if isinstance(v, list) else fix(k, v)
        return out

    def _single_call(self, messages, model, cap, tools):
        msgs = messages
        if RX_ANTHROPIC.match(model) and len(messages) >= 2:
            msgs = list(messages)
            for i in (0, len(msgs) - 1):
                c = msgs[i].get("content")
                if isinstance(c, str) and c:
                    d = dict(msgs[i])
                    d["content"] = [{"type": "text", "text": c, "cache_control": {"type": "ephemeral"}}]
                    msgs[i] = d
        body = {"model": model, "messages": msgs, "max_tokens": cap, "temperature": 0}
        effort = False
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
            if self.effort_ok and self.clock() >= LOW_EFFORT_CLOCK:
                body["reasoning"] = {"effort": "low"}
                effort = True
        res = model_client(self.room()).post_verbose("/inference/chat/completions", body)
        data = res.data
        if effort and not (isinstance(data, dict) and data.get("choices")):
            st = (res.error or {}).get("status")
            if isinstance(data, dict) and isinstance(data.get("error"), dict):
                st = count_of(data["error"].get("code"))
            if st == 400 or st == 422:
                self.effort_ok = False
                self.trace.append({"kind": "effort_refused", "model": model, "status": st})
                return self._single_call(messages, model, cap, tools)
        if isinstance(data, dict) and data.get("choices"):
            u = data.get("usage")
            cost = u.get("cost") if isinstance(u, dict) else None
            if _num(cost):
                self.spend += cost
            ch0 = data["choices"][0] if isinstance(data["choices"][0], dict) else {}
            msg = ch0.get("message") if isinstance(ch0.get("message"), dict) else {}
            if tools and not msg.get("tool_calls") and ch0.get("finish_reason") == "length":
                self.trace.append({"kind": "model_fail", "model": model, "status": 0,
                                   "body": "the reply stopped at the token cap without a tool call"})
                return None, 0
            if msg.get("content") or msg.get("tool_calls"):
                return msg, None
            self.trace.append({"kind": "empty_reply", "model": model})
            return None, 0
        err = res.error or {}
        status = err.get("status")
        body_txt = err.get("body")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            status = count_of(data["error"].get("code"))
            body_txt = data["error"].get("message")
        self.trace.append({"kind": "model_fail", "model": model, "status": status, "body": str(body_txt)[:200]})
        return None, status

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
                msg, status = self._single_call(messages, m, cap, tools)
                if msg is not None:
                    return msg
                if not (status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)):
                    break
                if tries >= (1 if status in (None, 408) else len(RETRY_WAITS)):
                    break
                w = RETRY_WAITS[tries]
                if self.waited + w > WAIT_BUDGET or self.room() - 1.2 * w < CALL_FLOOR:
                    break
                time.sleep(w * random.uniform(0.8, 1.2))
                self.waited += w
                tries += 1
            if status == 402:
                self.no_credit = True
                return None
        return None

    def ask_json(self, prompt, models, cap=1800):
        for m in models:
            msg = self.think([{"role": "user", "content": prompt}], [m], cap)
            o = json_obj(msg.get("content")) if msg else None
            if o is not None:
                return o
            if self.store_broke or self.no_credit or self.model_calls >= CALL_CAP:
                return None
        return None

    def env(self, name, args):
        body = dict(self.binding)
        body["step"] = self.step + 1
        body["action"] = {"name": name, "args": args}
        self.last_args[name] = args
        gap = time.monotonic() - self.last_start
        if gap < self.gap:
            time.sleep(self.gap - gap)
        self.last_start = time.monotonic()
        for _ in range(4):
            t0 = time.monotonic()
            res = _ENV_CLIENT.post_verbose("/environment/call", body)
            self.store_secs += time.monotonic() - t0
            self.store_calls += 1
            data = res.data
            if isinstance(data, dict):
                st = data.get("step")
                if isinstance(st, int) and not isinstance(st, bool):
                    self.step = st
                else:
                    self.step += 1
                if data.get("finished") is True:
                    self.finished = True
                if name != self.roles.get("ask"):
                    data["observation"] = self.unspell(name, data.get("observation"))
                obs = data.get("observation")
                self.note_read(name, args, obs)
                if name == self.facts_tool() and isinstance(obs, dict) and not is_failure(obs):
                    lid = self.listing_id(obs, False, 0)
                    if lid and attrs_of(obs) is None:
                        self.factless.add((name, lid))
                if name in self.page_tools() and not (isinstance(obs, dict) and obs.get("error")) and not is_failure(obs):
                    cj = None
                    for v in args.values():
                        got = []
                        if isinstance(v, str):
                            got.append(v.strip())
                        elif isinstance(v, list):
                            if cj is None:
                                cj = dump(obs)
                            got.extend(x.strip() for x in v if isinstance(x, str) and x.strip() in cj)
                        for g in got:
                            self.opened.add(g)
                            self.views.setdefault(g, set()).add(name)
                return data
            err = res.error or {}
            status = err.get("status")
            btxt = str(err.get("body") or "")
            self.trace.append({"kind": "env_fail", "tool": name, "status": status, "body": btxt[:200]})
            if status == 409:
                low = btxt.lower()
                if RX_409_END.search(low):
                    self.finished = True
                    return None
                m = RX_EXPECTED.search(btxt)
                if m:
                    k = int(m.group(1))
                    self.step = k - 1
                    body["step"] = k
                    continue
                if RX_REUSE.search(low):
                    self.step += 1
                    body["step"] = self.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.store_broke = True
                return None
        return None

    def unspell(self, tool, obs):
        views = [self.roles.get(r) for r in "open offer compare".split() if self.roles.get(r)]
        page = tool != self.roles.get("search") or tool in views
        r = fold(restore(obs, 0, False, page))
        if tool not in views:
            return r
        for rid in self.seen:
            if rid not in self.opened:
                self.row_keys |= set(self.rows[rid].keys())
        if not self.row_keys:
            return r

        def gather(rec):
            if not isinstance(rec, dict) or attrs_of(rec) is not None or not self.listing_id(rec, True, 1):
                return rec
            extras = {}
            for k, v in rec.items():
                if k not in self.row_keys and isinstance(v, (str, int, float)) and not isinstance(v, bool):
                    extras[k] = v
            if len(extras) < REC_MIN_FIELDS:
                return rec
            out = {k: v for k, v in rec.items() if k not in extras}
            out["attributes"] = extras
            return out
        if isinstance(r, dict):
            if self.listing_id(r, False, 0):
                return gather(r)
            return {k: ([gather(x) for x in v] if isinstance(v, list) else v) for k, v in r.items()}
        return r

    def _shared_shelf(self, b):
        seen = set()
        for e in items_of(b.get("items")):
            if isinstance(e, dict):
                s = word_form(e.get("shelf") or "")
                if s:
                    if s in seen:
                        return True
                    seen.add(s)
        return False

    def _complete(self, b):
        return bool(b) and bool(b.get("items")) and isinstance(b.get("roles"), dict) and bool(b["roles"].get("buy"))

    def read_brief(self):
        lines = []
        for t in self.tools:
            f = t["function"]
            lines.append("- %s: %s | parameters: %s" % (f["name"], f.get("description") or "", dump(f.get("parameters") or {})))
        prompt = BRIEF_PROMPT + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.pv.get("opening") is not None:
            prompt += "\n\nOPENING NOTE THE STORE SHOWED:\n" + dump(clip(self.pv["opening"], 1500))
        b = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit:
            if not b.get("items") or not isinstance(b.get("roles"), dict) or not b["roles"].get("buy"):
                b2 = self.ask_json(prompt, [HEAVY, LIGHT], 2500)
                if b2:
                    b = b2
            elif self._shared_shelf(b):
                b2 = self.ask_json(prompt, [HEAVY], 2500)
                if self._complete(b2):
                    b = b2
        while not b and not self.no_credit and self.model_calls < CALL_CAP and self.clock() < BRIEF_DEADLINE:
            time.sleep(OUTAGE_PAUSE)
            b = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        if b:
            self.waited = 0.0
        roles = b.get("roles") if isinstance(b.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        for k in "buy_args refuse_args".split():
            if not isinstance(b.get(k), dict):
                b[k] = {}
        self.brief = b
        buy, ref = self.roles.get("buy"), self.roles.get("refuse")
        ba, ra = b["buy_args"], b["refuse_args"]
        if buy and buy == ref and not (ba and ra and ba != ra):
            p = ("The store tool %s can either complete the purchase or close the episode with no purchase, according to "
                 "the arguments it is given. Its parameter schema: %s\nReply with a single JSON object only, holding two "
                 "fields: \"buy_args\", the constant arguments that make it complete the purchase, and \"refuse_args\", the "
                 "constant arguments that make it decline with no purchase. Include in both only the arguments that select "
                 "the mode." % (buy, dump(self.schemas.get(buy) or {})))
            o = self.ask_json(p, [LIGHT, HEAVY], 1500)
            if o and isinstance(o.get("buy_args"), dict) and isinstance(o.get("refuse_args"), dict) and o["buy_args"] != o["refuse_args"]:
                b["buy_args"], b["refuse_args"] = o["buy_args"], o["refuse_args"]
        raw = b.get("items")
        if isinstance(raw, (dict, str)):
            raw = [raw]
        if not isinstance(raw, list):
            raw = []
        items = []
        for e in raw:
            if not isinstance(e, dict):
                continue
            q = count_of(e.get("quantity"))
            q = q if q and q > 0 else 1
            ms = e.get("musts")
            ms = ms if isinstance(ms, list) else ([ms] if ms else [])
            musts = [str(x) for x in ms if x]
            nm = e.get("name")
            nm = str(nm) if nm is not None and str(nm).strip() else "the product"
            sh = e.get("shelf")
            sh = str(sh) if sh else ""
            items.append(Item(nm, sh, q, musts))
        merged = []
        newno = {}
        for i, it in enumerate(items):
            target = None
            sw = word_form(it.shelf)
            for j, m in enumerate(merged):
                if sw and it.qty == m.qty and sw == word_form(m.shelf) and not self._conflict(it, m):
                    target = j
                    break
            if target is None:
                merged.append(it)
                newno[i + 1] = len(merged)
            else:
                m = merged[target]
                for r in it.musts:
                    if r not in m.musts:
                        m.musts.append(r)
                newno[i + 1] = target + 1
                self.trace.append({"kind": "merge", "item": it.name, "into": m.name})
        if not merged:
            merged = [Item("the requested product", "", 1, [])]
        self.items = merged
        self.lookups = [str(lk).strip() for lk in items_of(b.get("lookups"))]
        self.scope_reqs()
        n = len(merged)
        for f in items_of(b.get("fits")):
            if not isinstance(f, dict):
                continue
            a, w = newno.get(count_of(f.get("item"))), newno.get(count_of(f.get("with")))
            if a and w and a != w and a <= n and w <= n and (a, w) not in self.fit_pairs:
                self.fit_pairs.append((a, w))
        allm = [(i, r) for i, it in enumerate(self.items, 1) for r in it.musts]
        for lk in items_of(b.get("lookups")):
            t = str(lk).strip()
            if not any(r == t for _, r in allm):
                p = word_form(t.partition(":")[0])
                for _, r in allm:
                    if word_form(r.partition(":")[0]) == p:
                        t = r
                        break
            no = next((i for i, r in allm if r == t), 1)
            if t and (no, t) not in self.refs:
                self.refs.append((no, t))
        self.unmix()
        bud = b.get("budget")
        if isinstance(bud, str) and RX_PLAIN_NUM.match(bud):
            bud = float(bud)
        self.budget = bud if _num(bud) and bud > 0 else None
        has_ask = bool(self.roles.get("ask"))
        mc = count_of(b.get("messages"))
        if mc is not None and mc >= 0:
            self.msg_limit = mc
        elif has_ask:
            self.msg_limit = 3
        sc = count_of(b.get("steps"))
        if sc is not None and sc >= 10 and sc < self.budget_steps:
            self.budget_steps = sc
        for d in items_of(b.get("process")):
            if not isinstance(d, dict):
                continue
            st = d.get("step")
            st = str(st).strip() if st is not None else ""
            if not st:
                continue
            w = str(d.get("when") or "").strip().lower()
            if not RX_STAGE.match(w):
                w = "order"
            k = count_of(d.get("messages"))
            self.duties.append({"text": st, "stage": w, "msgs": max(0, k) if k is not None else 0})
        if has_ask:
            counts = [d["msgs"] for d in self.duties if d["stage"] != "choose" and d["msgs"] > 0]
            self.msg_reserve = min(max(counts), self.msg_limit) if counts else 0

    def scope_reqs(self):
        pointed = set(self.lookups)
        sents = [p for p in RX_SENT_BREAK.split(self.request) if p.strip()]
        sroots = []
        for s in sents:
            rl = []
            for w in word_form(s).split(" "):
                rl.extend(sorted(roots(w)))
            sroots.append(rl)
        nroots = [roots(it.name) for it in self.items]

        def named(si):
            rl = sroots[si]
            rs = set(rl)
            out = []
            for i, nr in enumerate(nroots):
                if not nr or not nr <= rs:
                    continue
                wider = any(j != i and nroots[j] > nr and nroots[j] <= rs for j in range(len(nroots)))
                once = all(rl.count(r) == 1 for r in nr)
                if wider and once:
                    continue
                out.append(i)
            return out
        cands = sorted(set(r for it in self.items for r in it.musts if ":" in r and r not in pointed))
        for m in cands:
            holders = [i for i, it in enumerate(self.items) if m in it.musts]
            if len(holders) < 2:
                continue
            prop, _, val = m.partition(":")
            pr = roots(prop)
            figs = set(RX_FIGURE.findall(val))
            if not pr:
                continue
            keep = set()
            for si, s in enumerate(sents):
                if not pr <= set(sroots[si]) or not figs <= set(RX_FIGURE.findall(s)):
                    continue
                nh = [i for i in named(si) if i in holders]
                if not nh:
                    keep = set()
                    break
                keep |= set(nh)
            if not keep:
                continue
            for i in holders:
                if i not in keep and any(nroots[i] & nroots[k] for k in keep):
                    self.items[i].musts.remove(m)
                    self.trace.append({"kind": "scoped", "must": m, "left": i + 1, "kept": sorted(k + 1 for k in keep)})

    def _conflict(self, a, b):
        for r1 in a.musts:
            p1, _, v1 = r1.partition(":")
            for r2 in b.musts:
                p2, _, v2 = r2.partition(":")
                if word_form(p1) == word_form(p2) and word_form(v1) != word_form(v2):
                    return True
        return False

    def unmix(self):
        for n, it in enumerate(self.items, 1):
            ptrs = [t for (i, t) in self.refs if i == n and t in it.musts]
            rest = [r for r in it.musts if r not in ptrs]
            if not ptrs or not rest:
                continue
            p = (UNMIX_PROMPT + "\n\nREQUEST:\n" + self.request + "\n\nPRODUCT: " + it.name
                 + "\nREQUIREMENTS THAT REFER TO A DIFFERENT LISTING:\n" + "\n".join(ptrs)
                 + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            o = self.ask_json(p, [HEAVY, LIGHT], 1500) or {}
            lst = [x.strip() for x in items_of(o.get("describe_other")) if isinstance(x, str)]
            moved = [r for r in rest if r in lst]
            if not moved:
                continue
            new = ptrs[0] + " (that listing: " + "; ".join(moved) + ")"
            it.musts = [new if r == ptrs[0] else r for r in it.musts if r not in moved]
            self.refs = [(i, new if (i == n and t == ptrs[0]) else t) for (i, t) in self.refs]
            self.trace.append({"kind": "unmix", "item": n, "moved": moved})

    def described(self, n, req):
        if any(req == t for _, t in self.refs) or req in self.lookups:
            return False
        prop, colon, val = req.partition(":")
        if not colon or not prop.strip():
            return False
        if len(RX_ALNUM.findall(val.lower())) < 4:
            return False
        if RX_POINTER_WORDS.search(val):
            return False
        vs = stems(val)
        for j, other in enumerate(self.items, 1):
            if j != n and vs & set(s for s in stems(other.name) if len(s) >= 4):
                return False
        ck = self.category_key()
        blocks = []
        for rid in self.seen:
            rec = self.full.get(rid)
            if rec is None:
                continue
            a = attrs_of(rec)
            if a is None:
                continue
            if ck is None or self.owns_label(n, rec.get(ck)):
                blocks.append(a)
        if len(blocks) < 3:
            return False
        ps = stems(prop)
        for a in blocks:
            for k in a:
                if stems(k) & ps:
                    return False
        return True

    def ledger(self):
        lines = []
        several = len(self.items) > 1
        for n, it in enumerate(self.items, 1):
            head = ("item %d: %s" % (n, it.name)) if several else ("item: %s" % it.name)
            if it.shelf:
                head += " (store label: %s)" % it.shelf
            head += ", quantity %d" % it.qty
            deciding, desc = [], []
            for r in it.live():
                txt = r
                if r in self.ref_values:
                    txt += " (value read on the referenced page: %s)" % self.ref_values[r]
                (desc if self.described(n, r) else deciding).append(txt)
            if deciding:
                head += "; requirements: " + "; ".join(deciding)
            else:
                head += "; requirements: nothing besides the product type"
            if desc:
                head += ("; only in the shopper's phrasing (no attribute of any listing records these, so they settle "
                         "nothing): " + "; ".join(desc))
            dl = []
            live = it.live()
            for r in it.dropped:
                if r and r not in dl and r not in live:
                    dl.append(r)
            if dl:
                head += ("; DROPPED by the shopper after the request was written (no longer required, and this holds "
                         "also where the request's text still states them): " + "; ".join(dl))
            lines.append(head)
        ll = self.label_line()
        if ll:
            lines.append(ll)
        if self.budget is not None:
            unit = self.brief.get("budget_unit")
            s = "budget: at most %s %s %s" % (self.budget, unit if unit else "",
                                            "for each item" if self.brief.get("budget_scope") == "each" else "in total")
            lines.append(s.replace("  ", " ", 1))
        pk = self.pick()
        if pk == "cheapest":
            lines.append("pick: the lowest-priced matching listing (lowest price)")
        elif pk == "most" or pk == "least":
            lines.append("pick: the %s %s among the matching listings" % (pk, self.brief.get("pick_measure") or ""))
        else:
            lines.append("pick: any listing meeting every requirement")
        if self.brief.get("in_stock_only") is not False:
            lines.append("in stock only")
        for w in items_of(self.brief.get("wishes")):
            lines.append("wish (only breaks ties): %s" % (w if isinstance(w, str) else dump(w)))
        for r in items_of(self.brief.get("rules")):
            lines.append("rule the request gives: %s" % (r if isinstance(r, str) else dump(r)))
        if self.brief.get("quiet_decline") is True:
            lines.append("if nothing fits, the episode ends without naming or proposing any listing")
        for d in self.duties:
            if d["stage"] == "choose":
                when = "before choosing"
            elif d["stage"] == "order":
                when = "at the purchase; Cairn carries it out when you order"
            else:
                when = "at a decline; Cairn carries it out at that moment"
            lines.append("step the request requires (%s): %s" % (when, d["text"]))
        ft = self.facts_tool()
        offer = self.roles.get("offer")
        if ft and (self.facts.get(ft) is True or offer):
            s = "listing attributes are shown by %s" % ft
            pts = set(self.page_tools())
            if offer:
                pts.add(offer)
            none = sorted(t for t in pts if self.facts.get(t) is False)
            if none:
                s += "; " + ", ".join(none) + " show none"
            others = sorted(t for t in self.page_tools() if t != ft)
            if others:
                s += ("; " + " or ".join(others) + " display a listing's page, the page meant whenever a chain is "
                      "followed or a page is checked")
            lines.append(s)
        return "\n".join("- " + x for x in lines)

    def live_all(self):
        out = []
        for it in self.items:
            out.extend(it.live())
        return out

    def status(self):
        S, M, L = self.step, self.budget_steps, self.steps_left()
        r = self.reserve()
        out = "[Cairn] step %d of %d used, %d left." % (S, M, L)
        if L <= r + 1:
            out += (" FINAL STEPS: order now if every item already has a carted line; decline only when searching has "
                    "proved that some item lacks any matching listing, and a shortage of steps proves no such thing.")
        elif L <= 6 or (L <= len(self.items) + 5 and not self.checkouts):
            out += " Few steps remain: make the cart hold exactly the picked lines and order at once."
        elif not self.checkouts and L <= 12:
            out += " Steps are getting scarce: decide the picks, put them in the cart and order within a few more steps."
        elif not self.checkouts and not self.cart_touched and S >= M // 2:
            out += (" Half the step budget is spent with nothing in the cart: put the matching listing of every item in "
                    "it now, because the store may yet sell a listing out.")
        if self.clock() >= WIND_DOWN and L > r + 1:
            out += (" TIME IS ALMOST OVER: make the cart hold the picked lines and order within two more turns, since an "
                    "episode cut off by the clock scores zero.")
        live = self.live_all()
        pend = [t for (i, t) in self.refs if t in live and t not in self.ref_values]
        if pend and not self.checkouts:
            out += ("\n[Settle these first: visit the listing each one points at, and also every listing to which that "
                    "listing's page refers onward (only the listings that page directs onward, not every listing it "
                    "mentions), since the final page holds the value] " + " | ".join(pend[:3]))
        if self.leads:
            out += "\n[review candidates] " + ", ".join(self.leads)
        if self.fit_pairs and not self.checkouts:
            fp = self.fit_plan()
            if fp:
                out += "\n" + fp
        if self.fit_pairs:
            sets = self.read_sets()
            cids = set(self.cart_ids())
            if sets and not any(set(s["ids"]) == cids for s in sets):
                out += ("\n[complete sets made only of listings already read]: each line of these sets states every one of "
                        "its own requirements in its attribute block or its title, and the line chosen for each bound item "
                        "matches the device through a code; " + self.set_text(sets) + ". Make the cart consist of one of "
                        "the listed sets, unless some record contradicts a requirement.")
        if self.change_notes:
            out += "\n[changes so far] " + " | ".join(self.change_notes[-4:])
        out += "\n[ledger]\n" + self.ledger()
        return out


RX_THREE_DIGITS = re.compile(r"\d{3}")
RX_LETTERS3 = re.compile(r"[a-z]{3,}")
RX_RUN3 = re.compile(r"[a-z0-9]{3,}")
RX_CMP_WORDS = re.compile(r"\b(?:not|no|other than|unlike|under|less|below|up to|at most|more|above|over|at least|between|within|or)\b", re.I)
RX_PRICED = re.compile(r"price|money|cost|amount|spend|budget", re.I)
RX_CAP_WORD = re.compile(r"max|most|under|below|limit|cap|upto|up_to", re.I)
RX_PRICE_TEXT = re.compile(r"^\s*[^\d\s]{0,3}\s?(\d[\d,]*(?:\.\d+)?)\s*[A-Za-z]{0,6}\.?\s*$")


def gnum(x):
    try:
        return "%g" % x
    except Exception:
        return str(x)


def mark_of(tool, args):
    return str(tool) + dump(args)


class E2(object):
    def listing_id(self, obj, inrows, depth):
        if not isinstance(obj, dict) or len(obj) < 3:
            return None
        if self.id_key is not None:
            v = obj.get(self.id_key)
            if id_like(v, True):
                return v.strip()
        if not (inrows or depth == 0 or any(isinstance(x, dict) for x in obj.values())):
            return None
        k = learn_id([obj])
        if k is None:
            return None
        v = obj[k].strip()
        if "id" in str(k).lower() or v in self.rows:
            return v
        return None

    def row_list(self, lst):
        cands = [x for x in lst if isinstance(x, dict) and len(x) >= 3]
        if not cands or len(cands) * 2 < len(lst):
            return False
        if self.id_key is not None and all(id_like(x.get(self.id_key), True) for x in cands):
            return True
        k = learn_id(cands)
        if k is None:
            return False
        self.id_key = k
        return True

    def absorb(self, v, inrows=False, depth=0):
        if depth > 6:
            return
        if isinstance(v, dict):
            lid = self.listing_id(v, inrows, depth)
            if lid:
                if lid not in self.rows:
                    self.seen.append(lid)
                    self.rows[lid] = {}
                row = self.rows[lid]
                off = any(row.get(k) is True and x is False for k, x in v.items())
                on = any(row.get(k) is False and x is True for k, x in v.items())
                if off:
                    self.unavail.add(lid)
                elif on:
                    self.unavail.discard(lid)
                row.update(v)
                if attrs_of(v) is not None:
                    self.full[lid] = dict(row)
                elif lid in self.full:
                    self.full[lid].update(v)
                return
            for x in v.values():
                self.absorb(x, False, depth + 1)
        elif isinstance(v, list):
            rl = self.row_list(v)
            for x in v:
                self.absorb(x, rl, depth + 1)

    def enum_labels(self):
        out = set()

        def walk(s, d):
            if d > 6 or not isinstance(s, dict):
                return
            en = s.get("enum")
            if isinstance(en, list):
                for o in en:
                    if isinstance(o, str):
                        out.add(o.strip().lower())
            for x in s.values():
                if isinstance(x, dict):
                    walk(x, d + 1)
        for sch in self.schemas.values():
            walk(sch, 0)
        return out

    def category_key(self):
        vals = {}
        pres = {}
        lens = {}
        order = []
        for rid in self.seen:
            for k, x in self.rows[rid].items():
                if isinstance(x, str) and 1 <= len(x) <= 40 and not RX_THREE_DIGITS.search(x):
                    if k not in vals:
                        vals[k] = []
                        pres[k] = 0
                        lens[k] = 0
                        order.append(k)
                    if x not in vals[k]:
                        vals[k].append(x)
                    pres[k] += 1
                    lens[k] += len(x)
        total = max(1, len(self.rows))
        labels = None
        best = None
        bs = None
        for k in order:
            if pres[k] < 0.8 * total or len(vals[k]) > 40:
                continue
            if labels is None:
                labels = self.enum_labels()
            share = sum(1 for x in vals[k] if x.strip().lower() in labels) / float(len(vals[k]))
            sc = (share, -len(vals[k]), pres[k], lens[k] / float(pres[k]))
            if bs is None or sc > bs:
                best, bs = k, sc
        return best

    def label_matches(self, label):
        lw = word_form(label)
        ls = set(s for s in stems(label) if len(s) >= 4)
        for it in self.items:
            if lw == word_form(it.shelf) or lw == word_form(it.name):
                return True
            if ls & set(s for s in (stems(it.shelf) | stems(it.name)) if len(s) >= 4):
                return True
        return self.label_owner(label) is not None

    def key_table(self):
        ck = self.category_key()
        groups = {}
        gorder = []
        for rid in self.seen:
            rec = self.full.get(rid)
            if rec is None:
                continue
            a = attrs_of(rec) or {}
            cv = rec.get(ck) if ck else None
            lab = str(cv) if cv is not None and cv != "" else "listings"
            g = groups.get(lab)
            if g is None:
                g = groups[lab] = {"label": lab, "count": 0, "slots": {}, "sorder": []}
                gorder.append(lab)
            g["count"] += 1
            seenk = set()
            for k, x in a.items():
                wf = word_form(k)
                if not wf or wf in seenk:
                    continue
                seenk.add(wf)
                s = g["slots"].get(wf)
                if s is None:
                    s = g["slots"][wf] = {"key": wf, "count": 0, "shown": [], "vals": {}}
                    g["sorder"].append(wf)
                s["count"] += 1
                shown = str(x).strip().lower()
                meas = str(FIRST_SPELLING.get(x, x)).strip().lower() if isinstance(x, str) else shown
                s["vals"][rid] = (shown, meas)
                if shown not in s["shown"]:
                    s["shown"].append(shown)
        allg = [groups[l] for l in gorder]
        keep = [g for g in allg if self.label_matches(g["label"])] or allg
        keep = sorted(keep, key=lambda g: -g["count"])
        out = []
        for g in keep:
            slots = sorted([g["slots"][w] for w in g["sorder"]], key=lambda s: (-s["count"], len(s["key"])))
            kept = []
            for s in slots:
                ms = [v[1] for v in s["vals"].values()]
                if not ms:
                    continue
                mean = sum(len(m) for m in ms) / float(len(ms))
                if mean > 40 or sum(1 for m in ms if " / " in m) * 2 >= len(ms):
                    continue
                words = [set(RX_LETTERS3.findall(m)) for m in ms]
                shared = 0
                for i, w in enumerate(words):
                    others = set()
                    for j, w2 in enumerate(words):
                        if j != i:
                            others |= w2
                    if w & others:
                        shared += 1
                if len(ms) >= 4 and len(set(ms)) == len(ms) and mean > 12 and shared * 2 < len(ms):
                    continue
                twin = None
                for kk in kept:
                    if self.twin_keys(kk, s):
                        twin = kk
                        break
                if twin is not None:
                    for rid, v in s["vals"].items():
                        if rid not in twin["vals"]:
                            twin["vals"][rid] = v
                    for x in s["shown"]:
                        if x not in twin["shown"]:
                            twin["shown"].append(x)
                    twin["count"] = len(twin["vals"])
                    continue
                kept.append(s)
            kept = sorted(kept, key=lambda s: -s["count"])
            out.append({"label": g["label"], "count": g["count"], "keys": kept[:KEY_TABLE_SIZE]})
        return out

    def twin_keys(self, A, B):
        wa, wb = set(A["key"].split()), set(B["key"].split())
        if not wa or not wb or not (wa & wb):
            return False

        def agree(x, y):
            if x == y:
                return True
            xa, ya = RX_ALNUM.findall(x), RX_ALNUM.findall(y)
            if len(ya) < len(xa):
                short, longer = ya, xa
            else:
                short, longer = xa, ya
            return bool(short) and all(w in longer for w in short) and len(longer) <= 2 * len(short) + 2
        shared = [rid for rid in A["vals"] if rid in B["vals"] and A["vals"][rid][1].strip() and B["vals"][rid][1].strip()]
        ag = sum(1 for rid in shared if agree(A["vals"][rid][1], B["vals"][rid][1]))
        if not (wa <= wb or wb <= wa):
            return len(shared) >= 2 and ag == len(shared)
        if shared:
            return ag * 2 > len(shared)
        return bool(set(x for x in A["shown"] if x) & set(x for x in B["shown"] if x))

    def not_stated(self, label, kwf, shown):
        n = self.label_owner(label)
        if n is None:
            return ""
        sw = set(RX_RUN3.findall(" ".join(shown).lower()))
        for r in self.items[n - 1].live():
            prop, _, val = r.partition(":")
            if RX_CMP_WORDS.search(val):
                continue
            vw = RX_RUN3.findall(val.lower())
            if prop.strip() and word_form(prop) != kwf and (stems(prop) & stems(kwf)) and vw and not all(w in sw for w in vw):
                return " [not the stated %s: another property]" % prop.strip()
        return ""

    def item_groups(self):
        ck = self.category_key()
        groups = {}
        order = []
        for rid in self.seen:
            v = self.rows[rid].get(ck) if ck else None
            lab = str(v) if v is not None else ""
            if lab not in groups:
                groups[lab] = []
                order.append(lab)
            groups[lab].append(rid)
        pool0 = [l for l in order if self.label_matches(l)]
        own = dict((l, self.label_owner(l)) for l in pool0)
        fit = dict((l, self.label_fit(own[l], l) if own[l] else 0) for l in pool0)
        best = {}
        for l in pool0:
            if own[l]:
                best[own[l]] = max(best.get(own[l], 0), fit[l])
        pool = [l for l in pool0 if not own[l] or best[own[l]] < 0.5 or fit[l] >= LABEL_SHARE * best[own[l]]] or order
        pool = sorted(pool, key=lambda l: -len(groups[l]))
        return [(l, groups[l]) for l in pool[:max(1, len(self.items))]]

    def single_target(self, tool):
        sch = self.schemas.get(tool)
        if not isinstance(sch, dict):
            return None
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        req = [r for r in sch.get("required") if isinstance(r, str)] if isinstance(sch.get("required"), list) else []
        texts = [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum")]
        t = next((r for r in req if r in texts), None)
        if t is None and len(texts) == 1:
            t = texts[0]
        if t is None or any(r != t for r in req):
            return None
        return t

    def price(self, rid):
        if not self.price_key:
            return None
        v = (self.rows.get(rid) or {}).get(self.price_key)
        if isinstance(v, dict):
            for x in v.values():
                if _num(x):
                    return x
            return None
        if isinstance(v, str):
            m = RX_PRICE_TEXT.match(v)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except ValueError:
                    return None
            return None
        if _num(v):
            return v
        return None

    def sold_out(self, rid):
        return rid in self.unavail or rid in self.gone

    def buyable(self, rid):
        if self.sold_out(rid):
            return False
        if self.stock_key:
            v = (self.rows.get(rid) or {}).get(self.stock_key)
            if isinstance(v, str):
                s = v.strip()
                if RX_STOCK_WHOLE.match(s) or RX_STOCK_PART.search(s):
                    return False
            elif v is False:
                return False
            elif _num(v) and v <= 0:
                return False
        return True

    def obs_listings(self, obs):
        out = []
        if isinstance(obs, dict):
            lid = self.listing_id(obs, False, 0)
            if lid:
                return [(lid, obs)]
            for v in obs.values():
                if isinstance(v, list):
                    for x in v:
                        if isinstance(x, dict):
                            l = self.listing_id(x, True, 1)
                            if l:
                                out.append((l, x))
                elif isinstance(v, dict):
                    l = self.listing_id(v, False, 1)
                    if l:
                        out.append((l, v))
                    else:
                        for y in v.values():
                            if isinstance(y, list):
                                for x in y:
                                    if isinstance(x, dict):
                                        l2 = self.listing_id(x, True, 2)
                                        if l2:
                                            out.append((l2, x))
        elif isinstance(obs, list):
            for x in obs:
                if isinstance(x, dict):
                    l = self.listing_id(x, True, 1)
                    if l:
                        out.append((l, x))
        return out

    def remember_view(self, tool, obs):
        for lid, rec in self.obs_listings(obs):
            ks = set(rec.keys())
            self.tool_keys.setdefault(tool, set()).update(ks)
            vk = self.view_keys.get(lid)
            if vk is not None and vk[0] == self.version:
                vk[1].update(ks)
            else:
                self.view_keys[lid] = (self.version, set(ks))

    def shown_already(self, tool, lid):
        tk = self.tool_keys.get(tool)
        vk = self.view_keys.get(lid)
        return bool(tk) and vk is not None and vk[0] == self.version and tk <= vk[1] and lid not in self.gone

    def read_shape(self, tool, obs):
        if tool in self.shapes:
            return self.shapes[tool]
        if not isinstance(obs, dict):
            return None
        cands = [k for k, v in obs.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(x, dict) for x in v)]
        if not cands:
            return None
        self.shapes[tool] = None
        sample = {k: (v[:3] if k in cands else v) for k, v in obs.items()}
        p = (SHAPE_PROMPT + "\n\nTOOL: %s | %s\n\nRESULT (shortened to its first rows):\n%s"
             % (tool, dump(self.schemas.get(tool) or {}), dump(clip(sample, 120))[:4000]))
        a = self.ask_json(p, [LIGHT, HEAVY], 2500) or {}
        rk = a.get("rows")
        if not isinstance(rk, str) or rk not in cands:
            return None
        first = obs[rk][0]
        props = (self.schemas.get(tool) or {}).get("properties")
        props = props if isinstance(props, dict) else {}

        def key_in(v, where):
            return v if isinstance(v, str) and v in where else None
        sh = {"rows": rk, "count": key_in(a.get("count"), obs), "next": key_in(a.get("next"), obs),
              "next_arg": key_in(a.get("next_arg"), props), "price": key_in(a.get("price"), first),
              "stock": key_in(a.get("stock"), first)}
        if sh["count"] is not None:
            c = obs[sh["count"]]
            if not isinstance(c, int) or isinstance(c, bool):
                sh["count"] = None
        if not sh["next_arg"] and sh["next"] and obs.get(sh["next"]) is not None:
            sh["next_arg"] = self.page_param(tool, obs[sh["next"]])
        self.shapes[tool] = sh
        if self.price_key is None:
            if sh["price"]:
                self.price_key = sh["price"]
            else:
                mk = self.money_key(obs, sh)
                if mk:
                    self.price_key = mk
        if self.stock_key is None and sh["stock"]:
            self.stock_key = sh["stock"]
        self.trace.append({"kind": "shape", "tool": tool, "shape": sh})
        return sh

    def paging_plan(self, tool, obs):
        sh = self.read_shape(tool, obs)
        if not sh or not sh.get("count") or not sh.get("next") or not sh.get("next_arg") or not isinstance(obs, dict):
            return None
        rows = obs.get(sh["rows"])
        if not isinstance(rows, list) or not rows:
            return None
        c = obs.get(sh["count"])
        if not isinstance(c, int) or isinstance(c, bool):
            return None
        nx = obs.get(sh["next"])
        if not (nx is None or isinstance(nx, str) or (isinstance(nx, int) and not isinstance(nx, bool))):
            return None
        return {"total": c, "shown": len(rows), "arg": sh["next_arg"], "next": nx}

    def price_sorted(self, obs):
        ps = [self.price(lid) for lid, _ in self.obs_listings(obs)]
        if len(ps) < 3 or any(p is None for p in ps):
            return False
        return all(ps[i] <= ps[i + 1] for i in range(len(ps) - 1))

    def auto_page(self, tool, args, obs):
        self.paged = []
        if self.pick() == "any":
            return self.next_big_page(tool, args, obs)
        plan = self.paging_plan(tool, obs)
        if not plan:
            return ""
        sent = args.get(plan["arg"])
        if isinstance(sent, int) and not isinstance(sent, bool):
            later = sent not in (0, 1)
        else:
            later = bool(sent)
        if plan["next"] is None or later or plan["total"] <= plan["shown"]:
            return ""
        if self.pick() == "cheapest" and self.price_sorted(obs):
            return ""
        need = -(-(plan["total"] - plan["shown"]) // plan["shown"])
        L, r = self.steps_left(), self.reserve()
        if need > 2 and self.pages_read + 2 <= AUTO_PAGES and L - r - 2 >= 12:
            need = 2
        if need > 2 or self.pages_read + need > AUTO_PAGES or L - r - need < (LAST_PAGE_ROOM if need == 1 else 12):
            return ("[Cairn: this search matched %d listings but displayed only %d, and not sorted by price. The pick "
                    "weighs every listing that fits: search again with tighter words, namely the value the requirement "
                    "names, or request the next page by sending %s set to %s.]"
                    % (plan["total"], plan["shown"], plan["arg"], dump(plan["next"])))
        lines = []
        nxt = plan["next"]
        got = 0
        for _ in range(need):
            a2 = dict(args)
            a2[plan["arg"]] = nxt
            faults = []
            a2 = shape(a2, self.schemas.get(tool), "args", faults)
            if faults:
                break
            res = self.env(tool, a2)
            if res is None or self.finished:
                break
            self.pages_read += 1
            got += 1
            o2 = res.get("observation")
            self.observe(tool, o2)
            self.paged.append(o2)
            rend = self.render(o2)
            self.read_cache[mark_of(tool, a2)] = (self.version, self.step, rend)
            lines.append("%s=%s: %s" % (plan["arg"], dump(nxt), rend))
            p2 = self.paging_plan(tool, o2)
            nxt = p2["next"] if p2 else None
            if nxt is None:
                break
        self.trace.append({"kind": "auto_pages", "pages": got})
        if not got:
            return ""
        out = "[Cairn fetched the following pages of this search by itself, spending one step per page]\n" + "\n".join(lines)
        if nxt is not None:
            out += ("\n[Cairn: the search matched %d listings and Cairn has now read %d rows. Make the search tighter, "
                    "using the value the requirement names, or keep requesting pages, so that the remaining listings come "
                    "into view.]" % (plan["total"], plan["shown"] * (got + 1)))
        return out

    def cap_ignored(self, tool, args, obs):
        props = (self.schemas.get(tool) or {}).get("properties")
        props = props if isinstance(props, dict) else {}
        rank = self.cap_args(tool)
        caps = []
        for k, v in args.items():
            if k not in props or not _num(v):
                continue
            p = props.get(k) if isinstance(props.get(k), dict) else {}
            priced = bool(RX_PRICED.search(str(k) + " " + str(p.get("description") or "")))
            if (RX_CAP_WORD.search(str(k)) and priced) or (len(rank) == 1 and rank[0] == k):
                caps.append((v, k))
        if not caps or not self.price_key:
            return ""
        cap, name = caps[0]
        for c in caps[1:]:
            if c[0] < cap:
                cap, name = c
        over = [lid for lid, _ in self.obs_listings(obs) if (self.price(lid) or 0) > cap]
        if not over:
            return ""
        hints = []
        for k, v in args.items():
            p = props.get(k)
            if isinstance(v, str) and isinstance(p, dict) and isinstance(p.get("enum"), list):
                others = [o for o in p["enum"] if o != v]
                if others:
                    hints.append("%s=%s" % (k, dump(others[0])))
        note = "[Cairn: this search did not honour %s=%s; %d of its rows are priced above it." % (name, gnum(cap), len(over))
        if hints:
            note += (" Browse the category with %s, keeping the same %s, so that only the cheaper listings show."
                     % (", ".join(hints), name))
        return note + "]"

    def cheapest_here(self, obs):
        if self.pick() != "cheapest" or not self.price_key:
            return ""
        ids = []
        for o in [obs] + list(self.paged):
            for lid, _ in self.obs_listings(o):
                if lid not in ids:
                    ids.append(lid)
        ok = [(self.price(l), l) for l in ids if self.buyable(l) and self.price(l) is not None]
        if len(ok) < 2:
            return ""
        ok.sort(key=lambda p: (p[0], p[1]))
        return ("[Cairn: buyable rows here from the lowest price up: "
                + ", ".join("%s (%s)" % (l, gnum(p)) for p, l in ok[:6]) + "]")

    def wider_hold(self):
        if self.wider_held or self.pick() != "cheapest" or self.steps_left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = []
        for rid in self.cart_ids():
            p = self.price(rid)
            if p is not None:
                lines.append("%s at %s" % (rid, gnum(p)))
        return (HOLD + "%d steps remain and the pick is cheapest. A search may match just a portion of a listing's text, so "
                "searching by a requirement's value surfaces only part of the listings that fit, and rows arrive unordered "
                "by price. Ahead of any checkout, for each item run a few extra searches under different product names: "
                "the product's bare one-word name, its synonyms, and words lifted from the titles of fitting listings "
                "already seen. Consider only rows that can be bought and cost less than that item's current cart line%s. "
                "Then check the attribute records of the lowest-priced of those rows, replace the cart line with any one "
                "that fits, and then check out." % (self.steps_left(), (" (" + "; ".join(lines) + ")") if lines else ""))

    def fact_note(self, tool, has_attrs):
        if has_attrs:
            self.facts[tool] = True
        elif self.facts.get(tool) is not True:
            self.bare[tool] = self.bare.get(tool, 0) + 1
            if self.bare[tool] >= 2:
                self.facts[tool] = False

    def lists_things(self, tool):
        t = self.single_target(tool)
        if t:
            u = under_form(t)
            if RX_SHELF_ARG.search(u) or RX_QUERY_ARG.search(u):
                return True
        return tool in self.shapes


RX_RUN4 = re.compile(r"[a-z]{4,}")
RX_OR_WORD = re.compile(r"\bor\b", re.I)
RX_CHOICE_CUT = re.compile(r"\s*,\s*|\s+or\s+", re.I)
RX_NUM_UNIT = re.compile(r"^(\d[\d.]*)\s*([a-zA-Z]{2,4})$")
RX_LETTERS_NUM = re.compile(r"^[a-zA-Z]+\s+(\d{3,})$")
RX_NAME_WORD = re.compile(r"[a-z][a-z'\-]{2,}")


class E3(object):
    def read_full(self, rid):
        return rid in self.full

    def fill_facts(self, ids, reserve):
        texts = []
        ft = self.facts_tool()
        tgt = self.single_target(ft) if ft else None
        cmp_ = self.roles.get("compare")
        if (not ft or not tgt) and self.compare_facts() is not True:
            return []
        wanted = [i for i in ids if i in self.rows and not self.read_full(i) and not self.sold_out(i)]
        unread = []
        larg = self.compare_target(cmp_) if cmp_ else None
        while (cmp_ and larg and self.compare_facts() is True and len(wanted) >= 2 and self.steps_left() > reserve
               and not self.finished and not self.rushed()):
            room = self.batch_room()
            batch, wanted = wanted[:room], wanted[room:]
            faults = []
            a = shape({larg: batch}, self.schemas.get(cmp_), "args", faults)
            if faults:
                wanted = batch + wanted
                break
            res = self.env(cmp_, a)
            if res is None:
                wanted = batch + wanted
                break
            obs = res.get("observation")
            self.observe(cmp_, obs)
            self.remember_view(cmp_, obs)
            recs = self.obs_listings(obs)
            for lid, rec in recs:
                texts.append("%s: %s" % (lid, self.bounded_show(rec, 1500)))
            if recs:
                self.cmp_ok = max(self.cmp_ok, len(recs))
                unread += [i for i in batch if not self.read_full(i)]
            else:
                self.cmp_fail = min(self.cmp_fail or 999, len(batch))
                wanted = batch + wanted
                if self.batch_room() >= len(batch):
                    break
        if ft and tgt:
            for i in unread + wanted:
                if self.steps_left() <= reserve or self.finished or self.facts.get(ft) is False or self.rushed():
                    break
                if (ft, i) in self.factless:
                    continue
                faults = []
                a = shape({tgt: i}, self.schemas.get(ft), "args", faults)
                if faults:
                    break
                res = self.env(ft, a)
                if res is None:
                    break
                obs = res.get("observation")
                self.observe(ft, obs)
                self.remember_view(ft, obs)
                if isinstance(obs, dict) and self.listing_id(obs, False, 0):
                    self.fact_note(ft, attrs_of(obs) is not None)
                texts.append("%s: %s" % (i, self.bounded_show(obs, 1500)))
        if texts:
            self.trace.append({"kind": "fill_facts", "texts": len(texts)})
        return texts

    def show_links(self, ids, reserve):
        vt = self.value_template()
        if not vt or (len(self.items) < 2 and len(ids) < 2):
            return
        for i in ids:
            fi = self.fit_index()
            code = (fi.get(i) or {}).get("pointer")
            if not code or code in self.link_tried or any(m.get("own") == code for m in fi.values()):
                continue
            if self.steps_left() <= reserve or self.finished or self.rushed():
                return
            self.link_tried.add(code)
            tool, fixed, field = vt
            a = dict(fixed)
            a[field] = code
            faults = []
            a = shape(a, self.schemas.get(tool), "args", faults)
            if faults:
                return
            res = self.env(tool, a)
            if res is None:
                return
            self.observe(tool, res.get("observation"))
            self.trace.append({"kind": "link", "code": code})
            fi = self.fit_index()
            self.fill_facts([lid for lid, m in fi.items() if m.get("own") == code], max(reserve, self.reserve() + 6))

    def _props(self, tool):
        p = (self.schemas.get(tool) or {}).get("properties")
        return p if isinstance(p, dict) else {}

    def cap_args(self, tool):
        props = self._props(tool)
        cands = [k for k, p in props.items() if isinstance(p, dict) and ("integer" in kinds(p) or "number" in kinds(p))
                 and not RX_NON_PRICE.search(str(k)) and not RX_LOW_ARG.search(under_form(k))]
        if not cands:
            return []
        ev = self.ceiling_evidence(tool)
        proved = [k for k in cands if any(v == "holds" and a == k for (a, m), v in ev.items())]
        maxed = [k for k in cands if RX_CAP_ARG.search(under_form(k))]
        money = [k for k in cands if RX_PRICE_NAME.search(under_form(k))]
        for grp in (proved, [k for k in maxed if k in money], money, maxed):
            if grp:
                return grp
        return []

    def _mode(self, tool, args):
        props = self._props(tool)
        out = []
        for k, v in args.items():
            p = props.get(k)
            if isinstance(p, dict) and isinstance(p.get("enum"), list) and isinstance(v, (str, int, float, bool)):
                out.append((k, v))
        return tuple(sorted(out, key=lambda x: str(x[0])))

    def ceiling_evidence(self, tool):
        ck = self.category_key()
        pools = {}
        for rid in self.seen:
            p = self.price(rid)
            if p is not None:
                lab = str(self.rows[rid].get(ck, "")) if ck else ""
                pools.setdefault(lab, []).append(p)
        props = self._props(tool)
        out = {}
        for cm, rec in list(self.remembered.items()):
            t, args, res = rec
            if t != tool:
                continue
            recs = self.obs_listings(res)
            if not recs:
                continue
            prices = [self.price(lid) for lid, _ in recs]
            if any(p is None for p in prices):
                continue
            mode = self._mode(tool, args)
            pool = []
            if ck:
                for lab in set(str(self.rows.get(lid, {}).get(ck, "")) for lid, _ in recs):
                    pool.extend(pools.get(lab, []))
            dearest = max(prices)
            for k, v in args.items():
                if k not in props or not _num(v):
                    continue
                key = (k, mode)
                if dearest > v:
                    out[key] = "fails"
                elif out.get(key) != "fails" and pool:
                    if (sum(1 for p in pool if p <= v) / float(len(pool))) ** len(recs) < PRICE_LIMIT_CHANCE:
                        out[key] = "holds"
        return out

    def _worded(self, args, props, shelf):
        for k, v in args.items():
            if k == shelf:
                continue
            p = props.get(k)
            if isinstance(p, dict) and isinstance(p.get("enum"), list):
                continue
            if isinstance(v, str) and v.strip() and RX_QUERY_ARG.search(under_form(k)):
                return True
        return False

    def capped_template(self):
        for t in self.tools:
            tool = t["function"]["name"]
            props = self._props(tool)
            caps = self.cap_args(tool)
            shelves = [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum")
                       and RX_SHELF_ARG.search(under_form(k))]
            if len(caps) != 1 or len(shelves) != 1:
                continue
            cap, shelf = caps[0], shelves[0]
            ev = self.ceiling_evidence(tool)
            learned = [(dict(m), v) for (a, m), v in ev.items() if a == cap]
            opts = [k for k, p in props.items() if isinstance(p, dict) and isinstance(p.get("enum"), list) and len(p["enum"]) >= 2]
            req = (self.schemas.get(tool) or {}).get("required")
            req = req if isinstance(req, list) else []
            single = len(opts) == 1 and opts[0] in req
            chosen = {}
            for k in opts:
                en = props[k]["enum"]
                val = None
                for m, v in learned:
                    if v == "holds" and k in m:
                        val = m[k]
                        break
                if val is None:
                    for args in reversed(self.searched):
                        if shelf in args and k in args and not self._worded(args, props, shelf):
                            val = args[k]
                            break
                if val is None:
                    lm = [o for o in en if isinstance(o, str) and RX_LIST_MODE.search(o)]
                    if len(lm) == 1:
                        val = lm[0]
                if val is None and single:
                    worded = [args[k] for args in self.searched if k in args and self._worded(args, props, shelf)]
                    if worded:
                        never = [o for o in en if o not in worded]
                        if len(never) == 1:
                            val = never[0]
                if single:
                    spoiled = [m[k] for m, v in learned if v == "fails" and k in m]
                    if spoiled and (val is None or val in spoiled):
                        rest = [o for o in en if o not in spoiled]
                        val = rest[0] if len(rest) == 1 else None
                if val is None:
                    return None
                chosen[k] = val
            return (tool, chosen, shelf, cap)
        return None

    def walk_below(self, lines):
        tpl = self.capped_template()
        ck = self.category_key()
        if not tpl or not ck:
            return
        tool, opts, shelf, capa = tpl
        for lid in lines:
            p = self.price(lid)
            lab = (self.rows.get(lid) or {}).get(ck)
            if p is None or p < 2 or not lab or (lid, p) in self.capped:
                continue
            self.capped.add((lid, p))
            ceil = int(p) - 1 if p == int(p) else int(p)
            used = set()
            for n in range(WALK_PER_LINE):
                if self.steps_left() <= self.reserve() + (7 if n == 0 else WALK_RESERVE) or self.finished or self.rushed():
                    return
                if ceil < 1 or ceil in used:
                    break
                used.add(ceil)
                a = dict(opts)
                a[shelf] = lab
                a[capa] = ceil
                faults = []
                a = shape(a, self.schemas.get(tool), "args", faults)
                if faults:
                    return
                res = self.env(tool, a)
                if res is None:
                    return
                obs = res.get("observation")
                self.observe(tool, obs)
                recs = self.obs_listings(obs)
                self.trace.append({"kind": "walk", "line": lid, "price": p, "ceiling": ceil, "rows": len(recs)})
                plan = self.paging_plan(tool, obs)
                if (not recs or any((self.price(r) or 0) > ceil for r, _ in recs) or not plan
                        or plan["total"] <= plan["shown"] or self.price_sorted(obs)):
                    break
                if plan["total"] <= 2 * plan["shown"]:
                    if plan["next"] is not None and plan["next"] != "" and self.steps_left() > self.reserve() + 7 and not self.rushed():
                        a2 = dict(a)
                        a2[plan["arg"]] = plan["next"]
                        f2 = []
                        a2 = shape(a2, self.schemas.get(tool), "args", f2)
                        if not f2:
                            r2 = self.env(tool, a2)
                            if r2 is not None:
                                self.observe(tool, r2.get("observation"))
                    break
                ps = [self.price(r) for r, _ in recs if self.price(r) is not None]
                if not ps:
                    break
                m = min(ps)
                ceil = int(m) - 1 if m == int(m) else int(m)

    def _below_pool(self, lid, p, lab, lines):
        ck = self.category_key()
        key = str(lab).strip().lower()
        pool = []
        for rid in self.seen:
            if rid in lines or self.read_full(rid) or not self.buyable(rid):
                continue
            if str((self.rows[rid] or {}).get(ck, "")).strip().lower() != key:
                continue
            q = self.price(rid)
            if q is not None and q < p:
                pool.append(rid)
        return pool

    def cheaper_rows(self):
        if self.pick() != "cheapest":
            return []
        mk = self.cart_mark() + "#" + str(self.version)
        ck = self.category_key()
        if not ck or mk in self.cheaper_marks:
            return []
        self.cheaper_marks.add(mk)
        lines = self.cart_ids()
        self.walk_below(lines)
        room = self.batch_room() if self.compare_facts() is True else 0
        picks = []
        for lid in lines:
            p = self.price(lid)
            lab = (self.rows.get(lid) or {}).get(ck)
            if p is None or lab is None or str(lab) == "":
                continue
            pool = sorted(self._below_pool(lid, p, lab, lines), key=lambda r: self.price(r))
            vr = [r for r in pool if r in self.value_rows]
            if vr:
                pool.remove(vr[0])
                pool.insert(0, vr[0])
            k = room or 2
            c = 0
            for r in pool:
                if c >= k:
                    break
                if r not in picks:
                    picks.append(r)
                    c += 1
        cap = room * len(lines) if room else min(6, max(1, 3 * len(lines)))
        return self.fill_facts(picks[:cap], self.reserve() + 6)

    @soft("")
    def below_cart(self):
        ck = self.category_key()
        if not ck:
            return ""
        ents = []
        lines = self.cart_ids()
        for lid in lines:
            p = self.price(lid)
            lab = (self.rows.get(lid) or {}).get(ck)
            if p is None or lab is None or str(lab) == "":
                continue
            pool = sorted(self._below_pool(lid, p, lab, lines), key=lambda r: self.price(r))[:CHEAPER_SHOWN]
            ents += ["%s at %s" % (r, gnum(self.price(r))) for r in pool]
        if not ents:
            return ""
        return "Not read yet and priced below a line of the cart: " + ", ".join(ents) + "."

    def title_key(self):
        ck = self.category_key()
        st = {}
        for rid in self.seen:
            for k, v in self.rows[rid].items():
                if k == ck or not isinstance(v, str) or len(v) > TITLE_LIMIT:
                    continue
                s = st.setdefault(k, [0, 0, 0, set()])
                s[0] += len(v.split())
                s[1] += 1
                s[2] += len(v)
                s[3].add(v)
        n = len(self.rows)
        kept = [k for k, s in st.items() if s[1] * 5 >= n * 4 and s[0] >= 2 * s[1] and len(s[3]) * 2 >= s[1]]
        if not kept:
            return None
        most = max(len(st[k][3]) for k in kept)
        kept = [k for k in kept if len(st[k][3]) * 10 >= most * 9]
        return min(kept, key=lambda k: (st[k][2] / float(st[k][1]), str(k)))

    def hunt_words(self, n):
        it = self.items[n - 1]
        ck = self.category_key()
        tk = self.title_key()
        labels = []
        rids = []
        if ck:
            for rid in self.seen:
                lab = self.rows[rid].get(ck)
                if lab is not None and self.owns_label(n, lab):
                    labels.append(str(lab))
                    rids.append(rid)
        names = []
        if ck and tk:
            names = [str(self.rows[r].get(tk)) for r in rids if self.rows[r].get(tk) is not None]
        if ck:
            for rid in self.seen:
                rec = self.full.get(rid)
                if rec is None:
                    continue
                lab = rec.get(ck)
                if lab is not None and self.owns_label(n, lab):
                    a = attrs_of(rec) or {}
                    names.append(" ".join(v for v in a.values() if isinstance(v, str) and len(v) <= HUNT_VALUE_LEN))
        other = set()
        for j, o in enumerate(self.items, 1):
            if j != n:
                other |= set(RX_RUN4.findall((o.shelf + " " + o.name).lower()))
        if labels:
            cnt = {}
            for l in labels:
                cnt[l] = cnt.get(l, 0) + 1
            top = max(cnt.values())
            home = next(l for l in labels if cnt[l] == top).lower()
        else:
            home = (it.shelf or it.name).lower()
        cands = [w for w in RX_RUN4.findall(home) if w not in other]
        noun = ""
        for w in cands:
            if len(w) > len(noun):
                noun = w
        if n in self.hunt_nonoun:
            noun = ""
        if not names:
            return noun, []
        counts = {}
        for nm in names:
            for w in set(RX_NAME_WORD.findall(nm.lower())):
                counts[w] = counts.get(w, 0) + 1
        own = roots(it.shelf) | roots(it.name)
        ceilc = max(2, int(HUNT_COMMON * len(names)))
        words = [w for w, c in counts.items() if 2 <= c <= ceilc and w != noun and roots(w) and not (roots(w) & own)]
        words.sort(key=lambda w: (-counts[w], w))
        return noun, words

    def item_rows(self, n):
        ck = self.category_key()
        if not ck:
            return len(self.rows)
        return sum(1 for rid in self.seen if self.rows[rid].get(ck) is not None and self.owns_label(n, self.rows[rid].get(ck)))

    @soft("")
    def cheaper_hunt(self):
        mark = dump([it.live() for it in self.items]) + "|" + str(self.budget) + "|" + dump(sorted(self.gone))
        if self.pick() != "cheapest" or self.hunts >= CHEAPER_ROUNDS or mark == self.hunt_mark:
            return ""
        vt = self.value_template()
        if not vt or not self.price_key or self.clock() > HUNT_DEADLINE:
            return ""
        room = min(HUNT_STEP_CAP, self.steps_left() - self.reserve() - HUNT_RESERVE - 2 * len(self.items))
        if room < 3:
            return ""
        self.hunts += 1
        self.hunt_mark = mark
        share = max(3, room // len(self.items))
        spent = gain = 0
        for n in range(1, len(self.items) + 1):
            if spent >= room or self.finished:
                break
            s, g = self.item_hunt(n, vt, min(share, room - spent))
            spent += s
            gain += g
        self.trace.append({"kind": "cheaper_hunt", "steps": spent, "new_rows": gain, "cart": self.cart_mark()})
        if not spent:
            return ""
        self.wider_held = True
        return ("Cairn itself ran %d extra searches chasing a lower price, one step each, and they brought up %d new rows."
                % (spent, gain))

    def item_hunt(self, n, vt, room):
        tool, fixed, field = vt
        sch = self.schemas.get(tool)
        tset = self.hunt_tried.setdefault(n, set())
        noun, _ = self.hunt_words(n)
        sets = self.hunt_sets(n)
        st = {"spent": 0, "gain": 0, "idle": 0, "refused": False}

        def run(q):
            tset.add(q.lower())
            a = dict(fixed)
            a[field] = q
            f = []
            a = shape(a, sch, "args", f)
            if f:
                st["refused"] = True
                return False, False
            if st["spent"] >= room or self.finished:
                return False, False
            holds = any(vs and all(v.lower() in q.lower() for v in vs) for vs in sets)
            found = False
            visited = set()
            while not self.finished and self.clock() <= HUNT_DEADLINE + HUNT_GRACE:
                cm = self.canon(tool, a)
                if cm in visited:
                    break
                visited.add(cm)
                if cm in self.remembered:
                    obs = self.remembered[cm][2]
                    new = HUNT_PAGE_YIELD
                else:
                    if st["spent"] >= room:
                        break
                    before = self.item_rows(n)
                    res = self.env(tool, a)
                    if res is None:
                        break
                    st["spent"] += 1
                    obs = res.get("observation")
                    self.observe(tool, obs)
                    new = self.item_rows(n) - before
                    st["gain"] += new
                    st["idle"] = 0 if new else st["idle"] + 1
                recs = self.obs_listings(obs)
                if recs:
                    found = True
                if holds:
                    self.value_rows |= set(l for l, _ in recs)
                plan = self.paging_plan(tool, obs)
                if not plan or plan["next"] is None or plan["next"] == "" or (new < HUNT_PAGE_YIELD and plan["total"] > HUNT_WHOLE):
                    break
                a2 = dict(a)
                a2[plan["arg"]] = plan["next"]
                f2 = []
                a2 = shape(a2, sch, "args", f2)
                if f2:
                    break
                a = a2
            return True, found
        key = (n, tuple(tuple(x) for x in sets))
        if key not in self.hunt_stems:
            stem = ""
            for vs in sets:
                tries = []
                if vs:
                    for t in (" ".join(vs + [noun]), " ".join(vs)) + ((" ".join(vs[:-1] + [noun]),) if len(vs) >= 2 else ()):
                        t = t.strip()
                        if t and t not in tries:
                            tries.append(t)
                for t in tries:
                    issued, found = run(t)
                    if not issued:
                        return st["spent"], st["gain"]
                    if found:
                        if not stem:
                            stem = t
                        break
            if not stem and noun:
                issued, found = run(noun)
                if not issued:
                    return st["spent"], st["gain"]
                if found:
                    stem = noun
                else:
                    self.hunt_nonoun.add(n)
            self.hunt_stems[key] = stem
        stem = self.hunt_stems[key]
        while (st["spent"] < room and st["idle"] < HUNT_IDLE and not st["refused"] and not self.finished
               and self.clock() <= HUNT_DEADLINE + HUNT_GRACE):
            _, words = self.hunt_words(n)
            sw = set(stem.lower().split())
            words = [w for w in words if (w + " " + stem).strip().lower() not in tset and w not in sw]
            if not words:
                break
            run((words[0] + " " + stem).strip())
        return st["spent"], st["gain"]

    def hunt_sets(self, n):
        stext = " ".join(str(v).lower() for a in self.searched for v in a.values() if isinstance(v, str))
        plain = [v for v, _ in self.value_words(n) if not find_codes(v)]
        reft = set(t for _, t in self.refs)
        choices = []
        for r in self.items[n - 1].live():
            if choices:
                break
            if ":" not in r or r in reft:
                continue
            v = r.split(":", 1)[1].strip()
            if not RX_OR_WORD.search(v) or RX_OPEN_EXCLUDE.search(RX_OR_WORD.sub(" ", v)):
                continue
            parts = [p.strip(" .") for p in RX_CHOICE_CUT.split(v)]
            parts = [p for p in parts if p]
            if len(parts) < 2:
                continue
            choices = [p for p in parts if not find_codes(p) and len(p) <= HUNT_VALUE_LEN][:HUNT_CHOICES]
        raw = [plain + [c] for c in choices] or [list(plain)]
        out = []
        for vs in raw:
            other = []
            changed = False
            ok = False
            for v in vs:
                t = v.strip()
                if RX_UNIT_VALUE.match(t):
                    m = RX_NUM_UNIT.match(t)
                    o = (m.group(1) + m.group(2)) if " " in t else (m.group(1) + " " + m.group(2))
                    other.append(o)
                    if o != v:
                        changed = True
                        if v.lower() in stext and o.lower() not in stext:
                            ok = True
                else:
                    other.append(v)
            res = other if (changed and ok) else vs
            if res not in out:
                out.append(res)
        bare = []
        for v in out[0]:
            t = v.strip()
            m = RX_LETTERS_NUM.match(t)
            bare.append(m.group(1) if m else t)
        if bare != out[0] and bare not in out:
            out.append(bare)
        return out

    @soft(list)
    def fit_hunt(self):
        vt = self.value_template()
        if self.fit_hunted or self.clock() > HUNT_DEADLINE or not vt:
            return []
        res_ = self.reserve() + len(self.items) + 1
        spare = self.steps_left() - res_
        if spare < 3:
            return []
        self.fit_hunted = True
        ck = self.category_key()
        covered = set()
        if ck:
            for i, u in self.held_units().items():
                if u >= 1:
                    lab = (self.rows.get(i) or {}).get(ck)
                    if lab is not None:
                        covered.add(lab)
        N = len(self.items)
        hunt_items = [n for n in range(1, N + 1) if not any(self.owns_label(n, l) for l in covered)] or list(range(1, N + 1))
        wanted = {}
        for n in hunt_items:
            toks = []
            for v, _ in self.value_words(n):
                toks.extend(value_tokens(v))
            wanted[n] = toks
        tool, fixed, field = vt
        walked = self.unread_walk(self.roles.get("search") or tool, wanted, min(FIT_HUNT_PAGES, spare // 2))
        sroom = min(HUNT_STEP_CAP, self.steps_left() - self.reserve() - HUNT_RESERVE)
        total = 0
        if sroom >= 3:
            for n in hunt_items:
                vals = [v for v, _ in self.value_words(n)]
                noun, words = self.hunt_words(n)
                qs = []

                def addq(q):
                    q = q.strip()
                    if q and q not in qs:
                        qs.append(q)
                for ser in self.fit_series(n)[:SERIES_HUNTED]:
                    addq(ser + " " + (vals[0] if vals else "") + " " + noun)
                for v in vals:
                    addq(v + " " + noun)
                if noun:
                    for v in vals:
                        addq(v)
                for w in words[:FIT_HUNT_WORDS]:
                    addq(w + " " + noun)
                if noun:
                    addq(noun)
                share = max(2, sroom // len(hunt_items))
                used = 0
                for q in qs:
                    if total >= sroom or used >= share or self.finished or self.clock() > HUNT_DEADLINE + HUNT_GRACE:
                        break
                    a = dict(fixed)
                    a[field] = q
                    f = []
                    a = shape(a, self.schemas.get(tool), "args", f)
                    if f:
                        continue
                    r = self.env(tool, a)
                    if r is None:
                        return []
                    total += 1
                    used += 1
                    self.observe(tool, r.get("observation"))
        texts = []
        for n in hunt_items:
            toks = set(wanted[n])
            pool = []
            for rid in self.seen:
                if rid in self.full or not self.buyable(rid):
                    continue
                if ck and not self.owns_label(n, self.rows[rid].get(ck)):
                    continue
                if self.budget is not None and (self.price(rid) or 0) > self.budget:
                    continue
                pool.append(rid)

            def sc(rid):
                tw = text_words(dump(self.rows[rid]))
                p = self.price(rid)
                return (-sum(1 for t in toks if t in tw), p is None, p if p is not None else 0)
            picks = sorted(pool, key=sc)[:FIT_HUNT_OPENS]
            for i in range(0, len(picks), 4):
                if self.clock() > HUNT_DEADLINE + HUNT_GRACE:
                    break
                texts += self.fill_facts(picks[i:i + 4], res_)
        self.trace.append({"kind": "fit_hunt", "pages": walked, "searches": total, "opened": len(texts)})
        return texts

    def unread_walk(self, tool, wanted, room):
        items = list(wanted.keys())
        if not items or room <= 0:
            return 0
        ck = self.category_key()
        share = max(1, room // len(items))
        spent = 0
        visited = set()
        for limit in (share, room):
            for n in items:
                stop = min(room, spent + limit)
                toks = set(wanted[n])
                for cm, rec in list(self.remembered.items()):
                    if spent >= stop or self.finished:
                        break
                    t, args, res = rec
                    if t != tool:
                        continue
                    txt = " ".join(str(v).lower() for v in args.values() if isinstance(v, str))
                    if not (text_words(txt) & toks):
                        continue
                    owners = set(self.label_owner((self.rows.get(lid) or {}).get(ck) if ck else None)
                                 for lid, _ in self.obs_listings(res))
                    if owners and None not in owners and n not in owners:
                        continue
                    a, obs = args, res
                    while spent < stop and not self.finished:
                        plan = self.paging_plan(t, obs)
                        if not plan or plan["next"] is None or plan["next"] == "":
                            break
                        a2 = dict(a)
                        a2[plan["arg"]] = plan["next"]
                        f = []
                        a2 = shape(a2, self.schemas.get(t), "args", f)
                        if f:
                            break
                        c2 = self.canon(t, a2)
                        if c2 in visited:
                            break
                        visited.add(c2)
                        if c2 in self.remembered:
                            obs = self.remembered[c2][2]
                            a = a2
                            continue
                        r = self.env(t, a2)
                        if r is None:
                            return spent
                        spent += 1
                        obs = r.get("observation")
                        self.observe(t, obs)
                        a = a2
        return spent


RX_WS = re.compile(r"\s+")


def _whole(v):
    return isinstance(v, int) and not isinstance(v, bool)


class E4(object):
    def canon(self, tool, args):
        props = self._props(tool)
        kept = {}
        for k in sorted(args.keys(), key=str):
            v = args[k]
            p = props.get(k) if isinstance(props.get(k), dict) else {}
            if _whole(v):
                mn, df = p.get("minimum"), p.get("default")
                if (_num(mn) and v == mn) or (_num(df) and v == df):
                    continue
                if v == 1 and self.page_arg.get(tool) == k and "minimum" not in p and "default" not in p:
                    continue
            if isinstance(v, str):
                v = RX_WS.sub(" ", v.lower()).strip()
            kept[k] = v
        return str(tool) + dump(kept)

    def note_read(self, tool, args, obs):
        skip = [self.roles.get("offer") or self.roles.get("open")]
        for r in "compare ask cart cart_view buy refuse".split():
            skip.append(self.roles.get(r))
        if tool in skip:
            return
        self.is_new_query(tool, args)
        if isinstance(obs, (dict, list)) and not is_failure(obs):
            self.remembered[self.canon(tool, args)] = (tool, dict(args), obs)

    def is_new_query(self, tool, args):
        kind = dump({k: v for k, v in args.items() if not _whole(v)})
        mem = self.page_queries.setdefault(tool, set())
        if kind in mem:
            return False
        mem.add(kind)
        return True

    def _low(self, tool, k):
        p = self._props(tool).get(k)
        mn = p.get("minimum") if isinstance(p, dict) else None
        return mn if _whole(mn) else 1

    @soft("")
    def late_page(self, tool, args, obs):
        if tool in self.page_arg or not isinstance(obs, dict):
            return ""
        props = self._props(tool)
        cands = []
        for k, v in args.items():
            if _whole(v) and k in props and v > self._low(tool, k):
                cands.append(k)
        if len(cands) != 1:
            return ""
        if self.steps_left() <= self.reserve() + 4:
            return ""
        sh = self.read_shape(tool, obs)
        if not sh or not sh.get("count") or not sh.get("next") or obs.get(sh["next"]) is not None:
            return ""
        rows = obs.get(sh["rows"])
        c = obs.get(sh["count"])
        if not isinstance(rows, list) or not _whole(c) or c <= len(rows):
            return ""
        k = cands[0]
        lo = self._low(tool, k)
        a = dict(args)
        a[k] = lo
        f = []
        a = shape(a, self.schemas.get(tool), "args", f)
        if f:
            return ""
        res = self.env(tool, a)
        if res is None:
            return ""
        o2 = res.get("observation")
        self.observe(tool, o2)
        r2 = o2.get(sh["rows"]) if isinstance(o2, dict) else None
        if not isinstance(r2, list) or len(r2) < len(rows):
            return ""
        first_ids = set(l for l, _ in self.obs_listings(obs))
        if any(l in first_ids for l, _ in self.obs_listings(o2)):
            return ""
        self.page_arg[tool] = k
        return ("\n[Cairn] %s is this search's page number, and the call just made displayed its final page. Page %s of "
                "the same search:\n%s" % (k, dump(lo), self.render(o2)))

    @soft(False)
    def names_code(self, args):
        said = set()
        for s in strings_of(args):
            said |= set(find_codes(s))
        if not said:
            return False
        known = set()
        for m in self.fit_index().values():
            known |= set(m["models"]) | set(m["works"])
        return bool(said & known)

    def first_look(self, tool, args, fresh):
        k = self.page_arg.get(tool)
        if not k or not fresh:
            return args
        v = args.get(k)
        if _whole(v):
            lo = self._low(tool, k)
            if v != lo:
                a = dict(args)
                a[k] = lo
                return a
        return args

    @soft("")
    def shorter(self, tool, args, obs):
        vt = self.value_template()
        if not vt or vt[0] != tool:
            return ""
        if self.steps_left() <= self.reserve() + SHORTER_RESERVE or self.finished or not isinstance(obs, dict):
            return ""
        field = vt[2]
        q = args.get(field)
        if not isinstance(q, str) or len(q.split()) < 3 or self.obs_listings(obs):
            return ""
        for v in obs.values():
            if isinstance(v, (list, str)) and v:
                return ""
            if _num(v) and v > 0:
                return ""
        short = " ".join([w for w in q.split() if not find_codes(w)][:2])
        if not short:
            return ""
        a = {k: v for k, v in args.items() if not _whole(v)}
        a[field] = short
        f = []
        a = shape(a, self.schemas.get(tool), "args", f)
        if f:
            return ""
        res = self.env(tool, a)
        if res is None:
            return ""
        o2 = res.get("observation")
        self.observe(tool, o2)
        return ("\n[Cairn] No listing carries every one of those words; the identical search with the short words '%s' "
                "gives:\n%s" % (short, self.render(o2)))

    def record_search(self, args):
        self.searched.append(dict(args))
        if len(self.searched) > 8:
            self.searched = self.searched[-8:]

    @soft("")
    def opening_moves(self):
        vt = self.value_template()
        if not vt or self.budget_steps < OPEN_MIN_STEPS or len(self.items) < OPEN_MIN_ITEMS:
            return ""
        self.gap = OPENING_PAUSE
        stop = self.budget_steps - self.reserve() - self.msgs_free() - OPEN_LATER_STEPS * len(self.items) - HUNT_RESERVE // 2
        tool, fixed, field = vt
        sch = self.schemas.get(tool)
        lines = []
        vsearches = []
        reftexts = set(t for _, t in self.refs)
        for n, it in enumerate(self.items, 1):
            vals = []
            for r in it.musts:
                if ":" not in r or r in reftexts:
                    continue
                v = r.split(":", 1)[1].strip()
                if v and len(v.split()) <= 3 and not RX_OPEN_EXCLUDE.search(v):
                    vals.append(v)
            w = it.name.split()
            noun = it.name if len(w) <= 3 else " ".join(w[-2:])
            tries = []
            if vals:
                tries.append(" ".join(vals[:2] + [noun]))
            if noun not in tries:
                tries.append(noun)
            singles = [vals[0] + " " + noun, vals[1] + " " + noun] if len(vals) >= 2 else []
            while tries:
                t = tries.pop(0)
                a = dict(fixed)
                a[field] = t
                f = []
                a = shape(a, sch, "args", f)
                if f or self.finished or self.step >= stop:
                    break
                res = self.env(tool, a)
                if res is None:
                    return "\n".join(lines)
                obs = res.get("observation")
                self.observe(tool, obs)
                if self.obs_listings(obs):
                    self.record_search(a)
                    lines.append("%s: %s" % (t, self.render(obs)))
                    if vals and t != noun:
                        org = self.item_origins.setdefault(n, [])
                        for l, _ in self.obs_listings(obs):
                            if l not in org:
                                org.append(l)
                        vsearches.append((n, a, obs))
                    plan = self.paging_plan(tool, obs)
                    if (plan and plan["next"] is not None and plan["next"] != "" and self.item_rows(n) < OPEN_ROWS_PER_ITEM
                            and not self.finished and self.step < stop):
                        a2 = dict(a)
                        a2[plan["arg"]] = plan["next"]
                        f2 = []
                        a2 = shape(a2, sch, "args", f2)
                        if not f2:
                            r2 = self.env(tool, a2)
                            if r2 is not None:
                                o2 = r2.get("observation")
                                self.observe(tool, o2)
                                lines.append("next page: %s" % self.render(o2))
                    tries = [x for x in tries if x not in singles]
                    if self.item_rows(n) >= OPEN_ROWS_PER_ITEM:
                        break
                else:
                    fo = is_failure(obs)
                    lines.append("%s: %s" % (t, ("the store rejected the call: " + self.render(obs)) if fo else "no rows"))
                    if singles and t not in singles and t != noun and not fo:
                        tries = [x for x in singles if x not in tries] + tries
        lines += self.value_pages(tool, vsearches, stop)
        if self.rows and self.roles.get("ask") and self.msgs_free() > 0:
            found = [n for n in range(1, len(self.items) + 1) if self.item_rows(n) > 0]
            s0 = self.step
            while self.step - s0 < self.budget_steps // OPEN_WIDEN_SHARE and self.step < stop and not self.finished:
                if all(len(self.item_records(n)) >= OPEN_ROWS_PER_ITEM for n in found):
                    break
                before = len(self.full)
                self.widen()
                if len(self.full) <= before:
                    break
        self.gap = STORE_PAUSE
        sw = self.series_sweep()
        for rid in list(self.full.keys()):
            lines.append("read in full: %s" % self.page_text(rid))
        if sw:
            lines.append(sw)
        ev = self.evidence(list(self.full.keys()))
        if ev:
            lines.append(ev)
        return "\n".join(lines)

    def value_pages(self, tool, vsearches, stop):
        big = self.budget_steps >= BIG_STEPS
        if (self.pick() != "cheapest" and not big) or self.swept:
            return []
        ck = self.category_key()
        qual = []
        for n, a, obs in vsearches:
            plan = self.paging_plan(tool, obs)
            if not plan or plan["next"] is None or not (SWEEP_MIN <= plan["total"] <= SWEEP_MAX):
                continue
            if self.pick() == "cheapest" and self.price_sorted(obs):
                continue
            recs = self.obs_listings(obs)
            own = sum(1 for _, r in recs if ck and self.label_owner(r.get(ck)) == n)
            if own * 2 <= len(recs):
                continue
            qual.append((n, a, plan))
        if not qual:
            return []
        self.swept = True
        qual = sorted(qual, key=lambda q: -q[2]["total"])[:max(VS_SWEEPS, len(self.items)) if big else VS_SWEEPS]
        lines = []
        for n, a, plan in qual:
            lines += self.vs_sweep(tool, n, a, plan, stop)
        return lines

    def vs_sweep(self, tool, n, a, plan, stop):
        lines = []
        got = 0
        org = self.item_origins.setdefault(n, [])
        while plan and plan["next"] is not None and got < VS_PAGES and stop - self.step >= SWEEP_KEEP and not self.finished:
            a2 = dict(a)
            a2[plan["arg"]] = plan["next"]
            f = []
            a2 = shape(a2, self.schemas.get(tool), "args", f)
            if f:
                break
            cm = self.canon(tool, a2)
            if cm in self.remembered:
                obs = self.remembered[cm][2]
            else:
                res = self.env(tool, a2)
                if res is None:
                    break
                obs = res.get("observation")
                self.observe(tool, obs)
                self.pages_read += 1
                got += 1
                lines.append("next page: %s" % self.render(obs))
            for l, _ in self.obs_listings(obs):
                if l not in org:
                    org.append(l)
            a = a2
            plan = self.paging_plan(tool, obs)
        if lines:
            self.trace.append({"kind": "value_pages", "item": n, "pages": got})
        return lines

    def page_text(self, rid):
        rec = self.full.get(rid) or self.rows.get(rid) or {}
        a = attrs_of(rec) or {}
        rest = {k: v for k, v in rec.items() if not isinstance(v, (dict, list))}
        return "%s: attributes %s | %s" % (rid, dump(clip(a, 120))[:PAGE_ATTR_CLIP], dump(clip(rest, 200))[:PAGE_REST_CLIP])

    def observe(self, tool, obs):
        self.absorb(obs)
        lines = self.loose_lines(obs)
        if not lines:
            return ""
        meaning = self.read_note(lines)
        if meaning:
            self.queued.append("(seen on %s) %s" % (tool, meaning))
        return meaning

    def compare_target(self, cmp_):
        props = self._props(cmp_)
        tgt = next((k for k, p in props.items() if isinstance(p, dict) and "array" in kinds(p)), None)
        req = (self.schemas.get(cmp_) or {}).get("required")
        req = req if isinstance(req, list) else []
        if tgt is None or any(r != tgt for r in req):
            return None
        return tgt

    def widen(self):
        cmp_ = self.roles.get("compare")
        tgt = self.compare_target(cmp_) if cmp_ else None
        if not cmp_ or not tgt or self.compare_bare():
            return self.widen_singly()
        if self.steps_left() <= 12:
            return None
        mx = self._props(cmp_)[tgt].get("maxItems")
        cap = min(int(mx) if _num(mx) else 4, self.batch_room())
        groups = sorted(self.item_groups(), key=lambda g: sum(1 for r in g[1] if r in self.full))
        lanes = [[r for r in g[1] if r not in self.full] for g in groups]
        picks = []
        for rank in range(cap):
            for ln in lanes:
                if rank < len(ln):
                    picks.append(ln[rank])
        picks = picks[:cap]
        if len(picks) < 2:
            self.fill_facts(picks, 12)
            return None
        f = []
        a = shape({tgt: picks}, self.schemas.get(cmp_), "args", f)
        if f:
            return None
        before = len(self.full)
        res = self.env(cmp_, a)
        if res is not None:
            obs = res.get("observation")
            self.observe(cmp_, obs)
            self.remember_view(cmp_, obs)
            recs = self.obs_listings(obs)
            if recs:
                self.facts[cmp_] = self.facts.get(cmp_) is True or any(attrs_of(r) is not None for _, r in recs)
                self.cmp_ok = max(self.cmp_ok, len(recs))
                if self.facts[cmp_] is False:
                    return self.widen_singly()
            else:
                self.cmp_fail = min(self.cmp_fail or 999, len(picks))
        if len(self.full) <= before and len(picks) > 2 and self.steps_left() > 12:
            f2 = []
            a2 = shape({tgt: picks[:2]}, self.schemas.get(cmp_), "args", f2)
            if not f2:
                r2 = self.env(cmp_, a2)
                if r2 is not None:
                    o2 = r2.get("observation")
                    self.observe(cmp_, o2)
                    self.cmp_ok = max(self.cmp_ok, len(self.obs_listings(o2)))
        return None

    def widen_singly(self):
        ft = self.facts_tool()
        tgt = self.single_target(ft) if ft else None
        if not ft or not tgt:
            return None
        words = {}
        allw = set()
        for n, it in enumerate(self.items, 1):
            ws = set()
            for r in it.live():
                v = r.split(":", 1)[1] if ":" in r else r
                ws |= set(RX_RUN3.findall(v.lower()))
            words[n] = ws
            allw |= ws
        ck = self.category_key()
        lane_size = WIDEN_PER_ITEM if len(self.items) <= 2 else 4
        lanes = []
        heads = []
        for lab, ids in self.item_groups():
            opened = sum(1 for r in ids if r in self.full)
            pool = [r for r in ids if r not in self.full and self.buyable(r)]
            glab = (self.rows[ids[0]].get(ck) if (ck and ids) else None)
            owners = [n for n in range(1, len(self.items) + 1) if glab is not None and str(glab) != "" and self.owns_label(n, glab)]
            wanted = set()
            found = set()
            for n in owners:
                wanted |= words[n]
                found |= set(self.item_origins.get(n, []))
            if not owners:
                wanted = allw

            def key(r):
                hits = len(set(RX_RUN3.findall(word_form(dump(self.rows[r])))) & wanted)
                p = self.price(r)
                pk = ((p is None), p or 0) if self.pick() == "cheapest" else (0, 0)
                return (r not in found, -hits) + pk
            pool.sort(key=key)
            k = lane_size - opened
            lanes.append(pool[:k] if k > 0 else [])
            heads.append(max(0, 4 - opened))
        first = []
        for rank in range(4):
            for i, ln in enumerate(lanes):
                if rank < heads[i] and rank < len(ln):
                    first.append(ln[rank])
        more = []
        for rank in range(lane_size):
            for i, ln in enumerate(lanes):
                if heads[i] <= rank < len(ln):
                    more.append(ln[rank])
        cut = max(5, 2 * len(lanes) + 2)
        got = self.fill_facts(first[:min(cut, 8)], 22)
        self.fill_facts((first[cut:] + more)[:max(0, 10 - len(got))], 30)
        return None

    def ready_to_ask(self):
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.steps_left() <= 8 or self.clock() > INTERVIEW_DEADLINE:
            return False
        N = len(self.items)
        if self.turns < TURNS_BEFORE_ASK and any(not self.item_records(n) for n in range(1, N + 1)):
            return False
        if len(self.full) >= 3 * N + 1:
            return True
        return self.step >= max(8, self.budget_steps // 4)


RX_LETTER = re.compile(r"[a-zA-Z]")
RX_VALUE_EXCL = re.compile(r"\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\b", re.I)


def _crowded(rd):
    return any(x == 1 and not isinstance(x, bool) for x in items_of(rd.get("crowded")) if _num(x))


def _label_list(labs):
    if len(labs) >= 2:
        return ", ".join(labs[:-1]) + " or " + labs[-1]
    return labs[0]


class E5(object):
    def msg_args(self, text):
        tool = self.roles.get("ask")
        sch = self.schemas.get(tool) if tool else None
        sch = sch if isinstance(sch, dict) else {}
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        req = [r for r in sch.get("required") if isinstance(r, str)] if isinstance(sch.get("required"), list) else []
        texts = [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p) and "enum" not in p and "const" not in p]
        pool = [r for r in req if r in texts] or texts
        args = {}
        if pool:
            def key(k):
                u = under_form(k)
                ml = props[k].get("maxLength")
                return (not RX_NON_MSG_END.search(u), bool(RX_MSG_ARG.search(u)), ml if _num(ml) else 1000000)
            best = pool[0]
            bk = key(best)
            for k in pool[1:]:
                if key(k) > bk:
                    best, bk = k, key(k)
            args[best] = text
        else:
            for k, p in props.items():
                if not isinstance(p, dict):
                    continue
                if "array" in kinds(p):
                    args[k] = [text]
                    break
                sp = p.get("properties")
                if isinstance(sp, dict):
                    s = next((kk for kk, pp in sp.items() if isinstance(pp, dict) and "string" in kinds(pp)), None)
                    if s is not None:
                        args[k] = {s: text}
                        break
        if not args:
            return None
        for r in req:
            if r not in args:
                f = filler(props.get(r) or {})
                if f is NOFILL:
                    return None
                args[r] = f
        faults = []
        out = shape(args, sch, "args", faults)
        return None if faults else out

    def reset_counters(self, clear):
        for n in range(1, len(self.items) + 1):
            self.lane_found[n] = 0
            self.lane_spent[n] = 0
            self.lane_used[n] = set()
        if clear:
            self.followups = []

    def _ref_lines(self, told):
        return [t for (i, t) in self.refs if any(t in l for l in told)]

    def interview(self):
        self.asked = True
        if self.msgs_free() <= 0:
            return ""
        self.widen()
        cnt = self.msg_limit - self.msgs_used
        if self.listing_mode:
            cnt = max(cnt, self.list_cap())
        self.plan_lanes(cnt)
        self.reset_counters(True)
        sent_all = []
        told = []
        groups = self.group_items() if self.listing_mode else []
        while self.msgs_free() > 0 and not self.finished and self.steps_left() > 6 and self.clock() < QUESTION_DEADLINE:
            sent = []
            while groups and self.listing_mode and not sent:
                sent = self.group_ask(groups.pop(0), told)
            if not sent:
                sent = self.ask_next(told)
            if not sent:
                break
            sent_all += sent
        found = [n for n in range(1, len(self.items) + 1) if self.lane_found.get(n, 0) > 0]
        early = self.early_trail()
        vs = self.value_search(found) if found else ""
        if told:
            self.version += 1
        lines = ["[Cairn] Cairn's interviewer questioned the shopper as shown below, one step per question."]
        for n, q, a in sent_all:
            lines.append("Q: " + q)
            lines.append("A: " + a)
        lines.append("New firm requirements from the answers: %s. These must be applied to every pick, with a fresh search "
                     "on the new values wherever one is needed." % ("; ".join(told) if told else "none"))
        refs_in = self._ref_lines(told)
        if refs_in:
            lines.append("These needs take their value from a different listing: %s. Locate that listing, read its whole "
                         "page, and then open every further listing to which that page directs onward (only the listings "
                         "that page points onward to, not every listing it mentions); the final page along this chain "
                         "supplies the value, and an order is valid only once every one of them has been opened."
                         % " | ".join(refs_in))
        if early:
            lines.append(early)
        if vs:
            lines.append(vs)
        if self.queued:
            lines.append("[notes seen in the meantime] " + " | ".join(self.queued))
            self.queued = []
        return "\n".join(lines)

    def late_question(self):
        if self.late_asked or not self.roles.get("ask") or self.msgs_used >= self.msg_limit:
            return ""
        if any(d["stage"] == "order" and d["msgs"] >= 1 for d in self.duties):
            return ""
        if self.steps_left() < 8 or self.clock() > LATE_Q_DEADLINE or not self.judges():
            return ""
        self.late_asked = True
        if not self.lanes:
            self.plan_lanes(self.list_cap() if self.listing_mode else 1)
            self.reset_counters(False)
        b0 = self.budget
        told = []
        sent = self.ask_next(told, 6)
        if not sent:
            self.trace.append({"kind": "late_question", "sent": False})
            return ""
        if self.budget != b0:
            told.append("budget now %s" % self.budget)
        if not told:
            return ""
        self.version += 1
        n, q, a = sent[0]
        out = ["[Cairn] Cairn put a further question to the shopper ahead of the checkout, at a cost of one step.",
               "Q: " + q, "A: " + a, "New firm requirements: " + "; ".join(told) + ".",
               "The checkout did not happen. Test every cart line against these requirements (search again on the new "
               "values where necessary), repair the cart, and after that check out."]
        refs_in = self._ref_lines(told)
        if refs_in:
            out.append("These needs take their value from a different listing: %s. Locate that listing, read its whole "
                       "page, and then open every further listing to which that page directs onward (only the listings "
                       "that page points onward to, not every listing it mentions); the final page along this chain "
                       "supplies the value." % " | ".join(refs_in))
        vs = self.value_search([n])
        if vs:
            out.append(vs)
        return "\n".join(out)

    def _shelfy(self, v, labels):
        vw = word_form(v)
        cands = [word_form(it.shelf) for it in self.items] + [word_form(it.name) for it in self.items] + [word_form(l) for l in labels]
        cands = [c for c in cands if c]
        if vw in cands:
            return True
        vs = set(w for w in stems(v) if len(w) >= 4)
        for c in cands:
            if vs & set(w for w in stems(c) if len(w) >= 4):
                return True
        return False

    def search_template(self):
        tool = self.roles.get("search")
        if not tool:
            return None
        props = self._props(tool)
        req = (self.schemas.get(tool) or {}).get("required")
        req = req if isinstance(req, list) else []
        texts = [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum")]
        named = [k for k in texts if RX_QUERY_ARG.search(under_form(k))]
        ck = self.category_key()
        labels = set()
        if ck:
            for r in self.seen:
                v = self.rows[r].get(ck)
                if v is not None:
                    labels.add(str(v).strip().lower())
        for s in reversed(self.searched):
            cands = [k for k in (named or texts) if isinstance(s.get(k), str) and RX_LETTER.search(s[k])]
            if not named:
                ok = []
                for k in cands:
                    vals = [x.get(k) for x in self.searched if isinstance(x.get(k), str)]
                    if (len(set(vals)) >= 2 and any(len(v.split()) >= 2 for v in vals)
                            and not any(v.strip().lower() in labels for v in vals)):
                        ok.append(k)
                cands = ok
            if not cands:
                continue
            field = cands[0]
            kept = {}
            for k, v in s.items():
                p = props.get(k)
                if k != field and isinstance(p, dict) and (p.get("enum") or k in req):
                    kept[k] = v
            for k in list(kept):
                if isinstance(kept[k], str) and k not in req and self._shelfy(kept[k], labels):
                    del kept[k]
            for k in list(kept):
                if props[k].get("enum"):
                    cnt = {}
                    order = []
                    for x in self.searched:
                        if isinstance(x.get(field), str) and RX_LETTER.search(x[field]) and k in x:
                            d = dump(x[k])
                            if d not in cnt:
                                cnt[d] = 0
                                order.append((d, x[k]))
                            cnt[d] += 1
                    if order:
                        top = max(cnt.values())
                        kept[k] = next(v for d, v in order if cnt[d] == top)
            return (tool, kept, field)
        return None

    def value_template(self):
        st = self.search_template()
        if st:
            return st
        stool = self.roles.get("search")
        if not stool:
            return None
        sw = self.brief.get("search_words")
        if isinstance(sw, dict):
            tool = sw.get("tool") if isinstance(sw.get("tool"), str) and sw.get("tool") in self.schemas else stool
            field = sw.get("field")
            p = self._props(tool).get(field) if isinstance(field, str) else None
            if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum"):
                fixed = sw.get("args") if isinstance(sw.get("args"), dict) else {}
                a = dict(fixed)
                a[field] = "x"
                f = []
                a = shape(a, self.schemas.get(tool), "args", f)
                if not f:
                    a.pop(field, None)
                    return (tool, a, field)
        props = self._props(stool)
        req = (self.schemas.get(stool) or {}).get("required")
        req = req if isinstance(req, list) else []
        words = [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum")
                 and RX_QUERY_ARG.search(under_form(k))]
        last = self.searched[-1] if self.searched else None
        opts = [k for k, p in props.items() if isinstance(p, dict) and isinstance(p.get("enum"), list) and len(p["enum"]) >= 2]
        modes = {}
        for k in opts:
            en = props[k]["enum"]
            q = [o for o in en if isinstance(o, str) and RX_QUERY_ARG.search(o)]
            if len(q) == 1:
                modes[k] = q[0]
            elif k in req and len(en) == 2 and last is not None and k in last:
                diff = [o for o in en if o != last.get(k)]
                if len(diff) == 1:
                    modes[k] = diff[0]
        if len(words) != 1 or len(modes) > 1 or (opts and not modes):
            return None
        base = {}
        if last:
            for k in req:
                if k in last and k != words[0]:
                    base[k] = last[k]
        base.update(modes)
        return (stool, base, words[0])

    def value_words(self, n):
        it = self.items[n - 1]
        reft = set(t for _, t in self.refs)
        out = []
        for r in it.live():
            if r in reft or ":" not in r:
                continue
            v = r.split(":", 1)[1].strip()
            told = r in it.told
            m = RX_FLOOR_PHRASE.search(v)
            if m:
                if told:
                    out.append((m.group(1).strip(), True))
                continue
            if not v or RX_VALUE_EXCL.search(v):
                continue
            out.append((v, told))
        return out

    def value_search(self, items):
        vt = self.value_template()
        if not vt or self.steps_left() < self.reserve() + VALUE_ROOM:
            return ""
        allow = min(VALUE_RESERVE, self.steps_left() - self.reserve() - VALUE_RESERVE)
        tool, base, field = vt
        ck = self.category_key()
        lines = []
        for n in items:
            ph = self.value_words(n)
            if not any(f for p, f in ph) or allow <= 2:
                continue
            leadi = [i for i, (p, f) in enumerate(ph) if f and (len(p.split()) == 1 or RX_AMOUNT.fullmatch(p))]
            order = [ph[i] for i in leadi] + [ph[i] for i in range(len(ph)) if i not in leadi]
            reqpos = [i for i, (p, f) in enumerate(order) if not f]
            drop = reqpos[-1] if reqpos else 0
            words = [p for p, f in order]
            if len(words) >= 2:
                tries = [words, words[:drop] + words[drop + 1:]]
            elif leadi:
                tries = [[words[0], self.items[n - 1].name], [words[0]]]
            else:
                tries = [[words[0]]]
            for t in tries[:2]:
                a = dict(base)
                a[field] = " ".join(" ".join(t).split())
                f = []
                a = shape(a, self.schemas.get(tool), "args", f)
                if f or self.finished:
                    break
                res = self.env(tool, a)
                allow -= 1
                if res is None:
                    break
                obs = res.get("observation")
                self.observe(tool, obs)
                recs = self.obs_listings(obs)
                prices = [self.price(l) for l, _ in recs]
                bud = self.budget
                if bud is not None and recs and all(p is not None for p in prices) and all(p > 10 * bud for p in prices):
                    bud = None
                kept = []
                for lid, _ in recs:
                    if not self.buyable(lid):
                        continue
                    if ck:
                        lab = (self.rows.get(lid) or {}).get(ck)
                        if lab is None or not self.owns_label(n, lab):
                            continue
                    p = self.price(lid)
                    if bud is not None and p is not None and p > bud:
                        continue
                    if lid not in kept:
                        kept.append(lid)
                hl = self.home_label(n)
                if ck and hl:
                    kept = ([l for l in kept if word_form(self.rows[l].get(ck, "")) == hl]
                            + [l for l in kept if word_form(self.rows[l].get(ck, "")) != hl])
                lines.append("%s: %s" % (dump(a.get(field)), ", ".join(kept[:6]) if kept else "no row of this item's kind came back"))
                if kept:
                    picks = [l for l in kept if l not in self.full][:max(0, min(3, allow))]
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

    def home_label(self, n):
        ck = self.category_key()
        if not ck:
            return ""
        it = self.items[n - 1]
        for pool in (list(self.full.values()), [self.rows[r] for r in self.seen]):
            cnt = {}
            order = []
            for rec in pool:
                lab = rec.get(ck)
                if lab is None or not self.owns_label(n, lab):
                    continue
                w = word_form(lab)
                if w not in cnt:
                    cnt[w] = 0
                    order.append(w)
                cnt[w] += 1
            for t in (word_form(it.shelf), word_form(it.name)):
                if t and t in cnt:
                    return t
            if order:
                top = max(cnt.values())
                return next(w for w in order if cnt[w] == top)
        return ""

    def group_items(self):
        order = [n for n in self.plan_order if self.lanes.get(n)]
        order += [n for n in sorted(self.lanes) if n not in self.plan_order and self.lanes.get(n)]
        f = self.msgs_free()
        if f <= 0 or len(order) <= f:
            return []
        groups = []
        rest = order[:]
        g = f
        while rest and g > 0:
            k = -(-len(rest) // g)
            groups.append(rest[:k])
            rest = rest[k:]
            g -= 1
        return groups

    def remaining(self, n):
        used = self.lane_used.get(n, set())
        return [l for l in self.lanes.get(n, []) if word_form(l) not in used and not self.told_about(n, l)]

    def group_ask(self, group, told, floor=6):
        G = len(group)
        cap = self.list_cap()
        now = self.list_now()
        share = max(1, now // G) if now < cap else max(MIXED_LABELS, cap // G)
        parts = []
        for n in group:
            rem = self.remaining(n)
            if rem:
                parts.append((n, rem[:share]))
        if not parts:
            return []
        if len(parts) == 1:
            n, labs = parts[0]
            labs2, text = self.fit_ask(n, labs, False)
            if not text:
                return []
            sent = self.send([(n, labs2[0], text)], floor)
            if sent:
                self.finish_ask(n, labs2, False, sent, told)
            return sent
        chosen = None
        while parts and chosen is None:
            for c in range(share, 0, -1):
                cut = [(n, l[:c]) for n, l in parts]
                text = self.group_text(cut)
                a = self.msg_args(text)
                if a is not None and says(a, text):
                    chosen = cut
                    break
            if chosen is None:
                parts = parts[:-1]
        if not chosen:
            return []
        parts = chosen
        text = self.group_text(parts)
        sent = self.send([(parts[0][0], parts[0][1][0], text)], floor)
        if not sent:
            return []
        for n, _ in parts:
            self.lane_spent[n] = self.lane_spent.get(n, 0) + 1
        self.list_asks += 1
        before = dict((n, len(self.items[n - 1].told)) for n, _ in parts)
        rd = dict(self.read_answers([(0, sent[0][1], sent[0][2])], [LIGHT, HEAVY]))
        rd.pop("free", None)
        self.fold_reading(rd, told)
        answered = False
        for n, labs in parts:
            new = self.items[n - 1].told[before[n]:]
            if new:
                answered = True
                self.lane_found[n] = self.lane_found.get(n, 0) + 1
                self.settle(n, labs, False, new, False)
        if _crowded(rd) and not answered:
            self.listing_mode = False
            self.trace.append({"kind": "lists_off", "group": [n for n, _ in parts]})
            return sent
        if not answered:
            self.lane_used.setdefault(parts[0][0], set()).add(word_form(parts[0][1][0]))
        self.trace.append({"kind": "group_ask", "items": [n for n, _ in parts], "answered": answered})
        return sent

    def finish_ask(self, n, labels, shown, sent, told):
        self.lane_spent[n] = self.lane_spent.get(n, 0) + 1
        self.list_asks += 1
        before = [len(it.told) for it in self.items]
        rd = self.read_answers([(n, sent[0][1], sent[0][2])], [LIGHT, HEAVY])
        if len(labels) > 1:
            rd = dict(rd)
            rd.pop("free", None)
        self.fold_reading(rd, told)
        new = []
        for i, it in enumerate(self.items, 1):
            nn = it.told[before[i - 1]:]
            if nn:
                self.lane_found[i] = self.lane_found.get(i, 0) + 1
                new += nn
        self.settle(n, labels, shown, new, _crowded(rd))

    def list_cap(self):
        return LIST_ONE if len(self.items) == 1 else LIST_MANY

    def list_now(self):
        ramp = RAMP_ONE if len(self.items) == 1 else RAMP_MANY
        cap = self.list_cap()
        return min(cap, ramp[self.list_asks] if self.list_asks < len(ramp) else cap)

    def next_ask(self):
        scarce = self.msg_limit <= SCARCE_MSGS and not self.listing_mode
        N = len(self.items)

        def key(n):
            po = self.plan_order.index(n) if n in self.plan_order else len(self.plan_order) + n
            return (self.lane_found.get(n, 0) > 0, 0 if scarce else self.lane_spent.get(n, 0), po)
        for n in sorted(range(1, N + 1), key=key):
            rem = self.remaining(n)
            if rem:
                return n, (rem[:self.list_now()] if self.listing_mode else rem[:1])
        return None

    def told_about(self, n, label):
        ls = spelled(label)
        for t in self.items[n - 1].told:
            if names_match(ls, spelled(t.partition(":")[0])):
                return True
        return False

    def lookalikes(self, n, label):
        keyroots = {}
        korder = []
        for rid in self.item_listing_ids(n):
            rec = self.full.get(rid)
            if rec is None:
                continue
            for k, v in (attrs_of(rec) or {}).items():
                w = word_form(k)
                if w not in keyroots:
                    keyroots[w] = set()
                    korder.append(w)
                keyroots[w] |= roots(str(v))
        wc = {}
        for w in korder:
            for x in set(w.split()):
                wc[x] = wc.get(x, 0) + 1
        lw = word_form(label)
        plain = set(x for x in lw.split() if wc.get(x, 0) <= 2)
        own = keyroots.get(lw, set())
        scored = []
        for w in korder:
            if not w or w == lw:
                continue
            shared = plain & set(w.split())
            common = own & keyroots[w]
            if shared or common:
                scored.append((3 * len(shared) + min(3, len(common)), w))
        scored.sort(key=lambda s: -s[0])
        return [w for _, w in scored[:LOOKALIKE_KEYS]]

    def q_text(self, n, labels, shown):
        name = self.asked_name(n)
        if len(labels) >= 2:
            return "What %s would you like for the %s?" % (_label_list(labels), name)
        lab = labels[0]
        la = self.apart.get(lab, {}).get(n)
        if la:
            s = "Apart from the " + la[0] + (" and the " + la[1] if len(la) > 1 else "")
            return "%s, which %s would you like for the %s?" % (s, lab, name)
        if shown:
            ex = self.examples(n, lab)
            return "What %s would you like for the %s, for example %s or %s?" % (lab, name, ex[0], ex[1])
        return "What %s would you like for the %s?" % (lab, name)

    def group_text(self, parts):
        t = ", and ".join("what %s for the %s" % (_label_list(labs), self.asked_name(n)) for n, labs in parts)
        t = "W" + t[1:]
        i = t.find("for the")
        return t[:i] + "would you like " + t[i:] + "?"

    def fit_ask(self, n, labels, shown):
        for c in range(len(labels), 0, -1):
            text = self.q_text(n, labels[:c], shown)
            a = self.msg_args(text)
            if a is not None and says(a, text):
                return labels[:c], text
        return [], ""

    def ask_next(self, told, floor=6):
        got = None
        for _ in range(LIST_ONE):
            self.followups = [f for f in self.followups if not self.told_about(f[0], f[1][0])]
            if self.followups:
                n, labels, shown = self.followups.pop(0)
            else:
                na = self.next_ask()
                if not na:
                    return []
                n, labels = na
                shown = False
            if not labels:
                return []
            labs, text = self.fit_ask(n, labels, shown)
            if not text and shown:
                labs, text = self.fit_ask(n, labels, False)
                shown = False
            if text:
                got = (n, labs, shown, text)
                break
            self.lane_used.setdefault(n, set()).add(word_form(labels[0]))
            self.trace.append({"kind": "question_too_long", "item": n})
        if not got:
            return []
        n, labs, shown, text = got
        sent = self.send([(n, labs[0], text)], floor)
        if not sent:
            return []
        self.finish_ask(n, labs, shown, sent, told)
        return sent


RX_MEASURED = re.compile(r"^\s*[\d.,/x \-]+[a-z]{0,6}\s*$")
RX_LABEL_CUT = re.compile(r"\s(?:and|or)\s|/|,|;|\(")
RX_PLAIN_S = re.compile(r"(?<=[a-z]{3})s\b")
RX_PLAIN_OUR = re.compile(r"our\b")
RX_AND_WORD = re.compile(r"\band\b", re.I)
RX_AND_CUT = re.compile(r"\s*(?:,|\band\b)\s*", re.I)
RX_PLURAL_END = re.compile(r"[a-z]{2}[a-rtv-z]s$", re.I)
RX_Q_FILLER = re.compile(r"\b(?:and|for|like|me|or|please|tell|the|what|which|would|you)\b", re.I)


def catchall(k):
    w = word_form(k)
    return not RX_KIND_WORD.search(w) and bool(RX_CATCHALL.search(w))


def plain_form(s):
    return RX_PLAIN_OUR.sub("or", RX_PLAIN_S.sub("", word_form(s)))


class E6(object):
    def settle(self, n, labels, shown, new, crowded):
        used = self.lane_used.setdefault(n, set())
        if len(labels) == 1:
            lab = labels[0]
            used.add(word_form(lab))
            if crowded and self.listing_mode:
                self.listing_mode = False
                self.trace.append({"kind": "lists_off", "item": n})
            if new or not crowded:
                return
            if n not in self.one_product and self.name_parts(n):
                self.one_product.add(n)
                self.followups.insert(0, (n, [lab], shown))
                self.trace.append({"kind": "one_product", "item": n})
                return
            if not self.apart:
                la = self.lookalikes(n, lab)
                if la:
                    self.apart.setdefault(lab, {})[n] = la
                    self.followups.insert(0, (n, [lab], False))
                    self.trace.append({"kind": "set_apart", "item": n, "label": lab, "keys": la})
            return
        if crowded and not new:
            self.listing_mode = False
            self.followups.insert(0, (n, [labels[0]], False))
            self.trace.append({"kind": "lists_off", "item": n})
            return
        hit = self.which_label(labels, new) if new else None
        if hit is not None:
            ahead, usedl = labels[hit + 1:], labels[:hit + 1]
        else:
            ahead, usedl = list(reversed(labels[1:])), labels[:1]
        for l in usedl:
            used.add(word_form(l))
        rest = [l for l in self.lanes.get(n, []) if l not in labels]
        self.lane_misses[n] = 0 if hit is not None else self.lane_misses.get(n, 0) + 1
        m = self.lane_misses[n]
        if m and m % 2 == 0:
            mid = (len(ahead) - 1) / 2.0
            ahead = [x for _, x in sorted(enumerate(ahead), key=lambda p: (abs(p[0] - mid), p[0]))]
            self.lanes[n] = rest + ahead
        else:
            self.lanes[n] = ahead + rest
        self.trace.append({"kind": "list_settled", "item": n, "hit": hit})

    def which_label(self, labels, new):
        pooled = {}
        for g in self.key_table():
            for k in g["keys"]:
                s = pooled.setdefault(k["key"], set())
                for v in k["shown"][:4]:
                    s |= roots(v)
        best = None
        for nd in new:
            prop, _, val = nd.partition(":")
            ps = spelled(prop)
            scores = [len(spelled(l) & ps) for l in labels]
            if not labels or all(s == 0 for s in scores):
                vr = roots(val)
                scores = [len(pooled.get(word_form(l), set()) & vr) for l in labels]
            if scores and max(scores) > 0:
                pos = scores.index(max(scores))
                best = pos if best is None else min(best, pos)
        return best

    def ranked_keys(self, n, table):
        it = self.items[n - 1]
        settled = [stems(r.partition(":")[0]) for r in it.musts]
        rows = [k for g in table for k in g["keys"] if self.label_owner(g["label"]) == n]
        if len(self.items) == 1 and not rows:
            rows = [k for g in table for k in g["keys"]]
        measured = {}
        for g in table:
            for k in g["keys"]:
                ex = k["shown"][:4]
                measured[k["key"]] = sum(1 for v in ex if RX_MEASURED.match(v)) * 2 > len(ex)
        rows = sorted(rows, key=lambda k: (-k["count"], measured.get(k["key"], False), -len(k["shown"][:4])))
        keys = [k["key"] for k in rows if not any(names_match(s, stems(k["key"])) for s in settled)]
        prop = []
        dem = []
        for k in keys:
            if not RX_NON_PROP.search(k) or RX_MODELS_KEY.search(k):
                prop.append(k)
            else:
                dem.append(k)
        return [k for k in prop if not catchall(k)] + [k for k in prop if catchall(k)] + dem

    def _owner_raw(self, label):
        lw = word_form(label)
        lr = roots(label)
        best, bs = None, 0
        for n, it in enumerate(self.items, 1):
            if lw == word_form(it.shelf):
                sc = 3
            elif lw == word_form(it.name):
                sc = 2
            else:
                ir = roots(it.shelf) | roots(it.name)
                sc = len(lr & ir) / float(max(1, len(lr | ir)))
            if sc > bs:
                best, bs = n, sc
        return best

    def adopt(self):
        ck = self.category_key()
        if len(self.items) < 2 or not ck:
            return
        self.part_shelves()
        labels = []
        for rid in self.full:
            v = self.full[rid].get(ck)
            if v is not None and str(v) != "" and v not in labels:
                labels.append(v)
        strays = [l for l in labels if self.label_owner(l) is None]
        orphans = [n for n in range(1, len(self.items) + 1)
                   if not any(self._owner_raw(l) == n or word_form(l) == word_form(self.items[n - 1].shelf) for l in labels)]
        if strays and len(orphans) == 1:
            for l in strays:
                self.adopted[str(l)] = orphans[0]

    def label_owner(self, label):
        if label is None or not word_form(label):
            return None
        if str(label) in self.adopted:
            return self.adopted[str(label)]
        key = (str(label), len(self.items))
        if key not in self.owners_seen:
            self.owners_seen[key] = self._owner_raw(label)
        return self.owners_seen[key]

    def owns_label(self, n, label):
        if label is None or not word_form(label) or not (1 <= n <= len(self.items)):
            return False
        return self.label_owner(label) == n or word_form(label) == word_form(self.items[n - 1].shelf)

    def item_records(self, n):
        ck = self.category_key()
        if not ck:
            return []
        return [rid for rid, rec in self.full.items() if self.owns_label(n, rec.get(ck)) and attrs_of(rec) is not None]

    def plan_lanes(self, count):
        self.adopt()
        table = self.key_table()
        N = len(self.items)
        parts = [PLAN_PROMPT.replace("COUNT", str(max(1, count))), "REQUEST:\n" + self.request]
        il = ["item %d: %s - already settled by the request: %s" % (n, it.name, "; ".join(it.musts) if it.musts else "nothing")
              for n, it in enumerate(self.items, 1)]
        parts.append("ITEMS:\n" + "\n".join(il))
        af = [str(x).strip() for x in items_of(self.brief.get("ask_first")) if str(x).strip()]
        if af:
            parts.append("ASK FIRST (the request wants the shopper asked about these):\n" + "\n".join(af))
        kl = []
        num = 0
        prev = None
        for g in table:
            for k in g["keys"]:
                num += 1
                if g["label"] != prev:
                    kl.append("category %r:" % str(g["label"]))
                    prev = g["label"]
                line = "  %d. %s (on %d): %s" % (num, k["key"], k["count"], " | ".join(x[:40] for x in k["shown"][:4]))
                line += self.not_stated(g["label"], k["key"], k["shown"][:4])
                if catchall(k["key"]):
                    line += " [lumps several features together; ask last]"
                kl.append(line)
        parts.append("ATTRIBUTE KEYS SEEN:\n" + ("\n".join(kl) if kl else "(no listing has been read in full yet)"))
        o = self.ask_json("\n\n".join(parts), [HEAVY, LIGHT], 3500) or {}
        picks = o.get("ask")
        picks = [picks] if isinstance(picks, dict) else (picks if isinstance(picks, list) else [])
        lanes = dict((n, []) for n in range(1, N + 1))
        hinted = dict((n, set()) for n in range(1, N + 1))
        for p in picks:
            if not isinstance(p, dict):
                continue
            i = count_of(p.get("item"))
            i = min(max(i if i else 1, 1), N)
            lab = RX_LABEL_CUT.split(str(p.get("label") or ""), 1)[0].strip().rstrip("?.").strip()
            if not lab or len(lab.split()) > 5:
                continue
            if word_form(lab) not in [word_form(x) for x in lanes[i]]:
                lanes[i].append(lab)
            if str(p.get("hint") or "").strip():
                hinted[i].add(word_form(lab))
                if i not in self.plan_order:
                    self.plan_order.append(i)
        for n in range(1, N + 1):
            keys = self.ranked_keys(n, table)
            reads = max(1, len(self.item_records(n)))
            own = {}
            for g in table:
                if self.label_owner(g["label"]) == n:
                    for k in g["keys"]:
                        s = k["count"] / float(reads)
                        if s > own.get(k["key"], -1):
                            own[k["key"]] = s

            def store_sp(lab):
                pf = plain_form(lab)
                for k in own:
                    if plain_form(k) == pf:
                        return k
                return lab

            def share(lab):
                w = word_form(lab)
                if w in own:
                    return own[w]
                ls = stems(lab)
                best = 0
                for k, s in own.items():
                    ks = stems(k)
                    if names_match(ls, ks) or names_match(ks, ls):
                        best = max(best, s)
                return best
            hint_sp = set(word_form(store_sp(h)) for h in hinted[n])
            lane = []
            for lab in lanes[n]:
                l2 = store_sp(lab)
                if word_form(l2) not in [word_form(x) for x in lane]:
                    lane.append(l2)
            gkeys = [k for k in keys if not catchall(k) and own.get(k, 0.0) >= GUARD_SHARE][:GUARD_KEYS]
            for gk in gkeys:
                if gk in [word_form(x) for x in lane[:3]]:
                    continue
                ks = own.get(gk, 0)
                weak = None
                for x in lane[:3]:
                    if word_form(x) not in hint_sp and share(x) + GUARD_GAP <= ks:
                        weak = x
                        break
                if weak is None and len(lane) >= 3:
                    continue
                new = [x for x in lane if word_form(x) != gk]
                if weak is not None:
                    wi = new.index(weak)
                    if N > 1:
                        new = new[:wi] + [gk] + new[wi + 1:]
                        new.insert(min(3, len(new)), weak)
                    else:
                        new.insert(wi, gk)
                else:
                    new.append(gk)
                new3 = [word_form(x) for x in new[:3]]
                if any(word_form(x) in hint_sp and word_form(x) not in new3 for x in lane[:3]):
                    continue
                lane = new
            for k in keys:
                if word_form(k) not in [word_form(x) for x in lane]:
                    lane.append(k)
            self.lanes[n] = [x for x in lane if not catchall(x)] + [x for x in lane if catchall(x)]
        self.trace.append({"kind": "plan", "lanes": dict((n, l[:6]) for n, l in self.lanes.items())})

    def name_parts(self, n):
        nm = self.items[n - 1].name
        if not RX_AND_WORD.search(nm):
            return []
        ps = [p.strip() for p in RX_AND_CUT.split(nm) if p.strip()]
        if len(ps) >= 2 and all(RX_PLURAL_END.search(p) for p in ps):
            return ps
        return []

    def asked_name(self, n):
        it = self.items[n - 1]
        parts = self.name_parts(n)
        if not parts:
            return it.name
        if n not in self.one_product:
            return RX_AND_WORD.sub("or", it.name)
        ck = self.category_key()
        rr = []
        for r in self.seen:
            row = self.rows[r]
            if ck and not self.owns_label(n, row.get(ck)):
                continue
            s = set()
            for k, v in row.items():
                if isinstance(v, str) and k != ck:
                    s |= roots(v)
            rr.append(s)
        best, bs = parts[0], -1
        for p in parts:
            hr = roots(p.split()[-1])
            sc = sum(1 for r in rr if hr & r)
            if sc > bs:
                best, bs = p, sc
        return best

    def examples(self, n, label):
        lw = word_form(label)
        musts = " ".join(self.items[n - 1].musts).lower()
        for g in self.key_table():
            for k in g["keys"]:
                if k["key"] == lw:
                    return [v for v in k["shown"][:4] if v and len(v) <= 30 and v not in musts][:2]
        return []

    def send(self, qs, floor=6):
        sent = []
        tool = self.roles.get("ask")
        for n, label, text in qs:
            if self.finished or self.steps_left() <= floor or self.msgs_used >= self.msg_limit:
                break
            a = self.msg_args(text)
            if a is None:
                break
            res = self.env(tool, a)
            if res is None:
                break
            self.msgs_used += 1
            obs = res.get("observation")
            ans = self.spoken(obs)
            sent.append((n, text, ans))
            self.done_log.append("step %d: shopper message sent %r -> reply %r (cart then: %s)"
                                 % (self.step, text, ans[:300], self.cart_mark()))
            self.done_steps.append(self.step)
            self.trace.append({"kind": "message", "item": n, "label": label})
            if isinstance(obs, dict) and obs.get("error"):
                break
        return sent

    def fold_reading(self, rd, lines):
        N = len(self.items)

        def ino(x):
            i = count_of(x)
            return min(max(i if i else 1, 1), N)
        for e in items_of(rd.get("needs")):
            if not isinstance(e, dict) or e.get("need") is None or not str(e.get("need")).strip():
                continue
            i = ino(e.get("item"))
            t = str(e["need"]).strip()
            it = self.items[i - 1]
            if t not in it.told:
                it.told.append(t)
                lines.append(t if N == 1 else "item %d: %s" % (i, t))
            if e.get("points") is True and (i, t) not in self.refs:
                self.refs.append((i, t))
        for e in items_of(rd.get("drop")):
            if not isinstance(e, dict) or e.get("property") is None or not str(e.get("property")).strip():
                continue
            i = ino(e.get("item"))
            it = self.items[i - 1]
            pw = word_form(str(e["property"]).partition(":")[0])
            for r in it.musts + it.told:
                if r not in it.dropped and word_form(r.partition(":")[0]) == pw:
                    it.dropped.append(r)
                    lines.append("dropped " + r)
        for e in items_of(rd.get("free")):
            if isinstance(e, dict) and e.get("property"):
                self.items[ino(e.get("item")) - 1].free.append(str(e["property"]))
        b = rd.get("budget")
        if _num(b) and b > 0:
            self.budget = b

    def read_answers(self, sent, models=None):
        if self.quiet(sent):
            self.trace.append({"kind": "quiet_answer"})
            return {}
        self.reply_reads += 1
        lines = [REPLY_PROMPT, "", "ITEMS:"] + ["item %d: %s" % (n, it.name) for n, it in enumerate(self.items, 1)]
        lines += ["", "QUESTIONS AND ANSWERS:"]
        for j, (n, q, a) in enumerate(sent, 1):
            lines.append("%d. %s Q: %s" % (j, "(about item %d)" % n if n else "(about several items)", q))
            lines.append("A: %s" % a)
        return self.ask_json("\n".join(lines), models or [HEAVY, LIGHT], 2500) or {}

    @soft(False)
    def quiet(self, sent):
        if self.reply_reads == 0:
            return False
        known = set()
        for it in self.items:
            known |= spelled(it.name) | spelled(it.shelf)
            for r in it.musts + it.told:
                known |= spelled(" ".join(RX_ALNUM.findall(r.partition(":")[0].lower())))
        for g in self.key_table():
            known |= spelled(g["label"])
            for k in g["keys"]:
                known |= spelled(k["key"])
        for rec in self.full.values():
            for v in (attrs_of(rec) or {}).values():
                known |= spelled(" ".join(w for w in RX_ALNUM.findall(str(v).lower()) if len(w) >= 4))
        for n, q, ans in sent:
            qw = spelled(" ".join(RX_ALNUM.findall(RX_Q_FILLER.sub(" ", q).lower())))
            aw = spelled(" ".join(RX_ALNUM.findall(str(ans).lower())))
            if RX_HAS_DIGIT.search(str(ans)) or aw & known or aw & qw or RX_HOW_ASK.search(str(ans)):
                return False
        return True


RX_COUNT_KEY = re.compile(r"\b(?:qty|quantity|count|unit|units)\b")
RX_LIMIT_WORDS = re.compile(r"\b(?:not|no|least|most|under|over|min|max|minimum|maximum|unlike|different)\b")
RX_REMOVE_OPT = re.compile(r"remov|delet|drop|clear|take", re.I)
RX_VAL_SPLIT = re.compile(r"[,/;|]")
GATE_LEAD = "Cairn carried out the request's own steps before continuing:"


class E7(object):
    def gate(self, stage):
        due = [d for d in self.duties if d["stage"] == stage]
        if not due or self.finished or not self.gate_due(stage):
            return "go", ""
        buy, ref = self.roles.get("buy"), self.roles.get("refuse")
        tl = "\n".join("- %s: %s | parameters: %s" % (t["function"]["name"], t["function"].get("description") or "",
                                                      dump(t["function"].get("parameters") or {}))
                       for t in self.tools if t["function"]["name"] not in (buy, ref))
        said = []
        nudged = False
        phrase = "place the order" if stage == "order" else "end without buying anything"
        ask = self.roles.get("ask")
        for rnd in range(GATE_ROUNDS):
            blocks = [GATE_PROMPT.replace("STAGE", phrase), "REQUEST:\n" + self.request,
                      "STEPS DUE NOW:\n" + "\n".join("- " + d["text"] for d in due), "LEDGER:\n" + self.ledger(),
                      "CART: " + self.cart_text(),
                      "RECORD (shopper messages and store actions so far):\n"
                      + ("\n".join("%d. %s" % (i, l) for i, l in enumerate(self.done_log, 1)) if self.done_log else "(empty so far)")]
            if self.change_notes:
                blocks.append("NOTES SEEN:\n" + "\n".join(self.change_notes[-4:]))
            blocks.append("Messages left: %d   Steps left: %d (the %s itself uses one step)"
                          % (max(0, self.msg_limit - self.msgs_used), self.steps_left(), "order" if stage == "order" else "ending"))
            blocks.append("TOOLS:\n" + tl)
            o = self.ask_json("\n\n".join(blocks), [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if o is None:
                break
            if said:
                lines = []
                self.fold_reading({"needs": o.get("needs")}, lines)
                hold = str(o.get("hold") or "").strip()
                if lines or hold:
                    self.version += 1
                    txt = GATE_LEAD + "\n" + "\n".join(said) + "\n"
                    txt += ("The shopper's answer alters the task: %s. Repair the cart accordingly and check out once more."
                            % "; ".join(lines + ([hold] if hold else [])))
                    return "stop", txt
            call = o.get("call")
            talk = stage == "order" and any(d["msgs"] >= 1 for d in due)
            if talk:
                proof = set(i for i, s in enumerate(self.done_steps, 1) if s > self.cart_changed_at)
            else:
                proof = set(range(1, len(self.done_log) + 1))
            unproven = []
            for e in items_of(o.get("steps")):
                if not isinstance(e, dict):
                    continue
                cited = [count_of(x) for x in items_of(e.get("lines")) if not isinstance(x, bool)]
                if not (e.get("done") is True and any(c in proof for c in cited if c is not None)):
                    unproven.append(str(e.get("step") or ""))
            if not isinstance(call, dict) or not call:
                if (unproven and not said and not nudged and ask and self.msgs_used < self.msg_limit
                        and self.steps_left() >= 2):
                    nudged = True
                    self.done_log.append("(Cairn: none of the record lines written after the cart's latest change at step %d "
                                         "proves: %s. Messages left: %d, steps left: %d. Give the call that performs it.)"
                                         % (self.cart_changed_at, "; ".join(unproven)[:200], self.msg_limit - self.msgs_used,
                                            self.steps_left()))
                    self.done_steps.append(-1)
                    continue
                self.gate_passes.add((stage, self.cart_mark() + "#" + str(self.version)))
                break
            tool = call.get("tool")
            if not isinstance(tool, str) or tool not in self.schemas or tool in (buy, ref):
                break
            if self.steps_left() < 2 or (tool == ask and self.msgs_used >= self.msg_limit):
                break
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            f = []
            args = shape(args, self.schemas.get(tool), "args", f)
            if f:
                break
            if tool == ask:
                ss = strings_of(args)
                if not ss:
                    break
                text = max(ss, key=len)
                sent = self.send([(1, "", text)], 1)
                if not sent:
                    break
                self.gate_msgs += 1
                said.append("Q: %s\nA: %s" % (text, sent[0][2]))
                continue
            res = self.env(tool, args)
            if res is None:
                break
            obs = res.get("observation")
            self.absorb(obs)
            shown = self.render(obs)
            self.done_log.append("step %d %s %s -> %s (%s)" % (self.step, tool, dump(args), shown[:300], self.cart_mark()))
            self.done_steps.append(self.step)
            said.append("%s %s -> %s" % (tool, dump(args), shown[:400]))
        return "go", (GATE_LEAD + "\n" + "\n".join(said)) if said else ""

    def spoken(self, obs):
        if isinstance(obs, str):
            return obs
        out = []

        def walk(v, d):
            if d > 4:
                return
            if isinstance(v, str):
                if len(v.split()) >= 2:
                    out.append(v)
            elif isinstance(v, dict):
                for x in v.values():
                    walk(x, d + 1)
            elif isinstance(v, list):
                for x in v:
                    walk(x, d + 1)
        walk(obs, 0)
        if out:
            return " ".join(out)[:800]
        return dump(obs)[:400]

    def loose_lines(self, obs):
        if not isinstance(obs, dict) or self.listing_id(obs, False, 0):
            return []
        out = []
        for v in obs.values():
            if isinstance(v, str):
                if len(v.split()) >= 5:
                    out.append(v)
            elif isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                out += [x for x in v if len(x.split()) >= 5]
            elif isinstance(v, dict) and not self.listing_id(v, False, 1):
                out += [x for x in v.values() if isinstance(x, str) and len(x.split()) >= 6]
        res = []
        for l in out:
            if l not in self.heard and l not in res:
                res.append(l)
        return res

    def read_note(self, lines):
        for l in lines:
            self.heard.add(l)
        p = NOTE_PROMPT + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger() + "\n\nNOTICE:\n" + "\n".join(lines)
        o = self.ask_json(p, [LIGHT, SPARE], 2500) or {}
        gone = [str(x) for x in items_of(o.get("gone")) if isinstance(x, str) or _whole(x)]
        known = [g for g in gone if g in self.rows]
        for g in known:
            self.unavail.add(g)
        sold = bool(gone) or o.get("sold_out") is True
        if sold:
            self.note_sold = True
            if len(lines) == 1:
                self.sold_lines.add(lines[0])
        if known:
            self.note_named = True
        kind = o.get("kind")
        if kind and kind != "change":
            return ""
        rec = []
        N = len(self.items)
        b = self.budget_guard(o.get("budget"))
        bchg = False
        if _num(b) and b > 0 and b != self.budget:
            self.budget = b
            rec.append("budget now %s" % gnum(b))
            bchg = True
        dropped = False
        for e in items_of(o.get("drop")):
            if not isinstance(e, dict) or e.get("need") is None or not str(e.get("need")).strip():
                continue
            i = count_of(e.get("item"))
            it = self.items[min(max(i if i else 1, 1), N) - 1]
            nd = str(e["need"]).strip()
            live = it.live()
            m = nd if nd in live else next((r for r in live if word_form(r.partition(":")[0]) == word_form(nd.partition(":")[0])), None)
            it.dropped.append(m if m is not None else nd)
            rec.append("dropped " + nd)
            dropped = True
        for e in items_of(o.get("add")):
            if not isinstance(e, dict) or e.get("need") is None or not str(e.get("need")).strip():
                continue
            i = count_of(e.get("item"))
            it = self.items[min(max(i if i else 1, 1), N) - 1]
            nd = str(e["need"]).strip()
            if nd in it.dropped:
                it.dropped.remove(nd)
            if nd not in it.musts and nd not in it.told:
                it.told.append(nd)
            rec.append("added " + nd)
        if gone:
            rec.append("unavailable: " + ", ".join(gone))
        if o.get("price_moved") is True:
            rec.append("a price moved")
        if self.pick() == "cheapest" and (dropped or gone or o.get("sold_out") is True or bchg):
            self.wider_held = False
            self.cheaper_marks = set()
        summ = str(o.get("summary") or "").strip()
        if rec or summ:
            self.version += 1
            self.recent = []
            self.change_notes.append(summ if summ else "; ".join(rec))
        return (summ + (" (" + "; ".join(rec) + ")" if rec else "")).strip()

    def live_pairs(self):
        out = []
        for n, it in enumerate(self.items, 1):
            for r in it.live():
                out.append((n, r))
        return out

    def evidence_obs(self, obs):
        ids = []

        def walk(v, d, inrows):
            if d > 6:
                return
            if isinstance(v, dict):
                lid = self.listing_id(v, inrows, d)
                if lid and attrs_of(v) is not None:
                    if lid not in ids:
                        ids.append(lid)
                    return
                for x in v.values():
                    walk(x, d + 1, False)
            elif isinstance(v, list):
                for x in v:
                    walk(x, d + 1, True)
        walk(obs, 0, False)
        return self.evidence(ids)

    def evidence(self, ids):
        live = self.live_pairs()
        if not ids or not live:
            return ""
        several = len(self.items) >= 2
        lines = []
        for lid in ids[:12]:
            a = attrs_of(self.full.get(lid)) or {}
            parts = []
            for n, r in live[:8]:
                prop = r.partition(":")[0].strip()
                ps = stems(prop)
                hits = []
                for k, v in a.items():
                    if names_match(ps, stems(k)):
                        h = "%s=%s" % (k, str(v)[:80])
                        bo = self.bare_off(r, v)
                        if bo is not None:
                            u = self.measured_unit(r)
                            h += (" (a bare figure; taken in %s, the unit of that requirement, it amounts to %s %s)"
                                  % (u, gnum(bo), u))
                        hits.append(h)
                hits = hits[:3]
                if hits:
                    parts.append(("item %d: " % n if several else "") + prop + " -> " + ", ".join(hits))
            if parts:
                lines.append("%s: %s" % (lid, "; ".join(parts)))
        if not lines:
            return ""
        return ("[evidence: each key listed carries a requirement's property inside its name; a key about another property "
                "(another component or aspect) settles nothing either way, and a differently named key might still hold "
                "that same property] " + " | ".join(lines))

    def fresh_view(self):
        if self.cart_view is not None and self.cart_seen_at >= self.cart_changed_at:
            return self.cart_view
        return None

    def cart_text(self):
        v = self.fresh_view()
        if v is not None:
            return dump(clip(v, 300))[:3000]
        return "(from Cairn's cart record, listing: units) " + dump(self.cart)

    def units_held(self, lid):
        v = self.fresh_view()
        if v is not None:
            stack = [v]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    if any(isinstance(s, str) and s.strip() == lid for s in x.values()):
                        cs = []
                        for k, val in x.items():
                            if not isinstance(val, str) and count_of(val) is not None and RX_COUNT_KEY.search(word_form(k)):
                                cs.append(count_of(val))
                        if len(cs) == 1:
                            return cs[0]
                    for y in x.values():
                        if isinstance(y, (dict, list)):
                            stack.append(y)
                elif isinstance(x, list):
                    stack.extend(x)
        return self.cart.get(lid, 1)

    def held_units(self):
        return dict((r, self.units_held(r)) for r in self.cart_ids())

    def cart_ids(self):
        v = self.fresh_view()
        if v is not None:
            t = dump(v)
            return [r for r in self.seen if r in t]
        return list(self.cart.keys())

    def listing_record(self, lid):
        r = self.full.get(lid) or self.rows.get(lid)
        if r is None:
            return "(no record seen)"
        return dump(clip(r, 400))[:3000]

    def other_listings(self, skip, limit=30):
        ids = list(reversed(self.seen))
        if self.pick() == "cheapest":
            ids.sort(key=lambda r: (r not in self.full, self.price(r) is None, self.price(r) or 0))
        out = []
        for r in ids:
            if r in skip or self.sold_out(r):
                continue
            out.append(r + ": " + dump(clip(self.full.get(r) or self.rows.get(r), 160))[:900])
            if len(out) >= limit:
                break
        return out

    def fit_index(self):
        key = (len(self.seen), len(self.full), sum(len(v) for v in self.views.values()))
        if self.fit_cache is not None and self.fit_cache[0] == key:
            return self.fit_cache[1]
        idx = {}
        for rid in self.seen:
            rec = self.full.get(rid) or self.rows.get(rid) or {}
            strs = [v for v in rec.values() if isinstance(v, str)]
            ents = []
            for v in rec.values():
                if isinstance(v, dict):
                    ents += ["%s %s" % (k, x) for k, x in v.items() if isinstance(x, str)]
            idx[rid] = fit_marks(" ".join(strs) + " " + ". ".join(ents))
        notes = {}
        for m in idx.values():
            for code, codes in m["notes"]:
                notes.setdefault(code, []).extend(codes)
        for m in idx.values():
            m["fitset"] = set(m["works"]) | (set(notes.get(m["own"], [])) if m["own"] else set())
        self.fit_cache = (key, idx)
        return idx

    def item_listing_ids(self, n):
        ck = self.category_key()
        if not ck:
            return []
        return [r for r in self.seen if self.owns_label(n, self.rows[r].get(ck))]

    def series_system(self, a, b):
        idx = self.fit_index()
        for carrier, namer in ((a, b), (b, a)):
            cl = self.item_listing_ids(carrier)
            tagged = [r for r in cl if idx[r]["own_series"]]
            if len(tagged) < SERIES_ROWS or len(tagged) * 2 < len(cl):
                continue
            fam = set()
            for r in tagged:
                fam |= set(idx[r]["own_series"])
            if len(fam) > SERIES_BREADTH:
                continue
            if any(set(idx[r]["named_series"]) & fam for r in self.item_listing_ids(namer)):
                return carrier, namer, fam
        return None

    def link_holds(self, carried, naming, fam):
        idx = self.fit_index()
        c = set(idx[carried]["own_series"]) & fam
        n = set(idx[naming]["named_series"]) & fam
        if c and n:
            return bool(c & n)
        return None

    def link_faults(self, part, main, device, plines):
        sysm = self.series_system(part, main)
        if not sysm:
            return []
        carrier, namer, fam = sysm
        idx = self.fit_index()
        out = []
        for pl in plines:
            carried, naming = (pl, device) if carrier == part else (device, pl)
            if self.link_holds(carried, naming, fam) is not False:
                continue
            wanted = set(idx[naming]["named_series"]) & fam
            alts = [r for r in self.item_listing_ids(carrier) if set(idx[r]["own_series"]) & wanted and not self.sold_out(r)
                    and r != carried][:4]
            txt = ("%s (series %s) against %s, which takes series %s"
                   % (carried, ", ".join(sorted(set(idx[carried]["own_series"]) & fam)), naming, ", ".join(sorted(wanted))))
            if alts:
                txt += "; seen listings carrying that series: " + ", ".join(alts)
            else:
                txt += ("; search for that series together with the item's own words, or take a different %s, one with a "
                        "series carried by the wanted lines" % self.items[main - 1].name)
            out.append(txt)
        return out

    def fit_series(self, n):
        held = []
        cnt = {}
        idx = self.fit_index()
        cart = set(self.cart_ids())
        for a, b in self.fit_pairs:
            if n not in (a, b):
                continue
            sysm = self.series_system(a, b)
            if not sysm:
                continue
            fam = sysm[2]
            for it in (a, b):
                for r in self.item_listing_ids(it):
                    for f in sorted((set(idx[r]["own_series"]) | set(idx[r]["named_series"])) & fam):
                        cnt[f] = cnt.get(f, 0) + 1
                        if r in cart and f not in held:
                            held.append(f)
        rest = sorted([f for f in cnt if f not in held], key=lambda f: (-cnt[f], f))
        return (held + rest)[:SERIES_HUNTED]

    def covers(self, pid, did):
        idx = self.fit_index()
        fs, bar, dc = idx[pid]["fitset"], idx[pid]["barred"], idx[did]["models"]
        return any(code_covers(c, x) for c in fs for x in dc) and not any(code_covers(c, x) for c in bar for x in dc)

    def fit_faults(self):
        if not self.fit_pairs:
            return []
        idx = self.fit_index()
        ck = self.category_key()
        cart = self.cart_ids()
        out = []
        for part, main in self.fit_pairs:
            if not ck:
                continue
            dev = [r for r in cart if self.owns_label(main, (self.rows.get(r) or {}).get(ck))]
            pls = [r for r in cart if self.owns_label(part, (self.rows.get(r) or {}).get(ck))]
            if len(dev) != 1 or not pls:
                continue
            d = dev[0]
            dcodes = idx[d]["models"]
            lf = []
            for pl in pls:
                if dcodes and self.covers(pl, d):
                    continue
                lf += self.link_faults(part, main, d, [pl])
            if lf:
                out += lf
                continue
            if not dcodes:
                continue
            for pl in pls:
                fs = idx[pl]["fitset"]
                if not fs or self.covers(pl, d):
                    continue
                alts = [r for r in self.item_listing_ids(part) if self.covers(r, d) and not self.sold_out(r) and r not in cart][:4]
                txt = ("%s (%s) is meant for %s, against %s (%s)"
                       % (pl, self.items[part - 1].name, ", ".join(sorted(fs)[:4]), d, ", ".join(sorted(set(dcodes)))))
                if alts:
                    txt += "; compatible parts of the same sort: " + ", ".join(alts)
                else:
                    txt += ("; none of the seen parts of this sort suits the device. Hunt for a part that suits it or change "
                            "the device: " + self.fit_plan(True))
                out.append(txt)
        return out

    def fit_plan(self, short=False):
        if not self.fit_pairs:
            return ""
        idx = self.fit_index()
        lines = []
        for m in sorted(set(mm for p, mm in self.fit_pairs)):
            parts = sorted(set(p for p, mm in self.fit_pairs if mm == m))
            tally = {}
            order = []

            def add(c, p):
                if c not in tally:
                    tally[c] = {}
                    order.append(c)
                tally[c][p] = tally[c].get(p, 0) + 1
            for p in parts:
                for r in self.item_listing_ids(p):
                    if not self.sold_out(r):
                        for c in idx[r]["fitset"]:
                            add(c, p)
                sysm = self.series_system(p, m)
                if sysm and sysm[0] == p:
                    for r in self.item_listing_ids(p):
                        if not self.sold_out(r):
                            for c in set(idx[r]["own_series"]) & sysm[2]:
                                add(c, p)
            ranked = sorted(order, key=lambda c: (-len(tally[c]), -sum(tally[c].values())))[:6]
            segs = []
            for c in ranked:
                devs = [r for r in self.item_listing_ids(m) if c in idx[r]["named_series"]
                        or any(code_covers(c, x) for x in idx[r]["models"])][:2]
                counts = ", ".join("%d %s" % (tally[c][p], self.items[p - 1].name) for p in parts if p in tally[c])
                segs.append("%s: %s %s" % (c, counts, ("(devices seen: %s)" % ", ".join(devs)) if devs else "(no device has turned up yet)"))
            if segs:
                lines.append("codes the parts name for the %s -> %s" % (self.items[m - 1].name, "; ".join(segs)))
            if self.pick() == "cheapest":
                cs = self.cheapest_sets(m)
                if cs:
                    lines.append(cs)
        if not lines:
            return ""
        if short:
            return " | ".join(lines)
        return ("[fit plan: a part suits a device if the part's wording mentions the device's model number or another trade "
                "name of it, or if the device's wording mentions the series the part's code belongs to; try a single query "
                "combining one code (or one series) with the item's own terms, and if that returns nothing, browse the "
                "category the device belongs to and read the codes off its rows]\n" + "\n".join(lines))

    @soft("")
    def cheapest_sets(self, m):
        idx = self.fit_index()
        parts = sorted(set(p for p, mm in self.fit_pairs if mm == m))
        shelves = {}
        for p in parts:
            ls = [r for r in self.item_listing_ids(p) if self.price(r) is not None and self.buyable(r)]
            ls.sort(key=lambda r: (self.price(r), r))
            shelves[p] = (self.series_system(p, m), ls)
        sets = []
        for d in self.item_listing_ids(m):
            if (not idx[d]["models"] and not idx[d]["named_series"]) or self.price(d) is None or not self.buyable(d):
                continue
            tot = self.price(d) * self.items[m - 1].qty
            ids = [d]
            ok = True
            for p in parts:
                sysm, ls = shelves[p]
                fit = None
                for r in ls:
                    if self.covers(r, d) or (sysm and sysm[0] == p and self.link_holds(r, d, sysm[2]) is True):
                        fit = r
                        break
                if fit is None:
                    ok = False
                    break
                ids.append(fit)
                tot += self.price(fit) * self.items[p - 1].qty
            if ok:
                sets.append((tot, ids))
        sets.sort(key=lambda s: s[0])
        if not sets:
            return ""
        return ("cheapest complete sets observed (requirements still unverified): "
                + "; ".join("%s=%s" % ("+".join(ids), gnum(t)) for t, ids in sets[:CHEAPEST_SETS]))

    def missing_items(self):
        ck = self.category_key()
        if not ck:
            return []
        lines = [r for r in self.cart_ids() if self.units_held(r) > 0]
        if len(lines) >= len(self.items):
            return []
        out = []
        for n in range(1, len(self.items) + 1):
            if (any(self.owns_label(n, self.rows[r].get(ck)) for r in self.seen)
                    and not any(self.owns_label(n, (self.rows.get(r) or {}).get(ck)) for r in lines)):
                out.append(n)
        return out

    def states_all(self, n, rid):
        a = attrs_of(self.full.get(rid)) or {}
        for r in self.items[n - 1].live():
            prop, _, val = r.partition(":")
            vt = RX_VAL_SPLIT.sub(" ", word_form(val))
            vw = set(vt.split())
            if not vw or RX_LIMIT_WORDS.search(vt):
                return False
            ps = spelled(prop)
            ok = False
            for k, v in a.items():
                if names_match(ps, spelled(k)) and vw <= set(RX_VAL_SPLIT.sub(" ", word_form(v)).split()):
                    ok = True
                    break
            if not ok:
                return False
        return True

    def fill_short(self):
        bound = set(x for p in self.fit_pairs for x in p)
        miss = self.missing_items()
        for n in miss:
            it = self.items[n - 1]
            cands = [r for r in self.item_listing_ids(n) if r in self.full and r not in self.cart and self.buyable(r)
                     and self.price(r) is not None and self.states_all(n, r)]
            cands.sort(key=lambda r: self.price(r))
            pick = cands[0] if cands and n not in bound else None
            if pick is not None and _num(self.budget):
                held = sum((self.price(r) or 0) * self.units_held(r) for r in self.cart_ids())
                each = self.brief.get("budget_scope") == "each"
                cost = self.price(pick) * (1 if each else it.qty)
                if (cost if each else cost + held) > self.budget + 1e-9:
                    pick = None
            if pick is None:
                return False
            target = dict((r, self.cart[r]) for r in self.cart_ids() if r in self.cart)
            target[pick] = it.qty
            calls = self.cart_calls(target, set())
            if not calls or len(calls) > self.steps_left() - 1:
                return False
            for tool, args in calls:
                f = []
                a = shape(args, self.schemas.get(tool), "args", f)
                if f:
                    return False
                res = self.env(tool, a)
                if res is None:
                    return False
                self.note_cart(tool, a, res)
            self.trace.append({"kind": "fill_short", "item": n, "pick": pick})
        return True

    def part_shelves(self):
        ck = self.category_key()
        if not ck:
            return
        labels = sorted(set(str(self.rows[r].get(ck)).strip() for r in self.seen
                            if self.rows[r].get(ck) is not None and str(self.rows[r].get(ck)).strip()))
        moved = False
        for n, it in enumerate(self.items, 1):
            nr = roots(it.name)
            if not nr or not it.shelf:
                continue
            others = [o for j, o in enumerate(self.items, 1) if j != n]
            if not any(word_form(o.shelf) == word_form(it.shelf) for o in others):
                continue
            if any(roots(o.name) == nr for o in others):
                continue
            own = [l for l in labels if roots(l) == nr and word_form(l) != word_form(it.shelf)]
            if len(own) == 1 and not any(word_form(o.shelf) == word_form(own[0]) for o in self.items):
                self.trace.append({"kind": "shelf_parted", "item": n, "from": it.shelf, "to": own[0]})
                it.shelf = own[0]
                moved = True
        if moved:
            self.owners_seen = {}
            self.fit_cache = None

    def fuller_label(self, n, lab):
        it = self.items[n - 1]
        nr = roots(it.name)
        ck = self.category_key()
        if len(nr) < 2 or nr <= roots(lab) or not ck or (it.shelf and roots(lab) == roots(it.shelf)):
            return None
        for l in sorted(set(str(self.rows[r].get(ck)) for r in self.seen if self.rows[r].get(ck) not in (None, ""))):
            if nr <= roots(l):
                return l
        return None

    def named_label(self, n, lab):
        it = self.items[n - 1]
        ck = self.category_key()
        if not it.shelf or not ck or word_form(lab) == word_form(it.shelf):
            return None
        for r in self.seen:
            v = self.rows[r].get(ck)
            if v is not None and word_form(v) == word_form(it.shelf):
                return v
        return None

    def line_faults(self):
        self.adopt()
        ck = self.category_key()
        lines = [r for r in self.cart_ids() if self.units_held(r) > 0]
        if not lines or not ck:
            return []
        N = len(self.items)
        lab = dict((r, (self.rows.get(r) or {}).get(ck)) for r in lines)
        owners = dict((r, [n for n in range(1, N + 1) if self.owns_label(n, lab[r])]) for r in lines)
        known = [n for n in range(1, N + 1) if any(self.owns_label(n, self.rows[x].get(ck)) for x in self.seen)]
        faults = []
        for n in known:
            it = self.items[n - 1]
            mine = [r for r in lines if n in owners[r]]
            if not mine:
                t = "item %d (%s) lacks a cart line" % (n, it.name)
                own = self.label_text(n)
                if own:
                    stand = [r for r in lines if lab[r] is not None and str(lab[r]) != "" and word_form(lab[r]) != word_form(own)
                             and roots(lab[r]) & roots(own)]
                    if stand:
                        ss = []
                        for r in stand:
                            s = "%s bears its own label '%s'" % (r, lab[r])
                            if len(owners[r]) == 1:
                                s += " (that label belongs to item %d)" % owners[r][0]
                            ss.append(s)
                        t += (" (this item's own listings bear its category label '%s'; %s, so %s not count toward item %d, "
                              "whatever the title says: list the label '%s' and source this item's line there)"
                              % (own, ", ".join(ss), "this line does" if len(stand) == 1 else "these lines do", n, own))
                faults.append(("other", t))
                continue
            if len([r for r in mine if owners[r] == [n]]) > 1:
                faults.append(("other", "item %d (%s) has %d lines (%s); keep one" % (n, it.name, len(mine), ", ".join(mine))))
            for r in mine:
                u = self.units_held(r)
                if owners[r] == [n] and u != it.qty:
                    faults.append(("units", "%s is held at %s unit%s; item %d (%s) wants %d"
                                   % (r, u, "" if u == 1 else "s", n, it.name, it.qty)))
                fl = self.fuller_label(n, lab[r]) or self.named_label(n, lab[r])
                if fl:
                    faults.append(("other", "%s comes from the label %s while the store also has the label %s, which names "
                                   "item %d (%s) completely; source this line from %s instead" % (r, lab[r], fl, n, it.name, fl)))
        if len(known) == N:
            for r in lines:
                if not owners[r]:
                    faults.append(("other", "%s is of a kind nobody requested (%s); remove it"
                                   % (r, lab[r] if lab[r] is not None else "unknown kind")))
        for r in lines:
            if self.sold_out(r):
                faults.append(("other", "%s is sold out" % r))
        if _num(self.budget) and all(self.price(r) is not None for r in lines):
            if self.brief.get("budget_scope") == "each":
                for r in lines:
                    p = self.price(r)
                    if self.budget < p <= 10 * self.budget:
                        faults.append(("other", "%s costs %s, over the per-item limit %s" % (r, gnum(p), gnum(self.budget))))
            else:
                tot = sum(self.price(r) * self.units_held(r) for r in lines)
                if tot > self.budget + 1e-9 and tot <= 10 * self.budget:
                    faults.append(("other", "the cart totals %s, over the budget %s" % (gnum(tot), gnum(self.budget))))
        return faults

    def count_fixes(self):
        ck = self.category_key()
        if not ck:
            return []
        N = len(self.items)
        fixes = []
        cids = self.cart_ids()
        for r in cids:
            u = self.units_held(r) or 0
            own = [n for n in range(1, N + 1) if self.owns_label(n, (self.rows.get(r) or {}).get(ck))]
            if u <= 0 or len(own) != 1:
                continue
            n = own[0]
            cnt = sum(1 for x in cids if self.cart.get(x, 1) > 0 and self.owns_label(n, (self.rows.get(x) or {}).get(ck)))
            if cnt != 1 or u == self.items[n - 1].qty or self.sold_out(r):
                continue
            fixes.append((r, u, self.items[n - 1].qty))
        tool = self.roles.get("cart")
        if not fixes or not tool:
            return []
        props = self._props(tool)
        sch = self.schemas.get(tool)
        texts = [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum")]
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        idp = next((a for t, a in sorted(self.id_args) if t == tool and a in texts), None)
        if idp is None:
            idp = next((r for r in req if r in texts), None)
        if idp is None and len(texts) == 1:
            idp = texts[0]
        cnp = next((k for k, p in props.items() if isinstance(p, dict) and "integer" in kinds(p)), None)
        mdp = next((k for k, p in props.items() if isinstance(p, dict) and isinstance(p.get("enum"), list) and len(p["enum"]) >= 2), None)
        if not idp or not cnp:
            return []
        calls = []
        for r, u, w in fixes:
            if self.sets_counts(tool):
                calls.append({idp: r, cnp: w})
            elif mdp:
                en = props[mdp]["enum"]
                rm = next((o for o in en if RX_REMOVE_OPT.search(str(o))), en[1])
                ad = next((o for o in en if o != rm), en[0])
                calls.append({idp: r, mdp: rm, cnp: u})
                calls.append({idp: r, mdp: ad, cnp: w})
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
            res = self.env(tool, a)
            if res is None:
                break
            self.note_cart(tool, a, res)
            obs = res.get("observation")
            self.cart_said = dump(obs)
            self.observe(tool, obs)
            done.append("%s -> %s" % (dump(a), self.render(obs)[:300]))
            if self.finished:
                break
        self.trace.append({"kind": "count_fixes", "done": len(done)})
        return done

    def cart_check(self, planned=None, limit=SOFT_SPEND):
        ids = list(planned) if planned else self.cart_ids()
        key = "|".join(sorted(ids)) + "#" + str(self.version)
        js = self.judges(limit)
        if key in self.checked or not js or not ids:
            return ""
        self.checked.add(key)
        parts = [CHECK_PROMPT, "REQUEST:\n" + self.request, "REQUIREMENTS NOW BINDING:\n" + self.ledger()]
        if planned:
            parts.append("CART (only planned: judge it as though these listings sat in the cart, each at its asked "
                         "quantity): " + "; ".join("item %d: quantity %d" % (n, it.qty) for n, it in enumerate(self.items, 1))
                         + "\n" + ", ".join(ids))
        else:
            parts.append("CART:\n" + self.cart_text())
        recs = []
        for r in ids:
            so = "SOLD OUT (the store reports it cannot be purchased any more) " if self.sold_out(r) else ""
            recs.append("%s: %s%s" % (r, so, self.listing_record(r)))
        parts.append("RECORDS OF THE CARTED LISTINGS:\n" + "\n".join(recs))
        weak = [] if planned else self.weak_lines()
        if weak:
            parts.append("REQUIREMENTS THESE RECORDS LEAVE UNSTATED (a measured limit counts only when it is filed beneath a "
                         "key carrying the name of the property; a word the shopper wants counts only within some "
                         "attribute's value or within the title, never from an attribute that is related but holds a "
                         "different sort of value, and never from the description), so each listed requirement is to be "
                         "taken as unmet unless the line's record expresses it in different wording:\n"
                         + "\n".join("%s: %s" % (ln, "; ".join(rs)) for ln, n, rs in weak))
        cf = self.code_fits(ids)
        if cf:
            parts.append("FITS PROVEN BY CODES (these codes are how the store settles compatibility, between two listings or "
                         "between a listing and a device that is named, and every line below holds; a listing description "
                         "naming a different purpose, measurement or number of seats does not cancel it):\n" + "\n".join(cf))
        if self.pick() != "any":
            parts.append("OTHER LISTINGS SEEN (newest first):\n" + "\n".join(self.other_listings(set(ids), 15)))
        prompt = "\n\n".join(parts)
        ans = None
        if len(js) == 2 and not planned and not self.audited and self.pick() == "any" and not weak:
            ans = self.ask_json(prompt, [LIGHT], 3000)
            if ans is not None:
                js = [HEAVY] if ans.get("ok") is False and HEAVY in self.judges(limit) else []
        if js and not self.no_credit:
            a2 = self.ask_json(prompt, js, 3000)
            if a2 is not None:
                ans = a2
        self.trace.append({"kind": "cart_check", "key": key, "ok": None if ans is None else ans.get("ok")})
        self.verdicts[key] = ans is not None and ans.get("ok") is not False
        if ans is None or ans.get("ok") is not False:
            return ""
        self.leads = [l for l in self.leads if l not in ids]
        probs = [p if isinstance(p, str) else dump(p) for p in items_of(ans.get("problems"))][:5]
        better = [str(b) for b in items_of(ans.get("better")) if isinstance(b, (str, int)) and str(b) in self.rows and not self.sold_out(str(b))][:4]
        if not planned:
            self.check_better[key] = better
        txt = "A careful review finds the cart still wrong: " + ("; ".join(probs) if probs else "see the requirements in the ledger") + "."
        if better:
            txt += " Candidates that seem preferable: " + ", ".join(better) + "."
        txt += (" Repair the cart, or else demonstrate that the reviewer erred by reading the records afresh, and then "
                "check out.")
        return txt


RX_LINK_STOP1 = re.compile(r"^(?:the|and|with|from|that|comes|listing|page|its|for|of)$")
RX_LINK_STOP2 = re.compile(r"^(?:the|and|with|from|that|comes|listing|page|its|for)$")
RX_RUN_ANY = re.compile(r"[a-z0-9]+")
RX_FIRST_DIGITS = re.compile(r"\d+")
QUIET_TEXT = "I am sorry, but no product in this shop satisfies all of your requirements, so I am not making a purchase."


def leaves(v, out=None):
    if out is None:
        out = []
    if isinstance(v, dict):
        for x in v.values():
            leaves(x, out)
    elif isinstance(v, list):
        for x in v:
            leaves(x, out)
    elif v is not None:
        out.append(v)
    return out


def edge_in(word, text):
    return re.search(RX_TITLE_EDGE % re.escape(word), text) is not None


class E8(object):
    def page_read(self, lid):
        if lid not in self.opened:
            return False
        off = self.roles.get("offer")
        if off and self.single_target(off):
            return off in self.views.get(lid, set())
        return True

    def pending_refs(self):
        live = self.live_all()
        return [(i, t) for (i, t) in self.refs if t in live and t not in self.ref_values]

    def _trail_pre(self):
        return self.steps_left() >= TRAIL_ROOM and self.clock() <= HUNT_DEADLINE and bool(self.pending_refs())

    def early_trail(self):
        if not self._trail_pre():
            return ""
        r0, h0 = self.chain_rounds, self.chain_holds
        self.chain_rounds = 0
        self.chain_room = EARLY_TRAIL_ROUNDS
        try:
            txt, _ = self.trail(HUNT_DEADLINE)
        finally:
            self.chain_rounds, self.chain_holds = r0, h0
            self.chain_room = None
        return txt

    def decline_trail(self):
        if self.decline_chain_done or not self._trail_pre():
            return ""
        self.decline_chain_done = True
        txt, _ = self.trail(HUNT_DEADLINE)
        if not txt:
            return ""
        return txt + "\nRun searches for those values at once and purchase whatever matches; decline only if nothing matches."

    def _noname_keys(self):
        return set(k for k in (self.id_key, self.category_key(), self.title_key()) if k)

    def _short_vals(self, rid, bad):
        out = []
        for k, v in (self.rows.get(rid) or {}).items():
            if k not in bad and isinstance(v, str) and len(v.strip()) <= 40 and not RX_PLACEHOLDER.match(v.strip()):
                out.append(v.strip())
        return out

    def _names_in(self, rid, low, bad):
        for c in self._short_vals(rid, bad):
            if len(c) > 2 and edge_in(c.lower(), low):
                return True
        return False

    def opened_shop(self, desc):
        low = desc.lower()
        bad = self._noname_keys()
        return [r for r in list(self.opened) if r in self.rows and self._names_in(r, low, bad)]

    def sold_line_search(self):
        lines = [r for r in self.cart_ids() if r not in self.gone]
        if any(self.sold_out(r) for r in lines):
            return ""
        if len(lines) == 1:
            self.unavail.add(lines[0])
            return ""
        if len(lines) >= 2:
            return self.reread_offers()
        return ""

    def replacements(self, lost):
        js = self.judges()
        if not js:
            return []
        recs = []
        for r in reversed(self.seen):
            if r in lost or r in self.gone or self.sold_out(r):
                continue
            recs.append(dump(clip(self.full.get(r) or self.rows.get(r), 200))[:1200])
            if len(recs) >= 25:
                break
        if not recs:
            return []
        p = (SWAP_PROMPT + "\n\nREQUEST:\n" + self.request + "\n\nLIVE REQUIREMENTS:\n" + self.ledger()
             + "\n\nTHE LINE THAT SOLD OUT:\n" + "\n".join("%s: %s" % (r, self.listing_record(r)) for r in lost)
             + "\n\nCURRENT CART:\n" + self.cart_text() + "\n\nREMAINING RECORDS SEEN (latest first):\n" + "\n".join(recs))
        a = self.ask_json(p, list(reversed(js)), 3500) or {}
        cart = set(self.cart_ids())
        kept = []
        fits = a.get("fits")
        for e in (fits if isinstance(fits, list) else ([fits] if fits is not None else [])):
            i = e.get("id") if isinstance(e, dict) else e
            i = str(i) if i is not None else ""
            pr = e.get("price") if isinstance(e, dict) else None
            pr = pr if _num(pr) else float("inf")
            if i in self.rows and i not in lost and i not in self.gone and i not in cart and i not in [k for k, _ in kept]:
                kept.append((i, pr))
        if self.pick() == "cheapest":
            kept.sort(key=lambda x: x[1])
        return [k for k, _ in kept[:4]]

    def audit(self):
        if self.audited or not self.judges():
            return ""
        self.audited = True
        a = self._audit_call()
        fits = self._fits(a)
        if not fits and a:
            op = [str(x) for x in items_of(a.get("open")) if isinstance(x, (str, int)) and str(x) in self.rows
                  and str(x) not in self.opened and not self.sold_out(str(x))]
            if op and self._open_for_audit(op):
                a = self._audit_call()
                fits = self._fits(a)
        if not fits:
            return ""
        self.lead_items = fits
        self.leads = [i for v in fits.values() for i in v]
        txt = "The review believes these listings could match: " + "; ".join("item %s: %s" % (k, ", ".join(v)) for k, v in fits.items())
        reason = str((a or {}).get("reason") or "").strip()
        if reason:
            txt += " (" + reason + ")"
        return txt + ". Open these listings and purchase what really matches; decline only when every one of them fails."

    def _audit_call(self):
        ids = sorted(reversed(self.seen), key=lambda r: r not in self.full)
        recs = []
        for r in ids:
            if self.sold_out(r):
                continue
            if r in self.full:
                recs.append("OPENED " + dump(clip(self.full[r], 200))[:1200])
            else:
                recs.append("ROW " + dump(clip(self.rows[r], 120))[:500])
            if len(recs) >= 30:
                break
        if not recs:
            return None
        js = self.judges()
        if not js:
            return None
        p = (AUDIT_PROMPT + "\n\nREQUEST:\n" + self.request + "\n\nLIVE REQUIREMENTS:\n" + self.ledger()
             + "\n\nRECORDS SEEN (latest first):\n" + "\n".join(recs))
        a = self.ask_json(p, [LIGHT], 4000)
        if HEAVY in js and not self.no_credit and not self._fits(a):
            b = self.ask_json(p, [HEAVY], 4000)
            if b is not None:
                a = b
        return a

    def _fits(self, a):
        if not isinstance(a, dict) or not isinstance(a.get("fits"), dict):
            return {}
        out = {}
        for k, v in a["fits"].items():
            ids = [str(x) for x in (v if isinstance(v, list) else [v]) if isinstance(x, (str, int)) and not isinstance(x, bool)
                   and str(x) in self.rows and not self.sold_out(str(x))][:4]
            if ids:
                out[str(k)] = ids
        return out

    def _open_for_audit(self, op):
        cmp_ = self.roles.get("compare")
        tgt = self.compare_target(cmp_) if cmp_ else None
        if cmp_ and tgt and self.compare_facts() is not False:
            plan = [(cmp_, {tgt: op[:self.batch_room()]})]
        else:
            ft = self.facts_tool()
            t = self.single_target(ft) if ft else None
            if not ft or not t:
                return False
            plan = [(ft, {t: r}) for r in op[:2]]
        calls = self.cart_calls(dict(("#%d" % i, 1) for i in range(len(self.items))), set())
        cs = len(calls) if calls is not None else len(self.items)
        if self.steps_left() < len(plan) + max(5, self.reserve() + cs):
            return False
        before = len(self.full)
        for tool, args in plan:
            f = []
            a = shape(args, self.schemas.get(tool), "args", f)
            if f:
                return False
            res = self.env(tool, a)
            if res is None or self.finished:
                break
            self.observe(tool, res.get("observation"))
        return len(self.full) > before

    def cart_calls(self, wanted, drops):
        tool = self.roles.get("cart")
        if not tool:
            return None
        props = self._props(tool)
        sch = self.schemas.get(tool) or {}
        for k, p in props.items():
            if isinstance(p, dict) and "array" in kinds(p) and isinstance(p.get("items"), dict) and isinstance(p["items"].get("properties"), dict):
                ip = p["items"]["properties"]
                idk = next((kk for kk, pp in ip.items() if isinstance(pp, dict) and "string" in kinds(pp) and not pp.get("enum")), None)
                cnk = next((kk for kk, pp in ip.items() if isinstance(pp, dict) and "integer" in kinds(pp)), None)
                if not idk:
                    return None
                lst = []
                for lid, u in wanted.items():
                    el = {idk: lid}
                    if cnk:
                        el[cnk] = u
                    lst.append(el)
                return [(tool, {k: lst})]
        texts = [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum")]
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        tgt = next((a for t, a in sorted(self.id_args) if t == tool and a in texts), None)
        if tgt is None:
            tgt = next((r for r in req if r in texts), None)
        if tgt is None and len(texts) == 1:
            tgt = texts[0]
        cnt = next((k for k, p in props.items() if isinstance(p, dict) and "integer" in kinds(p)), None)
        mode = next((k for k, p in props.items() if isinstance(p, dict) and isinstance(p.get("enum"), list) and len(p["enum"]) >= 2), None)
        if not tgt:
            return None
        sets = self.sets_counts(tool)
        calls = []
        for lid in sorted(drops):
            held = self.cart.get(lid, 1)
            if mode:
                a = {tgt: lid, mode: props[mode]["enum"][1]}
                if cnt:
                    a[cnt] = held
            elif cnt and sets:
                a = {tgt: lid, cnt: 0}
            elif cnt and _whole(props[cnt].get("minimum")) and props[cnt]["minimum"] <= -held:
                a = {tgt: lid, cnt: -held}
            else:
                return None
            calls.append((tool, a))
        cids = set(self.cart_ids())
        for lid, u in wanted.items():
            held = self.cart.get(lid, 0) if lid in cids else 0
            if held == u:
                continue
            if held > 0 and not sets:
                return None
            a = {tgt: lid}
            if mode:
                a[mode] = props[mode]["enum"][0]
            if cnt:
                a[cnt] = u
            calls.append((tool, a))
        return calls

    def sets_counts(self, tool):
        props = self._props(tool)
        if any(isinstance(p, dict) and isinstance(p.get("enum"), list) and len(p["enum"]) >= 2 for p in props.values()):
            return False
        return any(isinstance(p, dict) and "integer" in kinds(p) and _num(p.get("minimum")) and p.get("minimum") == 0
                   for p in props.values())

    def whole_removal(self, args):
        tool = self.roles.get("cart")
        props = self._props(tool)
        cnt = next((k for k, p in props.items() if isinstance(p, dict) and "integer" in kinds(p)), None)
        rem = False
        for k, v in args.items():
            p = props.get(k)
            if isinstance(p, dict) and isinstance(p.get("enum"), list) and v in p["enum"][1:]:
                rem = True
        line = next((v for v in args.values() if isinstance(v, str) and v in self.cart), None)
        if cnt and rem and line is not None and cnt not in args:
            u = self.cart[line]
            mx = props[cnt].get("maximum")
            if _num(mx):
                u = min(u, mx)
            a = dict(args)
            a[cnt] = u
            return a
        return args

    def is_order(self, tool, args):
        if not tool or tool != self.roles.get("buy"):
            return False
        fa = self.brief.get("buy_args") if isinstance(self.brief.get("buy_args"), dict) else {}
        return all(k in args and args.get(k) == v for k, v in fa.items())

    def is_decline(self, tool, args):
        if not tool or tool != self.roles.get("refuse"):
            return False
        fa = self.brief.get("refuse_args") if isinstance(self.brief.get("refuse_args"), dict) else {}
        return all(k in args and args.get(k) == v for k, v in fa.items())

    def quiet_fill(self, args):
        tool = self.roles.get("refuse")
        sch = self.schemas.get(tool) or {}
        props = self._props(tool)
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        out = dict(args)
        for k, p in props.items():
            if not isinstance(p, dict) or "string" not in kinds(p) or p.get("enum"):
                continue
            if k not in out and k not in req:
                continue
            v = out.get(k)
            v = v if isinstance(v, str) else ("" if v is None else str(v))
            if not v.strip() or any(r in v for r in self.rows) or self.brief.get("quiet_decline") is not False:
                ml = p.get("maxLength")
                out[k] = QUIET_TEXT[:int(ml) if _num(ml) else 200]
        return out

    def map_ids(self, tool, args):
        if tool not in (self.roles.get("cart"), self.roles.get("open"), self.roles.get("offer"), self.roles.get("compare")) or not tool or not self.id_key:
            return args, ""
        props = self._props(tool)
        free = [k for k, p in props.items() if isinstance(p, dict) and ("string" in kinds(p) or "array" in kinds(p)) and not p.get("enum")]
        if len(free) != 1:
            return args, ""
        fk = free[0]
        if tool != self.roles.get("cart") and (tool, fk) not in self.id_args:
            return args, ""
        if fk not in args:
            return args, ""

        def mp(v):
            if not isinstance(v, str):
                return v, ""
            s = v.strip()
            if s in self.rows:
                return s, ""
            if id_like(s, True) or len(s) < 8:
                return v, ""
            low = s.lower()
            hits = [r for r in self.seen if any(isinstance(x, str) and x.strip().lower().startswith(low)
                                               for k, x in self.rows[r].items() if k != self.id_key)]
            if len(hits) == 1:
                return hits[0], ""
            if hits:
                return v, "more than one listing holds the text %r (%s); give a single listing id" % (s, ", ".join(hits[:6]))
            return v, "the text %r is no listing id; give the id as the result rows show it, in the %s field" % (s, self.id_key)
        v = args.get(fk)
        if isinstance(v, list):
            nv = []
            for x in v:
                y, pb = mp(x)
                if pb:
                    return args, pb
                nv.append(y)
        else:
            nv, pb = mp(v)
            if pb:
                return args, pb
        out = dict(args)
        out[fk] = nv
        return out, ""

    def render(self, obs):
        t = dump(clip(obs, ROW_CLIP if self.obs_listings(obs) else TEXT_CLIP))
        if len(t) > OBS_CAP:
            t = t[:OBS_CAP] + "...(cut)"
        return t

    def batch_room(self):
        cmp_ = self.roles.get("compare")
        tgt = self.compare_target(cmp_) if cmp_ else None
        if tgt:
            mx = self._props(cmp_)[tgt].get("maxItems")
            if _num(mx):
                return int(mx)
        if self.cmp_fail:
            return max(2, min(self.cmp_fail - 1, max(self.cmp_ok, 2)))
        return max(4, self.cmp_ok)

    def join_opens(self, calls):
        op = self.roles.get("open")
        cmp_ = self.roles.get("compare")
        ot = self.single_target(op) if op else None
        ct = self.compare_target(cmp_) if cmp_ else None
        if not (op and ot and cmp_ and ct and op != cmp_ and self.compare_facts() is not False):
            return calls
        picks = []
        for i, c in enumerate(calls):
            fn = c.get("function") or {}
            if fn.get("name") != op or not isinstance(fn.get("arguments"), str):
                continue
            a = json_obj(fn.get("arguments"))
            if not isinstance(a, dict) or list(a.keys()) != [ot]:
                continue
            lid = a[ot]
            if isinstance(lid, str) and lid.strip() in self.rows and lid.strip() not in self.gone:
                picks.append((i, lid.strip()))
        picks = picks[:self.batch_room()]
        if len(picks) < 2:
            return calls
        pidx = set(i for i, _ in picks)
        out = []
        for i, c in enumerate(calls):
            if i == picks[0][0]:
                out.append({"id": c.get("id"), "type": "function",
                            "function": {"name": cmp_, "arguments": dump({ct: [l for _, l in picks]})}})
            elif i in pidx:
                d = dict(c)
                d["joined"] = True
                out.append(d)
            else:
                out.append(c)
        return out


class GM(object):
    def label_text(self, n):
        ck = self.category_key()
        if not ck:
            return None
        it = self.items[n - 1]
        targets = [roots(it.name)]
        if it.shelf.strip():
            targets.append(roots(it.shelf))
        cands = sorted(set(str(self.rows[r].get(ck)).strip() for r in self.seen
                           if self.rows[r].get(ck) is not None and str(self.rows[r].get(ck)).strip()))
        for c in cands:
            if roots(c) in targets and self.label_owner(c) == n:
                return c
        return None

    def label_line(self):
        if len(self.items) < 2 or not self.category_key():
            return ""
        lt = {}
        for n in range(1, len(self.items) + 1):
            t = self.label_text(n)
            if t:
                lt[n] = t
        paired = set()
        for a in lt:
            for b in lt:
                if a != b and roots(lt[a]) < roots(lt[b]):
                    paired.add(a)
                    paired.add(b)
        if len(paired) < 2:
            return ""
        if all(self.items[n - 1].shelf and word_form(self.items[n - 1].shelf) == word_form(lt[n]) for n in paired):
            return ""
        return ("store labels: " + "; ".join("item %d (%s) under '%s'" % (n, self.items[n - 1].name, lt[n]) for n in sorted(paired))
                + ". Any listing counts only toward the one item whose category label it carries, whatever its title "
                  "says; find each of these items by listing that item's own label.")

    def label_fit(self, n, label):
        if label is None or str(label) == "" or not (1 <= n <= len(self.items)):
            return 0
        it = self.items[n - 1]
        lw = word_form(label)
        if lw == word_form(it.shelf) or self.adopted.get(str(label)) == n:
            return 3
        if lw == word_form(it.name):
            return 2
        lr = roots(label)
        ir = roots(it.shelf) | roots(it.name)
        return len(lr & ir) / float(max(1, len(lr | ir)))

    def budget_guard(self, v):
        if not _num(v) or v <= 0 or not _num(self.budget) or self.budget <= 0:
            return v
        b = self.budget
        if b / 30.0 <= v <= b * 30.0:
            return v
        for c in (v * 100.0, v / 100.0):
            if b / 3.0 <= c <= b * 3.0:
                r = round(c, 6)
                self.trace.append({"kind": "budget_unit", "read": v, "kept": r})
                return int(r) if r == int(r) else r
        return v

    def _later(self, sent):
        if isinstance(sent, int) and not isinstance(sent, bool):
            return sent not in (0, 1)
        return bool(sent)

    def held_owners(self):
        ck = self.category_key()
        if not ck:
            return set()
        return set(o for o in (self.label_owner((self.rows.get(r) or {}).get(ck)) for r in self.cart_ids()) if o)

    def req_words(self):
        out = {}
        for n, it in enumerate(self.items, 1):
            s = set()
            for r in it.live():
                v = r.split(":", 1)[1] if ":" in r else r
                s |= set(RX_RUN3.findall(word_form(v)))
            out[n] = s
        return out

    def money_key(self, obs, sh):
        recs = self.obs_listings(obs)
        if len(recs) < 3:
            return None
        other = set(v for k, v in sh.items() if k != "price" and isinstance(v, str))
        figs = {}
        order = []
        for _, rec in recs:
            for k, v in rec.items():
                if k == self.stock_key or k in other or RX_ID_FIELD.search(str(k)):
                    continue
                m = money(v)
                if m is not None and m > 0:
                    if k not in figs:
                        figs[k] = []
                        order.append(k)
                    figs[k].append(m)
        floor = max(3, -(-2 * len(recs) // 3))
        cands = [k for k in order if len(figs[k]) >= floor and RX_MONEY_FIELD.search(under_form(k))]
        if not cands:
            return None
        best = max(cands, key=lambda k: (len(set(figs[k])), len(figs[k]), -order.index(k)))
        return best if len(set(figs[best])) >= 3 else None

    def page_param(self, tool, val):
        if isinstance(val, bool) or not isinstance(val, (str, int)):
            return None
        sch = self.schemas.get(tool) or {}
        props = self._props(tool)
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        last = self.last_args.get(tool) or {}
        cands = []
        for k, p in props.items():
            if not isinstance(p, dict) or k in req or "enum" in p:
                continue
            ks = kinds(p)
            if ks and not any(kind_fit(x, val) for x in ks):
                continue
            lv = last.get(k)
            if isinstance(lv, str) and lv.strip():
                continue
            if RX_QUERY_ARG.search(under_form(k)):
                continue
            ml = p.get("maxLength")
            if isinstance(val, str) and _whole(ml) and len(val) > ml:
                continue
            if isinstance(val, int):
                mn, mx = p.get("minimum"), p.get("maximum")
                if (_num(mn) and val < mn) or (_num(mx) and val > mx):
                    continue
            cands.append((k, ks))
        if isinstance(val, int):
            ints = [c for c in cands if "integer" in c[1]]
            if ints:
                cands = ints
        return cands[0][0] if len(cands) == 1 else None

    def next_big_page(self, tool, args, obs):
        if self.budget_steps < BIG_STEPS or self.steps_left() - self.reserve() < BIG_SPARE:
            return ""
        plan = self.paging_plan(tool, obs)
        if not plan or self._later(args.get(plan["arg"])) or plan["next"] is None or plan["total"] <= plan["shown"]:
            return ""
        ck = self.category_key()
        if not ck:
            return ""
        rown = set(o for o in (self.label_owner(rec.get(ck)) for _, rec in self.obs_listings(obs)) if o)
        if not rown or rown <= self.held_owners():
            return ""
        a2 = dict(args)
        a2[plan["arg"]] = plan["next"]
        f = []
        a2 = shape(a2, self.schemas.get(tool), "args", f)
        if f:
            return ""
        res = self.env(tool, a2)
        if res is None or self.finished:
            return ""
        self.pages_read += 1
        o2 = res.get("observation")
        self.observe(tool, o2)
        self.paged.append(o2)
        rend = self.render(o2)
        self.read_cache[mark_of(tool, a2)] = (self.version, self.step, rend)
        self.trace.append({"kind": "next_big_page", "tool": tool})
        return ("[Cairn fetched this search's following page on its own, at a cost of one step]\n%s=%s: %s"
                % (plan["arg"], dump(plan["next"]), rend))

    def big_opens(self, obs):
        ck = self.category_key()
        if self.budget_steps < BIG_STEPS or self.steps_left() - self.reserve() < BIG_SPARE or not ck:
            return ""
        held = self.held_owners()
        words = self.req_words()
        pool = []
        seen = set()
        for o in [obs] + list(self.paged):
            for lid, rec in self.obs_listings(o):
                own = self.label_owner(rec.get(ck))
                if not own or own in held or lid in self.full or not self.buyable(lid) or self.sold_out(lid) or lid in seen:
                    continue
                seen.add(lid)
                runs = set(RX_RUN3.findall(word_form(dump(self.rows.get(lid) or rec))))
                pr = (self.price(lid) or 0) if self.pick() == "cheapest" else 0
                pool.append((-len(words.get(own, set()) & runs), pr, str(lid), lid))
        pool.sort(key=lambda x: x[:3])
        picks = [x[3] for x in pool[:BIG_OPENS]]
        if not picks:
            return ""
        texts = self.fill_facts(picks, self.reserve() + BIG_SPARE)
        if not texts:
            return ""
        self.trace.append({"kind": "big_opens", "ids": picks})
        return "[Cairn fetched the attribute records below on its own, at a cost of one step per listing]\n" + "\n".join(texts)

    @soft("")
    def end_reads(self, tool, obs):
        if self.budget_steps >= BIG_STEPS or self.pick() != "cheapest" or not self.paged or not self.price_key:
            return ""
        plan = self.paging_plan(tool, self.paged[-1])
        if plan and plan["next"] is not None:
            return ""
        ck = self.category_key()
        if self.steps_left() - self.reserve() < END_ROOM or not ck:
            return ""
        held = self.held_owners()
        pool = []
        for o in [obs] + list(self.paged):
            for lid, rec in self.obs_listings(o):
                own = self.label_owner(rec.get(ck))
                if (own and own not in held and lid not in self.full and self.price(lid) is not None and self.buyable(lid)
                        and not self.sold_out(lid) and lid not in pool):
                    pool.append(lid)
        picks = sorted(pool, key=lambda l: (self.price(l), str(l)))[:END_READS]
        texts = self.fill_facts(picks, self.reserve() + END_ROOM - END_READS - 1) if picks else []
        if not texts:
            return ""
        self.trace.append({"kind": "end_reads", "ids": picks})
        ev = self.evidence([l for l in picks if l in self.full])
        return ("[Cairn on its own fetched the attribute records of the lowest-priced rows of this search, at one step per "
                "row, since a cheapest pick is in force and the search has now been read through its final page]\n"
                + "\n".join(texts) + ("\n" + ev if ev else ""))

    def series_sweep(self):
        ck = self.category_key()
        st = self.roles.get("search")
        if (self.budget_steps < BIG_STEPS or not self.fit_pairs or not st or not ck
                or self.steps_left() - self.reserve() < BIG_SPARE):
            return ""
        idx = self.fit_index()
        lines = []
        for part, main in self.fit_pairs:
            sysm = self.series_system(part, main)
            if not sysm or sysm[0] != part:
                continue
            fam = sysm[2]
            proven = [r for r in self.item_listing_ids(part) if r in self.full and self.buyable(r) and not self.sold_out(r)
                      and self.proves(r, part)]
            wanted = set()
            for r in proven:
                wanted |= set(idx[r]["own_series"]) & fam
            key = (main, frozenset(wanted))
            if not proven or len(proven) > PINNED_SERIES or not wanted or key in self.series_swept:
                continue

            def names():
                ix = self.fit_index()
                return [r for r in self.item_listing_ids(main) if set(ix[r]["named_series"]) & wanted
                        and self.buyable(r) and not self.sold_out(r)]
            if names():
                continue
            self.series_swept.add(key)
            mi = self.items[main - 1]
            dw = set(RX_RUN3.findall(word_form(mi.name)))
            for r in mi.live():
                if ":" in r:
                    dw |= set(RX_RUN3.findall(word_form(r.split(":", 1)[1])))
            spent = 0
            for cm, rec in list(self.remembered.items()):
                t, a0, r0 = rec
                if t != st:
                    continue
                if main not in set(self.label_owner(x.get(ck)) for _, x in self.obs_listings(r0)):
                    continue
                if not (dw & text_words(" ".join(v for v in a0.values() if isinstance(v, str)).lower())):
                    continue
                walked = set([cm])
                a, obs = a0, r0
                while spent < SERIES_PAGES and self.steps_left() - self.reserve() >= BIG_SPARE and not self.finished and not names():
                    plan = self.paging_plan(t, obs)
                    if not plan or plan["next"] is None:
                        break
                    a2 = dict(a)
                    a2[plan["arg"]] = plan["next"]
                    f = []
                    a2 = shape(a2, self.schemas.get(t), "args", f)
                    if f:
                        break
                    c2 = self.canon(t, a2)
                    if c2 in walked:
                        break
                    walked.add(c2)
                    if c2 in self.remembered:
                        obs = self.remembered[c2][2]
                        a = a2
                        continue
                    res = self.env(t, a2)
                    spent += 1
                    if res is None:
                        break
                    obs = res.get("observation")
                    self.observe(t, obs)
                    self.pages_read += 1
                    lines.append("%s=%s of %s: %s" % (plan["arg"], dump(plan["next"]), dump(a0), self.render(obs)))
                    a = a2
                if names() or spent >= SERIES_PAGES:
                    break
            hits = [r for r in names() if r not in self.full][:BIG_OPENS]
            reads = self.fill_facts(hits, self.reserve() + BIG_SPARE // 2) if hits else []
            self.trace.append({"kind": "series_sweep", "main": main, "series": sorted(wanted), "hits": hits})
            if lines or reads:
                lines.append("[%s rows mentioning %s, the series carried by the sole opened listings of the %s proving each "
                             "requirement]: %s" % (mi.name, ", ".join(sorted(wanted)), self.items[part - 1].name,
                                                   ", ".join(hits) if hits else "none found"))
                lines += reads
        if not lines:
            return ""
        return ("[Cairn on its own went through further pages of searches for the item the parts are meant to fit, one step "
                "per page]\n" + "\n".join(lines))

    def cap_repeat(self, tool, args):
        caps = self.cap_args(tool)
        if len(caps) != 1 or caps[0] not in args:
            return ""
        cap = caps[0]
        if self.ceiling_evidence(tool).get((cap, self._mode(tool, args))) != "fails":
            return ""
        others = dict((k, v) for k, v in args.items() if k != cap)
        for cm, rec in reversed(list(self.remembered.items())):
            t, a, r = rec
            if t != tool or (cap in a and a.get(cap) == args.get(cap)):
                continue
            if dict((k, v) for k, v in a.items() if k != cap) != others or not self.obs_listings(r):
                continue
            self.trace.append({"kind": "cap_repeat", "tool": tool})
            return (SKIP + "this mode of the search ignores %s, as its rows went over it; so the identical search with %s=%s "
                    "has already given precisely these rows:\n%s" % (cap, cap, dump(a.get(cap)), self.render(r)[:3000]))
        return ""

    @soft("")
    def dear_page(self, added):
        ck = self.category_key()
        st = self.roles.get("search")
        if (self.budget_steps >= BIG_STEPS or self.pick() != "cheapest" or self.steps_left() - self.reserve() < DEAR_ROOM
                or not ck or not st):
            return ""
        for lid in added:
            p = self.price(lid)
            lab = (self.rows.get(lid) or {}).get(ck)
            own = self.label_owner(lab)
            if p is None or not own or own in self.dear_paged:
                continue
            for cm, rec in reversed(list(self.remembered.items())):
                t, a, r = rec
                if t != st:
                    continue
                recs = self.obs_listings(r)
                if lid not in [l for l, _ in recs]:
                    continue
                plan = self.paging_plan(t, r)
                if not plan or plan["next"] is None:
                    continue
                a2 = dict(a)
                a2[plan["arg"]] = plan["next"]
                f = []
                a2 = shape(a2, self.schemas.get(t), "args", f)
                if f or self.canon(t, a2) in self.remembered:
                    continue
                pr = [self.price(l) for l, _ in recs if self.buyable(l) and self.price(l) is not None]
                if len(pr) < 4 or sum(1 for x in pr if x < p) < DEAR_SHARE * len(pr):
                    continue
                self.dear_paged.add(own)
                res = self.env(t, a2)
                if res is None or self.finished:
                    return ""
                o2 = res.get("observation")
                self.observe(t, o2)
                self.pages_read += 1
                below = sorted([l for l, x in self.obs_listings(o2) if l not in self.full and self.buyable(l)
                                and self.price(l) is not None and self.price(l) < p and str(x.get(ck)) == str(lab)],
                               key=lambda l: (self.price(l), str(l)))[:2]
                texts = self.fill_facts(below, self.reserve() + DEAR_ROOM - 4) if below else []
                self.trace.append({"kind": "dear_page", "line": lid, "read": below})
                ev = self.evidence([l for l in below if l in self.full])
                out = ("[Cairn] %s is pricier than the majority of rows in the search that displayed it, so with a cheapest "
                       "pick Cairn fetched the following page of that search (%s=%s, one step): %s"
                       % (lid, plan["arg"], dump(plan["next"]), self.render(o2)))
                if texts:
                    out += ("\nand Cairn also read the attribute records of that page's lowest-priced rows that cost less "
                            "than the cart line (a single step apiece):\n" + "\n".join(texts))
                if ev:
                    out += "\n" + ev
                return out
        return ""


RX_PROOF_TOK = re.compile(r"[a-z0-9+#]+")
RX_LISTSEP = re.compile(r"[,/;|]")
RX_OR_WHOLE = re.compile(r"\bor\b")
RX_V3_WHOLE = re.compile(r"^(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or|same|listing)$")
RX_AMOUNT_SCAN = re.compile(r"(?<![a-z0-9])\d[\d.,]*\s*[a-z\"']{1,12}")
RX_AMOUNT_HEAD = re.compile(r"^[\d.,\s]+")
RX_DEC = re.compile(r"\d\.\d")
RX_LOWER_ONLY = re.compile(r"^[a-z]+$")
RX_WHOLE_NUM = re.compile(r"^\d+(?:\.\d+)?$")
RX_NUM = re.compile(r"\d+(?:\.\d+)?")
RX_NUM_LET = re.compile(r"(\d+(?:\.\d+)?)\s*([a-z]*)")
RX_HAS_LETTER = re.compile(r"[a-z]")
RX_LETTER_RUN = re.compile(r"[a-z]+")
RX_ABBR_CUT = re.compile(r"\s*,\s*|\s+or\s+")
RX_OR_SPLIT = re.compile(r"\s+or\s+")
PLACEHOLDER_LEAD = "whatever the listing "


def proof_tokens(text):
    return RX_PROOF_TOK.findall(word_form(text))


def list_pieces(text):
    out = []
    for p in RX_LISTSEP.split(str(text)):
        t = proof_tokens(p)
        if t:
            out.append(set(t))
    return out


class G4B(object):
    def limit_in(self, value, apart_or=False):
        wf = word_form(value)
        if apart_or:
            wf = RX_OR_WHOLE.sub(" ", wf)
        return bool(RX_OPEN_EXCLUDE.search(wf))

    def judged(self, n):
        return [r for r in self.items[n - 1].live() if not self.described(n, r)]

    def measured_unit(self, req):
        if ":" not in req:
            return None
        v = req.split(":", 1)[1]
        raw = v.lower()
        unit = None
        for m in RX_AMOUNT_SCAN.finditer(raw):
            ut = RX_AMOUNT_HEAD.sub("", m.group(0))
            if RX_V3_WHOLE.match(ut):
                continue
            unit = ut
            break
        if unit is None:
            return None
        if not (RX_DEC.search(raw) or RX_OPEN_EXCLUDE.search(word_form(v))):
            return None
        return unit if RX_LOWER_ONLY.match(unit) else ""

    def bare_off(self, req, value):
        if not self.measured_unit(req):
            return None
        t = str(value).strip().lower().replace(",", "")
        if not RX_WHOLE_NUM.match(t):
            return None
        num = float(t)
        m = RX_NUM.search(req.split(":", 1)[1].replace(",", ""))
        tgt = float(m.group(0)) if m else 0.0
        if num <= 0 or tgt <= 0:
            return None
        if num >= 10 * tgt or 10 * num <= tgt:
            return num
        return None

    def misread(self, block, req):
        pr = roots(req.partition(":")[0])
        cands = [v for k, v in block.items() if roots(k) & pr and RX_HAS_DIGIT.search(str(v))]
        return bool(cands) and all(self.bare_off(req, v) is not None for v in cands)

    def gives_qty(self, block, prop, unit):
        pr = roots(prop)
        for k, v in block.items():
            if not (roots(k) & pr):
                continue
            t = str(v).lower()
            if not RX_HAS_DIGIT.search(t):
                continue
            if not unit or not RX_HAS_LETTER.search(t) or re.search(r"\d[\s-]*" + re.escape(unit), t):
                return True
        return False

    def split_wording(self, req, texts):
        val = req.split(":", 1)[1] if ":" in req else ""
        if not texts or RX_LISTSEP.search(val) or self.limit_in(val) or self.measured_unit(req) is not None:
            return False
        return split_run(val, " | ".join(texts))

    def lists_codes(self, lid, codes):
        m = self.fit_index().get(lid)
        if not m:
            return False
        acc = list(m["works"]) + list(m["models"])
        return all(any(code_covers(a, c) for a in acc) and not any(code_covers(b, c) for b in m["barred"]) for c in codes)

    def title_of(self, lid):
        tk = self.title_key() or "title"
        rec = self.full.get(lid) or {}
        return str(rec.get(tk) or rec.get("title") or (self.rows.get(lid) or {}).get("title") or "")

    def states_word(self, lid, req, strict=False):
        block = attrs_of(self.full.get(lid)) or {}
        prop, _, val = req.partition(":")
        toks = value_tokens(val)
        if not toks:
            return True
        ps = spelled(prop)
        keyed = [v for k, v in block.items() if names_match(ps, spelled(k))]
        if keyed:
            if self.split_wording(req, [str(v) for v in keyed]):
                return False
            if not strict:
                return True
            kt = [glued(dump(v)) for v in keyed]
            return (any(t in x for x in kt for t in toks)
                    or any(kin(t, w) for x in kt for w in RX_LETTER_RUN.findall(x) for t in toks))
        codes = find_codes(val)
        if codes and self.lists_codes(lid, codes):
            return True
        page = glued(" ".join([dump(v) for v in block.values()] + [self.title_of(lid)]))
        if any(t in page for t in toks) or any(kin(t, w) for w in RX_LETTER_RUN.findall(page) for t in toks):
            return True
        vis = set(RX_ALNUM.findall(page))
        for piece in RX_ABBR_CUT.split(val.lower()):
            runs = RX_LETTER_RUN.findall(piece)
            for i in range(len(runs)):
                for j in range(i + 3, len(runs) + 1):
                    if "".join(r[0] for r in runs[i:j]) in vis:
                        return True
        return False

    def proves(self, lid, n, reqs=None):
        rec = self.full.get(lid)
        block = attrs_of(rec) if rec else None
        if not block:
            return False
        t = rec.get("title") or (self.rows.get(lid) or {}).get("title")
        ttoks = set(proof_tokens(t)) if t else set()
        reqs = list(reqs) if reqs is not None else self.judged(n)
        if not reqs:
            return False
        inw = (spelled(self.items[n - 1].name) | set(["of"])) if 1 <= n <= len(self.items) else set()
        lines = []
        for r in reqs:
            prop, _, val = r.partition(":")
            vt = set(proof_tokens(val))
            if not vt or self.limit_in(val):
                return False
            ps = spelled(prop)
            mk = [k for k in block if names_match(ps, spelled(k))]
            fuller = [k for k in mk if spelled(k) > ps and spelled(k) <= (ps | inw)]
            if fuller:
                mx = max(len(spelled(k)) for k in fuller)
                fuller = [k for k in fuller if len(spelled(k)) == mx]
            deciding = fuller or [k for k in mk if spelled(k) == ps]
            disagree = [k for k in mk if spelled(k) == ps and not any(vt <= p for p in list_pieces(block[k]))] if fuller else []
            sat = False
            for k in (deciding or mk):
                v = block[k]
                pieces = list_pieces(v)
                cov = [p for p in pieces if vt <= p]
                if self.split_wording(r, [str(v)]):
                    cov = []
                plural = word_form(k) != word_form(prop) and spelled(k) == ps
                if (cov and plural and len(pieces) > 1 and any(not (vt <= p) and p <= ttoks for p in pieces)
                        and not any(p <= ttoks for p in cov)):
                    cov = []
                if deciding and not cov:
                    return False
                if cov:
                    sat = True
                    if len(lines) < len(reqs):
                        line = "%s: %s" % (k, clip(str(v), 160))
                        if disagree:
                            dk = disagree[0]
                            line += (" (this key extends the property's name with the requested item's name, and so it wins "
                                     "over the plain key %s, which gives %s)" % (dk, clip(str(block[dk]), 60)))
                        lines.append(line)
            if not sat:
                return False
        self.proofs[lid] = lines
        return True

    def literal(self, lid, req):
        u = self.measured_unit(req)
        if u is None:
            return None
        val = req.split(":", 1)[1].strip()
        fl = RX_FLOOR_PHRASE.search(val)
        if not fl and self.limit_in(val):
            return None
        m = RX_NUM.search((fl.group(1) if fl else val).replace(",", ""))
        if not m:
            return None
        want = float(m.group(0))
        ps = spelled(req.partition(":")[0])
        kept = []
        for k, v in (attrs_of(self.full.get(lid)) or {}).items():
            if not names_match(ps, spelled(k)):
                continue
            mm = RX_NUM_LET.search(str(v).lower().replace(",", ""))
            if not mm:
                continue
            let = mm.group(2)
            if not u or not let or let.rstrip("s") == u.rstrip("s"):
                kept.append(float(mm.group(1)))
        if not kept:
            return None
        if fl:
            return "yes" if any(x >= want - 1e-9 for x in kept) else "no"
        return "yes" if any(abs(x - want) <= 1e-9 for x in kept) else "no"

    def resolved(self, n):
        ptrs = set(t for i, t in self.refs if i == n)
        out = []
        for r in self.judged(n):
            if r in ptrs:
                v = self.ref_values.get(r, "")
                if not v or v.startswith(PLACEHOLDER_LEAD) or self.limit_in(v):
                    return None
                out.append(r.partition(":")[0].strip() + ": " + v)
            else:
                out.append(r)
        return out

    def pointer_items(self):
        live = self.live_all()
        return set(i for i, t in self.refs if t in live)

    def meets_words(self, lid, n, literal=False):
        if lid not in self.full or not self.buyable(lid) or self.sold_out(lid) or self.price(lid) is None:
            return False
        reqs = self.resolved(n)
        if not reqs or any(self.described(n, r) for r in self.items[n - 1].live()):
            return False
        block = attrs_of(self.full[lid]) or {}
        for r in reqs:
            prop, _, val = r.partition(":")
            mu = self.measured_unit(r) is not None
            if literal and mu:
                if self.literal(lid, r) != "yes":
                    return False
                continue
            if mu or self.limit_in(val, True):
                return False
            ps = spelled(prop)
            if any(names_match(ps, spelled(k)) for k in block):
                ch = [c.strip() for c in RX_OR_SPLIT.split(val.strip()) if c.strip()] or [val.strip()]
                if not any(self.proves(lid, n, [prop.strip() + ": " + c]) for c in ch):
                    return False
            elif not self.states_word(lid, r, True):
                return False
        return True

    def _code_fit(self, dev, piece):
        idx = self.fit_index()
        dc = (idx.get(dev) or {}).get("models") or []
        pm = idx.get(piece)
        if not dc or not pm:
            return False
        return (any(code_covers(c, x) for c in pm["fitset"] for x in dc)
                and not any(code_covers(b, x) for b in pm["barred"] for x in dc))

    def keeps_links(self, n, cand, cart):
        for part, main in self.fit_pairs:
            if n not in (part, main):
                continue
            partner = main if n == part else part
            pl = set(self.item_listing_ids(partner))
            plines = [c for c in cart if c in pl]
            if not plines:
                return False
            sysm = self.series_system(part, main)
            for ln in plines:
                dev, piece = (ln, cand) if n == part else (cand, ln)
                if sysm:
                    carried, naming = (piece, dev) if sysm[0] == part else (dev, piece)
                    if self.link_holds(carried, naming, sysm[2]) is not True:
                        return False
                elif not self._code_fit(dev, piece):
                    return False
        return True

    def _cf(self, n):
        return 1 if self.brief.get("budget_scope") == "each" else self.items[n - 1].qty

    @soft(list)
    def read_sets(self, literal=False):
        if not self.fit_pairs or len(set(m for p, m in self.fit_pairs)) != 1:
            return []
        main = self.fit_pairs[0][1]
        parts = sorted(set(p for p, m in self.fit_pairs))
        N = len(self.items)
        qual = {}
        for n in range(1, N + 1):
            q = [r for r in self.item_listing_ids(n) if self.meets_words(r, n, literal)]
            if not q:
                return []
            qual[n] = q
        rem = {}
        rtot = 0.0
        for n in range(1, N + 1):
            if n == main or n in parts:
                continue
            c = min(qual[n], key=lambda r: self.price(r))
            rem[c] = self.items[n - 1].qty
            rtot += self.price(c) * self._cf(n)
        sets = []
        for d in qual[main]:
            lines = dict(rem)
            lines[d] = self.items[main - 1].qty
            tot = rtot + self.price(d) * self._cf(main)
            ok = True
            for p in parts:
                sysm = self.series_system(p, main)
                fits = []
                for r in qual[p]:
                    if sysm:
                        carried, naming = (r, d) if sysm[0] == p else (d, r)
                        f = self.link_holds(carried, naming, sysm[2]) is True
                    else:
                        f = self._code_fit(d, r)
                    if f:
                        fits.append(r)
                if not fits:
                    ok = False
                    break
                c = min(fits, key=lambda r: self.price(r))
                lines[c] = self.items[p - 1].qty
                tot += self.price(c) * self._cf(p)
            if not ok or len(lines) != N:
                continue
            if _num(self.budget) and self.brief.get("budget_scope") != "each" and tot > self.budget + 1e-9:
                continue
            sets.append({"ids": list(lines.keys()), "lines": lines, "total": tot})
        sets.sort(key=lambda s: (s["total"], sorted(s["ids"])))
        return sets

    def set_text(self, sets):
        out = []
        for s in sets[:CHEAPEST_SETS]:
            ls = " + ".join(("%s x%d" % (i, q)) if q > 1 else i for i, q in s["lines"].items())
            out.append("%s = %s" % (ls, gnum(s["total"])))
        return "; ".join(out)

    def _run_calls(self, calls, observe=True):
        done = []
        for tool, args in calls:
            f = []
            a = shape(args, self.schemas.get(tool), "args", f)
            if f:
                return done, False
            res = self.env(tool, a)
            if res is None or self.finished:
                return done, False
            self.note_cart(tool, a, res)
            if observe:
                self.observe(tool, res.get("observation"))
            done.append(dump(a))
        return done, True

    @soft(False)
    def cart_set(self, want):
        cids = self.cart_ids()
        rem = set(r for r in cids if r not in want)
        chg = dict((k, v) for k, v in want.items() if self.cart.get(k) != v)
        calls = self.cart_calls(chg, rem) if (chg or rem) else []
        if calls is None or not self.roles.get("cart") or len(calls) + 1 > self.steps_left():
            return False
        done, ok = self._run_calls(calls)
        if not ok:
            return False
        self.version += 1
        self.trace.append({"kind": "cart_set", "lines": want})
        return True

    def cheaper_proven(self):
        ck = self.category_key()
        if self.pick() != "cheapest" or not ck:
            return []
        cart = [r for r in self.cart_ids() if self.units_held(r) > 0]
        out = []
        N = len(self.items)
        for ln in cart:
            lab = (self.rows.get(ln) or {}).get(ck)
            own = [n for n in range(1, N + 1) if self.owns_label(n, lab)]
            p = self.price(ln)
            if len(own) != 1 or p is None:
                continue
            rr = self.resolved(own[0])
            if rr is None:
                continue
            cands = []
            for r in list(self.full.keys()):
                if r in cart or r in self.gone or not self.buyable(r) or self.sold_out(r):
                    continue
                if str((self.rows.get(r) or {}).get(ck)) != str(lab):
                    continue
                q = self.price(r)
                if q is None or not (q < p - 1e-9):
                    continue
                if self.proves(r, own[0], rr) and self.keeps_links(own[0], r, cart):
                    cands.append(r)
            if cands:
                c = min(cands, key=lambda r: self.price(r))
                self.proves(c, own[0], rr)
                out.append((ln, p, c, self.price(c)))
        return out

    def cheaper_text(self, recs):
        ents = []
        for ln, p, c, q in recs:
            e = "%s at %s instead of %s at %s" % (c, gnum(q), ln, gnum(p))
            if self.proofs.get(c):
                e += " (" + "; ".join(self.proofs[c]) + ")"
            ents.append(e)
        return ("The request chooses by lowest price, and an already-opened listing costs less and gives every requirement "
                "beneath the attribute key carrying that requirement's property: " + "; ".join(ents) + ". Swap that line, "
                "by the means the cart tool offers for removing and adding units, and then check out; the current line may "
                "stay only if the cheaper listing's record clearly violates one of the requirements." + self.swap_links(recs))

    @soft("")
    def swap_links(self, recs):
        bound = set(x for p in self.fit_pairs for x in p)
        if not bound:
            return ""
        kept = []
        cart = self.cart_ids()
        for ln, p, c, q in recs:
            if not any(ln in self.item_listing_ids(b) for b in bound):
                continue
            ids = [c if x == ln else x for x in cart]
            for l in self.code_fits(ids):
                if l.startswith(c) or re.search(r"(?:^| )" + re.escape(c) + r"(?: |$)", l):
                    kept.append(l)
        if not kept:
            return ""
        return (" Fit by code: " + "; ".join(kept[:3]) + ". The store settles compatibility through these codes; a purpose, "
                "a size or a count of seats that some description mentions cannot overturn that.")

    @soft("")
    def agreed_swap(self):
        if self.swap_agreed or self.pick() != "cheapest":
            return ""
        cids = self.cart_ids()
        better = self.check_better.get("|".join(sorted(cids)) + "#" + str(self.version), [])

        def link_ok(old, new):
            for n in range(1, len(self.items) + 1):
                if old in self.item_listing_ids(n):
                    return any(n in pr for pr in self.fit_pairs) and self.keeps_links(n, new, cids)
            return False
        pairs = [(ln, c) for ln, p, c, q in self.cheaper_proven() if (ln, c) in self.cheaper_named and (c in better or link_ok(ln, c))]
        if not pairs:
            return ""
        want = dict((c, self.cart.get(ln) or 1) for ln, c in pairs)
        calls = self.cart_calls(want, set(ln for ln, c in pairs))
        if not self.roles.get("cart") or not calls or self.steps_left() < len(calls) + 2 + self.reserve():
            return ""
        self.swap_agreed = True
        done, _ = self._run_calls(calls)
        cids2 = self.cart_ids()
        for ln, c in pairs:
            if ln not in cids2:
                self.dropped_ids.add(ln)
            if c in cids2:
                self.dropped_ids.discard(c)
        live = [r for r in cids2 if not self.sold_out(r) and r not in self.dropped_ids]
        if len(live) >= len(self.built):
            self.built = live
        self.version += 1
        self.trace.append({"kind": "agreed_swap", "pairs": pairs, "calls": done})
        if all(c in better for ln, c in pairs):
            why = "Cairn's reading of the pages and the separate review both point to the cheaper listing"
        else:
            why = "Cairn's reading points to the cheaper listing, whose own codes preserve compatibility with the remaining cart lines"
        return ("Cairn carried out the swap on its own, since %s: %s (%s). Check out now."
                % (why, "; ".join("%s instead of %s" % (c, ln) for ln, c in pairs), "; ".join(done)))

    def word_shown(self, ln, req):
        ck = self.category_key()
        lab = str((self.rows.get(ln) or {}).get(ck)) if ck else ""
        same = [r for r in self.seen if r != ln and ((str(self.rows[r].get(ck)) if ck else "") == lab)]
        opened = [r for r in same if attrs_of(self.full.get(r))]
        if len(opened) < 3 or any(self.states_word(r, req, True) for r in opened):
            return True
        toks = value_tokens(req.split(":", 1)[1] if ":" in req else "")
        tk = self.title_key() or "title"
        for r in same:
            t = glued(str(self.rows[r].get(tk) or ""))
            if any(x in t for x in toks):
                return True
        return False

    def weak_lines(self):
        ck = self.category_key()
        if not ck:
            return []
        ptrs = set(t for _, t in self.refs)
        out = []
        N = len(self.items)
        for ln in self.cart_ids():
            if self.units_held(ln) <= 0:
                continue
            block = attrs_of(self.full.get(ln))
            own = [n for n in range(1, N + 1) if self.owns_label(n, (self.rows.get(ln) or {}).get(ck))]
            if not block or len(own) != 1:
                continue
            n = own[0]
            weak = []
            for r in self.items[n - 1].live():
                if r in ptrs or ":" not in r or self.described(n, r):
                    continue
                u = self.measured_unit(r)
                if u is not None:
                    if not self.gives_qty(block, r.split(":", 1)[0], u) or self.misread(block, r):
                        weak.append(r)
                elif not self.limit_in(r.split(":", 1)[1], True) and not self.states_word(ln, r) and self.word_shown(ln, r):
                    weak.append(r)
            if weak:
                out.append((ln, n, weak))
        return out

    def weak_text(self):
        wl = self.weak_lines()
        if not wl:
            return ""
        ck = self.category_key()
        cart = self.cart_ids()
        each = self.brief.get("budget_scope") == "each" and _num(self.budget)
        parts = []
        for ln, n, reqs in wl[:2]:
            block = attrs_of(self.full.get(ln)) or {}
            props = [r.split(":", 1)[0].strip() for r in reqs]
            lab = str((self.rows.get(ln) or {}).get(ck))
            alts = [r for r in self.seen if r not in cart and r != ln and str(self.rows[r].get(ck)) == lab and self.buyable(r)
                    and not self.sold_out(r) and r not in self.gone and (not each or (self.price(r) or 0) <= self.budget)]

            def strong(r):
                b = attrs_of(self.full.get(r))
                if not b:
                    return False
                for q in reqs:
                    u = self.measured_unit(q)
                    if u is not None:
                        if not self.gives_qty(b, q.split(":", 1)[0], u) or self.misread(b, q):
                            return False
                    elif not self.states_word(r, q, True):
                        return False
                return True
            strongs = [r for r in alts if r in self.full and strong(r)]
            unopened = [r for r in alts if r not in self.full]
            proots = set()
            for p in props:
                proots |= roots(p)
            lrec = self.full[ln] if ln in self.full else (self.rows.get(ln) or {})
            ltw = roots(str(lrec.get("title") or ""))
            unopened.sort(key=lambda r: (not (roots(dump(self.rows[r])) & proots),
                                         -len(roots(str(self.rows[r].get("title") or "")) & ltw),
                                         self.price(r) is None, self.price(r) or 0))
            sizes = [r.split(":", 1)[0].strip() for r in reqs if self.measured_unit(r) is not None]
            words = [r.strip() for r in reqs if self.measured_unit(r) is None]
            cl = []
            if sizes:
                cl.append("%s fails to give %s under a key bearing each property's name (a number filed under some other "
                          "quantity's key, a number printed within the title, or no number whatever leaves grading unable "
                          "to decide the requirement, and the order gets failed on that account)" % (ln, ", ".join(sizes)))
                mis = [r.split(":", 1)[0].strip() for r in reqs if self.measured_unit(r) is not None and self.misread(block, r)]
                if mis:
                    cl.append("%s gives %s only as a number without a unit; such a number is taken in the unit the "
                              "requirement uses, and so taken it lies far outside what the requirement allows" % (ln, ", ".join(mis)))
            if words:
                cl.append("neither any attribute's value nor the title of %s contains the value wanted by %s (a neighbouring "
                          "attribute carrying a different sort of value, such as a series of numbers, or wording present "
                          "solely in the line's description, keeps the requirement undecided unless the page expresses it "
                          "some other way)" % (ln, "; ".join(words)))
            t = "; ".join(cl)
            if strongs:
                strongs.sort(key=lambda r: (self.price(r) is None, self.price(r) or 0))
                t += ("; the following already-opened listings of that same kind give it beneath the key of that very "
                      "property: " + ", ".join(("%s (%s)" % (r, gnum(self.price(r)))) if self.price(r) is not None else r
                                               for r in strongs[:3]))
            elif unopened and self.steps_left() > self.reserve() + 5:
                t += "; open first these same-kind listings: " + ", ".join(unopened[:3])
            else:
                continue
            parts.append(t)
        if not parts:
            return ""
        self.trace.append({"kind": "weak_hold", "lines": [w[0] for w in wl]})
        return (" | ".join(parts) + ". Change to a listing that both states each requirement (a quantity bound beneath the "
                "key of its own property; a wanted word inside an attribute or inside the title) and satisfies it, while "
                "obeying the pick rule; after that, check out. The line may stay only if there is no listing that does both.")

    def proven_basket(self):
        if not self.roles.get("cart") or not self.items:
            return False
        if self.pointer_items() | set(x for p in self.fit_pairs for x in p):
            return False
        lines = {}
        tot = 0.0
        for n in range(1, len(self.items) + 1):
            c = [r for r in self.item_listing_ids(n) if r in self.full and self.buyable(r) and not self.sold_out(r)
                 and self.price(r) is not None and self.proves(r, n)]
            if not c:
                return False
            b = min(c, key=lambda r: self.price(r))
            lines[b] = self.items[n - 1].qty
            tot += self.price(b) * self._cf(n)
        if _num(self.budget) and self.brief.get("budget_scope") != "each" and tot > self.budget + 1e-9:
            return False
        calls = self.cart_calls(lines, set())
        if not calls or len(calls) + 1 > self.steps_left():
            return False
        for tool, args in calls:
            f = []
            a = shape(args, self.schemas.get(tool), "args", f)
            if f:
                return False
            res = self.env(tool, a)
            if res is None:
                return False
            self.note_cart(tool, a, res)
        self.trace.append({"kind": "proven_basket", "lines": lines})
        return True


class G5A(object):
    @soft(dict)
    def proven_lines(self):
        each = self.brief.get("budget_scope") == "each"
        N = len(self.items)
        out = {}
        tot = 0.0
        for n in range(1, N + 1):
            cands = list(self.full.keys()) if N == 1 else self.item_listing_ids(n)
            cands = [r for r in cands if r not in self.gone and self.meets_words(r, n, True)]
            if each and _num(self.budget):
                cands = [r for r in cands if (self.price(r) or 0) <= self.budget + 1e-9]
            if not cands:
                return {}
            c = sorted(cands, key=lambda r: (self.price(r) or 0, str(r)))[0]
            out[n] = c
            tot += (self.price(c) or 0) * self.items[n - 1].qty
        if _num(self.budget) and not each and tot > self.budget + 1e-9:
            return {}
        return out

    def line_fit(self, n, cand, dev):
        idx = self.fit_index()
        for part, main in self.fit_pairs:
            if part != n or not dev or dev not in self.item_listing_ids(main):
                continue
            sysm = self.series_system(part, main)
            if sysm:
                carried, naming = (cand, dev) if sysm[0] == part else (dev, cand)
                h = self.link_holds(carried, naming, sysm[2])
                return 1 if h is True else (-1 if h is False else 0)
            dc = (idx.get(dev) or {}).get("models") or []
            cm = idx.get(cand) or {}
            fs = cm.get("fitset") or set()
            if not dc or not fs:
                return 0
            ok = (any(code_covers(c, x) for c in fs for x in dc)
                  and not any(code_covers(b, x) for b in cm.get("barred") or [] for x in dc))
            return 1 if ok else -1
        return 0

    def need_hits(self, lid, n):
        ptrs = set(t for i, t in self.refs if i == n)
        block = attrs_of(self.full.get(lid)) or {}
        met = judged = 0
        for r in self.items[n - 1].live():
            if self.described(n, r):
                continue
            if r in ptrs:
                v = self.ref_values.get(r, "")
                if not v or v.startswith(PLACEHOLDER_LEAD) or RX_OPEN_EXCLUDE.search(word_form(v)):
                    continue
                r = r.partition(":")[0].strip() + ": " + v
            prop, _, val = r.partition(":")
            if self.measured_unit(r) is not None:
                if not RX_FLOOR_PHRASE.search(val.strip()) and RX_OPEN_EXCLUDE.search(word_form(val)):
                    continue
                ok = self.literal(lid, r) == "yes"
            else:
                if RX_OPEN_EXCLUDE.search(RX_OR_WHOLE.sub(" ", word_form(val))):
                    continue
                ps = spelled(prop)
                if any(names_match(ps, spelled(k)) for k in block):
                    ch = [c.strip() for c in RX_OR_SPLIT.split(val.strip()) if c.strip()] or [val.strip()]
                    ok = any(self.proves(lid, n, [prop.strip() + ": " + c]) for c in ch)
                else:
                    ok = self.states_word(lid, r, True)
            judged += 1
            if ok:
                met += 1
        return met, judged

    @soft(False)
    def closest_cart(self):
        ck = self.category_key()
        N = len(self.items)
        if not self.roles.get("cart") or not self.roles.get("buy") or not ck or self.finished or N < 2:
            return False
        cur = [r for r in self.cart_ids() if self.units_held(r) > 0]
        if not cur and not self.ever_carted:
            return False
        each = self.brief.get("budget_scope") == "each"

        def owner(r):
            return self.label_owner((self.rows.get(r) or {}).get(ck)) or 0

        def cu(n):
            return 1 if each else self.items[n - 1].qty
        leads = {}
        for k, v in self.lead_items.items():
            if str(k).isdigit():
                leads[int(k)] = v
        lines = {}
        for r in cur:
            o = owner(r)
            if 1 <= o <= N and o not in lines and self.buyable(r) and r not in self.gone:
                lines[o] = r
        kept = dict(lines)

        def device(n):
            for part, main in self.fit_pairs:
                if part == n and main in lines:
                    return lines[main]
            return None

        def pool(n):
            used = set(lines.values())
            return [r for r in self.item_listing_ids(n) if r in self.full and self.buyable(r) and r not in self.gone
                    and self.price(r) is not None and r not in used]

        def rank(n, c):
            fs = self.line_fit(n, c, device(n))
            for part, main in self.fit_pairs:
                if main == n and part in lines:
                    fs += self.line_fit(part, lines[part], c)
            return (fs, c in self.ever_carted, c in leads.get(n, []), self.need_hits(c, n)[0], -(self.price(c) or 0), str(c))

        def held_sum():
            s = 0.0
            for m, r in lines.items():
                if kept.get(m) == r:
                    u = self.units_held(r) or 1
                    s += (self.price(r) or 0) * u
                else:
                    s += (self.price(r) or 0) * cu(m)
            return s

        def room(cost):
            if not _num(self.budget):
                return True
            if each:
                return cost <= self.budget + 1e-9
            return held_sum() + cost <= self.budget + 1e-9

        def ok_need(c, n):
            m, j = self.need_hits(c, n)
            return m > 0 or j == 0
        mains = sorted(set(m for p, m in self.fit_pairs))
        order = mains + [n for n in range(1, N + 1) if n not in mains]
        swapped = []
        for n in order:
            if n in lines:
                old = lines[n]
                if self.line_fit(n, old, device(n)) >= 0:
                    continue
                del lines[n]
                dv = device(n)
                cands = sorted([c for c in pool(n) if self.line_fit(n, c, dv) == 1 and ok_need(c, n)],
                               key=lambda c: rank(n, c), reverse=True)
                pick = next((c for c in cands if room((self.price(c) or 0) * cu(n))), None)
                if pick:
                    lines[n] = pick
                    swapped.append(old)
                else:
                    lines[n] = old
            else:
                dv = device(n)
                opts = sorted([c for c in pool(n) if self.line_fit(n, c, dv) >= 0 and ok_need(c, n)],
                              key=lambda c: rank(n, c), reverse=True)
                pick = next((c for c in opts if room((self.price(c) or 0) * cu(n))), None)
                if not pick:
                    self.trace.append({"kind": "closest_cart_gap", "item": n, "options": len(opts)})
                    return False
                lines[n] = pick
        want = {}
        for n, r in lines.items():
            if kept.get(n) == r:
                want[r] = self.cart.get(r) or self.units_held(r) or 1
            else:
                want[r] = self.items[n - 1].qty
        drops = set(swapped)
        for r in cur:
            if r not in want and r not in drops and (not self.buyable(r) or r in self.gone) and owner(r) in lines:
                drops.add(r)
        calls = self.cart_calls(want, drops)
        if calls is None or len(calls) + 1 > self.steps_left():
            return False
        made = 0
        for tool, args in calls:
            f = []
            a = shape(args, self.schemas.get(tool), "args", f)
            if f:
                return False
            res = self.env(tool, a)
            if res is None or self.finished:
                return False
            self.note_cart(tool, a, res)
            self.observe(tool, res.get("observation"))
            made += 1
        if made:
            self.version += 1
        self.trace.append({"kind": "closest_cart", "want": want, "drops": sorted(drops), "kept": sorted(kept.values())})
        return True

    def code_fits(self, ids):
        idx = self.fit_index()
        out = []
        for part, main in self.fit_pairs:
            ml = set(self.item_listing_ids(main))
            devs = [i for i in ids if i in ml]
            if len(devs) != 1:
                continue
            d = devs[0]
            sysm = self.series_system(part, main)
            dc = sorted(set((idx.get(d) or {}).get("models") or []))
            pl = set(self.item_listing_ids(part))
            for p in ids:
                if p not in pl:
                    continue
                pn, mn = self.items[part - 1].name, self.items[main - 1].name
                if sysm:
                    carried, naming = (p, d) if sysm[0] == part else (d, p)
                    if self.link_holds(carried, naming, sysm[2]) is True:
                        sh = sorted(set(idx[carried]["own_series"]) & set(idx[naming]["named_series"]) & sysm[2])
                        out.append("%s (%s) and %s (%s): %s bears the series %s, which %s names"
                                   % (p, pn, d, mn, carried, ", ".join(sh), naming))
                    continue
                pm = idx.get(p) or {}
                cov = [x for x in dc if any(code_covers(c, x) for c in pm.get("fitset") or [])
                       and not any(code_covers(b, x) for b in pm.get("barred") or [])]
                if cov:
                    out.append("%s (%s) names the code that identifies the device %s (%s): %s"
                               % (p, pn, d, mn, ", ".join(c.upper() for c in cov)))
        ck = self.category_key()
        N = len(self.items)
        for i in ids:
            own = [n for n in range(1, N + 1) if self.owns_label(n, (self.rows.get(i) or {}).get(ck) if ck else None)]
            if len(own) != 1:
                continue
            m = idx.get(i) or {}
            acc = list(m.get("works") or []) + list(m.get("models") or [])
            for r in self.items[own[0] - 1].live():
                codes = find_codes(r.split(":", 1)[1] if ":" in r else "")
                if codes and all(any(code_covers(a, c) for a in acc) and not any(code_covers(b, c) for b in m.get("barred") or [])
                                 for c in codes):
                    out.append("%s satisfies the requirement '%s': its fit clause lists %s" % (i, r, ", ".join(c.upper() for c in codes)))
        return out


RX_VTOK = re.compile(r"[a-z0-9][a-z0-9.]*")
RX_SQUASH_P = re.compile(r"[^a-z0-9.]")
RX_SQUASH = re.compile(r"[^a-z0-9]")
RX_QSEP = re.compile(r"[,;]+")
RX_KEYWORD3 = re.compile(r"[a-z]{3,}")
SEEN_HEAD = ("[Cairn opened the pages below while tracing the listings that requirements refer to; a page qualifies as one of "
             "them only if it displays every value stated in that listing's description]")


class G5B(object):
    def bounded_show(self, v, limit):
        t = self.render(v)
        if len(t) <= limit:
            return t
        return rec_dump(v, ROW_CLIP if self.obs_listings(v) else TEXT_CLIP, limit)

    def shop_names(self, lid):
        return [c for c in self._short_vals(lid, self._noname_keys()) if len(c) > 2]

    def names_shop(self, lid, text):
        low = str(text).lower()
        return any(edge_in(c.lower(), low) for c in self.shop_names(lid))

    def key_words(self):
        out = set()
        for rec in self.full.values():
            for k in (attrs_of(rec) or {}):
                out |= set(RX_KEYWORD3.findall(word_form(k)))
        return out

    def read_misses(self, lid, vw):
        if not vw or lid not in self.full:
            return False
        b = attrs_of(self.full.get(lid))
        lv = " ".join(str(x) for x in b.values()).lower() if b else ""
        if not lv:
            return False
        return any(not edge_in(w, lv) for w in vw)

    def shop_rows(self, desc, ids, vw=None):
        return [r for r in ids if r not in self.opened and r in self.rows and not (vw and self.read_misses(r, vw))
                and self.names_shop(r, desc)]

    def chain_match(self, text, pend):
        if text in pend:
            return text
        if len(pend) == 1:
            return pend[0]
        ws = set(RX_ALNUM.findall(text.lower()))
        if not ws or len(pend) < 2:
            return None
        sc = []
        for p in pend:
            pw = set(RX_ALNUM.findall(p.lower()))
            sc.append(len(ws & pw) / float(max(1, len(ws | pw))))
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        if sc[order[0]] >= 0.6 and sc[order[0]] - sc[order[1]] > 0.15:
            return pend[order[0]]
        return None

    def map_link_id(self, t):
        if not t or t in self.rows:
            return t
        cont = [r for r in self.seen if len(r) >= 6 and r in t]
        if len(cont) == 1:
            return cont[0]
        if len(t) < 3 or len(t) > 40 or not RX_HAS_DIGIT.search(t):
            return t
        low = t.lower()
        tk = self.title_key()
        if tk:
            hits = [r for r in self.seen if edge_in(low, str(self.rows[r].get(tk) or "").lower())]
            if len(hits) == 1:
                self.trace.append({"kind": "link_id", "from": t, "to": hits[0]})
                return hits[0]
            if hits:
                return t
        hits = [r for r in self.seen if edge_in(low, dump(self.full.get(r) or self.rows.get(r)).lower())]
        if len(hits) == 1:
            self.trace.append({"kind": "link_id", "from": t, "to": hits[0]})
            return hits[0]
        return t

    def lacks_values(self, lid, vtext, src):
        blocks = [b for b in (attrs_of(self.full.get(lid)), attrs_of(self.rows.get(lid))) if b]
        if not blocks or not str(vtext).strip():
            return []
        nw = set()
        for c in self.shop_names(lid):
            nw |= set(RX_ALNUM.findall(c.lower()))
        tk = self.title_key()
        held = " ".join(dump(b) for b in blocks) + " " + (str((self.rows.get(lid) or {}).get(tk) or "") if tk else "")
        held = RX_SQUASH_P.sub("", held.lower())
        raw = [x.strip(".") for x in RX_VTOK.findall(str(vtext).lower())]
        wanted = []
        for i, w in enumerate(raw):
            if not w or not (len(w) >= 3 or w[0].isdigit()) or w in nw:
                continue
            if w.isalpha() and i > 0 and raw[i - 1][:1].isdigit():
                continue
            wanted.append(w)
        if src is not None:
            sl = src.lower()
            wanted = [w for w in wanted if (w[0].isdigit() or roots(w)) and edge_in(w, sl)]
        out = []
        for w in wanted:
            if w not in held and w not in out:
                out.append(w)
        return out

    def sends_on(self, lid, req, val):
        phrase = " ".join(RX_ALNUM.findall(req.partition(":")[0].lower()))
        wv = RX_SQUASH.sub("", str(val).lower())
        if len(phrase) < 3 or not wv:
            return False
        tk = self.title_key()
        for rec in (self.rows.get(lid) or {}, self.full.get(lid) or {}):
            for k, v in rec.items():
                if k == tk or not isinstance(v, str) or len(v) <= 40:
                    continue
                if edge_in(phrase, " ".join(RX_ALNUM.findall(v.lower()))) and wv not in RX_SQUASH.sub("", v.lower()):
                    return True
        return False

    def trail(self, limit=None):
        pend = self.pending_refs()
        tool = self.roles.get("offer") or self.roles.get("open")
        tgt = self.single_target(tool) if tool else None
        seen_lines = []
        lost = []
        while pend:
            cap = self.chain_room if self.chain_room is not None else (6 if self.steps_left() > 15 else 4)
            if self.chain_rounds >= cap or self.spend >= HARD_SPEND or (limit is not None and self.clock() > limit):
                break
            self.chain_rounds += 1
            recs = []
            nopen = 0
            for rid in reversed(self.seen):
                rec = self.full.get(rid) or self.rows.get(rid)
                if self.page_read(rid) and rec is not None and nopen < 45:
                    recs.append("OPENED " + rec_dump(rec, 300, 1400))
                    nopen += 1
            for rid in reversed(self.seen):
                if len(recs) >= 60:
                    break
                rec = self.full.get(rid) or self.rows.get(rid)
                if not self.page_read(rid) or rec is None:
                    if rid in self.full:
                        recs.append("ROW " + dump(clip(self.full[rid], 120))[:700])
                    else:
                        recs.append("ROW " + dump(clip(self.rows.get(rid), 80))[:300])
            p = (CHAIN_PROMPT + "\n\nREQUIREMENTS WHOSE VALUE SITS ON ANOTHER LISTING:\n" + "\n".join(t for _, t in pend)
                 + "\n\nTHE SHOPPER'S REQUEST (each requirement listed above is a brief restatement of this request; wherever "
                   "the request's wording itself states how the desired value stands against the value found along the "
                   "chain, that wording fixes the value, overriding the requirement's wording):\n" + self.request[:3000])
            if self.trail_notes:
                p += "\n\nWHAT CAIRN VERIFIED AGAINST THE RECORDS:\n" + "\n".join(self.trail_notes[-6:])
            p += "\n\nRECORDS (newest first):\n" + "\n".join(recs)
            a = self.ask_json(p, [LIGHT], 4500)
            if (a is None and self.spend + HEAVY_ROOM <= HARD_SPEND and not self.store_broke and not self.no_credit
                    and self.model_calls < CALL_CAP):
                a = self.ask_json(p, [HEAVY], 4500)
            a = a or {}
            reach = []
            lost = []
            gaps = []
            again = False
            ptxt = [t for _, t in pend]
            for ch in items_of(a.get("chains")):
                if not isinstance(ch, dict):
                    continue
                req = self.chain_match(str(ch.get("need") or "").strip(), ptxt)
                if req is None:
                    continue
                links = [dict(l) for l in items_of(ch.get("links")) if isinstance(l, dict)]
                ids = [self.map_link_id(str(l.get("id") or "").strip()) for l in links]
                for pi, l in enumerate(links):
                    lid = ids[pi]
                    if pi == 0:
                        src = req
                    else:
                        prev = ids[pi - 1]
                        src = dump(self.full.get(prev) or self.rows.get(prev) or {}) if prev else None
                    vv = l.get("values")
                    vtext = vv if vv not in (None, "") else l.get("find")
                    vtext = "" if vtext is None else (vtext if isinstance(vtext, str) else str(vtext))
                    miss = []
                    if lid and (self.page_read(lid) or lid in self.full) and src:
                        miss = self.lacks_values(lid, vtext, src)
                        if miss and not self.page_read(lid):
                            kw = self.key_words()
                            if not any(w in kw for w in miss):
                                miss = []
                    if not miss:
                        continue
                    note = ('%s is not the listing portrayed by "%s": its attributes and its title both lack %s'
                            % (lid, str(l.get("about") or "")[:150], ", ".join(miss[:3])))
                    if note in self.trail_notes:
                        continue
                    self.trail_notes.append(note)
                    self.trace.append({"kind": "link_rejected", "id": lid, "missing": miss[:3]})
                    low = (req + " " + str(l.get("about") or "")).lower()
                    shop = next((c for c in self.shop_names(lid) if edge_in(c.lower(), low)), "")
                    vs = vv if isinstance(vv, str) else ("" if vv is None else str(vv))
                    sw = str(l.get("find") or "").strip() or (shop.lower() + " " + vs).strip()
                    nl = dict(l)
                    nl["id"] = ""
                    nl["find"] = sw
                    links = links[:pi] + [nl]
                    ids = ids[:pi] + [""]
                    again = True
                    break
                val = str(ch.get("value") or "").strip()
                allread = bool(links) and all(self.page_read(i) for i in ids)
                if ch.get("done") is True and allread and val:
                    note = ('%s mentions "%s" in its own prose but without the value %s: it therefore hands that property '
                            'over to another listing, and the chain continues from whatever its prose points to'
                            % (ids[-1], req[:80], val[:40]))
                    if self.sends_on(ids[-1], req, val) and note not in self.trail_notes:
                        self.trail_notes.append(note)
                        self.trace.append({"kind": "sends_on", "id": ids[-1]})
                        again = True
                        continue
                    self.ref_values[req] = val
                    self.version += 1
                    continue
                if allread and not str(links[-1].get("find") or "").strip():
                    self.ref_values[req] = val or (PLACEHOLDER_LEAD + "%s shows" % ids[-1])
                    self.version += 1
                    continue
                unread = [i for i in ids if i in self.rows and not self.page_read(i)]
                if unread:
                    for i in unread:
                        if i not in reach:
                            reach.append(i)
                    continue
                lk = next((l for l, i in zip(links, ids) if i not in self.rows), None) or {}
                desc = str(lk.get("about") or "").strip() or "the listing this requirement leads to"
                words = str(lk.get("find") or "").strip()
                lost.append("%s -> find %s%s" % (req, desc, (" (search: %s)" % words) if words else ""))
                gaps.append((desc, words))
            pend = [(i, t) for i, t in pend if t not in self.ref_values]
            if not reach and gaps and tool and tgt and self.steps_left() > 12:
                settled = []
                for gi, (desc, words) in enumerate(gaps[:2]):
                    if self.link_searches >= CHAIN_SEARCHES:
                        break
                    got, known = self.find_link(desc, words)
                    for x in got:
                        if x not in reach:
                            reach.append(x)
                    if known:
                        seen_lines.append("%s: a page, already opened, of the shop the chain names, and its attributes "
                                          "contain every value stated by its description; take the value off that page."
                                          % known)
                        settled.append(gi)
                lost = [l for i, l in enumerate(lost) if i not in settled]
            if not reach or not tool or not tgt or self.finished:
                if again and pend and tool and tgt and not self.finished:
                    continue
                break
            if self.chain_room is not None:
                room = min(3, EARLY_OPENS - len(seen_lines), self.steps_left() - (3 * len(self.items) + self.reserve() + 1))
            else:
                room = min(3, 6 - len(seen_lines)) if self.steps_left() >= len(reach) + 4 else 0
            if room <= 0:
                break
            for rid in reach[:room]:
                f = []
                args = shape({tgt: rid}, self.schemas.get(tool), "args", f)
                if f or self.rushed():
                    break
                res = self.env(tool, args)
                if res is None or self.finished:
                    break
                obs = res.get("observation")
                self.observe(tool, obs)
                seen_lines.append("%s: %s" % (rid, self.bounded_show(obs, 1500)))
        out = []
        if seen_lines:
            out.append(SEEN_HEAD + "\n" + "\n".join(seen_lines))
        if self.ref_values:
            out.append("[values taken from those pages] " + "; ".join("%s -> %s" % (k, v) for k, v in self.ref_values.items()))
        hold = False
        if lost and self.chain_holds < (3 if self.steps_left() > 15 else 2) and self.steps_left() > 5:
            self.chain_holds += 1
            hold = True
            out.append("The purchase is only valid once you have read completely every page that a requirement leads to (a "
                       "result row will not do), including each page that leads on to yet another listing. Not yet located "
                       "or read: %s. Search using the shop's name plus the values in the description (another product of "
                       "that kind sold by the same shop is the wrong one), open whichever row from that shop has a page "
                       "showing those values, use the value printed on that page, and then order." % " | ".join(lost[:3]))
        return "\n".join(out), hold

    def find_link(self, desc, words):
        bad = self._noname_keys()
        sr = self.shop_rows(desc, self.seen)
        osl = self.opened_shop(desc)
        nw = set()
        for r in sr + osl:
            for v in self._short_vals(r, bad):
                nw |= set(RX_RUN3.findall(v.lower()))
        src = words or desc
        vw = [w for w in RX_RUN_ANY.findall(src.lower()) if (len(w) >= 3 or w.isdigit()) and w not in nw and not RX_LINK_STOP1.match(w)]
        for r in osl:
            b = attrs_of(self.full.get(r))
            lv = " ".join(str(x) for x in b.values()).lower() if b else ""
            if vw and lv and all(edge_in(w, lv) for w in vw):
                self.trace.append({"kind": "known_link", "id": r})
                return [], r
        kw = self.key_words()
        p0 = RX_QSEP.sub(" ", words).split()
        parts = [p for i, p in enumerate(p0) if i == 0 or p.lower() not in kw]
        vt = self.value_template()
        vws = set(vw)
        votes = {}
        vorder = []
        if vt and parts:
            shop = parts[0]
            singles = [shop + " " + p for p in parts[1:] if len(p) >= 3 or RX_HAS_DIGIT.search(p)] if len(parts) >= 3 else []
            seq = []
            for q in [" ".join(parts)] + singles + [shop]:
                if q not in seq:
                    seq.append(q)
            for q in seq:
                if q == shop and votes:
                    break
                got = self.link_query(vt, q, desc, vws)
                if got is None:
                    break
                fresh = self.shop_rows(desc, got, vws)
                while not fresh and got and self.link_pages.get(q.lower()) and all(self.names_shop(x, desc) for x in got):
                    g2 = self.link_query(vt, q, desc, vws)
                    if not g2 or len(g2) == len(got):
                        break
                    got = g2
                    fresh = self.shop_rows(desc, got, vws)
                for r in fresh:
                    if r not in votes:
                        votes[r] = []
                        vorder.append(r)
                    votes[r].append(len(got))
                if fresh and q not in singles:
                    break
        terms = set(w for w in RX_RUN3.findall((desc + " " + words).lower()) if w not in nw and not RX_LINK_STOP2.match(w))

        def rank(ids, keep_all):
            sc = []
            for i, r in enumerate(ids):
                h = len(terms & set(RX_RUN3.findall(dump(self.rows.get(r)).lower())))
                if h or keep_all or not terms:
                    sc.append((-h, i, r))
            return [r for _, _, r in sorted(sc)]
        if vorder:
            ranked = rank(vorder, True)
            pos = dict((r, i) for i, r in enumerate(ranked))
            res = sorted(ranked, key=lambda r: (-len(votes[r]), min(votes[r]), pos[r]))
        else:
            fb = rank(self.shop_rows(desc, list(reversed(self.seen))), False)
            clear = [r for r in fb if not self.read_misses(r, vws)]
            res = clear + [r for r in fb if r not in clear]
        return res[:2], None

    def link_query(self, vt, q, desc, vw):
        tool, fixed, field = vt
        key = q.lower()
        mem = self.link_seen.get(key)
        nxt = self.link_pages.get(key) if mem is not None else None
        if mem is not None and (self.shop_rows(desc, mem, vw) or not nxt or not any(self.names_shop(x, desc) for x in mem)):
            return mem
        if self.link_searches >= CHAIN_SEARCHES or self.steps_left() <= 12:
            return mem
        a = dict(fixed)
        a[field] = q
        if nxt:
            a[nxt[0]] = nxt[1]
        f = []
        a = shape(a, self.schemas.get(tool), "args", f)
        if f:
            return mem
        self.link_searches += 1
        res = self.env(tool, a)
        if res is None:
            return mem
        obs = res.get("observation")
        self.observe(tool, obs)
        ids = [l for l, _ in self.obs_listings(obs)]
        plan = self.paging_plan(tool, obs)
        self.link_pages[key] = (plan["arg"], plan["next"]) if plan and plan["next"] is not None else None
        m2 = list(mem or [])
        for x in ids:
            if x not in m2:
                m2.append(x)
        self.link_seen[key] = m2
        self.trace.append({"kind": "link_query", "q": q[:80], "paged": bool(nxt), "shop_rows": self.shop_rows(desc, ids)[:4]})
        return m2

    def chain_texts(self):
        live = self.live_all()
        pend = [t for i, t in self.refs if t in live and t not in self.ref_values]
        if not pend:
            return "", ""
        rt = " | ".join(pend)
        pw = []
        used = set()
        tk = self.title_key()
        for _ in range(5):
            known = rt + " " + " ".join(pw)
            new = [r for r in self.seen if r not in used and self.page_read(r) and self.names_shop(r, known)]
            if not new:
                break
            for r in new:
                used.add(r)
                for k, v in (self.rows.get(r) or {}).items():
                    if k != tk and isinstance(v, str) and len(v) > 40:
                        pw.append(v)
        return rt, " ".join(pw)

    def page_first(self, tool, args):
        op, off = self.roles.get("open"), self.roles.get("offer")
        if tool != op or not off or off == op or off not in self.schemas:
            return tool, args, ""
        ot, ft = self.single_target(op), self.single_target(off)
        if not ot or not ft:
            return tool, args, ""
        v = args.get(ot)
        lid = v.strip() if isinstance(v, str) else None
        if not lid or lid not in self.rows or self.page_read(lid) or lid in self.gone:
            return tool, args, ""
        ck = self.category_key()
        lab = self.rows[lid].get(ck) if ck else None
        if lab is None or str(lab) == "" or self.label_owner(lab):
            return tool, args, ""
        rt, pw = self.chain_texts()
        if not rt or not self.names_shop(lid, rt + " " + pw):
            return tool, args, ""
        f = []
        a = shape({ft: lid}, self.schemas[off], "args", f)
        if f:
            return tool, args, ""
        self.trace.append({"kind": "page_first", "id": lid})
        return off, a, ("\n[Cairn] displayed the page of %s (%s) in place of the attribute view: a requirement is still "
                        "valued off a listing sold by this shop, and solely the page reveals where it refers onward or which "
                        "value its value is paired with (an order is valid only once that page has been opened). Call %s on "
                        "that listing once more if its attributes are still needed." % (lid, off, op))

    def reread_offers(self):
        tool = self.roles.get("offer") or self.roles.get("open")
        tgt = self.single_target(tool) if tool else None
        if not tool or not tgt:
            return ""
        cut = max(self.step - 2, self.refused_at)
        stale = [r for r in self.cart_ids() if r not in self.gone and self.offer_seen.get(r, -99) < cut]
        todo = stale
        if stale and self.steps_left() < len(stale) + 5:
            unseen = [r for r in stale if not self.page_read(r)]
            todo = unseen if self.steps_left() >= len(unseen) + self.reserve() else []
        if not todo:
            return ""
        lines = []
        for r in todo:
            f = []
            a = shape({tgt: r}, self.schemas.get(tool), "args", f)
            if f or self.rushed():
                break
            before = dict(self.rows.get(r) or {})
            res = self.env(tool, a)
            if res is None or self.finished:
                break
            self.offer_seen[r] = self.step
            obs = res.get("observation")
            self.observe(tool, obs)
            after = self.rows.get(r) or {}
            ch = [k for k in before if k in after and after[k] != before[k] and not isinstance(after[k], (dict, list))]
            if ch:
                lines.append("%s changed (%s): %s" % (r, ", ".join(str(k) for k in ch[:4]), dump(clip(obs, 300))[:900]))
        if lines:
            self.version += 1
        return "\n".join(lines)

    def late_show(self):
        tool = self.roles.get("offer") or self.roles.get("open")
        if not tool or not self.single_target(tool) or self.rushed():
            return ""
        un = [r for r in self.cart_ids() if not self.page_read(r) and r not in self.gone]
        if len(un) != 1:
            return ""
        self.reread_offers()
        if self.page_read(un[0]):
            self.trace.append({"kind": "late_show", "id": un[0]})
            return ("Cairn displayed the page of %s first (one step): an order requires every line's page to have been "
                    "displayed." % un[0])
        return ""

    def sold_peers(self, lost):
        ck = self.category_key()
        if self.pick() != "cheapest" or not ck:
            return []
        cart = set(self.cart_ids())
        out = []
        for ln in lost:
            p = self.price(ln)
            lab = (self.rows.get(ln) or {}).get(ck)
            if p is None or lab is None or str(lab) == "":
                continue
            pool = [r for r in self.seen if r not in lost and r not in cart and r not in self.full and self.buyable(r)
                    and not self.sold_out(r) and str(self.rows[r].get(ck)) == str(lab) and self.price(r) is not None
                    and self.price(r) <= p + 1e-9]
            pool = sorted(pool, key=lambda r: self.price(r))[:SOLD_PEERS]
            out += [r for r in pool if r not in out]
        return out

    def swap_hint(self):
        lost = [r for r in self.cart_ids() if r not in self.gone and self.sold_out(r)]
        if not lost:
            return ""
        for r in lost:
            self.gone[r] = self.step
        self.leads = [l for l in self.leads if l not in lost]
        self.version += 1
        peers = self.sold_peers(lost)
        if peers:
            self.fill_facts(peers, self.reserve() + 5)
        reps = self.replacements(lost)
        floor = max((self.price(r) or 0) for r in lost)
        txt = ("[Cairn: %s sold out from now on (the store shows it cannot be purchased any more); gone for good, do not "
               "open it again." % ", ".join(lost))
        if reps and self.steps_left() <= 6:
            txt += (" Steps are scarce: within this very turn put %s in the cart in the quantity asked and remove the line "
                    "that sold out (two calls), and then order. Other read listings that seem to match: %s."
                    % (reps[0], ", ".join(reps[1:]) or "none"))
        elif reps and self.pick() == "cheapest" and (self.price(reps[0]) is None or self.price(reps[0]) > floor + 1e-9):
            txt += (" Read listings that seem to match that item, lowest price first: %s. Since the request wants the lowest "
                    "price, unless you have already seen every matching listing, make one search capped below the price of "
                    "%s (through the search's price ceiling argument, if it has one) ahead of replacing the line that sold "
                    "out with the lowest-priced matching listing." % (", ".join(reps), reps[0]))
        elif reps:
            txt += (" Read listings that seem to match that item: %s. Within this turn remove the line that sold out and put "
                    "%s in the cart in the quantity asked, and order after that." % (", ".join(reps), reps[0]))
        else:
            txt += " Remove that line and put in the best other matching listing, searching when none of those already read matches."
        return txt + "]"


RX_FAULT_KIND = re.compile(r" sold out| over the ")
DONE_OPEN = "[Cairn] The decline was not sent because Cairn acted in its place"


class G6A(object):
    def planned_basket(self, lead):
        ck = self.category_key()

        def kind(r):
            if not ck:
                return r
            v = (self.rows.get(r) or {}).get(ck)
            return "" if v is None else str(v).strip().lower()
        N = len(self.items)
        planned = {}
        pkinds = set()
        for key, ids in self.lead_items.items():
            m = RX_FIRST_DIGITS.search(str(key))
            n = int(m.group(0)) if m else (1 if N == 1 else None)
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if pick and n and 1 <= n <= N and kind(pick) not in pkinds:
                planned[pick] = self.items[n - 1].qty
                pkinds.add(kind(pick))
        if lead not in planned:
            return None
        keep = {}
        drops = set()
        kkinds = set()
        for r in self.cart_ids():
            if r in planned:
                continue
            if not self.sold_out(r) and kind(r) not in pkinds:
                keep[r] = self.cart.get(r, 1)
                kkinds.add(kind(r))
            else:
                drops.add(r)
        if len(planned) + len(kkinds) < N:
            return None
        want = dict(keep)
        want.update(planned)
        return want, drops

    def complete_cart(self):
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or not self.roles.get("cart") or not self.roles.get("buy"):
            return ""
        self.completed = True
        pb = self.planned_basket(self.confirmed)
        if not pb:
            return ""
        want, drops = pb
        calls = self.cart_calls(want, drops)
        if not calls or self.steps_left() < len(calls) + 1:
            return ""
        lines = list(want.keys())
        self.fill_facts(lines, len(calls) + 3)
        self.show_links(lines, len(calls) + 3)
        self.cart_check(lines, HARD_SPEND)
        if not self.verdicts.get("|".join(sorted(lines)) + "#" + str(self.version)):
            return ""
        done = []
        for tool, args in calls:
            f = []
            a = shape(args, self.schemas.get(tool), "args", f)
            if f:
                return ""
            res = self.env(tool, a)
            if res is None or self.finished:
                return ""
            self.note_cart(tool, a, res)
            self.observe(tool, res.get("observation"))
            done.append(dump(a))
        buy = self.roles["buy"]
        f = []
        oa = shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", f)
        if f:
            return ""
        self.reread_offers()
        kept = []
        for att in range(2):
            if att == 1:
                self.reread_offers()
                self.cart_check(lines, HARD_SPEND)
                if (not self.verdicts.get("|".join(sorted(lines)) + "#" + str(self.version)) or self.steps_left() <= 0
                        or self.finished or any(self.sold_out(r) for r in lines)):
                    break
            g, gt = self.gate("order")
            if gt:
                kept.append(gt)
            if g != "go":
                break
            res = self.env(buy, oa)
            if res is None:
                break
            self.checkouts += 1
            if not self.finished:
                self.refused_at = self.step
            obs = res.get("observation")
            self.observe(buy, obs)
            kept.append(self.render(obs))
            if self.finished:
                break
        if not kept:
            return ""
        return (DONE_OPEN + ": the independent check passed %s, so Cairn changed the cart (%s) and placed the order. Store "
                "replies: %s" % (self.confirmed, "; ".join(done), " -> then: ".join(kept)))

    def direct_checkout(self, label, detail):
        buy = self.roles.get("buy")
        if not buy or self.finished or self.steps_left() <= 0:
            return []
        if any(RX_FAULT_KIND.search(t) for _, t in self.line_faults()):
            return []
        if self.steps_left() >= 2:
            self.reread_offers()
            if self.finished or self.steps_left() <= 0:
                return []
        f = []
        oa = shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", f)
        if f:
            return []
        kept = []
        for att in range(2):
            if att == 1 and (self.steps_left() <= 0 or self.finished or any(self.sold_out(r) for r in self.cart_ids())):
                break
            if self.steps_left() >= 2:
                g, gt = self.gate("order")
                if gt:
                    kept.append(gt)
                if g != "go":
                    break
            if self.finished or self.steps_left() <= 0:
                break
            res = self.env(buy, oa)
            self.trace.append({"kind": label, "detail": detail, "ok": res is not None})
            if res is None:
                break
            self.checkouts += 1
            if not self.finished:
                self.refused_at = self.step
            obs = res.get("observation")
            self.observe(buy, obs)
            kept.append(self.render(obs))
            if self.finished:
                break
        return kept

    def late_finish(self):
        if self.finished or not self.roles.get("buy") or not self.roles.get("cart"):
            return ""
        built = [r for r in self.built if not self.sold_out(r)]
        cids = self.cart_ids()
        ret = []
        if len(built) >= len(self.items):
            ret = [r for r in built if r not in cids]
            if self.fresh_view() is None:
                ret += [r for r in built if r in self.dropped_ids and r not in ret]
            ck = self.category_key()
            want = {}
            for r in ret:
                own = [n for n in range(1, len(self.items) + 1) if ck and self.owns_label(n, (self.rows.get(r) or {}).get(ck))]
                want[r] = self.items[own[0] - 1].qty if len(own) == 1 else 1
            calls = self.cart_calls(want, set()) if want else []
            if calls is None or len(calls) + 1 > self.steps_left():
                return ""
            for tool, args in calls:
                f = []
                a = shape(args, self.schemas.get(tool), "args", f)
                if f:
                    return ""
                res = self.env(tool, a)
                if res is None or self.finished:
                    return ""
                self.note_cart(tool, a, res)
                self.observe(tool, res.get("observation"))
        elif len(self.items) >= 2 and cids:
            if not self.fill_short() or len([r for r in self.cart_ids() if self.units_held(r) > 0]) < len(self.items):
                return ""
        else:
            return ""
        kept = self.direct_checkout("late_finish", ret)
        if not kept:
            return ""
        return (DONE_OPEN + ": this near the end of the step budget a decline earns zero when every requested item is "
                "already covered by a cart line; " + ("Cairn first put %s back into the cart, and " % ", ".join(ret) if ret else "")
                + "Cairn ordered. Store replies: " + " -> then: ".join(kept))

    def closest_order(self):
        if self.clock() < CLOSEST_CLOCK or not self.closest_cart():
            return ""
        kept = self.direct_checkout("closest_order", dict(self.cart))
        if not kept:
            return ""
        return (DONE_OPEN + ": at this point of the clock a decline earns zero, and a shortage of time is no proof that the "
                "shop lacks a fitting product; Cairn rebuilt the cart as the nearest basket among the listings it had read "
                "(%s) and ordered. Store replies: %s" % (self.cart_text(), " -> then: ".join(kept)))

    def short_cart(self):
        return len([r for r in self.cart_ids() if self.units_held(r) > 0]) < len(self.items) or bool(self.missing_items())

    def reopen(self):
        if (self.tools_reopened or self.checkouts or not self.roles.get("search")
                or self.steps_left() <= self.reserve() + 3 or not self.short_cart()):
            return ""
        if self.roles.get("search") in self.offered():
            return ""
        self.tools_reopened = True
        self.trace.append({"kind": "reopen", "steps_left": self.steps_left()})
        return ("Search and the views for reading listings are available again, because some requested item lacks a cart "
                "line: locate and open a listing that suits that item, put it in the cart, and order, all inside the "
                "remaining steps.")

    def late_review(self):
        self.decline_blocks = 2
        if not self.leads and not self.audited and self.clock() <= WIND_DOWN and self.judges():
            self.audited = True
            fits = self._fits(self._audit_call())
            if fits:
                self.lead_items = fits
                self.leads = [i for v in fits.values() for i in v]
        lead = next((l for l in self.leads if not self.sold_out(l)), None)
        pb = self.planned_basket(lead) if lead else None
        if not pb or self.clock() > WIND_DOWN:
            return ""
        want, drops = pb
        calls = self.cart_calls(want, drops)
        if not calls or len(calls) + 1 > self.steps_left():
            return ""
        ids = sorted(want.keys())
        self.cart_check(ids, HARD_SPEND)
        if not self.verdicts.get("|".join(ids) + "#" + str(self.version)):
            return ""
        self.confirmed = lead
        self.trace.append({"kind": "late_review", "lead": lead})
        return self.complete_cart()

    def offered(self):
        ends = [t for t in (self.roles.get("buy"), self.roles.get("refuse")) if t]
        allowed = [t["function"]["name"] for t in self.tools]
        ask = self.roles.get("ask")
        if ask and self.msg_args("x" * 200) is not None:
            allowed = [n for n in allowed if n != ask]
        if ends:
            L, r = self.steps_left(), self.reserve()
            keep = None
            if L <= r:
                keep = set(ends)
            elif L <= r + 3:
                keep = set(ends) | set([self.roles.get("cart"), self.roles.get("cart_view")])
                ft = self.facts_tool()
                if ft and L >= 3 and self.missing_items():
                    keep.add(ft)
            elif L <= len(self.items) + 5 and not self.checkouts and not (self.tools_reopened and self.short_cart()):
                keep = set(ends) | set([self.roles.get("cart"), self.roles.get("cart_view"), self.roles.get("offer")])
            if keep is not None:
                nar = [n for n in allowed if n in keep]
                if nar:
                    allowed = nar
        return allowed

    def zero_add(self, tool, args):
        props = self._props(tool)
        cnts = [k for k, v in args.items() if isinstance(props.get(k), dict) and "integer" in kinds(props[k]) and count_of(v) is not None]
        if self.units_known or len(cnts) != 1:
            return args, ""
        k = cnts[0]
        c = count_of(args[k])
        if c > 0:
            self.units_known = True
            return args, ""
        if c != 0:
            return args, ""
        ids = [v for v in args.values() if isinstance(v, str) and v in self.rows]
        if len(ids) != 1 or any(isinstance(v, (list, dict)) for v in args.values()):
            return args, ""
        for kk, v in args.items():
            p = props.get(kk)
            if isinstance(p, dict) and isinstance(p.get("enum"), list) and v in p["enum"][1:]:
                return args, ""
        lid = ids[0]
        cv = dump(self.cart_view) if self.cart_view is not None else ""
        if lid in self.ever_carted or lid in self.cart or lid in self.cart_ids() or lid in cv or lid in (self.cart_said or ""):
            return args, ""
        ck = self.category_key()
        own = [n for n in range(1, len(self.items) + 1) if ck and self.owns_label(n, (self.rows.get(lid) or {}).get(ck))]
        qs = set([self.items[own[0] - 1].qty]) if len(own) == 1 else set(it.qty for it in self.items)
        if len(qs) != 1:
            return args, ""
        u = max(1, list(qs)[0])
        mx, mn = props[k].get("maximum"), props[k].get("minimum")
        if _num(mx):
            u = min(u, mx)
        if (_num(mn) and mn > u) or u < 1:
            return args, ""
        self.units_known = True
        self.trace.append({"kind": "zero_add", "id": lid, "units": u})
        a = dict(args)
        a[k] = u
        return a, ("[Cairn] %s is a number of units, not an index into the rows or the cart lines; a zero count puts nothing "
                   "in the cart, so Cairn sent %s=%s." % (k, k, u))

    def note_cart(self, tool, args, res=None):
        self.cart_touched = True
        self.cart_changed_at = self.step
        props = self._props(tool)
        rem = False
        for k, v in args.items():
            p = props.get(k)
            if isinstance(p, dict) and isinstance(p.get("enum"), list) and len(p["enum"]) >= 2 and v in p["enum"][1:]:
                rem = True
        loose = None
        for v in args.values():
            if not isinstance(v, (str, bool)) and count_of(v) is not None:
                loose = count_of(v)
                break
        lines = []
        whole = False
        for k, v in args.items():
            if isinstance(v, list):
                p = props.get(k) if isinstance(props.get(k), dict) else {}
                it = p.get("items") if isinstance(p.get("items"), dict) else {}
                if not v and ("object" in kinds(it) or isinstance(it.get("properties"), dict)):
                    whole = True
                for el in v:
                    if isinstance(el, dict):
                        whole = True
                        lid = next((x for x in el.values() if isinstance(x, str) and x in self.rows), None)
                        c = next((count_of(x) for x in el.values() if not isinstance(x, str) and count_of(x) is not None), 1)
                        if lid is not None:
                            lines.append((lid, c))
                    elif isinstance(el, str) and el in self.rows:
                        lines.append((el, loose))
            elif isinstance(v, str) and v in self.rows:
                lines.append((v, loose))
        if whole:
            self.cart = dict((l, c) for l, c in lines if c is not None and c > 0)
            for l, c in self.cart.items():
                self.ever_carted[l] = c
            return
        hasint = any(isinstance(p, dict) and "integer" in kinds(p) for p in props.values())
        sets = self.sets_counts(tool)
        for lid, c in lines:
            u = self.cart.get(lid, 0)
            if rem:
                if hasint:
                    u -= (c or 1)
                else:
                    u = u - c if c is not None else 0
            elif sets:
                u = c if c is not None else 1
            else:
                u += c if c is not None else 1
            if u > 0:
                self.cart[lid] = u
                self.ever_carted[lid] = u
            else:
                self.cart.pop(lid, None)


RX_QUOTED_ID = re.compile(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"')
SKIP = "[Cairn] Not executed, and no step was spent: "
HOLD = "[Cairn] Not executed yet, and no step was spent: "
BARE_HOLD = "[Cairn] Not executed yet: "


class E9(object):
    def footer(self):
        return "\nCart now: " + self.cart_text()

    def plain_role_tools(self):
        return set(v for k, v in self.roles.items() if k != "search")

    def reader_tools(self):
        return set(self.roles.get(r) for r in "search open offer compare".split() if self.roles.get(r))

    def run(self):
        if self.finished:
            return list(self.trace)
        self.read_brief()
        self.trace.append({"kind": "brief", "brief": self.brief})
        op = self.pv.get("opening")
        shown_op = None
        if op is not None:
            shown_op = restore(op)
            self.absorb(shown_op)
        mv = ""
        if not self.finished:
            mv = self.opening_moves()
        self.gap = STORE_PAUSE
        msg = "SHOPPER'S REQUEST:\n" + self.request
        if op is not None:
            msg += "\n\nTHE STORE'S STARTING DISPLAY:\n" + self.render(shown_op)
        if mv:
            msg += ("\n\nCairn has already searched for every item and read part of each item's listings, one step per "
                    "read:\n" + mv + "\nJudge these listings by the requirements, and search more only where something "
                    "is still lacking.\n\n" + self.status())
        else:
            msg += ("\n\nBegin with a search for each item requested, sought within the category it belongs to, and read "
                    "several listings completely, so that the interviewer can then tell which properties such listings "
                    "show.\n\n" + self.status())
        self.convo = [{"role": "system", "content": GUIDE}, {"role": "user", "content": msg}]
        self.work(TURN_CAP, HARD_SPEND)
        if not self.finished:
            t = self.finish("again")
            if t and not self.finished and not self.store_broke and not self.no_credit and self.steps_left() > 1:
                self.convo.append({"role": "user", "content": "[Cairn] Cairn attempted to finish and " + t
                                   + "\nFollow the store's answer and then finish.\n" + self.status()})
                self.work(self.turns + 10, HARD_SPEND + 0.01)
        for _ in range(2):
            if self.finished:
                break
            t = self.finish("normal")
            if not t or self.finished or self.store_broke or self.no_credit or self.steps_left() <= 1:
                break
            self.convo.append({"role": "user", "content": "[Cairn] Cairn attempted to finish and " + t
                               + "\nFollow the store's answer and then finish.\n" + self.status()})
            self.work(self.turns + 6, HARD_SPEND + HEAVY_ROOM)
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "spend": round(self.spend, 5),
                           "calls": self.model_calls})
        return list(self.trace)

    def work(self, cap, spend_cap):
        for _ in range(CRASH_RETRIES):
            try:
                self.turn_loop(cap, spend_cap)
                return
            except Exception as e:
                self.trace.append({"kind": "crash", "error": ("%s: %s" % (type(e).__name__, e))[:400]})
                self.answer_open()

    def answer_open(self):
        idx = None
        for i in range(len(self.convo) - 1, -1, -1):
            if self.convo[i].get("tool_calls"):
                idx = i
                break
        if idx is None:
            return
        answered = set(m.get("tool_call_id") for m in self.convo[idx + 1:] if m.get("role") == "tool")
        for c in self.convo[idx]["tool_calls"]:
            if c.get("id") not in answered:
                self.convo.append({"role": "tool", "tool_call_id": c.get("id"),
                                   "content": "[Cairn] This call possibly did not execute, because something failed along "
                                              "the way. Inspect the present state before repeating it."})
        self.convo.append({"role": "user", "content": self.status()})

    def turn_loop(self, cap, spend_cap):
        idle = 0
        while True:
            if self.finished or self.store_broke or self.steps_left() < 1 or self.turns >= cap or self.spend >= spend_cap:
                return
            if not (self.clock() < LAST_TURN or (not self.cart_ids() and self.room() > EMPTY_CART_ROOM)):
                return
            if self.ready_to_ask():
                s = self.interview()
                if s:
                    self.convo.append({"role": "user", "content": s + "\n" + self.status()})
                continue
            self.trim()
            self.turns += 1
            off = set(self.offered())
            tools = [t for t in self.tools if t["function"]["name"] in off]
            msg = self.think(self.convo, [LIGHT, SPARE], DIRECTOR_ROOM, tools)
            if msg is None:
                self.misses += 1
                if self.no_credit or self.model_calls >= CALL_CAP:
                    return
                if self.misses > 2:
                    if self.clock() >= OUTAGE_CUTOFF:
                        return
                    time.sleep(OUTAGE_PAUSE)
                continue
            calls = []
            for j, c in enumerate(msg.get("tool_calls") or []):
                if not isinstance(c, dict):
                    continue
                fn = c.get("function") if isinstance(c.get("function"), dict) else {}
                raw = fn.get("arguments")
                if isinstance(raw, dict):
                    a = raw
                elif isinstance(raw, str) and raw.strip():
                    a = json_obj(raw) or {}
                else:
                    a = {}
                calls.append({"id": c.get("id") or "k%d_%d" % (self.turns, j), "type": "function",
                              "function": {"name": str(fn.get("name") or ""), "arguments": dump(a)}})
            content = msg.get("content")
            text = content if isinstance(content, str) else (dump(content) if content is not None else "")
            if not calls:
                idle += 1
                self.convo.append({"role": "assistant", "content": text or "(no call)"})
                ro = self.reopen()
                if ro:
                    idle = 0
                self.convo.append({"role": "user", "content": "[Cairn] Use one of the tools so the work goes on."
                                   + (" " + ro if ro else "") + "\n" + self.status()})
                if idle >= 3:
                    return
                continue
            idle = 0
            commit_tools = (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"))
            commit = any(c["function"]["name"] and c["function"]["name"] in commit_tools for c in calls)
            if (not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and self.steps_left() > 8
                    and self.clock() <= INTERVIEW_DEADLINE and commit):
                self.convo.append({"role": "assistant", "content": text or None, "tool_calls": calls})
                for c in calls:
                    self.convo.append({"role": "tool", "tool_call_id": c["id"], "content": SKIP + "the shopper is questioned first."})
                s = self.interview()
                self.convo.append({"role": "user", "content": (s or "[Cairn] The interviewer had no question to put.")
                                   + "\n" + self.status()})
                continue
            self.convo.append({"role": "assistant", "content": text or None, "tool_calls": calls})
            self.trace.append({"kind": "turn", "text": text[:300], "calls": [c["function"] for c in calls]})
            replies = self.dispatch(calls)
            for k, c in enumerate(calls):
                r = replies[k]
                if k == len(calls) - 1:
                    r = r + "\n" + self.status()
                self.convo.append({"role": "tool", "tool_call_id": c["id"], "content": r})

    def trim(self):
        tot = sum(len(dump(m.get("content"))) for m in self.convo)
        if tot <= HISTORY_CAP:
            return
        tms = [m for m in self.convo if m.get("role") == "tool"]
        for m in tms[:max(1, len(tms) // 2)]:
            c = m.get("content")
            if isinstance(c, str) and len(c) > 600:
                ids = []
                for x in RX_QUOTED_ID.findall(c):
                    if x in self.rows and x not in ids:
                        ids.append(x)
                m["content"] = c[:400] + "... (older result shortened; listing ids it held: " + ", ".join(ids[:20]) + ")"

    def dispatch(self, calls):
        calls = self.join_opens(calls)
        n = len(calls)
        return [self.run_call(c, n) for c in calls]

    def run_call(self, c, n):
        allowed = self.offered()
        if c.get("joined"):
            return ("[Cairn] This listing is in the comparison result given earlier in this turn, because Cairn read them "
                    "jointly in a single step.")
        if self.finished:
            return "[Cairn] Not executed: the episode has ended."
        tool = c["function"]["name"]
        R = self.roles
        if tool and tool == R.get("ask") and self.msg_args("x" * 200) is not None:
            return SKIP + "Cairn's interviewer sends all shopper messages."
        if tool not in self.schemas:
            return SKIP + "unknown tool; the tools are: " + ", ".join(sorted(allowed))
        if tool not in allowed:
            return SKIP + "too few steps remain for this tool; use one of these instead: " + ", ".join(sorted(allowed))
        raw = c["function"].get("arguments")
        if isinstance(raw, dict):
            args = raw
        elif isinstance(raw, str) and raw.strip():
            args = json_obj(raw)
            if args is None:
                return SKIP + "the argument text did not form exactly one JSON object."
        else:
            args = {}
        faults = []
        args = shape(args, self.schemas[tool], "args", faults)
        if faults:
            return SKIP + "; ".join(faults[:6])
        args, prob = self.map_ids(tool, args)
        if prob:
            return SKIP + prob + "."
        tail = []
        if tool == R.get("cart"):
            args = self.whole_removal(args)
            args, zn = self.zero_add(tool, args)
            if zn:
                tail.append(zn)
        ct = self.compare_target(tool) if tool == R.get("compare") else None
        if ct and isinstance(args.get(ct), list) and self.cmp_fail:
            room = self.batch_room()
            if len(args[ct]) > room:
                left = args[ct][room:]
                args = dict(args)
                args[ct] = args[ct][:room]
                tail.append("[Cairn] Only the first %d went out, as this store's comparison takes no more than %d; not yet "
                            "sent: %s." % (room, room, ", ".join(str(x) for x in left)))
        st = self.single_target(tool)
        if st and tool in (R.get("open"), R.get("offer")):
            v = args.get(st)
            if isinstance(v, str) and v.strip() in self.gone:
                return (SKIP + "this listing's sell-out was recorded at step %d and it will not return. Put a different "
                        "listing that fits in place of its cart line." % self.gone[v.strip()])
        if tool == R.get("cart") and any(not isinstance(v, str) and count_of(v) is not None and count_of(v) < 0 for v in args.values()):
            fv = self.fresh_view()
            vt = dump(fv) if fv is not None else ""
            for v in args.values():
                if isinstance(v, str) and v in self.rows and v not in self.cart and v not in vt and v not in self.cart_said:
                    return SKIP + "%s is not held, so no units can come off." % v + self.footer()
        fin_o = self.is_order(tool, args)
        fin_d = self.is_decline(tool, args)
        finishing = fin_o or fin_d
        room_at_call = self.steps_left()
        if finishing and n > 1:
            return SKIP + "a finishing call must go alone; send it once the results of the other calls have been read."
        if not finishing and self.steps_left() <= self.reserve() and (R.get("buy") or R.get("refuse")):
            return (SKIP + "the last few steps are reserved for the order (with the shopper approving the basket first when "
                    "that duty is due) or for a decline.")
        searchlike = tool not in self.plain_role_tools()
        fresh = False
        if searchlike:
            fresh = self.is_new_query(tool, args)
            a2 = self.first_look(tool, args, fresh)
            if a2 != args:
                tail.append("[Cairn] This being a new search, Cairn sent it for page one; you had sent %s." % dump(args))
                args = a2
        args = self.row_ids(tool, args)
        tool, args, rnote = self.page_first(tool, args)
        st = self.single_target(tool)
        if tool == R.get("search") and self.codes_dead and self.names_code(args):
            return (SKIP + "code searches in this store returned nothing. Search with words and attribute values instead, "
                    "browse the category listing, and take the codes from the rows shown.")
        mark = mark_of(tool, args)
        if not finishing and tool != R.get("cart") and mark in self.recent[-3:]:
            return SKIP + "this exact call was just made, with no change since; the earlier result stands."
        if tool == R.get("search"):
            cr = self.cap_repeat(tool, args)
            if cr:
                return cr
        readers = self.reader_tools()
        if tool in readers and mark in self.read_cache and self.read_cache[mark][0] == self.version:
            _, s, txt = self.read_cache[mark]
            return SKIP + "the identical call at step %d gave the result below, with no change since then.\n%s" % (s, txt[:3000])
        if tool in readers and tool != R.get("compare") and st:
            v = args.get(st)
            if isinstance(v, str):
                lid = v.strip()
                if self.shown_already(tool, lid) or (tool, lid) in self.factless:
                    return SKIP + "this view has nothing new to show for that listing; its record: " + self.listing_record(lid)
        if ct:
            lst = args.get(ct)
            if isinstance(lst, list) and lst:
                keep = [x for x in lst if not (isinstance(x, str) and self.shown_already(tool, x.strip()))]
                drop = [x for x in lst if x not in keep]
                if not keep:
                    return (SKIP + "these views have nothing new to show for those listings; their records:\n"
                            + "\n".join("%s: %s" % (x, self.listing_record(str(x).strip())) for x in lst))[:4000]
                if drop:
                    tail.append("[Cairn] Left out as already shown in full with no change since: %s." % ", ".join(str(x) for x in drop))
                    args = dict(args)
                    args[ct] = keep
                    mark = mark_of(tool, args)
        hunt_txt = ""
        trail_txt = ""
        if fin_o and self.steps_left() > 2:
            if not self.cheaper_holds and self.steps_left() <= self.reserve() + 3:
                cp = self.cheaper_proven()
                if cp and self.steps_left() >= self.reserve() + 1 + 2 * len(cp):
                    self.cheaper_holds += 1
                    self.trace.append({"kind": "cheaper_hold", "late": True, "pairs": [(a[0], a[2]) for a in cp]})
                    for a in cp:
                        self.cheaper_named.add((a[0], a[2]))
                    return HOLD + self.cheaper_text(cp) + self.footer()
            hunt_txt = self.cheaper_hunt()
            w = self.wider_hold()
            if w:
                return w + self.footer()
            changes = self.reread_offers()
            if self.note_sold:
                self.sold_line_search()
            cv = R.get("cart_view")
            if self.cart_touched and self.fresh_view() is None and cv and self.steps_left() > 5 and not self.rushed():
                f = []
                a = shape({}, self.schemas.get(cv), "args", f)
                if not f:
                    r = self.env(cv, a)
                    if r is not None:
                        obs = r.get("observation")
                        self.cart_view = obs
                        self.cart_seen_at = self.step
                        meaning = self.observe(cv, obs)
                        if meaning:
                            self.queued = []
                            return (HOLD + "viewing the cart revealed a change: " + meaning + self.footer()
                                    + "\nApply it, repair the cart where necessary, then order.")
            sh = self.swap_hint()
            gl = [r for r in self.cart_ids() if r in self.gone] if self.sold_holds < 2 else []
            if sh or gl:
                if not sh:
                    self.sold_holds += 1
                return HOLD + (sh if sh else "the carted lines %s are sold out; replace that carted line first." % ", ".join(gl)) + self.footer()
            trail_txt, hold = self.trail()
            if hold:
                return HOLD + trail_txt + self.footer()
            if self.fit_holds < 2 and self.steps_left() > self.reserve() + 3:
                ff = self.fit_faults()
                if ff:
                    self.fit_holds += 1
                    self.trace.append({"kind": "fit_hold", "faults": ff[:4]})
                    sets = self.read_sets()
                    t1 = ""
                    if sets:
                        t1 = (" Among the listings already opened there is a complete set, compatible through its codes, "
                              "which spells out each of the requirements: " + self.set_text(sets) + ".")
                    return (HOLD + "a carted part is incompatible with the carted device: " + "; ".join(ff[:3])
                            + ". Replace the part, or else the device, and then order." + t1 + self.footer())
            if self.line_holds < 2 and self.steps_left() > self.reserve() + 2:
                lf = self.line_faults()
                if lf and all(k == "units" for k, _ in lf):
                    fixes = self.count_fixes()
                    if fixes:
                        lf = self.line_faults()
                        if not lf:
                            self.line_holds += 1
                            return (HOLD + "Cairn adjusted the unit counts itself: " + "; ".join(fixes)
                                    + ". If the store rejected any of those fixes, correct those lines. Then order." + self.footer())
                if lf:
                    self.line_holds += 1
                    self.trace.append({"kind": "line_hold", "faults": [t for _, t in lf[:6]]})
                    return (HOLD + "the cart and the request do not match yet: " + "; ".join(t for _, t in lf[:6])
                            + ". Correct those lines with the means the cart tool offers for adding and removing units, and "
                              "then order." + self.footer())
            lines = self.cart_ids()
            fetched = self.fill_facts(lines, self.reserve() + (1 if len(lines) >= 2 else 3))
            self.show_links(lines, self.reserve() + 1)
            fetched = fetched + self.cheaper_rows()
            ablock = ("Cairn fetched the attribute records of the listings below at one step apiece:\n" + "\n".join(fetched)
                      + "\n") if fetched else ""
            if not self.cheaper_holds and self.steps_left() > self.reserve() + 3:
                cp = self.cheaper_proven()
                if cp:
                    self.cheaper_holds += 1
                    self.trace.append({"kind": "cheaper_hold", "late": False, "pairs": [(a[0], a[2]) for a in cp]})
                    for a in cp:
                        self.cheaper_named.add((a[0], a[2]))
                    return HOLD + ablock + self.cheaper_text(cp) + self.footer()
            if not self.weak_holds and self.steps_left() > self.reserve() + 3:
                wt = self.weak_text()
                if wt:
                    self.weak_holds += 1
                    return HOLD + ablock + wt + self.footer()
            ag = self.agreed_swap()
            if ag:
                return ("[Cairn] The order has not run yet (what waits is the order; the cart calls reported next did "
                        "consume steps): " + ag + self.footer())
            v = self.cart_check()
            if v:
                out = BARE_HOLD
                if trail_txt:
                    out += trail_txt + "\n"
                if hunt_txt:
                    out += hunt_txt + " " + self.below_cart() + "\n"
                if fetched:
                    out += ("Cairn fetched the attribute records of the listings below for this review, at one step apiece:\n"
                            + "\n".join(fetched) + "\n")
                return out + v + self.footer()
            if changes and self.spend > SOFT_SPEND:
                return (HOLD + "Cairn has just fetched the cart's listings again:\n" + changes
                        + "\nOrder again once all lines are still in stock and priced within budget; otherwise swap that line "
                          "before ordering.")
        if fin_d and self.steps_left() > 5:
            if self.fit_pairs and not self.set_holds:
                sets = self.read_sets(True)
                if sets:
                    self.set_holds += 1
                    self.trace.append({"kind": "set_hold", "sets": [s["lines"] for s in sets[:2]]})
                    return (HOLD + "a complete fitting set exists in this store: its listings were all opened before and, "
                            "between their attribute blocks and titles, spell out each of the requirements, and each line "
                            "bound to the device matches the device through its code: " + self.set_text(sets) + ". Make "
                            "the cart one of these sets and order; decline solely when some listing's record plainly "
                            "violates a requirement.")
            if not self.fit_pairs and not self.set_holds:
                pl = self.proven_lines()
                if pl:
                    self.set_holds += 1
                    self.trace.append({"kind": "proven_hold", "lines": pl})
                    ents = []
                    for n in sorted(pl):
                        p = self.price(pl[n])
                        ents.append("item %d (%s): %s%s" % (n, self.items[n - 1].name, pl[n], (" (%s)" % gnum(p)) if p is not None else ""))
                    return (HOLD + "opened, buyable listings satisfy all requirements in the way their pages put it "
                            "(every wanted value sits under a key bearing its property's name, and every figure is taken "
                            "literally from the page, even if it seems a typing slip): " + "; ".join(ents) + ". Put these "
                            "listings into the cart in the quantities requested and order; decline solely when some record "
                            "plainly violates a requirement, and in that case name the violated one in your own reply text "
                            "(not a shopper message) and send the decline once more.")
            t = self.decline_trail()
            if t:
                return HOLD + t
            self.fit_hunt()
            t = self.audit()
            if t:
                return HOLD + t
            if self.leads and self.steps_left() > 4 and self.decline_blocks == 0:
                self.decline_blocks = 1
                self.trace.append({"kind": "lead_hold", "leads": list(self.leads)})
                return (HOLD + "the review named listings that seem to fit: %s. Open them where necessary, cart whichever fits "
                        "best while obeying the pick rule, then order. Decline only if all of them plainly fail some "
                        "requirement, and in that case state in your own reply text (not a shopper message) which one fails "
                        "and send the decline once more." % ", ".join(self.leads))
            if self.leads and self.steps_left() > 4 and self.decline_blocks == 1:
                self.decline_blocks = 2
                first = self.leads[0]
                judged = [first]
                basket = False
                if len(self.items) > 1:
                    pb = self.planned_basket(first)
                    if pb:
                        judged = sorted(pb[0].keys())
                        basket = True
                self.cart_check(judged, HARD_SPEND)
                if self.verdicts.get("|".join(sorted(judged)) + "#" + str(self.version)) and first in self.leads:
                    self.confirmed = first
                    self.trace.append({"kind": "confirmed", "id": first, "judged": judged})
                    if basket:
                        return (HOLD + "a second, separate check run on that basket (%s) finds that it satisfies all "
                                "requirements, with exactly one line per requested item. Make the cart hold precisely these "
                                "lines in the quantities requested and order; decline solely when some record plainly "
                                "violates a requirement." % ", ".join(judged))
                    return (HOLD + "a second, separate check of the records of %s finds that it satisfies all requirements. "
                            "Cart it in the quantity asked and order; decline only when its record plainly breaks a "
                            "requirement." % first)
        if (fin_d and 2 <= self.steps_left() <= 5 and not self.confirmed and self.decline_blocks < 2
                and len([b for b in self.built if not self.sold_out(b)]) < len(self.items)):
            t = self.late_review()
            if t:
                return t + (" [episode finished]" if self.finished else "")
        if fin_d and self.steps_left() <= 5 and not self.late_held:
            self.late_held = True
            t = self.late_finish()
            if t:
                return t + (" [episode finished]" if self.finished else "")
            if not self.finished:
                built = [r for r in self.built if not self.sold_out(r)]
                cids = self.cart_ids()
                missing = [r for r in built if r not in cids]
                if self.fresh_view() is None:
                    missing += [r for r in built if r in self.dropped_ids and r not in missing]
                if len(built) >= len(self.items) and len(missing) + 1 <= self.steps_left():
                    self.trace.append({"kind": "late_decline_hold", "missing": missing})
                    t = (HOLD + "a decline scores zero whenever a fitting listing exists, and running low on steps is no "
                         "evidence about whether one exists. ")
                    if missing:
                        t += "Cart the lines %s again in the quantities asked and order." % ", ".join(missing)
                    else:
                        t += "Order the cart unchanged."
                    return t + (" Decline only if the searching proved that, for one of the requested items, no listing "
                                "satisfies its requirements, and in that case send the decline once more.")
        if (fin_d and not self.pushed and not self.leads and room_at_call > self.budget_steps // 3
                and self.steps_left() > self.reserve() + len(self.items) + 2):
            self.pushed = True
            return (HOLD + "%d steps remain. Before giving up, try untried kinds of search: list every requested item's "
                    "category and page onward through that listing using the page token, or offset value, returned by the "
                    "store; try one-word queries of values and alternative names of the property; open any listing that "
                    "could fit. Decline only after all that." % self.steps_left())
        if fin_d:
            t = self.complete_cart() or self.closest_order()
            if t:
                return t + (" [episode finished]" if self.finished else "")
            args = self.quiet_fill(args)
        if fin_o:
            lq = self.late_question()
            if lq:
                return BARE_HOLD + lq + self.footer()
        if fin_o and self.steps_left() == 2 and self.checkouts >= 1:
            ls = self.late_show()
            if ls:
                tail.insert(0, "[Cairn] " + ls)
        if finishing and self.steps_left() >= 2:
            g, gt = self.gate("order" if fin_o else "end")
            if g != "go":
                return BARE_HOLD + gt + self.footer()
            if gt:
                tail.insert(0, "[Cairn] " + gt)
        res = self.env(tool, args)
        if res is None:
            return "[Cairn] The call broke down through a network or server problem; you may retry it."
        obs = res.get("observation")
        if ct:
            lst = args.get(ct)
            if isinstance(lst, list) and len(lst) > 1:
                d = dump(obs)
                if not any(isinstance(x, str) and x in d for x in lst):
                    self.cmp_fail = min(self.cmp_fail or 999, len(lst))
                    tail.append("[Cairn] The comparison came back empty for %d listings. Send no more than %d per call."
                                % (len(lst), self.batch_room()))
                else:
                    self.cmp_ok = max(self.cmp_ok, len(lst))
        self.recent = (self.recent + [mark])[-3:]
        for k, v in args.items():
            vals = v if isinstance(v, list) and v else [v]
            if all(isinstance(x, str) and x in self.rows for x in vals):
                self.id_args.add((tool, k))
        self.absorb(obs)
        if tool == R.get("compare"):
            recs = self.obs_listings(obs)
            if recs:
                self.facts[tool] = self.facts.get(tool) is True or any(attrs_of(r) is not None for _, r in recs)
        if tool in readers and tool != R.get("search"):
            self.remember_view(tool, obs)
        pnotes = []
        if searchlike and not finishing:
            ap = self.auto_page(tool, args, obs)
            if ap:
                pnotes.append(ap)
            if fresh:
                lp = self.late_page(tool, args, obs)
                if lp:
                    pnotes.append(lp.strip("\n"))
            sh = self.shorter(tool, args, obs)
            if sh:
                pnotes.append(sh.strip("\n"))
            bo = self.big_opens(obs)
            if bo:
                pnotes.append(bo)
            er = self.end_reads(tool, obs)
            if er:
                pnotes.append(er)
        if tool in readers and not finishing:
            ss = self.series_sweep()
            if ss:
                pnotes.append(ss)
        if st and tool not in (R.get("cart"), R.get("buy"), R.get("refuse"), R.get("ask")):
            if isinstance(obs, dict) and self.listing_id(obs, False, 0):
                self.fact_note(tool, attrs_of(obs) is not None)
                if tool == R.get("open") and self.facts.get(tool) is False:
                    for t in self.tools:
                        nm = t["function"]["name"]
                        if (nm != tool and self.single_target(nm) and self.facts.get(nm) is not False
                                and nm not in (R.get("cart"), R.get("buy"), R.get("refuse"), R.get("ask"), R.get("search"))
                                and not self.lists_things(nm)):
                            self.roles["open"] = nm
                            self.trace.append({"kind": "open_role", "tool": nm})
                            break
                if self.facts.get(tool) is False and tool != self.roles.get("open") and not self.roles.get("offer"):
                    self.roles["offer"] = tool
                    self.trace.append({"kind": "offer_role", "tool": tool})
            if tool == self.roles.get("offer") or (tool == self.roles.get("open") and not self.roles.get("offer")):
                v = args.get(st)
                if isinstance(v, str):
                    self.offer_seen[v.strip()] = self.step
        if tool == R.get("cart"):
            self.note_cart(tool, args, res)
            self.cart_said = dump(obs)
            self.recent = []
            cnts = [count_of(v) for v in args.values() if not isinstance(v, str) and count_of(v) is not None]
            if cnts:
                for v in args.values():
                    if isinstance(v, str) and v in self.rows:
                        if cnts[0] <= 0 and not in_rows(v, obs):
                            self.dropped_ids.add(v)
                        elif cnts[0] > 0:
                            self.dropped_ids.discard(v)
                if cnts[0] > 0:
                    dl = self.dear_page([v for v in args.values() if isinstance(v, str) and v in self.rows])
                    if dl:
                        tail.append(dl)
            live = [r for r in self.cart_ids() if not self.sold_out(r) and r not in self.dropped_ids]
            if len(live) >= len(self.built):
                self.built = live
        if tool == R.get("search") and self.names_code(args):
            self.code_tries.append(bool(self.obs_listings(obs)))
            self.codes_dead = len(self.code_tries) >= 2 and not any(self.code_tries)
        if tool == R.get("search") and self.obs_listings(obs):
            self.record_search(args)
        if tool == R.get("cart_view"):
            self.cart_view = obs
            self.cart_seen_at = self.step
        if fin_o:
            self.checkouts += 1
            if not self.finished:
                self.refused_at = self.step
        out = self.render(obs)
        if pnotes:
            out += "\n" + "\n".join(pnotes)
        if tail:
            out += "\n" + "\n".join(tail)
        if searchlike and not finishing:
            ch = self.cheapest_here(obs)
            if ch:
                out += "\n" + ch
            ci = self.cap_ignored(tool, args, obs)
            if ci:
                out += "\n" + ci
        cacheable = out
        ev = self.evidence_obs(obs)
        if ev:
            out += "\n" + ev
        if tool == R.get("cart"):
            cv0 = [count_of(v) for v in args.values() if not isinstance(v, str) and count_of(v) is not None]
            if cv0 and cv0[0] == 0 and any(isinstance(v, str) and v in self.rows and in_rows(v, obs) for v in args.values()):
                out += ("\n[Cairn] A zero count did not remove the line, since the tool adds the number given to the line; "
                        "removing units takes a negative number.")
        self.note_sold = False
        self.note_named = False
        ll = self.loose_lines(obs)
        if ll:
            m = self.read_note(ll)
            if m:
                out += "\n[note read] " + m + " Carry this change into the picks before ordering."
        refused = fin_o and not self.finished
        again = refused and any(l in self.spoken(obs) for l in self.sold_lines)
        loose = self.note_sold and (refused or (not self.note_named and not fin_o))
        if (loose or again) and self.cart_ids() and not self.finished:
            t = self.sold_line_search()
            if t:
                out += "\n[Cairn fetched the cart lines again following the rejection]\n" + t
        self.note_sold = False
        sw = self.swap_hint()
        if sw:
            out += "\n" + sw
        if tool == R.get("cart") and self.fit_pairs and not self.finished:
            ff = self.fit_faults()
            if ff:
                out += "\n[Cairn] These lines are not yet compatible with each other: " + "; ".join(ff)
        if self.queued:
            out += "\n[notes seen during Cairn's own calls] " + " | ".join(self.queued)
            self.queued = []
        out += rnote
        if tool in readers:
            self.read_cache[mark] = (self.version, self.step, cacheable)
        if self.finished:
            out += " [episode finished]"
        return out

    def can_decline(self):
        tool = self.roles.get("refuse")
        if not tool:
            return False
        f = []
        shape(self.quiet_fill(dict(self.brief.get("refuse_args") or {})), self.schemas.get(tool), "args", f)
        return not f

    def finish(self, mode="normal"):
        if self.finished or self.store_broke or self.steps_left() <= 0:
            return ""
        fresh = self.fresh_view() is not None
        holding = bool(self.cart_ids())
        whole = True
        if holding and len(self.items) >= 2:
            filled = self.fill_short()
            whole = filled or not self.can_decline()
            if not whole and self.clock() >= CLOSEST_CLOCK and self.closest_cart():
                whole = True
            if whole and len(self.cart_ids()) >= 2:
                ids = self.cart_ids()
                self.fill_facts(ids, self.reserve())
                if self.steps_left() >= 3 and self.room() > 5:
                    self.show_links(ids, self.reserve())
            if not whole:
                self.trace.append({"kind": "short_cart_decline"})
        if self.finished or self.steps_left() <= 0:
            return ""
        buy = self.roles.get("buy")
        if not holding and buy and self.proven_basket():
            holding = True
        if not holding and buy and self.ever_carted and self.clock() >= CLOSEST_CLOCK and self.closest_cart():
            holding = True
        if buy and self.fit_pairs and not self.finished and self.steps_left() >= 2:
            sets = self.read_sets(True)
            cids = self.cart_ids()
            mixed = (not holding) or bool(self.fit_faults()) or len(cids) < len(self.items)
            if sets and mixed and not any(set(s["ids"]) == set(cids) for s in sets) and self.cart_set(sets[0]["lines"]):
                holding = True
                whole = True
        if buy and holding and whole and self.fit_pairs and not self.finished and self.steps_left() >= 3 and self.fit_faults():
            self.closest_cart()
        if self.finished or self.steps_left() <= 0:
            return ""
        blind = (self.cart_touched and not fresh and self.steps_left() >= 2 and not self.dropped_ids
                 and (self.checkouts == 0 or self.refused_at < self.cart_changed_at))
        if whole and buy and (holding or blind):
            f = []
            oa = shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", f)
            if not f:
                if self.steps_left() >= 2 and holding:
                    self.reread_offers()
                    if self.finished or self.steps_left() <= 0:
                        return ""
                if self.steps_left() >= 2:
                    self.gate("order")
                    if self.finished or self.steps_left() <= 0:
                        return ""
                res = self.env(buy, oa)
                self.trace.append({"kind": "forced_order", "ok": res is not None})
                if self.finished or self.steps_left() <= 0:
                    return ""
                if mode == "again" and res is not None:
                    self.checkouts += 1
                    self.refused_at = self.step
                    obs = res.get("observation")
                    self.observe(buy, obs)
                    return "the order was rejected: " + self.render(obs)
        ref = self.roles.get("refuse")
        if ref:
            f = []
            da = shape(self.quiet_fill(dict(self.brief.get("refuse_args") or {})), self.schemas.get(ref), "args", f)
            if not f:
                if self.steps_left() >= 2:
                    self.gate("end")
                    if self.finished or self.steps_left() <= 0:
                        return ""
                res = self.env(ref, da)
                self.trace.append({"kind": "forced_decline", "ok": res is not None})
                if res is not None and not self.finished:
                    obs = res.get("observation")
                    self.observe(ref, obs)
                    return "the decline was rejected: " + self.render(obs)
        return ""

class Episode(E1, E2, E3, E4, E5, E6, E7, E8, GM, G4B, G5A, G5B, G6A, E9):
    pass


def agent_main(problem_data):
    try:
        ep = Episode(problem_data)
    except Exception as e:
        return [{"kind": "crash", "error": ("%s: %s" % (type(e).__name__, e))[:400]}]
    try:
        return ep.run()
    except Exception as e:
        ep.trace.append({"kind": "crash", "error": ("%s: %s" % (type(e).__name__, e))[:400]})
        if not ep.finished:
            try:
                ep.finish()
            except Exception as e2:
                ep.trace.append({"kind": "crash", "error": ("%s: %s" % (type(e2).__name__, e2))[:400]})
        return list(ep.trace)


__all__ = ["agent_main"]
