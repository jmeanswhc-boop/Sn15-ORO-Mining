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
WIND_DOWN = 200.0
HUNT_DEADLINE = 200.0
HUNT_GRACE = 25.0
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

QUIET_WORD_LEN = 4

GATE_ROUNDS = 3
TRAIL_ROOM = 15
EARLY_TRAIL_ROUNDS = 3
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
RX_FIT_CODE = re.compile(r"\b[A-Z][A-Za-z]{0,5}[ -]?(?:\d{3,6}|\d{1,5}[xX]{1,3})\b")
PART_CODE = r"[A-Za-z]{1,5}\d{0,3}-?\d{3,7}"
RX_FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_ ./&'()\-]{0,38}")
RX_BULLET = re.compile(r"^ *(?:[-*•·]|\d{1,3}[.)]) +")
RX_PAIR_KEY = re.compile(r"name|key|label|attr|prop|field|spec|title|^k$", re.I)
RX_ID_FIELD = re.compile(r"(?i:(?:^|[ _-])(?:id|idx|sku|pid|ref|asin|ident\w*)(?=$|[ _-]))|^[iI][dD][A-Z0-9]|Ident|(?i:(?<![a-z])(?!(?:paid|valid|grid|solid|rigid|fluid|liquid|hybrid|rapid|void|acid|humid|android|said|laid)$)[a-z]*id$)")
RX_FAIL_KEY = re.compile(r"error|fail|(?<!de)fault|trouble(?!shoot)|denied|refus|invalid", re.I)
RX_PLACEHOLDER = re.compile(r"^(?:none|null|n/a|unknown)$", re.I)

RX_SEASON_YEAR = re.compile(r"^[A-Z][a-z]+ (?:19|20)\d\d$")
RX_GAP = re.compile(r"[ -]+")
RX_OWN_CODE = re.compile(r"[\[(]\s*(?:part\s+)?(" + PART_CODE + r")\s*[\])]", re.I)
RX_MODEL_MARK = re.compile(r"\bmodel\s*(?:no\.|no|number|#)?\s*[:.]?\s*(?!(?:no|number)\b)([A-Z][A-Za-z]{0,5}[ -]?(?:\d{3,6}|\d{1,5}[xX]{1,3}))\b", re.I)
RX_ALSO_SOLD = re.compile(r"\b(?:also sold as|also known as|sold also as)\s+([^.;]+)", re.I)
RX_WORKS_WITH = re.compile(r"(?<!not )(?<!never )\b(?:works with|compatible with|fits)\s*:?\s+([^.;]+)", re.I)
RX_EXCEPT_CUT = re.compile(r"\b(?:except|excluding|but not)\b", re.I)
RX_PART_ALSO = re.compile(r"\bpart\s+" + PART_CODE + r"\s+also\s+$", re.I)
RX_ALSO_FITS = re.compile(r"\bpart\s+(" + PART_CODE + r")\s+also fits\s*:?\s*([^.;]+)", re.I)
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
    'You are Terry, a careful purchaser who serves one single shopper inside a web shop. Your sole means of acting '
    "is a call to one of the store's tools. Grading is all-or-nothing: a score of 1 requires a finish that is "
    'exactly correct, and anything else scores 0. A correct purchase must hold all requested items, every one '
    'in the quantity asked, every one satisfying each requirement still in force when checkout happens (requirements '
    'that changed during the work count as changed), and its total may not exceed the budget. The single alternative '
    'correct finish: when the shop truly holds nothing that satisfies everything, the episode must instead close '
    'through the ending (no-purchase) tool, performed in the manner the request specifies.\n'
    '\n'
    'HOW THE STORE OPERATES\n'
    '- Every call to a tool spends one step; the supply of steps is finite, and reasoning is free. Every store '
    'has its own tool names, parameter names and result layouts, so study the schema and the description of every '
    'tool and pass parameter names and permitted values exactly as given.\n'
    '- A search can be restricted: it may filter by category; a keyword mode may read only the opening words of '
    'a query; pages may hold only a handful of rows; reaching the next page may need a token or an offset. Set '
    'the total the store reports against the number of rows you have seen; if the answer hinges on seeing every '
    'row, fetch further pages, passing back unchanged whatever token or offset came with the result. Keep keyword '
    "queries brief and lead with the value: the wanted value first, then the product noun. A store's search may "
    'match attribute values and seller names as well as titles, and may demand that every query word be present, '
    'so a handful of required values together with the noun can return almost the entire fitting set. A search '
    "by a part number or a model code can come back empty; a single attempt shows whether this store's search "
    'finds codes at all. When a filtered search returns zero rows, the usual cause is a wrong filter value or '
    'a wrong filter parameter: try again with the label copied letter for letter as the rows of a result show '
    'it, or switch to another filter parameter.\n'
    "- Rows in search results are summaries only. Shops divide a listing's record across views differently: one "
    'view may expose the attribute block (attributes and properties are the same thing here), another just the '
    'offer and a description, and either kind may appear in the side-by-side compare view. Once Terry has learned '
    'which tool exposes attributes, the ledger says so. Each property requirement is judged on those attributes, '
    'so inspect the attributes of every listing you might purchase. A single call that displays several listings '
    'at once saves steps, provided it displays what is required. Never reopen a listing whose state has not changed '
    'since you read it.\n'
    "- Asking the shopper is the job of Terry's interviewer: you never send the shopper a message yourself. The "
    'replies appear in this conversation, and the ledger records them.\n'
    '- A request may also demand intermediate steps: obtaining an approval, sending a message, verifying or reading '
    'something beforehand. Each appears in the ledger. Those that fall due right ahead of the order or of a no-purchase '
    'ending are carried out by Terry itself, at the moment you call the checkout or the ending tool. So never '
    'hold off waiting for them, and never decline on their account. The remaining steps (those not due at the '
    "finish) you perform yourself with the shop's tools; questions about products are the exception and go through "
    "Terry's interviewer.\n"
    '\n'
    'WHAT COUNTS AS FITTING\n'
    "- The kind of product is settled by the listing's category label: only a label that matches the requested "
    'item in its entirety qualifies; an adjacent label never does.\n'
    '- A property requirement is satisfied when the attribute the listing has for that property gives the value. '
    'How the key is written does not matter: spacing, underscore separators, prefixes or camel-cased joins. When '
    'the attribute is a list split by commas or slashes, a single matching element suffices. Matching works word '
    'by word; differences of letter case, of singular versus plural, and ordinary synonyms are tolerated. Containment '
    'counts: a value that includes all the words of the required value meets it despite extra words ("deep navy '
    'blue" satisfies "navy blue"; "very relaxed fit" satisfies "relaxed"). A value made of other words fails: '
    'an abbreviation or a code does not equal the word it abbreviates, and "navy blue" is not "royal blue". Numbers: '
    'convert the units first, then compare; a limit includes its boundary; a value given as a range must sit entirely '
    'within the limit.\n'
    "- An attribute keyed by the requirement's property (that exact name or an ordinary synonym of it) is decisive: "
    'if it states another value, the listing fails, whatever the title or any other key says. Exception: when '
    "a key prefixes the property name with the item's name (for a bag, a key meaning bag shape), its value counts "
    'even if the plain property key states otherwise. A key about another property, even a closely related one, '
    'settles nothing for this requirement: a handle\'s colour tells nothing about the body\'s colour. '
    'After listings are opened, Terry appends [evidence] lines naming, for each requirement, the attribute keys '
    'whose name matches it.\n'
    '- When a listing lacks any attribute for the property, the value sought still counts when it appears under '
    'some other attribute (such as a styles list or a features list that contains it), or in its description or '
    'title. It does not count when the figure is a different kind of quantity (a weight is no substitute for a '
    'capacity), or when a bare number appears with no text saying what it measures.\n'
    "- When the requirement's value is written into the category label (a season or a target audience inside the "
    'label), the category satisfies it. Conversely, a listing filed under the label that was asked for counts '
    'as a product of the requested kind even when its description or title mentions something different; such '
    'a listing is judged solely on its requirements, price and stock.\n'
    '- A requirement may refer to another listing instead of giving its value, for example "the same X as the '
    'item <shop> sells with <property>..." or "within N units of what that listing\'s page states". Resolve '
    'such a requirement first: locate the referenced listing (search on its words, brand, shop or category) and '
    'open its whole page with the single-item tool or the compare tool; a row from a search does not suffice. '
    'If the opened page sends you onward to one more listing that holds the property, locate and open that listing '
    'too by the same method, and so on: the final page in the chain supplies the value. If the requirement wants '
    'whatever goes with the referenced value (its partner or its counterpart), and that page says what its own '
    'value is paired with, then that paired value becomes the requirement. An order is valid only if every page '
    'along the chain was opened. If the requirement accepts any value but the one the referenced listing has, '
    'it removes only the listings sharing exactly that value, and that rule by itself must never cause a decline.\n'
    '- If a listing you intend to buy now points to a further page with more detail about itself, check that page '
    'before the purchase.\n'
    '- If one requested item has to suit another, as an accessory suits a device, pick both jointly, based on '
    "the listings' own wording: a compatibility list, or a series or model code that appears on both pages. If "
    'no accessory meeting its own requirements is compatible with the chosen device, change to a different device '
    'that satisfies its own requirements and that one of those accessories does fit. Never decline while that '
    'device switch has not been tried.\n'
    '- Honour exclusions and closed lists. Wishes (softly worded preferences such as "ideally" or "nice to have") '
    'serve only to break ties.\n'
    "- Sold-out listings must never be bought. The total (each line's price times its quantity, added across all "
    'lines) may not exceed the budget; a total equal to it is fine.\n'
    '\n'
    'CHOOSING\n'
    '- If the request wants the cheapest choice (the smallest price or the lowest total), the order must contain '
    'the lowest-priced listing out of ALL fitting listings. Establish the complete fitting set by paging '
    'through it (rows often arrive unsorted by price; Terry fetches further pages of short result lists itself), '
    'examine the candidates from the cheapest upward, and pick the earliest one that fits fully. If a search tool '
    'accepts a maximum price, use it: list that category with the cap set below what its present best pick costs, '
    'leaving only the cheaper ones to check.\n'
    '- If the request wants the greatest or the smallest amount of a measure, compare every fitting listing on '
    'that measure and take the extreme one; a listing silent on the measure is never the winner.\n'
    '- Otherwise any listing satisfying everything is correct; do not burn steps looking for a better one.\n'
    "- Put each item in the cart the moment an opened listing satisfies all of that item's requirements (for a "
    'price-ranked or measure-ranked pick: only after the fitting set has been compared), then go on to the next '
    'item. A store frequently sells out a line shortly after it enters the cart, and carting early keeps steps '
    'in hand for a replacement.\n'
    "- When several distinct items are requested, each needs its own cart line, every line at that item's requested "
    'quantity.\n'
    '\n'
    'CHANGES DURING THE WORK\n'
    '- Both the shopper and the store may alter the task: a fresh budget, an added or a removed requirement, a '
    'new price, a listing gone out of stock. Such changes show up within tool results, often on a checkout attempt '
    "as a refusal. Any note is valid whichever tool's result carries it. The ledger shows the requirements currently "
    'in force. A refusal does not end the episode: apply all changes, correct the cart and try checkout again.\n'
    '- Prices and stock may also shift without any note. When you call checkout, Terry re-checks the cart lines '
    'and views the cart itself, so do not spend a turn on either.\n'
    '\n'
    'FINISHING\n'
    '- Make the cart contain only the chosen lines, exactly those, removing everything else; then send checkout '
    'as a call by itself, never together with other calls in one turn.\n'
    '- A no-purchase ending is allowed only once a genuine search has found that no listing satisfies everything. '
    'Being low on steps proves no such thing: in that situation, check out with the best cart you hold. The ending '
    "follows the request's own wording; when a remark has to accompany it, keep the remark brief and name no listing.\n"
    '- Each reply must include one or more tool calls. Keep your text brief: the known facts, then the next action.'
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
    "Name the property inside a must with the very word the request used for it: \"for an evening event\" gives "
    "\"occasion: evening\"; \"a 12-inch blade\" gives \"blade length: 12 inch\". Never trade the property for a "
    "neighbouring one.\n"
    "A requirement whose value is taken from a different listing is written as ONE must that carries the other "
    "listing's full description, e.g. \"finish: different from that of the lamp that ShopX sells with a brass "
    "base\". All the words describing the other listing (its shop, its product kind, and any trailing \"with\" clause of "
    "properties and values) stay inside that single must; none of them ever becomes a must of this product.\n"
    "Price, budget, stock and quantity never go into musts. Narrative around the purchase (the intended user, the place "
    "the product goes, earlier trouble, the occasion that prompted the purchase) is context and never a must, except "
    "where it explicitly gives some property's value or bound. Words forming part of the product's own name, such as a "
    "season or a target group mentioned alongside the product, stay in name and never become musts.\n"
    "Exception that does create a must: when the request treats a purpose or event as a trait of the product, that IS "
    "a constraint on that very property (e.g. \"occasion: business\"), unless the store's category label already names "
    "it. This covers: something described as meant for a kind of wear (\"for business wear\"); something said to be made for a "
    "kind of event; a product followed by a separate clause stating its event; a plain statement that the "
    "occasion must equal a value. In contrast these stay context: one specific person's own event (wearing it at my own "
    "wedding), the recipient or place the purchase serves (a present for a co-worker), and its intended sit, feel "
    "or appearance when used.\n"
    "When one phrase names products linked by \"and\" (a single \"I want\" clause with a single quantity, often the "
    "title of a single department of the store), that is ONE item. Split into separate items only if every product gets "
    "a separate quantity or separate requirements.\n"
    "budget: the shopper's cap on the total spend, as a number in whatever unit the request uses; null if absent.\n"
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
    "rules: every instruction in the request about working method or how to finish, each at most 12 words.\n"
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
    "fits: one entry for every requested item that, by the request, must work with, fit or suit some OTHER requested "
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
    "Reading rules, for a requirement R on property P with value V: R holds when the listing's field for P states V; "
    "the key may be spelled any way, a single list member suffices, letter case, singular or plural and simple synonyms "
    "are fine, a value that includes V's words counts (\"dark forest green\" satisfies \"forest green\"), units are "
    "converted, a limit includes its own boundary, and a range has to fall wholly inside. A field whose key names P "
    "itself decides: if it states something other than V, the listing fails; except that a value filed under a key that "
    "prefixes P with the item's own kind (on a backpack, \"backpack color\") still counts even though the bare key for P "
    "disagrees. A key for another property, however related, decides nothing (a strap's colour says nothing about the "
    "main body's colour). Where no field for P exists at all, V stated under some other field, or in the title or "
    "description, is enough; but not a figure of a different sort (a weight is no capacity), and not a lone figure "
    "whose surrounding text never says what it measures. If R is stated by the category label itself, being in that "
    "category satisfies R. The category label must name the whole requested item, not just part of it. A requirement "
    "to differ from another listing's value is taken as met, except where the records reveal that very value.\n"
    "Cart rules: every requested item must have its own cart line carrying the quantity asked; the cart must hold "
    "nothing beyond those lines; no line may be shown as unavailable or sold out; and each line's price multiplied "
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
    "title or description, is enough, but a lone figure that never says what it measures is worth nothing; if the "
    "category label states R, that category meets R; the category label must name the item; a requirement to differ "
    "from another listing's value is taken as met, except where the records reveal that very value; an unavailable listing, or one "
    "over the budget, is not a fit."
)

AUDIT_PROMPT = (
    "An agent means to stop without buying, believing no listing in the store fits. Check the records shown below "
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
    'An agent is near checkout. Some requirements draw their value off a second listing instead of stating it, '
    "and one listing's page may refer the reader onward to yet another listing for the very property. An order "
    'only counts once the agent has fully opened every listing along every such chain. Trace the chain of every '
    'requirement, starting at its own description, link after link, through what the pages themselves say, using '
    'the records placed after this text. OPENED means that page was opened by the agent; ROW means the page is '
    'unread and the listing is known only from a search row or through its fields.\n'
    'Reply with one JSON object only: {"chains": [{"need": the requirement copied word for word, "links": [{"id": '
    'that listing\'s id if some record is this link, else "", "about": how this link is described, by the requirement '
    'or by the preceding page, "find": only where the link is not yet located, search words built from its shop '
    'plus whatever values the description supplies}], "done": true if the final link is OPENED and its page itself '
    "states the property's value; true as well when the requirement wants some value that differs from that one "
    'or is unlike it, because the value on that page is the yardstick; otherwise false, "value": the property\'s '
    'value as the final page states it, but if the requirement wants whatever pairs with that value and the page '
    'also names its partner, that partner; else ""}]}, one entry per requirement.\n'
    "A record counts as a link only when it matches the link's entire description: shop or brand, product kind AND "
    'each property value the description lists (a description of a 6-pack of cans excludes a listing from that '
    'shop with 6 bottles and no pack). If no record fits all of it, the link is unlocated, even when that shop '
    'carries further listings of the same kind. For an unlocated link, find holds a handful of search words: the '
    "shop's name plus the values in the description. Give the links of every chain in sequence and end the list "
    'at the earliest link that is unlocated or lacks an OPENED page, because what that page says is not yet known.'
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
    "A listing in the agent's cart sold out a moment ago. Pick substitutes for it among the remaining records shown "
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
RX_BLOCK_CUT = re.compile(r";\s+|;$|\s\|\s|\n")
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
RX_COUNT_TEXT = re.compile(r"^\s*(\d+)(?:\.0+)?\s*$")
RX_HAS_DIGIT = re.compile(r"\d")
RX_SPACE = re.compile(r"\s")
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
    if not (3 <= len(s) <= 64) or " " in s:
        return False
    return named or bool(RX_HAS_DIGIT.search(v))


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
        return v is None
    return True


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def shape(v, schema, path, faults, cuts=None):
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
            faults.append("%s must take one of %s" % (path, dump(en)))
    ml = schema.get("maxLength")
    if isinstance(v, str) and _num(ml) and len(v) > ml:
        c = cut_at_word(v, int(ml))
        if cuts is not None:
            cuts.append((path, int(ml), c))
        v = c
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
            v = [shape(x, it, "%s[%d]" % (path, i), faults, cuts) for i, x in enumerate(v)]
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
            out[k] = shape(x, props[k], "%s.%s" % (path, k), faults, cuts) if k in props else x
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
        self.list_taken = 0
        self.lists_read = False
        self.list_size_cap = 0
        self.shopper_lost = False
        self.asked_now = set()
        self.item_caps = {}
        self.effort_refused = False
        self.last_args = {}
        self.steps_at_call = 0
        self.cut_noted = set()
        self.fit_codes_dead = False
        self.own_rows = {}
        self.own_searches = []
        self.front_paged = False
        self.series_swept = set()
        self.late_drop = False
        self.dear_checked = set()
        self.search_reopened = False
        self.fit_scan_pages = 0
        self.fit_rows = {}
        self.swept = {}
        self.link_queries = set()
        self.code_memo = {}
        self.chain_page_ids = []
        self.chain_opened_mark = -1
        self.chain_findings = []
        self.early_chain = False
        self.link_memo = {}
        self.link_found = ""
        self.deferred = set()
        self.holds_fired = set()
        self.hold_causes = set()
        self.new_hold_causes = []
        self.cheaper_round = 0
        self.cheaper_named = []
        self.better_by_cart = {}
        self.decline_checks = set()
        self.ever_held = set()
        self.whole_set_memo = None
        self.repaired = []
        self.nearest_read = None
        self.nearest_sentence = ""
        self.near_misses = []
        self.near_made = False
        self.near_rescues = 0
        self.set_memo = None
        self.set_held = set()
        self.proven_mark = None
        self.proven_tries = 0
        self.proven_swaps = 0
        self.settle_held = False
        self.picked_lines = False

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


    def live_all(self):
        out = []
        for it in self.items:
            out.extend(it.live())
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
        note = "[Terry: this search did not honour %s=%s; %d of its rows are priced above it." % (name, gnum(cap), len(over))
        if hints:
            note += (" Browse the category with %s, keeping the same %s, to show only the cheaper listings."
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
        return ("[Terry: buyable rows here from the lowest price up: "
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
                "searching by a requirement's value surfaces only some fitting listings, and rows arrive unordered "
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
                texts.append("%s: %s" % (lid, self.render(rec)[:1500]))
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
                texts.append("%s: %s" % (i, self.render(obs)[:1500]))
        if texts:
            self.trace.append({"kind": "fill_facts", "texts": len(texts)})
        return texts


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
            if isinstance(p, dict) and isinstance(p.get("enum"), list):
                out.append((k, v if isinstance(v, (str, int, float, bool)) or v is None else dump(v)))
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
        return ("\n[Terry] %s is this search's page number, and the call just made displayed its final page. Page %s of "
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


    def record_search(self, args):
        self.searched.append(dict(args))
        if len(self.searched) > 8:
            self.searched = self.searched[-8:]


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




RX_LETTER = re.compile(r"[a-zA-Z]")
RX_VALUE_EXCL = re.compile(r"\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\b", re.I)






class E5(object):


    def _ref_lines(self, told):
        return [t for (i, t) in self.refs if any(t in l for l in told)]


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














RX_Q_FILLER = re.compile(r"\b(?:and|for|like|me|or|please|tell|the|what|which|would|you)\b", re.I)






class E6(object):



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








RX_COUNT_KEY = re.compile(r"\b(?:qty|quantity|count|unit|units)\b")
RX_LIMIT_WORDS = re.compile(r"\b(?:not|no|least|most|under|over|min|max|minimum|maximum|unlike|different)\b")
RX_REMOVE_OPT = re.compile(r"remov|delet|drop|clear|take", re.I)
RX_VAL_SPLIT = re.compile(r"[,/;|]")
GATE_LEAD = "Terry carried out the request's own steps before continuing:"


class E7(object):

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


    def fresh_view(self):
        if self.cart_view is not None and self.cart_seen_at >= self.cart_changed_at:
            return self.cart_view
        return None

    def cart_text(self):
        v = self.fresh_view()
        if v is not None:
            return dump(clip(v, 300))[:3000]
        return "(from Terry's cart record, listing: units) " + dump(self.cart)

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


RX_LINK_STOP1 = re.compile(r"^(?:the|and|with|from|that|comes|listing|page|its|for|of)$")
RX_LINK_STOP2 = re.compile(r"^(?:the|and|with|from|that|comes|listing|page|its|for)$")
RX_RUN_ANY = re.compile(r"[a-z0-9]+")
RX_FIRST_DIGITS = re.compile(r"\d+")
QUIET_TEXT = "Sorry, no listing in the store meets every one of your requirements, so no order is placed."


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

    def shop_rows(self, desc, ids):
        low = desc.lower()
        bad = self._noname_keys()
        return [r for r in ids if r not in self.opened and r in self.rows and self._names_in(r, low, bad)]

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
        return txt + ". Open these listings and purchase what really matches; decline only when none of them fits."


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


RX_QUOTED_ID = re.compile(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"')
SKIP = "[Terry] Not executed, and no step was spent: "
HOLD = "[Terry] Not executed yet, and no step was spent: "
BARE_HOLD = "[Terry] Not executed yet: "


class E9(object):
    def footer(self):
        return "\nCart now: " + self.cart_text()

    def plain_role_tools(self):
        return set(v for k, v in self.roles.items() if k != "search")

    def reader_tools(self):
        return set(self.roles.get(r) for r in "search open offer compare".split() if self.roles.get(r))


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
                                   "content": "[Terry] This call possibly did not execute, because something failed along "
                                              "the way. Inspect the present state before repeating it."})
        self.convo.append({"role": "user", "content": self.status()})


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


    def can_decline(self):
        tool = self.roles.get("refuse")
        if not tool:
            return False
        f = []
        shape(self.quiet_fill(dict(self.brief.get("refuse_args") or {})), self.schemas.get(tool), "args", f)
        return not f


STATUS_REFRESH = 4


COUNT_MIN = 2


COUNT_MAX = 20


TIMES_WORDS = 6


CAP_SANITY = 10


COND_NOTES = 4


KEEP_LINES_STEPS = 4


BUDGET_JUMP = 30


BUDGET_NEAR = 3


WHOLE_PRICE_ROWS = 3


STEP_SECS = 4.0


DIRECTOR_CAP = 8000


FIT_SCAN_PAGES = 4


FIT_SCAN_EPISODE_PAGES = 36


FIT_SCAN_ROOM = 12


FIT_SCAN_ENOUGH = 3


FIT_ROWS_SHOWN = 8


FIT_OPEN_ROOM = 15


FIT_OPENS_ITEM = 3


FIT_OPENS_SCAN = 6


FIT_SCAN_DEADLINE = 160.0


SHOPS_SWEPT = 3


SWEEP_KIND_WORDS = 3


CHAIN_SHOPS = 6


CHAIN_PAGES = 2


CHAIN_LONG_WORDS = 8


WIDE_LINK_SEARCHES = 5


LINK_VALUES = 2


EARLY_CHAIN_ROUNDS = 6


EARLY_CHAIN_PAGES = 12


ROUND_PAGES = 3


LINK_MEMO_SEARCHES = 6


LINK_GAPS = 2


LINK_STEPS = 12


LINK_CANDIDATES = 2


LINK_WORD_LEN = 3


LINK_ID_LEN = 6


TITLE_CODE_MIN = 3


TITLE_CODE_MAX = 40


REWORD_SHARE = 0.6


REWORD_LEAD = 0.15


FINDINGS_KEPT = 12


FINDING_WORDS = 3


LONG_TEXT_CHARS = 40


CHAIN_WIDEN_PASSES = 5


DECLINE_CHAIN_SECS = 21.0


LATE_HUNT_SEARCHES = 2


EXTRA_CHEAPER_ROUNDS = 2


EXTRA_CHEAPER_ROOM = 5


FRONT_PAGES = 2


FRONT_SPAN = 8


FRONT_ROOM = 6


FRONT_SEARCHES = 2


DEAR_ROOM = 12


DEAR_ROWS = 4


DEAR_SHARE = 2.0 / 3


READS_AFTER_PAGE = 2


DEAR_KEEP = 8


BIG_ROOM = 20


SERIES_PAGES = 6


SERIES_KEEP = 10


LOOSE_SHARE = 0.4


LOOSE_FLOOR = 0.5


CHOICE_SETS = 3


ONE_PAGE_ROOM = 6


SHORT_READ_ROOM = 14


SHORT_READ_KEEP = 11


PEER_KEEP = 5


DESCENT_WHOLE = 30


DESCENT_PAGES = 2


DESCENT_ROUNDS = 5


SINGULAR_ROWS = 2


HOLD_ROOM = 3


WEAK_LINES = 2


WEAK_SHOWN = 3


SAME_MEASURE_KEYS = 2


FIT_HOLD_SHOWN = 4


BARE_FACTOR = 10


LITERAL_SHOWN = 2


WEAK_GUARD_READS = 3


WEAK_UNREAD_ROOM = 5


SET_ROOM = 5


CODE_LINKS_SHOWN = 3


NEAREST_SECS = 26.0


NEAREST_SENTENCE = 12


LATE_ORDER_STEPS = 5


LATE_REVIEW_DEADLINE = 200.0


EXTRA_ROUND_TURNS = 6


EXTRA_ROUND_SPEND = 0.02


NEAREST_PICK_SECS = 12.0


NEAREST_RECORDS = 36


NEAREST_PER_ITEM = 4


NEAREST_RESCUES = 2


NEAREST_RECORD_CHARS = 700


SET_ITEMS = 6


SET_COMBOS = 300


PROVEN_TRIES = 2


PROVEN_SWAPS = 2


PROVEN_SECS = 40.0


SETTLE_STEPS = 12


SETTLE_SECS = 60.0


PICK_SECS = 30.0


PICK_ROWS = 10


RX_COND = re.compile(r"\b(?:unless|except\s+(?:if|when|where)|only\s+(?:if|when|where)|provided(?:\s+that)?|as\s+long\s+as|so\s+long\s+as|in\s+case|if)\b", re.I)


RX_WAIVER = re.compile(r"(?:unless|except)\b", re.I)


RX_COND_VERB = re.compile(r"\b(?:is|are|was|were|be|has|have|uses?|comes?\s+with|carries|takes)\b", re.I)


RX_POINTER_TALK = re.compile(r"\b(?:page|listing|listed|points?|pointer|sells?|sold|shop|store|seller|vendor)\b", re.I)


RX_SAME = re.compile(r"\b(?:same|identical|matching|match|matches|matched|equal|equals|equalling|alike)\b", re.I)


RX_AUX = re.compile(r"\b(?:should|must|shall|will|would|could|can|may|might|needs?\s+to|has\s+to|have\s+to|ought\s+to|is\s+to|are\s+to)\s+(?:be\s+)?", re.I)


RX_LEAD_FUNC = re.compile(r"(?:should|must|shall|will|would|could|can|may|might|need|needs|has|have|ought|is|are|was|were|be|been|to|that|which|it|its|the|a|an)\b *", re.I)


RX_JOINER = re.compile(r"\s+(?:and|or)\s+|\s*&\s*", re.I)


RX_CODE_INTRO = re.compile(r"\b(?:for|fit|fits|fitting|fitment|compatible(?:\s+with)?|works\s+with|model|code)\s*:?\s*$", re.I)


RX_FLOOR_BEFORE_SUM = re.compile(r"(?:\bno\s+(?:less|fewer)\s+than|(?<!\bno )(?<!\bnot )\b(?:at\s+least|min(?:imum)?|more\s+than|over|above|upwards\s+of))\s*$", re.I)


RX_MONEY_SIGN = re.compile(r"[^\w\s.,;:!?()\[\]{}\"'/%&+=<>#@~^|*\\\-]")


RX_REMOVAL_MODE = re.compile(r"remov|delet|drop|clear|take|subtract|minus|decreas|reduc", re.I)


RX_OFFSET_SEG = re.compile(r"(?:^|_)(?:offset|start|skip|from|begin)(?:_|$)", re.I)


NUM_PAT = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


UNIT_PAT = "[A-Za-z%°µ\"'][A-Za-z%°µ]{0,11}(?![A-Za-z])"


RX_FIG_UNIT = re.compile(r"(?<![\w.])(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d+))?\s*-?\s*(?P<unit>" + UNIT_PAT + ")")


RX_RANGE_UNIT = re.compile(r"(?<![\w.])(?P<a>\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(?P<b>\d+(?:\.\d+)?)\s*-?\s*(?P<unit>" + UNIT_PAT + ")")


RX_BETWEEN = re.compile(r"\bbetween\s+(?P<an>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<ad>\d+))?\s*(?:[A-Za-z%°µ\"']{1,12}\s*)?(?:and|to|-|–)\s*(?P<bn>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<bd>\d+))?\s*-?\s*(?P<unit>" + UNIT_PAT + ")", re.I)


RX_FLOOR_INC_END = re.compile(r"\b(?:at\s+least|no\s+(?:less|fewer|lower|smaller|shorter)\s+than|not\s+(?:less|fewer|below|under)(?:\s+than)?|min|minimum|from|upwards\s+of)\s*:?\s*$", re.I)


RX_FLOOR_STRICT_END = re.compile(r"(?<!no )(?<!not )\b(?:(?:more|greater|higher|larger|bigger|longer)\s+than|over|above|exceeding)\s*:?\s*$", re.I)


RX_CAP_INC_END = re.compile(r"\b(?:at\s+most|no\s+(?:more|greater|higher|larger|bigger|longer)\s+than|not\s+(?:more|over|above)(?:\s+than)?|max|maximum|up\s+to|within)\s*:?\s*$", re.I)


RX_CAP_STRICT_END = re.compile(r"(?<!no )(?<!not )\b(?:(?:less|fewer|lower|smaller|shorter)\s+than|under|below)\s*:?\s*$", re.I)


RX_FLOOR_AFTER = re.compile(r"\s*(?:\+|(?:or\s+(?:more|higher|greater|larger|above|over|better|longer)|and\s+(?:up|above|over|more)|plus)\b)", re.I)


RX_CAP_AFTER = re.compile(r"\s*(?:or\s+(?:less|lower|fewer|smaller|below|under|shorter)|and\s+(?:under|below|less))\b", re.I)


RX_NEGATION = re.compile(r"\b(?:not|no|never|without|excluding|except|unlike|anything\s+but|other\s+than|different\s+from)\b", re.I)


RX_DIFFERENCE = re.compile(r"\b(?:unlike|other\s+than|differ\w*|not\s+the\s+same)\b", re.I)


RX_FILLER = re.compile(r"(?:a|an|the|and|or|of)", re.I)


RX_LIST_SEP = re.compile(r"\s*[,;/|]\s*")


RX_LIST_PUNCT = re.compile(r"[,;/|()\[\]{}]+")


RX_OWN_PRICE_FIELD = re.compile(r"price|cost", re.I)


RX_FREE_TEXT_SEG = re.compile(r"(?:^|_)(?:text|texts|body|message|messages|note|notes|content|contents|question|questions|msg|say|comment|comments|reason|reasons|justif[a-z0-9]*|explan[a-z0-9]*|why|remark|remarks)(?:_|$)", re.I)


RX_PLAIN_NUMBER = re.compile(r"(?<![\w.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d)")


RX_BOUND_WORD = re.compile(r"(?:not|no|other\s+than|unlike|at\s+most|at\s+least|under|over|below|above|less|more|between|within|up\s+to|or|same|listing)", re.I)


RX_OWN_PRICE_TEXT = re.compile(r"\s*[^\d\s]{0,3} ?(\d[\d,]*(?:\.\d+)?)\s*[A-Za-z]{0,6}\.?\s*")


RX_BARE_NUM = re.compile(r"\s*\d+(?:\.\d+)?\s*")


RX_BRACKETED = re.compile(r"\([^)]*\)|\[[^\]]*\]")


RX_EDGE_JUNK = re.compile(r"^[^\w%]+|[^\w%]+$")


RX_RUN_WORD = re.compile(r"[a-z0-9%]+")


RX_FIT_CODE_ANY = re.compile(RX_FIT_CODE.pattern, re.I)


RX_LETTER_RUN = re.compile(r"[a-z]+")


RX_NAME_RUN = re.compile(r"[A-Za-z0-9]+")


RX_REL_TOKEN = re.compile(r"[A-Za-z0-9']+")


RX_SENT_CUT = re.compile(r"(?<=[.!?;])\s+|\n+")


RX_TIMES_COUNT = re.compile("(?<![\\w.,])(\\d{1,2})\\s*[xX×*]\\s+(?=[A-Za-z])")


RX_TIMES_STOP = re.compile(r"[.,;:!?()\n]")


RX_HAS_VOWEL = re.compile(r"[aeiouy]")


RX_ONLY_LETTERS = re.compile(r"[a-z]+")


RX_LIST_MARKS = re.compile(r"[,/;|]")


def is_money_sign(ch):
    return isinstance(ch, str) and len(ch) == 1 and bool(RX_MONEY_SIGN.fullmatch(ch))


def is_empty_field(v):
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict)):
        return not v
    return False


def compact_rows(v, depth=0):
    if depth > 4:
        return v
    if isinstance(v, list):
        if len(v) >= 2 and all(isinstance(x, dict) for x in v):
            return [dict((k, x) for k, x in r.items() if not is_empty_field(x)) for r in v]
        return [compact_rows(x, depth + 1) for x in v]
    if isinstance(v, dict):
        return dict((k, compact_rows(x, depth + 1)) for k, x in v.items())
    return v


def own_price_value(v):
    if isinstance(v, dict):
        for x in v.values():
            if _num(x):
                return float(x)
        return None
    if isinstance(v, str):
        m = RX_OWN_PRICE_TEXT.fullmatch(v)
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None
    if _num(v):
        return float(v)
    return None


def bound_word(w):
    w = (w or "").strip()
    return bool(w) and bool(RX_BOUND_WORD.fullmatch(w))


def first_unit_hit(rx, text):
    for m in rx.finditer(text or ""):
        if not bound_word(m.group("unit")):
            return m
    return None


def same_stem(a, b):
    a = str(a).lower()
    b = str(b).lower()
    if a == b:
        return True
    if not RX_ONLY_LETTERS.fullmatch(a) or not RX_ONLY_LETTERS.fullmatch(b):
        return False
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    if n < 4:
        return False
    for rest in (a[n:], b[n:]):
        if rest and not (2 <= len(rest) <= 4 and RX_HAS_VOWEL.search(rest)):
            return False
    return True


def words_hold(wanted, held):
    wanted = set(wanted)
    held = set(held)
    if not wanted:
        return False
    for w in wanted:
        if w in held:
            continue
        if any(same_stem(w, h) for h in held):
            continue
        return False
    return True


def initials(text):
    ws = RX_ALNUM.findall(str(text).lower())
    out = set()
    for i in range(len(ws)):
        for j in range(i + 3, len(ws) + 1):
            out.add("".join(w[0] for w in ws[i:j]))
    return out


def compact_text(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def is_conditional(req):
    req = str(req)
    val = req.split(":", 1)[1] if ":" in req else req
    return bool(RX_COND.search(val)) and not RX_POINTER_TALK.search(val)


def bare_number(v):
    if _num(v):
        return float(v)
    if isinstance(v, str) and RX_BARE_NUM.fullmatch(v):
        return float(v.strip())
    return None


def bare_note(value, lim):
    if not lim or not lim.get("unit") or lim.get("figure") is None or lim["figure"] <= 0:
        return ""
    x = bare_number(value)
    if x is None or x <= 0:
        return ""
    f = float(lim["figure"])
    if max(x / f, f / x) >= BARE_FACTOR:
        return "[a bare number with no unit; read in this requirement's unit it would be %s %s]" % (gnum(x), lim["unit"])
    return ""


def figure_value(whole, dec=None):
    s = str(whole).replace(",", "")
    if dec:
        s += "." + dec
    return float(s)


def req_limit(value):
    t = str(value or "")
    m = RX_BETWEEN.search(t)
    if m and not bound_word(m.group("unit")):
        a = figure_value(m.group("an"), m.group("ad"))
        b = figure_value(m.group("bn"), m.group("bd"))
        return {"kind": "range", "low": min(a, b), "high": max(a, b), "strict": False, "figure": a,
                "unit": m.group("unit"), "decimal": bool(m.group("ad") or m.group("bd"))}
    r = first_unit_hit(RX_RANGE_UNIT, t)
    f = first_unit_hit(RX_FIG_UNIT, t)
    if r is not None and (f is None or r.start() <= f.start()):
        before = t[:r.start()]
        if not any(rx.search(before) for rx in (RX_FLOOR_INC_END, RX_FLOOR_STRICT_END, RX_CAP_INC_END, RX_CAP_STRICT_END)):
            a = float(r.group("a"))
            b = float(r.group("b"))
            return {"kind": "range", "low": min(a, b), "high": max(a, b), "strict": False, "figure": a,
                    "unit": r.group("unit"), "decimal": "." in r.group("a") or "." in r.group("b")}
    if f is None:
        return None
    x = figure_value(f.group("num"), f.group("dec"))
    before = t[:f.start()]
    after = t[f.end():]
    strict = False
    if RX_FLOOR_INC_END.search(before) or RX_FLOOR_AFTER.match(after):
        kind = "min"
    elif RX_CAP_INC_END.search(before) or RX_CAP_AFTER.match(after):
        kind = "max"
    elif RX_FLOOR_STRICT_END.search(before):
        kind, strict = "min", True
    elif RX_CAP_STRICT_END.search(before):
        kind, strict = "max", True
    else:
        kind = "exact"
    return {"kind": kind, "low": x, "high": x, "strict": strict, "figure": x, "unit": f.group("unit"),
            "decimal": bool(f.group("dec"))}


def measured_limit(value):
    lim = req_limit(value)
    if lim is None:
        return None
    if lim["decimal"] or lim["kind"] != "exact" or RX_OPEN_EXCLUDE.search(str(value)):
        return lim
    return None


def single_root(word):
    r = roots(word)
    return next(iter(r)) if len(r) == 1 else ""


def relation_part(sentence, prop):
    sentence = str(sentence)
    pr = roots(prop)
    toks = list(RX_REL_TOKEN.finditer(sentence))
    start = 0
    for i, t in enumerate(toks):
        r = single_root(t.group(0))
        if r and r in pr:
            j = i
            while j + 1 < len(toks):
                r2 = single_root(toks[j + 1].group(0))
                if r2 and r2 in pr:
                    j += 1
                else:
                    break
            start = toks[j].end()
            break
    rest = sentence[start:]
    m = RX_AUX.search(rest)
    if m:
        rest = rest[m.end():]
    rest = rest.strip(" ,;:.")
    while True:
        m = RX_LEAD_FUNC.match(rest)
        if not m or m.end() == 0:
            break
        rest = rest[m.end():]
    return rest.strip(" ,;:.")


def name_pattern(name):
    runs = RX_NAME_RUN.findall(str(name or ""))
    if not runs:
        return None
    return re.compile(r"[\s\-]+".join(re.escape(r) for r in runs) + r"(?:s|es)?\b", re.I)


def _name_size(name):
    return len(RX_NAME_RUN.findall(str(name or "")))


def count_before_name(request, name, other_names):
    pat = name_pattern(name)
    if pat is None:
        return None
    size = _name_size(name)
    lead = "(?<![\\w.,])(\\d{1,2})\\s*(?:[xX×*]\\s*)?(?="
    rx = re.compile(lead + pat.pattern + ")", re.I)
    longer = []
    for o in other_names:
        if o and _name_size(o) > size:
            op = name_pattern(o)
            if op is not None:
                longer.append(op)
    nums = set()
    for m in rx.finditer(request or ""):
        if any(lp.match(request, m.end()) for lp in longer):
            continue
        nums.add(int(m.group(1)))
    if len(nums) == 1:
        n = nums.pop()
        if COUNT_MIN <= n <= COUNT_MAX:
            return n
    return None


def times_counts(request):
    out = []
    text = request or ""
    for m in RX_TIMES_COUNT.finditer(text):
        rest = RX_TIMES_STOP.split(text[m.end():], 1)[0]
        words = " ".join(rest.split()[:TIMES_WORDS])
        n = int(m.group(1))
        if words and n >= 1:
            out.append((n, words))
    return out


def sentences(text):
    return [p.strip() for p in RX_SENT_CUT.split(text or "") if p and p.strip()]


def figures_of(text):
    return [(float(m.group(0).replace(",", "")), m.start(), m.end()) for m in RX_PLAIN_NUMBER.finditer(str(text or ""))]


def record_leaves(v, depth=0):
    if depth > 6:
        return []
    out = []
    if isinstance(v, dict):
        for k, x in v.items():
            if isinstance(x, (dict, list)):
                out.extend(record_leaves(x, depth + 1))
            elif isinstance(x, bool) or x is None:
                continue
            elif isinstance(x, (str, int, float)):
                out.append(str(x) if depth == 0 else "%s %s" % (k, x))
    elif isinstance(v, list):
        for x in v:
            out.extend(record_leaves(x, depth + 1))
    elif isinstance(v, (str, int, float)) and not isinstance(v, bool):
        out.append(str(v))
    return out


def fold_word(w):
    w = re.sub(r"[^a-z0-9%]", "", str(w).lower())
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return RX_OUR_END.sub("or", w)


def key_words(text):
    t = RX_BRACKETED.sub(" ", str(text or ""))
    out = []
    for w in word_form(t).split():
        f = fold_word(w)
        if f:
            out.append(f)
    return out


def key_matches(key, prop):
    a = key_words(key)
    b = key_words(prop)
    return bool(a) and bool(b) and (set(a) == set(b) or a[-1] == b[-1])


def literal_words(text):
    t = RX_LIST_PUNCT.sub(" ", word_form(str(text or "")))
    out = set()
    for w in t.split():
        w = RX_EDGE_JUNK.sub("", w)
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w and not RX_FILLER.fullmatch(w):
            out.add(w)
    return out


def run_words(text):
    out = []
    for w in RX_RUN_WORD.findall(str(text or "").lower()):
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def cut_at_word(text, n):
    text = str(text)
    n = int(n)
    cut = text[:n]
    if n <= 0 or (len(text) > n and text[n].isspace()) or (cut and cut[-1].isspace()) or " " not in cut.strip():
        r = cut.rstrip()
        return r if r else cut
    r = cut[:cut.rfind(" ")].rstrip()
    return r if r else cut


def words_apart(value, text):
    wanted = [w for w in run_words(value) if not RX_FILLER.fullmatch(w)]
    if len(wanted) < 2:
        return None
    n = len(wanted)
    apart = False
    for part in RX_LIST_MARKS.split(str(text or "")):
        pw = run_words(part)
        for i in range(len(pw) - n + 1):
            if pw[i:i + n] == wanted:
                return "together"
        k = 0
        for w in pw:
            if k < n and w == wanted[k]:
                k += 1
        if k == n:
            apart = True
    return "apart" if apart else None


def plain_words_value(value):
    v = str(value or "")
    if RX_LIST_MARKS.search(v) or RX_HAS_DIGIT.search(v) or RX_OPEN_EXCLUDE.search(v) or RX_NEGATION.search(v):
        return False
    return len([w for w in run_words(v) if not RX_FILLER.fullmatch(w)]) >= 2


def unit_spelling(u):
    s = re.sub("[^a-z%°µ\"']", "", str(u or "").lower())
    if len(s) > 2 and s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s


def same_unit(a, b):
    a = unit_spelling(a)
    b = unit_spelling(b)
    if not a or not b:
        return False
    if a == b:
        return True
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    return len(s) >= 2 and l.startswith(s) and len(l) - len(s) <= 4


def text_codes(text):
    return [RX_GAP.sub(" ", m.group(0)).strip().lower() for m in RX_FIT_CODE_ANY.finditer(str(text or ""))]


def code_match(listing_code, required_code):
    return code_covers(listing_code, required_code) or code_covers(required_code, listing_code)


covers_code = code_match


def code_family(code):
    c = re.sub(r"[ -]", "", str(code or "").lower())
    m = RX_LETTER_RUN.match(c)
    return m.group(0) if m else ""


def figures_in_unit(value, unit, bare=True):
    t = str(value if value is not None else "")
    figs = []
    ranged = False
    for m in RX_RANGE_UNIT.finditer(t):
        if same_unit(m.group("unit"), unit):
            figs.append(float(m.group("a")))
            figs.append(float(m.group("b")))
            ranged = True
    t2 = RX_RANGE_UNIT.sub(lambda m: " " * len(m.group(0)), t)
    for m in RX_FIG_UNIT.finditer(t2):
        if same_unit(m.group("unit"), unit):
            figs.append(figure_value(m.group("num"), m.group("dec")))
    if not figs and bare:
        b = bare_number(value)
        if b is not None:
            figs.append(b)
    return figs, ranged


NEAREST_RULE_PROMPT = (
    "Study the shopper's request that follows. Judge one question only: does the request say what should happen when "
    "no listing the store has satisfies all of its requirements, namely that the closest candidate (fewest "
    "unmet requirements) is to be bought regardless? Reply with one JSON object and no text around it: "
    "{\"nearest\": true or false, \"sentence\": \"the request's own sentence that states this rule, copied verbatim, or "
    "an empty string\"}.")


NEAREST_PICK_PROMPT = (
    "Should no candidate satisfy every requirement, the shopper wants the candidate with the fewest unmet requirements "
    "ordered in its place. A listing that lacks stock, or one that would break the budget, counts that as one more "
    "unmet requirement. Ties between equally close candidates go to the cheapest, or to any candidate inside a margin "
    "on price that the request grants.\n"
    "Below come the items requested, the live requirements, and, per item, the listings already opened.\n"
    "Your task: one closest pick per item. It must match the item's kind, be available, hold the whole order within "
    "budget, and, for a part that must work with a second item of the request, suit whatever you pick for that other "
    "item. Reply with one JSON object and no text around it: {\"lines\": [{\"item\": <item number>, \"id\": \"<listing id "
    "copied exactly as displayed>\", \"misses\": [\"<each requirement this pick fails>\"]}]}. Omit an item ONLY when no "
    "displayed listing is of its kind.")


SETTLE_HOLD_TEXT = (
    HOLD + "the request accepts a closest listing only as a fallback for when no listing satisfies every requirement, "
    "yet cart line %(ids)s shows a value that contradicts a requirement while %(steps)s steps remain. Place a "
    "closest-listing order ONLY once a genuine search has turned up nothing that satisfies everything: browse the "
    "shelf of the item, filtered by the section or label it belongs to, keep paging, and search for the missing value "
    "together with the noun that names the product; open any candidate that might qualify. After that, check out, "
    "with the closest listing if no better candidate appeared.")


LINE_PICK_PROMPT = (
    "A checkout is imminent. For every cart line listed below, pick among that line's candidates (they include the "
    "line's current listing; all their records are shown) the one listing the shopper ought to receive: it must "
    "satisfy all live requirements and, should the pick rule rank by price or by a measure, it must beat the other "
    "candidates under that rule. The current listing STAYS when it satisfies everything and no other candidate is "
    "favoured by the pick rule. Reply with one JSON object and no text around it: {\"lines\": [{\"line\": \"<id of the "
    "cart line>\", \"pick\": \"<chosen listing id, copied exactly as displayed>\", \"why\": \"<brief reason>\"}]}.")


RX_SPACED_UNIT = re.compile(r"(\d)\s+([A-Za-z]{1,5})\b")


RX_GLUED_UNIT = re.compile(r"(\d)([A-Za-z]{1,5})\b")


RX_STOP_LIMIT = re.compile(r"length|max_tokens|max_output_tokens")


RX_REL_LEAD = re.compile(r"(?:the|a|an|of|one|ones|that|which)", re.I)


RX_ALPHA3_K02 = re.compile(r"[a-z]{3,}")


RX_AFTER_SUM_WORD = re.compile(r"\s*-?\s*([A-Za-z]{1,12})\b")


RX_AFTER_BUDGET_WORD = re.compile(r"\s*([A-Za-z]{1,12})\b")


RX_FOLD_RUN = re.compile(r"[a-z0-9%]+")


RX_LETTER_RUN_K02 = re.compile(r"[A-Za-z]+")


SLACK = 1e-9


def meets_limit(lim, figs, ranged=False):
    if not figs or not lim:
        return False
    kind = lim.get("kind")
    lo, hi = lim.get("low"), lim.get("high")
    strict = lim.get("strict")
    if kind == "min":
        ok = [(f > lo + SLACK) if strict else (f >= lo - SLACK) for f in figs]
    elif kind == "max":
        ok = [(f < hi - SLACK) if strict else (f <= hi + SLACK) for f in figs]
    elif kind == "range":
        ok = [lo - SLACK <= f <= hi + SLACK for f in figs]
    else:
        x = lim.get("figure")
        return any(abs(f - x) < SLACK for f in figs)
    return all(ok) if ranged else any(ok)


def breaks_limit(lim, figs):
    if not lim or lim.get("kind") == "exact":
        return False
    return any(not meets_limit(lim, [f], False) for f in figs)


def spacing_flip(value):
    v = str(value)
    if RX_SPACED_UNIT.search(v):
        return RX_SPACED_UNIT.sub(r"\1\2", v)
    return RX_GLUED_UNIT.sub(r"\1 \2", v)


def plain_text(v):
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def ref_note(req, value):
    prop, colon, _ = str(req).partition(":")
    prop = prop.strip() if colon else ""
    if not prop:
        prop = "this property"
    return ("the page this reference points at gives %s: %s; the requirement's other details only identify that page "
            "and do not constrain this item" % (prop, value if value is not None else ""))


def stopped_at_limit(choice):
    if not isinstance(choice, dict):
        return False
    for k in "finish_reason native_finish_reason".split():
        v = choice.get(k)
        if v is not None and RX_STOP_LIMIT.fullmatch(str(v).strip().lower()):
            return True
    return False


RX_VALUE_SPLIT_K03 = re.compile(r"\s*(?:,|/|\b(?:or|either|nor)\b)\s*", re.I)


RX_LEAD_FILL_K03 = re.compile(r"^(?:(?:the|an?|one|of)\s+)+", re.I)


RX_NOT_LEAD_K03 = re.compile(r"\s*not\b\s*", re.I)


RX_POSSESSIVE_K03 = re.compile(r"'s\b", re.I)


RX_PRICE_EXACT_K03 = re.compile(r"price|cost")


RX_HAS_ALNUM_K03 = re.compile(r"[A-Za-z0-9]")


RX_CHOICE_SEP_K04 = re.compile(r"\s*(?:,|/|;|\bor\b|\beither\b)\s*", re.I)


RX_OR_EITHER_K04 = re.compile(r"\b(?:or|either)\b", re.I)


RX_SPACED_NUM_K04 = re.compile(r"[A-Za-z][A-Za-z \-]*?\s(\d{3,})")


RX_TITLE_WORD_K04 = re.compile(r"[a-z]{3,}")


RX_HAS_ALNUM_K04 = re.compile(r"[A-Za-z0-9]")


RX_WS_OR_K05 = re.compile(r"\s+or\s+")


def record_cut(rec, sclip, limit):
    t = dump(clip(rec, sclip))
    if len(t) <= limit or not isinstance(rec, dict):
        return t[:limit]
    flat = [(k, v) for k, v in rec.items() if not isinstance(v, (dict, list))]
    nest = [(k, v) for k, v in rec.items() if isinstance(v, (dict, list))]
    return dump(clip(dict(flat + nest), sclip))[:limit]


RX_SEG_SPLIT_K06 = re.compile(r"[,;()]|\b(?:and|with|or)\b")


RX_OR_SPLIT_K06 = re.compile(r"\s+or\s+")


RX_WS_OR_K06 = re.compile(r"\s+or\s+")


RX_NUM_TOK_K06 = re.compile(r"\d+(?:\.\d+)?")


RX_LOW_RUN_K06 = re.compile(r"[a-z]+")


def _fold_s(w):
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _norm_k06(t):
    return " ".join(RX_RUN_ANY.findall(word_form(t)))


def _shares_k06(a, b):
    for w in a:
        if w in b or any(same_stem(w, x) for x in b):
            return True
    return False


RX_HAS_ALNUM_K07 = re.compile(r"[A-Za-z0-9]")


RX_DIGIT_RUN_K07 = re.compile(r"\d+")


def attr_block_raw(rec):
    if not isinstance(rec, dict):
        return None
    best = None
    for v in rec.values():
        if isinstance(v, dict) and len(v) >= 2:
            sc = {k: x for k, x in v.items() if isinstance(x, (str, int, float)) and not isinstance(x, bool)}
            if len(sc) >= 2 and (best is None or len(sc) > len(best)):
                best = sc
    return best


RX_OR_CHOICE_K08 = re.compile(r"\s+or\s+")


RX_ROUND_BR_K08 = re.compile(r"\([^)]*\)")


RX_OR_CHOICE_K09 = re.compile(r"\s+or\s+")


RX_WS_OR_K09 = re.compile(r"\s+or\s+")


RX_ROUND_BR_K09 = re.compile(r"\([^)]*\)")


RX_LETTER_K09 = re.compile(r"[A-Za-z]")


RX_PLACEHOLDER_REF = re.compile(r"^(?:as stated on|whatever the listing)", re.I)


ACTED = "[Terry] The decline was not sent because Terry acted in its place: "


READS_HEAD = "Terry fetched the attribute records of the listings below for this review, at one step apiece:\n"


def reads_heading(reads):
    if not reads:
        return ""
    return READS_HEAD + "\n".join(reads) + "\n"


RX_WS_OR_K10 = re.compile(r"\s+or\s+", re.I)


RX_TRUE_YES_K10 = re.compile(r"true|yes")


RX_MONEY_FAULT_K10 = re.compile(r"is sold out|above the budget|above the per-item limit|above the ceiling")


def combos(lists, cap):
    out = [[]]
    for lst in lists:
        nxt = []
        for c in out:
            for e in lst:
                nxt.append(c + [e])
        out = nxt[:cap]
    return out


class K02(object):
    def clock_bound(self):
        return (self.steps_left() - self.reserve()) * STEP_SECS > self.room()

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
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        res = model_client(self.room()).post_verbose("/inference/chat/completions", body)
        data = res.data
        if isinstance(data, dict) and data.get("choices"):
            u = data.get("usage")
            cost = u.get("cost") if isinstance(u, dict) else None
            if _num(cost):
                self.spend += cost
            ch = data["choices"][0]
            msg = ch.get("message") if isinstance(ch, dict) else None
            if tools and isinstance(msg, dict) and not msg.get("tool_calls") and stopped_at_limit(ch):
                self.trace.append({"kind": "cut_reply", "model": model, "len": len(str(msg.get("content") or ""))})
                return None, 0
            if isinstance(msg, dict) and (msg.get("content") or msg.get("tool_calls")):
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

    def env(self, name, args):
        self.last_args[name] = dict(args) if isinstance(args, dict) else {}
        body = dict(self.binding)
        body["step"] = self.step + 1
        body["action"] = {"name": name, "args": args}
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
        entries = [e for e in raw if isinstance(e, dict)]
        items = []
        for idx, e in enumerate(entries):
            q = count_of(e.get("quantity"))
            q = q if q and q > 0 else 1
            ms = e.get("musts")
            ms = ms if isinstance(ms, list) else ([ms] if ms else [])
            musts = [str(x) for x in ms if x]
            nm = e.get("name")
            if q == 1 and nm is not None and str(nm).strip():
                try:
                    others = [str(x.get("name") or "") for j, x in enumerate(entries) if j != idx]
                    c = count_before_name(self.request, str(nm), others)
                    if c is not None:
                        q = c
                        self.trace.append({"kind": "count_before_name", "item": str(nm), "qty": c})
                except Exception as ex:
                    self.trace.append({"kind": "soft_crash", "method": "count_before_name", "error": str(ex)[:200]})
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
            self.lookups.append(t)
            if not any(r == t for _, r in allm):
                p = word_form(t.partition(":")[0])
                for _, r in allm:
                    if word_form(r.partition(":")[0]) == p:
                        t = r
                        break
            no = next((i for i, r in allm if r == t), 1)
            if t and self._not_a_reference(no, t):
                continue
            if t and (no, t) not in self.refs:
                self.refs.append((no, t))
        try:
            self._keep_relation_wording()
        except Exception as ex:
            self.trace.append({"kind": "soft_crash", "method": "keep_relation_wording", "error": str(ex)[:200]})
        self.unmix()
        try:
            self._narrow_shared()
        except Exception as ex:
            self.trace.append({"kind": "soft_crash", "method": "narrow_shared", "error": str(ex)[:200]})
        bud = b.get("budget")
        if isinstance(bud, str) and RX_PLAIN_NUM.match(bud):
            bud = float(bud)
        self.budget = bud if _num(bud) and bud > 0 else None
        try:
            self._times_quantities()
        except Exception as ex:
            self.trace.append({"kind": "soft_crash", "method": "times_quantities", "error": str(ex)[:200]})
        try:
            self._read_item_caps()
        except Exception as ex:
            self.trace.append({"kind": "soft_crash", "method": "read_item_caps", "error": str(ex)[:200]})
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

    def _not_a_reference(self, no, t):
        val = t.split(":", 1)[1] if ":" in t else t
        if RX_COND.search(val) and not RX_POINTER_TALK.search(val):
            self.trace.append({"kind": "ref_skip_condition", "req": t})
            return True
        if len(self.items) >= 2 and not RX_POINTER_TALK.search(val):
            vr = roots(val)
            for j, other in enumerate(self.items, 1):
                if j == no:
                    continue
                nr = roots(other.name)
                if nr and nr <= vr:
                    self.trace.append({"kind": "ref_skip_relation", "req": t, "item": j})
                    return True
        return False

    def _keep_relation_wording(self):
        sents = sentences(self.request)
        for idx in range(len(self.refs)):
            no, t = self.refs[idx]
            prop, colon, val = t.partition(":")
            if not colon or not prop.strip() or not val.strip() or not RX_SAME.search(val):
                continue
            wanted = roots(val)
            best, bs = None, 0
            for s in sents:
                k = len(roots(s) & wanted)
                if k > bs:
                    best, bs = s, k
            if best is None or bs == 0 or RX_SAME.search(best):
                continue
            rel = relation_part(best, prop)
            shared = roots(rel) & roots(t)
            if len(shared) < 3:
                continue
            words = RX_REL_TOKEN.findall(rel)
            first = 0
            for i, w in enumerate(words):
                if single_root(w) in shared:
                    first = i
                    break
            if not any(len(w) >= 2 and not RX_REL_LEAD.fullmatch(w) for w in words[:first]):
                continue
            new = prop.strip() + ": " + rel
            if new == t:
                continue
            if 1 <= no <= len(self.items):
                it = self.items[no - 1]
                it.musts = [new if r == t else r for r in it.musts]
            self.refs[idx] = (no, new)
            self.lookups = [new if x == t else x for x in self.lookups]
            self.brief["lookups"] = [new if str(x) == t else x for x in items_of(self.brief.get("lookups"))]
            self.trace.append({"kind": "relation_wording", "old": t, "new": new})

    def _names_item(self, sentence, n):
        it = self.items[n - 1]
        pat = name_pattern(it.name)
        if pat is None:
            return False
        rx = re.compile("(?<![A-Za-z0-9])" + pat.pattern, re.I)
        mine = [(m.start(), m.end()) for m in rx.finditer(sentence)]
        if not mine:
            return False
        nr = roots(it.name)
        spans = []
        if nr:
            for j, o in enumerate(self.items, 1):
                if j == n:
                    continue
                orr = roots(o.name)
                op = name_pattern(o.name)
                if op is not None and nr < orr:
                    orx = re.compile("(?<![A-Za-z0-9])" + op.pattern, re.I)
                    spans.extend((m.start(), m.end()) for m in orx.finditer(sentence))
        return any(not any(a <= s and e <= b for a, b in spans) for s, e in mine)

    def _narrow_shared(self):
        if len(self.items) < 2:
            return
        pointers = set(t for _, t in self.refs) | set(str(x) for x in items_of(self.brief.get("lookups")))
        groups = {}
        for n, it in enumerate(self.items, 1):
            for r in it.musts:
                p, c, v = r.partition(":")
                k = word_form(r)
                if c and p.strip() and v.strip() and r not in pointers and n not in groups.get(k, []):
                    groups.setdefault(k, []).append(n)
        sents = sentences(self.request)
        for k, hs in groups.items():
            if len(hs) < 2:
                continue
            first = next(r for r in self.items[hs[0] - 1].musts if word_form(r) == k)
            p, _, v = first.partition(":")
            figs = [f for f, _, _ in figures_of(v)]
            proots = roots(p)
            wanted = set()
            if not figs:
                wanted = set(fold_word(w) for w in RX_LETTER_RUN_K02.findall(v))
                wanted = set(w for w in wanted if w and not RX_FILLER.fullmatch(w))
            if not proots or (not figs and not wanted):
                continue
            stating = []
            for s_ in sents:
                said = set(fold_word(w) for w in RX_NAME_RUN.findall(s_))
                if not (proots & roots(s_)) or not wanted <= said:
                    continue
                sf = [f for f, _, _ in figures_of(s_)]
                if all(any(abs(f - g) < SLACK for g in sf) for f in figs):
                    stating.append(s_)
            if not stating:
                continue
            kept = set()
            for s_ in stating:
                named = [h for h in hs if self._names_item(s_, h)]
                if not named:
                    kept = set()
                    break
                kept |= set(named)
            if not kept or kept >= set(hs):
                continue
            near = set()
            for h in kept:
                near |= roots(self.items[h - 1].name)
            gone = [h for h in hs if h not in kept and roots(self.items[h - 1].name) & near]
            for h in gone:
                it = self.items[h - 1]
                it.musts = [r for r in it.musts if word_form(r) != k]
            if gone:
                self.trace.append({"kind": "narrowed", "need": first[:120], "kept": sorted(kept), "removed": gone})

    def _times_quantities(self):
        counts = {}
        for c, words in times_counts(self.request):
            have = roots(words)
            scores = []
            for n, it in enumerate(self.items, 1):
                nr = roots(it.name)
                if nr:
                    scores.append((len(nr & have) / float(len(nr)), n))
            if not scores:
                continue
            scores.sort(key=lambda x: -x[0])
            best = scores[0]
            if best[0] <= 0.5 or (len(scores) > 1 and scores[1][0] == best[0]):
                continue
            counts.setdefault(best[1], set()).add(c)
        for n, it in enumerate(self.items, 1):
            cs = counts.get(n)
            if cs and len(cs) == 1:
                c = next(iter(cs))
                if c != it.qty:
                    self.trace.append({"kind": "times_count", "item": n, "qty": c, "was": it.qty})
                    it.qty = c

    def _money_marks(self):
        unit = self.brief.get("budget_unit")
        unit = str(unit) if unit is not None else ""
        ur = roots(unit)
        sign = None
        located = False
        uwords = None
        if _num(self.budget):
            for f, s, e in figures_of(self.request):
                if abs(f - self.budget) <= SLACK:
                    located = True
                    pre = self.request[max(0, s - 2):s].strip()
                    if pre and is_money_sign(pre[-1]) and sign is None:
                        sign = pre[-1]
                    m = RX_AFTER_BUDGET_WORD.match(self.request, e)
                    if m and ur and roots(m.group(1)) & ur:
                        uwords = ur
        if uwords is None and ur and sign is None:
            uwords = ur
        loose = not located and sign is None and uwords is None
        return sign, uwords, loose

    def _sums(self, text, marks):
        sign, uwords, loose = marks
        out = []
        for f, s, e in figures_of(text):
            before = text[:s].rstrip()
            mark = before[-1:]
            m = RX_AFTER_SUM_WORD.match(text, e)
            money = False
            if sign is not None and mark == sign:
                money = True
            elif loose and mark and is_money_sign(mark):
                money = True
            elif uwords and m and roots(m.group(1)) & uwords:
                money = True
            if not money:
                continue
            lead = before[:-1].rstrip() if (mark and is_money_sign(mark)) else before
            if RX_FLOOR_BEFORE_SUM.search(lead):
                continue
            out.append(f)
        return out

    def _read_item_caps(self):
        caps = {}
        if len(self.items) == 1 and self.budget is None:
            self.item_caps = caps
            return
        total = self.budget if _num(self.budget) else None
        marks = self._money_marks()
        for s in sentences(self.request):
            sums = [x for x in self._sums(s, marks) if total is None or abs(x - total) > SLACK]
            if len(sums) != 1 or sums[0] <= 0:
                continue
            sr = roots(s)
            named = []
            for n, it in enumerate(self.items, 1):
                for rr in (roots(it.name), roots(it.shelf)):
                    if rr and len(rr & sr) > len(rr) / 2.0:
                        named.append(n)
                        break
            if len(named) != 1:
                continue
            n = named[0]
            caps[n] = min(caps[n], sums[0]) if n in caps else sums[0]
        self.item_caps = caps
        if caps:
            self.trace.append({"kind": "item_caps", "caps": dict(caps)})

    def item_cap(self, n, request_only=False):
        c = []
        if n in self.item_caps:
            c.append(self.item_caps[n])
        if not request_only and self.brief.get("budget_scope") == "each" and _num(self.budget):
            c.append(self.budget)
        return min(c) if c else None

    def over_cap(self, n, lid):
        cap = self.item_cap(n)
        if cap is None:
            return False
        p = self.price(lid)
        if p is None:
            return False
        return p > cap + SLACK and p <= CAP_SANITY * cap

    def described(self, n, req):
        if any(req == t for _, t in self.refs) or req in self.lookups:
            return False
        prop, colon, val = req.partition(":")
        if not colon or not prop.strip():
            return False
        if len(RX_ALNUM.findall(val.lower())) < 4:
            return False
        if RX_HAS_DIGIT.search(val):
            return False
        if len(RX_ALPHA3_K02.findall(val.lower())) < 4:
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
        unit = self.brief.get("budget_unit")
        for n, it in enumerate(self.items, 1):
            head = ("item %d: %s" % (n, it.name)) if several else ("item: %s" % it.name)
            if it.shelf:
                head += " (store label: %s)" % it.shelf
            head += ", quantity %d" % it.qty
            deciding, desc = [], []
            for r in it.live():
                txt = r
                if r in self.ref_values:
                    txt += " (" + ref_note(r, self.ref_values[r]) + ")"
                (desc if self.described(n, r) else deciding).append(txt)
            if deciding:
                head += "; requirements: " + "; ".join(deciding)
            else:
                head += "; requirements: nothing besides the product type"
            cap = self.item_cap(n, True)
            if cap is not None:
                s = ("; price: one unit may cost at most %s %s (the limit the request itself sets for this item)"
                     % (gnum(cap), unit if unit else ""))
                head += s.replace("  ", " ")
            if desc:
                head += ("; only in the shopper's phrasing (no attribute of any listing records these, so they settle "
                         "nothing): " + "; ".join(desc))
            gone = []
            for r in it.dropped:
                if r not in gone:
                    gone.append(r)
            if gone:
                head += ("; withdrawn by the shopper after the request (these no longer count, even where the request "
                         "text still states them): " + "; ".join(gone))
            lines.append(head)
        if several:
            try:
                nl = self.nested_labels_line()
            except Exception:
                nl = ""
            if nl:
                lines.append(nl)
        if self.budget is not None:
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
                when = "at the purchase; Terry carries it out when you order"
            else:
                when = "at a decline; Terry carries it out at that moment"
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


class K03(object):
    def _init_k03(self):
        self.status_count = 0
        self.status_memo = {}
        self.descent_caps = {}

    def status_filter(self, key, text):
        last_text, last_at = self.status_memo.get(key, ("", -4))
        if text == last_text and self.status_count - last_at < STATUS_REFRESH:
            return ""
        self.status_memo[key] = (text, self.status_count)
        return text

    def _line_label(self, lid, ck):
        return str((self.rows.get(lid) or {}).get(ck) or "")

    def held_lines(self):
        out = []
        for lid in self.cart_ids():
            u = self.units_held(lid)
            if u is not None and _num(u) and u > 0:
                out.append(lid)
        return out

    def short_items(self, floor=None):
        if self.checkouts:
            return []
        if floor is not None and self.steps_left() <= floor:
            return []
        held = self.held_lines()
        if not held:
            return list(range(1, len(self.items) + 1))
        ck = self.category_key()
        if not ck:
            return []
        labels = [self._line_label(l, ck) for l in held]
        return [n for n in range(1, len(self.items) + 1) if not any(self.owns_label(n, lab) for lab in labels)]

    def status(self):
        self.status_count += 1
        S, M, L = self.step, self.budget_steps, self.steps_left()
        r = self.reserve()
        out = "[Terry] step %d of %d used, %d left." % (S, M, L)
        short = self.short_items(r + 3) if L <= len(self.items) + 5 else []
        if L <= r + 1:
            out += (" FINAL STEPS: order now if every item already has a carted line; decline only when searching has "
                    "proved that some item lacks any matching listing, and a shortage of steps proves no such thing.")
        elif short:
            if len(self.items) > 1:
                names = ", ".join("item %d (%s)" % (n, self.items[n - 1].name) for n in short)
            else:
                names = self.items[0].name
            out += (" Few steps remain and %s still has no cart line: first open the most likely listing for it "
                    "that has not been opened yet, then add the fitting listing to the cart, then order." % names)
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
        pend = [t for (i, t) in self.open_refs()]
        if pend and not self.checkouts:
            out += ("\n[Settle these first: open the listing each one points at, and every listing that listing's page "
                    "refers to; the last page holds the value] " + " | ".join(pend[:3]))
        if self.leads:
            out += "\n[review candidates] " + ", ".join(self.leads)
        if self.fit_pairs and not self.checkouts:
            fp = self.fit_plan()
            if fp:
                fp = self.status_filter("fit", fp)
                if fp:
                    out += "\n" + fp
        if self.fit_pairs:
            try:
                ws = self.whole_sets_note() or ""
            except Exception as e:
                self.trace.append({"kind": "soft_crash", "method": "whole_sets_note", "error": str(e)[:200]})
                ws = ""
            ws = self.status_filter("sets", ws)
            if ws:
                out += ws
        if self.pick() == "cheapest" and not self.checkouts:
            try:
                sc = self.set_check_line() or ""
            except Exception as e:
                self.trace.append({"kind": "soft_crash", "method": "set_check_line", "error": str(e)[:200]})
                sc = ""
            if sc:
                sc = self.status_filter("setcheck", sc)
                if sc:
                    out += "\n" + sc
        notes = self.condition_notes()
        if notes:
            t = ("\n[conditions between items, judged from the carted lines] " + " | ".join(notes))
            t = self.status_filter("cond", t)
            if t:
                out += t
        if self.change_notes:
            out += "\n[changes so far] " + " | ".join(self.change_notes[-4:])
        out += "\n[ledger]\n" + self.ledger()
        return out

    @soft(list)
    def condition_notes(self):
        notes = []
        ck = self.category_key()
        held = self.held_lines()
        for n, it in enumerate(self.items, 1):
            for r in it.live():
                if len(notes) >= COND_NOTES:
                    return notes
                prop, colon, val = r.partition(":")
                prop = prop.strip()
                if not colon or not prop:
                    continue
                m = RX_COND.search(val)
                if not m or RX_POINTER_TALK.search(val):
                    continue
                rule = val[:m.start()].strip(" \t,;:.")
                cond = val[m.end():].strip(" \t,;:.")
                waiver = bool(RX_WAIVER.match(m.group(0)))
                cr = roots(cond)
                cands = []
                for j, o in enumerate(self.items, 1):
                    if j != n:
                        cands.append((len(cr & (roots(o.name) | roots(o.shelf))), j))
                if not cands:
                    continue
                cands.sort(reverse=True)
                if cands[0][0] == 0 or (len(cands) > 1 and cands[1][0] == cands[0][0]):
                    continue
                om = cands[0][1]
                other = self.items[om - 1]
                head = "item %d '%s'" % (n, r.strip())
                if held and not ck:
                    continue
                lines = [l for l in held if self.owns_label(om, self._line_label(l, ck))] if held else []
                if not lines:
                    notes.append("%s: this is settled once item %d (%s) has been chosen" % (head, om, other.name))
                    continue
                line = lines[0]
                vm = RX_COND_VERB.search(cond)
                subj, allowed = (cond[:vm.start()], cond[vm.end():]) if vm else ("", cond)
                neg = False
                nm = RX_NOT_LEAD_K03.match(allowed)
                if nm:
                    neg = True
                    allowed = allowed[nm.end():]
                vals = []
                for piece in RX_VALUE_SPLIT_K03.split(allowed):
                    p = RX_LEAD_FILL_K03.sub("", piece.strip()).strip()
                    if p and RX_HAS_ALNUM_K03.search(p):
                        vals.append(p)
                block = attrs_of(self.full.get(line)) if line in self.full else None
                if not vals or not block:
                    notes.append("%s: verify on %s (the carted %s) whether %s holds" % (head, line, other.name, cond))
                    continue
                sr = roots(RX_POSSESSIVE_K03.sub("", subj)) - (roots(other.name) | roots(other.shelf))
                keys = [k for k in block if roots(k) & sr] if sr else []

                def has_val(k):
                    sv = spelled(str(block[k]))
                    return any(spelled(v) and spelled(v) <= sv for v in vals)
                if not keys:
                    keys = [k for k in block if has_val(k)]
                heldk = [k for k in keys if has_val(k)]
                holds = bool(heldk)
                if neg:
                    holds = not holds
                showk = (heldk or keys)[:2]
                shown = ", ".join("%s=%s" % (k, str(block[k])[:40]) for k in showk) or "no attribute covers it"
                state = "met" if holds else "not met"
                binds = (holds and not waiver) or (not holds and waiver)
                if binds:
                    note = ("%s: on %s (%s) the condition is %s, hence the rule applies: item %d must have %s: %s"
                            % (head, line, shown, state, n, prop, rule))
                    if waiver:
                        sw = [w for w in RX_NAME_RUN.findall(RX_POSSESSIVE_K03.sub("", subj)) if single_root(w) in sr]
                        st = ("whose " + " ".join(sw) + " is") if sw else "with"
                        note += " (a %s listing %s %s%s would lift the rule)" % (other.name, st, "not " if neg else "",
                                                                                 " or ".join(vals))
                else:
                    note = ("%s: on %s (%s) the condition is %s, hence for now this rule stays inactive"
                            % (head, line, shown, state))
                notes.append(note)
        return notes

    def _closeness(self, label, n):
        it = self.items[n - 1]
        lw = word_form(label)
        if self.adopted.get(str(label)) == n or lw == word_form(it.shelf):
            return 3.0
        if lw == word_form(it.name):
            return 2.0
        lr = roots(label)
        ir = roots(it.shelf) | roots(it.name)
        return len(lr & ir) / float(max(1, len(lr | ir)))

    def _close_groups(self, pairs):
        best = {}
        sc = {}
        for lab, ids in pairs:
            if not lab:
                continue
            o = self.label_owner(lab)
            if not o:
                continue
            s = self._closeness(lab, o)
            sc[lab] = (o, s)
            if s > best.get(o, -1):
                best[o] = s
        kept = []
        out = []
        for lab, ids in pairs:
            if lab in sc:
                o, s = sc[lab]
                if best[o] >= LOOSE_FLOOR and s < LOOSE_SHARE * best[o]:
                    out.append(lab)
                    continue
            kept.append((lab, ids))
        if out:
            self.trace.append({"kind": "loose_labels", "labels": out[:4]})
        return kept

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
        matched = [l for l in order if self.label_matches(l)]
        if matched:
            kept = self._close_groups([(l, groups[l]) for l in matched])
            pool = [l for l, _ in kept]
        else:
            pool = order
        pool = sorted(pool, key=lambda l: -len(groups[l]))
        return [(l, groups[l]) for l in pool[:max(1, len(self.items))]]

    def price_value(self, v):
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

    def own_price(self, rid):
        for rec in (self.rows.get(rid), self.full.get(rid)):
            if not isinstance(rec, dict):
                continue
            ks = [k for k in rec if RX_OWN_PRICE_FIELD.search(str(k))]
            ks = [k for k in ks if RX_PRICE_EXACT_K03.fullmatch(under_form(k))] + [k for k in ks if not RX_PRICE_EXACT_K03.fullmatch(under_form(k))]
            for k in ks:
                p = self.price_value(rec[k])
                if p is not None:
                    return p
        return None

    def price(self, rid):
        if not self.price_key:
            return self.own_price(rid)
        return self.price_value((self.rows.get(rid) or {}).get(self.price_key))

    @soft(None)
    def _guess_next_arg(self, tool, token):
        sch = self.schemas.get(tool) or {}
        props = sch.get("properties")
        props = props if isinstance(props, dict) else {}
        req = sch.get("required")
        req = set(x for x in req if isinstance(x, str)) if isinstance(req, list) else set()
        last = self.last_args.get(tool) or {}
        whole = False
        cands = []
        for k, p in props.items():
            if k in req or not isinstance(p, dict) or p.get("enum"):
                continue
            ks = kinds(p)
            if isinstance(token, bool):
                ok = "boolean" in ks
            elif _num(token):
                whole = isinstance(token, int)
                ok = "number" in ks or (whole and "integer" in ks)
                lo, hi = p.get("minimum"), p.get("maximum")
                if ok and _num(lo) and token < lo:
                    ok = False
                if ok and _num(hi) and token > hi:
                    ok = False
            elif isinstance(token, str):
                ok = "string" in ks
                ml, mn = p.get("maxLength"), p.get("minLength")
                if ok and _num(ml) and len(token) > ml:
                    ok = False
                if ok and _num(mn) and len(token) < mn:
                    ok = False
            else:
                ok = False
            if ok:
                cands.append(k)
        cands = [k for k in cands if not (isinstance(last.get(k), str) and last.get(k).strip())
                 and not RX_QUERY_ARG.search(under_form(k))]
        if isinstance(token, int) and not isinstance(token, bool) and len(cands) > 1:
            ints = [k for k in cands if "integer" in kinds(props[k])]
            if ints:
                cands = ints
        return cands[0] if len(cands) == 1 else None

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
        p = (SHAPE_PROMPT + "\n\nTOOL: %s | parameters: %s\n\nRESULT (first rows only):\n%s"
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
            g = self._guess_next_arg(tool, obs.get(sh["next"]))
            if g:
                sh["next_arg"] = g
                self.trace.append({"kind": "next_arg_guess", "tool": tool, "arg": g})
        self.shapes[tool] = sh
        if self.price_key is None and sh["price"]:
            self.price_key = sh["price"]
        if self.stock_key is None and sh["stock"]:
            self.stock_key = sh["stock"]
        self.trace.append({"kind": "shape", "tool": tool, "shape": sh})
        return sh

    def page_arg_of(self, tool):
        if tool in self.page_arg:
            return self.page_arg[tool]
        sh = self.shapes.get(tool)
        if sh and sh.get("next_arg"):
            p = self._props(tool).get(sh["next_arg"])
            if isinstance(p, dict) and "integer" in kinds(p):
                return sh["next_arg"]
        return None

    def first_page_value(self, tool, arg):
        p = self._props(tool).get(arg)
        p = p if isinstance(p, dict) else {}
        mn = p.get("minimum")
        if _whole(mn):
            return mn
        d = p.get("default")
        if _whole(d) and 0 <= d <= 1:
            return d
        if RX_OFFSET_SEG.search(under_form(arg)):
            return 0
        return 1

    def _later_page(self, args, plan):
        sent = args.get(plan["arg"])
        if isinstance(sent, int) and not isinstance(sent, bool):
            return sent not in (0, 1)
        return bool(str(sent or "").strip())

    def page_loop(self, tool, args, arg, nxt, count):
        lines = []
        got = 0
        for _ in range(count):
            a2 = dict(args)
            a2[arg] = nxt
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
            lines.append("%s=%s: %s" % (arg, dump(nxt), rend))
            p2 = self.paging_plan(tool, o2)
            nxt = p2["next"] if p2 else None
            if nxt is None:
                break
        self.trace.append({"kind": "auto_pages", "pages": got})
        return lines, nxt, got

    def auto_page(self, tool, args, obs):
        self.paged = []
        pk = self.pick()
        plan = self.paging_plan(tool, obs) if pk != "any" else None
        if not plan:
            if pk == "any":
                t = self.chain_page(tool, args, obs)
                if t:
                    return t
                return self.big_next_page(tool, args, obs) or ""
            return ""
        if plan["next"] is None or self._later_page(args, plan) or plan["total"] <= plan["shown"]:
            return ""
        if pk == "cheapest" and self.price_sorted(obs):
            return ""
        need = -(-(plan["total"] - plan["shown"]) // plan["shown"])
        L, r = self.steps_left(), self.reserve()
        if need > 2 and self.pages_read + 2 <= AUTO_PAGES and L - r - 2 >= 12:
            need = 2
        spare = ONE_PAGE_ROOM if need == 1 else 12
        if need > 2 or self.pages_read + need > AUTO_PAGES or L - r - need < spare:
            return ("[Terry: this search matched %d listings but displayed only %d, and not sorted by price. The pick "
                    "weighs every listing that fits: search again with tighter words, namely the value the requirement "
                    "names, or request the next page by sending %s set to %s.]"
                    % (plan["total"], plan["shown"], plan["arg"], dump(plan["next"])))
        lines, nxt, got = self.page_loop(tool, args, plan["arg"], plan["next"], need)
        if not got:
            return ""
        out = "[Terry fetched the following pages of this search by itself, spending one step per page]\n" + "\n".join(lines)
        if nxt is not None:
            out += ("\n[Terry: the search matched %d listings and Terry has now read %d rows. Make the search tighter, "
                    "using the value the requirement names, or keep requesting pages, so that the remaining listings come "
                    "into view.]" % (plan["total"], plan["shown"] * (got + 1)))
        return out

    @soft("")
    def big_next_page(self, tool, args, obs):
        if not self.clock_bound() or self.steps_left() - self.reserve() < BIG_ROOM or self.finished:
            return ""
        short = set(self.short_items())
        if not short:
            return ""
        ck = self.category_key()
        hit = False
        for lid, rec in self.obs_listings(obs):
            its = self.listing_items(lid, ck)
            if short & set(its):
                hit = True
                break
        if not hit:
            return ""
        plan = self.paging_plan(tool, obs)
        if not plan or plan["next"] is None or self._later_page(args, plan) or plan["total"] <= plan["shown"]:
            return ""
        lines, nxt, got = self.page_loop(tool, args, plan["arg"], plan["next"], 1)
        if not lines:
            return ""
        self.trace.append({"kind": "big_next_page", "tool": tool})
        return ("\n[Terry fetched this search's following page on its own, at the cost of one step; the search matched %d "
                "listings and displayed %d]\n" % (plan["total"], plan["shown"]) + lines[0])

    def open_refs(self):
        live = self.live_all()
        return [(i, t) for (i, t) in self.refs if t in live and t not in self.ref_values]

    def shop_values(self):
        bad = self._noname_keys()
        for k in (self.price_key, self.stock_key):
            if k:
                bad.add(k)
        cnt = {}
        for rid in self.seen:
            row = self.rows.get(rid) or {}
            for k, v in row.items():
                if k in bad or not isinstance(v, str):
                    continue
                s = v.strip()
                if (3 <= len(s) <= 40 and not RX_PLACEHOLDER.match(s) and s[:1].isupper() and not RX_HAS_DIGIT.search(s)
                        and len(s.split()) <= 4):
                    cnt[s] = cnt.get(s, 0) + 1
        return cnt

    def chain_shops(self):
        opens = self.open_refs()
        if not opens:
            return []
        longs = []
        for rid in sorted(self.opened):
            rec = self.full.get(rid) or self.rows.get(rid)
            if rec is None:
                continue
            for leaf in record_leaves(rec):
                if len(leaf.split()) >= CHAIN_LONG_WORDS:
                    longs.append(leaf)
        ranked = sorted(self.shop_values().items(), key=lambda kv: (-kv[1], kv[0]))
        found = []
        for texts in ([t for _, t in opens], longs):
            for v, _ in ranked:
                if len(found) >= CHAIN_SHOPS:
                    return found
                if v in found:
                    continue
                rx = re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])")
                if any(rx.search(t) for t in texts):
                    found.append(v)
        return found

    @soft("")
    def chain_page(self, tool, args, obs):
        if not self.open_refs():
            return ""
        said = " ".join(v for v in args.values() if isinstance(v, str)).lower()
        shops = [s for s in self.chain_shops()
                 if re.search(r"(?<![a-z0-9])" + re.escape(s.lower()) + r"(?![a-z0-9])", said)]
        if not shops:
            return ""
        plan = self.paging_plan(tool, obs)
        if not plan or plan["next"] is None or self._later_page(args, plan) or plan["total"] <= plan["shown"]:
            return ""
        need = -(-(plan["total"] - plan["shown"]) // plan["shown"])
        L, r = self.steps_left(), self.reserve()
        if need > CHAIN_PAGES or self.pages_read + need > AUTO_PAGES or L - r - need < 12:
            return ""
        lines, nxt, got = self.page_loop(tool, args, plan["arg"], plan["next"], need)
        self.trace.append({"kind": "chain_pages", "shops": shops, "pages": got})
        if not got:
            return ""
        return ("\n[Terry fetched the following pages of this search by itself, spending one step per page, because the "
                "search names a shop that a reference leads to]\n" + "\n".join(lines))

    def show_links(self, ids, reserve):
        vt = self.value_template()
        if not vt or not ids:
            return
        for i in ids:
            fi = self.fit_index()
            code = (fi.get(i) or {}).get("pointer")
            if code and code not in self.link_tried:
                owners = [lid for lid, m in fi.items() if m.get("own") == code]
                if owners and not any(self.read_full(o) for o in owners):
                    if self.steps_left() <= reserve or self.finished or self.rushed():
                        return
                    self.link_tried.add(code)
                    self.fill_facts([owners[0]], reserve)
                    self.trace.append({"kind": "link_open", "code": code, "id": owners[0]})
                    continue
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

    def plausible_rows(self, line, label, bound):
        own = self.label_owner(label) if label is not None and str(label).strip() else None
        if not own:
            return []
        toks = set()
        for v, _ in self.value_words(own):
            toks |= set(value_tokens(v))
        ck = self.category_key()
        if not toks or not ck:
            return []
        key = str(label).strip().lower()
        out = []
        for rid in self.seen:
            if rid == line:
                continue
            p = self.price(rid)
            if p is None or p >= bound:
                continue
            row = self.rows.get(rid) or {}
            if str(row.get(ck, "")).strip().lower() != key or not self.buyable(rid):
                continue
            if toks <= text_words(dump(row)):
                out.append(rid)
        out.sort(key=lambda r: (self.price(r), r))
        return out

    def walk_below(self, lines):
        tpl = self.capped_template()
        ck = self.category_key()
        if not tpl or not ck:
            return
        for lid in lines:
            p = self.price(lid)
            lab = (self.rows.get(lid) or {}).get(ck)
            if p is None or p < 2 or not lab or (lid, p) in self.capped:
                continue
            self.capped.add((lid, p))
            self._descent(lid, p, lab, tpl)

    def _descent(self, lid, p, lab, tpl):
        tool, opts, shelf, capa = tpl
        bound = p
        pl = self.plausible_rows(lid, lab, p)
        if pl:
            bound = self.price(pl[0]) + 1
        tried = set()
        for rnd in range(DESCENT_ROUNDS):
            cap = int(bound) - 1 if bound == int(bound) else int(bound)
            if (cap < 1 or cap in tried or self.steps_left() <= self.reserve() + (7 if rnd == 0 else WALK_RESERVE)
                    or self.finished or self.rushed()):
                return
            tried.add(cap)
            a = dict(opts)
            a[shelf] = lab
            a[capa] = cap
            faults = []
            a = shape(a, self.schemas.get(tool), "args", faults)
            if faults:
                return
            bp = self.plausible_rows(lid, lab, cap + 1)
            before = self.price(bp[0]) if bp else None
            pages = 0
            while True:
                res = self.env(tool, a)
                if res is None:
                    return
                obs = res.get("observation")
                self.observe(tool, obs)
                pages += 1
                recs = self.obs_listings(obs)
                plan = self.paging_plan(tool, obs)
                pl = self.plausible_rows(lid, lab, cap + 1)
                cheaper = bool(pl) and (before is None or self.price(pl[0]) < before)
                whole = bool(plan) and plan["total"] <= DESCENT_WHOLE
                if (not recs or (cheaper and not whole) or not plan or plan["next"] is None
                        or (not whole and pages >= DESCENT_PAGES) or self.price_sorted(obs)):
                    break
                if self.steps_left() <= self.reserve() + 7 or self.finished or self.rushed():
                    break
                a2 = dict(a)
                a2[plan["arg"]] = plan["next"]
                f2 = []
                a2 = shape(a2, self.schemas.get(tool), "args", f2)
                if f2:
                    break
                a = a2
            pl = self.plausible_rows(lid, lab, cap + 1)
            self.trace.append({"kind": "descent", "line": lid, "cap": cap, "pages": pages, "plausible": pl[:3]})
            if not pl or (before is not None and not self.price(pl[0]) < before):
                return
            bound = self.price(pl[0])

    def line_owner(self, lab):
        if len(self.items) == 1:
            return 1
        its = [n for n in range(1, len(self.items) + 1) if self.owns_label(n, lab)]
        return its[0] if len(its) == 1 else 0

    def cheaper_rows(self, extra=False):
        if self.pick() != "cheapest":
            return []
        mk = self.cart_mark() + "#" + str(self.version)
        ck = self.category_key()
        if not ck:
            return []
        if mk in self.cheaper_marks and not extra:
            return []
        self.cheaper_marks.add(mk)
        lines = self.cart_ids()
        if not extra:
            self.walk_below(lines)
        room = self.batch_room() if self.compare_facts() is True else 0
        picks = []
        for lid in lines:
            p = self.price(lid)
            lab = str((self.rows.get(lid) or {}).get(ck) or "")
            if p is None or not lab.strip():
                continue
            pool = sorted(self._below_pool(lid, p, lab, lines), key=lambda r: self.price(r))
            own = self.line_owner(lab)
            brk = [r for r in pool if self.row_breaks_limit(r, own)]
            pool = [r for r in pool if r not in brk] + brk
            lead = None
            for r in pool:
                if r in self.value_rows and r not in brk:
                    lead = r
                    break
            first = [r for r in self.plausible_rows(lid, lab, p) if r in pool and r not in brk]
            if lead and lead not in first:
                first.append(lead)
            pool = first + [r for r in pool if r not in first]
            for r in pool[:room or 2]:
                if r not in picks:
                    picks.append(r)
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
            lab = str((self.rows.get(lid) or {}).get(ck) or "")
            if p is None or not lab.strip():
                continue
            pool = sorted(self._below_pool(lid, p, lab, lines), key=lambda r: self.price(r))
            own = self.line_owner(lab)
            brk = [r for r in pool if self.row_breaks_limit(r, own)]
            pool = ([r for r in pool if r not in brk] + brk)[:CHEAPER_SHOWN]
            ents += ["%s at %s" % (r, gnum(self.price(r))) for r in pool]
        if not ents:
            return ""
        return "Not read yet and priced below a line of the cart: " + ", ".join(ents) + "."

    def cheaper_hunt(self):
        mark = dump([it.live() for it in self.items]) + "|" + str(self.budget) + "|" + dump(sorted(self.gone))
        late = self.late_drop and self.pick() == "cheapest"
        if self.pick() != "cheapest" or mark == self.hunt_mark:
            return ""
        if self.hunts >= CHEAPER_ROUNDS and not (late and self.hunts < CHEAPER_ROUNDS + 1):
            return ""
        vt = self.value_template()
        if not vt or not self.price_key or self.clock() > (HUNT_DEADLINE + HUNT_GRACE if late else HUNT_DEADLINE):
            return ""
        room = min(HUNT_STEP_CAP, self.steps_left() - self.reserve() - HUNT_RESERVE - 2 * len(self.items))
        if room < 3 and late:
            room = min(LATE_HUNT_SEARCHES, self.steps_left() - self.reserve() - 2 - 2 * len(self.items))
            if room >= 1:
                self.trace.append({"kind": "late_hunt_room", "room": room})
        if room < 3 and not (late and room >= 1):
            return ""
        if late:
            self.late_drop = False
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
        return ("Terry itself ran %d extra searches chasing a lower price, one step each, and they brought up %d new rows."
                % (spent, gain))


class K04(object):
    def hunt_sets(self, n, vals):
        ptr = set(t for _, t in self.refs)
        listed = None
        choices = []
        for m, r in self.live_pairs():
            if m != n or ":" not in r:
                continue
            if r in ptr or is_conditional(r):
                continue
            v = r.split(":", 1)[1]
            if RX_NEGATION.search(v):
                continue
            if RX_OPEN_EXCLUDE.search(RX_OR_EITHER_K04.sub(" ", v)):
                continue
            lim = req_limit(v)
            if lim and lim["kind"] != "exact":
                continue
            parts = [p.strip(" .") for p in RX_CHOICE_SEP_K04.split(v)]
            parts = [p for p in parts if RX_HAS_ALNUM_K04.search(p)]
            if len(parts) < 2:
                continue
            listed = v.strip()
            choices = [p for p in parts if not find_codes(p) and len(p) <= HUNT_VALUE_LEN][:CHOICE_SETS]
            break
        base = [v for v in vals if (not find_codes(v) or RX_SPACED_NUM_K04.fullmatch(v.strip())) and v.strip() != listed]
        sets = []
        if base:
            sets.append(list(base))
        for i, v in enumerate(base):
            m = RX_SPACED_NUM_K04.fullmatch(v.strip())
            if m:
                s = list(base)
                s[i] = m.group(1)
                sets.append(s)
                break
        for c in choices:
            sets.append(base + [c])
        return base, [s for s in sets if s]

    def title_singular(self, n, noun):
        if not noun:
            return ""
        tk = self.title_key()
        ck = self.category_key()
        if not tk or not ck:
            return ""
        noun = noun.lower()
        cnt = {}
        for rid in self.seen:
            row = self.rows.get(rid) or {}
            lab = row.get(ck)
            if lab is None or not self.owns_label(n, lab):
                continue
            for w in set(RX_TITLE_WORD_K04.findall(str(row.get(tk, "")).lower())):
                if w != noun and len(noun) - 3 <= len(w) < len(noun) and (noun.startswith(w) or same_stem(w, noun)):
                    cnt[w] = cnt.get(w, 0) + 1
        best = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
        return best[0][0] if best and best[0][1] >= SINGULAR_ROWS else ""

    def item_hunt(self, n, vt, room):
        tool, fixed, field = vt
        sch = self.schemas.get(tool)
        tset = self.hunt_tried.setdefault(n, set())
        noun, _ = self.hunt_words(n)
        vals = [v for v, _ in self.value_words(n)]
        base, sets = self.hunt_sets(n, vals)
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
            holds = bool(base) and all(v.lower() in q.lower() for v in base)
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
                    st["spent"] += 1
                    if res is None:
                        break
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
                if not plan or plan["next"] is None or (new < HUNT_PAGE_YIELD and plan["total"] > HUNT_WHOLE):
                    break
                a2 = dict(a)
                a2[plan["arg"]] = plan["next"]
                f2 = []
                a2 = shape(a2, sch, "args", f2)
                if f2:
                    break
                a = a2
            return True, found
        key = (n, tuple(base))
        singular = "" if key in self.hunt_stems else self.title_singular(n, noun)
        bare = set(x for x in (noun, singular) if x)

        def wider(ladder, pos):
            for t in ladder[pos + 1:]:
                if st["spent"] >= room or st["refused"] or self.finished or t in bare or t.lower() in tset:
                    continue
                self.trace.append({"kind": "wider_rung", "item": n, "query": t})
                run(t)
        if key not in self.hunt_stems:
            stem = ""
            for s in sets:
                ladder = []
                rungs = [" ".join(s + [noun]), " ".join(s + [singular]) if singular else "", " ".join(s),
                         " ".join(s[:-1] + [noun]) if len(s) >= 2 else ""]
                for t in rungs:
                    t = t.strip()
                    if t and t not in ladder:
                        ladder.append(t)
                for pos, t in enumerate(ladder):
                    issued, found = run(t)
                    if not issued:
                        return st["spent"], st["gain"]
                    if found:
                        if not stem:
                            stem = t
                        wider(ladder, pos)
                        break
            if not stem and singular:
                issued, found = run(singular)
                if not issued:
                    return st["spent"], st["gain"]
                if found:
                    stem = singular
            if not stem and noun:
                issued, found = run(noun)
                if not issued:
                    return st["spent"], st["gain"]
                if found:
                    stem = noun
            self.hunt_stems[key] = stem
            self.trace.append({"kind": "hunt_sets", "item": n, "sets": sets, "stem": stem})
        stem = self.hunt_stems[key]
        if base and st["spent"] < room and not st["refused"] and not self.finished:
            fl = [spacing_flip(v) for v in base]
            q = " ".join(fl + [noun]).strip()
            if any(x != y for x, y in zip(fl, base)) and q.lower() not in tset:
                self.trace.append({"kind": "flipped_hunt", "item": n, "query": q})
                run(q)
        while (st["spent"] < room and st["idle"] < HUNT_IDLE and not st["refused"] and not self.finished
               and self.clock() <= HUNT_DEADLINE + HUNT_GRACE):
            _, words = self.hunt_words(n)
            sw = set(stem.lower().split())
            words = [w for w in words if (w + " " + stem).strip().lower() not in tset and w not in sw]
            if not words:
                break
            run((words[0] + " " + stem).strip())
        return st["spent"], st["gain"]

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
            for i in self.cart_ids():
                if (self.units_held(i) or 0) > 0:
                    covered.add(str((self.rows.get(i) or {}).get(ck) or ""))
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
        picks = []
        for n in hunt_items:
            toks = set(wanted[n])
            pool = []
            for rid in self.seen:
                if rid in self.full or not self.buyable(rid):
                    continue
                if ck and not self.owns_label(n, str(self.rows[rid].get(ck) or "")):
                    continue
                if _num(self.budget) and (self.price(rid) or 0) > self.budget:
                    continue
                if self.over_cap(n, rid):
                    continue
                pool.append(rid)

            def sc(rid):
                tw = text_words(dump(self.rows[rid]))
                p = self.price(rid)
                return (-sum(1 for t in toks if t in tw), p if p is not None else float("inf"))
            picks += sorted(pool, key=sc)[:FIT_HUNT_OPENS]
        texts = []
        for i in range(0, len(picks), FIT_HUNT_OPENS):
            if self.clock() > HUNT_DEADLINE + HUNT_GRACE:
                break
            texts += self.fill_facts(picks[i:i + FIT_HUNT_OPENS], res_)
        self.trace.append({"kind": "fit_hunt", "pages": walked, "searches": total, "opened": len(texts)})
        return texts

    @soft("")
    def top_rows(self, tool, args, obs):
        if not (isinstance(obs, dict) and self.obs_listings(obs)) or self.steps_left() <= self.reserve() + 4 or self.finished:
            return ""
        props = self._props(tool)
        cands = [k for k, v in args.items() if _whole(v) and k in props and v > self.first_page_value(tool, k)
                 and not RX_PRICE_NAME.search(under_form(k))]
        if not cands:
            return ""
        if tool not in self.shapes:
            self.read_shape(tool, obs)
        pa = self.page_arg_of(tool)
        if not pa or pa not in cands:
            return ""
        fv = self.first_page_value(tool, pa)
        a = dict(args)
        a[pa] = fv
        f = []
        a = shape(a, self.schemas.get(tool), "args", f)
        if f or self.canon(tool, a) in self.remembered:
            return ""
        res = self.env(tool, a)
        if res is None:
            return ""
        o = res.get("observation")
        self.observe(tool, o)
        self.paged.append(o)
        rend = self.render(o)
        self.read_cache[mark_of(tool, a)] = (self.version, self.step, rend)
        self.trace.append({"kind": "top_rows", "tool": tool, "sent": fv, "rows": len(self.obs_listings(o))})
        return ("\n[Terry] Your fresh search began at %s=%s, beyond the opening rows, so Terry also fetched its opening "
                "rows with %s=%s, which took one step:\n%s" % (pa, dump(args.get(pa)), pa, dump(fv), rend))

    def seen_offsets(self, tool, args, pa):
        sh = self.shapes.get(tool)
        if not sh or not sh.get("rows") or not sh.get("next"):
            return None, False, False
        ident = self.canon(tool, dict((k, v) for k, v in args.items() if k != pa))
        best = None
        ended = False
        counting = False
        for cm, mem in list(self.remembered.items()):
            t, a, o = mem
            if t != tool:
                continue
            if self.canon(tool, dict((k, v) for k, v in a.items() if k != pa)) != ident:
                continue
            if not isinstance(o, dict) or sh["next"] not in o:
                continue
            nx = o.get(sh["next"])
            if nx is None:
                ended = True
                continue
            if not _whole(nx):
                continue
            sent = a.get(pa)
            start = sent if _whole(sent) else self.first_page_value(tool, pa)
            rows = o.get(sh["rows"])
            if isinstance(rows, list) and len(rows) >= 2 and nx - start == len(rows):
                counting = True
            if best is None or nx > best:
                best = nx
        return best, ended, counting

    @soft(lambda: None)
    def offset_skip(self, tool, args):
        pa = self.page_arg_of(tool)
        if not pa:
            return None
        v = args.get(pa)
        if not _whole(v) or v <= 0:
            return None
        best, ended, counting = self.seen_offsets(tool, args, pa)
        if best is None or ended or not counting or v >= best:
            return None
        p = self._props(tool).get(pa)
        p = p if isinstance(p, dict) else {}
        if _num(p.get("maximum")) and best > p["maximum"]:
            return None
        a = dict(args)
        a[pa] = best
        f = []
        a = shape(a, self.schemas.get(tool), "args", f)
        if f:
            return None
        self.trace.append({"kind": "offset_skip", "sent": v, "moved": best})
        note = ("\n[Terry] %s gives the index of the first row displayed (an offset), not a page number; the rows before %s "
                "were already displayed by this same search, so Terry sent %s=%s (you sent %s)." % (pa, best, pa, best, v))
        return a, note

    def first_look(self, tool, args, fresh):
        k = self.page_arg_of(tool)
        if not k or not fresh:
            return args
        v = args.get(k)
        if not _whole(v):
            return args
        fv = self.first_page_value(tool, k)
        if v == fv:
            return args
        a = dict(args)
        a[k] = fv
        return a

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
        extra = ""
        if self.pick() == "any":
            extra = self.chain_page(tool, a, o2) or ""
        return ("\n[Terry] No listing carries all of those words; the identical search with the short words '%s' "
                "gives:\n%s" % (short, self.render(o2))) + extra

    def remember_opening(self, n, tool, args, obs):
        rows = self.own_rows.setdefault(n, [])
        for lid, _ in self.obs_listings(obs):
            if lid not in rows:
                rows.append(lid)
        cm = self.canon(tool, args)
        if (n, cm) not in self.own_searches:
            self.own_searches.append((n, cm))

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
            if len(w) <= 3:
                noun = it.name
            else:
                parts = RX_JOINER.split(it.name.strip(), 1)
                head = parts[0].split() if len(parts) > 1 else []
                if 1 <= len(head) <= 3:
                    noun = " ".join(head)
                elif len(head) > 3:
                    noun = " ".join(head[-2:])
                else:
                    noun = " ".join(w[-2:])
            tries = []
            if vals:
                tries.append(" ".join(vals[:2] + [noun]))
            if noun not in tries:
                tries.append(noun)
            singles = []
            if len(vals) >= 2:
                for v in vals[:2]:
                    s = (v + " " + noun).strip()
                    if s not in singles:
                        singles.append(s)
            i = 0
            while i < len(tries):
                t = tries[i]
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
                    if vals and t != noun:
                        self.remember_opening(n, tool, a, obs)
                    if singles:
                        tries = tries[:i + 1] + [x for x in tries[i + 1:] if x not in singles or x == noun]
                    self.record_search(a)
                    lines.append("%s: %s" % (t, self.render(obs)))
                    plan = self.paging_plan(tool, obs)
                    if (plan and plan["next"] is not None and self.item_rows(n) < OPEN_ROWS_PER_ITEM
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
                    if self.item_rows(n) >= OPEN_ROWS_PER_ITEM:
                        break
                else:
                    lines.append("%s: %s" % (t, ("the store rejected the call: " + self.render(obs)) if is_failure(obs) else "no rows"))
                    if (singles and i == 0 and vals and not is_failure(obs)
                            and not (isinstance(obs, dict) and obs.get("error"))):
                        ins = [s for s in singles if s not in tries]
                        if ins:
                            tries = tries[:i + 1] + ins + tries[i + 1:]
                            self.trace.append({"kind": "single_openers", "item": n, "tries": ins})
                i += 1
        if self.pick() == "cheapest" or self.clock_bound():
            lines.extend(self.front_paging(stop) or [])
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
        sw = self.series_sweep()
        if sw:
            lines.append(sw)
        self.gap = STORE_PAUSE
        self.trace.append({"kind": "opening", "rows": len(self.rows), "opened": len(self.full)})
        for rid in list(self.full.keys()):
            lines.append("read in full: %s" % self.page_text(rid))
        ev = self.evidence(list(self.full.keys()))
        if ev:
            lines.append(ev)
        return "\n".join(lines)

    @soft(list)
    def front_paging(self, stop):
        big = self.clock_bound()
        ck = self.category_key()
        if self.front_paged or not self.own_searches or not ck or not (big or self.pick() == "cheapest"):
            return []
        vt = self.value_template()
        if not vt:
            return []
        tool0, _, field = vt
        cands = []
        for n, cm in self.own_searches:
            mem = self.remembered.get(cm)
            if not mem:
                continue
            t, a, o = mem
            if t != tool0 or a is None or o is None:
                continue
            plan = self.paging_plan(t, o)
            if not plan or plan["next"] is None:
                continue
            if not plan["shown"] or plan["total"] <= plan["shown"] or plan["total"] > FRONT_SPAN * plan["shown"]:
                continue
            if self.pick() == "cheapest" and self.price_sorted(o):
                continue
            recs = self.obs_listings(o)
            own = sum(1 for lid, rec in recs if isinstance(rec, dict) and self.owns_label(n, rec.get(ck)))
            if recs and own * 2 > len(recs):
                cands.append((n, t, a, o, plan))
        if not cands:
            return []
        self.front_paged = True
        k = max(FRONT_SEARCHES, len(self.items)) if big else FRONT_SEARCHES
        cands = sorted(cands, key=lambda c: -c[4]["total"])[:k]
        lines = []
        pages = 0
        for n, t, a, o, plan in cands:
            walked = set()
            cur_a, cur_o = a, o
            for _ in range(FRONT_PAGES):
                if stop - self.step < FRONT_ROOM or self.finished:
                    break
                p = self.paging_plan(t, cur_o)
                if not p or p["next"] is None:
                    break
                a2 = dict(cur_a)
                a2[p["arg"]] = p["next"]
                f = []
                a2 = shape(a2, self.schemas.get(t), "args", f)
                if f:
                    break
                cm2 = self.canon(t, a2)
                if cm2 in walked:
                    break
                walked.add(cm2)
                if cm2 in self.remembered:
                    o2 = self.remembered[cm2][2]
                else:
                    res = self.env(t, a2)
                    if res is None:
                        break
                    o2 = res.get("observation")
                    self.observe(t, o2)
                    pages += 1
                    lines.append("%s (%s=%s): %s" % (a2.get(field), p["arg"], dump(p["next"]), self.render(o2)))
                rows = self.own_rows.setdefault(n, [])
                for lid, _ in self.obs_listings(o2):
                    if lid not in rows:
                        rows.append(lid)
                cur_a, cur_o = a2, o2
        self.trace.append({"kind": "front_paging", "searches": len(cands), "pages": pages})
        return lines

    @soft("")
    def series_sweep(self):
        vt = self.value_template()
        ck = self.category_key()
        if (not self.fit_pairs or not ck or not vt or not self.clock_bound()
                or self.steps_left() - self.reserve() < BIG_ROOM or self.finished):
            return ""
        tool = vt[0]
        notes = []
        for a, b in self.fit_pairs:
            sysm = self.series_system(a, b)
            if not sysm or sysm[0] != a:
                continue
            fam = sysm[2]
            proven = [r for r in self.proven_listings(a) if self.buyable(r)]
            if not (1 <= len(proven) <= 2):
                continue
            idx = self.fit_index()
            carried = set()
            for r in proven:
                carried |= set((idx.get(r) or {}).get("own_series") or [])
            wanted = sorted(carried & set(fam))
            if not wanted or (b, tuple(wanted)) in self.series_swept:
                continue

            def naming():
                ix = self.fit_index()
                return [r for r in self.item_listing_ids(b)
                        if set((ix.get(r) or {}).get("named_series") or []) & set(wanted) and self.buyable(r)]
            if naming():
                continue
            dw = roots(self.items[b - 1].name)
            for r in self.item_reqs(b):
                dw |= roots(r.partition(":")[2])
            walks = []
            for cm, mem in list(self.remembered.items()):
                t, args, o = mem
                if t != tool:
                    continue
                txt = " ".join(v for v in args.values() if isinstance(v, str))
                if not (roots(txt) & dw):
                    continue
                if not any(isinstance(rec, dict) and (len(self.items) == 1 or self.owns_label(b, rec.get(ck)))
                           for _, rec in self.obs_listings(o)):
                    continue
                walks.append((args, o))
            if not walks:
                continue
            self.series_swept.add((b, tuple(wanted)))
            fresh = 0
            walked = set()
            for args, o in walks:
                cur_a, cur_o = args, o
                while (not naming() and fresh < SERIES_PAGES and self.steps_left() - self.reserve() >= BIG_ROOM
                       and not self.finished and not self.rushed()):
                    p = self.paging_plan(tool, cur_o)
                    if not p or p["next"] is None:
                        break
                    a2 = dict(cur_a)
                    a2[p["arg"]] = p["next"]
                    f = []
                    a2 = shape(a2, self.schemas.get(tool), "args", f)
                    if f:
                        break
                    cm2 = self.canon(tool, a2)
                    if cm2 in walked:
                        break
                    walked.add(cm2)
                    if cm2 in self.remembered:
                        o2 = self.remembered[cm2][2]
                    else:
                        res = self.env(tool, a2)
                        if res is None:
                            break
                        o2 = res.get("observation")
                        self.observe(tool, o2)
                        fresh += 1
                    cur_a, cur_o = a2, o2
                if naming() or fresh >= SERIES_PAGES or self.steps_left() - self.reserve() < BIG_ROOM or self.finished:
                    break
            nr = naming()
            texts = []
            if nr:
                texts = self.fill_facts([r for r in nr if not self.read_full(r)][:2], self.reserve() + SERIES_KEEP)
            dn = self.items[b - 1].name
            self.trace.append({"kind": "series_sweep", "device": b, "series": wanted, "pages": fresh, "naming": nr[:4]})
            note = ("[Terry paged through the searches for %s, hunting for a listing of it that mentions the series %s, "
                    "which is the series carried by the opened %s %s that satisfies its requirements (%d pages, each "
                    "taking one step)] " % (dn, ", ".join(wanted), self.items[a - 1].name, ", ".join(proven), fresh))
            if nr:
                note += "%s listings that mention it: %s" % (dn, ", ".join(nr[:4]))
            else:
                note += "None of the seen %s listings mentions it." % dn
            for t in texts:
                note += "\n" + t
            notes.append(note)
        return "\n".join(notes)

    def need_words(self, n):
        out = set()
        for r in self.items[n - 1].live():
            v = r.split(":", 1)[1] if ":" in r else r
            out |= set(RX_RUN3.findall(word_form(v)))
        return out

    def lane_rank(self, ids, pool):
        ck = self.category_key()
        lab = (self.rows.get(ids[0]) or {}).get(ck) if (ck and ids) else None
        lab = "" if lab is None else lab
        owners = [n for n in range(1, len(self.items) + 1) if self.owns_label(n, lab)] if str(lab) else []
        words = set()
        for n in (owners or range(1, len(self.items) + 1)):
            words |= self.need_words(n)
        opening = set()
        for n in owners:
            opening |= set(self.own_rows.get(n, []))
        cheap = self.pick() == "cheapest"

        def key(r):
            rw = set(RX_RUN3.findall(word_form(dump(self.rows.get(r) or {}))))
            k = (0 if r in opening else 1, -len(rw & words))
            if cheap:
                p = self.price(r)
                k = k + (p is None, p if p is not None else 0)
            return k
        return sorted(pool, key=key)

    def widen_singly(self):
        ft = self.facts_tool()
        tgt = self.single_target(ft) if ft else None
        if not ft or not tgt:
            return None
        lane_size = WIDEN_PER_ITEM if len(self.items) <= 2 else 4
        lanes = []
        heads = []
        for lab, ids in self.item_groups():
            opened = sum(1 for r in ids if r in self.full)
            pool = [r for r in ids if r not in self.full and self.buyable(r)]
            pool = self.lane_rank(ids, pool)
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
                    if self.over_cap(n, lid):
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
        return ("[Terry searched all the requirement values of each such item at once and read the leading rows of the "
                "right kind]\n" + "\n".join(lines))


class K05(object):
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
            self.reply_notes(obs)
            sent.append((n, text, ans))
            self.done_log.append("step %d: shopper message sent %r -> reply %r (cart then: %s)"
                                 % (self.step, text, ans[:300], self.cart_mark()))
            self.done_steps.append(self.step)
            self.trace.append({"kind": "message", "item": n, "label": label})
            if isinstance(obs, dict) and obs.get("error"):
                break
        return sent

    @soft(None)
    def reply_notes(self, obs):
        if isinstance(obs, str):
            passages = [obs]
        else:
            passages = []

            def walk(v, d):
                if d > 4:
                    return
                if isinstance(v, str):
                    if len(v.split()) >= 2:
                        passages.append(v)
                elif isinstance(v, dict):
                    for x in v.values():
                        walk(x, d + 1)
                elif isinstance(v, list):
                    for x in v:
                        walk(x, d + 1)
            walk(obs, 0)
        if len(passages) < 2:
            return None
        lines = [l for l in self.loose_lines(obs) if l != passages[0]]
        if not lines:
            return None
        meaning = self.read_note(lines)
        self.trace.append({"kind": "reply_note", "lines": len(lines), "meaning": (meaning or "")[:200]})
        if meaning:
            self.queued.append("(this came from inside a shopper's answer) " + meaning)
        return None

    def rescale_budget(self, fig):
        cur = self.budget
        if not (_num(fig) and fig > 0 and _num(cur) and cur > 0):
            return fig
        r = fig / float(cur)
        if 1.0 / BUDGET_JUMP <= r <= BUDGET_JUMP:
            return fig
        for c in (fig / 100.0, fig * 100.0):
            if 1.0 / BUDGET_NEAR <= c / float(cur) <= BUDGET_NEAR:
                out = int(c) if c == int(c) else round(c, 6)
                self.trace.append({"kind": "budget_rescale", "from": fig, "to": out})
                return out
        return fig

    def unit_budget(self, fig, answer=False):
        if answer:
            fig = self.rescale_budget(fig)
        cur = self.budget
        if not (isinstance(fig, float) and fig != int(fig) and fig > 0 and _num(cur) and cur > 0 and cur == int(cur)):
            return fig
        prices = [self.price(r) for r in list(self.rows.keys())]
        prices = [p for p in prices if p is not None]
        if len(prices) < WHOLE_PRICE_ROWS or any(p != int(p) for p in prices):
            return fig
        cand = round(fig * 100, 6)
        if 1.0 / BUDGET_NEAR <= fig / float(cur) <= BUDGET_NEAR or not (1.0 / BUDGET_JUMP <= cand / float(cur) <= BUDGET_JUMP):
            return fig
        out = int(cand) if cand == int(cand) else cand
        self.trace.append({"kind": "budget_decimal", "from": fig, "to": out})
        return out

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
            if e.get("points") is True and (i, t) not in self.refs and not is_conditional(t):
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
            self.budget = self.unit_budget(b, True)

    def closest_property(self, live, target):
        want = spelled(str(target).partition(":")[0])
        best = None
        bkey = None
        for r in live:
            held = spelled(r.partition(":")[0])
            sh = len(want & held)
            if not sh:
                continue
            inside = want <= held or held <= want
            if not inside and 2 * sh < max(len(want), len(held)):
                continue
            k = (1 if inside else 0, sh)
            if bkey is None or k > bkey:
                best, bkey = r, k
        return best

    def read_note(self, lines):
        for l in lines:
            self.heard.add(l)
        p = NOTE_PROMPT + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger() + "\n\nNOTICE:\n" + "\n".join(lines)
        o = self.ask_json(p, [LIGHT, SPARE], 2500) or {}
        gone = [str(x) for x in items_of(o.get("gone")) if isinstance(x, str) or _whole(x)]
        known = [g for g in gone if g in self.rows]
        for g in known:
            self.unavail.add(g)
        self.note_sold = bool(gone) or o.get("sold_out") is True
        self.note_named = bool(known)
        if self.note_sold and len(lines) == 1:
            self.sold_lines.add(lines[0])
        kind = o.get("kind")
        if kind is not None and kind != "change":
            return ""
        rec = []
        N = len(self.items)
        b = o.get("budget")
        bchg = False
        if _num(b) and b > 0:
            b = self.unit_budget(self.rescale_budget(b), False)
        if _num(b) and b > 0 and b != self.budget:
            self.budget = b
            rec.append("budget now %s" % b)
            bchg = True
        dropped = False
        drops = [e for e in items_of(o.get("drop")) if isinstance(e, dict) and e.get("need") is not None
                 and str(e.get("need")).strip()]
        if drops and self.pick() == "cheapest":
            self.late_drop = True
        for e in drops:
            i = count_of(e.get("item"))
            it = self.items[min(max(i if i else 1, 1), N) - 1]
            nd = str(e["need"]).strip()
            live = it.live()
            if nd in live:
                m = nd
            else:
                m = next((r for r in live if word_form(r.partition(":")[0]) == word_form(nd.partition(":")[0])), None)
                if m is None:
                    m = self.closest_property(live, nd)
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
            rec.append("a price changed")
        if self.pick() == "cheapest" and (dropped or gone or o.get("sold_out") is True or bchg):
            self.wider_held = False
            self.cheaper_marks = set()
        summ = str(o.get("summary") or "").strip()
        if rec or summ:
            self.version += 1
            self.recent = []
            self.change_notes.append(summ if summ else "; ".join(rec))
        return (summ + (" (" + "; ".join(rec) + ")" if rec else "")).strip()

    def split_note(self, req, value):
        if any(req == t for _, t in self.refs):
            return ""
        want = req.split(":", 1)[1].strip() if ":" in req else ""
        if not want or not plain_words_value(want):
            return ""
        if words_apart(want, str(value)) != "apart":
            return ""
        return ("[this attribute carries the words of '%s' only apart from each other, so it is a different value]" % want)

    def evidence(self, ids):
        live = self.live_pairs()
        if not ids or not live:
            return ""
        several = len(self.items) >= 2
        reft = set(t for _, t in self.refs)
        cands = []
        for n, r in live[:8]:
            lim = measured_limit(r.split(":", 1)[1]) if (":" in r and r not in reft) else None
            cands.append((n, r, lim))
        lines = []
        for lid in ids[:12]:
            a = attrs_of(self.full.get(lid)) or {}
            parts = []
            for n, r, lim in cands:
                prop = r.partition(":")[0].strip()
                ps = stems(prop)
                hits = []
                for k, v in a.items():
                    if not names_match(ps, stems(k)):
                        continue
                    h = "%s=%s" % (k, str(v)[:80])
                    if lim:
                        bn = bare_note(v, lim)
                        if bn:
                            h += " " + bn
                    sp = self.split_note(r, v)
                    if sp:
                        h += " " + sp
                    hits.append(h)
                    if len(hits) >= 3:
                        break
                if hits:
                    parts.append(("item %d: " % n if several else "") + prop + " -> " + ", ".join(hits))
            if parts:
                lines.append("%s: %s" % (lid, "; ".join(parts)))
        if not lines:
            return ""
        return ("[evidence: each key listed carries a requirement's property inside its name; a key about another property "
                "(another component or aspect) settles nothing either way, and a differently named key might still hold "
                "that same property] " + " | ".join(lines))

    def listing_record(self, lid):
        r = self.full.get(lid) or self.rows.get(lid)
        if r is None:
            return "(no record seen)"
        return record_cut(r, 400, 3000)

    def other_listings(self, skip, limit=30):
        ids = list(reversed(self.seen))
        if self.pick() == "cheapest":
            ids.sort(key=lambda r: (r not in self.full, self.price(r) is None, self.price(r) or 0))
        out = []
        for r in ids:
            if r in skip or self.sold_out(r):
                continue
            out.append(record_cut(self.full.get(r) or self.rows.get(r), 160, 900))
            if len(out) >= limit:
                break
        return out

    def covers(self, pid, did):
        idx = self.fit_index()
        fs, bar, dc = idx[pid]["fitset"], idx[pid]["barred"], idx[did]["models"]
        return any(code_match(c, x) for c in fs for x in dc) and not any(code_covers(c, x) for c in bar for x in dc)

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
            exempt = [pl for pl in pls if dcodes and self.covers(pl, d)]
            if exempt:
                self.trace.append({"kind": "fit_exempt", "lines": exempt})
            lf = self.link_faults(part, main, d, [pl for pl in pls if pl not in exempt])
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

    def plain_values(self, n):
        out = []
        for r, p, v in self.plain_reqs(n):
            if self.described(n, r) or find_codes(v) or req_limit(v) or RX_NEGATION.search(v):
                continue
            if RX_OPEN_EXCLUDE.search(RX_WS_OR_K05.sub(" ", v)) or not literal_words(v):
                continue
            out.append((p, v))
        return out

    def _listing_words(self, lid, cache):
        if lid in cache:
            return cache[lid]
        words = set()
        for x in (self.rows.get(lid) or {}).values():
            if isinstance(x, (str, int, float)) and not isinstance(x, bool):
                words |= literal_words(str(x))
        for v in (attrs_of(self.full.get(lid)) or {}).values():
            words |= literal_words(str(v))
        cache[lid] = words
        return words

    def holds_values(self, lid, n, pv, cache):
        words = self._listing_words(lid, cache)
        for p, v in pv:
            opts = [literal_words(o) for o in RX_WS_OR_K05.split(v)]
            opts = [o for o in opts if o]
            if any(words_hold(o, words) for o in opts):
                continue
            written = False
            for l in self.item_listing_ids(n):
                lw = self._listing_words(l, cache)
                if any(words_hold(o, lw) for o in opts):
                    written = True
                    break
            if written:
                return False
            if not words_hold(literal_words(p), words):
                return False
        return True

    def fit_plan(self, short=False):
        if not self.fit_pairs:
            return ""
        idx = self.fit_index()
        cache = {}
        lines = []
        for m in sorted(set(mm for p, mm in self.fit_pairs)):
            parts = []
            for p, mm in self.fit_pairs:
                if mm == m and p not in parts:
                    parts.append(p)
            pv = dict((p, self.plain_values(p)) for p in parts)
            mv = self.plain_values(m)
            tally = {}
            order = []
            holding = {}

            def add(c, p, r):
                if c not in tally:
                    tally[c] = {}
                    order.append(c)
                tally[c][p] = tally[c].get(p, 0) + 1
                if pv[p] and self.holds_values(r, p, pv[p], cache):
                    lst = holding.setdefault((c, p), [])
                    if r not in lst:
                        lst.append(r)
            for p in parts:
                for r in self.item_listing_ids(p):
                    if not self.sold_out(r) and not self.over_cap(p, r):
                        for c in sorted(idx[r]["fitset"]):
                            add(c, p, r)
                sysm = self.series_system(p, m)
                if sysm and sysm[0] == p:
                    for r in self.item_listing_ids(p):
                        if not self.sold_out(r) and not self.over_cap(p, r):
                            for c in sorted(set(idx[r]["own_series"]) & sysm[2]):
                                add(c, p, r)
            ranked = sorted(order, key=lambda c: (-len(tally[c]), -sum(tally[c].values()), c))
            valued = [p for p in parts if pv[p]]
            if valued:
                ranked.sort(key=lambda c: -sum(1 for p in valued if holding.get((c, p))))
            ranked = ranked[:6]
            devs = {}
            for c in ranked:
                ds = [r for r in self.item_listing_ids(m) if c in idx[r]["named_series"]
                      or any(code_covers(c, x) for x in idx[r]["models"])]
                if mv:
                    ds.sort(key=lambda r: 0 if self.holds_values(r, m, mv, cache) else 1)
                devs[c] = ds[:2]
            segs = []
            for c in ranked:
                counts = ", ".join("%d %s" % (tally[c][p], self.items[p - 1].name) for p in sorted(parts) if p in tally[c])
                seg = "%s: %s %s" % (c, counts, ("(devices seen: %s)" % ", ".join(devs[c])) if devs[c] else "(no device has turned up yet)")
                if valued:
                    groups = [", ".join(holding[(c, p)][:3]) for p in valued if holding.get((c, p))]
                    if groups:
                        seg += " [parts that carry all the required values: %s]" % "; ".join(groups)
                    else:
                        seg += " [no seen part carries all the required values]"
                segs.append(seg)
            if segs:
                lines.append("codes the parts name for the %s -> %s" % (self.items[m - 1].name, "; ".join(segs)))
            if valued:
                dl = self._device_change(m, ranked, devs, valued, holding)
                if dl:
                    lines.append(dl)
            for p in parts:
                cap = self.item_cap(p, True)
                if cap is not None:
                    lines.append("item %d (%s): one unit may cost at most %s (the request itself set this limit); a part "
                                 "priced above it was counted toward none of the codes listed" % (p, self.items[p - 1].name, gnum(cap)))
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
                "category the device belongs to and read the codes off its rows] " + " | ".join(lines))

    def _device_change(self, m, ranked, devs, valued, holding):
        ck = self.category_key()
        if not ck:
            return ""
        mine = [r for r in self.cart_ids() if self.owns_label(m, (self.rows.get(r) or {}).get(ck))]
        if len(mine) != 1:
            return ""
        d = mine[0]
        dcodes = [c for c in ranked if d in devs.get(c, [])]
        good = [c for c in ranked if all(holding.get((c, p)) for p in valued)]
        if not dcodes or not good or any(c in good for c in dcodes):
            return ""
        others = []
        for c in good:
            for r in devs.get(c, []):
                if r != d and r not in others:
                    others.append(r)
        nm = self.items[m - 1].name
        line = ("the %s %s takes the codes %s, and none of those codes' parts carries all the required values; codes whose "
                "parts do: %s" % (nm, d, ", ".join(dcodes[:3]), ", ".join(good[:3])))
        if others:
            line += " (devices seen for them: %s)" % ", ".join(others[:2])
        line += ": change the %s to one that takes such a code" % nm
        return line


class K06(object):
    @soft("")
    def cheapest_sets(self, m):
        idx = self.fit_index()
        parts = sorted(set(p for p, mm in self.fit_pairs if mm == m))
        shelves = {}
        for p in parts:
            ls = [r for r in self.item_listing_ids(p) if self.price(r) is not None and self.buyable(r) and not self.over_cap(p, r)]
            ls.sort(key=lambda r: (self.price(r), r))
            shelves[p] = (self.series_system(p, m), ls)
        sets = []
        for d in self.item_listing_ids(m):
            if (not idx[d]["models"] and not idx[d]["named_series"]) or self.price(d) is None or not self.buyable(d):
                continue
            if self.over_cap(m, d):
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
                + "; ".join("%s = %s" % (" + ".join(ids), gnum(t)) for t, ids in sets[:CHEAPEST_SETS]))

    def store_values(self, prop):
        ps = spelled(prop)
        out = set()
        for rid, rec in self.full.items():
            blk = attr_block_raw(rec) or {}
            for k, v in blk.items():
                if isinstance(v, (dict, list)):
                    continue
                if names_match(ps, spelled(k)):
                    out |= set(RX_VAL_SPLIT.sub(" ", word_form(v)).split())
        return out

    def states_all(self, n, rid):
        blk = attr_block_raw(self.full.get(rid)) or {}
        for r in self.items[n - 1].live():
            prop, _, val = r.partition(":")
            vw = set(RX_VAL_SPLIT.sub(" ", word_form(val)).split())
            if not vw or not vw <= self.store_values(prop):
                return False
            ps = spelled(prop)
            ok = False
            for k, v in blk.items():
                if names_match(ps, spelled(k)) and vw <= set(RX_VAL_SPLIT.sub(" ", word_form(v if isinstance(v, str) else str(v))).split()):
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
            cands = [r for r in cands if not self.over_cap(n, r)]
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

    def own_label(self, n):
        ck = self.category_key()
        if not ck or not (1 <= n <= len(self.items)):
            return None
        it = self.items[n - 1]
        targets = [s for s in (roots(it.name), roots(it.shelf)) if s]
        counts = {}
        rejected = set()
        for rid in self.seen:
            v = (self.rows.get(rid) or {}).get(ck)
            lab = str(v).strip() if v is not None else ""
            if not lab or lab in rejected:
                continue
            if lab in counts:
                counts[lab] += 1
                continue
            if any(roots(lab) == t for t in targets) and self.owns_label(n, lab):
                counts[lab] = 1
            else:
                rejected.add(lab)
        if not counts:
            return None
        return max(counts, key=lambda l: (counts[l], l))

    @soft("")
    def nested_labels_line(self):
        if len(self.items) < 2 or not self.category_key():
            return ""
        own = dict((n, self.own_label(n)) for n in range(1, len(self.items) + 1))
        joined = set()
        for a in own:
            for b in own:
                if a != b and own[a] and own[b] and roots(own[a]) < roots(own[b]):
                    joined.add(a)
                    joined.add(b)
        if len(joined) < 2:
            return ""
        if all(word_form(self.items[n - 1].shelf) == word_form(own[n]) for n in joined):
            return ""
        ents = ["item %d (%s) is sold under its own label '%s'" % (n, self.items[n - 1].name, own[n]) for n in sorted(joined)]
        return ("store labels: " + "; ".join(ents) + ". The category label, never the listing's title, settles which item "
                "a listing is for: source every item's line under its own label as listed.")

    @soft("")
    def no_line_note(self, n, lines, labels):
        ol = self.own_label(n)
        if not ol:
            return ""
        note = " (item %d's own label is '%s'" % (n, ol)
        ents = []
        for r in lines:
            lab = labels.get(r)
            if lab is None or not str(lab).strip():
                continue
            if word_form(lab) == word_form(ol) or not (roots(lab) & roots(ol)):
                continue
            e = "%s carries the label '%s'" % (r, lab)
            ows = [m for m in range(1, len(self.items) + 1) if self.owns_label(m, lab)]
            if len(ows) == 1:
                e += " (the label of item %d)" % ows[0]
            ents.append(e)
        if ents:
            note += "; " + ", ".join(ents) + ", which does not serve this item"
        return note + "; source this item's line under its own label '%s')" % ol

    @soft(list)
    def word_faults(self, lid, n):
        blk = attr_block_raw(self.full.get(lid))
        if not blk:
            return []
        out = []
        for r, p, v in self.plain_reqs(n):
            if self.described(n, r) or find_codes(v) or req_limit(v) or RX_NEGATION.search(v):
                continue
            if RX_OPEN_EXCLUDE.search(RX_WS_OR_K06.sub(" ", v)):
                continue
            opts = [literal_words(o) for o in RX_OR_SPLIT_K06.split(v)]
            opts = [o for o in opts if o]
            pk = set(key_words(p))
            keys = [k for k, c in blk.items() if not isinstance(c, (dict, list))
                    and (set(key_words(k)) == pk or self.fuller_key(n, k, p))]
            held = [(k, literal_words(str(blk[k]))) for k in keys]
            held = [(k, w) for k, w in held if w]
            if not opts or not held:
                continue
            if all(not any(_shares_k06(o, w) or _shares_k06(w, o) for o in opts) for k, w in held):
                out.append("%s states %s=%s, which is not the required %s" % (lid, held[0][0], str(blk[held[0][0]])[:60], r))
        return out

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
                faults.append(("other", "item %d (%s) lacks a cart line" % (n, it.name) + self.no_line_note(n, lines, lab)))
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
                if owners[r] == [n]:
                    cap = self.item_cap(n, True)
                    p = self.price(r)
                    if cap is not None and p is not None and cap + 1e-9 < p <= CAP_SANITY * cap:
                        eb = (self.brief.get("budget_scope") == "each" and _num(self.budget)
                              and self.budget < p <= 10 * self.budget)
                        if not eb:
                            faults.append(("other", "%s costs %s per unit, above the ceiling %s that the request set for "
                                           "item %d (%s)" % (r, gnum(p), gnum(cap), n, it.name)))
                    for wf in self.word_faults(r, n):
                        faults.append(("other", wf))
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
                        faults.append(("other", "%s costs %s, above the per-item limit %s" % (r, gnum(p), gnum(self.budget))))
            else:
                tot = sum(self.price(r) * self.units_held(r) for r in lines)
                if tot > self.budget + 1e-9 and tot <= 10 * self.budget:
                    faults.append(("other", "the cart totals %s, above the budget %s" % (gnum(tot), gnum(self.budget))))
        return faults

    def check_key(self, ids):
        k = "|".join(sorted(ids)) + "#" + str(self.version)
        if self.cheaper_round > 0:
            k += "~%d" % self.cheaper_round
        return k

    def cart_check(self, planned=None, limit=SOFT_SPEND):
        lead = planned is not None
        ids = list(planned) if lead else self.cart_ids()
        key = self.check_key(ids)
        js = self.judges(limit)
        if key in self.checked or not js or not ids:
            return ""
        self.checked.add(key)
        parts = [CHECK_PROMPT, "REQUEST:\n" + self.request, "REQUIREMENTS NOW BINDING:\n" + self.ledger()]
        if lead:
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
        weak = []
        if not lead:
            weak = self.weak_lines() or []
            if weak:
                self.trace.append({"kind": "weak_lines", "lines": [w[0] for w in weak]})
        if self.pick() != "any":
            parts.append("OTHER LISTINGS SEEN (newest first):\n" + "\n".join(self.other_listings(set(ids), 15)))
        prompt = "\n\n".join(parts)
        ans = None
        if len(js) > 1 and not lead and not self.audited and self.pick() == "any" and not weak:
            ans = self.ask_json(prompt, [LIGHT], 3000)
            if ans:
                js = [HEAVY] if ans.get("ok") is False and HEAVY in self.judges(limit) else []
        if js and not self.no_credit:
            ans = self.ask_json(prompt, js, 3000) or ans
        self.trace.append({"kind": "cart_check", "key": key, "ok": ans.get("ok") if ans else None})
        self.verdicts[key] = bool(ans) and ans.get("ok") is not False
        if not ans or ans.get("ok") is not False:
            return ""
        self.leads = [l for l in self.leads if l not in ids]
        probs = [p if isinstance(p, str) else dump(p) for p in items_of(ans.get("problems"))][:5]
        better = [str(b) for b in items_of(ans.get("better")) if isinstance(b, (str, int)) and str(b) in self.rows and not self.sold_out(str(b))][:4]
        if better and not lead and not self.cart_has_gaps() and all(self.listing_gaps(b) for b in better):
            self.trace.append({"kind": "check_set_aside", "better": better})
            self.verdicts[key] = True
            return ""
        nb = [b for b in better if not self.contradicts_words(b)]
        if len(nb) != len(better):
            self.trace.append({"kind": "better_dropped", "ids": [b for b in better if b not in nb]})
        better = nb
        if not lead:
            self.better_by_cart["|".join(sorted(ids))] = list(better)
        txt = "A careful review finds the cart still wrong: " + ("; ".join(probs) if probs else "see the requirements in the ledger") + "."
        if better:
            txt += " Candidates that seem preferable: " + ", ".join(better) + "."
        txt += (" Repair the cart, or else demonstrate that the reviewer erred by reading the records afresh, and then "
                "check out.")
        return txt

    def early_trail(self):
        if not self._trail_pre():
            return ""
        r0, h0 = self.chain_rounds, self.chain_holds
        self.chain_rounds = 0
        self.chain_room = EARLY_CHAIN_ROUNDS
        self.early_chain = True
        try:
            txt, _ = self.trail(HUNT_DEADLINE)
        finally:
            self.chain_rounds, self.chain_holds = r0, h0
            self.chain_room = None
            self.early_chain = False
        self.trace.append({"kind": "early_trail", "len": len(txt)})
        return txt

    def soft_call(self, name, fn, fallback):
        try:
            return fn()
        except Exception as e:
            self.trace.append({"kind": "soft_crash", "method": name, "error": ("%s: %s" % (type(e).__name__, e))[:300]})
            return fallback

    def trail(self, limit=None):
        pend = self.pending_refs()
        tool = self.roles.get("offer") or self.roles.get("open")
        tgt = self.single_target(tool) if tool else None
        seen_lines = []
        lost = []
        early = self.early_chain
        opened_here = 0
        while pend:
            room = self.chain_room if self.chain_room is not None else 6
            cap = room if early else min(6 if self.steps_left() > 15 else 4, room)
            if self.chain_rounds >= cap or self.spend >= HARD_SPEND or (limit is not None and self.clock() > limit):
                break
            self.chain_rounds += 1
            self.chain_opened_mark = len(self.opened)
            seen0 = len(self.seen)
            flagged = False
            recs = []
            nopen = 0
            for rid in reversed(self.seen):
                rec = self.full.get(rid) or self.rows.get(rid)
                if self.page_read(rid) and rec is not None and nopen < 45:
                    recs.append("OPENED " + record_cut(rec, 300, 1400))
                    nopen += 1
            for rid in reversed(self.seen):
                if len(recs) >= 60:
                    break
                rec = self.full.get(rid) or self.rows.get(rid)
                if not self.page_read(rid) or rec is None:
                    if rid in self.full:
                        recs.append("ROW " + record_cut(self.full[rid], 120, 700))
                    else:
                        recs.append("ROW " + record_cut(self.rows.get(rid), 80, 300))
            p = (CHAIN_PROMPT + "\n\nREQUIREMENTS WHOSE VALUE SITS ON ANOTHER LISTING:\n" + "\n".join(t for _, t in pend)
                 + "\n\nRECORDS (newest first):\n" + "\n".join(recs))
            a = self.ask_json(p, [LIGHT], 4500)
            if (a is None and self.spend + HEAVY_ROOM <= HARD_SPEND and not self.store_broke and not self.no_credit
                    and self.model_calls < CALL_CAP):
                a = self.ask_json(p, [HEAVY], 4500)
            a = a or {}
            reach = []
            lost = []
            gaps = []
            ptxt = [t for _, t in pend]
            for ch in items_of(a.get("chains")):
                if not isinstance(ch, dict):
                    continue
                need = str(ch.get("need") or "").strip()
                if need in ptxt:
                    req = need
                elif len(ptxt) == 1:
                    req = ptxt[0]
                else:
                    req = self.soft_call("reworded_req", lambda: self.reworded_req(need, ptxt), None)
                    if not req:
                        continue
                links = [l for l in items_of(ch.get("links")) if isinstance(l, dict)]
                links = [self.soft_call("map_title_code", lambda l=l: self.map_title_code(l), l) for l in links]
                ids = [str(l.get("id") or "").strip() for l in links]
                rj = self.soft_call("reject_links", lambda: self.reject_links(req, links, ids), (links, ids, False))
                links, ids, rejected = rj
                if rejected:
                    flagged = True
                val = str(ch.get("value") or "").strip()
                allread = bool(links) and all(self.page_read(i) for i in ids)
                if ch.get("done") is True and allread and val:
                    if self.soft_call("defers", lambda: self.defers(req, ids[-1], val), False):
                        flagged = True
                        continue
                    self.ref_values[req] = val
                    self.version += 1
                    for i in ids:
                        if i in self.rows and i not in self.chain_page_ids:
                            self.chain_page_ids.append(i)
                    continue
                if allread and not str(links[-1].get("find") or "").strip():
                    self.ref_values[req] = val or ("as stated on %s" % ids[-1])
                    self.version += 1
                    for i in ids:
                        if i in self.rows and i not in self.chain_page_ids:
                            self.chain_page_ids.append(i)
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
            if (not reach and gaps and tool and tgt and self.steps_left() > LINK_STEPS
                    and self.link_searches < LINK_MEMO_SEARCHES):
                drop = []
                for gi, (desc, words) in enumerate(gaps[:LINK_GAPS]):
                    if self.steps_left() <= LINK_STEPS or self.link_searches >= LINK_MEMO_SEARCHES or self.finished:
                        break
                    self.link_found = ""
                    for g in self.find_link(desc, words) or []:
                        if g not in reach:
                            reach.append(g)
                    if self.link_found:
                        seen_lines.append("%s: a page, already open, of the shop the chain names; it shows the values "
                                          "listed in the description; take the value off that page." % self.link_found)
                        drop.append(gi)
                lost = [l for i, l in enumerate(lost) if i not in drop]
            if early:
                batch = reach[:max(0, min(3, EARLY_CHAIN_PAGES - opened_here))]
                res_ = 3 * len(self.items) + self.reserve() + 2
            else:
                batch = reach[:max(0, min(3, 6 - len(seen_lines)))]
                res_ = len(batch) + self.reserve() + 1
            if not reach and flagged and pend and not self.finished and len(self.seen) > seen0:
                self.trace.append({"kind": "chain_extra_round"})
                continue
            if not reach or not tool or not tgt or self.steps_left() < res_ or self.finished:
                break
            for rid in batch:
                if early and self.steps_left() <= 3 * len(self.items) + self.reserve() + 1:
                    break
                f = []
                args = shape({tgt: rid}, self.schemas.get(tool), "args", f)
                if f or self.rushed():
                    break
                res = self.env(tool, args)
                if res is None or self.finished:
                    break
                opened_here += 1
                obs = res.get("observation")
                self.observe(tool, obs)
                seen_lines.append("%s: %s" % (rid, self.render(obs)[:1500]))
        out = []
        if seen_lines:
            out.append("[Terry opened these pages, which the requirements lead to]\n" + "\n".join(seen_lines))
        if self.ref_values:
            out.append("[values taken from those pages] " + "; ".join("%s -> %s" % (k, ref_note(k, v)) for k, v in self.ref_values.items()))
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

    def name_values(self, rid, bad=None):
        if bad is None:
            bad = self._noname_keys()
        return [c for c in self._short_vals(rid, bad) if len(c) > 2]

    def of_shop(self, rid, dlow, bad=None):
        if bad is None:
            bad = self._noname_keys()
        return self._names_in(rid, dlow, bad)

    def attr_name_words(self):
        out = set()
        for rec in self.full.values():
            for k in (attr_block_raw(rec) or {}):
                out |= set(RX_RUN_ANY.findall(word_form(k)))
        return out

    def attr_test(self, rid, vw):
        blk = attr_block_raw(self.full.get(rid) or self.rows.get(rid) or {}) or {}
        txt = " ".join(str(v) for v in blk.values()).lower()
        vw = [w for w in vw if w]
        if not vw or not txt.strip():
            return False
        return all(re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", txt) for w in vw)

    def keyed_pairs(self, rid, desc):
        blk = attr_block_raw(self.full.get(rid))
        if not blk or not desc or not self.of_shop(rid, str(desc).lower()):
            return None
        keys = {}
        for k, v in blk.items():
            if isinstance(v, (dict, list)):
                continue
            nk = _norm_k06(k)
            if nk:
                keys[nk] = _norm_k06(str(v))
        order = sorted(keys, key=lambda k: -len(k))
        if not order:
            return None
        desc = str(desc)
        head = desc.split("(", 1)[0]
        if ":" in head:
            label, body = desc.split(":", 1)
            own = _norm_k06(label)
        else:
            body = desc
            own = ""
        firsts = set(k.split()[0] for k in order)
        verdicts = []
        for s in RX_SEG_SPLIT_K06.split(body.lower()):
            ws = _norm_k06(s).split()
            while ws and len(ws[0]) <= 3 and ws[0] not in firsts:
                ws = ws[1:]
            st = " ".join(ws)
            key = next((k for k in order if st == k or st.startswith(k + " ")), None)
            if key is None:
                continue
            wanted = st[len(key):].split()
            if not wanted or len(wanted) > 4:
                continue
            if own and (key in own or own in key):
                continue
            content = keys[key]
            nums = [w for w in wanted if RX_NUM_TOK_K06.fullmatch(w)]
            if nums:
                cn = set(RX_NUM_TOK_K06.findall(content))
                good = all(x in cn for x in nums)
                if not good:
                    kw = set(_fold_s(w) for w in key.split())
                    cl = set(_fold_s(w) for w in RX_LOW_RUN_K06.findall(content))
                    if any(_fold_s(w) not in cl for w in wanted if not RX_NUM_TOK_K06.fullmatch(w) and _fold_s(w) not in kw):
                        continue
                verdicts.append(good)
            else:
                cw = set(_fold_s(w) for w in content.split())
                verdicts.append(all(_fold_s(w) in cw for w in wanted))
        if not verdicts:
            return None
        v = all(verdicts)
        if not v:
            self.trace.append({"kind": "keyed_pair_false", "id": rid})
        return v

    @soft(list)
    def find_link(self, desc, words):
        bad = self._noname_keys()
        dlow = str(desc).lower()
        aw = self.attr_name_words()
        said = [w for w in RX_RUN_ANY.findall(str(words or "").lower()) if not RX_LINK_STOP1.match(w)]
        qw = said[:1] + [w for w in said[1:] if w not in aw]
        shop = qw[0] if qw else ""
        nw = set()
        for r in self.seen:
            if self.of_shop(r, dlow, bad):
                for v in self.name_values(r, bad):
                    nw |= set(RX_RUN3.findall(v.lower()))
        vw = []
        for w in qw[1:]:
            if w not in vw and (len(w) >= 3 or RX_HAS_DIGIT.search(w)) and w not in nw and w != shop:
                vw.append(w)
        if not vw:
            for w in RX_RUN_ANY.findall(dlow):
                if (w not in vw and (len(w) >= 3 or w.isdigit()) and w not in nw and not RX_LINK_STOP1.match(w)
                        and w not in aw):
                    vw.append(w)
        for r in self.opened_shop(desc):
            if self.keyed_pairs(r, desc) is False:
                continue
            if self.attr_test(r, vw):
                self.link_found = r
                self.trace.append({"kind": "link_on_open_page", "id": r})
                return []
        described = [w for w in RX_RUN_ANY.findall(dlow) if (len(w) >= 3 or w.isdigit()) and not RX_LINK_STOP1.match(w)
                     and w not in nw]

        def lacking(r):
            blk = attr_block_raw(self.full.get(r))
            if not blk:
                return False
            txt = " ".join(str(v) for v in blk.values()).lower()
            return any(not re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", txt) for w in vw)

        def cand(r):
            return r not in self.opened and r in self.rows and self.of_shop(r, dlow, bad) and not lacking(r)

        def overlap(r):
            return len(set(described) & set(RX_RUN_ANY.findall(dump(self.rows.get(r)).lower())))
        found = {}

        def search(q, value):
            ids = self.link_search(q, desc, cand)
            got = [r for r in ids if cand(r)]
            for r in got:
                f = found.setdefault(r, [0, len(ids)])
                if value:
                    f[0] += 1
                f[1] = min(f[1], len(ids))
            return got
        if qw and self.value_template():
            g = search(" ".join(qw), False)
            if not g and len(qw) >= 2:
                anyv = False
                if len(qw) >= 3:
                    for w in vw:
                        if search(shop + " " + w, True):
                            anyv = True
                if not anyv:
                    search(shop, False)
        if found:
            ranked = sorted(found, key=lambda r: (overlap(r) == 0, -found[r][0], found[r][1], -overlap(r), r))
            self.trace.append({"kind": "link_found", "ids": ranked[:LINK_CANDIDATES]})
            return ranked[:LINK_CANDIDATES]
        rest = [r for r in reversed(self.seen) if r not in self.opened and self.of_shop(r, dlow, bad) and overlap(r) > 0]
        rest.sort(key=lambda r: (lacking(r), -overlap(r)))
        return rest[:LINK_CANDIDATES]

    def _next_pair(self, tool, obs):
        if not isinstance(obs, dict):
            return None
        sh = self.read_shape(tool, obs)
        if not sh or not sh.get("next") or not sh.get("next_arg"):
            return None
        v = obs.get(sh["next"])
        if (isinstance(v, str) and v.strip()) or _whole(v):
            return (sh["next_arg"], v)
        return None

    def _link_page(self, tool, mem):
        arg, v = mem["next"]
        a = dict(mem["args"])
        a[arg] = v
        f = []
        a = shape(a, self.schemas.get(tool), "args", f)
        if f:
            mem["next"] = None
            return False
        self.link_searches += 1
        res = self.env(tool, a)
        if res is None:
            mem["next"] = None
            return False
        obs = res.get("observation")
        self.observe(tool, obs)
        new = [l for l, _ in self.obs_listings(obs) if l not in mem["ids"]]
        mem["ids"].extend(new)
        mem["next"] = self._next_pair(tool, obs) if new else None
        mem["args"] = a
        self.trace.append({"kind": "link_page", "new": len(new)})
        return bool(new)

    def link_search(self, q, desc, cand):
        vt = self.value_template()
        if not vt or not str(q).strip():
            return []
        key = " ".join(str(q).lower().split())
        bad = self._noname_keys()
        dlow = str(desc).lower()

        def room():
            return (self.link_searches < LINK_MEMO_SEARCHES and self.steps_left() > LINK_STEPS and not self.finished
                    and not self.rushed())
        tool, fixed, field = vt
        mem = self.link_memo.get(key)
        if mem is None:
            if not room():
                return []
            a = dict(fixed)
            a[field] = q
            f = []
            a = shape(a, self.schemas.get(tool), "args", f)
            if f:
                return []
            self.link_searches += 1
            res = self.env(tool, a)
            if res is None:
                return []
            obs = res.get("observation")
            self.observe(tool, obs)
            mem = {"ids": [l for l, _ in self.obs_listings(obs)], "next": self._next_pair(tool, obs), "args": a}
            self.link_memo[key] = mem
            self.trace.append({"kind": "link_search", "query": q, "rows": len(mem["ids"])})
            for _ in range(6):
                ids = mem["ids"]
                if (not ids or not mem["next"] or not room() or not all(self.of_shop(r, dlow, bad) for r in ids)
                        or any(cand(r) for r in ids)):
                    break
                if not self._link_page(tool, mem):
                    break
        elif (mem["next"] and room() and not any(cand(r) for r in mem["ids"])
              and any(self.of_shop(r, dlow, bad) for r in mem["ids"])):
            self._link_page(tool, mem)
        return list(mem["ids"])


class K07(object):
    def reworded_req(self, need, ptxt):
        said = set(RX_RUN_ANY.findall(str(need).lower()))
        if not said or not ptxt:
            return None
        sc = []
        for i, t in enumerate(ptxt):
            w = set(RX_RUN_ANY.findall(t.lower()))
            sc.append((len(said & w) / float(max(1, len(said | w))), i, t))
        sc.sort(key=lambda x: (-x[0], x[1]))
        top = sc[0]
        if top[0] >= REWORD_SHARE and (len(sc) == 1 or top[0] - sc[1][0] > REWORD_LEAD):
            return top[2]
        return None

    def map_title_code(self, link):
        name = str(link.get("id") or "").strip()
        if not name or name in self.rows:
            return link
        inner = [r for r in self.rows if len(r) >= LINK_ID_LEN and r in name]
        if len(inner) == 1:
            l2 = dict(link)
            l2["id"] = inner[0]
            self.trace.append({"kind": "link_inner_id", "from": name, "to": inner[0]})
            return l2
        if not (TITLE_CODE_MIN <= len(name) <= TITLE_CODE_MAX) or not RX_HAS_DIGIT.search(name):
            return link
        rx = re.compile(r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?![a-z0-9])")
        tk = self.title_key()
        hits = [r for r in self.rows if tk and rx.search(str((self.rows.get(r) or {}).get(tk, "")).lower())] if tk else []
        if not hits:
            hits = [r for r in self.rows if rx.search(dump(self.full.get(r) or self.rows.get(r)).lower())]
        if len(hits) == 1:
            l2 = dict(link)
            l2["id"] = hits[0]
            self.trace.append({"kind": "link_title_code", "from": name, "to": hits[0]})
            return l2
        return link

    def add_finding(self, text):
        if text not in self.chain_findings:
            self.chain_findings.append(text)
        self.chain_findings = self.chain_findings[-FINDINGS_KEPT:]

    def reject_links(self, req, links, ids):
        aw = None
        tk = self.title_key()
        for idx in range(len(links)):
            lk, i = links[idx], ids[idx]
            if not i or i not in self.rows:
                break
            opened = self.page_read(i)
            if not opened and i not in self.full:
                continue
            vals = lk.get("values")
            vt = " ".join(str(x) for x in vals) if isinstance(vals, list) else ("" if vals is None else str(vals))
            toks = RX_RUN_ANY.findall(vt.lower())
            if not toks:
                continue
            src = req if idx == 0 else dump(self.full.get(ids[idx - 1]) or self.rows.get(ids[idx - 1]))
            sw = set(RX_RUN_ANY.findall(str(src).lower()))
            shop = set()
            for v in self.name_values(i):
                shop |= set(RX_RUN_ANY.findall(v.lower()))
            kept = []
            prev = None
            for t in toks:
                ok = (t in sw and t not in shop and (len(t) >= LINK_WORD_LEN or t[0].isdigit())
                      and not (t[0].isalpha() and prev is not None and prev.isdigit()))
                prev = t
                if ok:
                    kept.append(t)
            blk = attr_block_raw(self.full.get(i) or self.rows.get(i) or {})
            if not blk:
                continue
            title = str((self.rows.get(i) or {}).get(tk) or "") if tk else ""
            ptxt = compact_text(" ".join("%s %s" % (k, v) for k, v in blk.items()) + " " + title)
            missing = []
            for w in kept:
                if w not in missing and w not in ptxt:
                    missing.append(w)
            if not missing:
                continue
            if not opened:
                if aw is None:
                    aw = self.attr_name_words()
                if not any(w in aw for w in missing):
                    continue
            desc = str(lk.get("about") or "").strip() or "the listing this requirement leads to"
            self.add_finding("%s is not the described listing (%s) for %s: %s absent from its attribute block and title"
                             % (i, desc[:120], req[:120], ", ".join(missing[:FINDING_WORDS])))
            find = str(lk.get("find") or "").strip()
            if not find:
                hay = (desc + " " + req).lower()
                sv = [v for v in self.shop_values()
                      if re.search(r"(?<![a-z0-9])" + re.escape(v.lower()) + r"(?![a-z0-9])", hay)]
                longest = max(sv, key=len) if sv else ""
                uk = []
                for w in kept:
                    if w not in uk:
                        uk.append(w)
                find = " ".join(([longest] if longest else []) + uk)
            self.trace.append({"kind": "link_rejected", "id": i, "missing": missing[:FINDING_WORDS], "find": find})
            nl = list(links[:idx]) + [{"id": "", "about": desc, "find": find}]
            ni = list(ids[:idx]) + [""]
            return nl, ni, True
        return links, ids, False

    def defers(self, req, page, value):
        if (req, page) in self.deferred or ":" not in req:
            return False
        pw = set(key_words(req.split(":", 1)[0]))
        wanted = compact_text(value)
        rec = self.full.get(page) or self.rows.get(page)
        tk = self.title_key()
        title = (self.rows.get(page) or {}).get(tk) if tk else None
        for t in leaves(rec):
            if not isinstance(t, str) or len(t) <= LONG_TEXT_CHARS or t == title:
                continue
            if pw and pw <= set(key_words(t)) and wanted and wanted not in compact_text(t):
                self.deferred.add((req, page))
                self.add_finding("%s hands the property of %s on: its prose mentions the property, yet the value '%s' is "
                                 "absent; follow the listing it refers to" % (page, req[:120], str(value)[:60]))
                self.trace.append({"kind": "chain_defer", "req": req, "page": page})
                return True
        return False

    def chain_sellers(self):
        pend = self.pending_refs()
        if not pend:
            return set()
        cands = list(self.shop_values().keys())
        txt = " ".join(t for _, t in pend)

        def tok(v, t):
            return re.search(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", t) is not None
        found = set(v for v in cands if tok(v, txt))
        for _ in range(CHAIN_WIDEN_PASSES):
            prose = []
            for r in sorted(self.opened):
                if r not in self.rows or not self.page_read(r):
                    continue
                if not (set(self.name_values(r)) & found):
                    continue
                for leaf in record_leaves(self.full.get(r) or self.rows.get(r)):
                    if len(leaf.split()) >= CHAIN_LONG_WORDS:
                        prose.append(leaf)
            new = [v for v in cands if v not in found and any(tok(v, p) for p in prose)]
            if not new:
                break
            found |= set(new)
        return found

    @soft(None)
    def chain_page_view(self, tool, args):
        R = self.roles
        if not tool or tool != R.get("open"):
            return None
        allowed = set(self.offered())
        pt = R.get("offer")
        if not pt or pt == tool:
            cands = sorted(t for t in self.page_tools() if t != tool and t in allowed)
            pt = cands[0] if cands else None
        if not pt or pt == tool or pt not in allowed:
            return None
        ot = self.single_target(tool)
        ptt = self.single_target(pt)
        cmpt = None
        if not ptt and pt == R.get("compare"):
            cmpt = self.compare_target(pt)
        lid = str(args.get(ot) or "").strip() if ot else ""
        if not (ptt or cmpt) or lid not in self.rows or lid in self.gone or pt in self.views.get(lid, set()):
            return None
        ck = self.category_key()
        lab = (self.rows.get(lid) or {}).get(ck) if ck else None
        if lab is not None and str(lab).strip() and any(self.owns_label(n, lab) for n in range(1, len(self.items) + 1)):
            return None
        sellers = self.chain_sellers()
        if not sellers or not (sellers & set(self.name_values(lid))):
            return None
        a = {ptt: lid} if ptt else {cmpt: [lid]}
        f = []
        a = shape(a, self.schemas.get(pt), "args", f)
        if f:
            return None
        self.trace.append({"kind": "chain_page_view", "id": lid, "tool": pt})
        note = ("\n[Terry] %s was displayed through %s as the listing's page, in place of %s, because it belongs to a seller "
                "that a requirement leads to, whose page alone tells where it leads and earns credit for the purchase; you may "
                "repeat the %s call if its attributes are still needed." % (lid, pt, tool, tool))
        return pt, a, note

    def undisplayed(self, lid):
        reader = self.roles.get("offer") or self.roles.get("open")
        return lid not in self.offer_seen and reader not in self.views.get(lid, set())

    def reread_offers(self):
        tool = self.roles.get("offer") or self.roles.get("open")
        tgt = self.single_target(tool) if tool else None
        if not tool or not tgt:
            return ""
        cut = max(self.step - 2, self.refused_at)
        stale = [r for r in self.cart_ids() if r not in self.gone and self.offer_seen.get(r, -99) < cut]
        for r in self.chain_page_ids:
            if r not in stale and r not in self.gone and self.undisplayed(r):
                stale.append(r)
        if not stale:
            return ""
        if self.steps_left() < len(stale) + 5:
            und = [r for r in stale if self.undisplayed(r)]
            if not und or self.steps_left() < len(und) + self.reserve() + 1:
                return ""
            self.trace.append({"kind": "reread_undisplayed", "ids": und})
            stale = und
        lines = []
        for r in stale:
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

    @soft("")
    def late_page_show(self):
        if (not self.checkouts or self.steps_left() != self.reserve() + 1 or self.steps_left() > 2 or self.rushed()
                or self.finished):
            return ""
        tool = self.roles.get("offer") or self.roles.get("open")
        tgt = self.single_target(tool) if tool else None
        if not tool or not tgt:
            return ""
        unseen = [r for r in self.cart_ids() if (self.units_held(r) or 0) > 0 and r not in self.gone and self.undisplayed(r)]
        if len(unseen) != 1:
            return ""
        r = unseen[0]
        f = []
        a = shape({tgt: r}, self.schemas.get(tool), "args", f)
        if f:
            return ""
        res = self.env(tool, a)
        if res is None:
            return ""
        self.offer_seen[r] = self.step
        self.observe(tool, res.get("observation"))
        self.trace.append({"kind": "late_page_show", "id": r})
        if self.undisplayed(r):
            return ""
        return ("\n[Terry] Terry displayed the page of %s beforehand, at the cost of one step, because an ordered line earns "
                "credit only once its page has been displayed." % r)

    @soft("")
    def ignored_cap(self, tool, args):
        caps = [k for k in self.cap_args(tool) if k in args and _num(args[k])]
        if len(caps) != 1:
            return ""
        cap = caps[0]
        mode = self._mode(tool, args)
        if self.ceiling_evidence(tool).get((cap, mode)) != "fails":
            return ""
        want = self.canon(tool, dict((k, v) for k, v in args.items() if k != cap))
        for cm in reversed(list(self.remembered.keys())):
            t, a, o = self.remembered[cm]
            if t != tool or cap not in a or a.get(cap) == args.get(cap):
                continue
            if self.canon(tool, dict((k, v) for k, v in a.items() if k != cap)) != want:
                continue
            if not self.obs_listings(o):
                continue
            self.trace.append({"kind": "ignored_cap", "cap": cap})
            return (SKIP + "this mode of the store's search disregards %s (an earlier query in this mode returned rows dearer "
                    "than its cap), so this call would bring back exactly what the call with %s=%s brought:\n%s"
                    % (cap, cap, dump(a.get(cap)), self.render(o)[:3000]))
        return ""

    def code_tools(self):
        out = set()
        s = self.roles.get("search")
        if s:
            out.add(s)
        vt = self.value_template()
        if vt:
            out.add(vt[0])
        return out

    @soft("")
    def pricey_line(self, before):
        ck = self.category_key()
        if (self.pick() != "cheapest" or not ck or not self.price_key or self.clock_bound()
                or self.steps_left() - self.reserve() < DEAR_ROOM):
            return ""
        line = None
        owner = 0
        for r in list(self.cart.keys()):
            if (self.cart.get(r) or 0) > (before.get(r) or 0) and self.price(r) is not None:
                n = self.line_owner((self.rows.get(r) or {}).get(ck))
                if n and n not in self.dear_checked:
                    line, owner = r, n
                    break
        if not line:
            return ""
        self.dear_checked.add(owner)
        p = self.price(line)
        lab = str((self.rows.get(line) or {}).get(ck, "")).strip().lower()
        codet = self.code_tools()
        for cm in reversed(list(self.remembered.keys())):
            t, a, o = self.remembered[cm]
            ids = [l for l, _ in self.obs_listings(o)] if t in codet else []
            if line not in ids:
                continue
            plan = self.paging_plan(t, o)
            if not plan or plan["next"] is None:
                continue
            a2 = dict(a)
            a2[plan["arg"]] = plan["next"]
            f = []
            a2 = shape(a2, self.schemas.get(t), "args", f)
            if f or self.canon(t, a2) in self.remembered:
                continue
            priced = [l for l in ids if self.buyable(l) and self.price(l) is not None]
            cheaper = [l for l in priced if self.price(l) < p]
            if len(priced) < DEAR_ROWS or len(cheaper) < DEAR_SHARE * len(priced):
                return ""
            res = self.env(t, a2)
            if res is None:
                return ""
            o2 = res.get("observation")
            self.observe(t, o2)
            rend = self.render(o2)
            self.read_cache[mark_of(t, a2)] = (self.version, self.step, rend)
            picks = []
            for l, _ in self.obs_listings(o2):
                if (l not in picks and not self.read_full(l) and self.buyable(l) and self.price(l) is not None
                        and self.price(l) < p and str((self.rows.get(l) or {}).get(ck, "")).strip().lower() == lab):
                    picks.append(l)
            picks = sorted(picks, key=lambda l: (self.price(l), l))[:READS_AFTER_PAGE]
            opened = self.fill_facts(picks, self.reserve() + DEAR_KEEP) if picks else []
            proof = self.evidence([l for l in picks if self.read_full(l)])
            self.trace.append({"kind": "pricey_line", "line": line, "opened": len(opened)})
            txt = ("\n[Terry] %s costs %s, and %d of the %d priced buyable rows on that result page undercut it; rows here are "
                   "not sorted by price, so Terry fetched the following page (%s=%s, one step)%s:\n%s"
                   % (line, gnum(p), len(cheaper), len(priced), plan["arg"], dump(plan["next"]),
                      ", and also opened that page's cheaper rows of the same label" if opened else "", rend))
            if opened:
                txt += "\n" + "\n".join(opened)
            if proof:
                txt += "\n" + proof
            return txt
        return ""

    def _call_ids(self, obs):
        ids = []
        for o in [obs] + list(self.paged):
            for l, _ in self.obs_listings(o):
                if l not in ids:
                    ids.append(l)
        return ids

    @soft("")
    def big_row_reads(self, tool, args, obs):
        ck = self.category_key()
        if not ck or not self.clock_bound() or self.steps_left() - self.reserve() < BIG_ROOM or self.finished:
            return ""
        short = self.short_items()
        if not short:
            return ""
        cands = []
        for l in self._call_ids(obs):
            if self.read_full(l) or not self.buyable(l):
                continue
            sh = [n for n in self.listing_items(l, ck) if n in short]
            if sh:
                cands.append((l, sh[0]))
        cheap = self.pick() == "cheapest"

        def key(c):
            l, n = c
            rw = set(RX_RUN3.findall(word_form(dump(self.rows.get(l) or {}))))
            k = (-len(self.need_words(n) & rw),)
            if cheap:
                p = self.price(l)
                k = k + (p is None, p if p is not None else 0)
            return k + (l,)
        cands.sort(key=key)
        texts = self.fill_facts([l for l, _ in cands[:READS_AFTER_PAGE]], self.reserve() + BIG_ROOM)
        if not texts:
            return ""
        return ("\n[Terry opened these rows by itself, one step apiece, since each serves an item still absent from the "
                "cart]\n" + "\n".join(texts))

    @soft("")
    def short_cheap_reads(self, tool, args, obs):
        ck = self.category_key()
        if (self.pick() != "cheapest" or not self.price_key or not ck or self.clock_bound()
                or self.steps_left() - self.reserve() < SHORT_READ_ROOM or self.finished or not self.paged):
            return ""
        plan = self.paging_plan(tool, self.paged[-1])
        if plan and plan["next"] is not None:
            return ""
        short = self.short_items()
        pool = []
        for l in self._call_ids(obs):
            if self.read_full(l) or not self.buyable(l) or self.price(l) is None:
                continue
            if any(n in short for n in self.listing_items(l, ck)):
                pool.append(l)
        picks = sorted(pool, key=lambda l: (self.price(l), l))[:READS_AFTER_PAGE]
        texts = self.fill_facts(picks, self.reserve() + SHORT_READ_KEEP) if picks else []
        if not texts:
            return ""
        proof = self.evidence([l for l in picks if self.read_full(l)])
        out = ("\n[Terry opened the lowest-priced rows across this search's pages by itself, one step apiece, since the pick "
               "rule is the lowest price and those rows were still unopened]\n" + "\n".join(texts))
        if proof:
            out += "\n" + proof
        return out

    @soft(None)
    def open_sold_peers(self, lost):
        ck = self.category_key()
        if self.pick() != "cheapest" or not ck:
            return None
        cart = set(self.cart_ids())
        taken = []
        for r in lost:
            p = self.price(r)
            lab = (self.rows.get(r) or {}).get(ck)
            if p is None or lab is None or not str(lab).strip():
                continue
            key = str(lab).strip().lower()
            peers = [x for x in self.seen if x != r and not self.read_full(x) and x not in cart and self.buyable(x)
                     and self.price(x) is not None and self.price(x) <= p
                     and str((self.rows.get(x) or {}).get(ck, "")).strip().lower() == key]
            peers.sort(key=lambda x: (self.price(x), x))
            for x in peers[:READS_AFTER_PAGE]:
                if x not in taken:
                    taken.append(x)
        if taken:
            self.fill_facts(taken, self.reserve() + PEER_KEEP)
            self.trace.append({"kind": "sold_peers", "ids": taken})
        return None

    def pricier_swap(self, best, lost):
        bp = self.price(best)
        lp = [self.price(r) for r in lost if self.price(r) is not None]
        if bp is None or not lp:
            return True
        return bp > max(lp) + 1e-9

    def swap_hint(self):
        lost = [r for r in self.cart_ids() if r not in self.gone and self.sold_out(r)]
        if not lost:
            return ""
        for r in lost:
            self.gone[r] = self.step
        self.leads = [l for l in self.leads if l not in lost]
        self.version += 1
        self.open_sold_peers(lost)
        reps = self.replacements(lost)
        txt = ("[Terry: %s sold out from now on (the store shows it cannot be purchased any more); gone for good, do not "
               "open it again." % ", ".join(lost))
        if reps and self.steps_left() <= 6:
            txt += (" Steps are scarce: within this very turn put %s in the cart in the quantity asked and remove the line "
                    "that sold out (two calls), and then order. Other read listings that seem to match: %s."
                    % (reps[0], ", ".join(reps[1:]) or "none"))
        elif reps and self.pick() == "cheapest" and self.pricier_swap(reps[0], lost):
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

    def replacements(self, lost):
        js = self.judges()
        if not js:
            return []
        recs = []
        for r in reversed(self.seen):
            if r in lost or r in self.gone or self.sold_out(r):
                continue
            recs.append(record_cut(self.full.get(r) or self.rows.get(r), 200, 1200))
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

    def _audit_call(self):
        ids = sorted(reversed(self.seen), key=lambda r: r not in self.full)
        recs = []
        for r in ids:
            if self.sold_out(r):
                continue
            if r in self.full:
                recs.append("OPENED " + record_cut(self.full[r], 200, 1200))
            else:
                recs.append("ROW " + record_cut(self.rows[r], 120, 500))
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
            a = self.ask_json(p, [HEAVY], 4000) or a
        return a

    def complete_cart(self):
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or not self.roles.get("cart") or not self.roles.get("buy"):
            return ""
        self.completed = True
        want = self.basket_plan(self.confirmed)
        if not want:
            return ""
        drops = set(r for r in self.cart_ids() if r not in want)
        calls = self.cart_calls(want, drops)
        if not calls or self.steps_left() < len(calls) + 1:
            return ""
        lines = list(want.keys())
        self.fill_facts(lines, len(calls) + 3)
        self.show_links(lines, len(calls) + 3)
        self.cart_check(lines, HARD_SPEND)
        if not self.verdicts.get(self.check_key(lines)):
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
        rr = self.reread_offers()
        lost = [r for r in lines if not self.buyable(r)]
        if lost or self.finished or self.steps_left() <= 0:
            self.trace.append({"kind": "complete_cart_lost", "lost": lost})
            msg = ("[Terry] Terry acted in your place and the decline call did not run: the second check had confirmed %s, so "
                   "Terry arranged the cart (%s)" % (self.confirmed, "; ".join(done)))
            if lost:
                msg += ", yet the fresh offer reads showed %s unbuyable: swap that line and then order." % ", ".join(lost)
            else:
                msg += ", yet no step remained to order."
            if rr:
                msg += "\n" + rr
            return msg
        kept = []
        for att in range(2):
            if att == 1:
                self.reread_offers()
                self.cart_check(lines, HARD_SPEND)
                if (not self.verdicts.get(self.check_key(lines)) or self.steps_left() <= 0
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
        return ("[Terry] The decline was not sent because Terry acted in its place: the independent check passed %s, so "
                "Terry changed the cart (%s) and placed the order. Store replies: %s"
                % (self.confirmed, "; ".join(done), " -> then: ".join(kept)))

    def cart_short(self):
        held = [r for r in self.cart_ids() if (self.units_held(r) or 0) > 0]
        return len(held) < len(self.items) or bool(self.short_items())

    @soft("")
    def reopen_search(self):
        s = self.roles.get("search")
        if self.search_reopened or self.checkouts or not s or self.steps_left() <= self.reserve() + 3 or not self.cart_short():
            return ""
        if s in self.offered():
            return ""
        self.search_reopened = True
        self.trace.append({"kind": "search_reopened"})
        return ("[Terry] The search and listing tools are usable once more: for every item still lacking a cart line, "
                "locate, open and put one listing in the cart, then order within the remaining steps.")

    def offered(self):
        R = self.roles
        ends = [t for t in (R.get("buy"), R.get("refuse")) if t]
        allowed = [t["function"]["name"] for t in self.tools]
        ask = R.get("ask")
        if ask and self.msg_args("x" * 200) is not None:
            allowed = [n for n in allowed if n != ask]
        if ends:
            L, r = self.steps_left(), self.reserve()
            keep = None
            if L <= r:
                keep = set(ends)
            elif L <= r + 3:
                keep = set(ends) | set([R.get("cart"), R.get("cart_view")])
                if L >= 3 and self.short_items():
                    keep.add(self.facts_tool() or R.get("open"))
                    if L >= 4:
                        keep.add(R.get("search"))
            elif L <= len(self.items) + 5 and not self.checkouts:
                if self.short_items():
                    keep = None
                elif self.search_reopened and self.cart_short():
                    keep = None
                else:
                    keep = set(ends) | set([R.get("cart"), R.get("cart_view"), R.get("offer")])
            if keep is not None:
                nar = [n for n in allowed if n in keep]
                if nar:
                    allowed = nar
        return allowed

    def required_codes(self):
        reft = set(t for _, t in self.refs)
        out = []
        for n, r in self.live_pairs():
            if r in reft:
                continue
            v = r.split(":", 1)[1] if ":" in r else r
            for c in find_codes(v):
                if c not in out:
                    out.append(c)
        return out

    def strip_codes(self, text, req):
        spans = []
        for m in RX_FIT_CODE_ANY.finditer(text):
            c = RX_GAP.sub(" ", m.group(0)).strip().lower()
            if any(code_covers(c, rc) or code_covers(rc, c) for rc in req):
                spans.append(m)
        if not spans:
            return text, []
        t = text
        for m in reversed(spans):
            before = t[:m.start()]
            after = t[m.end():]
            im = RX_CODE_INTRO.search(before)
            if im:
                before = before[:im.start()]
            t = before + " " + after
        for m in spans:
            for d in RX_DIGIT_RUN_K07.findall(m.group(0)):
                t = re.sub(r"(?<![A-Za-z0-9_\-])" + d + r"(?![A-Za-z0-9_\-])", " ", t)
        return " ".join(t.split()), [m.group(0) for m in spans]

    def holds_req_code(self, args):
        req = self.required_codes()
        if not req:
            return False
        return any(isinstance(v, str) and self.strip_codes(v, req)[1] for v in args.values())

    def drop_codes(self, tool, args):
        req = self.required_codes()
        if not req:
            return args, "", ""
        props = self._props(tool)
        out = dict(args)
        gone = []
        for k, v in args.items():
            if not isinstance(v, str):
                continue
            p = props.get(k)
            if isinstance(p, dict) and p.get("enum"):
                continue
            t, sp = self.strip_codes(v, req)
            if not sp:
                continue
            if not RX_HAS_ALNUM_K07.search(t):
                return args, "", (SKIP + "fit codes draw no rows from this store's search (an earlier code search came back "
                                  "empty), and with the codes %s gone the value %s is empty. Query the item's kind plus its "
                                  "remaining required values in place of the code, because a part counts as fitting when the "
                                  "code appears in its own text." % (", ".join(sp), dump(v)))
            out[k] = t
            gone.extend(sp)
        if not gone:
            return args, "", ""
        self.trace.append({"kind": "codes_dropped", "codes": gone})
        note = ("\n[Terry] Fit codes draw no rows from this store's search (an earlier code search came back empty), so Terry "
                "dropped the named codes from this search (%s); a part counts as fitting when the code appears in its own "
                "text, and rows carrying it are flagged under the result." % dump(out))
        return out, note, ""

    def wanted_units(self, lid):
        if len(self.items) == 1:
            return self.items[0].qty
        ck = self.category_key()
        o = self.label_owner((self.rows.get(lid) or {}).get(ck)) if ck else None
        return self.items[o - 1].qty if o else 1

    def zero_count_fix(self, tool, args):
        props = self._props(tool)
        for k, v in args.items():
            p = props.get(k)
            if isinstance(v, str) and isinstance(p, dict) and p.get("enum") and RX_REMOVAL_MODE.search(v):
                return args, ""
        held = set(self.cart.keys()) | set(self.cart_ids())
        notes = []

        def fix(obj, pr):
            zk = [k for k, v in obj.items() if isinstance(pr.get(k), dict) and "integer" in kinds(pr[k])
                  and not isinstance(v, (str, bool)) and count_of(v) == 0]
            ids = [v.strip() for v in obj.values() if isinstance(v, str) and v.strip() in self.rows]
            if len(zk) == 1 and len(ids) == 1 and ids[0] not in held:
                want = self.wanted_units(ids[0])
                mx = pr[zk[0]].get("maximum")
                if isinstance(mx, (int, float)) and not isinstance(mx, bool):
                    want = min(want, int(mx))
                if want > 0:
                    obj = dict(obj)
                    obj[zk[0]] = want
                    notes.append("%s=%d for %s" % (zk[0], want, ids[0]))
            return obj
        a = fix(dict(args), props)
        for k, v in list(a.items()):
            p = props.get(k)
            if isinstance(v, list) and isinstance(p, dict) and isinstance(p.get("items"), dict) and isinstance(p["items"].get("properties"), dict):
                a[k] = [fix(x, p["items"]["properties"]) if isinstance(x, dict) else x for x in v]
        if not notes:
            return args, ""
        self.trace.append({"kind": "zero_count_fix", "notes": notes})
        return a, "[Terry] A zero count adds no units, since the count is what gets added, so Terry sent %s." % "; ".join(notes)


class K08a(object):
    def listing_items(self, rid, ck=None):
        if len(self.items) == 1:
            return [1]
        if ck is None:
            ck = self.category_key()
        if not ck:
            return []
        rec = self.rows.get(rid)
        if not isinstance(rec, dict) or ck not in rec:
            rec = self.full.get(rid) or rec or {}
        lab = rec.get(ck) if isinstance(rec, dict) else None
        if lab is None or not str(lab).strip():
            return []
        return [n for n in range(1, len(self.items) + 1) if self.owns_label(n, lab)]

    def owner_item(self, rid, ck=None):
        its = self.listing_items(rid, ck)
        return its[0] if len(its) == 1 else 0

    def item_reqs(self, n):
        if not (1 <= n <= len(self.items)):
            return []
        return [r for r in self.items[n - 1].live() if not is_conditional(r)]

    def plain_reqs(self, n):
        if not n:
            return []
        reft = set(t for _, t in self.refs)
        out = []
        for r in self.item_reqs(n):
            if r in reft:
                continue
            p, c, v = r.partition(":")
            if c and p.strip() and v.strip():
                out.append((r, p.strip(), v.strip()))
        return out

    def fuller_key(self, n, key, prop):
        if not (1 <= n <= len(self.items)):
            return False
        it = self.items[n - 1]
        own = set(key_words(it.name)) | set(key_words(it.shelf))
        kw = set(key_words(key))
        pw = set(key_words(prop))
        return bool(pw) and pw < kw and all(w in own for w in kw - pw)

    @soft(False)
    def row_breaks_limit(self, rid, n):
        if not n:
            return False
        txt = " ".join(record_leaves(self.rows.get(rid) or {}))
        reft = set(t for _, t in self.refs)
        for r in self.item_reqs(n):
            if r in reft:
                continue
            val = r.split(":", 1)[1] if ":" in r else r
            lim = req_limit(val)
            if not lim or lim["kind"] == "exact":
                continue
            figs, _ = figures_in_unit(txt, lim["unit"], False)
            if figs and breaks_limit(lim, figs):
                return True
        return False


class K08(object):
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
            for l in self.cart:
                self.ever_held.add(l)
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
                self.ever_held.add(lid)
            else:
                self.cart.pop(lid, None)

    def render(self, obs):
        recs = self.obs_listings(obs)
        shown = compact_rows(obs) if len(recs) >= 2 else obs
        t = dump(clip(shown, ROW_CLIP if recs else TEXT_CLIP))
        if len(t) > OBS_CAP:
            t = t[:OBS_CAP] + "...(cut)"
        return t

    def over_budget(self, lines):
        if not _num(self.budget) or not lines:
            return False
        ps = [self.price(r) for r in lines]
        if any(p is None for p in ps):
            return False
        if self.brief.get("budget_scope") == "each":
            return any(p > self.budget + 1e-9 for p in ps)
        return sum(self.price(r) * (u if u else 1) for r, u in lines.items()) > self.budget + 1e-9

    def keep_lines_guard(self, tool, args, later, start=None):
        if start is None:
            start = dict(self.cart)
        if self.steps_left() > KEEP_LINES_STEPS or not start:
            return ""
        saved = (dict(self.cart), self.cart_touched, self.cart_changed_at, set(self.ever_held))
        try:
            self.cart = dict(self.cart)
            self.note_cart(tool, args)
            for c in later:
                fn = c.get("function") if isinstance(c, dict) else None
                if not isinstance(fn, dict) or fn.get("name") != tool:
                    continue
                raw = fn.get("arguments")
                if isinstance(raw, dict):
                    a = raw
                elif isinstance(raw, str) and raw.strip():
                    a = json_obj(raw) or {}
                else:
                    a = {}
                f = []
                a = shape(a, self.schemas.get(tool), "args", f)
                if f or not a:
                    continue
                a, pb = self.map_ids(tool, a)
                if pb:
                    continue
                a = self.whole_removal(a)
                self.note_cart(tool, a)
            after = dict(self.cart)
        finally:
            self.cart, self.cart_touched, self.cart_changed_at, self.ever_held = saved
        if len(after) >= len(start):
            return ""
        removed = [r for r in start if r not in after and not self.sold_out(r)]
        if not removed or self.over_budget(start):
            return ""
        self.trace.append({"kind": "keep_lines", "removed": removed, "left": self.steps_left()})
        return (SKIP + "only %d steps remain and this call would take %s off the cart, so it would hold fewer lines than now. "
                "Retain %s as cart lines, and place the order immediately."
                % (self.steps_left(), ", ".join(removed), ", ".join(start)))

    def item_codes(self, n):
        reft = set(t for _, t in self.refs)
        out = []
        for r in self.item_reqs(n):
            if r in reft:
                continue
            v = r.split(":", 1)[1] if ":" in r else r
            for c in find_codes(v):
                if c not in out:
                    out.append(c)
        return out

    def code_words(self, lid):
        rec = (self.full.get(lid) if lid in self.full else self.rows.get(lid)) or {}
        txt = " ".join(record_leaves(rec))
        mark = (lid in self.full, len(rec) if isinstance(rec, dict) else 0, len(compact_text(txt)))
        m = self.code_memo.get(lid)
        if m and m[0] == mark:
            return m[1]
        cw = text_codes(txt)
        self.code_memo[lid] = (mark, cw)
        return cw

    def code_complete(self, lid, codes):
        if not codes:
            return False
        cw = self.code_words(lid)
        return all(any(code_match(w, c) for w in cw) for c in codes)

    def fitting_rows(self, n):
        codes = self.item_codes(n)
        if not codes:
            return []
        out = [r for r in self.seen if n in self.listing_items(r) and self.buyable(r) and self.code_complete(r, codes)]
        out.sort(key=lambda r: (self.price(r) is None, self.price(r) or 0, r))
        return out

    def mark_fits(self, items, ids):
        for n in items:
            codes = self.item_codes(n)
            lst = self.fit_rows.setdefault(n, [])
            for l in ids:
                if l not in lst and n in self.listing_items(l) and self.buyable(l) and self.code_complete(l, codes):
                    lst.append(l)

    def live_fits(self, n):
        return sum(1 for x in self.fit_rows.get(n, []) if self.buyable(x)
                   and (x not in self.full or self.verdict_rank(x, n)[0] == 0))

    def cheapest_unread(self, n, ids):
        pool = [x for x in ids if x not in self.full and self.buyable(x) and not self.over_cap(n, x)]
        pool.sort(key=lambda x: (self.price(x) is None, self.price(x) or 0, x))
        return pool[:FIT_OPENS_ITEM]

    def open_fits(self, picks):
        if not picks:
            return 0
        self.fill_facts(picks, self.reserve() + FIT_SCAN_ROOM)
        return sum(1 for p in picks if self.read_full(p))

    def verdict_rank(self, lid, n):
        if lid not in self.full:
            return 1, "not opened"
        diffs = []
        miss = []
        for q in self.item_reqs(n):
            v = q.split(":", 1)[1] if ":" in q else q
            if find_codes(v):
                continue
            vd, sh = self.literal_check(lid, n, q)
            p = q.split(":", 1)[0].strip()
            if vd == "differs":
                diffs.append("%s (wanted %s)" % (sh or p, v.strip()))
            elif vd == "missing":
                miss.append(p)
        if diffs:
            return 3, "opened, states " + "; ".join(diffs[:2])
        if miss:
            return 2, "opened, its page lacks " + ", ".join(miss[:3])
        return 0, "opened, satisfies the remaining requirements literally"

    def fit_pages(self, tool, args, obs, items, ids, stop):
        walked = set([self.canon(tool, args)])
        pages = 0
        cur_a, cur_o = args, obs
        for _ in range(16):
            if (pages >= FIT_SCAN_PAGES or self.fit_scan_pages >= FIT_SCAN_EPISODE_PAGES
                    or self.steps_left() <= self.reserve() + FIT_SCAN_ROOM or self.rushed() or self.finished or stop()):
                break
            plan = self.paging_plan(tool, cur_o)
            if not plan or plan["next"] is None:
                break
            a2 = dict(cur_a)
            a2[plan["arg"]] = plan["next"]
            f = []
            a2 = shape(a2, self.schemas.get(tool), "args", f)
            if f:
                break
            cm = self.canon(tool, a2)
            if cm in walked:
                break
            walked.add(cm)
            if cm in self.remembered:
                o2 = self.remembered[cm][2]
            else:
                res = self.env(tool, a2)
                if res is None:
                    break
                o2 = res.get("observation")
                self.observe(tool, o2)
                pages += 1
                self.fit_scan_pages += 1
            new = [l for l, _ in self.obs_listings(o2) if l not in ids]
            ids.extend(new)
            self.mark_fits(items, [l for l, _ in self.obs_listings(o2)])
            cur_a, cur_o = a2, o2
        return pages

    def sweep_names(self, n):
        fams = set(code_family(c) for c in self.item_codes(n))
        fams.discard("")
        if not fams:
            return []
        bad = self._noname_keys()
        for k in (self.price_key, self.stock_key):
            if k:
                bad.add(k)
        cnt = {}
        for r in self.seen:
            if n not in self.listing_items(r):
                continue
            if not any(code_family(w) in fams for w in self.code_words(r)):
                continue
            for k, v in (self.rows.get(r) or {}).items():
                if k in bad or not isinstance(v, str):
                    continue
                s = v.strip()
                if (2 < len(s) <= 40 and s[:1].isupper() and not RX_HAS_DIGIT.search(s) and len(s.split()) <= 4
                        and not RX_PLACEHOLDER.match(s)):
                    cnt[s] = cnt.get(s, 0) + 1
        done = self.swept.get(n, set())
        names = sorted([s for s in cnt if s not in done], key=lambda s: (-cnt[s], s))
        return names[:SHOPS_SWEPT]

    def maker_sweep(self, n, tool, args):
        vt = self.value_template()
        if not vt:
            return [], 0, 0
        vtool, fixed, field = vt
        if tool == vtool and isinstance(args.get(field), str):
            src = args.get(field)
        else:
            src = max([v for v in args.values() if isinstance(v, str)], key=len, default="")
        stripped, _ = self.strip_codes(src, self.item_codes(n))
        kw = [w for w in stripped.split() if not RX_HAS_DIGIT.search(w)][:SWEEP_KIND_WORDS]
        if not kw:
            return [], 0, 0
        snap = set(self.fit_rows.get(n, []))

        def new_fits():
            return [x for x in self.fit_rows.get(n, []) if x not in snap and x not in self.full and self.buyable(x)]
        queries = []
        pages = 0
        for name in self.sweep_names(n):
            if (new_fits() or self.steps_left() <= self.reserve() + FIT_SCAN_ROOM or self.clock() >= FIT_SCAN_DEADLINE
                    or self.rushed() or self.finished or self.fit_scan_pages >= FIT_SCAN_EPISODE_PAGES):
                break
            self.swept.setdefault(n, set()).add(name)
            q = name + " " + " ".join(kw)
            a = dict(fixed)
            a[field] = q
            f = []
            a = shape(a, self.schemas.get(vtool), "args", f)
            if f:
                break
            res = self.env(vtool, a)
            if res is None:
                break
            o = res.get("observation")
            self.observe(vtool, o)
            queries.append(q)
            self.fit_scan_pages += 1
            ids = [l for l, _ in self.obs_listings(o)]
            self.mark_fits([n], ids)
            pages += self.fit_pages(vtool, a, o, [n], ids, lambda: bool(new_fits()))
        opened = 0
        nf = new_fits()
        if nf:
            opened = self.open_fits(self.cheapest_unread(n, nf))
        self.trace.append({"kind": "maker_sweep", "item": n, "queries": queries, "pages": pages, "opened": opened})
        return queries, pages, opened

    @soft("")
    def fit_scan(self, tool, args, obs):
        N = len(self.items)
        code_items = [n for n in range(1, N + 1) if self.item_codes(n)]
        if not code_items:
            return ""
        ck = self.category_key()
        ids = self._call_ids(obs)
        active = [n for n in code_items if any(n in self.listing_items(l, ck) for l in ids)]
        if not active:
            return ""
        self.mark_fits(active, ids)
        pages = self.fit_pages(tool, args, obs, active, ids,
                               lambda: all(self.live_fits(n) >= FIT_SCAN_ENOUGH for n in active))
        opened = 0
        if (self.steps_left() > self.reserve() + FIT_OPEN_ROOM and self.clock() < FIT_SCAN_DEADLINE and not self.rushed()
                and not self.finished):
            picks = []
            for n in active:
                for x in self.cheapest_unread(n, self.fit_rows.get(n, [])):
                    if x not in picks:
                        picks.append(x)
            opened += self.open_fits(picks[:FIT_OPENS_SCAN])
        queries = []
        for n in active:
            if self.live_fits(n) == 0 and self.clock() < FIT_SCAN_DEADLINE and not self.finished:
                q, pg, op = self.maker_sweep(n, tool, args)
                queries += q
                pages += pg
                opened += op
        lines = []
        several = N >= 2
        for n in active:
            nm = self.items[n - 1].name
            head = ("item %d (%s)" % (n, nm)) if several else nm
            codes = ", ".join(self.item_codes(n))
            cnt = sum(1 for l in ids if n in self.listing_items(l, ck))
            fits = [x for x in self.fit_rows.get(n, []) if self.buyable(x)]
            vr = dict((x, self.verdict_rank(x, n)) for x in fits)
            fits.sort(key=lambda x: (vr[x][0], self.price(x) is None, self.price(x) or 0, x))
            shown = ["%s%s: %s" % (x, (" at " + gnum(self.price(x))) if self.price(x) is not None else "", vr[x][1])
                     for x in fits[:FIT_ROWS_SHOWN]]
            if shown:
                lines.append("%s: these rows carry %s -> %s (out of %d rows of the item's label in this search)"
                             % (head, codes, "; ".join(shown), cnt))
            else:
                lines.append("%s: not one of the %d rows of the item's label read here carries %s; this store cannot search "
                             "by code, so look for the item's kind using a different required value" % (head, cnt, codes))
        txt = ("\n[Terry fit scan: a part qualifies only if its code appears in that part's text"
               + ("; Terry read %d extra pages, one step apiece" % pages if pages else "") + "] " + " | ".join(lines))
        if opened:
            txt += ("\n[Terry] Terry opened %d fitting rows, one step apiece; put into the cart a row that satisfies the "
                    "requirements instead of reopening it." % opened)
        if queries:
            txt += "\n[Terry] Terry additionally queried the makers of that code family: %s." % "; ".join("'%s'" % q for q in queries)
        self.trace.append({"kind": "fit_scan", "items": active, "pages": pages, "opened": opened, "queries": queries,
                           "marked": dict((n, len(self.fit_rows.get(n, []))) for n in active)})
        return txt

    @soft(None)
    def forced_offer_views(self):
        tool = self.roles.get("offer") or self.roles.get("open")
        tgt = self.single_target(tool) if tool else None
        if not tool or not tgt:
            return None
        ids = []
        for r in self.cart_ids() + list(self.chain_page_ids):
            if r not in ids and r not in self.gone and self.undisplayed(r):
                ids.append(r)
        viewed = []
        for r in ids:
            if self.steps_left() <= self.reserve() or self.rushed() or self.finished:
                break
            f = []
            a = shape({tgt: r}, self.schemas.get(tool), "args", f)
            if f:
                break
            res = self.env(tool, a)
            if res is None:
                break
            self.offer_seen[r] = self.step
            self.observe(tool, res.get("observation"))
            viewed.append(r)
        if viewed:
            self.trace.append({"kind": "forced_offer_views", "ids": viewed})
        return None

    def barred_codes(self, lid, codes):
        bar = (self.fit_index().get(lid) or {}).get("barred") or []
        return [c for c in codes if any(code_covers(b, c) or code_covers(c, b) for b in bar)]

    def _scattered(self, choice, texts):
        if not plain_words_value(choice):
            return False
        res = [words_apart(choice, t) for t in texts]
        return "apart" in res and "together" not in res

    def word_elsewhere(self, lid, val, blk):
        tk = self.title_key()
        title = str((self.rows.get(lid) or {}).get(tk) or "") if tk else ""
        attrs = [str(v) for v in blk.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
        choices = [c for c in RX_OR_CHOICE_K08.split(val) if c.strip()] or [val]
        for place, texts in (("title", [title]), ("attributes", attrs)):
            joined = " ".join(texts)
            if not joined.strip():
                continue
            held = literal_words(joined)
            runs = set(RX_RUN_ANY.findall(joined.lower()))
            for ch in choices:
                lw = literal_words(ch)
                if lw and words_hold(lw, held) and not self._scattered(ch, texts):
                    return "met", "%s: %s" % (place, joined[:80])
                cw = [w for w in RX_RUN_ANY.findall(ch.lower()) if not RX_FILLER.fullmatch(w)]
                if len(cw) >= 3 and initials(" ".join(cw)) & runs:
                    return "met", "%s (initials): %s" % (place, joined[:80])
        return "missing", ""

    def word_by_keys(self, lid, n, prop, val, blk, pkeys):
        choices = [c for c in RX_OR_CHOICE_K08.split(val) if c.strip()] or [val]
        deciding = list(pkeys)
        remark = ""
        fk = [k for k in pkeys if self.fuller_key(n, k, prop)]
        if fk:
            best = sorted(fk, key=lambda k: (len(key_words(k)), k))[-1]
            deciding = [best]
            plain = [k for k in pkeys if set(key_words(k)) == set(key_words(prop))]
            if plain:
                pv = literal_words(str(blk[plain[0]]))
                if not any(literal_words(c) and words_hold(literal_words(c), pv) for c in choices):
                    remark = " [%s overrules %s=%s]" % (best, plain[0], str(blk[plain[0]])[:60])
        tk = self.title_key()
        title = str((self.rows.get(lid) or {}).get(tk) or "") if tk else ""
        tw = literal_words(title) if title else set()
        pws = word_form(prop).split()
        plast = pws[-1] if pws else ""
        for k in deciding:
            raw = str(blk[k])
            parts = [p for p in RX_LIST_SEP.split(raw) if p.strip()]
            kws = word_form(k).split()
            kl = kws[-1] if kws else ""
            plural = len(kl) > len(plast) and fold_word(kl) == fold_word(plast) and len(parts) >= 2
            for c in choices:
                lw = literal_words(c)
                if not lw:
                    continue
                if not any(words_hold(lw, literal_words(p)) for p in parts):
                    continue
                if self._scattered(c, [raw]):
                    continue
                if (plural and tw and not words_hold(lw, tw)
                        and any(literal_words(p) and literal_words(p) <= tw and not words_hold(lw, literal_words(p)) for p in parts)):
                    continue
                return "met", "%s=%s%s" % (k, raw, remark)
        k0 = deciding[0]
        return "differs", "%s=%s%s" % (k0, blk[k0], remark)

    def literal_check(self, lid, n, req):
        prop, c, val = str(req).partition(":")
        prop = prop.strip()
        val = val.strip()
        if not c or not prop or not val:
            return "unknown", ""
        if any(req == t for _, t in self.refs):
            rv = self.ref_values.get(req)
            if not rv or RX_PLACEHOLDER_REF.match(str(rv)) or RX_DIFFERENCE.search(val) or RX_OPEN_EXCLUDE.search(str(rv)):
                return "unknown", ""
            val = str(rv)
        codes = find_codes(val)
        if codes:
            if self.code_complete(lid, codes):
                bar = self.barred_codes(lid, codes)
                if bar:
                    return "differs", "fit clause excluding %s" % ", ".join(bar)
                return "met", "text naming %s" % ", ".join(codes)
            return "differs", "text not naming %s" % ", ".join(codes)
        lim = req_limit(val)
        if RX_NEGATION.search(val) and (lim is None or lim["kind"] == "exact"):
            return "unknown", ""
        blk = (attr_block_raw(self.full.get(lid)) or {}) if lid in self.full else {}
        pkeys = [k for k in blk if key_matches(k, prop)]
        if not pkeys and not lim:
            return self.word_elsewhere(lid, val, blk)
        if not pkeys:
            return "missing", ""
        if not lim:
            return self.word_by_keys(lid, n, prop, val, blk, pkeys)
        first = None
        for k in pkeys:
            figs, ranged = figures_in_unit(blk[k], lim["unit"], True)
            if not figs:
                continue
            shown = "%s=%s" % (k, blk[k])
            if first is None:
                first = shown
            if meets_limit(lim, figs, ranged):
                return "met", shown
        return "differs", first or "%s=%s" % (pkeys[0], blk[pkeys[0]])

    def within_money(self, n, lid):
        p = self.price(lid)
        if p is None or self.over_cap(n, lid):
            return False
        if _num(self.budget):
            cost = p if self.brief.get("budget_scope") == "each" else p * self.items[n - 1].qty
            return cost <= self.budget + 1e-9
        return True

    def literal_pool(self, n):
        if len(self.items) == 1:
            return [r for r in self.seen if r in self.full]
        ck = self.category_key()
        if not ck:
            return []
        return [r for r in self.seen if r in self.full and self.owns_label(n, str((self.rows.get(r) or {}).get(ck) or ""))]

    def proven(self, n, skip=()):
        reqs = self.item_reqs(n)
        out = []
        for r in self.literal_pool(n):
            if r in skip or not self.buyable(r) or not self.within_money(n, r):
                continue
            shown = []
            ok = True
            for q in reqs:
                v, s = self.literal_check(r, n, q)
                if v != "met":
                    ok = False
                    break
                shown.append(s)
            if ok:
                out.append((r, self.price(r) or 0, shown))
        out.sort(key=lambda x: (x[1], x[0]))
        return out

    def proven_listings(self, n):
        return [r for r, _, _ in self.proven(n)]

    def literal_picks(self):
        out = {}
        for n in range(1, len(self.items) + 1):
            pr = self.proven(n) if self.item_reqs(n) else []
            if not pr:
                return {}
            out[n] = pr
        return out

    @soft("")
    def checkout_holds(self):
        if self.steps_left() <= self.reserve() + HOLD_ROOM:
            return ""
        lines = [r for r in self.cart_ids() if (self.units_held(r) or 0) > 0]
        if not lines:
            return ""
        for kind in "unitless sizelist weak fitcode lookalike".split():
            if kind in self.holds_fired:
                continue
            self.new_hold_causes = []
            if kind == "unitless":
                t = self.unitless_hold(lines)
            elif kind == "sizelist":
                t = self.size_list_hold(lines)
            elif kind == "weak":
                t = self.weak_evidence_hold(lines)
            elif kind == "fitcode":
                t = self.fit_code_hold(lines)
            else:
                t = self.lookalike_hold(lines)
            if t:
                self.holds_fired.add(kind)
                self.hold_causes |= set(self.new_hold_causes)
                self.trace.append({"kind": "checkout_hold", "hold": kind})
                return t
        return ""

    def unitless_hold(self, lines):
        for r in lines:
            n = self.owner_item(r)
            blk = attr_block_raw(self.full.get(r))
            if not n or not blk:
                continue
            for q, p, v in self.plain_reqs(n):
                lim = req_limit(v)
                if not lim or lim["figure"] <= 0 or (r, q) in self.hold_causes:
                    continue
                ps = spelled(p)
                for k, val in blk.items():
                    x = bare_number(val)
                    if x is None or x <= 0:
                        continue
                    if not names_match(ps, spelled(RX_ROUND_BR_K08.sub(" ", str(k)))):
                        continue
                    f = lim["figure"]
                    if max(x / f, f / x) >= BARE_FACTOR:
                        self.new_hold_causes = [(r, q)]
                        return ("%s shows %s=%s, a plain number without any unit, whereas the requirement '%s' wants a value "
                                "in %s; taken literally the two are tenfold or more apart, so nothing on the line proves the "
                                "requirement is met. Replace it with a listing whose attributes state the property, unit "
                                "included, and satisfy it; the line may stay solely if the store has no such replacement."
                                % (r, k, val, q, lim["unit"]))
        return ""

    def size_list_hold(self, lines):
        tk = self.title_key()
        for r in lines:
            n = self.owner_item(r)
            blk = attr_block_raw(self.full.get(r))
            title = str((self.rows.get(r) or {}).get(tk) or "").strip() if tk else ""
            if not n or not blk or not title:
                continue
            for q, p, v in self.plain_reqs(n):
                lim = req_limit(v)
                if not lim:
                    continue
                want = lim["figure"]
                unit = lim["unit"]
                for k in blk:
                    if not key_matches(k, p):
                        continue
                    figs, _ = figures_in_unit(blk[k], unit, False)
                    if len(figs) < 2 or not any(abs(f - want) <= 1e-9 for f in figs):
                        continue
                    tf, _ = figures_in_unit(title, unit, False)
                    if not tf or any(abs(f - want) <= 1e-9 for f in tf):
                        continue
                    sizes = ", ".join(sorted(set(gnum(f) for f in tf)))
                    return ("%s lists %s=%s, yet its title (%s) gives %s %s; so what is carted is that version rather than "
                            "the %s %s one that the requirement '%s' needs. Cart the wanted version's own listing in its "
                            "place; the line may stay solely if the store has no such replacement."
                            % (r, k, blk[k], title[:100], sizes, unit, gnum(want), unit, q))
        return ""

    def _root_key(self, k, prop):
        return bool(roots(RX_ROUND_BR_K08.sub(" ", str(k))) & roots(prop))

    def weak_evidence_hold(self, lines):
        ck = self.category_key()
        tk = self.title_key()
        found = []
        for r in lines:
            if len(found) >= WEAK_LINES:
                break
            n = self.owner_item(r)
            blk = attr_block_raw(self.full.get(r))
            if not n or not blk:
                continue
            for q, p, v in self.plain_reqs(n):
                lim = req_limit(v)
                if not lim or RX_POINTER_TALK.search(v):
                    continue
                if not (lim["kind"] != "exact" or lim["decimal"]):
                    continue
                if (r, q) in self.hold_causes:
                    continue
                strict = self.unit_under_keys(n, r, p, lim["unit"])
                proof = any(self._root_key(k, p) and RX_HAS_DIGIT.search(str(x))
                            and (not strict or figures_in_unit(x, lim["unit"], True)[0]) for k, x in blk.items())
                if proof:
                    continue
                found.append((r, n, q, p, lim, blk))
                break
        parts = []
        anysame = False
        for r, n, q, p, lim, blk in found:
            lab = str((self.rows.get(r) or {}).get(ck, "") if ck else "").strip().lower()

            def same(x):
                if lab:
                    return str((self.rows.get(x) or {}).get(ck, "")).strip().lower() == lab
                return len(self.items) == 1
            pv = [(x, pr) for x, pr, _ in self.proven(n) if x != r and same(x)][:WEAK_SHOWN]
            unread = []
            if self.steps_left() > self.reserve() + WEAK_UNREAD_ROOM - 1 and (not pv or self.pick() == "cheapest"):
                ceil = min(pr for _, pr in pv) if pv else None
                tw = literal_words(str((self.rows.get(r) or {}).get(tk) or "")) if tk else set()
                pool = [x for x in self.seen if x != r and not self.read_full(x) and same(x) and self.buyable(x)
                        and self.within_money(n, x) and (ceil is None or (self.price(x) is not None and self.price(x) < ceil))]
                pool.sort(key=lambda x: (-len(literal_words(str((self.rows.get(x) or {}).get(tk) or "")) & tw) if tk else 0,
                                         self.price(x) if self.price(x) is not None else float("inf"), x))
                unread = pool[:WEAK_SHOWN]
            if not pv and not unread:
                continue
            self.new_hold_causes.append((r, q))
            t = ("%s gives its %s under no key carrying that property's name (%s), and the grader decides the order on that "
                 "attribute." % (r, p, q))
            sk = [k for k in blk if not self._root_key(k, p) and figures_in_unit(blk[k], lim["unit"], False)[0]][:2]
            if sk:
                anysame = True
                t += (" Its keys %s give a value in %s; should that be the same quantity under a different key name, the line "
                      "already satisfies the requirement and remains." % (", ".join("%s=%s" % (k, blk[k]) for k in sk), lim["unit"]))
            if pv:
                t += (" These opened listings of the same label give it and pass every requirement literally: %s."
                      % ", ".join("%s at %s" % (x, gnum(pr)) for x, pr in pv))
            if unread:
                t += (" These unread rows of the same label might give it: %s."
                      % ", ".join("%s at %s" % (x, gnum(self.price(x)) if self.price(x) is not None else "?") for x in unread))
            parts.append(t)
        if not parts:
            return ""
        return (" ".join(parts) + " Exchange a line like that for a listing whose attributes list that property beneath a key "
                "bearing its name and satisfy it, while still obeying the pick rule; the line may stay solely if the store has "
                "no such replacement" + (" or if its other key measures the same thing" if anysame else "") + ".")


class K09(object):
    def unit_under_keys(self, n, line, prop, unit):
        pr = roots(prop)
        for o, rec in self.full.items():
            if o == line or n not in self.listing_items(o):
                continue
            blk = attr_block_raw(rec) or {}
            for k, v in blk.items():
                if roots(RX_ROUND_BR_K09.sub(" ", str(k))) & pr and figures_in_unit(v, unit, False)[0]:
                    return True
        return False

    @soft("")
    def fit_code_hold(self, lines):
        for r in lines:
            n = self.owner_item(r)
            codes = self.item_codes(n) if n else []
            if not n or not codes or self.code_complete(r, codes):
                continue
            fams = set(code_family(c) for c in codes)
            kin = []
            for w in self.code_words(r):
                if code_family(w) in fams and w not in kin:
                    kin.append(w)
            if r not in self.full and not kin:
                continue
            good = [x for x in self.fitting_rows(n) if x != r][:FIT_HOLD_SHOWN]
            t = "%s is a misfit: its page lacks the required code %s" % (r, ", ".join(codes))
            if kin:
                t += " (it carries %s instead)" % ", ".join(kin[:3])
            if good:
                t += (". Seen rows carrying the code, lowest price first: %s. Put one of these rows that also satisfies the "
                      "item's remaining requirements in the cart, replacing %s."
                      % (", ".join(x + ((" at " + gnum(self.price(x))) if self.price(x) is not None else "") for x in good), r))
            else:
                t += (". No seen row carries the code, and code searches in this store return nothing: search the item's "
                      "kind together with a second required value, and cart a part whose own text carries the code.")
            return t
        return ""

    @soft("")
    def lookalike_hold(self, lines):
        for r in lines:
            n = self.owner_item(r)
            blk = attr_block_raw(self.full.get(r))
            if not n or not blk:
                continue
            for q, p, v in self.plain_reqs(n):
                lim = req_limit(v)
                if not lim:
                    continue
                if RX_NEGATION.search(v) and lim["kind"] == "exact":
                    continue
                if any(key_matches(k, p) for k in blk) or (r, q) in self.hold_causes:
                    continue
                look = [k for k in blk if figures_in_unit(blk[k], lim["unit"], False)[0]]
                if not look:
                    continue
                alts = []
                for o, rec in self.full.items():
                    if o == r:
                        continue
                    ob = attr_block_raw(rec) or {}
                    ok = [k for k in ob if key_matches(k, p)]
                    if not ok or n not in self.listing_items(o) or not self.buyable(o):
                        continue
                    cov = sum(1 for q2, p2, v2 in self.plain_reqs(n) if any(key_matches(k, p2) for k in ob))
                    pr = self.price(o)
                    alts.append((cov, pr if pr is not None else float("inf"), o, ok[0], ob[ok[0]]))
                if not alts:
                    continue
                alts.sort(key=lambda a: (-a[0], a[1], a[2]))
                self.new_hold_causes = [(r, q)]
                ents = ["%s (%s=%s%s)" % (o, k, val, (", " + gnum(pr)) if pr != float("inf") else "")
                        for cov, pr, o, k, val in alts[:WEAK_SHOWN]]
                return ("%s has no attribute carrying the name of the property (%s); its %s=%s is a nearby measure in that "
                        "same unit, not the property itself. Opened listings of this item that bear the property's name in a "
                        "key: %s. Cart, in the line's place, whichever of them satisfies the requirement while obeying the "
                        "pick rule; the line may stay solely when none of them satisfies it."
                        % (r, q, look[0], blk[look[0]], ", ".join(ents)))
        return ""

    @soft(lambda: ("", []))
    def extra_cheaper_rounds(self):
        reads = []
        obj = ""
        for _ in range(EXTRA_CHEAPER_ROUNDS):
            if (self.steps_left() <= self.reserve() + EXTRA_CHEAPER_ROOM or self.finished or self.rushed()
                    or not self.below_cart()):
                break
            got = self.cheaper_rows(True)
            self.trace.append({"kind": "extra_cheaper_round", "round": self.cheaper_round + 1, "opened": len(got)})
            if not got:
                break
            self.cheaper_round += 1
            reads += got
            obj = self.cart_check()
            if obj:
                break
        return obj, reads

    @soft(False)
    def contradicts_words(self, lid):
        n = self.owner_item(lid)
        blk = attr_block_raw(self.full.get(lid))
        if not n or not blk:
            return False
        for q, p, v in self.plain_reqs(n):
            if RX_HAS_DIGIT.search(v) or RX_NEGATION.search(v):
                continue
            want = literal_words(v)
            if not want:
                continue
            sets = [literal_words(str(blk[k])) for k in blk if key_matches(k, p)]
            if sets and all(s and not (s & want) for s in sets):
                return True
        return False

    @soft("")
    def decline_chain_round(self):
        if (not self.pending_refs() or len(self.opened) <= max(0, self.chain_opened_mark) or self.steps_left() < 3
                or self.room() < DECLINE_CHAIN_SECS):
            return ""
        r0, h0, room0 = self.chain_rounds, self.chain_holds, self.chain_room
        before = dict(self.ref_values)
        self.chain_rounds = 0
        self.chain_room = 1
        try:
            txt, _ = self.trail(None)
        finally:
            self.chain_rounds, self.chain_holds, self.chain_room = r0, h0, room0
        gained = [k for k in self.ref_values if k not in before][:3]
        self.trace.append({"kind": "decline_chain_round", "gained": gained})
        cut = txt.find("The purchase is only valid once")
        if cut >= 0:
            txt = txt[:cut].rstrip()
        return txt

    @soft("")
    def literal_hold(self):
        if "literal" in self.decline_checks or self.leads or self.steps_left() <= self.reserve() + HOLD_ROOM:
            return ""
        if self.steps_left() <= SET_ROOM or self.fit_pairs or "wholeset" in self.decline_checks:
            return ""
        picks = self.literal_picks()
        if not picks:
            return ""
        if _num(self.budget) and self.brief.get("budget_scope") != "each":
            taken = set()
            tot = 0
            for n in sorted(picks):
                ch = next((x for x in picks[n] if x[0] not in taken), None)
                if ch is None:
                    return ""
                taken.add(ch[0])
                tot += ch[1] * self.items[n - 1].qty
            if tot > self.budget + 1e-9:
                self.trace.append({"kind": "literal_hold_over_budget", "total": tot})
                return ""
        self.decline_checks.add("literal")
        several = len(self.items) > 1
        segs = []
        for n in sorted(picks):
            ents = ["%s at %s (%s)" % (x, gnum(pr), ", ".join(s for s in sh if s) or "its attributes")
                    for x, pr, sh in picks[n][:LITERAL_SHOWN]]
            segs.append(("item %d (%s): " % (n, self.items[n - 1].name) if several else "") + "; ".join(ents))
        self.trace.append({"kind": "literal_hold", "named": dict((n, [x for x, _, _ in picks[n][:LITERAL_SHOWN]]) for n in picks)})
        return ("opened listings already satisfy every requirement in the wording of their own attribute values: "
                + " | ".join(segs) + ". Take each figure literally, as printed, rather than as it seems it should read. "
                "Cart, for each item, the preferable one of those named, in the item's quantity and obeying the pick rule, "
                "and order; decline only when each named listing plainly violates a requirement.")

    @soft("")
    def nearest_hold(self):
        if ("nearest" in self.decline_checks or self.room() < NEAREST_SECS or not (self.opened or self.full)
                or self.no_credit):
            return ""
        self.decline_checks.add("nearest")
        if self.nearest_read is not None:
            if not self.nearest_read:
                return ""
            sent = self.nearest_sentence
        else:
            sent = self.read_nearest()
            if not sent:
                return ""
        return ("the request says \"%s\", so a decline goes against its wish when no listing satisfies everything. For each "
                "item, cart the read listing with the smallest number of unmet requirements (an out-of-stock listing misses "
                "one more, as does a price above the budget; honour whatever tolerance the request grants) in the item's "
                "quantity, then order, and name in the reply text the requirement it misses." % sent)

    @soft("")
    def late_proven_order(self):
        cart = self.roles.get("cart")
        buy = self.roles.get("buy")
        if "lateorder" in self.decline_checks or not cart or not buy or self.steps_left() <= 0 or self.finished:
            return ""
        self.decline_checks.add("lateorder")
        picks = self.literal_picks()
        if not picks:
            return ""
        ck = self.category_key()
        plan = {}
        for n in sorted(picks):
            labelled = bool(ck) and any(self.owns_label(n, (self.rows.get(x) or {}).get(ck)) for x in self.seen)
            ch = None
            for x, pr, sh in picks[n]:
                if x in plan or not self.buyable(x) or self.over_cap(n, x):
                    continue
                if labelled and not self.owns_label(n, (self.rows.get(x) or {}).get(ck)):
                    continue
                ch = x
                break
            if ch is None:
                return ""
            plan[ch] = self.items[n - 1].qty
        if _num(self.budget) and self.brief.get("budget_scope") != "each":
            if sum((self.price(x) or 0) * u for x, u in plan.items()) > self.budget + 1e-9:
                return ""
        drops = set(r for r in self.cart_ids() if r not in plan)
        calls = self.cart_calls(plan, drops)
        if calls is None or len(calls) + 1 > self.steps_left():
            return ""
        done = []
        for tool, args in calls:
            f = []
            a = shape(args, self.schemas.get(tool), "args", f)
            res = None if f else self.env(tool, a)
            if f or res is None or self.finished:
                self.trace.append({"kind": "late_proven_order_failed", "done": len(done)})
                if done:
                    return (ACTED + "opened listings satisfy every requirement literally, so Terry started putting %s in the "
                            "cart (%s), but a cart call failed partway." % (", ".join(plan), "; ".join(done)) + self.footer())
                return ""
            self.note_cart(tool, a, res)
            self.observe(tool, res.get("observation"))
            done.append(dump(a))
        head = (ACTED + "opened listings satisfy every requirement literally, so Terry put %s in the cart %s"
                % (", ".join("%s x%d" % (x, u) for x, u in plan.items()),
                   ("(%s)" % "; ".join(done)) if done else "(the cart already held them)"))
        g, gt = self.gate("order")
        if g != "go":
            return head + ". " + (gt or "")
        f = []
        oa = shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", f)
        if f or self.steps_left() <= 0 or self.finished:
            return head + ". Order now."
        res = self.env(buy, oa)
        self.trace.append({"kind": "late_proven_order", "plan": plan, "ok": res is not None})
        if res is None:
            return head + ". The order call failed; order now."
        self.checkouts += 1
        if not self.finished:
            self.refused_at = self.step
        obs = res.get("observation")
        self.observe(buy, obs)
        out = head
        if gt:
            out += ". Terry also performed the request's pre-order actions: " + gt
        return out + ". Then Terry ordered. The store replied: " + self.render(obs)

    def label_of(self, lid):
        ck = self.category_key()
        if not ck:
            return ""
        v = (self.rows.get(lid) or {}).get(ck)
        return str(v).strip().lower() if v is not None else ""

    def resolved_list(self, n):
        out = []
        reft = set(t for _, t in self.refs)
        for r in self.item_reqs(n):
            p, c, v = r.partition(":")
            p, v = p.strip(), v.strip()
            if not c or not p or not v or self.described(n, r):
                continue
            if r in reft:
                rv = self.ref_values.get(r)
                if (not rv or RX_PLACEHOLDER_REF.match(str(rv)) or RX_OPEN_EXCLUDE.search(str(rv))
                        or RX_DIFFERENCE.search(v) or RX_NEGATION.search(v)):
                    return None
                v = str(rv).strip()
            out.append((r, p, v))
        return out

    def codes_stated(self, lid, codes):
        return bool(codes) and self.code_complete(lid, codes) and not self.barred_codes(lid, codes)

    def word_stated(self, lid, n, prop, value, blk, tight):
        choices = [c for c in RX_OR_CHOICE_K09.split(value) if c.strip()] or [value]
        cw = [literal_words(c) for c in choices]
        if not any(cw):
            return True
        own = [k for k in blk if key_matches(k, prop)]
        if own:
            if not tight:
                return True
            for k in own:
                kv = literal_words(str(blk[k]))
                for c, w in zip(choices, cw):
                    if w and _shares_k06(w, kv) and not self._scattered(c, [str(blk[k])]):
                        return True
            return False
        codes = find_codes(value)
        if codes and self.codes_stated(lid, codes):
            return True
        tk = self.title_key()
        title = str((self.rows.get(lid) or {}).get(tk) or "") if tk else ""
        attrs = [str(v) for v in blk.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
        for texts in ([title], attrs):
            joined = " ".join(texts)
            if not joined.strip():
                continue
            held = literal_words(joined)
            runs = set(RX_RUN_ANY.findall(joined.lower()))
            for c, w in zip(choices, cw):
                if w and _shares_k06(w, held) and not self._scattered(c, texts):
                    return True
                cr = [x for x in RX_RUN_ANY.findall(c.lower()) if not RX_FILLER.fullmatch(x)]
                if len(cr) >= 3 and initials(" ".join(cr)) & runs:
                    return True
        return False

    def stem_keys(self, blk, prop):
        pr = roots(prop)
        return [k for k in blk if roots(RX_ROUND_BR_K09.sub(" ", str(k))) & pr and RX_HAS_DIGIT.search(str(blk[k]))]

    def limit_shown(self, blk, prop, lim):
        for k in self.stem_keys(blk, prop):
            v = str(blk[k])
            if not lim.get("unit") or not RX_LETTER_K09.search(v) or figures_in_unit(v, lim["unit"], False)[0]:
                return True
        return False

    def misread_keys(self, blk, prop, lim):
        sk = self.stem_keys(blk, prop)
        if sk and all(bare_note(blk[k], lim) for k in sk):
            return sk
        return []

    def could_state(self, line, n, prop, value, label):
        peers = [x for x in self.seen if x != line and self.label_of(x) == label]
        read = [x for x in peers if attr_block_raw(self.full.get(x))]
        if len(read) < WEAK_GUARD_READS:
            return True
        for x in read:
            if self.word_stated(x, n, prop, value, attr_block_raw(self.full.get(x)), True):
                return True
        tk = self.title_key()
        if tk:
            vw = literal_words(RX_WS_OR_K09.sub(" ", value))
            for x in peers:
                if literal_words(str((self.rows.get(x) or {}).get(tk) or "")) & vw:
                    return True
        return False

    @soft(list)
    def weak_lines(self):
        if not self.category_key():
            return []
        reft = set(t for _, t in self.refs)
        out = []
        for r in self.held_lines():
            blk = attr_block_raw(self.full.get(r))
            its = self.listing_items(r)
            if not blk or len(its) != 1:
                continue
            n = its[0]
            ents = []
            for q in self.item_reqs(n):
                p, c, v = q.partition(":")
                p, v = p.strip(), v.strip()
                if not c or not p or not v or q in reft or find_codes(v) or self.described(n, q):
                    continue
                lim = measured_limit(v)
                if lim:
                    if not self.limit_shown(blk, p, lim):
                        ents.append((q, p, "key"))
                    elif self.misread_keys(blk, p, lim):
                        ents.append((q, p, "bare"))
                    continue
                if RX_OPEN_EXCLUDE.search(RX_WS_OR_K09.sub(" ", v)) or RX_NEGATION.search(v):
                    continue
                if not self.word_stated(r, n, p, v, blk, False) and self.could_state(r, n, p, v, self.label_of(r)):
                    ents.append((q, p, "word"))
            if ents:
                out.append((r, ents))
        return out

    def weak_satisfied(self, lid, req, prop, kind):
        blk = attr_block_raw(self.full.get(lid))
        if not blk:
            return False
        v = req.split(":", 1)[1].strip() if ":" in req else ""
        n = self.owner_item(lid) or 1
        if kind == "word":
            return self.word_stated(lid, n, prop, v, blk, False)
        lim = measured_limit(v)
        return bool(lim) and self.limit_shown(blk, prop, lim) and not self.misread_keys(blk, prop, lim)

    def item_money_ok(self, n, lid):
        p = self.price(lid)
        if p is None:
            return True
        if self.over_cap(n, lid):
            return False
        if _num(self.budget) and self.brief.get("budget_scope") == "each" and p > self.budget + 1e-9:
            return False
        return True

    @soft("")
    def weak_line_hold(self, reads):
        if "weakline" in self.fired_marks() or self.steps_left() <= self.reserve() + HOLD_ROOM:
            return ""
        tk = self.title_key()
        cart = set(self.cart_ids())
        texts = []
        causes = []
        done = 0
        for r, ents in self.weak_lines():
            ents = [e for e in ents if (r, e[0]) not in self.hold_causes]
            if not ents or done >= WEAK_LINES:
                continue
            done += 1
            n = self.owner_item(r)
            lab = self.label_of(r)
            pool = [x for x in self.seen if x != r and x not in cart and lab and self.label_of(x) == lab
                    and self.buyable(x) and self.item_money_ok(n, x)]
            named = [x for x in pool if x in self.full and all(self.weak_satisfied(x, q, p, k) for q, p, k in ents)]
            named.sort(key=lambda x: (self.price(x) is None, self.price(x) or 0, x))
            named = named[:WEAK_SHOWN]
            unread = []
            if not named and self.steps_left() > self.reserve() + WEAK_UNREAD_ROOM:
                tw = literal_words(str((self.rows.get(r) or {}).get(tk) or "")) if tk else set()

                pks = [set(key_words(p)) for q, p, k in ents]

                def has_prop(x):
                    rw = set(key_words(dump(self.rows.get(x) or {})))
                    return any(pk and pk <= rw for pk in pks)
                um = [x for x in pool if x not in self.full]
                um.sort(key=lambda x: (0 if has_prop(x) else 1,
                                       -len(literal_words(str((self.rows.get(x) or {}).get(tk) or "")) & tw) if tk else 0,
                                       self.price(x) is None, self.price(x) or 0, x))
                unread = um[:WEAK_SHOWN]
            if not named and not unread:
                continue
            cl = []
            blk = attr_block_raw(self.full.get(r)) or {}
            for q, p, k in ents:
                if k == "key":
                    cl.append("no key bearing the name %s gives its figure (%s)" % (p, q))
                elif k == "bare":
                    v = q.split(":", 1)[1] if ":" in q else q
                    lim = measured_limit(v) or {}
                    mk = self.misread_keys(blk, p, lim) if lim else []
                    cl.append("%s appears only as a bare number far from the requirement (%s), to be read in %s (%s)"
                              % (p, ", ".join("%s=%s" % (x, blk[x]) for x in mk), lim.get("unit") or "its unit", q))
                else:
                    v = q.split(":", 1)[1].strip() if ":" in q else q
                    cl.append("neither an attribute value nor the title gives '%s' (%s); description text does not count" % (v, q))
            t = "%s fails to show all requirements in the graded way: %s." % (r, "; ".join(cl))
            if named:
                t += (" Read listings of the same label that state it: %s."
                      % ", ".join(x + ((" at " + gnum(self.price(x))) if self.price(x) is not None else "") for x in named))
            else:
                t += (" Unread rows of the same label that might state it: %s."
                      % ", ".join(x + ((" at " + gnum(self.price(x))) if self.price(x) is not None else "") for x in unread))
            texts.append(t)
            causes += [(r, q) for q, p, k in ents]
        if not texts:
            return ""
        self.holds_fired.add("weakline")
        self.hold_causes |= set(causes)
        self.trace.append({"kind": "weak_line_hold", "lines": len(texts)})
        return (reads_heading(reads) + " ".join(texts) + " Replace such a line by one that displays all requirements and "
                "satisfies them, obeying the pick rule, then order; the line may stay solely when no such replacement exists.")

    def fired_marks(self):
        return self.holds_fired

    @soft(list)
    def code_links(self, ids):
        idx = self.fit_index()
        out = []
        for part, main in self.fit_pairs:
            devs = [i for i in ids if main in self.listing_items(i)]
            if len(devs) != 1:
                continue
            d = devs[0]
            sysm = self.series_system(part, main)
            dcodes = sorted(set((idx.get(d) or {}).get("models") or []))
            for pl in ids:
                if pl == d or part not in self.listing_items(pl):
                    continue
                if sysm:
                    carried, naming = (pl, d) if sysm[0] == part else (d, pl)
                    if self.link_holds(carried, naming, sysm[2]) is True:
                        common = sorted(set(idx[carried]["own_series"]) & set(idx[naming]["named_series"]) & set(sysm[2]))
                        out.append("%s works with %s (series %s)" % (pl, d, ", ".join(common)))
                elif dcodes:
                    fs = (idx.get(pl) or {}).get("fitset") or set()
                    bar = (idx.get(pl) or {}).get("barred") or []
                    hits = [x for x in dcodes if any(code_match(c, x) for c in fs) and not any(code_covers(b, x) for b in bar)]
                    if hits:
                        out.append("%s works with %s (%s appear in its fit clause)" % (pl, d, ", ".join(hits)))
        for i in ids:
            its = self.listing_items(i)
            if len(its) != 1:
                continue
            fs = (self.fit_index().get(i) or {}).get("fitset") or set()
            bar = (self.fit_index().get(i) or {}).get("barred") or []
            for q in self.item_reqs(its[0]):
                codes = find_codes(q.split(":", 1)[1]) if ":" in q else []
                if codes and all(any(code_match(c, x) for c in fs) for x in codes) and not any(code_covers(b, x) for b in bar for x in codes):
                    out.append("%s meets '%s': its fit clause lists %s" % (i, q, ", ".join(codes)))
        return out

    def keeps_links(self, n, new, old):
        rest = [r for r in self.held_lines() if r != old]
        idx = self.fit_index()
        for part, main in self.fit_pairs:
            if n not in (part, main):
                continue
            other = main if n == part else part
            partners = [r for r in rest if other in self.listing_items(r)]
            if not partners:
                return False
            for pt in partners:
                device, piece = (pt, new) if n == part else (new, pt)
                sysm = self.series_system(part, main)
                if sysm:
                    carried, naming = (piece, device) if sysm[0] == part else (device, piece)
                    if self.link_holds(carried, naming, sysm[2]) is not True:
                        return False
                else:
                    dc = (idx.get(device) or {}).get("models") or []
                    fs = (idx.get(piece) or {}).get("fitset") or set()
                    bar = (idx.get(piece) or {}).get("barred") or []
                    if not dc or not any(code_match(c, x) for c in fs for x in dc) or any(code_covers(b, x) for b in bar for x in dc):
                        return False
        return True

    def established(self, lid, n, rlist):
        blk = attr_block_raw(self.full.get(lid))
        if not blk:
            return None
        out = []
        for q, p, v in rlist:
            codes = find_codes(v)
            if codes:
                if not self.codes_stated(lid, codes):
                    return None
                out.append("names " + ", ".join(codes))
                continue
            if not literal_words(v) or RX_OPEN_EXCLUDE.search(RX_WS_OR_K09.sub(" ", v)) or measured_limit(v):
                return None
            own = [k for k in blk if key_matches(k, p)]
            if not own:
                return None
            vd, sh = self.word_by_keys(lid, n, p, v, blk, own)
            if vd != "met":
                return None
            out.append(sh)
        return out

    @soft(list)
    def cheaper_pairs(self):
        ck = self.category_key()
        if self.pick() != "cheapest" or not ck:
            return []
        bound = set(x for pr in self.fit_pairs for x in pr)
        cart = set(self.cart_ids())
        out = []
        for r in self.held_lines():
            p = self.price(r)
            its = self.listing_items(r, ck)
            lab = self.label_of(r)
            if p is None or len(its) != 1 or not lab:
                continue
            n = its[0]
            rl = self.resolved_list(n)
            if not rl:
                continue
            pool = [x for x in self.full if x not in cart and self.label_of(x) == lab and self.buyable(x)
                    and self.price(x) is not None and self.price(x) < p - 1e-9 and self.within_money(n, x)]
            pool.sort(key=lambda x: (self.price(x), x))
            for x in pool:
                est = self.established(x, n, rl)
                if est is None:
                    continue
                if n in bound and not self.keeps_links(n, x, r):
                    continue
                out.append((r, x, est))
                break
        return out

    @soft("")
    def cheaper_hold(self, reads=None, late=False):
        if "cheaper" in self.holds_fired:
            return ""
        margin = self.reserve() + HOLD_ROOM
        if late and self.steps_left() > margin:
            return ""
        if not late and self.steps_left() <= margin:
            return ""
        pairs = self.cheaper_pairs()
        if not pairs:
            return ""
        if late and self.steps_left() < self.reserve() + 1 + 2 * len(pairs):
            return ""
        self.holds_fired.add("cheaper")
        for r, x, _ in pairs:
            if (r, x) not in self.cheaper_named:
                self.cheaper_named.append((r, x))
        self.trace.append({"kind": "cheaper_hold", "pairs": [(r, x) for r, x, _ in pairs], "late": late})
        segs = []
        for r, x, est in pairs:
            s = "%s at %s in place of %s at %s" % (x, gnum(self.price(x)), r, gnum(self.price(r)))
            if any(e for e in est):
                s += " (%s)" % ", ".join(e for e in est if e)
            segs.append(s)
        txt = ("the pick favours the cheapest offer, and fully read listings that establish every requirement under keys "
               "bearing their names undercut cart lines: " + "; ".join(segs) + ". Swap each pair through the cart tool "
               "(remove the line and cart the cheaper one in the item's quantity), then order; a line may stay solely when "
               "the named record plainly fails a requirement.")
        return (reads_heading(reads) if (reads and not late) else "") + txt

    @soft("")
    def agreed_swap(self):
        tool = self.roles.get("cart")
        if "agreed" in self.holds_fired or not tool or self.pick() != "cheapest" or not self.cheaper_named:
            return ""
        cur = dict(((r, x), est) for r, x, est in self.cheaper_pairs())
        ids = self.cart_ids()
        better = self.better_by_cart.get("|".join(sorted(ids)), [])
        bound = set(x for pr in self.fit_pairs for x in pr)
        for old, new in self.cheaper_named:
            if (old, new) not in cur:
                continue
            n = self.owner_item(old)
            if new in better:
                reason = "Terry's hold and the cart check's verdict on this exact cart both preferred it"
            elif n in bound and self.keeps_links(n, new, old):
                links = [l for l in self.code_links([new if i == old else i for i in ids]) if new in l][:CODE_LINKS_SHOWN]
                reason = "it keeps every fit link with the cart" + (" (%s)" % "; ".join(links) if links else "")
            else:
                continue
            units = self.units_held(old) or self.cart.get(old) or (self.items[n - 1].qty if n else 1)
            want = dict((r, self.cart.get(r, 1)) for r in ids if r != old and r in self.cart)
            want[new] = units
            calls = self.cart_calls(want, set([old]))
            if not calls or len(calls) + 2 + self.reserve() > self.steps_left():
                continue
            self.holds_fired.add("agreed")
            done = []
            for t, args in calls:
                f = []
                a = shape(args, self.schemas.get(t), "args", f)
                if f:
                    break
                res = self.env(t, a)
                if res is None:
                    break
                self.note_cart(t, a, res)
                obs = res.get("observation")
                self.observe(t, obs)
                done.append("%s -> %s" % (dump(a), self.render(obs)[:200]))
                if self.finished:
                    break
            if old not in self.cart:
                self.dropped_ids.add(old)
            self.dropped_ids.discard(new)
            if old in self.built:
                self.built = [new if b == old else b for b in self.built]
            elif new in self.cart and new not in self.built:
                self.built.append(new)
            self.trace.append({"kind": "agreed_swap", "old": old, "new": new})
            return (HOLD + "the cheapest offer wins under this pick: %s costs %s against %s at %s and establishes every "
                    "requirement, and %s; therefore Terry made the swap itself: %s. Order now."
                    % (new, gnum(self.price(new)), old, gnum(self.price(old)), reason,
                       "; ".join(done) if done else "no call succeeded"))
        return ""


class K10(object):
    @soft("")
    def late_holds(self, reads):
        t = self.cheaper_hold(reads, False)
        if t:
            return HOLD + t
        t = self.weak_line_hold(reads)
        if t:
            return HOLD + t
        return self.agreed_swap()

    def single_device_shape(self):
        mains = []
        parts = []
        for p, m in self.fit_pairs:
            if m not in mains:
                mains.append(m)
            if p not in parts:
                parts.append(p)
        if len(mains) != 1:
            return None
        return mains[0], parts

    def figure_as_written(self, blk, prop, lim):
        if lim["kind"] != "min" and lim["kind"] != "exact":
            return False
        for k in blk:
            if not key_matches(k, prop):
                continue
            figs, _ = figures_in_unit(blk[k], lim["unit"], True)
            if not figs:
                continue
            f = figs[0]
            if lim["kind"] == "min":
                ok = (f > lim["low"] + SLACK) if lim["strict"] else (f >= lim["low"] - SLACK)
            else:
                ok = abs(f - lim["figure"]) <= SLACK
            if ok:
                return True
        return False

    def qualifies(self, lid, n):
        if lid not in self.full or not self.buyable(lid) or self.price(lid) is None:
            return False
        if any(self.described(n, q) for q in self.item_reqs(n)):
            return False
        tr = self.resolved_list(n)
        blk = attr_block_raw(self.full.get(lid))
        if not tr or not blk:
            return False
        for q, p, v in tr:
            codes = find_codes(v)
            if codes:
                if not self.codes_stated(lid, codes):
                    return False
                continue
            lim = measured_limit(v)
            if lim or RX_OPEN_EXCLUDE.search(RX_WS_OR_K10.sub(" ", v)):
                if not (lim and self.figure_as_written(blk, p, lim)):
                    return False
                continue
            pk = [k for k in blk if key_matches(k, p)]
            if pk:
                if self.word_by_keys(lid, n, p, v, blk, pk)[0] != "met":
                    return False
                continue
            if not self.word_stated(lid, n, p, v, blk, True):
                return False
        return True

    def suits(self, part, main, pl, dl, idx=None):
        idx = idx or self.fit_index()
        sysm = self.series_system(part, main)
        if sysm:
            carried, naming = (pl, dl) if sysm[0] == part else (dl, pl)
            if self.link_holds(carried, naming, sysm[2]) is True:
                return True
        dm = idx.get(dl) or {}
        pm = idx.get(pl) or {}
        dc = dm.get("models") or []
        fs = pm.get("fitset") or set()
        bar = pm.get("barred") or []
        return bool(dc) and any(code_match(c, x) for c in fs for x in dc) and not any(code_covers(b, x) for b in bar for x in dc)

    def fit_verdict(self, part, main, pl, dl, idx=None):
        idx = idx or self.fit_index()
        if self.suits(part, main, pl, dl, idx):
            return "yes"
        sysm = self.series_system(part, main)
        if sysm:
            carried, naming = (pl, dl) if sysm[0] == part else (dl, pl)
            if self.link_holds(carried, naming, sysm[2]) is False:
                return "no"
        dc = (idx.get(dl) or {}).get("models") or []
        fs = (idx.get(pl) or {}).get("fitset") or set()
        bar = (idx.get(pl) or {}).get("barred") or []
        if dc and ((fs and not any(code_match(c, x) for c in fs for x in dc)) or any(code_covers(b, x) for b in bar for x in dc)):
            return "no"
        return "unknown"

    @soft(list)
    def complete_baskets(self):
        if not self.fit_pairs:
            return []
        shp = self.single_device_shape()
        if not shp:
            return []
        dev, parts = shp
        state = (len(self.seen), len(self.full), self.version, len(self.gone), len(self.unavail), str(self.budget),
                 sum(len(v) for v in self.views.values()))
        if self.whole_set_memo is not None and self.whole_set_memo[0] == state:
            return self.whole_set_memo[1]
        N = len(self.items)
        q = {}
        for n in range(1, N + 1):
            lst = [r for r in self.item_listing_ids(n) if self.qualifies(r, n) and self.item_money_ok(n, r)]
            lst.sort(key=lambda r: (self.price(r), r))
            q[n] = lst
        idx = self.fit_index()
        out = []
        for d in q.get(dev, []):
            lines = {d: self.items[dev - 1].qty}
            ok = True
            for n in range(1, N + 1):
                if n == dev:
                    continue
                ch = None
                for r in q[n]:
                    if r in lines:
                        continue
                    if n in parts and not self.suits(n, dev, r, d, idx):
                        continue
                    ch = r
                    break
                if ch is None:
                    ok = False
                    break
                lines[ch] = self.items[n - 1].qty
            if not ok:
                continue
            tot = sum(self.price(r) * u for r, u in lines.items())
            if _num(self.budget) and self.brief.get("budget_scope") != "each" and tot > self.budget + SLACK:
                continue
            out.append((tot, lines))
        out.sort(key=lambda b: (b[0], sorted(b[1])))
        self.whole_set_memo = (state, out)
        return out

    def cart_equals(self, lines):
        held = dict((r, self.units_held(r)) for r in self.held_lines())
        return held == dict(lines)

    def basket_text(self, baskets):
        return "; ".join(" + ".join("%s x%d" % (r, u) for r, u in lines.items()) + " = " + gnum(tot)
                         for tot, lines in baskets[:CHEAPEST_SETS])

    @soft("")
    def whole_sets_note(self):
        bs = self.complete_baskets()
        if not bs or any(self.cart_equals(l) for _, l in bs):
            return ""
        return ("\n[complete baskets among the read listings: every line is proven for its item by its own attributes and "
                "every part suits the device] " + self.basket_text(bs) + ". Switch the cart over to one of those baskets, "
                "then place the order.")

    @soft("")
    def basket_decline_hold(self):
        if "wholeset" in self.decline_checks or self.steps_left() <= SET_ROOM:
            return ""
        bs = self.complete_baskets()
        if not bs:
            return ""
        self.decline_checks.add("wholeset")
        self.decline_checks.add("literal")
        self.trace.append({"kind": "basket_decline_hold", "baskets": [sorted(l) for _, l in bs[:2]]})
        return ("complete baskets made of listings read earlier satisfy every requirement in the own wording of their "
                "attributes, their parts compatible with the device: " + self.basket_text(bs) + ". Cart one of those "
                "baskets with the asked quantities (the lowest-total basket under a lowest-price pick) and place the order; "
                "a decline is permitted ONLY when some record plainly breaks a requirement.")

    @soft(False)
    def cart_rewrite(self, wanted):
        if not self.roles.get("cart"):
            return False
        drops = set(r for r in self.cart_ids() if r not in wanted)
        calls = self.cart_calls(wanted, drops)
        if calls is None or len(calls) + 1 > self.steps_left():
            return False
        for t, args in calls:
            f = []
            a = shape(args, self.schemas.get(t), "args", f)
            if f:
                return False
            res = self.env(t, a)
            if res is None or self.finished:
                return False
            self.note_cart(t, a, res)
            self.observe(t, res.get("observation"))
            counts = [count_of(v) for v in a.values() if not isinstance(v, str) and count_of(v) is not None]
            if counts:
                for v in a.values():
                    if isinstance(v, str) and v in self.rows:
                        if counts[0] <= 0 or v in drops:
                            self.dropped_ids.add(v)
                        else:
                            self.dropped_ids.discard(v)
        self.built = [r for r in wanted if not self.sold_out(r)]
        return True

    @soft(False)
    def forced_basket(self):
        bs = self.complete_baskets()
        if not bs or self.steps_left() < 2 or any(self.cart_equals(l) for _, l in bs):
            return False
        held = self.held_lines()
        if held and not self.fit_faults() and not self.missing_items() and len(held) >= len(self.items):
            return False
        ok = self.cart_rewrite(dict(bs[0][1]))
        self.trace.append({"kind": "forced_basket", "ok": ok})
        return ok

    @soft(lambda: None)
    def basket_plan(self, lead):
        ck = self.category_key()
        N = len(self.items)

        def kind(r):
            if not ck:
                return r
            v = (self.rows.get(r) or {}).get(ck)
            return "" if v is None else str(v).strip().lower()

        def owner(r):
            for key, ids in self.lead_items.items():
                m = RX_FIRST_DIGITS.search(str(key))
                if r in ids and m and 1 <= int(m.group(0)) <= N:
                    return int(m.group(0))
            o = self.owner_item(r, ck)
            if o:
                return o
            return 1 if N == 1 else None
        planned = {}
        pitems = set()
        pkinds = set()
        for r in [lead] + [l for l in self.leads if l != lead]:
            if not r or self.sold_out(r):
                continue
            n = owner(r)
            if not n or n in pitems or kind(r) in pkinds:
                continue
            planned[r] = self.items[n - 1].qty
            pitems.add(n)
            pkinds.add(kind(r))
        if lead not in planned:
            return None
        keep = {}
        kkinds = set()
        for r in self.cart_ids():
            if r in planned:
                continue
            if not self.sold_out(r) and kind(r) not in pkinds:
                keep[r] = self.cart.get(r, 1)
                kkinds.add(kind(r))
        if len(planned) + len(kkinds) < N:
            return None
        out = dict(keep)
        out.update(planned)
        return out

    @soft(list)
    def code_checkout(self):
        buy = self.roles.get("buy")
        if not buy:
            return []
        f = []
        oa = shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", f)
        if f:
            return []
        if self.steps_left() >= 2:
            self.forced_offer_views()
        lf = self.line_faults()
        bad = [t for _, t in lf if RX_MONEY_FAULT_K10.search(t)]
        if bad:
            self.trace.append({"kind": "code_checkout_refused", "faults": bad[:3]})
            return []
        texts = []
        for _ in range(2):
            if self.finished or self.steps_left() <= 0 or any(self.sold_out(r) for r in self.cart_ids()):
                break
            if self.steps_left() >= 2:
                g, gt = self.gate("order")
                if gt:
                    texts.append(gt)
                if g != "go" or self.finished or self.steps_left() <= 0:
                    break
            res = self.env(buy, oa)
            self.trace.append({"kind": "code_checkout", "reply": (dump(res.get("observation")) if res else "none")[:200]})
            if res is None:
                break
            self.checkouts += 1
            if not self.finished:
                self.refused_at = self.step
            obs = res.get("observation")
            self.observe(buy, obs)
            texts.append(self.render(obs))
            if self.finished:
                break
        return texts

    @soft("")
    def late_completion(self):
        if ("latecomplete" in self.decline_checks or self.late_held or not self.roles.get("cart")
                or not self.roles.get("buy") or self.steps_left() <= 0):
            return ""
        self.decline_checks.add("latecomplete")
        N = len(self.items)
        lines = [b for b in self.built if not self.sold_out(b)]
        cids = self.cart_ids()
        absent = [r for r in lines if r not in cids]
        if self.fresh_view() is None:
            absent += [r for r in lines if r in self.dropped_ids and r not in absent]
        restored = []
        if len(lines) >= N:
            if absent:
                want = dict((r, self.cart[r]) for r in cids if r in self.cart and r not in absent)
                for a in absent:
                    o = self.owner_item(a)
                    want[a] = self.items[o - 1].qty if o else 1
                calls = self.cart_calls(want, set())
                if not calls or len(calls) + 1 > self.steps_left():
                    return ""
                for t, args in calls:
                    f = []
                    a2 = shape(args, self.schemas.get(t), "args", f)
                    if f:
                        return ""
                    res = self.env(t, a2)
                    if res is None or self.finished:
                        return ""
                    self.note_cart(t, a2, res)
                    self.observe(t, res.get("observation"))
                for a in absent:
                    self.dropped_ids.discard(a)
                restored = absent
        elif N >= 2 and cids:
            if not self.fill_short() or self.missing_items() or len(self.held_lines()) < N:
                return ""
        else:
            return ""
        texts = self.code_checkout()
        self.trace.append({"kind": "late_completion", "restored": restored, "texts": len(texts)})
        if not texts and not restored:
            return ""
        out = (ACTED + "only a few steps remain and the cart contains, or used to contain, a line serving every requested "
               "item; declining at this point earns nothing, so Terry ")
        if restored:
            out += "returned %s to the cart and " % ", ".join(restored)
        if texts:
            out += "placed the order: " + " | then: ".join(texts)
        else:
            out += "could not place the order. Repair whichever line the store rejects, and then order."
        return out

    @soft("")
    def late_rescue(self):
        L = self.steps_left()
        if not (2 <= L <= LATE_ORDER_STEPS) or self.confirmed or self.decline_blocks >= 2:
            return ""
        if len([b for b in self.built if not self.sold_out(b)]) >= len(self.items):
            return ""
        self.decline_blocks = 2
        if not self.leads and not self.audited and self.clock() <= LATE_REVIEW_DEADLINE and self.judges():
            self.audited = True
            fits = self._fits(self._audit_call())
            if fits:
                self.lead_items = fits
                self.leads = [i for v in fits.values() for i in v]
        lead = next((l for l in self.leads if not self.sold_out(l)), None)
        if not lead:
            return ""
        bp = self.basket_plan(lead)
        if not bp or self.clock() > LATE_REVIEW_DEADLINE:
            return ""
        drops = set(r for r in self.cart_ids() if r not in bp)
        calls = self.cart_calls(bp, drops)
        if calls is None or len(calls) > self.steps_left() - 1:
            return ""
        self.cart_check(list(bp), HARD_SPEND)
        ok = self.verdicts.get(self.check_key(list(bp)))
        self.trace.append({"kind": "late_rescue", "basket": list(bp), "ok": ok})
        if not ok:
            return ""
        self.confirmed = lead
        return self.complete_cart()

    def literal_tally(self, lid, n):
        judged = met = 0
        for q in self.item_reqs(n):
            v = self.literal_check(lid, n, q)[0]
            if v != "unknown":
                judged += 1
                if v == "met":
                    met += 1
        return judged, met

    @soft(lambda: None)
    def ranked_basket(self, repair=False):
        self.repaired = []
        N = len(self.items)
        cl = [r for r in self.held_lines() if not self.sold_out(r)]
        mains = sorted(set(m for p, m in self.fit_pairs))
        order = mains + [n for n in range(1, N + 1) if n not in mains]
        partset = set(p for p, m in self.fit_pairs)
        chosen = {}
        for n in order:
            for r in cl:
                if self.listing_items(r) == [n]:
                    chosen[n] = r
                    break
        removed = {}
        if repair:
            for p, m in self.fit_pairs:
                if p in chosen and m in chosen and self.fit_verdict(p, m, chosen[p], chosen[m]) == "no":
                    removed[p] = chosen.pop(p)
        lines = dict((chosen[n], self.items[n - 1].qty) for n in chosen)
        each = self.brief.get("budget_scope") == "each"
        for n in order:
            if n in chosen:
                continue
            qty = self.items[n - 1].qty
            spent = sum((self.price(r) or 0) * u for r, u in lines.items())
            cands = []
            for r in self.item_listing_ids(n):
                if r not in self.full or r in lines or r in self.gone or not self.buyable(r) or self.price(r) is None:
                    continue
                works = True
                rej = False
                for p, m in self.fit_pairs:
                    if n != p and n != m:
                        continue
                    partner = chosen.get(m if n == p else p)
                    if partner is None:
                        continue
                    v = self.fit_verdict(p, m, r if n == p else partner, partner if n == p else r)
                    if v == "no":
                        rej = True
                        break
                    if v != "yes":
                        works = False
                if rej or (repair and n in partset and not works):
                    continue
                j, mt = self.literal_tally(r, n)
                if j and not mt:
                    continue
                cands.append((r, works, mt))
            cands.sort(key=lambda c: (not c[1], c[0] not in self.ever_held, c[0] not in self.leads, -c[2], self.price(c[0]), c[0]))
            pick = None
            for r, w, mt in cands:
                pr = self.price(r)
                if _num(self.budget):
                    if each and pr > self.budget + SLACK:
                        continue
                    if not each and spent + pr * qty > self.budget + SLACK:
                        continue
                pick = r
                break
            if pick is None:
                if n in removed:
                    lines[removed[n]] = qty
                    chosen[n] = removed[n]
                    continue
                return None
            if n in removed:
                self.repaired.append((removed[n], pick))
            lines[pick] = qty
            chosen[n] = pick
        return lines

    @soft(False)
    def misfit_swap(self):
        if not self.fit_pairs or self.steps_left() < 3 or not self.fit_faults():
            return False
        bl = self.ranked_basket(True)
        if not bl or not self.repaired:
            return False
        olds = set(o for o, _ in self.repaired)
        cids = self.cart_ids()
        want = dict((r, self.cart[r]) for r in self.cart if r in cids and r not in olds)
        for o, nw in self.repaired:
            want[nw] = bl[nw]
        ok = self.cart_rewrite(want)
        self.trace.append({"kind": "misfit_swap", "repaired": self.repaired, "ok": ok})
        return ok

    def keep_nearest(self, ans):
        ans = ans if isinstance(ans, dict) else {}
        s = str(ans.get("sentence") or "").strip().strip('"').strip()
        nv = ans.get("nearest")
        found = nv is True or bool(RX_TRUE_YES_K10.fullmatch(str(nv).strip().lower()))
        valid = found and len(s) >= NEAREST_SENTENCE and plain_text(s) in plain_text(self.request)
        self.nearest_read = bool(valid)
        self.nearest_sentence = s if valid else ""
        return self.nearest_read

    def read_nearest(self):
        a = self.ask_json(NEAREST_RULE_PROMPT + "\n\nREQUEST:\n" + self.request, [LIGHT], 800)
        self.trace.append({"kind": "nearest_read", "reply": dump(a)[:200]})
        self.keep_nearest(a)
        return self.nearest_sentence

    def ask_nearest(self, fetch=True):
        if (self.nearest_read is None and fetch and not self.no_credit and self.model_calls < CALL_CAP
                and self.room() >= NEAREST_SECS):
            self.read_nearest()
        return self.nearest_read is True

    def money_price(self, lid):
        p = self.price(lid)
        return p if p is not None else self.own_price(lid)

    def basket_money_ok(self, lines):
        if not _num(self.budget):
            return True
        ps = dict((r, self.money_price(r)) for r in lines)
        if any(p is None for p in ps.values()):
            return False
        if self.brief.get("budget_scope") == "each":
            return all(p <= self.budget + SLACK for p in ps.values())
        return sum(ps[r] * u for r, u in lines.items()) <= self.budget + SLACK

    @soft(lambda: None)
    def nearest_choice(self):
        N = len(self.items)
        share = max(NEAREST_PER_ITEM, NEAREST_RECORDS // max(1, N))
        pool = []
        for n in range(1, N + 1):
            ls = [r for r in self.item_listing_ids(n) if not self.sold_out(r) and r not in self.gone and self.buyable(r)]
            ls.sort(key=lambda r: (r not in self.full, self.money_price(r) is None, self.money_price(r) or 0, r))
            pool += [(n, r) for r in ls[:share]]
        if not pool:
            return None
        shown = pool[:NEAREST_RECORDS]
        lst = "\n".join("item %d - %s: %s" % (n, r, self.listing_record(r)[:NEAREST_RECORD_CHARS]) for n, r in shown)
        il = "\n".join("item %d: %s, quantity %d" % (n, it.name, it.qty) for n, it in enumerate(self.items, 1))
        p = (NEAREST_PICK_PROMPT + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + il + "\n\nREQUIREMENTS NOW BINDING:\n"
             + self.ledger() + "\n\nLISTINGS (BY ITEM):\n" + lst)
        a = self.ask_json(p, [HEAVY, LIGHT], 1500) or {}
        sid = set(r for _, r in shown)
        choice = {}
        misses = []
        for e in items_of(a.get("lines")):
            if not isinstance(e, dict):
                continue
            n = count_of(e.get("item"))
            i = str(e.get("id") or "").strip()
            if not n or not (1 <= n <= N) or i not in sid or n in choice or i in choice.values():
                continue
            choice[n] = i
            for m in items_of(e.get("misses")):
                if str(m).strip():
                    misses.append("%s: %s" % (i, str(m).strip()))
        self.trace.append({"kind": "nearest_choice", "choice": choice, "misses": misses[:6]})
        if not choice:
            return None
        lines = dict((choice[n], self.items[n - 1].qty) for n in sorted(choice))
        if not self.basket_money_ok(lines):
            return None
        self.near_misses = misses[:4]
        return lines

    @soft(False)
    def nearest_cart(self):
        if not self.roles.get("cart") or not self.items:
            return False
        held = [r for r in self.held_lines() if not self.sold_out(r) and r not in self.gone]
        if held and not self.missing_items() and len(held) >= len(self.items):
            return True
        if self.near_rescues >= NEAREST_RESCUES or self.steps_left() < 2:
            return False
        self.near_rescues += 1
        lines = None
        if not self.no_credit and self.room() >= NEAREST_PICK_SECS:
            lines = self.nearest_choice()
        if not lines:
            rb = self.ranked_basket(False)
            lines = rb if rb and self.basket_money_ok(rb) else None
        if not lines:
            return False
        ok = self.cart_rewrite(lines)
        if ok:
            self.near_made = True
        self.trace.append({"kind": "nearest_cart", "lines": lines, "ok": ok})
        return ok

    def fill_free_text(self, args, text, always):
        buy = self.roles.get("buy")
        sch = self.schemas.get(buy) or {}
        props = sch.get("properties")
        props = props if isinstance(props, dict) else {}
        req = sch.get("required")
        req = set(x for x in req if isinstance(x, str)) if isinstance(req, list) else set()
        out = dict(args)
        free = [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum")
                and k not in out and (k in req or (always and RX_FREE_TEXT_SEG.search(under_form(k))))]
        if free and text:
            p = props[free[0]]
            ml = p.get("maxLength")
            out[free[0]] = cut_at_word(text, int(ml)) if (_whole(ml) and ml > 0) else text[:300]
        return out

    def nearest_buy_args(self):
        t = "No listing satisfies all of the requirements, so this order carries the closest listings"
        if self.near_misses:
            t += ". Missed: " + "; ".join(self.near_misses)
        else:
            t += "."
        return self.fill_free_text(dict(self.brief.get("buy_args") or {}), t, bool(self.near_misses))

    def checked_reqs(self, n):
        return [q for q in self.item_reqs(n) if not self.described(n, q)]

    def listing_gaps(self, lid, n=None):
        if lid not in self.full:
            return []
        if n is None:
            n = self.owner_item(lid)
        if not n:
            return []
        out = []
        for q in self.checked_reqs(n):
            v = self.literal_check(lid, n, q)[0]
            if v == "differs" or v == "missing":
                out.append(q)
        return out

    def cart_gaps(self):
        out = []
        for r in self.held_lines():
            if r not in self.full:
                continue
            n = self.owner_item(r)
            if not n:
                continue
            for q in self.checked_reqs(n):
                v = self.literal_check(r, n, q)[0]
                if v == "differs":
                    out.append("%s contradicts %s" % (r, q))
                elif v == "missing":
                    out.append("%s does not state %s" % (r, q))
        return out

    def cart_has_gaps(self):
        return bool(self.cart_gaps())

    def cand_pool(self, n):
        out = []
        for r in self.item_listing_ids(n):
            if r not in self.full or r in self.gone or not self.buyable(r) or not self.within_money(n, r):
                continue
            unc = []
            bad = False
            for q in self.checked_reqs(n):
                v = self.literal_check(r, n, q)[0]
                if v == "differs":
                    bad = True
                    break
                if v != "met":
                    unc.append(q.split(":", 1)[0].strip())
            if bad:
                continue
            out.append((len(unc), self.price(r) or 0, r, unc))
        out.sort(key=lambda e: (e[0], e[1], e[2]))
        return out[:SET_ITEMS * 3]

    @soft(lambda: None)
    def opened_basket(self):
        state = (len(self.seen), len(self.full), self.version, len(self.gone), len(self.unavail), str(self.budget),
                 len(self.ref_values), sum(len(v) for v in self.views.values()))
        if self.set_memo is not None and self.set_memo[0] == state:
            return self.set_memo[1]
        self.set_memo = (state, None)
        N = len(self.items)
        pools = dict((n, self.cand_pool(n)) for n in range(1, N + 1))
        if any(not pools[n] for n in pools):
            return None
        idx = self.fit_index() if self.fit_pairs else None
        mains = sorted(set(m for p, m in self.fit_pairs))
        pmap = {}
        for p, m in self.fit_pairs:
            pmap[p] = m
        best = None
        total_budget = _num(self.budget) and self.brief.get("budget_scope") != "each"
        for combo in combos([pools[m][:SET_ITEMS] for m in mains], SET_COMBOS):
            chosen = dict(zip(mains, combo))
            used = set()
            pick = {}
            ok = True
            for n in range(1, N + 1):
                if n in chosen:
                    e = chosen[n]
                    if e[2] in used:
                        ok = False
                        break
                else:
                    m = pmap.get(n)
                    mc = chosen.get(m) if m else None
                    e = None
                    for c in pools[n]:
                        if c[2] in used:
                            continue
                        if mc is not None and self.fit_verdict(n, m, c[2], mc[2], idx) == "no":
                            continue
                        e = c
                        break
                    if e is None:
                        ok = False
                        break
                used.add(e[2])
                pick[n] = e
            if not ok:
                continue
            tot = sum(pick[n][1] * self.items[n - 1].qty for n in pick)
            if total_budget and tot > self.budget + SLACK:
                continue
            key = (sum(pick[n][0] for n in pick), tot)
            if best is None or key < best[0]:
                best = (key, dict(pick))
        if not best:
            return None
        (unc, tot), pick = best
        lines = dict((pick[n][2], self.items[n - 1].qty) for n in sorted(pick))
        txt = " + ".join("item %d: %s at %s%s" % (n, pick[n][2], gnum(pick[n][1]),
                                                  (" x%d" % self.items[n - 1].qty) if self.items[n - 1].qty > 1 else "")
                         for n in sorted(pick)) + " = " + gnum(tot)
        if unc:
            txt += ("; these properties are still unconfirmed on the listings' pages: "
                    + "; ".join("item %d: %s" % (n, ", ".join(pick[n][3])) for n in sorted(pick) if pick[n][3]))
        else:
            txt += "; each examined requirement is confirmed"
        res = (unc, tot, lines, txt)
        self.set_memo = (state, res)
        return res


class K11(object):
    def _init_k11(self):
        self.turn_start = {}

    def cart_total(self):
        held = self.held_lines()
        if not held:
            return None
        if any(self.price(r) is None for r in held):
            return None
        return sum(self.price(r) * (self.units_held(r) or 0) for r in held)

    def names_other_item(self, n, val):
        vr = roots(val)
        for j, o in enumerate(self.items, 1):
            if j != n:
                orr = roots(o.name)
                if orr and orr <= vr:
                    return True
        return False

    def strictly_proven(self, lid, n):
        if lid not in self.full or lid in self.gone or not self.buyable(lid) or not self.within_money(n, lid):
            return False
        reqs = self.item_reqs(n)
        if not reqs or any(is_conditional(r) for r in self.items[n - 1].live()):
            return False
        bound = any(n == p or n == m for p, m in self.fit_pairs)
        for q in reqs:
            v = q.split(":", 1)[1] if ":" in q else q
            if bound and self.names_other_item(n, v):
                continue
            if self.literal_check(lid, n, q)[0] != "met":
                return False
        it = self.items[n - 1]
        ck = self.category_key()
        row = self.rows.get(lid) or {}
        lab = str(row.get(ck, "")) if ck else ""
        left = (roots(it.name) | roots(it.shelf)) - roots(lab)
        if not left:
            return True
        tk = self.title_key()
        pool = roots(lab) | roots(str(row.get(tk, "")) if tk else "")
        for v in (attr_block_raw(self.full.get(lid)) or {}).values():
            if not isinstance(v, (dict, list)):
                pool |= roots(str(v))
        return left <= pool

    @soft(False)
    def turned_down(self, ids):
        pre = "|".join(sorted(ids)) + "#"
        return any(k.startswith(pre) and not v for k, v in self.verdicts.items())

    @soft(lambda: None)
    def proven_basket(self):
        N = len(self.items)
        ck = self.category_key()
        lists = {}
        for n in range(1, N + 1):
            lst = [r for r in self.seen if ck and self.owns_label(n, (self.rows.get(r) or {}).get(ck)) and self.strictly_proven(r, n)]
            lst.sort(key=lambda r: (self.price(r) or 0, r))
            if not lst:
                return None
            lists[n] = lst
        mains = sorted(set(m for p, m in self.fit_pairs))
        pmap = {}
        for p, m in self.fit_pairs:
            pmap[p] = m
        best = None
        total_budget = _num(self.budget) and self.brief.get("budget_scope") != "each"
        for combo in combos([lists[m][:SET_ITEMS] for m in mains], SET_COMBOS):
            if len(set(combo)) != len(combo):
                continue
            pick = dict(zip(mains, combo))
            used = set(combo)
            ok = True
            for n in range(1, N + 1):
                if n in pick:
                    continue
                m = pmap.get(n)
                dev = pick.get(m) if m else None
                e = next((r for r in lists[n] if r not in used and (dev is None or self.suits(n, m, r, dev))), None)
                if e is None:
                    ok = False
                    break
                pick[n] = e
                used.add(e)
            if not ok:
                continue
            tot = sum((self.price(pick[n]) or 0) * self.items[n - 1].qty for n in pick)
            if total_budget and tot > self.budget + SLACK:
                continue
            if best is None or tot < best[0]:
                best = (tot, pick)
        if not best:
            return None
        tot, pick = best
        lines = dict((pick[n], self.items[n - 1].qty) for n in sorted(pick))
        text = " + ".join("item %d: %s at %s%s" % (n, pick[n], gnum(self.price(pick[n]) or 0),
                                                   (" x%d" % self.items[n - 1].qty) if self.items[n - 1].qty > 1 else "")
                          for n in sorted(pick))
        return tot, lines, text

    @soft("")
    def set_check_line(self):
        r = self.opened_basket()
        if not r:
            return ""
        return ("[set check: only listings opened so far are compared, and listings not yet opened could cost less; this is "
                "the cheapest total with no requirement contradicted by the attributes"
                + (" (no part in it was seen to clash with its device)" if self.fit_pairs else "") + "] " + r[3])

    @soft("")
    def set_check_hold(self):
        if self.steps_left() <= self.reserve() + len(self.items) + 2:
            return ""
        r = self.opened_basket()
        if not r or r[0] > 0:
            return ""
        mk = self.cart_mark()
        if mk in self.set_held:
            return ""
        if set(r[2]) == set(self.held_lines()):
            return ""
        tot = self.cart_total()
        gaps = self.cart_gaps()
        dearer = self.pick() == "cheapest" and tot is not None and tot > r[1] + SLACK
        if not gaps and not dearer:
            return ""
        self.set_held.add(mk)
        self.trace.append({"kind": "set_check_hold", "cart": tot, "set": r[1], "gaps": gaps[:4]})
        t = HOLD
        if gaps:
            t += "; ".join(gaps[:4]) + ". "
        t += ("The following listings, among those opened so far, state each requirement"
              + ((", and together they cost %s, below the cart's %s" % (gnum(r[1]), gnum(tot))) if dearer else "")
              + ": " + r[3] + ". Cart these in place of the present lines, unless one of them fails a point this check "
              "cannot see (a note about compatibility, a stated condition, a value phrased on the page in other words); "
              "after that, order.")
        return t

    def proven_order(self):
        R = self.roles
        if (self.proven_tries >= PROVEN_TRIES or self.finished or self.checkouts or self.pick() != "any"
                or not R.get("cart") or not R.get("buy") or not self.convo):
            return False
        if R.get("ask") and not self.asked and self.msgs_free() > 0:
            return False
        if (self.steps_left() <= self.reserve() + len(self.items) + 3 or self.room() < PROVEN_SECS
                or self.pending_refs()):
            return False
        mark = (len(self.full), len(self.ref_values), self.version, len(self.change_notes), str(self.budget))
        if mark == self.proven_mark:
            return False
        self.proven_mark = mark
        pb = self.proven_basket()
        if not pb:
            return False
        tot, lines, text = pb
        if self.turned_down(list(lines)):
            self.trace.append({"kind": "proven_order_skipped", "why": "turned down", "ids": sorted(lines)})
            return False
        self.proven_tries += 1
        if set(lines) != set(self.held_lines()):
            if not self.cart_rewrite(dict(lines)):
                self.trace.append({"kind": "proven_order_cart_failed"})
                self.convo.append({"role": "user", "content": "[Terry] Opened listings state each requirement (%s), and Terry "
                                   "started carting them, but one cart call failed. Cart now: %s\n%s"
                                   % (text, self.cart_text(), self.status())})
                return True
        if self.finished:
            return True
        buy = R["buy"]
        oa = self.fill_free_text(dict(self.brief.get("buy_args") or {}),
                                 "Each carted line has every requirement stated on its own page: " + text, False)
        call = {"id": "proven%d" % self.proven_tries, "type": "function", "function": {"name": buy, "arguments": dump(oa)}}
        self.convo.append({"role": "assistant", "content": "[Terry] Terry's code found these opened listings to state each "
                           "requirement in their attribute blocks: %s = %s; it carted them and is now ordering." % (text, gnum(tot)),
                           "tool_calls": [call]})
        self.trace.append({"kind": "proven_order", "lines": lines, "total": tot})
        rep = self.dispatch([call])
        self.convo.append({"role": "tool", "tool_call_id": call["id"], "content": rep[0] + "\n" + self.status()})
        return True

    @soft("")
    def proven_swap(self):
        pk = self.pick()
        if (pk == "most" or pk == "least" or self.proven_swaps >= PROVEN_SWAPS
                or self.steps_left() <= self.reserve() + len(self.items) + 2 or self.pending_refs()):
            return ""
        pb = self.proven_basket()
        if not pb:
            return ""
        tot, lines, text = pb
        old = self.held_lines()
        if set(lines) == set(old) or not self.cart_gaps() or self.turned_down(list(lines)):
            return ""
        self.proven_swaps += 1
        if not self.cart_rewrite(dict(lines)):
            return ""
        self.trace.append({"kind": "proven_swap", "old": old, "new": lines})
        return text + " = " + gnum(tot)

    @soft("")
    def settle_hold(self):
        if self.settle_held:
            return ""
        if self.steps_left() <= max(SETTLE_STEPS, self.budget_steps // 4) or self.room() <= SETTLE_SECS:
            return ""
        bad = []
        for r in self.held_lines():
            if r not in self.full:
                continue
            n = self.owner_item(r)
            if n and any(self.literal_check(r, n, q)[0] == "differs" for q in self.checked_reqs(n)):
                bad.append(r)
        if not bad or not self.ask_nearest(True):
            return ""
        self.settle_held = True
        self.trace.append({"kind": "settle_hold", "lines": bad, "left": self.steps_left()})
        return SETTLE_HOLD_TEXT % {"ids": ", ".join(bad[:3]), "steps": self.steps_left()}

    @soft("")
    def needless_decline_hold(self):
        if "needless" in self.decline_checks:
            return ""
        cids = self.cart_ids()
        lines = [b for b in self.built if b in cids and not self.sold_out(b) and b not in self.gone
                 and (self.units_held(b) or 0) > 0]
        cov = set()
        for b in lines:
            cov |= set(self.listing_items(b))
        if not lines or len(cov) < len(self.items):
            return ""
        self.decline_checks.add("needless")
        self.trace.append({"kind": "needless_decline_hold", "lines": lines})
        return ("the cart has a line you assembled for every item asked for (%s); a decline earns zero while the store offers "
                "a fitting listing. Order this cart, repairing a line first when its record violates a requirement; decline "
                "ONLY if the searching showed that one of the requested items has no listing satisfying its requirements, "
                "and name that item in the reply text." % ", ".join(lines))

    @soft("")
    def line_pick(self):
        if (self.picked_lines or self.no_credit or self.room() < PICK_SECS or HEAVY not in self.judges(SOFT_SPEND)
                or self.steps_left() <= self.reserve() + 3):
            return ""
        held = self.held_lines()
        ranked = self.pick() != "any"
        cands = {}
        for r in held:
            lab = self.label_of(r)
            n = self.owner_item(r)
            pool = [x for x in self.full if lab and self.label_of(x) == lab and x not in held and self.buyable(x)
                    and x not in self.gone]
            if not ranked:
                lg = len(self.listing_gaps(r, n)) if n else 0
                pool = [x for x in pool if len(self.listing_gaps(x, n)) < lg] if (n and lg) else []
            pool.sort(key=lambda x: (self.price(x) is None, self.price(x) or 0, x))
            if pool:
                cands[r] = [r] + pool[:PICK_ROWS]
        if not cands:
            return ""
        self.picked_lines = True
        blocks = []
        for r, cl in cands.items():
            blocks.append("CARTED LINE %s; its candidates:\n" % r
                          + "\n".join("%s: %s" % (x, self.listing_record(x)[:900]) for x in cl))
        p = (LINE_PICK_PROMPT + "\n\nREQUEST:\n" + self.request + "\n\nLIVE REQUIREMENTS:\n" + self.ledger()
             + "\n\nCART:\n" + self.cart_text() + "\n\n" + "\n\n".join(blocks))
        a = self.ask_json(p, [HEAVY], 1500) or {}
        swaps = []
        for e in items_of(a.get("lines")):
            if not isinstance(e, dict):
                continue
            ln = str(e.get("line") or "").strip()
            pk = str(e.get("pick") or "").strip()
            if ln not in cands or pk not in cands[ln] or pk == ln:
                continue
            n = self.owner_item(ln)
            if not n or not self.item_money_ok(n, pk):
                continue
            if not ranked and not (len(self.listing_gaps(pk, n)) < len(self.listing_gaps(ln, n))):
                continue
            want = dict((x, (self.units_held(x) or 1)) for x in held if x != ln)
            want[pk] = self.units_held(ln) or self.items[n - 1].qty
            if not self.basket_money_ok(want):
                continue
            swaps.append((ln, pk, str(e.get("why") or "").strip()[:200]))
        self.trace.append({"kind": "line_pick", "swaps": swaps})
        if not swaps:
            return ""
        segs = ["%s in place of %s%s" % (pk, ln, (" (%s)" % why) if why else "") for ln, pk, why in swaps]
        return (HOLD + "a second reading of the listings already read chooses " + "; ".join(segs) + ". Make those swaps "
                "(remove the old line, then add the replacement at the quantity requested) and order; leave a line as it is "
                "ONLY if its named record plainly fails a requirement.")

    def dispatch(self, calls):
        calls = self.join_opens(calls)
        n = len(calls)
        self.turn_start = dict(self.cart)
        out = []
        for i, c in enumerate(calls):
            out.append(self.run_call(c, n, calls[i + 1:]))
        return out

    def run_call(self, c, n, later=()):
        allowed = self.offered()
        self.steps_at_call = self.steps_left()
        if c.get("joined"):
            return ("[Terry] This listing is in the comparison result given earlier in this turn, because Terry read them "
                    "jointly in a single step.")
        if self.finished:
            return "[Terry] Not executed: the episode has ended."
        tool = c["function"]["name"]
        R = self.roles
        if tool and tool == R.get("ask") and self.msg_args("x" * 200) is not None:
            return SKIP + "Terry's interviewer sends all shopper messages."
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
        cuts = []
        args = shape(args, self.schemas[tool], "args", faults, cuts)
        if faults:
            return SKIP + "; ".join(faults[:6])
        args, prob = self.map_ids(tool, args)
        if prob:
            return SKIP + prob + "."
        cvnote = ""
        cv = self.chain_page_view(tool, args)
        if cv:
            tool, args, cvnote = cv
        zcnote = ""
        if tool == R.get("cart"):
            args = self.whole_removal(args)
            args, zcnote = self.zero_count_fix(tool, args)
            kl = self.keep_lines_guard(tool, args, later, self.turn_start)
            if kl:
                return kl + " Cart now: " + self.cart_text()
        tail = []
        if zcnote:
            tail.append(zcnote)
        if cvnote:
            tail.append(cvnote.strip("\n"))
        for path, ml, val in cuts:
            key = (tool, path)
            if key in self.cut_noted:
                continue
            self.cut_noted.add(key)
            nm = path[5:] if path.startswith("args.") else path
            tail.append("[Terry] The argument %s takes no more than %d characters, so what the store got was '%s'; keep "
                        "later values of it within the limit, with the strongest words placed first." % (nm, ml, val))
        ct = self.compare_target(tool) if tool == R.get("compare") else None
        if ct and isinstance(args.get(ct), list) and self.cmp_fail:
            room = self.batch_room()
            if len(args[ct]) > room:
                left = args[ct][room:]
                args = dict(args)
                args[ct] = args[ct][:room]
                tail = ["[Terry] Only the first %d went out, as this store's comparison takes no more than %d; not yet "
                        "sent: %s." % (room, room, ", ".join(str(x) for x in left))]
        st = self.single_target(tool)
        if st and tool in (R.get("open"), R.get("offer")):
            v = args.get(st)
            if isinstance(v, str) and v.strip() in self.gone:
                return (SKIP + "this listing's sell-out was recorded at step %d and that listing stays gone. Put a different "
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
        if finishing and n > 1:
            return SKIP + "a finishing call must go alone; send it once the results of the other calls have been read."
        if not finishing and self.steps_left() <= self.reserve() and (R.get("buy") or R.get("refuse")):
            return (SKIP + "the last few steps are reserved for checking out (with the shopper approving the basket first when "
                    "that duty is due) or for a decline.")
        searchlike = tool not in self.plain_role_tools()
        fresh = False
        if searchlike:
            fresh = self.is_new_query(tool, args)
            a2 = self.first_look(tool, args, fresh)
            if a2 != args:
                tail.append("[Terry] This being a new search, Terry sent it for page one; you had sent %s." % dump(args))
                args = a2
            elif not fresh and not finishing:
                osk = self.offset_skip(tool, args)
                if osk:
                    args = osk[0]
                    tail.append(osk[1].strip("\n"))
        args = self.row_ids(tool, args)
        codet = self.code_tools()
        if self.fit_codes_dead and not finishing and tool in codet:
            a3, note, skip = self.drop_codes(tool, args)
            if skip:
                return skip
            if note:
                args = a3
                tail.append(note.strip("\n"))
        if tool == R.get("search") and self.codes_dead and self.names_code(args):
            return (SKIP + "code searches in this store returned nothing. Search with words and attribute values instead, "
                    "browse the category listing, and take the codes from the rows shown.")
        mark = mark_of(tool, args)
        if not finishing and tool != R.get("cart") and mark in self.recent[-3:]:
            return SKIP + "this exact call was just made, with no change since; the earlier result stands."
        if not finishing and tool in codet:
            ic = self.ignored_cap(tool, args)
            if ic:
                return ic
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
                    tail.append("[Terry] Left out as already shown in full with no change since: %s." % ", ".join(str(x) for x in drop))
                    args = dict(args)
                    args[ct] = keep
                    mark = mark_of(tool, args)
        hunt_txt = ""
        trail_txt = ""
        if fin_o and self.steps_left() > 2:
            t = self.settle_hold()
            if t:
                return t + self.footer()
            t = self.cheaper_hold(None, True)
            if t:
                return HOLD + t + self.footer()
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
                    txt = (HOLD + "a carted part is incompatible with the carted device: " + "; ".join(ff[:3])
                           + ". Replace the part, or else the device, and then order.")
                    ws = self.complete_baskets()
                    if ws:
                        txt += (" Whole sets already read, each satisfying all requirements and fitting together: "
                                + self.basket_text(ws) + ".")
                    return txt + self.footer()
            swapped = False
            ps = self.proven_swap()
            if ps:
                swapped = True
                self.show_links(self.cart_ids(), self.reserve() + 1)
                note = ("[Terry] Some carted line left a requirement unstated, or the whole cart was dearer than necessary, so "
                        "Terry swapped in opened listings whose attribute blocks state each requirement: " + ps)
                trail_txt = note + "\n" + trail_txt if trail_txt else note
                tail.insert(0, note)
            t = self.checkout_holds()
            if t:
                return HOLD + t + self.footer()
            if self.line_holds < 2 and self.steps_left() > self.reserve() + 2:
                lf = self.line_faults()
                if lf and all(k == "units" for k, _ in lf):
                    fixes = self.count_fixes()
                    if fixes:
                        lf = self.line_faults()
                        if not lf:
                            self.line_holds += 1
                            return (HOLD + "Terry adjusted the unit counts itself: " + "; ".join(fixes)
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
            cheap = self.cheaper_rows()
            fetched = fetched + cheap
            t = self.late_holds(fetched)
            if t:
                return t + self.footer()
            if not swapped:
                t = self.set_check_hold()
                if t:
                    return t + self.footer()
                t = self.line_pick()
                if t:
                    return t + self.footer()
            v = self.cart_check()
            if cheap and not v:
                v2, more = self.extra_cheaper_rounds()
                fetched = fetched + list(more or [])
                v = v2
            if v:
                out = BARE_HOLD
                if trail_txt:
                    out += trail_txt + "\n"
                if hunt_txt:
                    out += hunt_txt + " " + self.below_cart() + "\n"
                if fetched:
                    out += ("Terry fetched the attribute records of the listings below for this review, at one step apiece:\n"
                            + "\n".join(fetched) + "\n")
                return out + v + self.footer()
            if changes and self.spend > SOFT_SPEND:
                return (HOLD + "Terry has just fetched the cart's listings again:\n" + changes
                        + "\nOrder again once each line can be bought and the total fits the budget; otherwise swap that line "
                          "before ordering.")
        if fin_d and self.steps_left() > 5:
            t = self.basket_decline_hold()
            if t:
                return HOLD + t
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
            if not self.leads and "literal" not in self.decline_checks:
                ctx = self.decline_chain_round()
                lt = self.literal_hold()
                if lt:
                    return HOLD + (ctx + "\n" if ctx else "") + lt
            if self.leads and self.steps_left() > 4 and self.decline_blocks == 1 and len(self.items) >= 2:
                first = self.leads[0]
                bp = self.basket_plan(first)
                if bp:
                    self.decline_blocks = 2
                    self.cart_check(list(bp), HARD_SPEND)
                    if self.verdicts.get(self.check_key(list(bp))) and first in self.leads:
                        self.confirmed = first
                        self.trace.append({"kind": "confirmed_basket", "id": first, "basket": bp})
                        return (HOLD + "a separate check approved the whole basket (%s) as satisfying all requirements; make "
                                "the cart hold precisely those lines at the quantities requested, removing any other line, "
                                "then order; decline ONLY if some record plainly violates a requirement."
                                % ", ".join("%s x%d" % (r, u) for r, u in bp.items()))
            if self.leads and self.steps_left() > 4 and self.decline_blocks == 1:
                self.decline_blocks = 2
                first = self.leads[0]
                self.cart_check([first], HARD_SPEND)
                if self.verdicts.get(self.check_key([first])) and first in self.leads:
                    self.confirmed = first
                    self.trace.append({"kind": "confirmed", "id": first})
                    return (HOLD + "a second, separate check of the records of %s finds that it satisfies all requirements. "
                            "Cart it in the quantity asked and order; decline only when its record plainly breaks a "
                            "requirement." % first)
            t = self.needless_decline_hold()
            if t:
                return HOLD + t
            t = self.nearest_hold()
            if t:
                return HOLD + t
        if fin_d and 2 <= self.steps_left() <= LATE_ORDER_STEPS:
            t = self.late_rescue()
            if t or self.finished:
                return (t + (" [episode finished]" if self.finished else "")) if t else "[Terry] Not executed: the episode has ended."
        if fin_d and self.steps_left() <= LATE_ORDER_STEPS and not self.late_held:
            t = self.late_completion()
            if t or self.finished:
                return (t + (" [episode finished]" if self.finished else "")) if t else "[Terry] Not executed: the episode has ended."
        if fin_d and self.steps_left() <= 5 and not self.late_held:
            built = [r for r in self.built if not self.sold_out(r)]
            cids = self.cart_ids()
            missing = [r for r in built if r not in cids]
            if self.fresh_view() is None:
                missing += [r for r in built if r in self.dropped_ids and r not in missing]
            if len(built) >= len(self.items) and len(missing) + 1 <= self.steps_left():
                self.late_held = True
                self.trace.append({"kind": "late_decline_hold", "missing": missing})
                t = (HOLD + "a decline scores zero whenever a fitting listing exists, and running low on steps is no evidence "
                     "about whether one exists. ")
                if missing:
                    t += "Cart the lines %s again in the quantities asked and order." % ", ".join(missing)
                else:
                    t += "Order the cart unchanged."
                return t + (" Decline only if the searching proved that, for one of the requested items, no listing "
                            "satisfies its requirements, and in that case send the decline once more.")
        if (fin_d and not self.pushed and not self.leads and self.steps_at_call > self.budget_steps // 3
                and self.steps_left() > self.reserve() + len(self.items) + 2):
            self.pushed = True
            return (HOLD + "%d steps remain. Before giving up, try untried kinds of search: list every requested item's "
                    "category and page onward through that listing using the page token, or offset value, returned by the "
                    "store; try one-word queries of values and alternative names of the property; open any listing that "
                    "could fit. Decline only after all that." % self.steps_left())
        if fin_d and self.steps_left() <= LATE_ORDER_STEPS:
            t = self.late_proven_order()
            if t:
                return t + (" [episode finished]" if self.finished else "")
        if fin_d:
            if self.confirmed:
                t = self.complete_cart()
                if t:
                    return t + (" [episode finished]" if self.finished else "")
            if self.finished:
                return "[Terry] Not executed: the episode has ended."
            buy = R.get("buy")
            if buy and self.steps_left() >= 2 and self.ask_nearest(True) and self.nearest_cart():
                if self.finished or self.steps_left() <= 0:
                    return "[Terry] Not executed: the episode has ended."
                f = []
                oa = shape(self.nearest_buy_args(), self.schemas.get(buy), "args", f)
                if f:
                    oa = dict(self.brief.get("buy_args") or {})
                tool, args = buy, oa
                fin_o, fin_d = True, False
                self.trace.append({"kind": "nearest_order", "args": oa})
                tail.insert(0, "[Terry] Since the request accepts the closest listing if none satisfies all requirements, "
                               "Terry placed an order for the closest listings rather than declining: " + self.cart_text())
            else:
                args = self.quiet_fill(args)
        if fin_o:
            lq = self.late_question()
            if lq:
                return BARE_HOLD + lq + self.footer()
            lp = self.late_page_show()
            if lp:
                tail.insert(0, lp.strip("\n"))
            if self.finished:
                return "[Terry] Not executed: the episode has ended."
        if finishing and self.steps_left() >= 2:
            g, gt = self.gate("order" if fin_o else "end")
            if g != "go":
                return BARE_HOLD + gt + self.footer()
            if gt:
                tail.insert(0, "[Terry] " + gt)
        if fin_o and self.steps_left() >= 3 and not self.finished:
            self.show_links(self.cart_ids(), 1)
            if self.finished or self.steps_left() <= 0:
                return "[Terry] Not executed: the episode has ended."
        res = self.env(tool, args)
        if res is None:
            return "[Terry] The call broke down through a network or server problem; you may retry it."
        obs = res.get("observation")
        if ct:
            lst = args.get(ct)
            if isinstance(lst, list) and len(lst) > 1:
                d = dump(obs)
                if not any(isinstance(x, str) and x in d for x in lst):
                    self.cmp_fail = min(self.cmp_fail or 999, len(lst))
                    tail.append("[Terry] The comparison came back empty for %d listings. Send no more than %d per call."
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
        p0 = self.pages_read
        pnotes = []
        failed = is_failure(obs)
        if searchlike and not finishing:
            ap = self.auto_page(tool, args, obs)
            if ap:
                pnotes.append(ap.strip("\n"))
            if fresh:
                lp = self.late_page(tool, args, obs)
                if lp:
                    pnotes.append(lp.strip("\n"))
            sh = self.shorter(tool, args, obs)
            if sh:
                pnotes.append(sh.strip("\n"))
            if fresh and not pnotes:
                tr = self.top_rows(tool, args, obs)
                if tr:
                    pnotes.append(tr.strip("\n"))
            if tool in codet and not failed:
                fs = self.fit_scan(tool, args, obs)
                if fs:
                    pnotes.append(fs.strip("\n"))
            if not failed:
                br = self.big_row_reads(tool, args, obs)
                if br:
                    pnotes.append(br.strip("\n"))
            if self.pages_read > p0:
                sc = self.short_cheap_reads(tool, args, obs)
                if sc:
                    pnotes.append(sc.strip("\n"))
        if tool in readers and not finishing:
            ss = self.series_sweep()
            if ss:
                pnotes.append(ss.strip("\n"))
        if (not self.fit_codes_dead and not finishing and not failed and tool in codet
                and not (isinstance(obs, dict) and obs.get("error")) and not self.obs_listings(obs)
                and self.holds_req_code(args)):
            self.fit_codes_dead = True
            self.trace.append({"kind": "fit_codes_dead"})
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
        dear = ""
        if tool == R.get("cart"):
            before = dict(self.cart)
            self.note_cart(tool, args, res)
            if not self.finished:
                dear = self.pricey_line(before)
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
        if dear:
            out += "\n" + dear.strip("\n")
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
                out += ("\n[Terry] A zero count did not remove the line, since the tool adds the number given to the line; "
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
                out += "\n[Terry fetched the cart lines again following the rejection]\n" + t
        self.note_sold = False
        sw = self.swap_hint()
        if sw:
            out += "\n" + sw
        if tool == R.get("cart") and self.fit_pairs and not self.finished:
            ff = self.fit_faults()
            if ff:
                out += "\n[Terry] These lines are not yet compatible with each other: " + "; ".join(ff)
        if self.queued:
            out += "\n[notes seen during Terry's own calls] " + " | ".join(self.queued)
            self.queued = []
        if tool in readers:
            self.read_cache[mark] = (self.version, self.step, cacheable)
        if self.finished:
            out += " [episode finished]"
        return out

    def finish(self, mode="normal"):
        if self.finished or self.store_broke or self.steps_left() <= 0:
            return ""
        buy = self.roles.get("buy")
        fresh = self.fresh_view() is not None
        holding = bool(self.cart_ids())
        blind = (self.cart_touched and not fresh and self.steps_left() >= 2 and not self.dropped_ids
                 and (not self.checkouts or self.cart_changed_at > self.refused_at))
        changed = False
        if self.fit_pairs and buy:
            changed = self.forced_basket()
        if changed:
            holding = bool(self.cart)
        rescued = False
        short = not self.held_lines() or bool(self.missing_items())
        if not changed and buy and self.steps_left() >= 2 and not self.finished and short and self.ask_nearest(True):
            rescued = self.nearest_cart()
            if rescued:
                holding = True
        if self.finished or self.steps_left() <= 0:
            return ""
        whole = True
        if holding and len(self.items) >= 2:
            filled = rescued or self.fill_short()
            whole = filled or not self.can_decline()
        if whole and len(self.cart_ids()) >= 2:
            ids = self.cart_ids()
            self.fill_facts(ids, self.reserve())
            if self.steps_left() >= 3 and self.room() > 5:
                self.show_links(ids, self.reserve())
        elif holding and len(self.items) == 1 and self.steps_left() >= 3 and self.room() > 5:
            self.show_links(self.cart_ids(), self.reserve())
        if not whole:
            self.trace.append({"kind": "short_cart_decline"})
        if holding and whole and self.fit_pairs and buy and not self.finished:
            self.misfit_swap()
        if self.finished or self.steps_left() <= 0:
            return ""
        if whole and buy and (holding or blind):
            oa = self.nearest_buy_args() if self.near_made else dict(self.brief.get("buy_args") or {})
            f = []
            oa = shape(oa, self.schemas.get(buy), "args", f)
            if not f:
                if holding and self.steps_left() >= 3 and self.room() > 5:
                    self.show_links(self.cart_ids(), self.reserve())
                    if self.finished or self.steps_left() <= 0:
                        return ""
                self.forced_offer_views()
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
            msg += ("\n\nTerry has already searched for every item and read part of each item's listings, one step per "
                    "read:\n" + mv + "\nJudge these listings by the requirements, and search more only where something "
                    "is still lacking.\n\n" + self.status())
        else:
            msg += ("\n\nBegin with a search for each item requested, sought within its own category, and read "
                    "several listings completely, so that the interviewer can then tell which properties such listings "
                    "show.\n\n" + self.status())
        self.convo = [{"role": "system", "content": GUIDE}, {"role": "user", "content": msg}]
        self.work(TURN_CAP, HARD_SPEND)
        if not self.finished:
            t = self.finish("again")
            if t and not self.finished and not self.store_broke and not self.no_credit and self.steps_left() > 1:
                self.convo.append({"role": "user", "content": "[Terry] Terry attempted to finish and " + t
                                   + "\nFollow the store's answer and then finish.\n" + self.status()})
                self.work(self.turns + 10, HARD_SPEND + 0.01)
        t = ""
        if not self.finished:
            t = self.finish("normal")
        extra = 0
        while (extra < 2 and t.startswith("the decline was rejected") and not self.finished and not self.store_broke
               and not self.no_credit and self.steps_left() > 1):
            extra += 1
            self.trace.append({"kind": "extra_round", "round": extra, "left": self.steps_left()})
            self.convo.append({"role": "user", "content": "[Terry] Terry attempted to finish and " + t
                               + "\nFollow the store's answer and then finish.\n" + self.status()})
            self.work(self.turns + EXTRA_ROUND_TURNS, HARD_SPEND + EXTRA_ROUND_SPEND)
            t = ""
            if extra == 1 and not self.finished:
                t = self.finish("normal")
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "spend": round(self.spend, 5),
                           "calls": self.model_calls})
        return list(self.trace)

    def turn_loop(self, cap, spend_cap):
        idle = 0
        quiet = False
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
            if self.proven_order():
                if self.finished:
                    return
                continue
            if (self.nearest_read is None and self.clock() >= WIND_DOWN
                    and (not self.cart_ids() or self.missing_items()) and self.room() >= NEAREST_SECS + CALL_FLOOR):
                self.ask_nearest(True)
            self.trim()
            self.turns += 1
            off = set(self.offered())
            tools = [t for t in self.tools if t["function"]["name"] in off]
            if quiet and HEAVY in self.judges(SOFT_SPEND):
                models = [HEAVY, LIGHT]
            else:
                models = [LIGHT, SPARE]
            msg = self.think(self.convo, models, DIRECTOR_CAP, tools)
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
            quiet = not calls
            if not calls:
                idle += 1
                ro = self.reopen_search()
                if ro:
                    idle = 0
                self.convo.append({"role": "assistant", "content": text or "(no call)"})
                self.convo.append({"role": "user", "content": "[Terry] Use one of the tools so the work goes on.\n"
                                   + (ro + "\n" if ro else "") + self.status()})
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
                self.convo.append({"role": "user", "content": (s or "[Terry] The interviewer had no question to put.")
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


class KXA(object):
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
        if res is None:
            return None
        obs = res.get("observation")
        self.observe(cmp_, obs)
        self.remember_view(cmp_, obs)
        recs = self.obs_listings(obs)
        if recs:
            self.facts[cmp_] = self.facts.get(cmp_) is True or any(attrs_of(r) is not None for _, r in recs)
            self.cmp_ok = max(self.cmp_ok, len(recs))
        else:
            self.cmp_fail = min(self.cmp_fail or 999, len(picks))
        self.trace.append({"kind": "widen", "ids": picks, "opened": len(self.full) - before})
        if recs and self.facts[cmp_] is False:
            return self.widen_singly()
        if len(self.full) == before and len(picks) > 2 and self.steps_left() > 12:
            f2 = []
            a2 = shape({tgt: picks[:2]}, self.schemas.get(cmp_), "args", f2)
            if not f2:
                r2 = self.env(cmp_, a2)
                if r2 is not None:
                    o2 = r2.get("observation")
                    self.observe(cmp_, o2)
                    self.cmp_ok = max(self.cmp_ok, len(self.obs_listings(o2)))
        return None




class KXB(object):


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
                known |= set(w for w in spelled(" ".join(RX_ALNUM.findall(str(v).lower()))) if len(w) >= 4)
        for n, q, ans in sent:
            qw = spelled(" ".join(RX_ALNUM.findall(RX_Q_FILLER.sub(" ", q).lower())))
            aw = spelled(" ".join(RX_ALNUM.findall(str(ans).lower())))
            if RX_HAS_DIGIT.search(str(ans)) or aw & known or aw & qw or RX_HOW_ASK.search(str(ans)):
                return False
        return True

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
                    self.done_log.append("(Terry: none of the record lines written after the cart's latest change at step %d "
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
            self.done_log.append("step %d: %s %s -> %s (cart then: %s)" % (self.step, tool, dump(args), shown[:300], self.cart_mark()))
            self.done_steps.append(self.step)
            said.append("%s %s -> %s" % (tool, dump(args), shown[:400]))
        return "go", (GATE_LEAD + "\n" + "\n".join(said)) if said else ""

    def _open_for_audit(self, op):
        cmp_ = self.roles.get("compare")
        tgt = self.compare_target(cmp_) if cmp_ else None
        if cmp_ and tgt and not self.compare_bare():
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

    def join_opens(self, calls):
        op = self.roles.get("open")
        cmp_ = self.roles.get("compare")
        ot = self.single_target(op) if op else None
        ct = self.compare_target(cmp_) if cmp_ else None
        if not (op and ot and cmp_ and ct and op != cmp_) or self.compare_bare():
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

    def quiet_fill(self, args):
        tool = self.roles.get("refuse")
        sch = self.schemas.get(tool) or {}
        props = self._props(tool)
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        out = dict(args)
        for k, p in props.items():
            if not isinstance(p, dict) or "string" not in kinds(p) or p.get("enum"):
                continue
            v = out.get(k)
            if v is None and k not in req:
                continue
            v = str(v or "")
            if not v or any(r in v for r in self.rows) or self.brief.get("quiet_decline") is not False:
                ml = p.get("maxLength")
                out[k] = QUIET_TEXT[:int(ml) if _num(ml) else 200]
        return out



# ==================================================================================================================
# TERRY'S ASKING ENGINE (Singles). The episode calls ready_to_ask(), interview(), late_question() and msg_args().
# ==================================================================================================================

ASK_PAGES = 3             # opened pages per item before the questions start (else a quarter of the steps)
ASK_BY = 165.0            # seconds: no interview starts after this
QUESTION_END = 210.0      # seconds: no question is sent after this
LATE_BY = 190.0           # seconds: no question just before a checkout after this
LATE_STEPS = 8            # steps a question just before a checkout needs left
ASK_PLANNERS = [LIGHT, HEAVY]   # the planner's models, in turn
# details never asked about: a listing's choice of variants and facts about the offer rather than the product
NOT_ASKED = re.compile(r'^(options?|variants?|variations?|warranty( \w+)?|shipping( \w+)?)$', re.I)
TEXT_FIELD = re.compile(r'(?:^|_)(?:texts?|body|messages?|msg|notes?|contents?|questions?|say|chat|reply|ask)(?:_|$)', re.I)
NOT_TEXT = re.compile(r'(?:^|_)(?:id|ids|type|format|code|lang|language|kind|mode|session|recipient|channel|to|user)$', re.I)
# a reply with no view on what was asked ("no preference on the colour"), which names the property all the same
NO_VIEW = re.compile(r"\b(?:no|any|don ?'?t|doesn ?'?t|do not|does not|not)\b(?:\W+\w+){0,3}?\W+(?:prefer\w*|mind|matter\w*|care|"
                     r"fuss\w*|particular|bothered|picky|sure|idea)\b|\bwhatever\b|\bhaven ?'?t (?:thought|considered)\b", re.I)

ASK_PLAN = """You help a shop assistant prepare its next messages to a shopper. The shopper holds a few wishes they left unsaid, each about one property of one requested product. Each message names ONE property of one product, and messages are few, so the order matters. A wish is told only when a message names that exact property; one the shopper has no view on gets a "don't mind", and one they never think about gets a puzzled reply, so every message spent on such a property is lost. You see the request, the products with what the request already settles for each, the messages sent with the shopper's replies, and CANDIDATES: the properties the shop's own pages record for the product to ask about now, the most widely recorded first, with PAGES: on how many of the pages read each one stands, and some of its values. Return ONE JSON object and nothing else:
{"first": up to 2 candidates that a remark in the request points at without giving their value, likeliest first ([] when no remark clearly does), "skip": candidates that are no property of the product itself (codes, model or part numbers, facts about the offer, stock, delivery, warranty or the shop; the condition it is sold in, new or used, is a property of it), "rank": up to 6 of the first 12 CANDIDATES, likeliest wish first: one that most of the pages record and whose values differ between them (so it tells the listings apart) before one whose pages all agree or that few record; among those, the order a buyer of this kind of product chooses by (its type or style, then form or fit, size or capacity, material, colour; a detail most buyers never think about last or not at all), "extra": when CANDIDATES name fewer than 6 properties of the product itself, up to 4 more that people choose this kind of product by and CANDIDATES lack, each in a shop's usual short wording, else [], "why": a few words}
- A remark points at a property when it is clearly about it: how the thing looks, what it is made of, how it fits or works, what it holds or runs on, the person or place it serves. Read the story for them: a worry or a past mishap names the property that went wrong (a lid that leaked last time: its closure; a cable that fell short: its length), being seen or photographed points at its looks (colour, pattern, finish), the place it goes or must suit at its style, size or mounting, the person meant to use or wear it at the fit for them (gender, age, size), and what it holds or comes in at its flavour, type or pack. Put that property in "first" under the CANDIDATES name that means it. Background with no such link points at nothing.
- Write every property exactly as CANDIDATES write it. Never put in "first" a property the request already gives a value for or one that was already asked. A candidate marked [not the stated ...] is another property than that requirement, whatever words its name shares with it."""

ASK_READ = """You read a shopper's replies to an assistant's questions in an online shop simulation and list what they told. A reply that only says they have no preference, did not follow the question, or wants it put again tells nothing. Return ONE JSON object and nothing else:
{"needs": [{"item": the item number the wish is about, "need": "property: wanted value", "points": true when that value is to be read off another listing's page, else false}], "drop": [{"item": item number, "property": a requirement the shopper now gives up}], "free": [{"item": item number, "property": a property a reply says they do not mind}], "budget": a new total spending limit a reply gives, as a number, or null}
- Name a need by the property the reply itself names, which may differ from the one asked about; several replies about one property make one need. The value is what a listing could state, in the reply's own words with its numbers, units and bounds ("at least 30 cm", "oval", "not wicker"): leave out the framing around it ("only", "must be", "and nothing else", "matching").
- A value given by pointing at another listing (the same as, unlike or near the one on some shop's product, that product picked out by a few of its properties) is ONE need on the property the reply names, with "points": true; its value is the reply's words from the comparison to the end of that description, unchanged. The properties describing that other product are not needs of this item.
- Roundabout or folksy wording that names a value still states that value; refusing to trade or give a value up states it as wanted; write "not X" only when the reply rejects X itself."""


def _ask_key(key):
    """A property name in plain words: camel case and underscores split into words."""
    return ' '.join(re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', str(key)).replace('_', ' ').lower().split())


def _ask_words(text):
    return {w for w in re.findall(r'[a-z0-9]+', str(text).lower()) if len(w) > 2}


def _ask_overlaps(prop, stated):
    """True when the property is one the request already settles (same name, or a vaguer form of it)."""
    mine = _ask_words(prop)
    return any(mine and _ask_words(x) and (mine == _ask_words(x) or mine <= _ask_words(x)) for x in stated)


def _ask_list(value):
    if isinstance(value, str):
        try:
            got = json.loads(value) if value.strip().startswith('[') else None
        except ValueError:
            got = None
        value = got if isinstance(got, list) else [value] if value.strip() else []
    return [_ask_key(x) for x in value if isinstance(x, str) and x.strip()] if isinstance(value, list) else []


def _ask_filler(spec):
    """A value for a required field of the message tool that is not the text: its first allowed, fixed or default
    value, else the smallest number, false, or a short word; None when the schema gives no safe one."""
    if not isinstance(spec, dict):
        return None
    if isinstance(spec.get('enum'), list) and spec['enum']:
        return spec['enum'][0]
    for word in ('const', 'default'):
        if word in spec:
            return spec[word]
    got = kinds(spec)
    if 'integer' in got or 'number' in got:
        return spec['minimum'] if _num(spec.get('minimum')) else 1
    if 'boolean' in got:
        return False
    return 'shopper'[:spec.get('maxLength') or 7] if 'string' in got else None


def _ask_core(name, item_words):
    """A property's own words: its name less the words of the product's name (a product P's "P A B" is its "A B")."""
    words = [w for w in re.findall(r'[a-z0-9]+', str(name).lower()) if len(w) > 2]
    core = [w for w in words if w not in item_words]
    return core or words


def _ask_same(a, b, item_words):
    """Two names of one property: the same first own word, and one name's words inside the other's ("A" and "A B"
    are one; "B" and "A B" stay two)."""
    ca, cb = _ask_core(a, item_words), _ask_core(b, item_words)
    return bool(ca and cb) and ca[0] == cb[0] and (set(ca) <= set(cb) or set(cb) <= set(ca))


def _ask_item_words(name):
    """The words of a product's name, singular and plural."""
    out = set()
    for w in re.findall(r'[a-z0-9]+', str(name).lower()):
        out |= {w, w[:-1] if w.endswith('s') else w + 's', w[:-2] if w.endswith('es') else w}
    return out


def _ask_noun(name):
    """The product as the request names it (a name given as alternatives, "A or B", reads as two products: the first
    names it; a bracketed gloss is no part of it)."""
    plain_name = ' '.join(re.sub(r'\([^)]*\)?', ' ', str(name)).split()) or str(name)
    return re.split(r'\s+or\s+', plain_name)[0] or plain_name or 'item'


class TerryAsk(object):
    """Terry's questions to the shopper (Singles): each message names one property of one product, the highest
    ranked one not asked yet; products take turns, one whose wish is still untold first. The ranking is a model's
    reading of the properties the item's opened pages record (most widely recorded first, with their values), hints
    in the request's story first. The replies are read once into the items' requirements."""

    def ask_state(self):
        state = getattr(self, '_asking', None)
        if state is None:
            n = len(self.items)
            state = {'plans': {}, 'done': {i: [] for i in range(1, n + 1)}, 'count': {i: 0 for i in range(1, n + 1)},
                     'told': set(), 'spent': set(), 'qa': []}
            self._asking = state
        return state

    def ask_note(self, kind, **data):
        data['kind'] = kind
        self.trace.append(data)

    def ask_together(self, jobs):
        """Model-only jobs (no store calls) side by side; their results in order (None for one that failed)."""
        if len(jobs) <= 1:
            return [job() for job in jobs]
        from concurrent.futures import ThreadPoolExecutor
        out = []
        with ThreadPoolExecutor(max_workers=min(6, len(jobs))) as pool:
            futures = [pool.submit(job) for job in jobs]
            for future in futures:
                try:
                    out.append(future.result())
                except Exception as exc:
                    self.ask_note('ask_job_crash', error=('%s: %s' % (type(exc).__name__, exc))[:300])
                    out.append(None)
        return out

    # ---------------------------------------------------------------- when to ask

    def ready_to_ask(self):
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        if self.steps_left() <= 8 or self.clock() > ASK_BY:
            return False
        read = [len(self.item_records(n)) for n in range(1, len(self.items) + 1)]
        if read and all(r >= ASK_PAGES for r in read):
            return True
        return self.step >= max(8, self.budget_steps // 4)

    # ---------------------------------------------------------------- the message tool

    def msg_args(self, text):
        """The message tool's arguments for one text: its text field (one named like a message before any other free
        text field, a field named like an id or an address last; else a list of texts or an object holding one),
        other required fields filled with a safe value."""
        schema = self.schemas.get(self.roles.get('ask') or '') or {}
        schema = schema if isinstance(schema, dict) else {}
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        required = [k for k in schema.get('required') or [] if isinstance(k, str)] if isinstance(schema.get('required'), list) else []

        def plain_key(key):
            return _ask_key(key).replace(' ', '_')
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in kinds(v) and 'enum' not in v and 'const' not in v]
        args = {}
        named = [k for k in texts if not NOT_TEXT.search(plain_key(k))]
        if named:
            field = max(named, key=lambda k: (bool(TEXT_FIELD.search(plain_key(k))), k in required,
                                              props[k].get('maxLength') if _num(props[k].get('maxLength')) else 10 ** 6,
                                              -texts.index(k)))
            args[field] = text
        else:
            for key, spec in props.items():
                if not isinstance(spec, dict):
                    continue
                inner = spec.get('properties') if isinstance(spec.get('properties'), dict) else {}
                inner_texts = [k for k, v in inner.items() if isinstance(v, dict) and 'string' in kinds(v) and 'enum' not in v]
                if 'array' in kinds(spec):
                    args[key] = [text]
                    break
                if inner_texts:
                    pick = max(inner_texts, key=lambda k: (not NOT_TEXT.search(plain_key(k)), bool(TEXT_FIELD.search(plain_key(k)))))
                    args[key] = {pick: text}
                    break
        if not args and texts:
            args[texts[0]] = text  # only fields named like an id or an address take text: the first of them
        if not args:
            return None
        for key in required:
            if key in args:
                continue
            value = _ask_filler(props.get(key))
            if value is None:
                return None
            args[key] = value
        faults = []
        args = shape(args, schema, 'args', faults)
        return None if faults or not isinstance(args, dict) else args

    # ---------------------------------------------------------------- what to ask

    def ask_candidates(self, number):
        """The properties the item's opened pages record, the most widely recorded first (then the ones whose values
        differ most), with on how many pages each stands and some of its values. A property spelt two ways counts
        once; a field holding several choices in one value, a long text or a value unique to every page (an
        identifier) is no candidate."""
        ids = list(self.item_records(number))
        per = {}
        for ident in ids:
            for key, value in (attrs_of(self.full.get(ident)) or {}).items():
                name = _ask_key(key)
                if name and value not in (None, '') and not isinstance(value, (dict, list)):
                    per.setdefault(name, {}).setdefault(ident, ' '.join(str(value).lower().split())[:80])
        rows = []
        for name, vals in per.items():
            texts = list(vals.values())
            joined = sum(' / ' in t for t in texts)
            mean = sum(len(t) for t in texts) / len(texts)
            if (joined >= 2 and joined * 3 >= len(texts)) or mean > 40 or \
                    (len(texts) >= 3 and len(set(texts)) == len(texts) and mean > 12):
                continue
            rows.append((name, vals))
        rows.sort(key=lambda r: (-len(r[1]), -len(set(r[1].values())), len(r[0].split()), r[0]))
        kept = []
        own = _ask_item_words(self.items[number - 1].name)
        for name, vals in rows:
            # one property under several names counts once, under its most recorded name
            if not any(_ask_same(name, other, own) for other, _ in kept):
                kept.append((name, vals))
        stats = {name: 'on %d of %d pages: ' % (len(vals), len(ids)) + ' | '.join(v[:30] for v in list(dict.fromkeys(vals.values()))[:4])
                 for name, vals in kept}
        return [n for n, _ in kept], stats

    def ask_plan(self, number, cands, stats):
        """The model's reading of one item's candidates: the ones a remark in the request points at (asked first), its
        ranking of the most recorded ones, any that is no property of the product (never asked), and a few more."""
        item = self.items[number - 1]
        stated = [m.split(':', 1)[0].strip() for m in item.musts if m.split(':', 1)[0].strip()]
        pages = {}
        for c in cands[:16]:
            near = [x for x in stated if _ask_words(x) & _ask_words(c) and _ask_words(x) != _ask_words(c)]
            pages[c] = stats.get(c, '') + ''.join(' [not the stated %s]' % x for x in near[:1])
        st = self.ask_state()
        history = '\n'.join('Q: %s\nA: %s' % (q, a) for _, q, a in st['qa']) or '(none yet)'
        products = '\n'.join('%d. %s - settled by the request: %s' % (n, it.name, '; '.join(it.musts) or 'nothing')
                             for n, it in enumerate(self.items, 1))
        hints = [str(x) for x in items_of(self.brief.get('ask_first')) if str(x).strip()]
        prompt = (ASK_PLAN + '\n\nREQUEST:\n%s\n\nPRODUCTS:\n%s' % (self.request, products)
                  + ('\n\nTHE REQUEST ASKS TO BE ASKED ABOUT: %s' % dump(hints) if hints else '')
                  + '\n\nMESSAGES SENT AND REPLIES:\n%s\n\nMessages left: %d.' % (history, self.msgs_free())
                  + '\n\nPRODUCT TO ASK ABOUT NOW: %s\nCANDIDATES: %s\nPAGES: %s' % (item.name, dump(cands[:24]), dump(pages)))
        got = self.ask_json(prompt, ASK_PLANNERS, 2000) or {}
        plan = {k: _ask_list(got.get(k)) for k in ('first', 'skip', 'rank', 'extra')}
        self.ask_note('ask_plan', item=number, plan={k: v[:6] if isinstance(v, list) else v for k, v in plan.items()}, cands=cands[:10])
        return plan

    def ask_told_about(self, number, prop):
        mine = _ask_words(prop)
        return any(mine and mine <= _ask_words(t.split(':', 1)[0]) for t in self.items[number - 1].told)

    def ask_fresh(self, number, prop):
        """A property still worth a message about the item: none the request settles, asked or told already."""
        st = self.ask_state()
        own = _ask_item_words(self.items[number - 1].name)
        stated = {m.split(':', 1)[0].strip().lower() for m in self.items[number - 1].musts if m.split(':', 1)[0].strip()}
        return bool(prop) and not NOT_ASKED.search(prop) and not _ask_overlaps(prop, stated) \
            and not any(_ask_same(prop, x, own) for x in st['done'][number]) and not self.ask_told_about(number, prop)

    def ask_prepare(self):
        """The plans of the items the messages will reach (products take turns), side by side."""
        st = self.ask_state()
        want = [n for n in self.ask_turns() if n not in st['plans']][:max(1, self.msgs_free())]
        jobs = []
        for number in want:
            names, stats = self.ask_candidates(number)
            cands = [c for c in names if self.ask_fresh(number, c)]
            jobs.append(lambda n=number, c=cands, s=stats: (n, self.ask_plan(n, c, s)))
        for got in self.ask_together(jobs):
            if isinstance(got, tuple) and isinstance(got[1], dict):
                st['plans'][got[0]] = got[1]

    def ask_order(self, number):
        """The item's properties in the order to ask them: hinted ones first (dropped when no opened page records them
        and the pages record three or more), then the planner's ranking over the most recorded, then the rest as
        recorded, then the planner's extras."""
        st = self.ask_state()

        def fresh(p):
            return self.ask_fresh(number, p)
        names, stats = self.ask_candidates(number)
        cands = [c for c in names if fresh(c)]
        if number not in st['plans']:
            st['plans'][number] = self.ask_plan(number, cands, stats)
        plan = st['plans'][number]
        skip = set(plan.get('skip') or [])
        first = [p for p in plan.get('first') or [] if fresh(p) and p not in skip]
        extra = [p for p in plan.get('extra') or [] if fresh(p) and p not in skip][:4]
        recorded = [c for c in cands if c not in skip]
        ranked = [p for p in plan.get('rank') or [] if p in recorded[:12]]
        recorded = list(dict.fromkeys(ranked + recorded))
        if len(recorded) >= 3:
            first = [p for p in first if p in recorded]
        return list(dict.fromkeys(first + recorded + extra))

    def ask_flat(self, number):
        """An item whose opened pages leave nothing to choose by: one listing found, or pages that agree on every
        property they record (a wish would rule none of them out)."""
        ids = list(self.item_records(number))
        if len(ids) < 2:
            return True
        per = {}
        for ident in ids:
            for key, value in (attrs_of(self.full.get(ident)) or {}).items():
                if value not in (None, '') and not isinstance(value, (dict, list)):
                    per.setdefault(_ask_key(key), set()).add(' '.join(str(value).lower().split()))
        return not any(len(v) >= 2 for k, v in per.items() if not NOT_ASKED.search(k))

    def ask_turns(self):
        """The products in the order they take turns: the listed order, one with nothing to choose by last."""
        numbers = list(range(1, len(self.items) + 1))
        if len(numbers) == 1:
            return numbers
        flat = {n: self.ask_flat(n) for n in numbers}
        return sorted(numbers, key=lambda n: (flat[n], n))

    def ask_next_item(self):
        """The item asked next: products take turns, one whose wish is still untold first."""
        st = self.ask_state()
        turns = self.ask_turns()
        live = [n for n in turns if n not in st['spent']]
        if not live:
            return 0
        return min(live, key=lambda n: (n in st['told'], st['count'][n], turns.index(n)))

    def ask_one(self, floor=6):
        """Send one question (one property of one product) and note whether its reply told the wish."""
        st = self.ask_state()
        while not self.finished:
            number = self.ask_next_item()
            if not number:
                return []
            order = self.ask_order(number)
            if not order:
                st['spent'].add(number)
                continue
            prop = order[0]
            st['done'][number].append(prop)
            text = 'For the %s, what %s would you like?' % (_ask_noun(self.items[number - 1].name), prop)
            if not self.msg_args(text):
                return []  # the tool takes no text this way, whatever the text
            batch = self.send([(number, prop, text)], floor)
            if not batch:
                return []
            st['count'][number] += 1
            st['qa'] += batch
            # a wish names its property ("where X is concerned, only Y"); one with no view on it names it too
            reply = batch[0][2]
            if _ask_words(prop) and _ask_words(prop) <= _ask_words(reply) and not NO_VIEW.search(reply):
                st['told'].add(number)
            return batch
        return []

    # ---------------------------------------------------------------- reading the replies

    def ask_read(self, sent, told):
        """Read the replies once into the items' requirements (appending each new one to `told`), the properties the
        shopper does not mind, requirements given up and a new budget."""
        if not sent:
            return
        prompt = ASK_READ + '\n\nITEMS:\n' + '\n'.join('item %d: %s' % (n, it.name) for n, it in enumerate(self.items, 1))
        prompt += '\n\nQUESTIONS AND REPLIES:\n' + '\n'.join('(about item %d) Q: %s\nA: %s' % (n, q, a) for n, q, a in sent)
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        needs = [e for e in items_of(got.get('needs')) if isinstance(e, dict) and ':' in str(e.get('need') or '')]
        got['needs'] = needs
        budget = self.budget
        self.fold_reading(got, told)
        if self.budget != budget:
            told.append('budget now %s' % self.budget)
        self.ask_note('ask_read', told=told[:8])

    # ---------------------------------------------------------------- the interview and the late question

    def interview(self):
        self.asked = True
        if self.msgs_free() <= 0 or not self.msg_args('For the item, what size would you like?'):
            return ''
        self.widen()
        st = self.ask_state()
        self.ask_prepare()
        sent = []
        while self.msgs_free() > 0 and not self.finished and self.steps_left() > 6 and self.clock() < QUESTION_END:
            batch = self.ask_one()
            if not batch:
                break
            sent += batch
        told = []
        self.ask_read(sent, told)
        heard = [n for n in range(1, len(self.items) + 1) if any(t not in self.items[n - 1].dropped for t in self.items[n - 1].told)]
        chain = self.early_trail()
        searched = self.value_search(heard) if heard else ''
        if told:
            self.version += 1
        lines = ["[Terry] Terry's questions to the shopper, one step each:"]
        for _, q, a in sent:
            lines += ['Q: ' + q, 'A: ' + a]
        lines.append('New firm requirements from these answers: %s. Apply them to every pick (search again with the new '
                     'values where needed).' % ('; '.join(told) if told else 'none'))
        pointed = self._ref_lines(told)
        if pointed:
            lines.append('These take their value from another listing: %s. Find that listing, open its page in full, and '
                         'open every listing its page refers to; the last page gives the value, and the order counts only '
                         'after all of them are opened.' % ' | '.join(pointed))
        if chain:
            lines.append(chain)
        if searched:
            lines.append(searched)
        if self.queued:
            lines.append('[notes seen meanwhile] ' + ' | '.join(self.queued))
            self.queued = []
        return '\n'.join(lines)

    def late_question(self):
        """Before a checkout, one more question with a message the order itself does not need (the messages kept for a
        step that comes with ending without a purchase are free once the agent buys). Returns the summary for the
        loop model when the answer adds a requirement, else ''."""
        if self.late_asked or not self.roles.get('ask') or self.msgs_used >= self.msg_limit:
            return ''
        if any(d.get('stage') == 'order' and (d.get('msgs') or 0) >= 1 for d in self.duties):
            return ''
        if self.steps_left() < LATE_STEPS or self.clock() > LATE_BY or not self.judges():
            return ''
        self.late_asked = True
        before = {n: len(it.told) for n, it in enumerate(self.items, 1)}
        batch = self.ask_one(LATE_STEPS - 2)
        if not batch:
            self.ask_note('late_question', sent=False)
            return ''
        told = []
        self.ask_read(batch, told)
        _, question, answer = batch[0]
        self.ask_note('late_question', q=question, a=answer[:200], told=told)
        if not told:
            return ''
        self.version += 1
        out = ["[Terry] Before checking out, Terry asked the shopper once more (one step):", 'Q: ' + question, 'A: ' + answer,
               'New firm requirements: ' + '; '.join(told) + '.',
               'The checkout did not happen. Check every cart line against them (search again with the new values '
               'where needed), fix the cart, then check out.']
        pointed = self._ref_lines(told)
        if pointed:
            out.append('These take their value from another listing: %s. Find that listing, open its page in full, and '
                       'open every listing its page refers to; the last page gives the value.' % ' | '.join(pointed))
        grown = [n for n, it in enumerate(self.items, 1) if len(it.told) > before[n]]
        searched = self.value_search(grown) if grown else ''
        if searched:
            out.append(searched)
        return '\n'.join(out)


class Episode(TerryAsk, KXB, KXA, K11, K10, K09, K08a, K08, K07, K06, K05, K04, K03, K02, E1, E2, E3, E4, E5, E6, E7, E8, E9):
    def __init__(self, problem):
        E1.__init__(self, problem)
        self._init_k03()
        self._init_k11()


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
