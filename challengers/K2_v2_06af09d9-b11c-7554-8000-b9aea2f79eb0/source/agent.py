"""K2: a shopping agent for one shopper inside an ORO store world.

Our fork of the public Cairn v2 agent (race-17 winner), which builds on the public Halo v2 and Know Thyself agents.
The code is restructured throughout and is our own form of that design, with its behaviour kept: a light model walks the
store with the world's own tools, while code keeps the ledger, plans the shopper interview, hunts the cheapest fitting
rows, follows requirements that point at other listings, and checks the cart before an order and the evidence before a
decline. A heavy model reads the request, plans the questions and checks the final cart.

Beyond the base:
- with only two messages left for questions on a one-item request, the first question already names three planned
  properties (the base asked one, then three). Live replies of races 15-17 show a three-property list loses little against
  a single question when the shopper refuses lists and covers more planned properties when it takes them.
- a listing's own page can hand one of its properties to another listing ("its socket is to be taken from the ... CPU, not
  from this item"; "for color, check the ... dash cam, as this item's color is taken from that product"). Stores of
  races 13-17 all write such notes, in different words and fields. K2 reads them in any of these forms and marks the
  property as not stated on that page, so neither its checks nor the model buy on the page's own value. In race 17 the
  base bought such listings 10 times on their overridden value and passed once.
"""
from __future__ import annotations
import json
import random
import re
from time import monotonic, sleep
from typing import Any
from src.agent.proxy_client import ProxyClient
LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'
LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
HEAVY_ROOM = 0.02
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
BRIEF_BY = 120.0
BRIEF_GAP = 10.0
OUTAGE_BY = 150.0
SHOW_CHARS = 450
HUNT_STEPS = 24
ENV_GAP = 0.25
OPEN_GAP = 0.5
TABLE_KEYS = 18
CHEAPER_SHOWN = 4
QUIET_WORD = 4
OPEN_LATER = 2
PAGE_ATTR_CHARS = 700
PAGE_REST_CHARS = 300
NO_NAME = ('none', 'null', 'n/a', 'unknown')
ASK_TURNS = 2
MESSAGE_FIELD = '(?:^|_)(?:texts?|body|messages?|notes?|contents?|questions?|msg|say)(?:_|$)'
NOT_MESSAGE = '(?:^|_)(?:id|ids|type|format|code|lang|language|kind)$'
TRAIL_ROOM = 15
FIT_PAGES = 2
CRASH_TURNS = 3
OPEN_MIN_STEPS = 20
OPEN_ROWS = 5
OPEN_ITEMS = 2
OPEN_SHARE = 5
OPEN_EACH = 6
OPEN_NOT = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or|same|listing)\\b'
SHORT_KEEP = 8
SET_SHOWN = 2
STOCK_OUT = '^(?:no|none|false|0|n)$|out[ _-]?of[ _-]?stock|sold[ _-]?out|unavailable|not[ _-]?available'
HUNT_WHOLE = 49
HUNT_GRACE = 25.0
HUNT_VALUE_CHARS = 30
ASK_WORDS = '\\b(?:one|each|single|first|time|many|much|several|list|once|separate\\w*|split|short\\w*|questions?|ask\\w*|turn|lot|too|all)\\b'
HUNT_ROUNDS = 2
HUNT_KEEP = 10
HUNT_IDLE = 4
HUNT_PAGE_NEW = 5
HUNT_FIT_WORDS = 4
HUNT_FIT_OPENS = 4
HUNT_COMMON = 0.6
HUNT_BY = 200.0
TITLE_CHARS = 200
MIX_LABELS = 3
TEXT_CHARS = 3000
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3
TASK_SECONDS = 300.0
TIME_MARGIN = 20.0
CALL_FLOOR = 6.0
ASK_BY = 165.0
QUESTION_BY = 210.0
WIND_DOWN = 200.0
LAST_TURN = 255.0
EMPTY_ROOM = 12.0
LATE_BY = 190.0
LATE_STEPS = 8
SCARCE_MESSAGES = 2
LIST_SIZE = 12
LIST_MIXED = 8
LIST_STEPS = (1, 3)  # labels in the episode's first and second question, before the full list size
LIST_STEPS_MIXED = (3, 3)  # the same on several items, where each question goes to another item (simulated: 3,3,8 level or above 1,3,8 on 2-3 items)
TWO_LEFT = 2  # (I1) question messages left at the first question, at or below which a one-item request starts wider
TWO_LEFT_FIRST = 3  # (I1) labels of that first question
CAP_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
CAP_NOT = 'result|row|page|count|size|num|item|step'
CAP_PRICED = 'price|cost|budget|spend|money|amount'  # a limit argument's name, when no search has shown one holding yet
CAP_LOW = '(?:^|_)(?:min|minimum|low|lowest|least|from|start|offset|floor)(?:_|$)'
EARLY_TRAILS = 3  # chain rounds the early follow may spend; the checkout keeps its own
CAP_CHANCE = 0.05  # how rarely rows drawn at random from their labels would all sit under a value, for that value to count as a limit held
WALK_ROOM = 10  # steps beyond the finish a walk listing after the first needs: a refusal at the next checkout still needs a swap
WALK_CALLS = 4  # capped shelf listings per cart line at one checkout (race 16: 456 rows on a shelf, 6 a page)
SHELF_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MODE = 'field|filter|facet|brows|categor|list|structured'
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_short: dict[int, ProxyClient] = {}
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
# Prompt texts are composed from paragraphs, bullet lists, "key": meaning fields and sentences, so each rule can be
# read, reordered or replaced on its own line.
def _paragraphs(*blocks: str) -> str:
    return "\n\n".join(blocks)


def _lines(*rows: str) -> str:
    return "\n".join(rows)


def _bullets(*rows: str) -> str:
    return _lines(*("- " + row for row in rows))


def _sentences(*parts: str) -> str:
    return " ".join(parts)


def _fields(*pairs: tuple[str, str]) -> str:
    return _lines(*(f'"{key}": {meaning}' for key, meaning in pairs))


GUIDE = _paragraphs(
    _sentences(
        'You are K2, a careful buyer working for one shopper in an online store.',
        "You can act only by calling the store's tools.",
        'The episode is graded 1 or 0: it earns 1 only when the finish is exactly right.',
        'A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.',
    ),
    _lines('HOW THE STORE WORKS', _bullets(
        _sentences(
            'Every tool call is one step and steps are limited; thinking costs nothing.',
            "Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.",
        ),
        _sentences(
            'Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page.',
            'Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them.',
            "Keyword queries work best short and value-first (the wanted value plus the product noun); a store's search can read attribute values and shop names as well as titles and may want every word, so a few required values plus the noun can list nearly the whole fitting set, and a model or part code may find nothing (one try tells).",
            'A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.',
        ),
        _sentences(
            'Search rows are summaries.',
            "Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either.",
            'The ledger names the tool that shows attributes once K2 knows it.',
            'Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need.',
            'Never open a listing again when nothing has changed since you read it.',
        ),
        _sentences(
            "Questions to the shopper are handled for you by K2's interviewer, so you never message the shopper yourself.",
            'Its answers arrive in the conversation and are added to the ledger.',
        ),
        _sentences(
            'The request can also ask for steps along the way: an approval, a message, something to check or read first.',
            'The ledger lists them.',
            "K2 carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through K2's interviewer).",
        ),
    )),
    _lines('WHAT COUNTS AS A FIT', _bullets(
        "The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.",
        _sentences(
            "A requirement on a property is met when the listing's attribute for that property states the value.",
            'Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough.',
            'Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy).',
            'A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green").',
            'Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.',
        ),
        _sentences(
            "An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say.",
            'A value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else.',
            "A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement.",
            'K2 adds [evidence] lines after opened listings with the keys whose names match a requirement.',
        ),
        _sentences(
            'When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts.',
            'A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.',
        ),
        _sentences(
            "When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.",
            'A listing under the requested label is that kind of product even when its title or description names something else, so judge it on the requirements, price and stock.',
        ),
        _sentences(
            'A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ...").',
            "Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value; when the requirement asks for what goes with that listing's value (its partner, its counterpart) and the page names what its value pairs with, the requirement is that other value.",
            'The order counts only when every page of that chain was opened.',
            "When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.",
        ),
        'When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.',
        _sentences(
            "When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry).",
            'When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.',
        ),
        'Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.',
        'Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.',
    )),
    _lines('CHOOSING', _bullets(
        _sentences(
            'When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit.',
            'Find the whole fitting set (page through it: result rows are often in no price order, and K2 reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits.',
            'When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.',
        ),
        'When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.',
        'Otherwise any listing that meets everything is right: do not waste steps hunting for something better.',
        _sentences(
            'Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on.',
            'Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.',
        ),
        'A request for several different items needs one cart line per item, each at its requested quantity.',
    )),
    _lines('CHANGES WHILE YOU WORK', _bullets(
        _sentences(
            'The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out.',
            "They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force.",
            'The episode goes on after a refusal: apply every change, fix the cart and check out again.',
        ),
        "Price and stock can change silently: K2 re-reads the cart's lines and looks at the cart when you call the checkout, so do not spend a turn on either.",
    )),
    _lines('FINISHING', _bullets(
        'Put exactly the chosen lines in the cart (remove anything else), then check out with a call on its own.',
        _sentences(
            'End without a purchase only after a real search finds no listing that meets everything.',
            'Running short of steps proves nothing of the kind: then check out the best cart you have.',
            "Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.",
        ),
        'Every reply must call at least one tool. Keep your text short: what you know, then the next move.',
    )),
)


BRIEF = _paragraphs(
    _sentences(
        'You prepare a shopping task for an agent.',
        "Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:",
    ),
    _fields(
        ('items', _sentences(
            'a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list.',
            'Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property.',
            'A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}.',
            'Never put price, budget, stock or quantity into musts.',
            'The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name".',
            'A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context.',
            'A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.',
        )),
        ('budget', "the most the shopper will pay in total as a number in the request's own unit, or null."),
        ('budget_unit', 'that unit in words (for example "cents"), or null.'),
        ('budget_scope', '"total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.'),
        ('in_stock_only', 'true unless the request clearly accepts items that are not available.'),
        ('pick', _sentences(
            '"cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any".',
            'Remarks that money is tight are not a pick rule.',
        )),
        ('pick_measure', 'what is maximised or minimised for "most"/"least", else "".'),
        ('messages', 'how many messages the shopper allows the agent to send (integer), or null when not stated.'),
        ('steps', 'how many steps, moves or tool calls the request allows (integer), or null.'),
        ('lookups', 'a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].'),
        ('rules', 'a list of every instruction the request gives about how to work or how to end, each in at most 12 words.'),
        ('wishes', 'a list of things the shopper would merely like.'),
        ('ask_first', 'a list of things the request explicitly says to check with the shopper before choosing (empty when none).'),
        ('process', _sentences(
            'a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step.',
            'Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.',
        )),
        ('quiet_decline', 'true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.'),
        ('fits', 'a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.'),
        ('roles', 'an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).'),
        ('search_words', '{"tool": the tool that searches listings by free words (when one tool lists a category and another takes words, the one that takes words), "field": its parameter that takes the words, "args": the fixed arguments that make it search by words (for a tool with a mode argument, that mode), else {}}.'),
        ('buy_args', 'the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.'),
        ('refuse_args', 'the fixed arguments that make the "refuse" tool end without a purchase, else {}.'),
    ),
) + "\n"


PLAN = _paragraphs(
    _sentences(
        'You choose the questions a shopping agent sends to a shopper.',
        'Shoppers hold a few firm preferences they did not write down, each about one property of one requested product.',
        'A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.',
    ),
    _sentences(
        "You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values.",
        'An unwritten preference may be on a property that no listed key names.',
    ),
    _lines(
        'Choose the properties in this order:',
        '0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.',
        _sentences(
            '1.',
            'Hinted properties.',
            "The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it.",
            'For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.',
        ),
        "Name a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item's three most carried keys.",
        _sentences(
            '2.',
            "Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item's listings; once five or more listings were read, avoid keys seen on only one or two of them.",
        ),
        _sentences(
            'Skip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics.',
            'Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it.',
            'Rank every item on its own: its hinted properties first, then its listed keys.',
        ),
        '"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.',
        _sentences(
            'Choose up to COUNT for each item, best first.',
            'Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order',
        ),
    ),
)


HEAR = _lines(
    'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:',
    '{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}',
    _sentences(
        '- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker").',
        'Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need.',
        'Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").',
    ),
    _sentences(
        '- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true.',
        "Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it.",
        'The properties describing the other product are not needs of this item.',
    ),
    '- "free": the property of each question the shopper said they do not mind about.',
    _sentences(
        '- Folksy or roundabout wording that names a value still states that value.',
        'Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write "not X" only when the answer rejects X itself.',
    ),
)


NOTE = _lines(
    'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:',
    '{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}',
    'A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.',
)


CHECK = _lines(
    'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.',
    _sentences(
        'Reading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails, though a value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value.',
        'Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget.',
        'The newest record of a listing wins.',
    ),
    "When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.",
    _sentences(
        'Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}.',
        'Use ok=false only for a problem the records clearly show or a requirement no record addresses.',
    ),
)


READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'


AUDIT = _sentences(
    'A shopping agent wants to end without buying because it thinks nothing fits.',
    'Check the listing records shown against the requirements, with these reading rules: ',
) + READING + _lines(
    '',
    'Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.',
    'Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}',
)


UNMIX = _sentences(
    "A shopper's request was read into firm requirements on a product.",
    'Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought.',
    "Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing.",
    'Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}',
)


TRAIL = _lines(
    _sentences(
        'A shopping agent is about to check out.',
        "Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property.",
        'The order counts only when the agent has opened in full every listing along each such chain.',
        "Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page;",
        'ROW: its page not read, seen only as a search result row or by its attributes).',
    ),
    'Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page; when the requirement asks for what goes with that value and the page names what its value pairs with, that other value; or ""}]}',
    _sentences(
        'A record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind.',
        'Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.',
    ),
    "List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.",
)


GATE = _lines(
    _sentences(
        'A shopping agent is about to STAGE.',
        "The shopper's request asks for the steps below along the way.",
        'Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now.',
        "A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.",
    ),
    _sentences(
        'When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments.',
        'A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need.',
        'Give no call when every due step is done or when the messages or steps left cannot do the missing one.',
    ),
    'Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}',
)


SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + _lines(
    '',
    'Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}',
)


SHAPE = _sentences(
    'A store tool returned the result below.',
    "Name its parts by the exact keys the result and the tool's parameters use.",
    'Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}',
)

GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5}[ -]?(?:\\d{3,6}|\\d{1,5}[xX]{1,3})'
PART_CODE = '[A-Z]{1,5}\\d{0,3}-?\\d{3,7}'
VALUE_ROOM = 12  # steps beyond the finish before a value search runs at all (race 16: limit 35, answers at about step 15)
VALUE_KEEP = 8  # steps a value search leaves beyond the finish, for opening and buying what it found
LOOK_ALIKES = 2  # keys set aside in one question (race 16: one or two)
LINK_SEARCHES = 3  # series searched per item in a decline hunt (race 16: up to 9 series)
LINK_ROWS = 3  # rows of one item carrying a series code before the codes count as a system (race 16 covers: nearly every row)
LINK_FAMILIES = 12  # more distinct series than this are model numbers, not a system (race 16: 9)

def _codes(text: Any) -> list[str]:
    """Model and series codes named in a text, lower-cased with single spaces, dates left out."""
    source = str(text) if text else ''
    found: list[str] = []
    for code in re.findall('\\b' + FIT_CODE + '\\b', source):
        if _a_dated(code):
            continue
        found.append(_a_squeeze(code, ' ').strip().lower())
    return found

def _code_hit(pattern: str, code: str) -> bool:
    """True when a code named by an accessory covers a device's code. Spaces and hyphens are ignored; a series ending
    in x's after a digit ("abc12xx") stands for any code of that length with the same start ("abc1234")."""
    wide = _a_squeeze(pattern, '')
    plain = _a_squeeze(code, '')
    stem = wide.rstrip('x')
    is_series = stem != wide and stem[-1:].isdigit()
    if not is_series:
        return wide == plain
    return len(plain) == len(wide) and plain.startswith(stem)

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own_code = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    see_more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    model_codes = _a_codes_after('\\bmodel\\s*(?:no\\.?|number|#)?\\s*[:.]?\\s*(?!(?:no|number)\\b)(' + FIT_CODE + ')\\b', text)
    alias_codes = _a_codes_after('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text)
    fits, barred = _a_fit_lists(text)
    series, bare_series = _series_marks(text)
    marks: dict[str, Any] = {'own': _a_upper_group(own_code), 'codes': model_codes + alias_codes}
    marks.update(works=fits, bars=barred, also=_a_cross_fits(text), more=_a_upper_group(see_more))
    marks.update(fam=series, named=bare_series)
    return marks

def _series_marks(text: str) -> tuple[list[str], list[str]]:
    """Two sorted lists: the series prefixes of the full codes a record carries ("AB1" from "AB1-2345"), and the
    series-like bare words left once those codes and any excluding clause are cut out. Whether the bare words really
    are series is judged later against the codes of other records (K2.link_system)."""
    prefixes = _a_upper_sorted('\\b([A-Za-z]{1,5}\\d{1,3})-\\d{2,7}\\b', text)
    rest = text
    for cut in ('\\b[A-Za-z]{1,5}\\d{1,3}-\\d{2,7}\\b',
                '(?i)\\b(?:except|excluding|but not|not for|not compatible with|does not (?:fit|take))\\b[^.;]*'):
        rest = re.sub(cut, ' ', rest)
    return prefixes, _a_upper_sorted('\\b[A-Za-z]{1,5}\\d{1,3}\\b', rest)
PAIR_MARKS = (': ', ' = ', '=', ' -> ', ' - ')
FIELD_MARKS = (' | ', '\n', ' || ', ' ; ', '; ', '\t', ' // ', ' · ', ' • ', '|', ';', ', ')
FIELD_NAME = "[A-Za-z][A-Za-z0-9_ ./&'()-]{0,38}"
LIST_BULLET = '^\\s*(?:[-*•·]|\\d{1,3}[.)])\\s+'
RECORD_FIELDS = 3
NOTE_FIELDS = 5
FIELD_SHARE = 0.15
PAIR_SHARE = 0.7
BLOCK_NAME = 60
BLOCK_VALUE = 40
TAG_TEXT = 20000
PAIR_KEY = 'name|key|label|attr|prop|field|spec|title|^k$'
ID_FIELD = '(?:^|[_ -])(?:id|idx|sku|pid|ref|asin|ident\\w*)(?:$|[_ -])|^id(?-i:[A-Z0-9])|(?-i:Ident)|(?<![a-z])(?!(?:paid|valid|grid|solid|rigid|fluid|liquid|hybrid|rapid|void|acid|humid|android|said|laid)$)[a-z]*id$'
FAIL_KEY = 'error|fail|(?<!de)fault|trouble(?!shoot)|denied|refus|invalid'

def _failed(observation: Any) -> bool:
    """True for a result that says nothing but that the call went wrong: an object of at most three keys where a
    key naming a failure holds something, or a lone short code without spaces."""
    if not isinstance(observation, dict):
        return False
    size = len(observation)
    if size == 0 or size > 3:
        return False
    for key, value in observation.items():
        if value and re.search(FAIL_KEY, str(key), re.I):
            return True
    if size != 1:
        return False
    only = next(iter(observation.values()))
    return isinstance(only, str) and 0 < len(only) <= 40 and ' ' not in only

def _plain(text: str) -> Any:
    """The typed value a field's text stands for: true/false, None for an empty or null word, an int or float for
    a plain or money amount, otherwise the stripped text."""
    bare = text.strip()
    low = bare.lower()
    if low in _A_FIXED_WORDS:
        return _A_FIXED_WORDS[low]
    money = re.fullmatch('[$€£]?\\s?((?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?)', bare)
    if money is None or re.match('0\\d', money.group(1)):
        return bare
    digits = money.group(1)
    amount = float(digits.replace(',', ''))
    if '.' in digits or not amount.is_integer():
        return amount
    return int(amount)

def _block_value(text: str) -> Any:
    """Like _plain, but an attribute block keeps words as words: "none" or "true" stay the text they are."""
    value = _plain(text)
    as_words = isinstance(value, bool) or (value is None and bool(text.strip()))
    return text.strip() if as_words else value

def _pair(text: str, mark: str) -> tuple[str, str] | None:
    """A (name, value) pair when the text starts with a short field name followed by `mark`, else None."""
    cut = _a_cut_at(text, mark)
    if cut is None:
        return None
    name = re.sub(LIST_BULLET, '', cut[0]).strip()
    if len(name.split()) > 5 or not re.fullmatch(FIELD_NAME, name):
        return None
    return (name, cut[1].strip())

def _split_fields(text: str, between: str, mark: str) -> tuple[list[tuple[str, str]], int]:
    """Cuts a text at `between` into (name, value) fields and returns them with their count. A piece without a name
    is glued back to the field before it, separator and all, since values may contain the separator. On a line cut, a
    bare "name:" line collects the bulleted or indented lines under it as its value. The text must open with a field."""
    parts = [p for p in text.split(between) if p.strip()]
    if not parts or not _pair(parts[0].strip(), mark):
        return ([], 0)
    by_lines = between == '\n'
    fields: list[tuple[str, str]] = []
    collecting = False
    for part in parts:
        bare = part.strip()
        if collecting and _a_indented(part):
            name, held = fields[-1]
            fields[-1] = (name, held + '\n' + bare if held else bare)
            continue
        title = _a_heading(bare) if by_lines else None
        collecting = title is not None
        if collecting:
            fields.append((title, ''))
            continue
        pair = _pair(bare, mark)
        if pair:
            fields.append(pair)
        elif fields:
            name, held = fields[-1]
            fields[-1] = (name, held + between + bare)
    # every named piece opened exactly one field, so the field count is the named count
    return (fields, len(fields))

def _attr_block(text: str) -> dict[str, Any] | None:
    """A listing's attributes packed into one field, in any of the written forms stores use: a braced block, "a=b; c=d"
    pieces, bulleted "name: value" lines, or short "name: value" pieces between commas. Names may be any short text, as
    catalogs spell them, while prose with stray colons ("Fits: A, B. Condition: new") is left as text (None)."""
    bare = text.strip()
    if len(bare) < 5:
        return None
    if bare[0] in '{[' and bare[-1] in '}]':
        return _a_braced(bare)
    if '=' in bare:
        return _a_equals_block(bare)
    bullets = _a_bullet_block(bare)
    return bullets if bullets is not None else _a_comma_block(bare)

def _record_text(text: str) -> dict[str, Any] | None:
    """A text read as one record, or None for prose. Field separators are tried outermost first, and the first that
    yields a record decides; within it the pair mark naming the most fields wins. A name repeated keeps its first value,
    and a value that is itself an attribute block becomes a nested record."""
    fields = next((found for found in (_a_best_split(text, between) for between in FIELD_MARKS if between in text) if found is not None), None)
    if fields is None:
        return None
    firsts: dict[str, str] = {}
    for name, value in fields:
        firsts.setdefault(name, value)
    return {name: _a_field_value(name, value) for name, value in firsts.items()}

def _table_text(text: str) -> list[dict[str, Any]] | None:
    """The rows of a text table: a line of column names, then lines with as many cells. Markdown rule lines are dropped."""
    lines = []
    for line in text.splitlines():
        if not line.strip() or re.fullmatch('[\\s|:+=-]+', line):
            continue
        lines.append(line.strip().strip('|').strip())
    if len(lines) < 2:
        return None
    for between in ('|', '\t', ';', ','):
        rows = _a_grid(lines, between)
        if rows is not None:
            return rows
    return None

def _records_text(text: str) -> Any:
    """What a text result of a store tool holds in structured form, or None when it holds no records. Tried in turn:
    JSON, blocks or lines that are each a record, one record or a table, the same after markup is removed, and lines
    whose name and value are set apart by a tab or wide spacing."""
    bare = text.strip()
    if len(bare) < 12:
        return None
    if bare[0] in '{[':
        try:
            return _restore(json.loads(bare))
        except (ValueError, RecursionError):
            pass
    for parts in (re.split('\\n\\s*\\n', bare), bare.splitlines()):
        run = _a_record_run(parts)
        if run is not None:
            return run
    single = _record_text(bare)
    if single is not None:
        return single
    if '\n' in bare:
        table = _table_text(bare)
        if table is not None:
            return table
    if len(bare) <= TAG_TEXT and re.search('<[a-zA-Z/][^<>]{0,40}>', bare):
        untagged = _a_strip_tags(bare)
        if untagged != bare and len(untagged) >= 12:
            return _records_text(untagged)
    if '\n' not in bare:
        return None
    spaced = '\n'.join((re.sub('(?:[ ]{2,}|\t)', ': ', line.strip(), count=1) for line in bare.splitlines() if line.strip()))
    return _record_text(spaced) if spaced != bare else None

def _block_text(text: str) -> Any:
    """A text field of a row that has fields of its own becomes a record only when it is made wholly of name=value pieces."""
    if len(text) < 5 or '=' not in text or not re.search('; | \\| |\\n', text):
        return text
    parts = _a_semi_parts(text.strip())
    whole = len(parts) >= 2 and all((re.fullmatch('[^=:]{1,40}=[^=]{0,200}', p.strip()) for p in parts))
    return (_attr_block(text) or text) if whole else text

def _pairs_block(rows: list[Any], in_record: bool) -> dict[str, Any] | None:
    """A list that spells out attributes as one object: name/value objects, [name, value] lists, or, inside a record,
    short "name: value" texts. Lists of offers, highlights or notes are left alone (None)."""
    if len(rows) < 2:
        return None
    head = rows[0]
    if isinstance(head, dict):
        return _a_object_pairs(rows)
    if isinstance(head, (list, tuple)):
        return _a_list_pairs(rows)
    if in_record and all((isinstance(r, str) for r in rows)):
        return _a_text_pairs(rows)
    return None

def _column_rows(node: dict[str, Any]) -> dict[str, Any]:
    """When a result lists its column names once and each row as a bare list, the rows come back as objects and the
    names list is dropped."""
    match = next(((head, key) for head in [k for k, v in node.items() if _a_names_list(v)] for key in node if key != head and _a_rows_of(node[key], len(node[head]))), None)
    if match is None:
        return node
    head, key = match
    rebuilt = {k: v for k, v in node.items() if k != head}
    rebuilt[key] = [dict(zip(node[head], row)) for row in node[key]]
    return rebuilt

def _id_texts(record: dict[str, Any]) -> dict[str, Any]:
    """Numeric identifiers turned to text, the form listings are keyed by; only in records that also hold some text with
    spaces (tool schemas turn the text back into a number where needed)."""
    if not any((isinstance(v, str) and ' ' in v for v in record.values())):
        return record
    converted: dict[str, Any] = {}
    for key, value in record.items():
        converted[key] = str(value) if _a_numeric_id(key, value) else value
    return converted

def _keyed_rows(node: dict[str, Any]) -> list[dict[str, Any]] | None:
    """An object whose keys are listing codes and whose values are records sharing their fields, as a list of rows,
    each given an "id" with its code unless it already carries an identifier."""
    rows = list(node.values())
    if len(rows) < 2:
        return None
    if not all((isinstance(r, dict) and len(r) >= RECORD_FIELDS for r in rows)) or not all(map(_a_code_key, node)):
        return None
    if len(set.intersection(*(set(r) for r in rows))) < RECORD_FIELDS:
        return None
    return [row if _holds_id(row) else {'id': str(code), **row} for code, row in node.items()]

def _holds_id(node: dict[str, Any]) -> bool:
    for key in node:
        if re.search(ID_FIELD, str(key), re.I):
            return True
    return False

def _flat_block(node: dict[str, Any]) -> dict[str, Any]:
    """Groups inside an attribute block lifted to one level: {"size": {"width": 3}} becomes {"size width": 3}."""
    if any((isinstance(v, dict) for v in node.values())):
        return dict(_a_flat_pairs(node))
    return node

def _restore_field(value: Any, depth: int, carded: bool, page: bool=True) -> Any:
    """One field restored. A text is only replaced by the record it spells when that record looks like a listing (an
    identifier key or value, or many fields), so a store's note full of colons and semicolons stays the note."""
    restored = _restore(value, depth + 1, carded, page)
    if not isinstance(value, str) or not isinstance(restored, dict):
        return restored
    return value if _a_note_like(restored) else restored

def _restore(value: Any, depth: int=0, in_record: bool=False, page: bool=True) -> Any:
    """Brings a store result into the one shape the agent reads: lists of row objects, and a listing's attributes as a
    nested object. Worlds write results in many ways (race 15 sent each record as a text with "a=b; c=d" attributes
    inside, and an unread result blinds every check built on records), so all of them are unpacked here, in one place."""
    if depth > 7:
        return value
    if isinstance(value, dict):
        return _a_restore_object(value, depth, page)
    if isinstance(value, list):
        block = _pairs_block(value, in_record and page)
        return block if block else [_restore_field(v, depth, False, page) for v in value]
    if not isinstance(value, str):
        return value
    found = _records_text(value)
    return value if found is None else found

_FIRST_SPELLING: dict[str, str] = {}

def _join_twins(held: str, value: str) -> str:
    """The values of two spellings of a key merged into one list with each element once (case aside). The first
    spelling's original value is remembered for the joined text, so measures of how a key's values read still see it."""
    have = {part.lower() for part in _a_list_parts(held)}
    extra: list[str] = []
    for part in _a_list_parts(value):
        low = part.lower()
        if part and low not in have:
            have.add(low)
            extra.append(part)
    if not extra:
        return held
    joined = ', '.join(([held] if held.strip() else []) + extra)
    _FIRST_SPELLING[joined] = _FIRST_SPELLING.get(held, held)
    return joined

def _glued(text: str) -> str:
    """Lower-cased, with a number and a short unit after it written as one word, so "65 W" reads as "65w"."""
    lowered = text.lower()
    return re.sub('(\\d)\\s+([a-z]{1,3})(?![a-z0-9])', '\\1\\2', lowered)

def _text_words(text: str) -> set[str]:
    """A text's words, plus the number-and-unit words of its glued form."""
    forms = [text.lower()]
    forms.append(_glued(forms[0]))
    loose, glued = (set(re.findall('[a-z0-9]+', form)) for form in forms)
    return loose | glued

def _value_tokens(value: str) -> list[str]:
    """The words of a value that a search or a row can carry: those of three or more characters; failing that, short
    ones with a letter (XL, 4g, 65w); failing that, its numbers."""
    low = _glued(value)
    for pattern in ('[a-z0-9]{3,}', '[0-9]+[a-z]|[0-9]*[a-z][a-z0-9]+', '[0-9]+'):
        tokens = re.findall(pattern, low)
        if tokens:
            break
    return tokens

def _fold_twins(node: Any, depth: int=0) -> Any:
    """Each property on one line. When a block spells a key twice ("clock shape: rectangle", "clock_shape: rectangular,
    square"), the first spelling stays and carries both values if they differ, so any one element still matches. Roughly
    one listing in ten in races 13-15 and a practice world had such twins with different values, and reading them as a
    contradiction dropped listings the store accepted; duplicate lines also cost tokens on every later model turn."""
    if depth > 5:
        return node
    if isinstance(node, list):
        return [_fold_twins(v, depth + 1) for v in node]
    if not isinstance(node, dict):
        return node
    folded = _a_fold_object(node, depth)
    if len(folded) < 2 and len(node) >= 2:
        # folding left a single line of several: keep the keys as they were
        return {k: _fold_twins(v, depth + 1) for k, v in node.items()}
    return folded

def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def _clip(value: Any, limit: int, depth: int = 0) -> Any:
    """Long strings cut at `limit` characters, through nested lists and objects (eight levels at most)."""
    if depth > 7:
        return '...'
    deeper = depth + 1
    if isinstance(value, dict):
        return {key: _clip(part, limit, deeper) for key, part in value.items()}
    if isinstance(value, list):
        return [_clip(part, limit, deeper) for part in value]
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + '...'
    return value

def _json_obj(text: Any) -> dict[str, Any] | None:
    """The JSON object in a model reply: the whole reply (fences dropped), else its outermost braces."""
    if not isinstance(text, str):
        return None
    body = text.strip()
    fenced = _FENCED.match(body)
    if fenced:
        body = fenced.group(1)
    valid, found = _parsed_object(body)
    if valid:
        return found
    first, last = body.find('{'), body.rfind('}')
    if first < 0 or first >= last:
        return None
    return _parsed_object(body[first:last + 1])[1]

def _words(key: str) -> str:
    spaced = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', spaced).strip().lower()

def _snake(key: str) -> str:
    return '_'.join(_words(key).split(' '))

def _stem(text: str) -> frozenset[str]:
    return frozenset(word[:-1] if len(word) > 3 and word.endswith('s') else word for word in _words(text).split())

def _roots(text: str) -> set[str]:
    return {root for root in map(_root, _words(text).split()) if len(root) >= 3 and root not in _ROOT_SKIP}

def _spelled(text: str) -> frozenset[str]:
    """A name's words with plural -s and British -our folded, so "colours" and "color" compare equal."""
    return frozenset(re.sub('our$', 'or', _singular(word)) for word in _words(text).split())

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    """Whether a key spelling names a property: the property's words all in the key, or a key of two words or more
    all in the property."""
    if not (prop and key):
        return False
    return prop <= key or (len(key) >= 2 and key <= prop)

def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [entry for entry in kind or [] if isinstance(entry, str)]

def _fits_kind(value: Any, kind: str) -> bool:
    test = _KIND_TEST.get(kind)
    return True if test is None else test(value)

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """`value` fitted to a JSON schema where it plainly can be; what cannot be fixed is added to `faults`."""
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)

    def fits(candidate: Any) -> bool:
        return any(_fits_kind(candidate, kind) for kind in kinds)

    if kinds and not fits(value):
        value = _coerced(value, kinds)
        if not fits(value):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    value = _shape_choice(value, schema.get('enum'), where, faults)
    if isinstance(value, str):
        value = _shape_text(value, schema, where, faults)
    if _plain_number(value):
        _check_range(value, schema, where, faults)
    if isinstance(value, list):
        value = _shape_entries(value, schema, where, faults)
    if isinstance(value, dict):
        value = _shape_fields(value, schema, where, faults)
    return value

def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, (str, dict)) and value else []

def _count(value: Any) -> int | None:
    """A whole count from a number or digits, else None (true/false are not counts)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        return int(float(value))
    return None

def _id_like(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    bare = value.strip()
    if not 3 <= len(bare) <= 64 or ' ' in bare:
        return False
    return named or bool(re.search('\\d', value))

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    scored = [(score, key) for key in rows[0] for score in [_a_id_score(rows, key)] if score is not None]
    if not scored:
        return None
    return max(scored, key=lambda entry: entry[0])[1]

def _in_rows(node: Any, ident: str, depth: int = 0) -> bool:
    """Whether some list (six levels down at most) holds a row carrying `ident` as one of its text values."""
    if depth > 5:
        return False
    if isinstance(node, dict):
        return any(_in_rows(child, ident, depth + 1) for child in node.values())
    if not isinstance(node, list):
        return False
    return any(_row_holds(row, ident) for row in node) or any(_in_rows(child, ident, depth + 1) for child in node)

def _llm_within(seconds: float) -> ProxyClient:
    """The model client whose timeout fits the time left: the usual one, or one with a timeout in whole 5 s steps."""
    if seconds >= 50:
        return _llm
    limit = max(5, int(seconds) // 5 * 5)
    client = _llm_short.get(limit)
    if client is None:
        client = _llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return client

def _fact_block(node: Any) -> dict[str, Any] | None:
    """The plain fields of the nested object with the most of them (two at least): a listing's attribute block."""
    if not isinstance(node, dict):
        return None
    blocks = [_scalars(value) for value in node.values() if isinstance(value, dict) and len(value) >= 2]
    blocks = [block for block in blocks if len(block) >= 2]
    return max(blocks, key=len) if blocks else None

def _soft(fallback: Any) -> Any:
    """Decorator for methods that help the episode but are not essential: an exception is written to the trace and the
    method returns the fallback (called first when it is a factory), and the episode carries on."""

    def decorate(method: Any) -> Any:
        where = method.__name__

        def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return method(self, *args, **kwargs)
            except Exception as exc:
                self.trace.append(_a_crash_entry(where, exc))
            return fallback() if callable(fallback) else fallback
        return guarded
    return decorate

class _Stamped(list):
    """A trace list that stamps each new entry with the clock time it was added at."""

    def __init__(self, clock: Any):
        list.__init__(self)
        self.clock = clock

    def append(self, entry: dict[str, Any]) -> None:
        stamp = round(self.clock(), 1)
        if 'at' not in entry:
            entry['at'] = stamp
        list.append(self, entry)

def _filler(spec: dict[str, Any]) -> Any:
    """A safe value for a required field the agent has nothing to put in, or None when the schema offers none."""
    options = spec.get('enum')
    if isinstance(options, list) and options:
        return options[0]
    if 'const' in spec:
        return spec['const']
    if 'default' in spec:
        return spec['default']
    kinds = _kinds(spec)
    if any((kind in kinds for kind in ('integer', 'number'))):
        lowest = spec.get('minimum')
        return lowest if isinstance(lowest, (int, float)) else 1
    if 'boolean' in kinds:
        return False
    if 'string' not in kinds:
        return None
    return 'question'[:spec.get('maxLength') or 8]

def _strings(node: Any) -> list[str]:
    """Every text in a call's arguments, at any depth, in order."""
    below = _a_children(node)
    if below is None:
        return [node] if isinstance(node, str) else []
    texts: list[str] = []
    for child in below:
        texts.extend(_strings(child))
    return texts

def _says(node: Any, text: str) -> bool:
    """True when some value in a call's arguments, at any depth, is exactly `text`."""
    below = _a_children(node)
    if below is None:
        return node == text
    return any((_says(child, text) for child in below))


# (R1) A listing's own page can hand one of its properties to another listing: "Its socket is to be taken from the
# Dorvextor CPU with ddr4 ..., not from this item", "For color, check the Torqinoan dash cam, as this item's color is
# taken from that product" (race 17), "For the material, go by the Bed Sheets product from Raraluzen ..., not this item"
# (race 16). The page's own value for that property is then not the listing's value. In race 17, the base agent bought such a
# listing 10 times on the strength of its own (overridden) value and passed once (K2 v1: 7 of 13); a hanger whose facts
# said "black, gray, pink, beige" but whose colour came from a dash cam failed two tasks in both runs. So the readable
# result marks that property as not stated on this page, and the note stays in the page for the model to follow.
_HANDED_OFF = re.compile(
    r"\b(?:not|rather than|instead of) (?:from |by |to |at )?(?:this|that) (?:item|one|listing|product|[a-z]+)\b"
    r"|\b(?:this|that) (?:item|one|listing|product)'s [a-z][\w ]{0,30}? is taken from\b"
    r"|\b(?:taken|read|comes?|drawn) from (?:a |an |the )?(?:different|other|another|separate)\b"
    r"|\b(?:look|refer|go|turn) (?:to|at|by) (?:a |an |the )?(?:different|other|another|separate) (?:product|item|listing)\b"
    r"|\b(?:isn't|is not|aren't|are not|not) taken from (?:here|this)\b|\bdoesn't take it from here\b"
    r"|^\s*for (?:the |its )?[a-z][\w '-]{0,40}?, (?:refer to|look to|look at|go by|check|see|use) (?:a |an |the )?[^.]*\b(?:product|item|listing|from)\b",
    re.I)
_HANDED_WHAT = (
    re.compile(r"^\s*for (?:the |its |this item's |their |his |her )?([a-z][\w '-]{0,40}?)\s*,", re.I),
    re.compile(r"\btake (?:the |its |their )?([a-z][\w '-]{0,40}?) from\b", re.I),
    re.compile(r"\b(?:this|that) (?:item|one|listing|product)'s ([a-z][\w '-]{0,30}?) is taken from\b", re.I),
    re.compile(r"^\s*(?:its |the |their |this item's )?([a-z][\w '-]{0,40}?) (?:is to be |should be |must be |isn't |is not |is |are )?(?:taken|read|comes?|drawn) from\b", re.I),
)
_HANDED_NOTE = 'not stated on this page: it says this value is taken from another listing (see its notes)'


def _handed_off(text: str) -> list[str]:
    """The properties a page's text hands to another listing, in the words the page uses ([] when none)."""
    found = []
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        if len(sentence) > 400 or not _HANDED_OFF.search(sentence):
            continue
        for pattern in _HANDED_WHAT:
            hit = pattern.search(sentence)
            if hit:
                words = _words(hit.group(1))
                if words and len(words.split()) <= 5 and words not in found:
                    found.append(words)
                break
    return found


def _same_property(name: str, key: Any) -> bool:
    a, b = set(_words(name).split()), set(_words(str(key)).split())
    a = {w[:-1] if len(w) > 3 and w.endswith('s') else w for w in a}
    b = {w[:-1] if len(w) > 3 and w.endswith('s') else w for w in b}
    return bool(a) and a <= b      # the key names the property (perhaps more narrowly); a wider name ('color
                                    # temperature') never takes a shorter key ('color')


def _without_handed_off(node: Any, depth: int = 0) -> Any:
    """`node` with each listing record's handed-off properties marked as not stated there; other values untouched."""
    if depth > 6:
        return node
    if isinstance(node, list):
        return [_without_handed_off(entry, depth + 1) for entry in node]
    if not isinstance(node, dict):
        return node
    out = {key: _without_handed_off(value, depth + 1) for key, value in node.items()}
    text = ' '.join(value for value in node.values() if isinstance(value, str) and len(value) >= 30)
    names = _handed_off(text) if text else []
    if not names:
        return out
    blocks = [key for key, value in out.items() if isinstance(value, dict) and sum(1 for v in value.values() if not isinstance(v, (dict, list))) >= 2]
    if not blocks:
        return out
    for name in names:
        marked = False
        for key in blocks:
            block = out[key]
            hits = [k for k in block if _same_property(name, k)]
            if hits:
                out[key] = {k: (_HANDED_NOTE if k in hits else v) for k, v in block.items()}
                marked = True
        if not marked:
            out[blocks[0]] = {**out[blocks[0]], name: _HANDED_NOTE}
    return out

class Item:
    """One requested product: its name, store shelf, quantity and firm needs, plus what the shopper said later."""

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name, self.shelf, self.qty, self.musts = name, shelf, qty, musts
        self.told: list[str] = []          # needs the shopper stated in answers
        self.dropped: list[str] = []       # needs the shopper gave up
        self.free: list[str] = []          # properties the shopper does not mind about

_B_WHEN_TEXT = {
    'choose': 'before choosing',
    'order': 'at the order (K2 does it when you check out)',
    'end': 'at an ending without a purchase (K2 does it then)',
}


_B_FINAL_NOTE = (' FINAL STEPS: check out now when the cart holds a line for each item;'
                 + ' end without a purchase only when your search showed that an item has no fitting listing'
                 + ' (running out of steps does not show that).')


_B_FEW_NOTE = ' Few steps left: set the cart to the chosen lines and check out now.'


_B_LOW_NOTE = (' Steps are running low: settle on your picks, fill the cart'
               + ' and check out within the next few steps.')


_B_HALF_NOTE = (' Half the steps are used and the cart is empty:'
                + " cart each item's fitting listing now (the store may still sell one out).")


_B_CLOCK_NOTE = (' TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns;'
                 + ' an episode that runs out of time earns nothing.')


_B_RESOLVE_HEAD = ('\n[resolve first: open the listing each of these points at,'
                   + ' and every listing its page refers to; the last page gives the value] ')


_B_UNSTATED_HEAD = ("; described in the shopper's words,"
                    + ' which no listing attribute states (decides nothing): ')


_B_PAGE_TAIL = "; a listing's page (what a chain of pages or a page check means) is shown by "


def _b_cache_marked(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies of the messages with the first and the last text turned into cacheable blocks (Anthropic models)."""
    copied = [dict(m) for m in messages]
    for entry in (copied[0], copied[-1]):
        text = entry.get('content')
        if isinstance(text, str) and text:
            entry['content'] = [{'type': 'text', 'text': text, 'cache_control': {'type': 'ephemeral'}}]
    return copied


def _b_flipped(before: dict[str, Any], after: dict[str, Any], was: bool, now: bool) -> bool:
    """Some flag that was `was` in the stored row is `now` in the new copy of it."""
    return any(before.get(key) is was and value is now for key, value in after.items())


_C_RANGE_WORDS = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b'


def _c_is_whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _c_alnum3(text: str) -> set[str]:
    """The lower-cased letter/digit runs of three or more characters in a text."""
    return set(re.findall('[a-z0-9]{3,}', text.lower()))


def _c_shaped(schema: Any, value: dict[str, Any]) -> dict[str, Any] | None:
    """The arguments fitted to the tool schema, or None when they cannot be made to fit."""
    problems: list[str] = []
    fitted = _shape(value, schema, 'args', problems)
    return None if problems else fitted


def _c_key_in(where: Any, key: Any) -> str | None:
    """A key the shape reader named, kept only when it really exists in the given mapping."""
    if isinstance(key, str) and key in where:
        return key
    return None


def _c_category_rank(values: set[str], hits: int, known: set[str]) -> tuple[float, int, int, float]:
    """Sort key of a candidate category column: most values that are schema labels, then fewest distinct values,
    then the most rows carrying it, then the longest values."""
    size = len(values)
    on_schema = sum(v.strip().lower() in known for v in values) / size
    avg_len = sum(map(len, values)) / size
    return (-on_schema, size, -hits, -avg_len)


def _c_past_first_page(given: Any) -> bool:
    """The call already asked for a later page (a page number past 1, or any cursor text)."""
    if isinstance(given, int) and not isinstance(given, bool):
        return given not in (0, 1)
    return bool(str(given or '').strip())


def _c_deal(lanes: list[list[str]], bounds: list[tuple[int, int]], rounds: int) -> list[str]:
    """Take row k of every lane in turn (k = 0, 1, ...), within each lane's [low, high) window."""
    dealt: list[str] = []
    for k in range(rounds):
        for lane, (low, high) in zip(lanes, bounds):
            if low <= k < min(high, len(lane)):
                dealt.append(lane[k])
    return dealt


_d_hedge_words = ('not', 'no', 'other than', 'unlike', 'alongside', 'listing', 'page', 'shown', 'at most', 'at least',
                  'under', 'over', 'below', 'above', 'less', 'more', 'between', 'within', 'up to', 'or')


_d_hedged = '\\b(?:' + '|'.join(_d_hedge_words) + ')\\b'


def _d_label(record, key):
    """The category label a record carries under `key`, as text ('' when it carries none)."""
    return str((record or {}).get(key) or '')


def _d_has_letter(value):
    return isinstance(value, str) and bool(re.search('[a-z]', value, re.I))


def _d_text_fields(props):
    """Free-text string fields of a tool schema, and the ones among them whose name reads like a query."""
    free = [name for name, spec in props.items() if isinstance(spec, dict) and 'string' in _kinds(spec) and 'enum' not in spec]
    return free, [name for name in free if re.search(QUERY_WORDS, _snake(name), re.I)]


def _d_long_words(text):
    return {w for w in _stem(text) if len(w) >= 4}


def _d_catch_all_last(labels, folded=False):
    """Labels in their order, with the broad "features / specs" style ones moved to the end."""
    broad = [x for x in labels if re.search(CATCH_ALL, _words(x) if folded else x)]
    return [x for x in labels if x not in broad] + broad


def _d_first_best(scores):
    """Index of the first top score, or None when no score is above zero."""
    top = max(scores, default=0)
    return scores.index(top) if top > 0 else None


def _d_plain(text):
    # plural -s and British -our folded away, for matching a planned label to a listing key
    return re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))


def _d_share(own, label):
    """Share of the item's opened listings carrying the key a label names (exact key, else the best near key)."""
    exact = _words(label)
    if exact in own:
        return own[exact]
    mine = _stem(label)
    return max((share for key, share in own.items() if _names(mine, _stem(key)) or _names(_stem(key), mine)), default=0.0)


_E_SOLD_TAG = 'SOLD OUT (the store says it can no longer be bought). '


_E_LIMIT_WORDS = frozenset({'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max',
                            'minimum', 'maximum'})


_E_REVIEW_END = 'Fix the cart (or show the review is wrong by re-reading the records), then check out.'


def _e_sections(head: str, parts: list[tuple[str, str]]) -> str:
    """A prompt: its head, then every (label, body) pair after a blank line."""
    return head + ''.join(f'\n\n{label}{body}' for label, body in parts)


def _e_has_words(text: str, least: int) -> bool:
    return len(text.split()) >= least


def _e_phrases(node: Any, depth: int):
    """Strings of two words or more found walking a reply down to four levels, in reading order."""
    if depth > 4:
        return
    if isinstance(node, str):
        if _e_has_words(node, 2):
            yield node
        return
    if isinstance(node, dict):
        node = list(node.values())
    if isinstance(node, list):
        for child in node:
            yield from _e_phrases(child, depth + 1)


def _e_popped_dicts(root: Any):
    """Dicts of a nested value in last-in-first-out order (the order a plain stack walk meets them)."""
    stack: list[Any] = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack += node
        elif isinstance(node, dict):
            yield node
            stack += [v for v in node.values() if isinstance(v, (dict, list))]


def _e_dumped(value: Any, clip: int, cut: int) -> str:
    return _dump(_clip(value, clip))[:cut]


def _e_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _e_tokens(text: Any) -> set[str]:
    return set(re.sub('[,/;|]', ' ', _words(text)).split())


def _e_stated(need: str, block: dict[str, Any]) -> bool:
    """A requirement is stated when an attribute named like its property holds every word of its value; a value that
    is a limit or an exclusion never counts as stated."""
    prop, _, value = need.partition(':')
    wanted = _e_tokens(value)
    if not wanted or wanted & _E_LIMIT_WORDS:
        return False
    name = _spelled(prop)
    return any(_names(name, _spelled(k)) and wanted <= _e_tokens(str(v)) for k, v in block.items())


def _e_need_entries(value: Any) -> list[dict[str, Any]]:
    return [e for e in _items(value) if isinstance(e, dict) and str(e.get('need') or '').strip()]


def _e_first_key(props: dict[str, Any], test) -> str | None:
    return next((k for k, v in props.items() if isinstance(v, dict) and test(v)), None)


def _e_switch_modes(options: list[str]) -> tuple[str, str]:
    """The enum value that takes units off a line, and the one that puts them on."""
    off = next((o for o in options if re.search('remov|delet|drop|clear|take', o, re.I)), options[1])
    on = next((o for o in options if o != off), options[0])
    return (off, on)


_f_decline_text = 'Sorry, nothing in the store meets all of your requirements, ' + 'so I am not buying anything.'


def _f_section(title: str, body: str) -> str:
    return '\n\n' + title + ':\n' + body


def _f_free_text(spec: dict[str, Any]) -> bool:
    return 'string' in _kinds(spec) and 'enum' not in spec


def _f_whole_number(spec: dict[str, Any]) -> bool:
    return 'integer' in _kinds(spec)


def _f_has_choice(spec: dict[str, Any]) -> bool:
    return isinstance(spec.get('enum'), list) and len(spec['enum']) >= 2


def _f_first_field(props: dict[str, Any], test) -> str | None:
    """First property whose schema is an object passing `test`."""
    for field, spec in props.items():
        if isinstance(spec, dict) and test(spec):
            return field
    return None


def _f_marks_removal(field_spec: Any, value: Any) -> bool:
    """The value picks a non-first option of an enum field (the cart tool's remove-style modes)."""
    options = (field_spec or {}).get('enum')
    return isinstance(options, list) and len(options) >= 2 and value in options[1:]


def _f_nested_lines(props: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The first array property whose entries are objects, with the entry properties."""
    for field, spec in props.items():
        if not (isinstance(spec, dict) and 'array' in _kinds(spec)):
            continue
        entry = spec.get('items')
        if isinstance(entry, dict) and isinstance(entry.get('properties'), dict):
            return field, entry['properties']
    return None


def _f_bulk_call(field: str, entry_props: dict[str, Any], lines: dict[str, int]) -> list[dict[str, Any]] | None:
    """One call that lists every wanted line inside an array of objects."""
    id_field = _f_first_field(entry_props, _f_free_text)
    unit_field = _f_first_field(entry_props, _f_whole_number)
    if not id_field:
        return None
    entries = []
    for ident, units in lines.items():
        entry = {id_field: ident}
        if unit_field:
            entry[unit_field] = units
        entries.append(entry)
    return [{field: entries}]


def _f_fit_entry(entry: Any) -> tuple[str, Any]:
    if isinstance(entry, dict):
        return str(entry.get('id')), entry.get('price')
    return str(entry), None


def _f_price_key(price: Any) -> float:
    numeric = isinstance(price, (int, float)) and not isinstance(price, bool)
    return float(price) if numeric else float('inf')


def _f_units_after(have: int, count: int | None, removal: bool, sets: bool, spec: dict[str, Any]) -> int:
    """Units of a line once a cart call naming it with `count` has gone through."""
    if removal:
        counted = any(('integer' in _kinds(v) for v in spec.values() if isinstance(v, dict)))
        if counted:
            return have - (count or 1)
        return have - count if count else 0
    step = 1 if count is None else count
    return step if sets else have + step


_FENCED = re.compile('^```[A-Za-z]*\\s*(.*?)\\s*```$', re.S)


def _parsed_object(text: str) -> tuple[bool, dict[str, Any] | None]:
    """(whether `text` is valid JSON, the object it holds or None)."""
    try:
        got = json.loads(text)
    except ValueError:
        return False, None
    return True, got if isinstance(got, dict) else None


def _singular(word: str) -> str:
    """A plural -s dropped (not from -ss words or short ones)."""
    return word[:-1] if len(word) > 3 and word.endswith('s') and not word.endswith('ss') else word


_ROOT_SKIP = ('and', 'for', 'the', 'with')


def _root(word: str) -> str:
    return word[:-2] if re.search('(?:ch|sh|ss|x|z)es$', word) else _singular(word)


def _plain_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_KIND_TEST = {
    'string': lambda value: isinstance(value, str),
    'integer': lambda value: isinstance(value, int) and not isinstance(value, bool),
    'number': _plain_number,
    'boolean': lambda value: isinstance(value, bool),
    'array': lambda value: isinstance(value, list),
    'object': lambda value: isinstance(value, dict),
    'null': lambda value: value is None,
}


def _coerced(value: Any, kinds: list[str]) -> Any:
    """The value converted to the first listed kind it can plainly become (int-valued float, numeric text, number as
    text, true/false text, a lone value as a list); unchanged when none applies."""
    for kind in kinds:
        if kind == 'integer' and isinstance(value, float) and value.is_integer():
            return int(value)
        if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value):
            number = float(value)
            return int(number) if kind == 'integer' or number.is_integer() else number
        if kind == 'string' and _plain_number(value):
            return str(value)
        if kind == 'boolean' and isinstance(value, str) and value.strip().lower() in ('true', 'false'):
            return value.strip().lower() == 'true'
        if kind == 'array' and value is not None and not isinstance(value, list):
            return [value]
    return value


def _shape_choice(value: Any, options: Any, where: str, faults: list[str]) -> Any:
    """A value outside the schema's choice list: matched to a choice case-blind, else reported."""
    if not isinstance(options, list) or value in options:
        return value
    if isinstance(value, str):
        wanted = value.strip().lower()
        same = [option for option in options if isinstance(option, str) and option.lower() == wanted]
        if same:
            value = same[0]
    if value not in options:
        faults.append(f'{where}: allowed values are {options}')
    return value


def _shape_text(value: str, schema: dict[str, Any], where: str, faults: list[str]) -> str:
    longest, shortest = schema.get('maxLength'), schema.get('minLength')
    if isinstance(longest, int) and len(value) > longest:
        value = value[:longest]
    if isinstance(shortest, int) and len(value) < shortest:
        faults.append(f'{where} is too short')
    return value


def _check_range(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> None:
    low, high = schema.get('minimum'), schema.get('maximum')
    if isinstance(low, (int, float)) and value < low:
        faults.append(f'{where} is below {low}')
    if isinstance(high, (int, float)) and value > high:
        faults.append(f'{where} is above {high}')


def _shape_entries(value: list[Any], schema: dict[str, Any], where: str, faults: list[str]) -> list[Any]:
    most, least, each = schema.get('maxItems'), schema.get('minItems'), schema.get('items')
    if isinstance(most, int) and len(value) > most:
        faults.append(f'{where} allows at most {most} entries')
    if isinstance(least, int) and len(value) < least:
        faults.append(f'{where} needs at least {least} entries')
    if not isinstance(each, dict):
        return value
    return [_shape(entry, each, f'{where}[{index}]', faults) for index, entry in enumerate(value)]


def _shape_fields(value: dict[str, Any], schema: dict[str, Any], where: str, faults: list[str]) -> dict[str, Any]:
    known = schema.get('properties')
    if not isinstance(known, dict):
        known = {}
    if schema.get('additionalProperties') is False:
        faults.extend(f'{where}.{name} is not an allowed argument' for name in [n for n in value if n not in known])
    for name in schema.get('required') or []:
        if name not in value:
            faults.append(f'{where}.{name} is required')
    return {key: _shape(part, known.get(key), f'{where}.{key}', faults) if key in known else part
            for key, part in value.items()}


def _row_holds(row: Any, ident: str) -> bool:
    return isinstance(row, dict) and ident in [str(text).strip() for text in row.values() if isinstance(text, str)]


def _scalars(block: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in block.items() if isinstance(value, (str, int, float)) and not isinstance(value, bool)}


_A_FIXED_WORDS = {'true': True, 'false': False, 'none': None, 'null': None, '': None}


def _a_squeeze(text: str, fill: str) -> str:
    """Runs of spaces and hyphens replaced by `fill`."""
    return re.sub('[\\s-]+', fill, text)


def _a_dated(code: str) -> bool:
    """A capitalised word followed by a year ("June 2021") has the look of a code but is a date."""
    return re.fullmatch('[A-Z][a-z]+ (?:19|20)\\d\\d', code) is not None


def _a_upper_group(match: Any) -> str:
    return '' if match is None else match.group(1).upper()


def _a_codes_after(pattern: str, text: str) -> list[str]:
    """The codes inside the first group of every case-blind match of `pattern`."""
    return [code for hit in re.finditer(pattern, text, re.I) for code in _codes(hit.group(1))]


def _a_fit_lists(text: str) -> tuple[list[str], list[str]]:
    """Codes a text says it fits, and the codes it excludes from that list ("fits A, B except C"). A "fits" clause
    that belongs to a named part ("part X also fits") is that part's business and skipped."""
    fits: list[str] = []
    barred: list[str] = []
    pattern = '\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)'
    for hit in re.finditer(pattern, text, re.I):
        lead_in = text[max(0, hit.start() - 40):hit.start()]
        if re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', lead_in, re.I):
            continue
        kept, *excluded = re.split('(?i)\\b(?:except|excluding|but not)\\b', hit.group(1), maxsplit=1)
        fits.extend(_codes(kept))
        if excluded:
            barred.extend(_codes(excluded[0]))
    return fits, barred


def _a_cross_fits(text: str) -> list[tuple[str, list[str]]]:
    """For each "part X also fits ..." clause: the part's code and the codes it lists."""
    pattern = '\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)'
    found: list[tuple[str, list[str]]] = []
    for hit in re.finditer(pattern, text, re.I):
        found.append((hit.group(1).upper(), _codes(hit.group(2))))
    return found


def _a_upper_sorted(pattern: str, text: str) -> list[str]:
    return sorted({hit.upper() for hit in re.findall(pattern, text)})


def _a_cut_at(text: str, mark: str) -> tuple[str, str] | None:
    """The text before and after the first `mark`; a text ending in the mark's bare form counts with an empty rest."""
    head, found, rest = text.partition(mark)
    if found:
        return head, rest
    closing = mark.strip()
    if text.endswith(closing):
        return text[:-len(closing)], ''
    return None


def _a_heading(bare: str) -> str | None:
    """The name of a line like "Features:" that opens a list below it."""
    if not bare.endswith(':'):
        return None
    name = bare[:-1].strip()
    return name if re.fullmatch(FIELD_NAME, name) else None


def _a_indented(part: str) -> bool:
    return bool(re.match(LIST_BULLET, part)) or part[:1] in ' \t'


def _a_braced(bare: str) -> dict[str, Any] | None:
    """A block written in braces: JSON when it parses, else its quoted or bare name:value pairs (two or more)."""
    try:
        parsed = json.loads(bare)
    except (ValueError, RecursionError):
        loose = dict(re.findall('[\'"]?([A-Za-z][\\w ./-]{0,38})[\'"]?\\s*[:=]\\s*[\'"]?([^,\'"}\\]]*)', bare))
        if len(loose) < 2:
            return None
        return {name.strip(): _plain(raw) for name, raw in loose.items()}
    return parsed if isinstance(parsed, dict) else None


def _a_semi_parts(text: str) -> list[str]:
    """The non-blank pieces of a text cut at "; ", a closing ";", " | " or a line break."""
    return [p for p in re.split(';\\s+|;$|\\s\\|\\s|\\n', text) if p.strip()]


def _a_equals_block(bare: str) -> dict[str, Any] | None:
    """A block of name=value pieces. A piece without its own name continues the previous value; a block whose first
    piece has no name is not a block. A single pair is accepted only when it is short and its name has no colon."""
    block: dict[str, Any] = {}
    last = ''
    for part in _a_semi_parts(bare):
        name, found, value = part.partition('=')
        key = name.strip()
        if found and 0 < len(key) <= BLOCK_NAME and key not in block:
            last = key
            block[key] = _block_value(value)
            continue
        if not last:
            return None
        block[last] = '{}; {}'.format(block[last], part.strip())
    if len(block) >= 2:
        return block
    lone_ok = bool(block) and ':' not in last and len(last) <= 30 and len(bare) <= 90
    return block if lone_ok else None


def _a_bullet_block(bare: str) -> dict[str, Any] | None:
    """Two or more bulleted lines, each "name: value"."""
    lines = [line for line in bare.splitlines() if line.strip()]
    if len(lines) < 2 or not all((re.match(LIST_BULLET, line) for line in lines)):
        return None
    pairs = [_pair(line.strip(), ': ') for line in lines]
    if not all(pairs):
        return None
    return {name: _plain(value) for name, value in pairs}


def _a_comma_block(bare: str) -> dict[str, Any] | None:
    """Short "name: value" pieces between commas (or else semicolons), names without dots."""
    if ': ' not in bare:
        return None
    for between in (', ', '; '):
        if between not in bare:
            continue
        pairs = [_pair(part.strip(), ': ') for part in bare.split(between)]
        if len(pairs) < 2 or not all(pairs):
            continue
        if all(('.' not in name and len(value) <= BLOCK_VALUE for name, value in pairs)):
            return {name: _plain(value) for name, value in pairs}
    return None


def _a_best_split(text: str, between: str) -> list[tuple[str, str]] | None:
    """The fields of the pair mark that names the most fields when the text is cut at `between` (first such mark on a
    tie), or None when no mark names enough of the pieces or the names mostly repeat."""
    pieces = sum((1 for p in text.split(between) if p.strip()))
    share = PAIR_SHARE if between == ', ' else FIELD_SHARE
    floor = max(RECORD_FIELDS, share * pieces)
    top, chosen = 0, None
    for mark in PAIR_MARKS:
        fields, named = _split_fields(text, between, mark)
        if named < floor or len({name for name, _ in fields}) * 2 <= len(fields):
            continue
        if named > top:
            top, chosen = named, fields
    return chosen


def _a_field_value(name: str, value: str) -> Any:
    inner = _attr_block(value)
    if inner:
        return inner
    if re.search(ID_FIELD, name, re.I):
        return value.strip()
    return _plain(value)


def _a_grid(lines: list[str], between: str) -> list[dict[str, Any]] | None:
    """The rows of the lines cut at `between` when the first line is a heading of three or more names and every line
    has as many cells."""
    grid = [[cell.strip() for cell in line.split(between)] for line in lines]
    head = grid[0]
    width = len(head)
    if width < 3 or not all((re.fullmatch(FIELD_NAME, cell) for cell in head)):
        return None
    if any((len(row) != width for row in grid[1:])):
        return None
    # a heading cell starts with a letter, so it is never a bare number or price
    return [dict(zip(head, map(_plain, row))) for row in grid[1:]]


def _a_record_run(parts: list[str]) -> list[dict[str, Any]] | None:
    """Pieces that are mostly records (at least two, four in five of the pieces) all opening with the same field name."""
    if len(parts) < 2:
        return None
    good = [found for found in map(_record_text, parts) if found]
    if len(good) < 2 or len(good) * 5 < len(parts) * 4:
        return None
    return good if len({next(iter(found)) for found in good}) == 1 else None


def _a_strip_tags(text: str) -> str:
    """Markup taken out: inline tags dropped, table cells joined as "name: value", every other tag run a line break."""
    for pattern, filler in (('</?(?:b|i|em|strong|span|a|u|small)\\b[^<>]{0,40}>', ''), ('</t[dh]>\\s*<t[dh][^<>]{0,40}>', ': ')):
        text = re.sub(pattern, filler, text, flags=re.I)
    return re.sub('\\s*(?:<[^<>]{1,40}>\\s*)+', '\n', text).strip()


def _a_short_name(name: Any) -> bool:
    return isinstance(name, str) and 0 < len(name) <= BLOCK_NAME


def _a_object_pairs(rows: list[Any]) -> dict[str, Any] | None:
    """Objects that all hold the same two keys, the first naming the attribute, the second its scalar value."""
    first = rows[0]
    if any((not isinstance(r, dict) or set(r) != set(first) or len(r) != 2 for r in rows)):
        return None
    name_key, value_key = list(first)
    if not re.search(PAIR_KEY, str(name_key), re.I):
        return None
    if not all((_a_short_name(r[name_key]) and not isinstance(r[value_key], (dict, list)) for r in rows)):
        return None
    return {r[name_key].strip(): r[value_key] for r in rows}


def _a_list_pairs(rows: list[Any]) -> dict[str, Any] | None:
    """[name, scalar value] lists."""
    fine = all((isinstance(r, (list, tuple)) and len(r) == 2 and _a_short_name(r[0]) and not isinstance(r[1], (dict, list)) for r in rows))
    return {r[0].strip(): r[1] for r in rows} if fine else None


def _a_text_pairs(rows: list[str]) -> dict[str, Any] | None:
    """Texts each "name: value" (or "name=value") with a name of at most three words and a short value."""
    pairs = [_pair(r.strip(), ': ') or _pair(r.strip(), '=') for r in rows]
    if not all((p and len(p[1]) <= BLOCK_VALUE and len(p[0].split()) <= 3 for p in pairs)):
        return None
    return {name: _plain(value) for name, value in pairs}


def _a_names_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 2 and all((isinstance(c, str) and re.fullmatch(FIELD_NAME, c) for c in value))


def _a_rows_of(value: Any, width: int) -> bool:
    return isinstance(value, list) and bool(value) and all((isinstance(r, (list, tuple)) and len(r) == width for r in value))


def _a_numeric_id(key: Any, value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and bool(re.search(ID_FIELD, key, re.I)) and abs(value) >= 100


def _a_code_key(key: Any) -> bool:
    spelled = str(key)
    return ' ' not in spelled and re.search('\\d', spelled) is not None


def _a_flat_pairs(node: dict[str, Any]):
    """The node's entries, a group's scalar entries lifted with the group name in front."""
    for key, value in node.items():
        if not isinstance(value, dict):
            yield (key, value)
            continue
        for inner, item in value.items():
            if not isinstance(item, (dict, list)):
                yield (f'{key} {inner}', item)


def _a_note_like(found: dict[str, Any]) -> bool:
    """A few fields with no identifier among their keys or values: a remark, not a listing."""
    return len(found) < NOTE_FIELDS and not _holds_id(found) and not any((_id_like(v) for v in found.values()))


def _a_grouped(value: Any) -> bool:
    """An attribute object holding groups (objects) but no lists, whose first entry is a group or a scalar, and no id."""
    if not isinstance(value, dict):
        return False
    entries = list(value.values())
    if not any((isinstance(x, dict) for x in entries)) or any((isinstance(x, list) for x in entries)):
        return False
    return isinstance(entries[0], (dict, str, int, float)) and not _holds_id(value)


def _a_restore_object(raw: dict[str, Any], depth: int, page: bool) -> Any:
    node = _id_texts(_column_rows(raw))
    node = {k: (_keyed_rows(v) if isinstance(v, dict) else None) or v for k, v in node.items()}
    has_id = _holds_id(node)
    carded = len(node) >= RECORD_FIELDS or has_id
    listing = has_id or any((_id_like(v) for v in node.values()))
    roomy = len(node) > 2
    lift = depth <= 3 and len(node) >= RECORD_FIELDS
    out: dict[str, Any] = {}
    for key, field in node.items():
        if roomy and isinstance(field, str):
            field = _block_text(field) if listing else field
        else:
            field = _restore_field(field, depth, carded, page)
        out[key] = _flat_block(field) if lift and _a_grouped(field) else field
    if len(out) != 1:
        return out
    # a one-key wrapper around a whole record, when the original held text first, is that record
    only = next(iter(out.values()))
    wrapped = isinstance(only, dict) and len(only) >= RECORD_FIELDS and isinstance(next(iter(raw.values())), str)
    return only if wrapped else out


def _a_list_parts(text: str) -> list[str]:
    """The stripped elements of a comma list; a comma between two digits belongs to a number."""
    return [part.strip() for part in re.split('(?<!\\d),|,(?!\\d)', text)]


def _a_joinable(held: Any, value: Any, name: str) -> bool:
    return isinstance(held, str) and isinstance(value, str) and not re.search(ID_FIELD, name, re.I)


def _a_fold_object(node: dict[str, Any], depth: int) -> dict[str, Any]:
    """The object with each property under its first spelling; a later text spelling adds its new elements to it."""
    out: dict[str, Any] = {}
    first_key: dict[str, str] = {}
    for key, value in node.items():
        name = _words(key)
        first = first_key.get(name)
        if first is None:
            first_key[name] = key
        elif out[first] == value:
            continue
        elif _a_joinable(out[first], value, name):
            out[first] = _join_twins(out[first], value)
            continue
        out[key] = _fold_twins(value, depth + 1)
    return out


def _a_id_score(rows: list[dict[str, Any]], key: Any) -> tuple[bool, float] | None:
    """How well a column serves as the rows' identifier: (its name says id, mean length), or None when some row's value
    does not look like an id or two rows share one."""
    named = bool(re.search(ID_FIELD, str(key), re.I))
    values = [row.get(key) for row in rows]
    if not all((_id_like(v, named) for v in values)):
        return None
    values = [v.strip() for v in values]
    if len(rows) > 1 and len(set(values)) < len(values):
        return None
    return (named, sum(map(len, values)) / len(values))


def _a_crash_entry(where: str, exc: Exception) -> dict[str, Any]:
    return {'kind': 'soft_crash', 'where': where, 'error': f'{type(exc).__name__}: {exc}'[:300]}


def _a_children(node: Any) -> Any:
    """The values below a dict or list node, or None for a leaf."""
    if isinstance(node, dict):
        return node.values()
    return node if isinstance(node, list) else None


_H_DEFAULT_STEPS = 30


_H_ENV_TRIES = 4


_H_FINAL_MARKS = ('finish', 'quarantin', 'finaliz')


_H_SKIP_MARKS = ('reuse', 'different action')


_H_GIVE_UP = (400, 401, 402, 403, 404, 422)


_H_PLAIN_BUDGET = '\\s*\\d+(\\.\\d+)?\\s*'


_H_PRICE_TEXT = '\\s*[^\\d\\s]{0,3}\\s?(\\d[\\d,]*(?:\\.\\d+)?)\\s*[A-Za-z]{0,6}\\.?\\s*'


_H_CAP_WORDS = 'max|most|under|below|limit|cap|upto|up_to'


_H_MONEY_WORDS = 'price|cost|amount|budget|spend|money'


_H_STAGES = ('choose', 'order', 'end')


_H_MODE_FORMAT = '{"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase}'


_H_MODE_TAIL = '\nReturn ONE JSON object and nothing else: ' + _H_MODE_FORMAT + ' (only the arguments that choose the mode).'


_H_BARE_LIMIT = 2


def _h_number(value: Any) -> bool:
    """A real int or float (a bool is not a number here)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _h_tool_ok(tool: Any) -> bool:
    if not isinstance(tool, dict):
        return False
    spec = tool.get('function')
    return isinstance(spec, dict) and isinstance(spec.get('name'), str)


def _h_parameters(spec: dict[str, Any]) -> dict[str, Any]:
    params = spec.get('parameters')
    if isinstance(params, dict):
        return params
    return {'type': 'object', 'properties': {}}


def _h_tool_line(spec: dict[str, Any]) -> str:
    return f"- {spec['name']}: {spec.get('description', '')} | parameters: {_dump(spec.get('parameters') or {})}"


def _h_usable_brief(brief: dict[str, Any]) -> bool:
    """A brief names at least one item and a checkout tool."""
    if not _items(brief.get('items')):
        return False
    roles = brief.get('roles')
    return isinstance(roles, dict) and bool(roles.get('buy'))


def _h_must_key(must: str) -> str:
    return _words(must.split(':')[0])


def _h_clash(kept: list[str], incoming: list[str]) -> bool:
    """Two musts lists name the same property with different values somewhere."""
    for first in kept:
        for second in incoming:
            if _h_must_key(first) != _h_must_key(second):
                continue
            if _words(first.partition(':')[2]) != _words(second.partition(':')[2]):
                return True
    return False


def _h_agree(x: str, y: str) -> bool:
    """Two attribute texts say the same: equal, or the shorter's words all inside a not much longer one."""
    if x == y:
        return True
    left = re.findall('[a-z0-9]+', x.lower())
    right = re.findall('[a-z0-9]+', y.lower())
    if len(left) > len(right):
        left, right = right, left
    if not left:
        return False
    return set(left) <= set(right) and len(right) <= 2 * len(left) + 2


def _h_words3(text: str) -> set[str]:
    return set(re.findall('[a-z]{3,}', text))


def _h_recurring(values: list[str]) -> int:
    """How many values share a 3+ letter word with the other values of the key."""
    shared = 0
    for value in values:
        rest = ' '.join(other for other in values if other is not value)
        if _h_words3(value) & _h_words3(rest):
            shared += 1
    return shared


def _h_named_ids(value: Any, shown: str) -> list[str]:
    """Listing ids an argument named: a lone text always, list members only when the page shows them."""
    if not isinstance(value, list):
        return [value.strip()] if isinstance(value, str) else []
    return [entry.strip() for entry in value if isinstance(entry, str) and entry.strip() in shown]


_J_FIRST_WALK_ROOM = 7  # steps beyond the finish the first capped listing of a cart line (and a closing page) needs


def _j_below(price: Any) -> int:
    """The whole-number price limit just under `price`: one less for a whole price, the floor otherwise."""
    whole = int(price)
    return whole - 1 if float(price).is_integer() else whole


def _j_plain_number(value: Any) -> bool:
    """A real int or float argument value (a flag is not a number here)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _j_query_keys(call: dict[str, Any]) -> list[str]:
    """The arguments of a past call that carried search words: non-blank text under a name that says query."""
    return [name for name, given in call.items()
            if isinstance(given, str) and given.strip() and re.search(QUERY_WORDS, _snake(name), re.I)]


def _j_words4(text: str) -> list[str]:
    """The runs of four or more lower-case letters in a text."""
    return re.findall('[a-z]{4,}', text)


def _j_fact_line(record: dict[str, Any]) -> str:
    """The short text values of a page's attribute block, joined by spaces."""
    block = _fact_block(record) or {}
    return ' '.join(str(text) for text in block.values() if isinstance(text, str) and len(text) <= HUNT_VALUE_CHARS)


def _j_stem_tries(values: list[str], noun: str) -> list[str]:
    """The first searches for one item, without repeats: all required values with the noun, the values alone, all values
    but the last with the noun, and the noun alone."""
    tries: list[str] = []
    if values:
        tries.append(' '.join(values + [noun]).strip())
        tries.append(' '.join(values))
    if len(values) >= 2:
        tries.append(' '.join(values[:-1] + [noun]).strip())
    if noun:
        tries.append(noun)
    return list(dict.fromkeys(tries))


_K_QUIET_ROLES = ('ask', 'cart', 'cart_view', 'buy', 'refuse')


_K_SEARCH_AGAIN = ' (search again with the new values where needed)'


_K_POINTED_HEAD = ' These take their value from another listing: '


_K_POINTED_STEPS = ('. Find that listing, open its page in full, and open every listing its page refers to;'
                    + ' the last page gives the value')


_K_VALUE_HEAD = ("[K2 searched each such item's requirement values together"
                 + ' and read the first fitting-kind rows] ')


def _k_whole(value: Any) -> bool:
    """An integer that is not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _k_without_numbers(args: dict[str, Any]) -> dict[str, Any]:
    """The arguments with their whole numbers taken out (flags and text stay, in order)."""
    return {key: value for key, value in args.items() if not _k_whole(value)}


def _k_floor(spec: Any) -> Any:
    """Where a numeric argument starts: the schema's integer minimum, else 1."""
    least = spec.get('minimum')
    return least if isinstance(least, int) else 1


def _k_folded(value: Any) -> Any:
    """Text in lower case with single spaces; anything else as it is."""
    if not isinstance(value, str):
        return value
    return ' '.join(value.lower().split())


def _k_has_content(value: Any) -> bool:
    """A non-empty list or text, or a positive number."""
    if isinstance(value, (list, str)):
        return bool(value)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _k_first_of_kind(props: Any, kind: str) -> str | None:
    """The first property whose schema allows `kind`."""
    for key, spec in props.items():
        if isinstance(spec, dict) and kind in _kinds(spec):
            return key
    return None


def _k_text_holder(spec: Any) -> str | None:
    """The first text field inside an object argument."""
    if not isinstance(spec, dict):
        return None
    return _k_first_of_kind(spec.get('properties') or {}, 'string')


def _k_message_rank(key: str, spec: dict[str, Any]) -> tuple[bool, bool, Any]:
    """How well a text field suits the message: not an id-like name, a message-like name, then the most room."""
    snake = _snake(key)
    return (not re.search(NOT_MESSAGE, snake, re.I), bool(re.search(MESSAGE_FIELD, snake, re.I)),
            spec.get('maxLength') or 10 ** 6)


def _k_mode_choice(key: str, options: list[Any], last: dict[str, Any], required: list[Any]) -> list[tuple[str, Any]]:
    """The option a mode argument takes for a words search, as a one-pair list: the single option that names a query,
    else the option not used last time of a required two-way mode; [] when neither fits."""
    reads = [option for option in options if isinstance(option, str) and re.search(QUERY_WORDS, option, re.I)]
    if len(reads) == 1:
        return [(key, reads[0])]
    unused = [option for option in options if option != last.get(key)]
    if key in required and len(options) == 2 and len(unused) == 1:
        return [(key, unused[0])]
    return []


_M_MEASURE = '^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$'


_M_MODEL_KEY = '\\w[ _]models?\\b'


_M_ALNUM = '[A-Za-z0-9]+'


_M_FILLER = '\\b(?:what|which|would|you|like|for|the|please|tell|me|and|or)\\b'


_M_DONE_HEAD = 'Before going on, K2 did the steps the request asks for:\n'


_M_CHANGED = "The shopper's reply changes the task: "


_M_FIX_CART = '. Fix the cart for it, then check out again.'


_M_EVIDENCE_HEAD = ("[evidence: keys whose names contain a requirement's property;"
                    + ' one naming a different property (another part or aspect) decides nothing,'
                    + ' and a key with another name may still hold the same property] ')


def _m_bump(counts: dict[int, int], number: int) -> None:
    """Add one to a per-item tally."""
    counts[number] = counts.get(number, 0) + 1


def _m_even_shares(numbers: list[int], slots: int) -> list[list[int]]:
    """Cut `numbers` into `slots` consecutive runs whose sizes differ by at most one, the longer runs first."""
    runs: list[list[int]] = []
    rest = numbers
    for open_slots in range(slots, 0, -1):
        take = -(-len(rest) // open_slots)
        runs.append(rest[:take])
        rest = rest[take:]
    return runs


def _m_either(labels: list[str]) -> str:
    """'a, b or c' for a list of two or more labels."""
    return f"{', '.join(labels[:-1])} or {labels[-1]}"


def _m_middle_out(labels: list[str]) -> list[str]:
    """The labels reordered from the middle outwards (the earlier one first at equal distance)."""
    centre = (len(labels) - 1) / 2
    return [labels[i] for i in sorted(range(len(labels)), key=lambda i: (abs(i - centre), i))]


def _m_mostly_measures(values: list[str]) -> bool:
    """Whether more than half of a key's values read as a number with an optional short unit."""
    return 2 * len([v for v in values if re.match(_M_MEASURE, v)]) > len(values)


def _m_is_property(key: str) -> bool:
    """Keys naming an option, code, link or similar are not shopper properties; '... model' keys are."""
    return not re.search(NOT_PROPERTY, key) or bool(re.search(_M_MODEL_KEY, key))


def _m_catch_all_last(keys: list[str]) -> list[str]:
    broad = [key for key in keys if re.search(CATCH_ALL, key)]
    return [key for key in keys if key not in broad] + broad


def _m_owner_score(words: str, roots: set[str], item: Any) -> float:
    """How well a category label fits an item: its shelf exactly, its name exactly, else the word overlap."""
    if words == _words(item.shelf or ''):
        return 3.0
    if words == _words(item.name):
        return 2.0
    theirs = _roots(item.shelf or '') | _roots(item.name)
    return len(roots & theirs) / max(1, len(roots | theirs))


def _m_spelled_words(text: str) -> frozenset[str]:
    """The spelled-out word set of a text's letter and digit runs."""
    return _spelled(' '.join(re.findall(_M_ALNUM, text)))


def _m_record_text(record: dict[str, Any]) -> str:
    """A record's top-level texts, then each nested block's 'key value' texts."""
    flat = ' '.join(str(value) for value in record.values() if isinstance(value, str))
    pairs = [f'{key} {value}' for block in record.values() if isinstance(block, dict)
             for key, value in block.items() if isinstance(value, str)]
    return flat + ' ' + '. '.join(pairs)


def _m_crowded(heard: dict[str, Any]) -> bool:
    """Whether the reading says the shopper wants one question at a time."""
    return any(_count(x) == 1 for x in _items(heard.get('crowded')))


_P_PLAN_HEAD = ("[fit plan: a device fits a part when the part's text names the device's model code"
                + " or another name it is sold as, or when the device's text names the series of the part's code;"
                + " try one search of a code or series with the item's words,"
                + " and if it finds nothing list the device's category and read the codes on its rows] ")


_P_HOLD_HEAD = ('The order counts only after every page a requirement points at has been opened in full'
                + ' (a search row is not enough), following each page that refers to a further listing.'
                + ' Still to find and open: ')


_P_HOLD_TAIL = (". Search for it by its shop's name together with the values its description gives"
                + " (the shop's other listings of that kind are not it),"
                + ' open the row of that shop whose page shows those values, apply the value its page gives, then check out.')


_P_SHOP_PAGE_NOTE = (': an opened page of the shop this chain names that shows the values its description gives;'
                     + ' read the value from it')


_P_REFUSE_PUSH = ('Search for those values now and buy what fits;'
                  + ' end without a purchase only if nothing does.')


_P_FILLER = frozenset({'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'})


def _p_mark(index: dict[str, Any], ident: str) -> dict[str, Any]:
    """The fit marks the index keeps for one listing ({} when it has none)."""
    return index['marks'].get(ident) or {}


def _p_tags(index: dict[str, Any], ident: str, field: str) -> set[str]:
    """One kind of fit mark of a listing (its series, the series it names, its bars...) as a set."""
    return set(_p_mark(index, ident).get(field) or [])


def _p_hits_any(patterns: Any, codes: Any) -> bool:
    """Whether some pattern covers some code."""
    return any(_code_hit(pattern, code) for pattern in patterns for code in codes)


def _p_codes_fit(fits: Any, bars: Any, codes: Any) -> bool:
    """A part fits a device when a code it works with covers one of the device's codes and none it rules out does."""
    return _p_hits_any(fits, codes) and not _p_hits_any(bars, codes)


def _p_snippet(value: Any, depth: int, width: int) -> str:
    """A clipped JSON rendering of a record, cut to `width` characters."""
    return _dump(_clip(value, depth))[:width]


def _p_word_in(word: str, text: str) -> bool:
    """Whether `word` stands in `text` as a whole token (no letter or digit glued to either side)."""
    return re.search('(?<![a-z0-9])' + re.escape(word) + '(?![a-z0-9])', text) is not None


def _p_tokens3(text: str) -> set[str]:
    """The runs of three or more letters or digits in a text."""
    return set(re.findall('[a-z0-9]{3,}', text))


def _p_scalars(node: Any) -> Any:
    """Every leaf value under a record, as text (dict values and list entries are walked into)."""
    if isinstance(node, dict):
        node = list(node.values())
    if not isinstance(node, list):
        yield str(node)
        return
    for child in node:
        yield from _p_scalars(child)


def _p_name_fields(row: dict[str, Any], skip: tuple[Any, ...], floor: int) -> list[str]:
    """The short text fields of a row that could carry a seller's or maker's name: not under a key in `skip`, longer than
    `floor` and at most 40 characters once trimmed, and not a placeholder such as 'none'."""
    fields = []
    for key, value in row.items():
        if not isinstance(value, str) or key in skip:
            continue
        trimmed = value.strip()
        if floor < len(trimmed) <= 40 and trimmed.lower() not in NO_NAME:
            fields.append(trimmed)
    return fields


def _p_changed_keys(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Plain fields of a row whose value differs between two reads of it."""
    return [key for key, value in after.items()
            if key in before and before[key] != value and not isinstance(value, (dict, list))]


_Q_SKIPPED = 'Not run (no step used): '


_Q_WAITING = 'Not run yet (no step used). '


_Q_PENDING = 'Not run yet. '


_Q_TAG = '\n[K2] '


_Q_CART_HEAD = '\nCart now: '


_Q_JOINED = 'Shown in the side-by-side result above: K2 opened these listings together in one step.'


_Q_MOVED_HEAD = '\n\nK2 has searched for each item and read some of its listings (one step each):\n'


_Q_MOVED_TAIL = '\nJudge these against the requirements and search further only for what is still missing.\n\n'


_Q_START = ('\n\nStart by searching for each item in its category and opening several listings in full,'
            + ' so the interviewer can see which properties these listings carry.\n\n')


_Q_LATE_DECLINE = ('Ending without a purchase earns nothing when the store has a fitting listing,'
                   + ' and running short of steps does not show that it has none. ')


_Q_LATE_DECLINE_END = (' End without a purchase only if your search showed that some requested item has no listing'
                       + ' that meets its requirements; then send the ending call again.')


_Q_LEADS_END = ('. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out.'
                + ' End without a purchase only if each of them clearly fails a requirement;'
                + ' then say which one in your text and send the ending call again.')


_Q_LOST_CALL = 'The call may not have run: something went wrong on the way. Look at the state before you repeat it.'


def _q_crash_note(exc: Exception) -> dict[str, Any]:
    """The trace entry for an exception that broke off a stretch of turns."""
    return {'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}


def _q_plain_text(content: Any) -> str:
    """The text part of a model message: a string as it is, anything else dumped (nothing when empty)."""
    if isinstance(content, str):
        return content
    return _dump(content) if content else ''


def _q_tool_names(names: set[str]) -> str:
    return ', '.join(sorted(names))


class _q_Step:
    """One tool call of a model turn on its way through K2's checks: the call as it will be sent, what is known
    about it so far, and the notes that go after the store's result in its reply."""

    def __init__(self, name: Any) -> None:
        self.name = name
        self.args: dict[str, Any] = {}
        self.notes: list[str] = []
        self.many: str | None = None
        self.target: str | None = None
        self.ending = False
        self.fresh = False
        self.mark = ''
        self.readers: set[str] = set()



class K2:

    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.tools = [tool for tool in self.view.get('tools') or [] if _h_tool_ok(tool)]
        self.schemas = {tool['function']['name']: _h_parameters(tool['function']) for tool in self.tools}
        self.max_steps = self._h_step_cap(self.view.get('max_steps'))
        self.started = monotonic()
        self.step = 0
        self.finished = bool(self.view.get('finished'))
        self._h_spend_state()
        self._h_brief_state()
        self._h_cart_state()
        self._h_lead_state()
        self._h_view_state()
        self._h_search_state()

    def left(self) -> int:
        return self.max_steps - self.step

    def clock(self) -> float:
        return monotonic() - self.started

    def model_room(self) -> float:
        usable = TASK_SECONDS - TIME_MARGIN
        return usable - self.clock()

    def rushed(self) -> bool:
        """True when the time left cannot hold two store calls at this run's pace (at least a second each):
        then a read the order does not need is skipped."""
        if self.env_count:
            pace = max(self.env_spent / self.env_count, 1.0)
        else:
            pace = 1.0
        return pace * 2 >= self.model_room()

    def msgs_free(self) -> int:
        held_back = max(0, self.reserve - self.gate_msgs)
        return self.msg_limit - self.msg_used - held_back

    def cart_mark(self) -> str:
        lines = sorted(self.cart_lines.items())
        return _dump(lines)

    def row_ids(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Arguments of a listing or cart call with any code that exactly one seen title carries swapped for that row's id."""
        if not self.rows or name not in self._h_role_tools(('open', 'offer', 'compare', 'cart')):
            return args
        title = self.title_key()
        if not title:
            return args
        props = (self.schemas.get(name) or {}).get('properties') or {}
        swapped: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, list):
                swapped[key] = [self._h_code_row(entry, key, props, title) for entry in value]
            else:
                swapped[key] = self._h_code_row(value, key, props, title)
        return swapped

    def page_tools(self) -> set[str]:
        single = self.roles.get('offer') or self.roles.get('open')
        return set(filter(None, (single, self.roles.get('compare'))))

    def facts_tool(self) -> str | None:
        # a tool already seen showing attributes wins; else the opener, unless it was seen showing none
        learned = None
        for tool, shows in self.facts_from.items():
            if shows and self.single_target(tool):
                learned = tool
                break
        if learned:
            return learned
        opener = self.roles.get('open')
        if self.facts_from.get(opener or '') is False or not self.single_target(opener):
            return None
        return opener

    def compare_facts(self) -> bool | None:
        side_by_side = self.roles.get('compare') or ''
        return self.facts_from.get(side_by_side)

    def compare_bare(self) -> bool:
        seen = self.compare_facts()
        if seen is not None:
            return seen is False
        # untested: assume it is bare when another page is known to carry the attributes
        return self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))

    def gate_due(self, stage: str) -> bool:
        if not self._b_stage_steps(stage):
            return False
        return (stage, f'{self.cart_mark()}#{self.version}') not in self.gate_ok

    def tail(self) -> int:
        speaks = any(entry['messages'] > 0 for entry in self._b_stage_steps('order'))
        if speaks and self.msg_used < self.msg_limit and self.gate_due('order'):
            return 2
        return 1

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        if not self.cost <= SPEND_HARD:
            return []
        room_for_heavy = self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD
        return [HEAVY, LIGHT] if room_for_heavy else [LIGHT]

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if not self._b_may_call():
                return None
            self.calls += 1
            message, status = self._b_with_retries(model, messages, max_tokens, tools)
            if message is not None:
                return message
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        cacheable = model.startswith('anthropic/') and len(messages) > 1
        body: dict[str, Any] = {'model': model, 'messages': _b_cache_marked(messages) if cacheable else messages,
                                'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body.update(tools=tools, tool_choice='auto')
        result = _llm_within(self.model_room()).post_verbose('/inference/chat/completions', body)
        data = result.data
        if not (isinstance(data, dict) and data.get('choices')):
            return None, self._b_failure_status(model, result)
        usage = data.get('usage') or {}
        spent = usage.get('cost')
        if isinstance(spent, (int, float)):
            self.cost += float(spent)
        message = data['choices'][0].get('message') or {}
        if message.get('content') or message.get('tool_calls'):
            return message, 200
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
        return None, 0

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        for model in models:
            reply = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens) or {}
            parsed = _json_obj(reply.get('content'))
            if parsed is not None:
                return parsed
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                break
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = {**self.binding, 'step': self.step + 1, 'action': {'name': name, 'args': args}}
        self._h_space_out()
        for _ in range(_H_ENV_TRIES):
            result = self._h_timed_post(payload)
            if isinstance(result.data, dict):
                return self._h_took_step(name, args, result.data)
            if not self._h_after_refusal(result, payload):
                return None
        return None

    def unspelled(self, name: str, observation: Any) -> Any:
        """The result as the agent reads it; a property that a listing's own page says is taken from another listing
        is marked as not stated on that page (R1), so neither the checks nor the model take the page's value for it."""
        return _without_handed_off(self._spelled_as_given(name, observation))

    def _spelled_as_given(self, name: str, observation: Any) -> Any:
        """The result as the agent reads it (_restore, twins folded). On a listing view, a record with enough plain values
        under keys no search row has gets those values gathered into one 'attributes' object."""
        pages = self._h_role_tools(('open', 'offer', 'compare'))
        on_page = name in pages
        readable = _fold_twins(_restore(observation, page=on_page or name != self.roles.get('search')))
        if not on_page:
            return readable
        self.row_keys.update(self._h_unopened_keys())
        search_keys = self.row_keys
        if not search_keys:
            return readable
        if not isinstance(readable, dict):
            return readable
        if self.ident(readable):
            return self._h_gather(readable, search_keys)
        folded = {}
        for key, value in readable.items():
            folded[key] = [self._h_gather(entry, search_keys) for entry in value] if isinstance(value, list) else value
        return folded

    def read_request(self) -> None:
        prompt = self._h_brief_prompt()
        brief = self._h_ask_brief(prompt)
        if brief:
            self.waited = 0
        self.brief = brief
        raw_roles = brief.get('roles')
        if not isinstance(raw_roles, dict):
            raw_roles = {}
        self.roles = {job: tool for job, tool in raw_roles.items() if isinstance(tool, str) and tool in self.schemas}
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        self._h_split_modes(brief)
        self.items.extend(self._h_item_of(entry) for entry in _items(brief.get('items')) if isinstance(entry, dict))
        renumbered = self._h_merge_items()
        if not self.items:
            self.items = [Item('the requested product', '', 1, [])]
        self._h_bind_fits(brief, renumbered)
        for lookup in _items(brief.get('lookups')):
            number, text = self._h_lookup_ref(str(lookup).strip())
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.unmix()
        self._h_read_terms(brief)
        self._h_read_process(brief)

    @staticmethod
    def shared_shelf(brief: dict[str, Any]) -> bool:
        labels: set[str] = set()
        for entry in _items(brief.get('items')):
            if not isinstance(entry, dict):
                continue
            label = _words(str(entry.get('shelf') or ''))
            if label in labels:
                return True
            if label:
                labels.add(label)
        return False

    def unmix(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = [need for owner, need in self.refs if owner == number and need in item.musts]
            others = [must for must in item.musts if must not in pointers]
            if pointers and others:
                self._b_unmix_item(number, item, pointers, others)

    def described(self, number: int, need: str) -> bool:
        if self._b_is_pointer(need):
            return False
        prop, _, value = need.partition(':')
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
            return False
        item = self.items[number - 1]
        if self._b_names_other_item(item, value):
            return False
        keys, read = self._b_item_keys(item)
        if read < 3:
            return False
        wanted = _stem(prop)
        return all(not _stem(key) & wanted for key in keys)

    def ledger(self) -> str:
        lines = [self._b_item_line(number, item) for number, item in enumerate(self.items, 1)]
        lines += self._b_terms_lines(self.brief)
        lines += [f"step the request asks for, {_B_WHEN_TEXT[entry['when']]}: {entry['step']}" for entry in self.process]
        facts = self._b_facts_line()
        if facts:
            lines.append(facts)
        return '\n'.join(['- ' + line for line in lines])

    def status(self) -> str:
        left = self.left()
        pieces = [f'[K2] step {self.step} of {self.max_steps} used, {left} left.', self._b_pace_note(left)]
        if self.clock() >= WIND_DOWN and left > self.tail() + 1:
            pieces.append(_B_CLOCK_NOTE)
        live = {need for _, need in self.needs_now()}
        unresolved = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if unresolved and self.buys == 0:
            pieces += [_B_RESOLVE_HEAD, ' | '.join(unresolved[:3])]
        if self.leads:
            pieces += ['\n[review candidates] ', ', '.join(self.leads)]
        if self.bound and self.buys == 0:
            plan = self.fit_plan()
            if plan:
                pieces += ['\n', plan]
        if self.notes:
            pieces += ['\n[changes so far] ', ' | '.join(self.notes[-4:])]
        pieces += ['\n[ledger]\n', self.ledger()]
        return ''.join(pieces)

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        key = self.id_key
        if key and _id_like(node.get(key), True):
            return node[key].strip()
        if not in_list and depth != 0 and not any(isinstance(v, dict) for v in node.values()):
            return None
        guess = _learn_id([node])
        if not guess:
            return None
        candidate = node[guess].strip()
        return candidate if 'id' in str(guess).lower() or candidate in self.rows else None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [entry for entry in node if isinstance(entry, dict) and len(entry) >= 3]
        if not rows or 2 * len(rows) < len(node):
            return False
        key = self.id_key
        if key and all(_id_like(row.get(key), True) for row in rows):
            return True
        guess = _learn_id(rows)
        if not guess:
            return False
        self.id_key = guess
        return True

    def absorb(self, observation: Any) -> None:
        self._b_visit(observation, 0, False)

    def labels(self) -> set[str]:
        """Every enum option in the tools' schemas (lower case), walked depth first to depth 6."""
        found: set[str] = set()
        pending = [(schema, 0) for schema in reversed(list(self.schemas.values()))]
        while pending:
            schema, depth = pending.pop()
            if depth > 6 or not isinstance(schema, dict):
                continue
            found.update(option.strip().lower() for option in schema.get('enum') or [] if isinstance(option, str))
            pending.extend((child, depth + 1) for child in reversed(list(schema.values())) if isinstance(child, dict))
        return found

    def category_key(self) -> str | None:
        tally = self._c_short_texts()
        total = max(1, len(self.rows))
        known = self.labels()
        ranked = [(_c_category_rank(values, hits, known), key) for key, (values, hits) in tally.items() if hits >= 0.8 * total and len(values) <= 40]
        if not ranked:
            return None
        return min(ranked, key=lambda entry: entry[0])[1]

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        groups = self._h_fact_groups(self.category_key())
        item_keys = self._c_item_keys()
        wanted = {label for label in groups if self._c_label_wanted(label, item_keys)}
        if wanted:
            groups = {label: group for label, group in groups.items() if label in wanted}
        table = []
        for label, group in sorted(groups.items(), key=lambda pair: -pair[1]['n']):
            for name, slot in self._h_group_keys(group)[:TABLE_KEYS]:
                table.append((label, name, slot['n'], [text[:40] for text in slot['values'][:4]]))
        return table

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        number = self.owner_of(label)
        if not number:
            return ''
        for must in self.items[number - 1].musts:
            prop, _, stated = must.partition(':')
            if re.search(_C_RANGE_WORDS, stated, re.I):
                continue
            need = _c_alnum3(stated)
            on_show = _c_alnum3(' '.join(values))
            title = prop.strip()
            if not title or _words(prop) == name or not _stem(prop) & _stem(name):
                continue
            if need and not need <= on_show:
                return f' [not the stated {title}: another property]'
        return ''

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        first, second = set(name.split()), set(other.split())
        if not (first and second and first & second):
            return False
        mine, theirs = spec['plain'], slot['plain']
        both = [ident for ident in mine if ident in theirs and mine[ident].strip() and theirs[ident].strip()]
        agreeing = len([ident for ident in both if _h_agree(mine[ident], theirs[ident])])
        nested = first <= second or second <= first
        if not nested:
            return len(both) >= 2 and agreeing == len(both)
        if not both:
            return bool((set(spec['values']) & set(slot['values'])) - {''})
        return agreeing * 2 > len(both)

    def item_groups(self) -> list[list[str]]:
        cat_key = self.category_key()
        by_label: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or '') if cat_key else ''
            by_label.setdefault(label, []).append(ident)
        item_keys = self._c_item_keys()
        pool = [ids for label, ids in by_label.items() if self._c_label_wanted(label, item_keys)] or list(by_label.values())
        pool.sort(key=len, reverse=True)
        return pool[:max(1, len(self.items))]

    def single_target(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or '') or {}
        target = self._c_text_slot(schema, True)
        if not tool or not target:
            return None
        return target if all(k == target for k in schema.get('required') or []) else None

    def price_of(self, ident: str) -> float | None:
        if not self.price_key:
            return None
        value = (self.rows.get(ident) or {}).get(self.price_key)
        if isinstance(value, dict):
            value = next((part for part in value.values() if _h_number(part)), None)
        elif isinstance(value, str):
            figure = re.fullmatch(_H_PRICE_TEXT, value)
            if not figure:
                return None
            value = float(figure.group(1).replace(',', ''))
        return float(value) if _h_number(value) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        if not self.stock_key:
            return True
        stock = (self.rows.get(ident) or {}).get(self.stock_key)
        if isinstance(stock, str):
            return re.search(STOCK_OUT, stock.strip(), re.I) is None
        if stock is False:
            return False
        return not (_h_number(stock) and stock <= 0)

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, list):
            return self._h_listed(observation)
        if not isinstance(observation, dict):
            return []
        whole = self.ident(observation)
        if whole:
            return [(whole, observation)]
        found: list[tuple[str, dict[str, Any]]] = []
        for value in observation.values():
            if isinstance(value, list):
                found.extend(self._h_listed(value))
                continue
            if not isinstance(value, dict):
                continue
            inner = self.ident(value, False, 1)
            if inner:
                found.append((inner, value))
                continue
            for nested in value.values():
                if isinstance(nested, list):
                    found.extend(self._h_listed(nested))
        return found

    def note_view(self, name: str, observation: Any) -> None:
        for ident, record in self.records_in(observation):
            fields = set(record)
            self.tool_keys.setdefault(name, set()).update(fields)
            self.view_keys[ident] = (self.version, self._c_current_view(ident) | fields)

    def shown_already(self, name: str, ident: str) -> bool:
        wanted = self.tool_keys.get(name)
        return bool(wanted) and wanted <= self._c_current_view(ident) and ident not in self.gone

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.shapes[name] = None
        got = self.ask_json(self._c_shape_prompt(name, observation, lists), [LIGHT, HEAVY], 2500) or {}
        rows = got.get('rows')
        if rows not in lists or not rows:
            return None
        shape = self._c_shape_from(name, got, observation, rows)
        self.shapes[name] = shape
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.read_shape(name, observation)
        if not shape or not (shape['count'] and shape['next'] and shape['next_arg']):
            return None
        if not isinstance(observation, dict):
            return None
        rows = observation.get(shape['rows'])
        total = observation.get(shape['count'])
        cursor = observation.get(shape['next'])
        if not (isinstance(rows, list) and rows and _c_is_whole(total)):
            return None
        usable = cursor is None or (isinstance(cursor, (str, int)) and not isinstance(cursor, bool))
        return (total, len(rows), shape['next_arg'], cursor) if usable else None

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(ident) for ident, _ in self.records_in(observation)]
        if len(prices) < 3 or None in prices:
            return False
        return all(low <= high for low, high in zip(prices, prices[1:]))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        if pick not in ('cheapest', 'most', 'least'):
            return ''
        plan = self.paging(name, observation)
        if not plan:
            return ''
        count, shown, param, cursor = plan
        if cursor is None or _c_past_first_page(args.get(param)) or count <= shown:
            return ''
        if pick == 'cheapest' and self.price_sorted(observation):
            return ''
        pages = self._c_pages_to_read(count, shown)
        if pages is None:
            return ' '.join((f'\n[K2] This search matched {count} listings and showed {shown}, in no price order.',
                             "The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with " + f'{param}={_dump(cursor)}.'))
        texts, cursor = self._c_read_pages(name, args, param, cursor, pages)
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        if not texts:
            return ''
        report = '\n[K2 read the next pages of this search itself, one step a page]\n' + '\n'.join(texts)
        if cursor is not None:
            report += f'\n[K2] {count} listings matched and K2 has read {shown * (len(texts) + 1)} rows of them; ' + "narrow the search with the requirement's value or page on to see the rest."
        return report

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A note when a search sent a price cap but still returned rows above it."""
        props = (self.schemas.get(name) or {}).get('properties') or {}
        limits = self.cap_keys(name)
        lone = limits[0] if len(limits) == 1 else None
        caps = []
        for key, value in args.items():
            if key not in props or not _h_number(value):
                continue
            if re.search(_H_CAP_WORDS, key, re.I) and self._h_priced_arg(props, key) or key == lone:
                caps.append(key)
        if not caps or not self.price_key:
            return ''
        tightest = min(caps, key=lambda key: args[key])
        cap = args[tightest]
        over = len([ident for ident, _ in self.records_in(observation) if (self.price_of(ident) or 0) > cap])
        if not over:
            return ''
        return f'\n[K2] This search did not apply {tightest}={cap:g}: {over} of its rows cost more{self._h_mode_hint(args, props, tightest)}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        priced = []
        for ident in found:
            if self.buyable(ident) and self.price_of(ident) is not None:
                priced.append((self.price_of(ident), ident))
        if len(priced) < 2:
            return ''
        listed = ', '.join(f'{ident} ({price:g})' for price, ident in sorted(priced)[:6])
        return '\n[K2] buyable rows here, cheapest first: ' + listed

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get('pick') != 'cheapest':
            return ''
        if self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        held = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        cart_line = f" ({'; '.join(held)})" if held else ''
        sentences = (f'Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest.',
                     "Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order.",
                     'Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen)',
                     "and look only at buyable rows priced below the cart's line" + cart_line + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.')
        return ' '.join(sentences)

    def note_facts(self, tool: str, shown: bool) -> None:
        """Learn whether a single-listing view shows attributes: yes once any page had them, no after two bare pages."""
        if shown:
            self.facts_from[tool] = True
            return
        if self.facts_from.get(tool) is True:
            return
        bare = self.bare_pages.get(tool, 0) + 1
        self.bare_pages[tool] = bare
        if bare >= _H_BARE_LIMIT:
            self.facts_from[tool] = False

    def lists_things(self, tool: str) -> bool:
        """True for a listing tool: one whose single argument names a shelf or carries search words, or one whose
        page shape is already known."""
        argument = _snake(self.single_target(tool) or '')
        if any(re.search(pattern, argument, re.I) for pattern in (SHELF_ARG, QUERY_WORDS)):
            return True
        return tool in self.shapes

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Open the pages of listings not read in full yet (several at a time through the compare tool when it answers,
        then one by one), while more than `keep` steps remain; the page texts, for the model."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        direct = bool(tool and target)
        if not direct and self.compare_facts() is not True:
            return []
        texts: list[str] = []
        queue = [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i)]
        unread, queue = self._j_compare_reads(queue, keep, texts)
        if not direct:
            return texts
        self._j_direct_reads(tool, target, unread + queue, keep, texts)
        if texts:
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def show_links(self, ids: list[str], keep: int) -> None:
        """For carts of several lines: follow the part code a cart line points to with one search per code, and open
        the pages that search shows while steps remain. What it finds goes to the ledger rather than the model."""
        plan = self.value_template()
        if not plan:
            return
        if len(self.items) < 2 and len(ids) < 2:
            return
        tool, base, field = plan
        for ident in ids:
            code = self._j_link_code(ident)
            if code is None:
                continue
            if self._j_out_of_room(keep):
                return
            self.link_tried.add(code)
            reply = self._j_send(tool, {**base, field: code})
            if reply is None:
                return
            self.observe(tool, reply.get('observation'))
            self.trace.append({'kind': 'show_link', 'line': ident, 'code': code})
            self.fill_facts(self._j_owned_by(code), max(keep, self.tail() + 6))

    def cap_template(self) -> tuple[str, dict[str, Any], str, str] | None:
        """The first tool with exactly one shelf argument and one price-cap argument, with a value for each of its
        option arguments: the mode a held limit was seen under, else the mode of the latest shelf listing without
        search words, else the one option that names a listing. None when such a tool has an option with no clear value."""
        found = next(((tool, schema, caps[0], shelves[0]) for tool, schema in self.schemas.items()
                      for caps, shelves in [self._j_cap_and_shelf(tool, schema)] if len(caps) == 1 and len(shelves) == 1), None)
        if found is None:
            return None
        tool, schema, cap, shelf = found
        props = (schema or {}).get('properties') or {}
        learned = [(dict(mode), held) for (key, mode), held in self.cap_modes(tool).items() if key == cap]
        choices = {key: spec['enum'] for key, spec in props.items()
                   if isinstance(spec, dict) and isinstance(spec.get('enum'), list) and len(spec['enum']) >= 2}
        single = len(choices) == 1 and next(iter(choices)) in ((schema or {}).get('required') or [])
        base: dict[str, Any] = {}
        for key, options in choices.items():
            pick = self._j_mode_pick(key, options, props, shelf, learned, single)
            if pick is None:
                return None
            base[key] = pick
        return (tool, base, shelf, cap)

    def cap_keys(self, tool: str) -> list[str]:
        """Numeric arguments of `tool` that may cap a listing's price, best evidence first: those searches showed
        holding a limit, then those named both as a maximum and a price, then any priced one, then any maximum."""
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        numbers = [key for key, spec in props.items() if isinstance(spec, dict) and {'integer', 'number'} & set(_kinds(spec))
                   and not re.search(CAP_NOT, key, re.I) and not re.search(CAP_LOW, _snake(key), re.I)]
        held = {key for (key, _), ok in self.cap_modes(tool).items() if ok}
        flags = {key: (key in held, bool(re.search(CAP_ARG, _snake(key), re.I)), bool(re.search(CAP_PRICED, _snake(key), re.I)))
                 for key in numbers}
        for needed in ((0,), (1, 2), (2,), (1,)):
            group = [key for key, marks in flags.items() if all(marks[i] for i in needed)]
            if group:
                return group
        return []

    def cap_modes(self, tool: str) -> dict[tuple[str, tuple[tuple[str, Any], ...]], bool]:
        """For each numeric argument of `tool` and each mode (the option arguments of the call) its searches used:
        False once a search's rows held one priced above the argument's value; True when all rows sat at or below it
        and rows picked at random from the known rows of their labels would seldom do so (under CAP_CHANCE), so the
        limit rather than the query kept dearer rows away."""
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        cat_key = self.category_key()
        pools = self._j_label_prices(cat_key)
        out: dict[tuple[str, tuple[tuple[str, Any], ...]], bool] = {}
        for mark, (name, args) in self.read_args.items():
            if name != tool:
                continue
            ids = [ident for ident, _ in self.records_in(self.read_calls.get(mark))]
            shown = [self.price_of(ident) for ident in ids]
            if not ids or None in shown:
                continue
            mode = tuple(sorted((k, v) for k, v in args.items()
                                if isinstance(props.get(k), dict) and isinstance(props[k].get('enum'), list) and isinstance(v, (str, int, float))))
            labels = {self._j_kind(ident, cat_key) for ident in ids} if cat_key else set()
            pool = [price for label in labels for price in pools.get(label, [])]
            dearest = max(shown)
            for key, value in args.items():
                if key not in props or not _j_plain_number(value):
                    continue
                slot = (key, mode)
                if dearest > value:
                    out[slot] = False
                elif out.get(slot) is not False and pool and (sum(p <= value for p in pool) / len(pool)) ** len(ids) < CAP_CHANCE:
                    out[slot] = True
        return out

    def list_below(self, cart: list[str]) -> None:
        """List each cart line's shelf under a price cap just below its price, walking the cap down while the
        listing stays capped and long, so that cheaper rows of its label come into view."""
        plan = self.cap_template()
        cat_key = self.category_key()
        if not plan or not cat_key:
            return
        for line in cart:
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            if price is None or price < 2 or not label:
                continue
            if (line, price) in self.capped:
                continue
            self.capped.add((line, price))
            if self._j_walk_line(plan, line, price, label):
                return

    def cheaper_rows(self) -> list[str]:
        """When the pick is the cheapest, once per cart state: list each cart line's shelf below its price, then open
        the lowest unread rows of the same label (one already matched by all required values first)."""
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.cart_mark() + f'#{self.version}'
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        self.list_below(cart)
        room = self.compare_room() if self.compare_facts() is True else 0
        picks: list[str] = []
        for line in cart:
            pool = self._j_cheaper_than(line, cart, cat_key)
            first = next((ident for ident in pool if ident in self.value_rows), None)
            if first is not None:
                pool = [first] + [ident for ident in pool if ident != first]
            picks.extend([ident for ident in pool[:room or 2] if ident not in picks])
        if room:
            opens = room * len(cart)
        else:
            opens = min(6, CHEAPER_OPENS * max(1, len(cart)))
        return self.fill_facts(picks[:opens], self.tail() + 6)

    @_soft('')
    def below_cart(self) -> str:
        """A note naming the unread buyable rows, of a cart line's label, that cost less than that line."""
        cat_key = self.category_key()
        cart = self.cart_ids()
        cheaper = [f'{ident} at {self.price_of(ident):g}'
                   for line in cart for ident in self._j_cheaper_than(line, cart, cat_key)[:CHEAPER_SHOWN]]
        if not cheaper:
            return ''
        return ''.join(('Still unread and below a cart line: ', ', '.join(cheaper), '.'))

    def title_key(self) -> str | None:
        """The row key that holds a listing's name. Candidates are short text keys most rows carry, with more than one
        word per value and mostly distinct values; of those with nearly the most distinct values (a seller repeats, a
        listing's name does not), the one with the shortest values."""
        cat_key = self.category_key()
        texts: dict[str, list[str]] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if key == cat_key or not isinstance(value, str) or len(value) > TITLE_CHARS:
                    continue
                texts.setdefault(key, []).append(value)
        total = len(self.rows)
        kept = []
        for key, values in texts.items():
            count = len(values)
            distinct = len(set(values))
            if count * 5 < total * 4 or sum(len(v.split()) for v in values) < count * 2 or distinct * 2 < count:
                continue
            kept.append((sum(len(v) for v in values) / count, key, distinct))
        most = max((distinct for _, _, distinct in kept), default=0)
        best = min(((mean, key) for mean, key, distinct in kept if distinct * 10 >= most * 9), default=None)
        return None if best is None else best[1]

    def hunt_words(self, number: int) -> tuple[str, list[str]]:
        """For one item: its noun (a word of its usual store label that no other item's label has, since a search also
        reads labels) and the words a further search may add, most frequent first. Words on nearly every row add
        nothing and a word on one row finds little, so only the words in between are kept."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        title = self.title_key()
        mine = [row for row in self.rows.values() if self.owns(item, str(row.get(cat_key) or ''))] if cat_key else []
        names = [str(row.get(title) or '') for row in mine] if title else []
        if cat_key:
            names.extend(_j_fact_line(record) for record in self.full.values() if self.owns(item, str(record.get(cat_key) or '')))
        noun = self._j_item_noun(number, item, [str(row.get(cat_key) or '') for row in mine])
        if not names:
            return (noun, [])
        return (noun, self._j_split_words(item, names, noun))

    def rows_of_item(self, number: int) -> int:
        """How many known rows belong to the item (all of them without a label key)."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        if not cat_key:
            return len(self.rows)
        return len([row for row in self.rows.values() if self.owns(item, str(row.get(cat_key) or ''))])

    @_soft('')
    def hunt_cheaper(self) -> str:
        """Keep searching, with no model call, while the pick is the cheapest and the store shows only part of its stock.

        A search can stop at a page cap while it reports the full total (race 14: 96 rows of 199; race 15: 49 rows of up
        to 1000, with neither price order nor a price filter), so the lowest price of a kind may lie beyond every first
        search. Race 15's search matched every word against titles, categories, brands, sellers and attribute values
        (22642 of 91849 matched words were found only in an attribute value), so an item's required values together
        list almost its whole fitting set, and one more word splits a set too long for the cap. Dropping a requirement
        widens the set: that settled 7 race-15 tasks, nearly every agent bought a dearer listing, and a rare passing
        run had kept trying new words (9 cents after 98 and 20). Each search costs one step; cheaper_rows then opens
        the lowest rows."""
        mark = _dump([self.needs_now(), self.budget, sorted(self.gone)])
        if self.brief.get('pick') != 'cheapest' or self.hunts >= HUNT_ROUNDS or self.hunt_mark == mark:
            return ''
        plan = self.value_template()
        if not plan or not self.price_key or self.clock() > HUNT_BY:
            return ''
        room = min(HUNT_STEPS, self.left() - self.tail() - HUNT_KEEP - 2 * len(self.items))
        if room < 3:
            return ''
        self.hunts += 1
        self.hunt_mark = mark
        cart = self.cart_ids()
        spent, fresh = self._j_hunt_all(plan, room)
        self.trace.append({'kind': 'hunt', 'steps': spent, 'new_rows': fresh, 'cart': cart})
        if not spent:
            return ''
        self.wider_held = True
        return ''.join((f'K2 ran {spent} more searches for the lowest price, one step each, ',
                        f'and they showed {fresh} more rows.'))

    def hunt_item(self, number: int, tool: str, base: dict[str, Any], field: str, room: int) -> tuple[int, int]:
        """At most `room` searches for one item: first a stem (its required values together, with or without the
        noun), then the stem narrowed by one further word at a time. Returns the steps spent and the rows gained."""
        tried = self.hunt_tried.setdefault(number, set())
        noun, _ = self.hunt_words(number)
        values = [value for value, _ in self.value_words(number)]
        hunt = {'number': number, 'tool': tool, 'base': base, 'field': field, 'room': room, 'values': values,
                'tried': tried, 'spent': 0, 'gained': 0, 'idle': 0, 'issued': False, 'refused': False}
        key = (number, tuple(values))
        stem = self.hunt_stem.get(key)
        if stem is None:
            stem = self._j_find_stem(hunt, noun)
            if stem is None:
                return (hunt['spent'], hunt['gained'])
            self.hunt_stem[key] = stem
        while self._j_may_widen(hunt):
            _, words = self.hunt_words(number)
            word = next((w for w in words if f'{w} {stem}'.strip().lower() not in tried and w not in stem.lower().split()), None)
            if word is None:
                break
            self._j_hunt_query(f'{word} {stem}'.strip(), hunt)
        return (hunt['spent'], hunt['gained'])

    @_soft(list)
    def hunt_fits(self) -> list[str]:
        """Keep searching, with no model call, before an ending without a purchase.

        Race 15: 495 of 2340 episodes of 13 leading agents ended without a purchase on a task with a right answer, and
        in 433 of them another agent held a cart the store's answer set accepted, so a fit was within reach; practice
        worlds show the same (a fit beyond the page cap, "no in-stock jeans list a slim fit"). A requirement's value and
        its words find listings the first searches missed, and the rows that name them are opened so the audit can
        judge their attributes. Returns the page texts read."""
        if self.fit_hunted or self.clock() > HUNT_BY:
            return []
        plan = self.value_template()
        if not plan:
            return []
        tool, base, field = plan
        keep = self.tail() + len(self.items) + 1
        spare = self.left() - keep
        if spare < 3:
            return []
        self.fit_hunted = True
        cat_key = self.category_key()
        numbers = self._j_open_items(cat_key)
        wanted = {number: {w for value, _ in self.value_words(number) for w in _value_tokens(value)} for number in numbers}
        paged = self.unread_pages(tool, wanted, min(FIT_PAGES, spare // 2))
        room = min(HUNT_STEPS, self.left() - self.tail() - HUNT_KEEP)
        spent = 0
        if room >= 3:
            spent = self._j_fit_searches(numbers, plan, room)
            if spent is None:
                return []
        picks = [ident for number in numbers for ident in self._j_fit_pool(number, wanted, cat_key)[:HUNT_FIT_OPENS]]
        read: list[str] = []
        for start in range(0, len(picks), HUNT_FIT_OPENS):
            if self._j_hunt_late():
                break
            read.extend(self.fill_facts(picks[start:start + HUNT_FIT_OPENS], keep))
        self.trace.append({'kind': 'hunt_fits', 'pages': paged, 'searches': spent, 'opened': len(read)})
        return read

    def unread_pages(self, tool: str, wanted: dict[int, set[str]], room: int) -> int:
        """Read up to `room` more pages of the searches already made that share a word with an item's required values:
        first an even share per item, then whatever room is left, before any new search words. Searches showing only other
        items' listings are skipped. Returns the steps spent. (Accepted rows were often on unread later pages of the
        episode's own searches: about a quarter of passing carts in races 13-15 were first seen past page 1.)"""
        used = 0
        cat_key = self.category_key()
        even_share = max(1, room // max(1, len(wanted)))
        for limit in (even_share, room):
            for number, words in wanted.items():
                stop = min(room, used + limit)
                for mark, (name, args) in list(self.read_args.items()):
                    if name != tool or not self._k_may_page(mark, args, words, number, cat_key):
                        continue
                    used, cut = self._k_walk_on(tool, mark, args, used, stop)
                    if cut:
                        return used
        return used

    def call_mark(self, name: str, args: dict[str, Any]) -> str:
        """A key that spells a call one way: arguments sorted, text lower-cased with single spaces, and a whole number
        left out while it sits at its starting value."""
        props = self._k_props(name)
        kept = {key: _k_folded(value) for key, value in sorted(args.items()) if not self._k_at_start(name, key, value, props)}
        return name + _dump(kept)

    def note_read(self, name: str, args: dict[str, Any], seen: Any) -> None:
        """Keep what a search or listing call showed, whoever made the call; page reads and shopper or cart calls are left."""
        if name in self.page_tools():
            return
        if any(name == self.roles.get(role) for role in _K_QUIET_ROLES):
            return
        self.query_is_new(name, args)
        if isinstance(seen, (dict, list)) and not _failed(seen):
            mark = self.call_mark(name, args)
            self.read_calls[mark], self.read_args[mark] = seen, (name, dict(args))

    def query_is_new(self, name: str, args: dict[str, Any]) -> bool:
        """Note this search by its non-numeric arguments; True the first time those were used with this tool."""
        mark = _dump(_k_without_numbers(args))
        asked = self.page_queries.setdefault(name, set())
        if mark in asked:
            return False
        asked.add(mark)
        return True

    @_soft('')
    def last_page_first(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A fresh search that started past its first page shows a few rows and no next page, so it looks complete (race 15:
        models asked the browse tool for its top page in 1716 of 2845 calls and saw 7 of 342 rows with "next: none"). When
        such a reply counts more rows than it shows, repeat it with its single raised numeric argument at its lowest value:
        at least as many rows, all different listings, show that argument is the page number."""
        if name in self.page_args or not isinstance(observation, dict):
            return ''
        props = self._k_props(name)
        raised = [key for key, value in args.items() if _k_whole(value) and key in props and value > _k_floor(props[key])]
        if len(raised) != 1 or self.left() <= self.tail() + 4:
            return ''
        short = self._k_short_last_page(name, observation)
        if short is None:
            return ''
        rows_key, shown = short
        key = raised[0]
        low = _k_floor(props[key])
        call = self._k_shaped(name, {**args, key: low})
        if call is None:
            return ''
        reply = self._k_step(name, call)
        if reply is None:
            return ''
        seen = reply.get('observation')
        first_rows = seen.get(rows_key) if isinstance(seen, dict) else None
        if not isinstance(first_rows, list) or len(first_rows) < shown:
            return ''
        if {i for i, _ in self.records_in(seen)} & {i for i, _ in self.records_in(observation)}:
            return ''
        self.page_args[name] = key
        self.trace.append({'kind': 'page_arg', 'tool': name, 'arg': key})
        head = f'\n[K2] {key} is the page number of this search: the call above showed its last page.'
        return head + f' Page {low} of the same search:\n' + self.show(seen)

    @_soft(False)
    def names_code(self, args: dict[str, Any]) -> bool:
        """Whether the search text names a model or part code that some listing's text gave as a code."""
        named: set[str] = set()
        for value in args.values():
            if isinstance(value, str):
                named.update(_codes(value))
        if not named:
            return False
        listed = (mark.get(field) or [] for mark in self.fit_index()['marks'].values() for field in ('codes', 'works'))
        return not named.isdisjoint(set().union(*listed))

    def first_look(self, name: str, args: dict[str, Any], fresh: bool) -> dict[str, Any]:
        """Once the tool's page argument is known, a new search is sent for its first page."""
        key = self.page_args.get(name)
        if key and fresh and isinstance(args.get(key), int):
            return {**args, key: _k_floor(self._k_props(name).get(key, {}))}
        return args

    @_soft('')
    def shorter_search(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Repeat a words search of three or more words that found nothing with just its first two plain words, without a
        model turn. Race 15: a search needs every word, and 54% of six-word, 42% of five-word but only 10% of one-word
        queries came back empty."""
        plan = self.value_template()
        text = self._k_missed_words(plan, name, args, observation)
        if text is None:
            return ''
        short = ' '.join([word for word in text.split() if not _codes(word)][:2])
        if not short:
            return ''
        call = self._k_shaped(name, _k_without_numbers({**args, plan[2]: short}))
        if call is None:
            return ''
        reply = self._k_step(name, call)
        if reply is None:
            return ''
        seen = reply.get('observation')
        self.trace.append({'kind': 'shorter', 'from': text, 'to': short, 'rows': len(self.records_in(seen))})
        return '\n[K2] Nothing has all those words;' + f' the same search with {short!r} shows:\n' + self.show(seen)

    @_soft('')
    def first_moves(self) -> str:
        """Before the model's first turn, search every item once and read some of its listings, so the interview can
        begin right away.

        A race model turn takes 5-10 s (race 15: 207 s per task on average; two agents lost 16 and 30 of 180 episodes to
        the 300 s limit, all with 3+ items), and the first turns would only do this. One-item tasks have the time and keep
        the model's own first searches. An item is searched by its required values with its noun, then by its noun alone
        when that finds nothing. Returns what the store showed, or '' when the search tool needs the model's reading of
        its modes first (the model then starts as usual)."""
        plan = self.value_template()
        if not plan or self.max_steps < OPEN_MIN_STEPS or len(self.items) < OPEN_ITEMS:
            return ''
        self.env_gap = OPEN_GAP
        shown: list[str] = []
        held = self.tail() + self.msgs_free() + OPEN_LATER * len(self.items) + HUNT_KEEP // 2
        stop_at = self.step + self.left() - held
        for number, item in enumerate(self.items, 1):
            if not self._k_open_item(plan, number, item, stop_at, shown):
                return '\n'.join(shown)
        if self.rows and self.roles.get('ask') and self.msgs_free() > 0:
            self._k_open_more(stop_at)
        self.env_gap = ENV_GAP
        self.trace.append({'kind': 'first_moves', 'rows': len(self.rows), 'opened': len(self.full)})
        shown.extend('read in full: ' + self.page_text(ident, record) for ident, record in self.full.items())
        proof = self.evidence(list(self.full.values()))
        if proof:
            shown.append(proof)
        return '\n'.join(shown)

    def page_text(self, ident: str, record: dict[str, Any]) -> str:
        """A listing read in full, for the model: the attributes it judges first, then the other plain fields."""
        attrs = _dump(_clip(_fact_block(record) or {}, 120))[:PAGE_ATTR_CHARS]
        flat = {key: value for key, value in record.items() if not isinstance(value, (dict, list))}
        others = _dump(_clip(flat, 200))[:PAGE_REST_CHARS]
        return f'{ident}: attributes {attrs} | {others}'

    def more_rows_for_keys(self, tool: str, args: dict[str, Any], seen: Any, stop_at: int, number: int) -> list[str]:
        """Read the next page when the first shows too few of the item's listings to tell their attributes; what it showed."""
        call = self._k_page_after(tool, args, seen)
        if call is None or self.rows_of_item(number) >= OPEN_EACH or self.finished or self.step >= stop_at:
            return []
        reply = self._k_step(tool, call)
        return [] if reply is None else ['next page: ' + self.show(reply.get('observation'))]

    def observe(self, name: str, observation: Any) -> str:
        self.absorb(observation)
        lines = self.loose_lines(observation)
        meaning = self.take_note(lines) if lines else ''
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def widen(self) -> None:
        """Read listings in full with one compare call over several items' unread listings (fewest read first, taken in
        turns); one at a time when the compare tool cannot take a plain list or shows no attributes."""
        tool = self.roles.get('compare')
        schema = self.schemas.get(tool or '') or {}
        target = _k_first_of_kind(schema.get('properties') or {}, 'array')
        if not self._k_batch_compare(tool, schema, target):
            self.widen_singly()
            return
        if self.left() <= 12:
            return
        most = schema['properties'][target].get('maxItems')
        picks = self._k_round_robin(min(most if isinstance(most, int) else 4, self.compare_room()))
        if len(picks) < 2:
            self.fill_facts(picks, 12)
            return
        args = self._k_shaped(tool, {target: picks})
        if args is None:
            return
        before = len(self.full)
        reply = self.act(tool, args)
        if reply is None:
            return
        rows = self._k_take_compare(tool, reply.get('observation'), picks, before)
        if rows and self.facts_from.get(tool) is False:
            self.widen_singly()
        elif len(self.full) == before and len(picks) > 2 and self.left() > 12:
            self._k_compare_pair(tool, target, picks)

    def widen_singly(self) -> None:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        wanted: set[str] = set()
        for _, need in self.needs_now():
            wanted |= _c_alnum3(need.split(':', 1)[-1])
        each = WIDEN_EACH if len(self.items) <= 2 else 4
        cheapest = self.brief.get('pick') == 'cheapest'
        lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in self.item_groups():
            opened = sum(i in self.full for i in ids)
            lanes.append(self._c_lane(ids, wanted, cheapest)[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = _c_deal(lanes, [(0, head) for head in heads], 4)
        later = _c_deal(lanes, [(head, each) for head in heads], each)
        cap = max(5, 2 * len(lanes) + 2)
        read = self.fill_facts(first[:min(cap, 8)], 22)
        self.fill_facts((first[cap:] + later)[:max(0, 10 - len(read))], 30)

    def ready_to_ask(self) -> bool:
        """Whether the interview may start now: messages, steps and time allow it, every item has a read listing (or the
        model already had its turns), and enough is read or enough steps are gone."""
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0 or self.left() <= 8 or self.clock() > ASK_BY:
            return False
        if self.loop_calls < ASK_TURNS and not all(map(self.records_of_item, range(1, len(self.items) + 1))):
            return False
        return len(self.full) >= 3 * len(self.items) + 1 or self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        """Arguments of the ask tool that carry `text`, required fields filled; None when the schema leaves no way."""
        schema = self.schemas.get(self.roles.get('ask') or '') or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        args = self._k_message_slot(props, required, text)
        if not args:
            return None
        for key in required:
            if key in args:
                continue
            value = _filler(props.get(key) or {})
            if value is None:
                return None
            args[key] = value
        faults: list[str] = []
        shaped = _shape(args, schema, 'args', faults)
        return None if faults else shaped

    def interview(self) -> str:
        """Ask the shopper the planned questions while messages, steps and time last; the summary for the loop model."""
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.widen()
        count = self.msg_limit - self.msg_used
        self._k_fresh_lanes(max(count, self.list_cap()) if self.listing else count)
        self.follow_up = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        groups = self.groups_for(self.msgs_free()) if self.listing else []
        while self._k_may_ask_more():
            batch = self._k_next_batch(groups, told)
            if not batch:
                break
            sent.extend(batch)
        answered = [n for n, hits in self.lane_found.items() if hits > 0]
        chain = self.early_trail()
        searched = self.value_search(answered) if answered else ''
        if told:
            self.version += 1
        return self._k_interview_summary(sent, told, chain, searched)

    def late_question(self) -> str:
        """Just before checking out, ask the next planned question with a message the order does not need.

        The interview keeps back every message the request reserves for a process step, whether that step comes with the
        order or with an ending without a purchase; once K2 is about to buy, the ones kept for that ending are free.
        Race 13 allowed two messages: the base asked a single question in 28 of 180 episodes and left a hidden need
        unasked in 12 of them, and no episode with an unasked need passed. Returns the loop model's summary when the
        answer adds a requirement, else ''."""
        if not self._k_late_allowed():
            return ''
        self.late_asked = True
        if not self.lanes:
            self._k_fresh_lanes(self.list_cap() if self.listing else 1)
        told: list[str] = []
        budget_before = self.budget
        batch = self.ask_next(told, LATE_STEPS - 2)
        if not batch:
            self.trace.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
            return ''
        if self.budget != budget_before:
            told.append(f'budget now {self.budget}')
        number, question, answer = batch[0]
        self.trace.append({'kind': 'late_ask', 'q': question, 'a': answer[:200], 'told': told})
        if not told:
            return ''
        self.version += 1
        summary = ('K2 asked the shopper one more question before checking out (one step):'
                   + f'\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told)
                   + '. The cart was not checked out: check each line against them' + _K_SEARCH_AGAIN
                   + ', fix the cart, then check out.' + self._k_pointed(told, '.'))
        searched = self.value_search([number])
        return summary + ('\n' + searched if searched else '')

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        free, named = _d_text_fields(props)
        candidates = named or self._d_varied_fields(free)
        for args in reversed(self.searched):
            fields = [k for k in candidates if _d_has_letter(args.get(k))]
            if not fields or not tool:
                continue
            field = fields[0]
            shelves = self._d_shelf_names()
            kept = {k: v for k, v in args.items() if k != field and self._d_keeps(props, required, shelves, k, v)}
            self._d_common_modes(kept, props, field)
            return (tool, kept, field)
        return None

    def value_words(self, number: int) -> list[tuple[str, bool]]:
        """Phrases to search for an item's requirement values, each flagged when an answer (not the request) gave it.
        A lower limit with a unit from an answer becomes its amount; relational or pointer values are skipped."""
        told = self.items[number - 1].told
        pointers = [need for _, need in self.refs]
        mine = [need for n, need in self.needs_now() if n == number and need not in pointers and ':' in need]
        return [word for word in (self._d_value_word(need, told) for need in mine) if word]

    def value_template(self) -> tuple[str, dict[str, Any], str] | None:
        """The search a value search copies: search_template(); else the field and fixed arguments the brief names; else,
        when every search so far only listed a category, the tool's one query field with its one mode argument set to
        read it (the option naming words or a query, or the other option of a required two-way mode). None when the
        schema leaves a doubt."""
        plan = self.search_template()
        tool = self.roles.get('search')
        if plan or not tool:
            return plan
        given = self._k_told_template(tool)
        if given is not None:
            return given
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        named = [key for key, spec in props.items() if isinstance(spec, dict) and 'string' in _kinds(spec)
                 and 'enum' not in spec and re.search(QUERY_WORDS, _snake(key), re.I)]
        last = self.searched[-1] if self.searched else {}
        enums = {key: spec['enum'] for key, spec in props.items()
                 if isinstance(spec, dict) and isinstance(spec.get('enum'), list) and len(spec['enum']) >= 2}
        modes = [pair for key, options in enums.items() for pair in _k_mode_choice(key, options, last, required)]
        if len(named) != 1 or len(modes) > 1 or (enums and not modes):
            return None
        query = named[0]
        base = {key: last[key] for key in required if key in last and key != query}
        base.update(modes)
        return (tool, base, query)

    def value_search(self, numbers: list[int]) -> str:
        """For each item with a value the shopper's answers gave, search its required values together and read the
        first rows of the item's kind; what the searches showed, '' when none ran."""
        plan = self.value_template()
        if not plan or self.left() < self.tail() + VALUE_ROOM:
            return ''
        notes: list[str] = []
        room = min(8, self.left() - self.tail() - VALUE_KEEP)
        for number in numbers:
            pairs = self.value_words(number)
            item = self.items[number - 1]
            if not any(told for _, told in pairs) or room <= 2:
                continue
            room = self._k_search_values(plan, number, item, pairs, room, notes)
        if not notes:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [line[:200] for line in notes]})
        return _K_VALUE_HEAD + '\n'.join(notes)

    def home_label(self, number: int) -> str:
        """The item's own category label (in _words form): its shelf or name when some listing carries that label,
        else the label most of its opened listings carry (its search rows when none is opened); '' if unknown."""
        cat_key = self.category_key()
        if not cat_key:
            return ''
        item = self.items[number - 1]
        for pool in (self.full, self.rows):
            tally = self._d_tally(item, pool, cat_key)
            named = next((w for w in (_words(item.shelf), _words(item.name)) if w in tally), None)
            if named is not None:
                return named
            if tally:
                return max(tally, key=tally.get)
        return ''

    def groups_for(self, free: int) -> list[list[int]]:
        """Splits the items with planned labels over the free messages when there are more items than messages:
        consecutive, near-equal groups (the planner's order first), so every item is asked at least once. Race 15 had
        19 of 90 tasks with 4-5 items and 3 messages; a message naming two items still drew needs for either."""
        ordered = [n for n in self.plan_order if n in self.lanes]
        ordered.extend(n for n in self.lanes if n not in self.plan_order)
        askable = [n for n in ordered if self.lanes.get(n)]
        if free > 0 and len(askable) > free:
            return _m_even_shares(askable, free)
        return []

    def ask_group(self, group: list[int], told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Ask about several items in one message, each with its leading open labels. The reply speaks to the first
        tracked property named, so each item's strongest keys come first in its part, and a reply that discloses
        nothing settles only the first item's first label."""
        share = self._m_group_share(len(group))
        parts = []
        for number in group:
            left = self._m_open_labels(number)
            if left:
                parts.append((number, left[:share]))
        if len(parts) < 2:
            if not parts:
                return []
            number, labels = parts[0]
            fitted, text = self.fit_ask(number, labels, False)
            return self._m_send_one(number, fitted, False, text, floor, told) if fitted else []
        chosen = self._m_fit_mixed(parts, share)
        if chosen is None:
            return []
        parts, text = chosen
        first, first_label = parts[0][0], parts[0][1][0]
        batch = self.send([(first, first_label, text)], floor)
        if not batch:
            return []
        for number, _ in parts:
            _m_bump(self.lane_spent, number)
        self.list_asks += 1
        grown, heard = self._m_read_reply([(0, text, batch[0][2])], True, told)
        answered = False
        for number, labels in parts:
            extra = grown[number - 1]
            if not extra:
                continue
            answered = True
            _m_bump(self.lane_found, number)
            self.settle(number, labels, False, extra, False)
        if not answered:
            if _m_crowded(heard):
                self._m_lists_off(first)
                return batch
            self.lane_used.setdefault(first, set()).add(_words(first_label))
        self.trace.append({'kind': 'mixed_ask', 'parts': parts, 'answered': answered})
        return batch

    def finish_ask(self, number: int, labels: list[str], shown: bool, batch: list[tuple[int, str, str]], told: list[str]) -> None:
        """Take a sent question's answer into the items' needs, then settle the labels it named."""
        _m_bump(self.lane_spent, number)
        self.list_asks += 1
        grown, heard = self._m_read_reply(batch, len(labels) > 1, told)
        fresh = []
        for index, extra in enumerate(grown, 1):
            if extra:
                _m_bump(self.lane_found, index)
                fresh += extra
        self.settle(number, labels, shown, fresh, _m_crowded(heard))

    def list_cap(self) -> int:
        if len(self.items) != 1:
            return LIST_MIXED
        return LIST_SIZE

    def list_now(self) -> int:
        """How many labels the next list names: fewer in the first questions of an episode, before any reply has
        shown how this shopper deals with a list."""
        if self.list_asks == 0 and len(self.items) == 1 and self.msgs_free() <= TWO_LEFT:
            return min(TWO_LEFT_FIRST, self.list_cap())   # (I1) two messages: three properties at once
        ramp = LIST_STEPS_MIXED if len(self.items) != 1 else LIST_STEPS
        cap = self.list_cap()
        if self.list_asks >= len(ramp):
            return cap
        return min(ramp[self.list_asks], cap)

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        rank = {n: i for i, n in enumerate(self.plan_order)}
        scarce = self.msg_limit <= SCARCE_MESSAGES and not self.listing
        width = self.list_now() if self.listing else 1

        def priority(n: int) -> tuple[bool, int, int]:
            # items with no answer yet first, then the least asked (unless messages are scarce), then plan order
            return (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), rank.get(n, len(rank) + n))
        for number in sorted(lanes, key=priority):
            open_labels = [label for label in lanes[number] if _words(label) not in used[number] and not self.told_about(number, label)]
            if open_labels:
                return number, open_labels[:width]
        return 0, []

    def told_about(self, number: int, label: str) -> bool:
        """True when an answer already named this property of the item: the label and a told property name contain
        one another (a one-word told name only covers a one-word label), with plurals and colour/color folded."""
        asked = _spelled(label)
        props = (_spelled(t.partition(':')[0]) for t in self.items[number - 1].told)
        return any(_names(asked, prop) for prop in props)

    def look_alikes(self, number: int, label: str) -> list[str]:
        """At most two of the item's keys a reader could confuse with `label`: they share a word with it, or the
        listings show the same values under both. Words carried by many keys of the item, and one-character values,
        count for nothing."""
        target = _words(label)
        words = self._m_key_words(number)
        spread: dict[str, int] = {}
        for word in (w for name in words for w in set(name.split())):
            spread[word] = spread.get(word, 0) + 1
        rare = {w for w in target.split() if spread.get(w, 0) <= 2}
        target_values = words.get(target, set())
        ranked = []
        for position, (name, values) in enumerate(words.items()):
            if not name or name == target:
                continue
            shared = rare.intersection(name.split())
            common = target_values & values
            if shared or common:
                ranked.append((3 * len(shared) + min(len(common), 3), position, name))
        ranked.sort(key=lambda entry: (-entry[0], entry[1]))
        return [entry[2] for entry in ranked[:LOOK_ALIKES]]

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        lead = labels[0]
        if len(labels) > 1:
            return 'What ' + _m_either(labels) + self._m_tail(number)
        alike = self.apart.get(lead, {}).get(number)
        if not alike:
            return self.question(number, lead, self.samples(number, lead) if shown else None)
        return f"Apart from the {' and the '.join(alike)}, which {lead}" + self._m_tail(number)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """The longest prefix of `labels` whose question the message tool's schema carries uncut (the schema shaper
        trims text over a length cap, so labels drop from the end instead), together with that question; ([], '')
        when even a single label does not fit."""
        size = len(labels)
        while size > 0:
            head = labels[:size]
            text = self.ask_text(number, head, shown)
            if self._m_fits(text):
                return head, text
            size -= 1
        return [], ''

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Send the next question (pending follow-ups before planned labels), read the answer into the items' needs
        (each new one also appended to `told`) and settle the labels it named. Returns the batch sent, [] if none."""
        chosen = self._m_fitted_ask()
        if chosen is None:
            return []
        number, labels, shown, text = chosen
        return self._m_send_one(number, labels, shown, text, floor, told)

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        """Record what an answer showed about the labels its question named.

        A single label is handled by _m_settle_single. For a list, the shopper answers about the first property it
        tracks, and a list naming none of its properties gets the same reply as a property it does not mind. So a
        disclosed need settles its label and all labels before it, and the labels after it lead the item's next list.
        A reply disclosing nothing settles only the first label (a need there would have been told); the rest come
        next in reverse order, so that what the shopper does not mind stops standing in front of them, and after a
        second such reply in a row they are taken from the middle outwards with the unnamed labels leading, as the
        order would otherwise go back to the first list's. A reply asking for one question at a time ends lists for
        the episode, and the first label of the list is asked on its own."""
        used = self.lane_used.setdefault(number, set())
        if len(labels) == 1:
            self._m_settle_single(number, labels, shown, new, crowded, used)
            return
        if crowded and not new:
            self._m_lists_off(number)
            self.follow_up.insert(0, (number, labels[:1], False))
            return
        hit = self.answered_label(labels, new) if new else -1
        if hit < 0:
            ahead = labels[:0:-1]
            used.update(_words(label) for label in labels[:1])
            misses = self.lane_misses.get(number, 0) + 1
        else:
            ahead = labels[hit + 1:]
            used.update(_words(label) for label in labels[:hit + 1])
            misses = 0
        self.lane_misses[number] = misses
        others = [label for label in self.lanes.get(number) or [] if label not in labels]
        regroup = misses > 0 and misses % 2 == 0
        if regroup:
            ahead = _m_middle_out(ahead)
        self.lanes[number] = others + ahead if regroup else ahead + others
        self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        """Position of the earliest label the reply answered: each new need goes to the label sharing most words with
        its property, or failing that to the label whose opened values share most words with its value; -1 if none."""
        value_roots = self._d_value_roots()
        picks = []
        for need in new:
            prop, _, value = need.partition(':')
            pick = _d_first_best([len(_spelled(label) & _spelled(prop)) for label in labels])
            if pick is None:
                said = _roots(value)
                pick = _d_first_best([len(value_roots.get(_words(label), set()) & said) for label in labels])
            if pick is not None:
                picks.append(pick)
        return min(picks, default=-1)

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        settled = [_stem(must.split(':', 1)[0]) for must in self.items[number - 1].musts]
        rows = [row for row in table if self.owner_of(row[0]) == number]
        numeric = {row[1]: _m_mostly_measures(row[3]) for row in table}
        if not rows and len(self.items) == 1:
            rows = list(table)
        # most carried first; measured keys after the others at equal reach; then the keys with more values
        rows.sort(key=lambda row: (-row[2], numeric.get(row[1], False), -len(row[3])))
        open_keys = [row[1] for row in rows if not any(_names(prop, _stem(row[1])) for prop in settled)]
        properties = [key for key in open_keys if _m_is_property(key)]
        return _m_catch_all_last(properties) + [key for key in open_keys if key not in properties]

    def adopt_labels(self) -> None:
        cat_key = self.category_key()
        if not cat_key or len(self.items) < 2:
            return
        self.part_shelves()
        labels = {str(record.get(cat_key) or '') for record in self.full.values()} - {''}
        owner = {label: self.owner_of(label) for label in labels}
        strays = [label for label, n in owner.items() if not n]
        if not strays:
            return
        claimed = {n for label, n in owner.items() if label not in self.adopted}
        orphans = [n for n in range(1, len(self.items) + 1) if n not in claimed]
        if len(orphans) != 1:
            return
        self.adopted.update(dict.fromkeys(strays, orphans[0]))
        self.trace.append({'kind': 'adopted', 'item': orphans[0], 'labels': strays})

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        if label in self.adopted:
            return self.adopted[label]
        memo = (label, len(self.items))
        if self.owners_seen.get(memo) is None:
            words, roots = _words(label), _roots(label)
            scores = [_m_owner_score(words, roots, item) for item in self.items]
            top = max(scores, default=0.0)
            self.owners_seen[memo] = scores.index(top) + 1 if top > 0 else 0
        return self.owners_seen[memo]

    def owns(self, item: Item, label: str) -> bool:
        if not label or item not in self.items:
            return False
        if self.owner_of(label) == self.items.index(item) + 1:
            return True
        return _words(label) == _words(item.shelf or '')

    def records_of_item(self, number: int) -> int:
        item = self.items[number - 1]
        cat_key = self.category_key()
        if not cat_key:
            return 0
        return len([r for r in self.full.values() if self.owns(item, _d_label(r, cat_key)) and _fact_block(r)])

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        self.adopt_labels()
        table = self.key_table()
        plan = self.ask_json(self._d_plan_prompt(count, table), [HEAVY, LIGHT], 3500) or {}
        picks, hints = self._d_read_plan(plan)
        lanes = {number: self._d_lane_for(number, chosen, table, hints[number]) for number, chosen in picks.items()}
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def name_parts(self, number: int) -> list[str]:
        """The separate products an item name lists with "and" or commas, when every one is a plural noun ("door
        handles and locks"); [] for one product spelled with "and" ("salt and pepper grinder")."""
        name = self.items[number - 1].name
        if not re.search('\\band\\b', name, re.I):
            return []
        pieces = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I)]
        parts = [p for p in pieces if p]
        if len(parts) < 2 or not all(re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts):
            return []
        return parts

    def item_said(self, number: int) -> str:
        """How a question names the item. Plural products joined with "and" are asked with "or" (an "and" drew
        one-at-a-time replies in race 13). After such a reply, the single product whose head noun the item's own
        rows mention most outside their category label (the first on a tie)."""
        item = self.items[number - 1]
        parts = self.name_parts(number)
        if not parts:
            return item.name
        if number not in self.one_product:
            return re.sub('\\band\\b', 'or', item.name, flags=re.I)
        texts = self._d_row_words(item)
        mentions = [sum(1 for words in texts if _roots(p.split()[-1]) & words) for p in parts]
        return parts[mentions.index(max(mentions))]

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        ending = f', for example {samples[0]} or {samples[1]}?' if samples else '?'
        return f'What {label} would you like for the {self.item_said(number)}{ending}'

    def samples(self, number: int, label: str) -> list[str]:
        want = _words(label)
        values = next((vals for _, key, _, vals in self.key_table() if key == want), None)
        if values is None:
            return []
        settled = ' '.join(self.items[number - 1].musts).lower()
        return [v for v in values if v and len(v) <= 30 and v not in settled][:2]

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for number, _label, text in questions:
            args = self.message_args(text) if self._d_may_message(floor) else None
            if not args:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            self.msg_used += 1
            answer = reply.get('observation')
            words = self.spoken(answer)
            sent.append((number, text, words))
            self._d_log_exchange(text, words)
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        self._d_take_needs(_items(heard.get('needs')), told)
        self._d_take_drops(_items(heard.get('drop')), told)
        for entry in _items(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                self.items[self._d_slot(entry.get('item')) - 1].free.append(str(entry['property']))
        amount = heard.get('budget')
        if isinstance(amount, (int, float)) and not isinstance(amount, bool) and amount > 0:
            self.budget = amount

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        if not self.quiet_reply(sent):
            self.hear_reads += 1
            return self.ask_json(self._m_hear_prompt(sent), list(models), 2500) or {}
        self.trace.append({'kind': 'quiet_reply', 'a': sent[0][2][:80]})
        return {}

    @_soft(False)
    def quiet_reply(self, sent: list[tuple[int, str, str]]) -> bool:
        """True when an answer can go unread: it has no digit, names no word of the questions, items, listing keys
        or values, and says nothing about how to be asked. Race 15: 5112 of 5160 replies without a need were like
        this (race 14: 5050 of 5055), and with the keys counted few need replies slip through (race 15: 35 of 1355
        by question words alone). Each reading is a model call of some 5-10 s in a 300 s episode. The first answer of
        an episode is always read, since it shows whether the shopper takes lists."""
        if self.hear_reads < 1:
            return False
        known = self._m_known_words()
        for _, question, answer in sent:
            asked = known | _m_spelled_words(re.sub(_M_FILLER, ' ', question, flags=re.I))
            said = _m_spelled_words(answer)
            if re.search('\\d', answer) or said & asked or re.search(ASK_WORDS, answer, re.I):
                return False
        return True

    def gate(self, stage: str) -> tuple[bool, str]:
        due = [p for p in self.process if p['when'] == stage]
        if not due or self.finished or not self.gate_due(stage):
            return True, ''
        ordering = stage == 'order'
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = self._m_gate_tools(ending)
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            got = self.ask_json(self._m_gate_prompt(ordering, due, tools), [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told, hold = self._m_gate_heard(got, said)
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.version += 1
                changes = told + [hold] if hold else told
                done = _M_DONE_HEAD + '\n'.join(said) + '\n' if said else ''
                return False, done + _M_CHANGED + '; '.join(changes) + _M_FIX_CART
            name = str((call or {}).get('tool') or '')
            unproven = self._m_gate_unproven(got, due, ordering)
            if not call:
                if unproven and self._m_may_nudge(said, nudged):
                    nudged = True
                    self._m_gate_nudge(unproven)
                    continue
                self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
                break
            if name not in self.schemas or name in ending:
                break
            asking = name == self.roles.get('ask')
            if self.left() < 2 or (asking and self.msg_used >= self.msg_limit):
                break
            faults: list[str] = []
            given = call.get('args')
            args = _shape(given if isinstance(given, dict) else {}, self.schemas[name], 'args', faults)
            if faults:
                break
            line = self._m_gate_message(args) if asking else self._m_gate_call(name, args)
            if line is None:
                break
            said.append(line)
        if not said:
            return True, ''
        return True, _M_DONE_HEAD + '\n'.join(said)

    def spoken(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        phrases = list(_e_phrases(observation, 0))
        if not phrases:
            return _dump(observation)[:400]
        return ' '.join(phrases)[:800]

    def loose_lines(self, observation: Any) -> list[str]:
        if not isinstance(observation, dict) or self.ident(observation):
            return []
        found: list[str] = []
        for value in observation.values():
            found += self._e_loose_from(value)
        return [line for line in found if line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = _e_sections(NOTE, [('ITEMS AND REQUIREMENTS NOW:\n', self.ledger()), ('THE NOTE:\n', '\n'.join(lines))])
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        gone = self._e_note_stock(got, lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed: list[str] = []
        budget = got.get('budget')
        moved = _e_positive(budget) and budget != self.budget
        if moved:
            self.budget = budget
            changed.append(f'budget now {budget}')
        drops = _e_need_entries(got.get('drop'))
        adds = _e_need_entries(got.get('add'))
        changed += [self._e_drop_need(entry) for entry in drops]
        changed += [self._e_add_need(entry) for entry in adds]
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price changed')
        summary = str(got.get('summary') or '').strip()
        if self.brief.get('pick') == 'cheapest' and (drops or self.note_sold or moved):
            self.wider_held = False
            self.cheaper_marks = set()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or '; '.join(changed))
        if changed:
            summary += ' (' + '; '.join(changed) + ')'
        return summary.strip()

    def needs_now(self) -> list[tuple[int, str]]:
        return [(number, need) for number, item in enumerate(self.items, 1) for need in item.musts + item.told if need not in item.dropped]

    def evidence(self, observation: Any) -> str:
        ids = self._m_record_ids(observation)
        needs = self.needs_now()
        if not (ids and needs):
            return ''
        many = len(self.items) > 1
        lines = [line for line in (self._m_evidence_line(ident, needs, many) for ident in ids[:12]) if line]
        return _M_EVIDENCE_HEAD + ' | '.join(lines) if lines else ''

    def cart_text(self) -> str:
        if not self._e_view_fresh():
            return "lines by K2's record of the cart calls (listing: units): " + _dump(self.cart_lines)
        return _e_dumped(self.cart_view, 300, 3000)

    def held_units(self, ident: str) -> int:
        if self._e_view_fresh():
            for node in _e_popped_dicts(self.cart_view):
                units = self._e_units_on(node, ident)
                if units is not None:
                    return units
        return self.cart_lines.get(ident, 1)

    def cart_ids(self) -> list[str]:
        if not self._e_view_fresh():
            return list(self.cart_lines)
        view = _dump(self.cart_view)
        return [i for i in self.rows if i in view]

    def record_of(self, ident: str) -> str:
        record = self._e_record(ident)
        if not record:
            return '(no record seen)'
        return _e_dumped(record, 400, 3000)

    def others(self, skip: list[str], limit: int=30) -> str:
        order = list(reversed(self.seen_order))
        if self.brief.get('pick') == 'cheapest':
            order.sort(key=self._e_cheap_first)
        kept = [i for i in order if i not in skip and not self.sold_out(i)]
        return '\n'.join(_e_dumped(self._e_record(i), 160, 900) for i in kept[:max(1, limit)])

    def fit_index(self) -> dict[str, Any]:
        state = (len(self.seen_order), len(self.full), sum(map(len, self.views.values())))
        cached = self.fit_cached
        if cached and cached[0] == state:
            return cached[1]
        marks: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            marks[ident] = _fit_marks(_m_record_text(self.full.get(ident) or self.rows.get(ident) or {}))
        notes: dict[str, set[str]] = {}
        for mark in marks.values():
            for part, codes in mark['also']:
                notes.setdefault(part, set()).update(codes)
        self.fit_cached = (state, {'marks': marks, 'notes': notes})
        return self.fit_cached[1]

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or [])
        own = mark.get('own')
        if own:
            found.update(index['notes'].get(own, set()))
        return found

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        if not cat_key:
            return []
        return [i for i in self.seen_order if self.owns(item, self._e_label(i, cat_key))]

    def link_system(self, part: int, main: int, index: dict[str, Any]) -> tuple[int, int, set[str]] | None:
        """Look for a series system tying two items together. Either item may be the carrier: most of its rows show a code
        from a small set of series, and at least one row of the other item names one of those series. Returns
        (carrier, namer, series) or None."""
        for carrier, namer in ((part, main), (main, part)):
            families = self._p_series_of(carrier, index)
            if families is None:
                continue
            if any(families & _p_tags(index, ident, 'named') for ident in self.item_ids(namer)):
                return (carrier, namer, families)
        return None

    def link_holds(self, carried: str, namer: str, families: set[str], index: dict[str, Any]) -> bool | None:
        """True or False by whether the carrying row's series is among those the naming row names; None when one of
        the two shows no series of the system."""
        own = _p_tags(index, carried, 'fam') & families
        named = _p_tags(index, namer, 'named') & families
        if not (own and named):
            return None
        return not own.isdisjoint(named)

    def link_faults(self, part: int, main: int, device: str, lines: list[str], index: dict[str, Any]) -> list[str]:
        system = self.link_system(part, main, index)
        if not system:
            return []
        carrier, families = system[0], system[2]
        found = []
        for line in lines:
            # the cart line and the device swap roles depending on which item carries the series codes
            carried, namer = (line, device) if carrier == part else (device, line)
            if self.link_holds(carried, namer, families, index) is False:
                found.append(self._p_series_fault(carrier, carried, namer, families, main, index))
        return found

    def series_for(self, number: int) -> list[str]:
        """Series worth searching for an item tied to another by a series system: first the ones the cart's lines
        already hold, then the rest by how many rows carry or name them. Empty when no system ties the item."""
        index = self.fit_index()
        cart = self.cart_ids()
        counts: dict[str, int] = {}
        held: list[str] = []
        for ident, family in self._p_linked_hits(number, index):
            counts[family] = counts.get(family, 0) + 1
            if ident in cart and family not in held:
                held.append(family)
        rest = sorted((family for family in counts if family not in held), key=lambda f: (-counts[f], f))
        return (held + rest)[:LINK_SEARCHES]

    def fit_faults(self) -> list[str]:
        if not self.bound:
            return []
        index = self.fit_index()
        cart = self.cart_ids()
        found: list[str] = []
        for part, main in self.bound:
            found += self._p_pair_faults(part, main, cart, index)
        return found

    def fit_plan(self, short: bool=False) -> str:
        if not self.bound:
            return ''
        index = self.fit_index()
        lines: list[str] = []
        for main in sorted({m for _, m in self.bound}):
            lines.extend(self._p_plan_lines(main, index))
        if not lines:
            return ''
        joined = ' | '.join(lines)
        return joined if short else _P_PLAN_HEAD + joined

    @_soft('')
    def cheapest_sets(self, main: int, parts: list[int], index: dict[str, Any]) -> str:
        """For a cheapest pick: the smallest order totals made of one device of the item plus one fitting part of each
        bound item, from the buyable rows seen so far. Requirements are not checked here; the point is that the whole
        order is compared, not each line alone."""
        systems = {part: self.link_system(part, main, index) for part in parts}
        shelves = {part: self._p_priced_shelf(part, index) for part in parts}
        options = []
        for device in self.item_ids(main):
            whole = self._p_whole_set(device, main, parts, systems, shelves, index)
            if whole:
                options.append(whole)
        if not options:
            return ''
        options.sort(key=lambda o: o[0])
        listed = '; '.join(f"{' + '.join(picks)} = {total:g}" for total, picks in options[:SET_SHOWN])
        return 'lowest whole sets seen, requirements not yet checked: ' + listed

    def missing_items(self) -> list[int]:
        """Items with no line in a cart holding fewer lines than items; items no seen listing is labelled for are not counted."""
        cat_key = self.category_key()
        ids = self._e_held_ids()
        if not cat_key or len(ids) >= len(self.items):
            return []
        shelf = [str(r.get(cat_key) or '') for r in self.rows.values()]
        carried = [self._e_label(i, cat_key) for i in ids]
        return [n for n, item in enumerate(self.items, 1) if self._e_owns_any(item, shelf) and not self._e_owns_any(item, carried)]

    def states_needs(self, ident: str, number: int) -> bool:
        """Whether a listing's attributes state each of the item's requirements word for word; a limit or an exclusion is
        not read here and counts as not stated."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        return all(_e_stated(need, block) for owner, need in self.needs_now() if owner == number)

    def fill_short_cart(self) -> bool:
        """Before a forced order: add, for each item with no line, the cheapest opened listing that states its requirements.
        False when an item stays uncovered, so the episode ends by refusing rather than with a short order."""
        cart = self.roles.get('cart') or ''
        paired = {n for pair in self.bound for n in pair}
        for number in self.missing_items():
            item = self.items[number - 1]
            pick = None if number in paired else self._e_cheapest_fit(number)
            if not pick or not self._e_affordable(pick, item.qty):
                return False
            calls = self.cart_calls(self._e_lines_with(pick, item.qty), [])
            if not calls or len(calls) + 1 > self.left():
                return False
            if not self._e_send_cart(cart, calls):
                return False
            self.trace.append({'kind': 'filled_line', 'item': number, 'line': pick})
        return True

    def part_shelves(self) -> None:
        """When two requested items share a store label, the one whose own name is another seen label (that no item
        holds yet) moves to that label."""
        labels = self._p_store_labels(self.category_key())
        moved = False
        for item in self.items:
            target = self._p_own_shelf(item, labels)
            if target is None:
                continue
            self.trace.append({'kind': 'shelf_parted', 'item': item.name, 'from': item.shelf, 'to': target})
            item.shelf = target
            moved = True
        if moved:
            self.owners_seen = {}
            self.fit_cached = None

    def fuller_label(self, number: int, label: str) -> str:
        cat_key = self.category_key()
        item = self.items[number - 1]
        mine = _roots(item.name)
        theirs = _roots(label)
        if len(mine) < 2 or mine <= theirs or not cat_key:
            return ''
        if item.shelf and theirs == _roots(item.shelf):
            return ''
        shelf = sorted({str(r.get(cat_key) or '') for r in self.rows.values()})
        return next((seen for seen in shelf if seen and mine <= _roots(seen)), '')

    def named_label(self, number: int, label: str) -> str:
        """When the request gave this item a store label that some seen row carries, and `label` is a different one: that
        label as the row shows it. '' otherwise."""
        wanted = _words(self.items[number - 1].shelf or '')
        cat_key = self.category_key()
        if not (wanted and cat_key) or _words(label) == wanted:
            return ''
        for row in self.rows.values():
            shown = str(row.get(cat_key) or '')
            if _words(shown) == wanted:
                return shown
        return ''

    def line_faults(self) -> list[str]:
        self.adopt_labels()
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not ids or not cat_key:
            return []
        label = {i: str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids}
        known = self._p_listed_items(cat_key)
        owner = {i: self._p_owners(label[i]) for i in ids}
        found: list[str] = []
        for number, item in enumerate(self.items, 1):
            if number in known:
                found += self._p_item_faults(number, item, ids, owner, label)
        if len(known) == len(self.items):
            found += [f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out" for i in ids if not owner[i]]
        found += [f'{i} is sold out' for i in ids if self.sold_out(i)]
        return found + self._p_budget_faults(ids)

    def count_fixes(self) -> list[tuple[str, int, int]]:
        cat_key = self.category_key()
        if not cat_key:
            return []
        cart = self.cart_ids()
        fixes: list[tuple[str, int, int]] = []
        for ident in cart:
            have = self.held_units(ident) or 0
            tag = self._e_label(ident, cat_key)
            owners = [n for n, item in enumerate(self.items, 1) if self.owns(item, tag)]
            if len(owners) != 1 or have <= 0:
                continue
            item = self.items[owners[0] - 1]
            peers = [i for i in cart if (self.cart_lines.get(i, 1) or 0) > 0 and self.owns(item, self._e_label(i, cat_key))]
            if len(peers) == 1 and have != item.qty and not self.sold_out(ident):
                fixes.append((ident, have, item.qty))
        return fixes

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        target = self._e_id_field(name, schema, props)
        count = _e_first_key(props, lambda spec: 'integer' in _kinds(spec))
        mode = _e_first_key(props, lambda spec: isinstance(spec.get('enum'), list) and len(spec['enum']) >= 2)
        if not (name and target and count):
            return []
        calls = self._e_count_calls(wrong, name, props, target, count, mode)
        if calls is None or len(calls) + 3 > self.left():
            return []
        done = self._e_run_counts(name, schema, calls)
        self.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    def check_key(self, ids: list[str]) -> str:
        return self._e_versioned('|'.join(sorted(ids)))

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        planned = ids is not None
        listing = ids if planned else self.cart_ids()
        key = self.check_key(listing)
        if key in self.checked or not self.judges(limit) or not listing:
            return ''
        self.checked.add(key)
        ranked = self.brief.get('pick') in ('cheapest', 'most', 'least')
        prompt = self._e_check_prompt(listing, planned, ranked)
        got = self._e_check_verdict(prompt, planned, ranked, limit)
        self.trace.append({'kind': 'check', 'cart': listing, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in listing]
        return self._e_review_text(got)

    def refuse_trail(self) -> str:
        """A one-time chain follow before declining while steps remain; returns what it found ('' for nothing)."""
        if self.refuse_trailed or not self._p_chains_waiting():
            return ''
        self.refuse_trailed = True
        found, _ = self.trail(HUNT_BY)
        if not found:
            return ''
        return found + _P_REFUSE_PUSH

    def early_trail(self) -> str:
        """Follow the pointing requirements early on, with a small round allowance of its own, so the checkout's
        allowance stays untouched."""
        if not self._p_chains_waiting():
            return ''
        saved_trails, saved_blocks = self.trails, self.trail_blocks
        self.trails, self.trail_room = 0, EARLY_TRAILS
        try:
            found, _ = self.trail(HUNT_BY)
        finally:
            self.trails = saved_trails
            self.trail_blocks = saved_blocks
            self.trail_room = None
        self.trace.append({'kind': 'early_trail', 'left': self.left(), 'values': list(self.ref_values.values())[:3]})
        return found

    def page_read(self, ident: str) -> bool:
        """A listing counts as read once it was opened, and, if the store has a page view separate from the attribute
        view, once that page view showed it."""
        if ident not in self.opened:
            return False
        page = self.roles.get('offer')
        if not page or not self.single_target(page):
            return True
        return page in self.views.get(ident, ())

    def trail(self, stop: float | None=None) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.offer_tool()
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self._p_may_trail(stop):
            self.trails += 1
            prompt = self._p_trail_prompt(pending)
            got = self.ask_json(prompt, self._p_trail_models(), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach, lost, gaps = self._p_take_chains(got, pending)
            pending = [need for need in pending if need not in self.ref_values]
            can_open = bool(tool and target)
            if not reach and gaps and can_open and self.left() > 12 and self.link_searches < 3:
                reach = self.find_link(*gaps[0])
                if self.trace and self.trace[-1].get('kind') == 'link_known':
                    seen.append(str(self.trace[-1]['id']) + _P_SHOP_PAGE_NOTE)
                    lost = lost[1:]
            if not reach or not can_open or self.left() < len(reach) + 4 or self.finished:
                break
            self._p_open_pages(tool, target, reach[:max(0, min(3, 6 - len(seen)))], seen)
        return self._p_trail_report(seen, lost)

    def not_maker_keys(self) -> tuple[Any, ...]:
        """Keys that never hold a seller's or maker's name (the id, the category label, the title). Every other short text
        field may, and a match still needs the name word for word in the description."""
        label_key = self.category_key()
        title = self.title_key()
        return (self.id_key, label_key, title)

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        return self._p_shop_named(about, ids, True)

    def opened_shop(self, about: str) -> list[str]:
        return self._p_shop_named(about, self.opened, False)

    def find_link(self, about: str, words: str) -> list[str]:
        skip = self.not_maker_keys()
        shop_words = self._p_shop_words(about, skip)
        asked = (words or about).lower()
        values = {w for w in re.findall('[a-z0-9]+', asked) if len(w) >= 3 or w.isdigit()} - shop_words - _P_FILLER
        if self._p_shop_page_shows(about, values):
            return []
        plan = self.value_template()
        memo = ' '.join(words.lower().split())
        known: list[str] = []
        if memo and memo in self.link_seen:
            known = self.shop_rows(about, self.link_seen[memo])
        elif plan and words:
            known = self._p_link_search(plan, words, about, memo)
        kind = _p_tokens3(asked) - shop_words - _P_FILLER
        newest = self._p_of_kind(known, kind) or self._p_of_kind(self.shop_rows(about, list(reversed(self.seen_order))), kind)
        return newest[:2]

    def offer_tool(self) -> str | None:
        page = self.roles.get('offer')
        return page if page else self.roles.get('open')

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not (tool and target):
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if not stale or self.left() < len(stale) + 5:
            return ''
        changes = []
        for ident in stale:
            before = dict(self.rows.get(ident) or {})
            reply = self._p_open_one(tool, target, ident)
            if reply is None:
                break
            observation = reply.get('observation')
            self.offer_seen[ident] = self.step
            self.observe(tool, observation)
            moved = _p_changed_keys(before, self.rows.get(ident) or {})
            if moved:
                changes.append(f"{ident} changed ({', '.join(moved[:4])}): " + _p_snippet(observation, 300, 900))
        self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(changes)})
        if changes:
            self.version += 1
        return '\n'.join(changes)

    def sold_out(self, ident: str) -> bool:
        return any((ident in pool for pool in (self.unavailable, self.gone)))

    def swap_hint(self) -> str:
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ''
        self._f_retire(lost)
        picks = self.replacements(lost)
        head = '[K2] Sold out now: ' + ', '.join(lost) + ' (the store shows it cannot be bought). It will not come back, so do not open it again.'
        if not picks:
            return head + ' Remove that line from the cart and add the best other listing that fits;' + ' search for one if none you have read fits.'
        if self.left() <= 6:
            return head + self._f_rushed_swap(picks)
        if self.brief.get('pick') == 'cheapest':
            return head + self._f_cheapest_swap(picks)
        return ''.join([head, ' Listings already read that appear to fit that item instead: ', ', '.join(picks),
                        '. In this turn take the sold-out line out of the cart and add ', picks[0],
                        ' at the requested quantity, then check out.'])

    def find_sold(self) -> str:
        held = [i for i in self.cart_ids() if i not in self.gone]
        if not held or any((self.sold_out(i) for i in held)):
            return ''
        if len(held) > 1:
            return self.reread_offers()
        self.unavailable.add(held[0])
        return ''

    def replacements(self, lost: list[str]) -> list[str]:
        if not self.judges():
            return []
        others = self._f_newest(25, lambda i: i in lost or i in self.gone or self.sold_out(i))
        records = [_dump(_clip(self.full.get(i) or self.rows.get(i), 200))[:1200] for i in others]
        if not records:
            return []
        prompt = ''.join([self._f_prompt_head(SWAP),
                          _f_section('SOLD-OUT CART LINE', '\n'.join((f'{i}: {self.record_of(i)}' for i in lost))),
                          _f_section('CART NOW', self.cart_text()),
                          _f_section('OTHER LISTING RECORDS SEEN (newest first)', '\n'.join(records))])
        got = self.ask_json(prompt, self.judges()[::-1], 3500)
        fits = self._f_swap_fits(got, lost)
        order = sorted(fits, key=lambda pair: pair[0]) if self.brief.get('pick') == 'cheapest' else fits
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return [ident for _, ident in order][:4]

    def audit(self) -> str:
        if self.audited or not self.judges():
            return ''
        self.audited = True
        got, named = self._f_audit_pass()
        if not named:
            unread = [i for i in self._f_live_ids((got or {}).get('open')) if i not in self.opened]
            if unread and self.open_rows(unread):
                got, named = self._f_audit_pass()
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return self._f_audit_note(named, got.get('reason'))

    def audit_call(self) -> dict[str, Any] | None:
        opened_first = sorted(reversed(self.seen_order), key=lambda i: i not in self.full)
        chosen = [ident for ident in opened_first if not self.sold_out(ident)][:30]
        if not chosen:
            return None
        records = [self._p_audit_record(ident) for ident in chosen]
        prompt = ''.join([AUDIT, '\n\nREQUEST:\n', self.request, '\n\nREQUIREMENTS IN FORCE:\n', self.ledger(),
                          '\n\nLISTING RECORDS SEEN (newest first):\n', '\n'.join(records)])
        judges = self.judges()
        got = self.ask_json(prompt, [LIGHT] if judges else [], 4000)
        if self._p_wants_heavy(judges, got):
            got = self.ask_json(prompt, [HEAVY], 4000) or got
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        per_item: dict[str, list[str]] = {}
        for item, raw in fits.items():
            per_item[str(item)] = self._f_live_ids(raw)[:4]
        return {k: ids for k, ids in per_item.items() if ids}

    def open_rows(self, rows: list[str]) -> bool:
        plan = self._p_open_plan(rows)
        if plan is None:
            return False
        cart_steps = len(self.cart_calls({f'#{n}': 1 for n in range(len(self.items))}, []) or self.items)
        if self.left() < len(plan) + max(5, self.tail() + cart_steps):
            return False
        before = len(self.full)
        for name, raw in plan:
            faults: list[str] = []
            args = _shape(raw, self.schemas.get(name), 'args', faults)
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
        nested = _f_nested_lines(props)
        if nested is not None:
            return _f_bulk_call(nested[0], nested[1], lines)
        texts = [k for k, s in props.items() if isinstance(s, dict) and _f_free_text(s)]
        target = self._f_cart_target(name, schema, texts)
        count = _f_first_field(props, _f_whole_number)
        mode = _f_first_field(props, _f_has_choice)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            call = self._f_drop_call(name, props, target, count, mode, ident)
            if call is None:
                return None
            calls.append(call)
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if have and (not self.cart_sets(name)):
                return None
            call = {target: ident}
            if mode:
                call[mode] = props[mode]['enum'][0]
            if count:
                call[count] = want
            calls.append(call)
        return calls

    def complete_cart(self) -> str:
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        blocked = self.completed or not self.confirmed or self.sold_out(self.confirmed)
        if blocked or not (cart and buy):
            return ''
        self.completed = True
        cat_key = self.category_key()
        plan = self._f_lead_plan(cat_key)
        if self.confirmed not in plan:
            return ''
        split = self._f_cart_split(cat_key, plan)
        if split is None:
            return ''
        lines, drop = split
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + 1 > self.left():
            return ''
        if not self._f_cart_verified(lines, len(calls) + 3):
            return ''
        done = self._f_set_cart(cart, calls)
        if done is None:
            return ''
        args, fits = self._f_shaped(buy, dict(self.brief.get('buy_args') or {}))
        if not fits:
            return ''
        replies = self._f_checkout(buy, args, lines, drop)
        if not replies:
            return ''
        return 'Not run (K2 acted instead): the independent check confirmed ' + self.confirmed + ', so K2 set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        return self._f_role_call('buy', 'buy_args', name, args)

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        return self._f_role_call('refuse', 'refuse_args', name, args)

    def offered(self) -> list[dict[str, Any]]:
        ask = self.roles.get('ask')
        tools = [t for t in self.tools if t['function']['name'] != ask or not self._q_interviewer_speaks()]
        window = self._q_end_window()
        if window is None:
            return tools
        # near the end only the finishing tools (and what fills the cart) stay on offer, if the store has any of them
        return [t for t in tools if t['function']['name'] in window] or tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        out = dict(args)
        for key, spec in (schema.get('properties') or {}).items():
            if not (isinstance(spec, dict) and _f_free_text(spec)):
                continue
            value = out.get(key)
            if value is None and key not in (schema.get('required') or []):
                continue
            text = str(value or '')
            if not text or self._f_should_decline(text):
                out[key] = _f_decline_text[:spec.get('maxLength', 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        key = self._f_id_field(name)
        if key is None:
            return (args, '')
        out = dict(args)
        if key not in args:
            return (out, '')
        raw = args[key]
        many = isinstance(raw, list)
        fixed = [self._f_fix_id(v) for v in raw] if many else [self._f_fix_id(raw)]
        problem = next((p for _, p in fixed if p), '')
        if problem:
            return (args, problem)
        out[key] = [v for v, _ in fixed] if many else fixed[0][0]
        return (out, '')

    def cart_sets(self, name: str) -> bool:
        specs = [s for s in (self.schemas.get(name, {}).get('properties') or {}).values() if isinstance(s, dict)]
        if any((_f_has_choice(s) for s in specs)):
            return False
        return any(('integer' in _kinds(s) and s.get('minimum') == 0 for s in specs))

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Give a count-less removal of a held line the full number of units held (else the store drops one)."""
        spec = self.schemas.get(name, {}).get('properties') or {}
        count = _f_first_field(spec, _f_whole_number)
        removal = any((_f_marks_removal(spec.get(k), v) for k, v in args.items()))
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.cart_lines), None)
        if not (count and removal and ident) or count in args:
            return args
        held = self.cart_lines[ident]
        return {**args, count: min(held, spec[count].get('maximum', held))}

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get('properties') or {}
        removal = any([_f_marks_removal(spec.get(key), value) for key, value in args.items()])
        loose = next((_count(v) for v in args.values() if _count(v) is not None and (not isinstance(v, str))), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            found, replaces = self._f_lines_from(spec.get(key), value, loose)
            lines += found
            whole = whole or replaces
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_sets(name)
        for ident, count in lines:
            have = _f_units_after(self.cart_lines.get(ident, 0), count, removal, sets, spec)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def show(self, observation: Any) -> str:
        width = SHOW_CHARS if self.records_in(observation) else TEXT_CHARS
        text = _dump(_clip(observation, width))
        if len(text) <= OBS_LIMIT:
            return text
        return text[:OBS_LIMIT] + '...(cut)'

    def compare_target(self) -> str | None:
        schema = self._g_compare_schema()
        listed = None
        for key, spec in (schema.get('properties') or {}).items():
            if isinstance(spec, dict) and 'array' in _kinds(spec):
                listed = key
                break
        # the list field must be the only thing the tool insists on
        if listed and all(need == listed for need in schema.get('required') or []):
            return listed
        return None

    def compare_room(self) -> int:
        fields = self._g_compare_schema().get('properties') or {}
        limit = (fields.get(self.compare_target() or '') or {}).get('maxItems')
        if isinstance(limit, int):
            return limit
        if not self.compare_bad:
            return max(4, self.compare_ok)
        # a failed batch size is known: stay below it, but never under two
        return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener = self.roles.get('open')
        joiner = self.roles.get('compare')
        field = self.single_target(opener)
        many = self.compare_target()
        if not (opener and joiner and field and many) or opener == joiner or self.compare_bare():
            return calls
        picks = []
        for position, call in enumerate(calls):
            wanted = self._g_open_pick(call, opener, field)
            if wanted is not None:
                picks.append((position, wanted))
        picks = picks[:self.compare_room()]
        if len(picks) < 2:
            return calls
        batch = dict(picks)
        lead = picks[0][0]
        idents = list(batch.values())
        merged = []
        for position, call in enumerate(calls):
            if position == lead:
                call = {'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': _dump({many: idents})}}
            elif position in batch:
                call = {'id': call['id'], 'joined': True}
            merged.append(call)
        self.trace.append({'kind': 'joined', 'ids': idents})
        return merged

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Answer each call of one model turn, in order: the store's result with K2's notes, or why it was held back."""
        batch = self.join_opens(calls)
        return [self._q_settle(call, len(batch)) for call in batch]

    def can_refuse(self) -> bool:
        refuse = self.roles.get('refuse')
        return bool(refuse) and self._g_decline_args(refuse)[1]

    def finish(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ''
        buy = self.roles.get('buy')
        refuse = self.roles.get('refuse')
        fresh = self._g_cart_fresh()
        holding = bool(self.cart_ids() if fresh else self.cart_lines)
        whole = self._g_settle_cart() if holding and len(self.items) >= 2 else True
        if self._g_over():
            return ''
        if whole and buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            outcome = self._g_forced_buy(buy, again)
            if outcome is not None:
                return outcome
        return self._g_forced_refuse(refuse) if refuse else ''

    def trim(self) -> None:
        if sum(len(_dump(m.get('content'))) for m in self.chat) <= HISTORY_LIMIT:
            return
        results = [m for m in self.chat if m.get('role') == 'tool']
        for message in results[:max(1, len(results) // 2)]:
            self._g_shorten(message)

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self._q_take_opening()
        moved = '' if self.finished else self.first_moves()
        self.env_gap = ENV_GAP
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': self._q_intro(opening, moved)}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            self._q_close_out()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5),
                           'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        tries = CRASH_TURNS
        while tries > 0:
            tries -= 1
            try:
                self.turns(loop_cap, spend_cap)
            except Exception as exc:
                self.trace.append(_q_crash_note(exc))
                self.answer_open_calls()
            else:
                return

    def answer_open_calls(self) -> None:
        """Reply to every call of the last model turn that has no reply yet, so the model can go on from there."""
        answered = set()
        for message in reversed(self.chat):
            if message.get('tool_calls'):
                break
            answered.add(message.get('tool_call_id'))
        else:
            return
        for call in message['tool_calls']:
            if call['id'] not in answered:
                self._q_tool_reply(call['id'], _Q_LOST_CALL)
        self._q_say(self.status())

    def turns(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while self._q_may_go_on(loop_cap, spend_cap):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self._q_say(summary + '\n' + self.status())
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                if self._q_after_miss():
                    break
                continue
            calls = [self._q_clean_call(n, c) for n, c in enumerate(message.get('tool_calls') or []) if isinstance(c, dict)]
            content = _q_plain_text(message.get('content'))
            if calls:
                idle = 0
                self._q_take_turn(content, calls)
                continue
            idle += 1
            self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
            self._q_say('Call one of the tools to continue.\n' + self.status())
            if idle >= 3:
                break

    def _b_stage_steps(self, stage: str) -> list[dict[str, Any]]:
        return [entry for entry in self.process if entry['when'] == stage]

    def _b_may_call(self) -> bool:
        """Whether one more model request fits the call cap and the time left (a late skip is traced)."""
        if self.calls >= ALL_CALLS:
            return False
        if self.model_room() >= CALL_FLOOR:
            return True
        self.trace.append({'kind': 'late_call_skipped', 'at': round(self.clock(), 1)})
        return False

    def _b_with_retries(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        attempt = 0
        while True:
            message, status = self.call_model(model, messages, max_tokens, tools)
            if message is not None:
                return message, status
            wait = self._b_retry_wait(status, attempt)
            if wait is None:
                return None, status
            pause = wait * (0.8 + 0.4 * random.random())
            self.waited += wait
            attempt += 1
            sleep(pause)

    def _b_retry_wait(self, status: Any, attempt: int) -> int | None:
        """The nominal wait before retrying a failed request, or None when the failure is final."""
        transient = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
        if not transient:
            return None
        allowed = 1 if status in (None, 408) else len(WAITS)
        if attempt >= allowed:
            return None
        wait = WAITS[attempt]
        if self.waited + wait > WAIT_TOTAL or self.model_room() - wait * 1.2 < CALL_FLOOR:
            return None
        return wait

    def _b_failure_status(self, model: str, result: Any) -> Any:
        """Trace a failed model request and return its status (the provider's own error code when it sent one)."""
        fault = result.error or {}
        status, said = fault.get('status'), fault.get('body')
        data = result.data
        reported = data.get('error') if isinstance(data, dict) else None
        if isinstance(reported, dict):
            status, said = _count(reported.get('code')), reported.get('message')
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(said or '')[:200]})
        return status

    def _b_unmix_item(self, number: int, item: Item, pointers: list[str], others: list[str]) -> None:
        """Ask which plain requirements only describe the listing a pointer requirement points at, and fold them
        into that pointer."""
        prompt = ''.join([UNMIX, '\n\nREQUEST:\n', self.request, f'\n\nPRODUCT: {item.name}',
                          '\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n', '\n'.join(pointers),
                          '\nOTHER REQUIREMENTS:\n', '\n'.join(others)])
        answer = self.ask_json(prompt, [HEAVY, LIGHT], 1500) or {}
        flagged = {str(x).strip() for x in _items(answer.get('describe_other'))}
        moved = [must for must in others if must in flagged]
        if not moved:
            return
        anchor = pointers[0]
        joined = anchor + ' (that listing: ' + '; '.join(moved) + ')'
        item.musts = [joined if must == anchor else must for must in item.musts if must not in moved]
        self.refs = [(number, joined) if ref == (number, anchor) else ref for ref in self.refs]
        self.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})

    def _b_is_pointer(self, need: str) -> bool:
        if any(pointed == need for _, pointed in self.refs):
            return True
        return need in {str(x) for x in _items(self.brief.get('lookups'))}

    def _b_names_other_item(self, item: Item, value: str) -> bool:
        """The value carries a long word of another requested item's name."""
        value_words = _stem(value)
        for other in self.items:
            if other is not item and any(len(w) >= 4 and w in value_words for w in _stem(other.name)):
                return True
        return False

    def _b_item_keys(self, item: Item) -> tuple[set[str], int]:
        """Attribute keys over the read records of the item's category, and how many such records were read."""
        cat_key = self.category_key()
        keys: set[str] = set()
        read = 0
        for record in self.full.values():
            if cat_key and not self.owns(item, str(record.get(cat_key) or '')):
                continue
            block = _fact_block(record)
            if not block:
                continue
            read += 1
            keys.update(_words(k) for k in block)
        return keys, read

    def _b_item_line(self, number: int, item: Item) -> str:
        live = [m for m in item.musts + item.told if m not in item.dropped]
        verdicts = [(m, self.described(number, m)) for m in live]
        told = [m for m, unstated in verdicts if unstated]
        needs = [self._b_with_ref(m) for m, unstated in verdicts if not unstated]
        pieces = [f'item {number}: ' if len(self.items) > 1 else 'item: ', f'{item.name}']
        if item.shelf:
            pieces.append(f' (store label: {item.shelf})')
        pieces += [f', quantity {item.qty}', '; requirements: ', '; '.join(needs) if needs else 'none beyond the kind of product']
        if told:
            pieces += [_B_UNSTATED_HEAD, '; '.join(told)]
        return ''.join(pieces)

    def _b_with_ref(self, need: str) -> str:
        if need not in self.ref_values:
            return need
        return need + f' (the page it points at gives: {self.ref_values[need]})'

    def _b_terms_lines(self, brief: dict[str, Any]) -> list[str]:
        """Budget, pick rule, stock, wishes, rules and the decline wording."""
        lines = []
        if self.budget is not None:
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            lines.append(' '.join(['budget: at most', f'{self.budget}', f"{brief.get('budget_unit') or ''}", scope]).replace('  ', ' '))
        lines.append(self._b_pick_line(brief))
        if brief.get('in_stock_only', True) is not False:
            lines.append('in stock only')
        lines += [f'wish (tie-breaker only): {wish}' for wish in _items(brief.get('wishes'))]
        lines += [f'rule from the request: {rule}' for rule in _items(brief.get('rules'))]
        if brief.get('quiet_decline') is True:
            lines.append('if nothing fits: end without naming or suggesting any listing')
        return lines

    @staticmethod
    def _b_pick_line(brief: dict[str, Any]) -> str:
        pick = brief.get('pick')
        if pick == 'cheapest':
            return 'pick: the cheapest listing that fits (lowest price)'
        if pick not in ('most', 'least'):
            return 'pick: any listing that fits everything'
        measure = f"{brief.get('pick_measure') or ''}"
        return ' '.join(['pick: the', pick, measure, 'among fitting listings']).replace('  ', ' ')

    def _b_facts_line(self) -> str | None:
        """Which tool shows listing attributes, once K2 knows it."""
        facts = self.facts_tool()
        if not facts or (self.facts_from.get(facts) is not True and not self.roles.get('offer')):
            return None
        pages = self.page_tools()
        bare = sorted(t for t in pages | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False)
        others = sorted(pages - {facts})
        text = [f'listing attributes come from {facts}']
        if bare:
            text.append(f"; {', '.join(bare)} show none")
        if others:
            text += [_B_PAGE_TAIL, ' or '.join(others)]
        return ''.join(text)

    def _b_pace_note(self, left: int) -> str:
        """The step-budget warning for this turn, or '' when there is none."""
        if left <= self.tail() + 1:
            return _B_FINAL_NOTE
        idle = self.buys == 0
        if left <= 6 or (idle and left <= len(self.items) + 5):
            return _B_FEW_NOTE
        if not idle:
            return ''
        if left <= 12:
            return _B_LOW_NOTE
        if not self.cart_touched and self.step >= self.max_steps // 2:
            return _B_HALF_NOTE
        return ''

    def _b_visit(self, node: Any, depth: int, in_list: bool) -> None:
        """Walk a store result: every listing found is merged into the row store; other containers are searched."""
        if depth > 6:
            return
        if isinstance(node, list):
            listed = self.learn_rows(node)
            for child in node:
                self._b_visit(child, depth + 1, listed)
            return
        if not isinstance(node, dict):
            return
        ident = self.ident(node, in_list, depth)
        if ident:
            self._b_store_row(ident, node)
            return
        for child in node.values():
            self._b_visit(child, depth + 1, False)

    def _b_store_row(self, ident: str, node: dict[str, Any]) -> None:
        if ident not in self.rows:
            self.seen_order.append(ident)
            self.rows[ident] = {}
        stored = self.rows[ident]
        if _b_flipped(stored, node, True, False):
            self.unavailable.add(ident)
        elif _b_flipped(stored, node, False, True):
            self.unavailable.discard(ident)
        stored.update(node)
        if _fact_block(node):
            self.full[ident] = dict(stored)
        elif ident in self.full:
            self.full[ident].update(node)

    def _c_short_texts(self) -> dict[str, list[Any]]:
        # column -> [distinct short texts without long numbers, number of rows holding such a text]
        tally: dict[str, list[Any]] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if not isinstance(value, str) or not 0 < len(value) <= 40 or re.search('\\d{3}', value):
                    continue
                entry = tally.setdefault(key, [set(), 0])
                entry[0].add(value)
                entry[1] += 1
        return tally

    def _c_item_keys(self) -> list[tuple[str, str, set[str]]]:
        # per wanted item: shelf phrase, name phrase, and the 4+ letter stems of both
        keys = []
        for item in self.items:
            shelf = item.shelf or ''
            stems = {w for w in _stem(shelf) | _stem(item.name) if len(w) >= 4}
            keys.append((_words(shelf), _words(item.name), stems))
        return keys

    def _c_label_wanted(self, label: str, item_keys: list[tuple[str, str, set[str]]]) -> Any:
        phrase = _words(label)
        stems = {w for w in _stem(label) if len(w) >= 4}
        named = any(phrase in (shelf, name) or stems & long for shelf, name, long in item_keys)
        return named or self.owner_of(label)

    def _c_text_slot(self, schema: dict[str, Any], lone: bool) -> str | None:
        # the free-text argument of a tool: the first required one, else the only one (lone) or the first one
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v]
        required = next((k for k in schema.get('required') or [] if k in texts), None)
        if required:
            return required
        if texts and (len(texts) == 1 or not lone):
            return texts[0]
        return None

    def _c_current_view(self, ident: str) -> set[str]:
        # the fields seen for a listing since the last change of requirements (empty when seen only before it)
        version, fields = self.view_keys.get(ident, (-1, set()))
        return fields if version == self.version else set()

    def _c_shape_prompt(self, name: str, observation: dict[str, Any], lists: list[str]) -> str:
        preview = {k: (v[:3] if k in lists else v) for k, v in observation.items()}
        parts = (SHAPE, '\n\nTOOL: ', name, ' | parameters: ', _dump(self.schemas.get(name) or {}),
                 '\n\nRESULT (first rows only):\n', _dump(_clip(preview, 120))[:4000])
        return ''.join(parts)

    def _c_shape_from(self, name: str, got: dict[str, Any], observation: dict[str, Any], rows: str) -> dict[str, Any]:
        sample = observation[rows][0]
        params = (self.schemas.get(name) or {}).get('properties') or {}
        count = _c_key_in(observation, got.get('count'))
        if count and not _c_is_whole(observation[count]):
            count = None
        return {'rows': rows, 'count': count, 'next': _c_key_in(observation, got.get('next')),
                'next_arg': _c_key_in(params, got.get('next_arg')), 'price': _c_key_in(sample, got.get('price')),
                'stock': _c_key_in(sample, got.get('stock'))}

    def _c_pages_to_read(self, count: int, shown: int) -> int | None:
        # pages left to read, trimmed to two when budget allows; None when reading them itself would cost too much
        pages = (count - 1) // shown
        room = self.left() - self.tail()
        if pages > 2 and self.pages_read + 2 <= AUTO_PAGES and room - 2 >= 12:
            pages = 2
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or room - pages < 12:
            return None
        return pages

    def _c_read_pages(self, name: str, args: dict[str, Any], param: str, cursor: Any, pages: int) -> tuple[list[str], Any]:
        texts: list[str] = []
        while len(texts) < pages:
            call = _c_shaped(self.schemas.get(name), {**args, param: cursor})
            if call is None:
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
            texts.append(f'{param}={_dump(cursor)}: {shown_text}')
            following = self.paging(name, seen)
            cursor = following[3] if following else None
            if cursor is None:
                break
        return texts, cursor

    def _c_price_or_inf(self, ident: str) -> float:
        price = self.price_of(ident)
        return float('inf') if price is None else price

    def _c_lane(self, ids: list[str], wanted: set[str], cheapest: bool) -> list[str]:
        # unopened buyable listings of one group, those sharing the most requirement words first (cheapest
        # first among equals when the pick is the cheapest)
        pool = [i for i in ids if i not in self.full and self.buyable(i)]
        if cheapest:
            pool.sort(key=self._c_price_or_inf)
        pool.sort(key=lambda i: -len(wanted & _c_alnum3(_dump(self.rows.get(i) or {}))))
        return pool

    def _d_varied_fields(self, free: list[str]) -> list[str]:
        """Without a query-named field: the text fields earlier searches filled with differing phrases (one of two or
        more words) and never with a category label."""
        cat_key = self.category_key()
        labels = {_d_label(r, cat_key).strip().lower() for r in self.rows.values()} - {''} if cat_key else set()

        def varied(k: str) -> bool:
            distinct = {str(a.get(k)) for a in self.searched if a.get(k)}
            phrased = any(len(str(a.get(k)).split()) >= 2 for a in self.searched)
            return len(distinct) >= 2 and phrased and not any(str(a.get(k) or '').strip().lower() in labels for a in self.searched)
        return [k for k in free if varied(k)]

    def _d_shelf_names(self) -> set[str]:
        cat_key = self.category_key()
        names = {_words(item.shelf or '') for item in self.items} | {_words(item.name) for item in self.items}
        if cat_key:
            names |= {_words(_d_label(row, cat_key)) for row in self.rows.values()}
        return names

    def _d_keeps(self, props: dict[str, Any], required: list[str], shelves: set[str], key: str, value: Any) -> bool:
        """Whether a copied search keeps this argument: required ones always; option fields unless their text names a
        shelf (an item's own category, which would narrow the words search)."""
        if key in required:
            return True
        if not isinstance((props.get(key) or {}).get('enum'), list):
            return False
        if not isinstance(value, str):
            return True
        own = _d_long_words(value)
        return _words(value) not in shelves and not any(own & _d_long_words(s) for s in shelves if s)

    def _d_common_modes(self, kept: dict[str, Any], props: dict[str, Any], field: str) -> None:
        """Set each kept option field to the value most worded searches used (in place)."""
        worded = [a for a in self.searched if _d_has_letter(a.get(field))]
        for key in [k for k in kept if isinstance((props.get(k) or {}).get('enum'), list)]:
            pairs = [(a[key], _dump(a[key])) for a in worded if key in a]
            if not pairs:
                continue
            dumps = [d for _, d in pairs]
            best = max(dumps, key=dumps.count)
            kept[key] = next(v for v, d in pairs if d == best)

    def _d_value_word(self, need: str, told: list[str]) -> tuple[str, bool] | None:
        value = need.split(':', 1)[1].strip()
        edge = re.search(FLOOR, value, re.I)
        if edge:
            return (edge.group(1).strip(), True) if need in told else None
        if value and not re.search(_d_hedged, value, re.I):
            return (value, need in told)
        return None

    def _d_tally(self, item: Item, pool: dict[str, Any], cat_key: str) -> dict[str, int]:
        tally: dict[str, int] = {}
        for label in (_d_label(record, cat_key) for record in pool.values()):
            if label and self.owns(item, label):
                key = _words(label)
                tally[key] = tally.get(key, 0) + 1
        return tally

    def _d_value_roots(self) -> dict[str, set[str]]:
        roots: dict[str, set[str]] = {}
        for _, key, _, values in self.key_table():
            bucket = roots.setdefault(_words(key), set())
            for value in values:
                bucket.update(_roots(value))
        return roots

    def _d_plan_prompt(self, count: int, table: list[tuple[str, str, int, list[str]]]) -> str:
        items = []
        for number, item in enumerate(self.items, 1):
            items.append(f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing'))
        sections = [PLAN.replace('COUNT', str(max(1, count))), 'REQUEST:\n' + self.request, 'ITEMS:\n' + '\n'.join(items)]
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            sections.append('ASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first))
        sections.append('ATTRIBUTE KEYS SEEN:\n' + ('\n'.join(self._d_key_lines(table)) or '(no full listing records yet)'))
        return '\n\n'.join(sections)

    def _d_key_lines(self, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        """The listing keys for the planner, numbered, under a header line per category."""
        lines: list[str] = []
        previous = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != previous:
                lines.append(f'category {label!r}:')
            previous = label
            shown = ' | '.join(values)
            caveat = self.not_stated(label, name, values)
            broad = ' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''
            lines.append(f'  {index}. {name} (on {carried}): {shown}{caveat}{broad}')
        return lines

    def _d_slot(self, value: Any) -> int:
        """An item number a model gave, clamped to the request's items (1 when missing or unreadable)."""
        return min(max(1, _count(value) or 1), len(self.items))

    def _d_read_plan(self, plan: dict[str, Any]) -> tuple[dict[int, list[str]], dict[int, set[str]]]:
        """The planner's labels per item (first phrase only, at most five words, no repeats) and the hinted ones;
        an item with a hinted label joins the plan order."""
        picks: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        hints: dict[int, set[str]] = {n: set() for n in picks}
        for pick in _items(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = self._d_slot(pick.get('item'))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            if _words(label) not in {_words(other) for other in picks[number]}:
                picks[number].append(label)
            if not str(pick.get('hint') or '').strip():
                continue
            hints[number].add(_words(label))
            if number not in self.plan_order:
                self.plan_order.append(number)
        return picks, hints

    def _d_item_shares(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> dict[str, float]:
        """For each key on the item's own categories, the best share of its opened listings carrying it."""
        read = max(1, self.records_of_item(number))
        own: dict[str, float] = {}
        for label, key, carried, _ in table:
            if self.owner_of(label) == number:
                own[key] = max(own.get(key, 0.0), carried / read)
        return own

    def _d_lane_for(self, number: int, chosen: list[str], table: list[tuple[str, str, int, list[str]]], hints: set[str]) -> list[str]:
        """An item's question order: the planner's labels in the listings' own spelling, frequent keys guarded into
        the first three, then every other listing key, with catch-all keys last."""
        keys = self.lane_keys(number, table)
        own = self._d_item_shares(number, table)

        def spelled(label: str) -> str:
            flat = _d_plain(label)
            return next((k for k in own if _d_plain(k) == flat), label)
        hints = {_words(spelled(h)) for h in hints}
        lane: list[str] = []
        taken: set[str] = set()
        for pick in chosen:
            form = spelled(pick)
            if _words(form) not in taken:
                lane.append(form)
                taken.add(_words(form))
        guards = [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and not re.search(CATCH_ALL, k)]
        for key in guards[:GUARD_KEYS]:
            lane = self._d_with_guard(lane, key, own, hints) or lane
        taken = {_words(x) for x in lane}
        for key in keys:
            if _words(key) not in taken:
                lane.append(key)
                taken.add(_words(key))
        return _d_catch_all_last(lane, folded=True)

    def _d_with_guard(self, lane: list[str], key: str, own: dict[str, float], hints: set[str]) -> list[str] | None:
        """The lane with a well-carried key moved into its first three, in place of a much rarer unhinted label (or
        appended when the lane is short). None leaves the lane as it is: the key is already up front, nothing there
        is weak enough, or the move would push a hinted label out of the first three."""
        word = _words(key)
        front = lane[:3]
        if word in [_words(x) for x in front]:
            return None
        weight = own.get(key, 0.0)
        weak = next((i for i, x in enumerate(front) if _words(x) not in hints and _d_share(own, x) + GUARD_GAP <= weight), None)
        if weak is None and len(lane) >= 3:
            return None
        placed = [x for x in lane if _words(x) != word]
        if weak is None:
            placed.append(key)
        elif len(self.items) > 1:
            moved, placed[weak] = placed[weak], key
            placed.insert(min(3, len(placed)), moved)
        else:
            placed.insert(min(len(placed), weak), key)
        kept_front = {_words(y) for y in placed[:3]}
        if any(_words(x) in hints and _words(x) not in kept_front for x in front):
            return None
        return placed

    def _d_row_words(self, item: Item) -> list[set[str]]:
        """Root words of each search row of the item's kind, from its text fields other than the category label."""
        cat_key = self.category_key()
        rows = [row for row in self.rows.values() if not cat_key or self.owns(item, _d_label(row, cat_key))]
        return [_roots(' '.join(v for k, v in row.items() if k != cat_key and isinstance(v, str))) for row in rows]

    def _d_may_message(self, floor: int) -> bool:
        return not self.finished and self.left() > floor and self.msg_used < self.msg_limit

    def _d_log_exchange(self, text: str, words: str) -> None:
        said = words[:300]
        exchange = f'message to the shopper {text!r} -> reply {said!r}'
        self.done_log.append(f'step {self.step}: {exchange} (cart then: {self.cart_mark()})')
        self.done_steps.append(self.step)
        self.trace.append({'kind': 'ask', 'q': text, 'a': said})

    def _d_take_needs(self, entries: list[Any], told: list[str]) -> None:
        many = len(self.items) > 1
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            need = str(entry.get('need') or '').strip()
            if not need:
                continue
            number = self._d_slot(entry.get('item'))
            known = self.items[number - 1].told
            if need not in known:
                known.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in self.refs:
                self.refs.append((number, need))

    def _d_take_drops(self, entries: list[Any], told: list[str]) -> None:
        for entry in entries:
            if not isinstance(entry, dict) or not str(entry.get('property') or '').strip():
                continue
            item = self.items[self._d_slot(entry.get('item')) - 1]
            gone = _words(str(entry['property']).partition(':')[0])
            hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.partition(':')[0]) == gone]
            item.dropped.extend(hit)
            told.extend(f'dropped {m}' for m in hit)

    def _e_versioned(self, text: str) -> str:
        return f'{text}#{self.version}'

    def _e_loose_from(self, value: Any) -> list[str]:
        """Sentences carried by one field of a reply that is not itself a listing."""
        if isinstance(value, str):
            return [value] if _e_has_words(value, 5) else []
        if isinstance(value, list):
            if not value or not all(isinstance(v, str) for v in value):
                return []
            return [v for v in value if _e_has_words(v, 5)]
        if isinstance(value, dict) and not self.ident(value, False, 1):
            return [v for v in value.values() if isinstance(v, str) and _e_has_words(v, 6)]
        return []

    def _e_note_stock(self, got: dict[str, Any], lines: list[str]) -> list[str]:
        """Record which listings the note says are gone; returns every id the note names as gone."""
        gone = [str(g) for g in _items(got.get('gone')) if isinstance(g, (str, int))]
        listed = [g for g in gone if g in self.rows]
        self.unavailable.update(listed)
        self.note_sold = bool(gone) or got.get('sold_out') is True
        self.note_named = bool(listed)
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        return gone

    def _e_item_for(self, entry: dict[str, Any]) -> Any:
        position = _count(entry.get('item')) or 1
        return self.items[min(max(1, position), len(self.items)) - 1]

    def _e_drop_need(self, entry: dict[str, Any]) -> str:
        item = self._e_item_for(entry)
        target = str(entry['need']).strip()
        live = [m for m in item.musts + item.told if m not in item.dropped]
        matched = [m for m in live if m == target]
        if not matched:
            head = _words(target.split(':')[0])
            matched = next(([m] for m in live if _words(m.split(':')[0]) == head), [])
        item.dropped.extend(matched or [target])
        return f'dropped {target}'

    def _e_add_need(self, entry: dict[str, Any]) -> str:
        item = self._e_item_for(entry)
        need = str(entry['need']).strip()
        if need in item.dropped:
            item.dropped.remove(need)
        if need not in item.musts and need not in item.told:
            item.told.append(need)
        return f'added {need}'

    def _e_view_fresh(self) -> bool:
        """Whether the last cart view was read after the last cart change."""
        return self.cart_view is not None and self.cart_seen_at >= self.cart_at

    def _e_units_on(self, node: dict[str, Any], ident: str) -> int | None:
        """The unit count a cart-view entry gives for the listing, when the entry names it and has one count field."""
        if ident not in [v.strip() for v in node.values() if isinstance(v, str)]:
            return None
        counts: list[int] = []
        for key, value in node.items():
            if isinstance(value, str):
                continue
            number = _count(value)
            if number is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _words(key)):
                counts.append(number)
        return counts[0] if len(counts) == 1 else None

    def _e_record(self, ident: str) -> Any:
        return self.full.get(ident) or self.rows.get(ident)

    def _e_label(self, ident: str, cat_key: str) -> str:
        return str((self.rows.get(ident) or {}).get(cat_key) or '')

    def _e_held_ids(self) -> list[str]:
        return [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]

    def _e_owns_any(self, item: Any, labels: list[str]) -> bool:
        return any(self.owns(item, label) for label in labels)

    def _e_cheap_first(self, ident: str) -> tuple[bool, float]:
        price = self.price_of(ident)
        return (ident not in self.full, price if price is not None else float('inf'))

    def _e_cheapest_fit(self, number: int) -> str | None:
        fits = [i for i in self.item_ids(number) if i in self.full and i not in self.cart_lines and self.buyable(i) and self.price_of(i) is not None and self.states_needs(i, number)]
        if not fits:
            return None
        return min(fits, key=lambda i: self.price_of(i) or 0.0)

    def _e_affordable(self, pick: str, qty: int) -> bool:
        if not isinstance(self.budget, (int, float)):
            return True
        each = self.brief.get('budget_scope') == 'each'
        held = sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in self.cart_ids()))
        cost = (self.price_of(pick) or 0.0) * (1 if each else qty)
        over = cost + (0.0 if each else held) > self.budget + 1e-09
        return not over

    def _e_lines_with(self, pick: str, qty: int) -> dict[str, int]:
        lines = {i: self.cart_lines[i] for i in self.cart_ids() if i in self.cart_lines}
        lines[pick] = qty
        return lines

    def _e_send_cart(self, cart: str, calls: list[dict[str, Any]]) -> bool:
        for planned in calls:
            faults: list[str] = []
            args = _shape(planned, self.schemas.get(cart), 'args', faults)
            if faults or self.act(cart, args) is None:
                return False
            self.note_cart(cart, args)
        return True

    def _e_id_field(self, name: str, schema: dict[str, Any], props: dict[str, Any]) -> str | None:
        """The cart tool's argument that names the listing: one learned from earlier calls, a required text field, or
        the only text field."""
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v]
        learned = next((k for t, k in self.id_args if t == name and k in texts), None)
        if learned:
            return learned
        required = next((k for k in schema.get('required') or [] if k in texts), None)
        if required:
            return required
        return texts[0] if len(texts) == 1 else None

    def _e_count_calls(self, wrong: list[tuple[str, int, int]], name: str, props: dict[str, Any], target: str, count: str, mode: str | None) -> list[dict[str, Any]] | None:
        """Cart calls that bring each line to the wanted count; None when the cart tool cannot express it."""
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if self.cart_sets(name):
                calls.append({target: ident, count: want})
                continue
            if not mode:
                return None
            off, on = _e_switch_modes([str(o) for o in props[mode]['enum']])
            calls.append({target: ident, mode: off, count: have})
            calls.append({target: ident, mode: on, count: want})
        return calls

    def _e_run_counts(self, name: str, schema: dict[str, Any], calls: list[dict[str, Any]]) -> list[str]:
        done: list[str] = []
        for planned in calls:
            faults: list[str] = []
            args = _shape(planned, schema, 'args', faults)
            if faults:
                break
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            seen = reply.get('observation')
            self.note_cart(name, args)
            self.cart_said = _dump(seen)
            self.observe(name, seen)
            done.append(_dump(args) + ' -> ' + self.show(seen)[:300])
        return done

    def _e_check_prompt(self, listing: list[str], planned: bool, ranked: bool) -> str:
        parts = [('REQUEST:\n', self.request), ('REQUIREMENTS IN FORCE:\n', self.ledger())]
        if planned:
            quantity = '; '.join(f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1))
            parts.append(('CART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n', ', '.join(listing)))
        else:
            parts.append(('CART:\n', self.cart_text()))
        records = [f'{i}: ' + (_E_SOLD_TAG if self.sold_out(i) else '') + self.record_of(i) for i in listing]
        parts.append(("RECORDS OF THE CART'S LISTINGS:\n", '\n'.join(records)))
        if ranked:
            parts.append(('OTHER LISTINGS SEEN (newest first):\n', self.others(listing, 15)))
        return _e_sections(CHECK, parts)

    def _e_check_verdict(self, prompt: str, planned: bool, ranked: bool, limit: float) -> dict[str, Any] | None:
        """A light screen first when it can settle the cart, then the judges still due."""
        judges = self.judges(limit)
        got = None
        if len(judges) > 1 and not (planned or self.audited or ranked):
            got = self.ask_json(prompt, [LIGHT], 3000)
            if got:
                rejected = got.get('ok') is False and HEAVY in self.judges(limit)
                judges = [HEAVY] if rejected else []
        if not judges or self.no_credit:
            return got
        return self.ask_json(prompt, judges, 3000) or got

    def _e_review_text(self, got: dict[str, Any]) -> str:
        problems = [str(p) for p in _items(got.get('problems'))[:5]]
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        pieces = ['A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])]
        if better:
            pieces.append('Listings that look better: ' + ', '.join(better))
        pieces.append(_E_REVIEW_END)
        return '. '.join(pieces)

    def _f_shaped(self, tool: str, args: dict[str, Any]) -> tuple[Any, bool]:
        """Arguments fitted to the tool's schema, and whether they fitted without a fault."""
        problems: list[str] = []
        fitted = _shape(args, self.schemas.get(tool), 'args', problems)
        return fitted, not problems

    def _f_retire(self, lost: list[str]) -> None:
        """Mark sold-out cart lines as gone for good and drop them from the leads."""
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1

    def _f_rushed_swap(self, picks: list[str]) -> str:
        others = ', '.join(picks[1:]) or 'none'
        return ''.join([' Few steps are left: in this turn add ', picks[0],
                        ' at the requested quantity and remove the sold-out line (two calls), then check out.',
                        ' Other listings already read that appear to fit: ', others, '.'])

    def _f_cheapest_swap(self, picks: list[str]) -> str:
        return ''.join([' Listings already read that appear to fit that item instead, cheapest first: ', ', '.join(picks),
                        '. The pick rule is the cheapest, so unless you have already seen the whole fitting set,',
                        ' run one search under ', picks[0], "'s price (the search's maximum price, if it has one)",
                        ' before you swap the sold-out line for the cheapest fitting listing.'])

    def _f_newest(self, limit: int, skip) -> list[str]:
        """Seen listings, newest first, leaving out those `skip` rejects, at most `limit` of them."""
        chosen: list[str] = []
        for ident in reversed(self.seen_order):
            if skip(ident):
                continue
            chosen.append(ident)
            if len(chosen) >= limit:
                break
        return chosen

    def _f_prompt_head(self, lead: str) -> str:
        return lead + _f_section('REQUEST', self.request) + _f_section('REQUIREMENTS IN FORCE', self.ledger())

    def _f_swap_fits(self, got: dict[str, Any] | None, lost: list[str]) -> list[tuple[float, str]]:
        """(price or inf, id) of each new, distinct, known listing the swap reviewer proposed."""
        fits: list[tuple[float, str]] = []
        taken: set[str] = set()
        for entry in _items((got or {}).get('fits')):
            ident, price = _f_fit_entry(entry)
            if ident in taken or ident not in self.rows or ident in lost or ident in self.gone or ident in self.cart_ids():
                continue
            taken.add(ident)
            fits.append((_f_price_key(price), ident))
        return fits

    def _f_audit_pass(self) -> tuple[dict[str, Any] | None, dict[str, list[str]]]:
        got = self.audit_call()
        return got, self.audit_fits(got)

    def _f_audit_note(self, named: dict[str, list[str]], reason: Any) -> str:
        listed = '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items()))
        why = f' ({reason})' if reason else ''
        return 'A careful review thinks these listings may fit: ' + listed + why + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def _f_live_ids(self, raw: Any) -> list[str]:
        """Ids in `raw` that are known rows and not sold out."""
        return [str(i) for i in _items(raw) if str(i) in self.rows and (not self.sold_out(str(i)))]

    def _f_cart_target(self, name: str, schema: dict[str, Any], texts: list[str]) -> str | None:
        """The id argument of the cart tool: a learned id argument, else a required text field, else the only one."""
        learned = next((k for t, k in self.id_args if t == name and k in texts), None)
        if learned:
            return learned
        required = next((k for k in schema.get('required') or [] if k in texts), None)
        if required:
            return required
        return texts[0] if len(texts) == 1 else None

    def _f_drop_call(self, name: str, props: dict[str, Any], target: str, count: str | None, mode: str | None, ident: str) -> dict[str, Any] | None:
        """The call that takes a whole line out of the cart, or None when the tool cannot express it."""
        held = self.cart_lines.get(ident, 1)
        call: dict[str, Any] = {target: ident}
        if mode:
            call[mode] = props[mode]['enum'][1]
            if count:
                call[count] = held
            return call
        if not count:
            return None
        if self.cart_sets(name):
            call[count] = 0
            return call
        floor = props[count].get('minimum')
        if isinstance(floor, int) and floor <= -held:
            call[count] = -held
            return call
        return None

    def _f_kind_of(self, cat_key: str | None, ident: str) -> str:
        """The listing's category text (the id itself when no category field is known)."""
        if not cat_key:
            return ident
        return str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower()

    def _f_item_index(self, number: Any) -> int | None:
        found = re.search('\\d+', str(number))
        if found:
            return int(found.group())
        return 1 if len(self.items) == 1 else None

    def _f_lead_plan(self, cat_key: str | None) -> dict[str, int]:
        """First buyable lead of each requested item -> its quantity, one listing per category."""
        plan: dict[str, int] = {}
        taken: set[str] = set()
        for number, ids in self.lead_items.items():
            index = self._f_item_index(number)
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if not (pick and index and 1 <= index <= len(self.items)):
                continue
            sort = self._f_kind_of(cat_key, pick)
            if sort in taken:
                continue
            plan[pick] = self.items[index - 1].qty
            taken.add(sort)
        return plan

    def _f_cart_split(self, cat_key: str | None, plan: dict[str, int]) -> tuple[dict[str, int], list[str]] | None:
        """Final lines (kept cart lines, then the plan) and lines to drop; None when items stay uncovered."""
        planned = {self._f_kind_of(cat_key, p) for p in plan}
        keep: dict[str, int] = {}
        drop: list[str] = []
        for ident in self.cart_ids():
            if ident in plan:
                continue
            if self.sold_out(ident) or self._f_kind_of(cat_key, ident) in planned:
                drop.append(ident)
            else:
                keep[ident] = self.cart_lines.get(ident, 1)
        if len(plan) + len({self._f_kind_of(cat_key, i) for i in keep}) < len(self.items):
            return None
        return {**keep, **plan}, drop

    def _f_cart_verified(self, lines: dict[str, int], room: int) -> bool:
        """Gather facts and links for the final lines, then ask the independent check."""
        self.fill_facts(list(lines), room)
        self.show_links(list(lines), room)
        self.check_cart(list(lines), SPEND_HARD)
        return bool(self.verdicts.get(self.check_key(list(lines))))

    def _f_set_cart(self, cart: str, calls: list[dict[str, Any]]) -> list[str] | None:
        """Send the cart calls; the sent arguments as text, or None when one could not be sent."""
        sent = []
        for call in calls:
            args, fits = self._f_shaped(cart, call)
            if not fits:
                return None
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return None
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
            sent.append(_dump(args))
        return sent

    def _f_checkout(self, buy: str, args: dict[str, Any], lines: dict[str, int], drop: list[str]) -> list[str]:
        """Order, and when the store refused, re-check the cart and order once more."""
        replies: list[str] = []
        if self._f_order_once(buy, args, lines, drop, replies) and self._f_still_orderable(lines):
            self._f_order_once(buy, args, lines, drop, replies)
        return replies

    def _f_still_orderable(self, lines: dict[str, int]) -> bool:
        self.reread_offers()
        self.check_cart(list(lines), SPEND_HARD)
        if not self.verdicts.get(self.check_key(list(lines))):
            return False
        return self.left() >= 1 and not self.finished and not any((self.sold_out(i) for i in lines))

    def _f_order_once(self, buy: str, args: dict[str, Any], lines: dict[str, int], drop: list[str], replies: list[str]) -> bool:
        """One pass through the order gate and the buy call; True when another try may follow."""
        go, said = self.gate('order')
        if said:
            replies.append(said)
        if not go:
            return False
        reply = self.act(buy, args)
        self.trace.append({'kind': 'completed', 'lines': lines, 'drop': drop, 'reply': _clip(reply, 200)})
        if reply is None:
            return False
        self.buys += 1
        if not reply.get('finished'):
            self.refused_at = self.step
        self.observe(buy, reply.get('observation'))
        replies.append(self.show(reply.get('observation')))
        return not (self.finished or reply.get('finished'))

    def _f_role_call(self, role: str, preset: str, name: str, args: dict[str, Any]) -> bool:
        """The call uses the role's tool and repeats every fixed argument the brief gives for it."""
        if name != self.roles.get(role):
            return False
        fixed = self.brief.get(preset)
        if not isinstance(fixed, dict):
            return True
        return all((args.get(k) == v for k, v in fixed.items()))

    def _f_should_decline(self, text: str) -> bool:
        """Free text that names a listing is replaced, and so is any text unless the brief keeps own words."""
        return any((ident in text for ident in self.rows)) or self.brief.get('quiet_decline') is not False

    def _f_id_field(self, name: str) -> str | None:
        """The single free field of a listing tool that carries listing ids, if that is known."""
        handled = {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None}
        if name not in handled or not self.id_key:
            return None
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _kinds(v) or 'array' in _kinds(v)) and ('enum' not in v)]
        if len(free) != 1:
            return None
        if name != self.roles.get('cart') and (name, free[0]) not in self.id_args:
            return None
        return free[0]

    def _f_fix_id(self, value: Any) -> tuple[Any, str]:
        """Map a title-like value to the one listing whose text starts with it; a problem text when it cannot."""
        if not isinstance(value, str):
            return (value, '')
        bare = value.strip()
        if bare in self.rows:
            return (bare, '')
        if _id_like(value, True) or len(bare) < 8:
            return (value, '')
        start = bare.lower()
        hits = [i for i, row in self.rows.items() if any((k != self.id_key and isinstance(v, str) and v.strip().lower().startswith(start) for k, v in row.items()))]
        if len(hits) == 1:
            return (hits[0], '')
        if not hits:
            return (value, f'{value!r} is not a listing id; pass the id from the result rows (their {self.id_key} field)')
        return (value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id")

    def _f_lines_from(self, field_spec: Any, value: Any, loose: int | None) -> tuple[list[tuple[str, int | None]], bool]:
        """Cart lines one argument names, and whether it lists the whole cart (an array of line objects)."""
        if isinstance(value, str):
            return ([(value, loose)] if value in self.rows else []), False
        if not isinstance(value, list):
            return [], False
        entry = (field_spec or {}).get('items') or {}
        replaces = not value and isinstance(entry, dict) and (entry.get('type') == 'object' or 'properties' in entry)
        found: list[tuple[str, int | None]] = []
        for line in value:
            if isinstance(line, dict):
                replaces = True
                named = self._f_line_entry(line)
                if named:
                    found.append(named)
            elif isinstance(line, str) and line in self.rows:
                found.append((line, loose))
        return found, replaces

    def _f_line_entry(self, line: dict[str, Any]) -> tuple[str, int] | None:
        ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
        units = next((_count(v) for v in line.values() if _count(v) is not None and (not isinstance(v, str))), 1)
        return (ident, units) if ident else None

    def _g_compare_schema(self) -> dict[str, Any]:
        """Schema of the side-by-side tool, or an empty dict when the store has none."""
        return self.schemas.get(self.roles.get('compare') or '') or {}

    def _g_open_pick(self, call: dict[str, Any], opener: str, field: str) -> str | None:
        """The listing id a plain 'open one listing' call asks for, if it may join a side-by-side batch."""
        fn = call.get('function') or {}
        packed = fn.get('arguments')
        if not isinstance(packed, str) or fn.get('name') != opener:
            return None
        args = _json_obj(packed)
        if not args or set(args) != {field}:
            return None
        wanted = str(args.get(field) or '').strip()
        return wanted if wanted in self.rows and wanted not in self.gone else None

    def _g_cart_fresh(self) -> bool:
        return self.cart_view is not None and self.cart_seen_at >= self.cart_at

    def _g_fit_args(self, tool: str | None, start: dict[str, Any]) -> tuple[Any, bool]:
        """Shape preset arguments for a tool; returns (args, True when they fit its schema)."""
        problems: list[str] = []
        shaped = _shape(start, self.schemas.get(tool), 'args', problems)
        return shaped, not problems

    def _g_decline_args(self, refuse: str) -> tuple[Any, bool]:
        return self._g_fit_args(refuse, self.quiet(refuse, dict(self.brief.get('refuse_args') or {})))

    def _g_over(self) -> bool:
        return self.finished or self.left() <= 0

    def _g_settle_cart(self) -> Any:
        """Complete a multi-item cart before the forced ending; falsy when it stays short and declining is possible."""
        whole = self.fill_short_cart() or not self.can_refuse()
        if not whole:
            self.trace.append({'kind': 'short_cart_refused', 'lines': self.cart_ids()})
            return whole
        if len(self.cart_ids()) >= 2:
            self.fill_facts(self.cart_ids(), self.tail())
            if self.left() >= 3 and self.model_room() > 5:
                self.show_links(self.cart_ids(), self.tail())
        return whole

    def _g_closing_call(self, tool: str, args: Any, stage: str, kind: str) -> tuple[bool, Any]:
        """Pass the approval gate and send an ending call; (False, None) when the episode closed first."""
        if self.left() >= 2:
            self.gate(stage)
        if self._g_over():
            return False, None
        reply = self.act(tool, args)
        self.trace.append({'kind': kind, 'reply': _clip(reply, 200)})
        return True, reply

    def _g_forced_buy(self, buy: str, again: bool) -> str | None:
        """Check out on K2's own; None means fall through to ending without a purchase."""
        args, ok = self._g_fit_args(buy, dict(self.brief.get('buy_args') or {}))
        if not ok:
            return None
        sent, reply = self._g_closing_call(buy, args, 'order', 'forced_buy')
        if not sent or self._g_over():
            return ''
        if reply is None or not again:
            return None
        self.buys += 1
        self.refused_at = self.step
        self.observe(buy, reply.get('observation'))
        return 'the checkout was refused: ' + self.show(reply.get('observation'))

    def _g_forced_refuse(self, refuse: str) -> str:
        args, ok = self._g_decline_args(refuse)
        if not ok:
            return ''
        sent, reply = self._g_closing_call(refuse, args, 'end', 'forced_refuse')
        if not sent or reply is None or self.finished:
            return ''
        self.observe(refuse, reply.get('observation'))
        return 'ending without a purchase was refused: ' + self.show(reply.get('observation'))

    def _g_shorten(self, message: dict[str, Any]) -> None:
        """Cut an old tool result down, keeping the listing ids it mentioned."""
        content = message.get('content') or ''
        if len(content) <= 600:
            return
        quoted = dict.fromkeys(re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content))
        known = [i for i in quoted if i in self.rows][:20]
        message['content'] = content[:400] + '...(older result shortened; listings in it: ' + ', '.join(known) + ')'

    @staticmethod
    def _h_step_cap(raw: Any) -> int:
        steps = _count(raw)
        if steps is None and isinstance(raw, float):
            steps = int(raw)
        if steps and steps > 0:
            return steps
        return _H_DEFAULT_STEPS

    def _h_spend_state(self) -> None:
        # failures, waits and what the model calls have cost so far
        self.broke = self.no_credit = False
        self.waited = self.misses = 0
        self.cost = 0.0
        self.calls = self.loop_calls = 0
        self.trace: list[dict[str, Any]] = _Stamped(self.clock)

    def _h_brief_state(self) -> None:
        # what the request asks for, and the talk with the shopper
        self.brief: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.items: list[Item] = []
        self.budget: Any = None
        self.msg_limit = self.msg_used = 0
        self.process: list[dict[str, Any]] = []
        self.reserve = self.gate_msgs = 0
        self.gate_ok: set[tuple[str, str]] = set()
        self.done_log, self.done_steps = [], []
        self.asked = False
        self.qa: list[tuple[str, str]] = []
        self.rows, self.full = {}, {}
        self.seen_order: list[str] = []
        self.heard: set[str] = set()
        self.notes: list[str] = []
        self.version = 0

    def _h_cart_state(self) -> None:
        # the cart as K2 last saw and changed it, and what went missing from the store
        self.cart_touched = False
        self.cart_at = self.cart_seen_at = -1
        self.cart_view: Any = None
        self.note_named = False
        self.cart_lines: dict[str, int] = {}
        self.checked: set[str] = set()
        self.audited = False
        self.offer_seen, self.gone = {}, {}
        self.unavailable, self.sold_lines = set(), set()
        self.note_sold = False
        self.held = 0
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ''
        self.refused_at = -1

    def _h_lead_state(self) -> None:
        # candidate listings, verdicts and the guards that hold a checkout back
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
        self.lane_misses, self.adopted = {}, {}
        self.capped: set[tuple[str, float]] = set()
        self.factless: set[tuple[str, str]] = set()
        self.trail_room = self.id_key = None
        self.facts_from: dict[str, bool] = {}
        self.queued: list[str] = []
        self.buys = self.refusals = 0
        self.recent, self.chat = [], []
        self.opened: set[str] = set()
        self.views: dict[str, set[str]] = {}
        self.fit_cached = None
        self.owners_seen: dict[tuple[str, int], int] = {}
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.trails = self.trail_blocks = 0
        self.compare_ok = self.compare_bad = 0

    def _h_view_state(self) -> None:
        # which keys each tool shows, read caches, result shapes and the paging and lane bookkeeping
        self.tool_keys: dict[str, set[str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.pages_read = 0
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key = self.stock_key = None
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.paged_obs, self.plan_order = [], []
        self.lanes, self.lane_found, self.lane_spent, self.lane_used = {}, {}, {}, {}
        self.late_asked = False
        self.one_product: set[int] = set()
        self.list_asks = 0
        self.apart: dict[str, dict[int, list[str]]] = {}
        self.follow_up: list[tuple[int, list[str], bool]] = []
        self.listing = True
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0

    def _h_search_state(self) -> None:
        # searches made, store pacing, and the hunt for values not yet seen
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.hunts = 0
        self.env_gap = ENV_GAP
        self.env_spent = 0.0
        self.env_count = 0
        self.read_calls: dict[str, Any] = {}
        self.read_args: dict[str, tuple[str, dict[str, Any]]] = {}
        self.value_rows, self.row_keys = set(), set()
        self.refuse_trailed = False
        self.env_at = 0.0
        self.hunt_stem: dict[tuple[int, tuple[str, ...]], str] = {}
        self.hunt_no_noun: set[int] = set()
        self.bare_pages: dict[str, int] = {}
        self.code_tries: list[bool] = []
        self.codes_dead = False
        self.hear_reads = 0
        self.fit_hunted = False
        self.hunt_mark = ''
        self.hunt_tried: dict[int, set[str]] = {}
        self.page_args: dict[str, str] = {}
        self.page_queries: dict[str, set[str]] = {}

    def _h_role_tools(self, jobs: tuple[str, ...]) -> set[Any]:
        return {self.roles.get(job) for job in jobs}

    def _h_code_row(self, value: Any, key: str, props: dict[str, Any], title: str) -> Any:
        if not isinstance(value, str):
            return value
        code = value.strip()
        if code in self.rows or not 3 <= len(code) <= 30:
            return value
        if not re.search('\\d', value) or 'enum' in (props.get(key) or {}):
            return value
        pattern = '(?<![a-z0-9])' + re.escape(code.lower()) + '(?![a-z0-9])'
        match = None
        for ident, row in self.rows.items():
            if not re.search(pattern, str(row.get(title) or '').lower()):
                continue
            if match is not None:
                return value
            match = ident
        if match is None:
            return value
        self.trace.append({'kind': 'row_id', 'sent': value, 'id': match})
        return match

    def _h_space_out(self) -> None:
        """Keep the minimum gap between two store calls."""
        wait = self.env_at + self.env_gap - monotonic()
        if wait > 0:
            sleep(wait)
        self.env_at = monotonic()

    def _h_timed_post(self, payload: dict[str, Any]) -> Any:
        began = monotonic()
        result = _env.post_verbose('/environment/call', payload)
        self.env_spent += monotonic() - began
        self.env_count += 1
        return result

    def _h_took_step(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Book an accepted store call: step count, finish flag, the readable result, and which listings a page opened."""
        reported = data.get('step')
        self.step = reported if isinstance(reported, int) else self.step + 1
        if data.get('finished'):
            self.finished = True
        if name != self.roles.get('ask'):
            data['observation'] = self.unspelled(name, data.get('observation'))
        seen = data.get('observation')
        self.note_read(name, args, seen)
        if name == self.facts_tool() and isinstance(seen, dict) and not _failed(seen):
            ident = self.ident(seen)
            if ident and not _fact_block(seen):
                self.factless.add((name, ident))
        if name not in self.page_tools() or isinstance(seen, dict) and seen.get('error') or _failed(seen):
            return data
        shown = _dump(seen)
        for value in args.values():
            named = _h_named_ids(value, shown)
            self.opened.update(named)
            for ident in named:
                self.views.setdefault(ident, set()).add(name)
        return data

    def _h_after_refusal(self, result: Any, payload: dict[str, Any]) -> bool:
        """Trace a refused store call; True when it is worth sending again (payload's step already corrected)."""
        error = result.error or {}
        status = error.get('status')
        body = str(error.get('body') or '')
        self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
        if status != 409:
            if status == 402:
                self.broke = True
            return status not in _H_GIVE_UP
        low = body.lower()
        if any(mark in low for mark in _H_FINAL_MARKS):
            self.finished = True
            return False
        expected = re.search('expected (\\d+)', body)
        if expected:
            self.step = int(expected.group(1)) - 1
        elif any(mark in low for mark in _H_SKIP_MARKS):
            self.step += 1
        else:
            return False
        payload['step'] = self.step + 1
        return True

    def _h_unopened_keys(self) -> Any:
        """Keys of the rows only seen in searches (one at a time, so the set fills in the order they come)."""
        for ident, row in self.rows.items():
            if ident in self.opened:
                continue
            yield from row

    def _h_gather(self, record: Any, search_keys: set[str]) -> Any:
        if not isinstance(record, dict) or _fact_block(record) or not self.ident(record, True):
            return record
        extra = {}
        for key, value in record.items():
            if key not in search_keys and (isinstance(value, str) or _h_number(value)):
                extra[key] = value
        if len(extra) < RECORD_FIELDS:
            return record
        regrouped = {key: value for key, value in record.items() if key not in extra}
        regrouped['attributes'] = extra
        return regrouped

    def _h_brief_prompt(self) -> str:
        tool_text = '\n'.join(_h_tool_line(tool['function']) for tool in self.tools)
        prompt = ''.join([BRIEF, '\nTOOLS:\n', tool_text, '\n\nREQUEST:\n', self.request])
        opening = self.view.get('opening')
        if opening is None:
            return prompt
        return prompt + '\n\nOPENING NOTE FROM THE STORE:\n' + _dump(_clip(opening, 1500))

    def _h_ask_brief(self, prompt: str) -> dict[str, Any]:
        """The cheap model first; Sonnet when that brief lacks items or a checkout tool, or (as a second opinion) when it
        puts two items on one shelf; then retries spaced out while the brief is still empty."""
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit:
            if not _h_usable_brief(brief):
                brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
            elif self.shared_shelf(brief):
                second = self.ask_json(prompt, [HEAVY], 2500) or {}
                if _h_usable_brief(second):
                    brief = second
        while self._h_brief_retry_due(brief):
            sleep(BRIEF_GAP)
            brief = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        return brief

    def _h_brief_retry_due(self, brief: dict[str, Any]) -> bool:
        if brief or self.no_credit or self.calls >= ALL_CALLS:
            return False
        return self.clock() < BRIEF_BY

    def _h_split_modes(self, brief: dict[str, Any]) -> None:
        """One tool both checks out and refuses: ask which fixed arguments pick each mode, unless the brief already says."""
        buy = self.roles.get('buy')
        if not buy or buy != self.roles.get('refuse'):
            return
        buy_args, refuse_args = brief['buy_args'], brief['refuse_args']
        if buy_args and refuse_args and buy_args != refuse_args:
            return
        head = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: '
        fixed = self.ask_json(head + _dump(self.schemas.get(buy)) + _H_MODE_TAIL, [LIGHT, HEAVY], 1500) or {}
        checkout, decline = fixed.get('buy_args'), fixed.get('refuse_args')
        if isinstance(checkout, dict) and isinstance(decline, dict) and checkout != decline:
            brief['buy_args'], brief['refuse_args'] = checkout, decline

    @staticmethod
    def _h_item_of(entry: dict[str, Any]) -> Item:
        wanted = _count(entry.get('quantity'))
        units = wanted if wanted and wanted > 0 else 1
        needs = [str(must) for must in _items(entry.get('musts')) if must]
        return Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), units, needs)

    def _h_twin_of(self, item: Item, merged: list[Item]) -> Item | None:
        """An already kept item on the same shelf with the same quantity whose musts do not contradict this one's."""
        if not item.shelf.strip():
            return None
        shelf = _words(item.shelf)
        for kept in merged:
            if kept.qty == item.qty and _words(kept.shelf) == shelf and not _h_clash(kept.musts, item.musts):
                return kept
        return None

    def _h_merge_items(self) -> dict[int, int]:
        """Fold duplicate items into the first of their kind; returns old item number -> new item number."""
        merged: list[Item] = []
        renumbered: dict[int, int] = {}
        for number, item in enumerate(self.items, 1):
            twin = self._h_twin_of(item, merged)
            if twin is None:
                merged.append(item)
                renumbered[number] = len(merged)
                continue
            renumbered[number] = merged.index(twin) + 1
            twin.musts.extend([must for must in item.musts if must not in twin.musts])
            self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
        self.items = merged
        return renumbered

    def _h_bind_fits(self, brief: dict[str, Any], renumbered: dict[int, int]) -> None:
        total = len(self.items)
        for pair in _items(brief.get('fits')):
            if not isinstance(pair, dict):
                continue
            part = renumbered.get(_count(pair.get('item')) or 0)
            main = renumbered.get(_count(pair.get('with')) or 0)
            if not part or not main or part == main:
                continue
            if 1 <= part <= total and 1 <= main <= total and (part, main) not in self.bound:
                self.bound.append((part, main))

    def _h_lookup_ref(self, text: str) -> tuple[int, str]:
        """The item a lookup must belongs to, with the lookup replaced by that item's must on the same property when the
        brief's copy differs."""
        if not any(text in item.musts for item in self.items):
            key = _h_must_key(text)
            spelled = [must for item in self.items for must in item.musts if _h_must_key(must) == key]
            if spelled:
                text = spelled[0]
        for number, item in enumerate(self.items, 1):
            if text in item.musts:
                return number, text
        return 1, text

    def _h_read_terms(self, brief: dict[str, Any]) -> None:
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch(_H_PLAIN_BUDGET, budget):
            budget = float(budget)
        if _h_number(budget) and budget > 0:
            self.budget = budget
        limit = _count(brief.get('messages'))
        if limit is None or limit < 0:
            if self.roles.get('ask'):
                self.msg_limit = 3
        else:
            self.msg_limit = limit
        steps = _count(brief.get('steps'))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps

    def _h_read_process(self, brief: dict[str, Any]) -> None:
        for entry in _items(brief.get('process')):
            if not isinstance(entry, dict) or not str(entry.get('step') or '').strip():
                continue
            when = str(entry.get('when') or '').strip().lower()
            self.process.append({'step': str(entry['step']).strip(), 'when': when if when in _H_STAGES else 'order', 'messages': max(0, _count(entry.get('messages')) or 0)})
        if not self.roles.get('ask'):
            return
        spoken = [entry['messages'] for entry in self.process if entry['messages'] and entry['when'] in ('order', 'end')]
        self.reserve = min(max(spoken), self.msg_limit) if spoken else 0

    def _h_fact_groups(self, cat_key: str | None) -> dict[str, dict[str, Any]]:
        """Seen listings' attributes per category label: each key's count, distinct values, and per-listing texts."""
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            block = _fact_block(record) or {}
            label = 'listings'
            if cat_key:
                label = str(record.get(cat_key) or 'listings')
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            filed: set[str] = set()
            for key, value in block.items():
                name = _words(key)
                if name and name not in filed:
                    filed.add(name)
                    self._h_file_fact(group['keys'], name, ident, value)
        return groups

    @staticmethod
    def _h_file_fact(keys: dict[str, Any], name: str, ident: str, value: Any) -> None:
        slot = keys.setdefault(name, {'n': 0, 'values': [], 'per': {}, 'plain': {}})
        slot['n'] += 1
        raw = str(value).strip()
        text = raw.lower()
        slot['per'][ident] = text
        slot['plain'][ident] = _FIRST_SPELLING.get(raw, text).strip().lower()
        if text not in slot['values']:
            slot['values'].append(text)

    @staticmethod
    def _h_useless_key(values: list[str]) -> bool:
        """Values too long, mostly lists ('a / b'), or (four or more) all different, longish and seldom sharing a word."""
        mean = sum(len(text) for text in values) / len(values)
        if mean > 40:
            return True
        if 2 * len([text for text in values if ' / ' in text]) >= len(values):
            return True
        if len(values) < 4 or len(set(values)) != len(values) or mean <= 12:
            return False
        return _h_recurring(values) * 2 < len(values)

    def _h_group_keys(self, group: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """A group's useful keys, a key that repeats an earlier kept one folded into it, most listed first."""
        kept: list[tuple[str, dict[str, Any]]] = []
        for name, slot in sorted(group['keys'].items(), key=lambda pair: (-pair[1]['n'], len(pair[0]))):
            values = list(slot['plain'].values())
            if not values or self._h_useless_key(values):
                continue
            twin = None
            for other, spec in kept:
                if self.twins(other, spec, name, slot):
                    twin = spec
                    break
            if twin is None:
                kept.append((name, slot))
                continue
            for ident, text in slot['per'].items():
                twin['per'].setdefault(ident, text)
                twin['plain'].setdefault(ident, slot['plain'][ident])
                if text not in twin['values']:
                    twin['values'].append(text)
            twin['n'] = len(twin['per'])
        kept.sort(key=lambda pair: -pair[1]['n'])
        return kept

    def _h_listed(self, rows: list[Any]) -> list[tuple[str, dict[str, Any]]]:
        found = []
        for row in rows:
            if isinstance(row, dict):
                ident = self.ident(row, True)
                if ident:
                    found.append((ident, row))
        return found

    def _h_priced_arg(self, props: dict[str, Any], key: str) -> bool:
        spec = props.get(key)
        if not isinstance(spec, dict):
            spec = {}
        return bool(re.search(_H_MONEY_WORDS, f"{key} {spec.get('description') or ''}", re.I))

    @staticmethod
    def _h_mode_hint(args: dict[str, Any], props: dict[str, Any], tightest: str) -> str:
        """Point at the other values of the search's mode arguments, which may honour the cap."""
        others = {}
        for key, value in args.items():
            if isinstance(value, str) and isinstance((props.get(key) or {}).get('enum'), list):
                others[key] = [option for option in props[key]['enum'] if option != value]
        if not any(others.values()):
            return ''
        modes = ', '.join(f'{key}={_dump(options[0])}' for key, options in others.items() if options)
        return f'; list the category with {modes} and the same {tightest} to see only cheaper listings'

    def _j_compare_reads(self, queue: list[str], keep: int, texts: list[str]) -> tuple[list[str], list[str]]:
        """Read queued listings in batches through the compare tool, adding each row's text to `texts`. Returns the
        batch members still without a full page, and what is left of the queue (a batch the tool did not answer goes
        back to its front)."""
        side = self.roles.get('compare')
        many = self.compare_target()
        unread: list[str] = []
        while self._j_may_batch(side, many, queue, keep):
            size = self.compare_room()
            batch, rest = queue[:size], queue[size:]
            rows = self._j_batch_rows(side, many, batch)
            if rows is None:
                return unread, batch + rest
            texts.extend(self._j_excerpt(ident, record) for ident, record in rows)
            if not rows:
                self.compare_bad = min(self.compare_bad or 999, len(batch))
                queue = batch + rest
                if self.compare_room() >= len(batch):
                    break
                continue
            self.compare_ok = max(self.compare_ok, len(rows))
            unread.extend(i for i in batch if i not in self.full)
            queue = rest
        return unread, queue

    def _j_may_batch(self, side: str | None, many: str | None, queue: list[str], keep: int) -> bool:
        """Whether another compare batch is worth a step."""
        if not (side and many) or self.compare_facts() is not True:
            return False
        if len(queue) < 2 or self.left() <= keep:
            return False
        return not (self.finished or self.rushed())

    def _j_batch_rows(self, side: str, many: str, batch: list[str]) -> list[Any] | None:
        """One compare call for a batch: its rows, or None when the call could not be made or failed."""
        reply = self._j_send(side, {many: batch})
        if reply is None:
            return None
        return self.records_in(self._j_viewed(side, reply))

    def _j_direct_reads(self, tool: str, target: str, queue: list[str], keep: int, texts: list[str]) -> None:
        """Open queued listings one page at a time, adding each page's text to `texts`."""
        for ident in queue:
            if self.facts_from.get(tool) is False or self._j_out_of_room(keep):
                return
            if (tool, ident) in self.factless:
                continue
            reply = self._j_send(tool, {target: ident})
            if reply is None:
                return
            page = self._j_viewed(tool, reply)
            if isinstance(page, dict) and self.ident(page):
                self.note_facts(tool, bool(_fact_block(page)))
            texts.append(self._j_excerpt(ident, page))

    def _j_excerpt(self, ident: str, value: Any) -> str:
        """A listing's text as the model reads it: its id, then the first 1500 characters of the shown value."""
        return f'{ident}: ' + self.show(value)[:1500]

    def _j_viewed(self, tool: str, reply: dict[str, Any]) -> Any:
        """Record a page call's observation as seen and viewed; the observation."""
        seen = reply.get('observation')
        self.observe(tool, seen)
        self.note_view(tool, seen)
        return seen

    def _j_fit_args(self, tool: str, wanted: dict[str, Any]) -> dict[str, Any] | None:
        """The arguments fitted to the tool's schema, or None when they break it."""
        problems: list[str] = []
        fitted = _shape(wanted, self.schemas.get(tool), 'args', problems)
        return None if problems else fitted

    def _j_send(self, tool: str, wanted: dict[str, Any]) -> dict[str, Any] | None:
        """Call the tool with the arguments fitted to its schema; None when they break it or the call fails."""
        fitted = self._j_fit_args(tool, wanted)
        if fitted is None:
            return None
        return self.act(tool, fitted)

    def _j_out_of_room(self, keep: int) -> bool:
        """No step to spare above `keep`, the episode is over, or time presses."""
        return self.left() <= keep or self.finished or self.rushed()

    def _j_link_code(self, ident: str) -> str | None:
        """The code the line's page refers to, when it is new and no known listing carries it as its own."""
        code = (self.fit_index()['marks'].get(ident) or {}).get('more')
        if not code or code in self.link_tried or self._j_owned_by(code):
            return None
        return code

    def _j_owned_by(self, code: str) -> list[str]:
        """The listings whose own code is `code`."""
        return [ident for ident, mark in self.fit_index()['marks'].items() if mark.get('own') == code]

    def _j_cap_and_shelf(self, tool: str, schema: Any) -> tuple[list[str], list[str]]:
        """A tool's price-cap arguments and its free-text shelf arguments."""
        caps = self.cap_keys(tool)
        props = (schema or {}).get('properties') or {}
        shelves = [key for key, spec in props.items()
                   if isinstance(spec, dict) and 'string' in _kinds(spec) and 'enum' not in spec and re.search(SHELF_ARG, _snake(key), re.I)]
        return caps, shelves

    def _j_mode_pick(self, key: str, options: list[Any], props: dict[str, Any], shelf: str,
                     learned: list[tuple[dict[str, Any], bool]], single: bool) -> Any:
        """The value of one option argument for a capped shelf listing, or None when none is clear."""
        pick = next((mode[key] for mode, held in learned if held and key in mode), None)
        if pick is None:
            pick = self._j_shelf_mode(key, props, shelf)
            fallback = self._j_mode_fallback(key, options, single)
            if pick is None:
                pick = fallback
        if not single:
            return pick
        spoiled = {mode[key] for mode, held in learned if not held and key in mode}
        if not spoiled or (pick is not None and pick not in spoiled):
            return pick
        rest = [option for option in options if option not in spoiled]
        return rest[0] if len(rest) == 1 else None

    def _j_shelf_mode(self, key: str, props: dict[str, Any], shelf: str) -> Any:
        """The option value of the latest listing by shelf that carried no search words of its own."""
        def plain(call: dict[str, Any]) -> bool:
            enum_keys = [k for k in _j_query_keys(call) if k != shelf and not (isinstance(props.get(k), dict) and 'enum' in props[k])]
            return not enum_keys
        return next((call[key] for call in reversed(self.searched) if shelf in call and key in call and plain(call)), None)

    def _j_mode_fallback(self, key: str, options: list[Any], single: bool) -> Any:
        """Without a learned mode: the one option naming a listing, else (for a lone required option) the one option
        no worded search has used."""
        named = [option for option in options if isinstance(option, str) and re.search(LIST_MODE, option, re.I)]
        worded = {call[key] for call in self.searched if key in call and _j_query_keys(call)}
        unworded = [option for option in options if option not in worded]
        if len(named) == 1:
            return named[0]
        if single and worded and len(unworded) == 1:
            return unworded[0]
        return None

    def _j_label_prices(self, cat_key: str | None) -> dict[str, list[float]]:
        """The known prices of the rows, by their lower-cased label (all under '' without a label key)."""
        pools: dict[str, list[float]] = {}
        for ident, row in self.rows.items():
            price = self.price_of(ident)
            if price is None:
                continue
            label = str(row.get(cat_key) or '').strip().lower() if cat_key else ''
            pools.setdefault(label, []).append(price)
        return pools

    def _j_kind(self, ident: str, cat_key: str | None) -> str:
        """A row's label, stripped and lower-cased."""
        return str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower()

    def _j_walk_line(self, plan: tuple[str, dict[str, Any], str, str], line: str, price: Any, label: str) -> bool:
        """Capped listings of one cart line's shelf, each cap below the lowest price the last one showed; True when the
        whole pass must stop (steps, time, or a call that could not be made)."""
        tool, base, shelf, cap = plan
        limit = _j_below(price)
        tried: set[int] = set()
        for _ in range(WALK_CALLS):
            margin = WALK_ROOM if tried else _J_FIRST_WALK_ROOM
            if self.left() <= self.tail() + margin or self.finished or self.rushed():
                return True
            if limit < 1 or limit in tried:
                return False
            tried.add(limit)
            args = self._j_fit_args(tool, {**base, shelf: label, cap: limit})
            if args is None:
                return True
            reply = self.act(tool, args)
            if reply is None:
                return True
            seen = reply.get('observation')
            self.observe(tool, seen)
            rows = self.records_in(seen)
            self.trace.append({'kind': 'list_below', 'line': line, 'price': price, 'limit': limit, 'rows': len(rows)})
            page = self.paging(tool, seen)
            if self._j_walk_done(rows, limit, page, seen):
                return False
            if page[0] <= 2 * page[1]:
                self._j_closing_page(tool, args, page)
                return False
            known = [p for p in (self.price_of(ident) for ident, _ in rows) if p is not None]
            if not known:
                return False
            limit = _j_below(min(known))
        return False

    def _j_walk_done(self, rows: list[Any], limit: int, page: Any, seen: Any) -> bool:
        """Whether a capped listing ends the walk: no rows, a row above the cap, no further page, or price order."""
        if not rows or any((self.price_of(ident) or 0) > limit for ident, _ in rows):
            return True
        if not page or page[0] <= page[1]:
            return True
        return self.price_sorted(seen)

    def _j_closing_page(self, tool: str, args: dict[str, Any], page: Any) -> None:
        """Read the second page of a capped listing that two pages cover, when the steps allow it."""
        if page[3] is None:
            return
        call = self._j_fit_args(tool, {**args, page[2]: page[3]})
        if call is None or self.left() <= self.tail() + _J_FIRST_WALK_ROOM or self.rushed():
            return
        more = self.act(tool, call)
        if more is not None:
            self.observe(tool, more.get('observation'))

    def _j_cheaper_than(self, line: str, cart: list[str], cat_key: str | None) -> list[str]:
        """Buyable unread rows of a cart line's label priced below it, cheapest first (none when the line has no
        price or no label)."""
        price = self.price_of(line)
        kind = self._j_kind(line, cat_key) if cat_key else ''
        if price is None or not kind:
            return []
        endless = float('inf')
        pool = [ident for ident in self.seen_order
                if ident not in cart and ident not in self.full and self.buyable(ident) and self._j_kind(ident, cat_key) == kind
                and (self.price_of(ident) if self.price_of(ident) is not None else endless) < price]
        pool.sort(key=lambda ident: self.price_of(ident) or 0.0)
        return pool

    def _j_item_noun(self, number: int, item: Any, labels: list[str]) -> str:
        """The longest word of the item's most common label (else its shelf or name) that no other item uses; empty
        once a search by the noun alone has found nothing."""
        foreign: set[str] = set()
        for other_number, other in enumerate(self.items, 1):
            if other_number != number:
                foreign.update(_j_words4(((other.shelf or '') + ' ' + other.name).lower()))
        if labels:
            tally: dict[str, int] = {}
            for label in labels:
                tally[label] = tally.get(label, 0) + 1
            usual = max(tally, key=tally.get).lower()
        else:
            usual = (item.shelf or item.name).lower()
        nouns = [word for word in _j_words4(usual) if word not in foreign]
        if number in self.hunt_no_noun or not nouns:
            return ''
        return max(nouns, key=len)

    def _j_split_words(self, item: Any, names: list[str], noun: str) -> list[str]:
        """Words that appear in at least two of the names and at most HUNT_COMMON of them, not the noun and not
        sharing a root with the item's own words; commonest first, then alphabetical."""
        counts: dict[str, int] = {}
        for name in names:
            for word in {w.lower() for w in re.findall("[A-Za-z][A-Za-z'-]{2,}", name)}:
                counts[word] = counts.get(word, 0) + 1
        own = _roots(item.shelf or '') | _roots(item.name)
        ceiling = max(2, int(len(names) * HUNT_COMMON))
        kept = []
        for word, seen in counts.items():
            if seen < 2 or seen > ceiling or word == noun:
                continue
            roots = _roots(word)
            if roots and not roots & own:
                kept.append(word)
        return sorted(kept, key=lambda word: (-counts[word], word))

    def _j_hunt_all(self, plan: tuple[str, dict[str, Any], str], room: int) -> tuple[int, int]:
        """Hunt each item in turn within its share of `room`; the steps spent and the rows gained."""
        tool, base, field = plan
        count = len(self.items)
        share = max(3, room // max(1, count))
        spent = fresh = 0
        for number in range(1, count + 1):
            if spent >= room or self.finished:
                break
            used, gained = self.hunt_item(number, tool, base, field, min(share, room - spent))
            spent += used
            fresh += gained
        return spent, fresh

    def _j_find_stem(self, hunt: dict[str, Any], noun: str) -> str | None:
        """The first stem search that shows rows ('' when none does), or None when a search could not be sent."""
        for query in _j_stem_tries(hunt['values'], noun):
            hunt['issued'] = False
            if self._j_hunt_query(query, hunt):
                return query
            if not hunt['issued']:
                return None
            if query == noun:
                self.hunt_no_noun.add(hunt['number'])
                noun = ''
        return ''

    def _j_may_widen(self, hunt: dict[str, Any]) -> bool:
        """Whether one more narrowed search fits: room left, recent searches still found rows, none refused, time left."""
        if hunt['spent'] >= hunt['room'] or hunt['idle'] >= HUNT_IDLE or hunt['refused']:
            return False
        return not self.finished and not self._j_hunt_late()

    def _j_hunt_late(self) -> bool:
        """Past the hunting time, grace included."""
        return self.clock() > HUNT_BY + HUNT_GRACE

    def _j_hunt_query(self, query: str, hunt: dict[str, Any]) -> bool:
        """Run one search to its last page or the end of the room, never paying again for a page this episode already
        read; whether any page showed a row."""
        hunt['tried'].add(query.lower())
        tool = hunt['tool']
        args = self._j_fit_args(tool, {**hunt['base'], hunt['field']: query})
        if args is None:
            hunt['refused'] = True
            return False
        if hunt['spent'] >= hunt['room'] or self.finished:
            return False
        hunt['issued'] = True
        values = hunt['values']
        whole = bool(values) and all(v.lower() in query.lower() for v in values)
        found = False
        walked: set[str] = set()
        while not self.finished and not self._j_hunt_late():
            mark = self.call_mark(tool, args)
            if mark in walked:
                return found
            walked.add(mark)
            page_read = self._j_hunt_page(mark, args, hunt)
            if page_read is None:
                return found
            seen, new = page_read
            rows = self.records_in(seen)
            found = found or bool(rows)
            if whole:
                self.value_rows.update(ident for ident, _ in rows)
            page = self.paging(tool, seen)
            if not page or page[3] is None or (new < HUNT_PAGE_NEW and page[0] > HUNT_WHOLE):
                return found
            args = self._j_fit_args(tool, {**args, page[2]: page[3]})
            if args is None:
                return found
        return found

    def _j_hunt_page(self, mark: str, args: dict[str, Any], hunt: dict[str, Any]) -> tuple[Any, int] | None:
        """One page of a hunting search: the stored one when already read (counted as new enough to go on), else a
        paid call, with the item's rows it added. None when the room is used up or the call failed."""
        if mark in self.read_calls:
            return self.read_calls[mark], HUNT_PAGE_NEW
        if hunt['spent'] >= hunt['room']:
            return None
        number = hunt['number']
        before = self.rows_of_item(number)
        reply = self.act(hunt['tool'], args)
        hunt['spent'] += 1
        if reply is None:
            return None
        seen = reply.get('observation')
        self.observe(hunt['tool'], seen)
        new = self.rows_of_item(number) - before
        hunt['gained'] += new
        hunt['idle'] = 0 if new else hunt['idle'] + 1
        return seen, new

    def _j_open_items(self, cat_key: str | None) -> list[int]:
        """The items no cart line with units covers (all items when every one is covered)."""
        held = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in self.cart_ids() if (self.held_units(i) or 0) > 0] if cat_key else []
        open_numbers = [number for number, item in enumerate(self.items, 1) if not any(self.owns(item, label) for label in held)]
        return open_numbers or list(range(1, len(self.items) + 1))

    def _j_fit_queries(self, number: int) -> list[str]:
        """The searches for one item's fit, without repeats: series with the first value and noun, each value with the
        noun, each value alone (when there is a noun), the first further words with the noun, and the noun."""
        pairs = self.value_words(number)
        noun, words = self.hunt_words(number)
        values = [value for value, _ in pairs]
        leads = values[:1] or ['']
        tries = [f'{family} {value} {noun}'.strip() for family in self.series_for(number) for value in leads]
        tries.extend(f'{value} {noun}'.strip() for value in values)
        if noun:
            tries.extend(values)
        tries.extend(f'{word} {noun}'.strip() for word in words[:HUNT_FIT_WORDS])
        if noun:
            tries.append(noun)
        return list(dict.fromkeys(tries))

    def _j_fit_searches(self, numbers: list[int], plan: tuple[str, dict[str, Any], str], room: int) -> int | None:
        """Each open item's fit searches within its share of `room`; the steps spent, or None when a call failed."""
        tool, base, field = plan
        share = max(2, room // max(1, len(numbers)))
        spent = 0
        for number in numbers:
            used = 0
            for query in self._j_fit_queries(number):
                if spent >= room or used >= share or self.finished or self._j_hunt_late():
                    break
                args = self._j_fit_args(tool, {**base, field: query})
                if args is None:
                    continue
                reply = self.act(tool, args)
                spent += 1
                used += 1
                if reply is None:
                    return None
                self.observe(tool, reply.get('observation'))
        return spent

    def _j_fit_pool(self, number: int, wanted: dict[int, set[str]], cat_key: str | None) -> list[str]:
        """The item's unread buyable rows within the budget, those sharing most required words first, then cheapest."""
        item = self.items[number - 1]
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        pool = [ident for ident in self.seen_order
                if ident not in self.full and self.buyable(ident)
                and (not cat_key or self.owns(item, str((self.rows.get(ident) or {}).get(cat_key) or '')))
                and (cap is None or (self.price_of(ident) or 0) <= cap)]
        words = wanted.get(number) or set()
        endless = float('inf')
        pool.sort(key=lambda ident: (-len(words & _text_words(_dump(self.rows.get(ident) or {}))),
                                     self.price_of(ident) if self.price_of(ident) is not None else endless))
        return pool

    def _k_props(self, tool: Any) -> dict[str, Any]:
        """The argument properties the schema of `tool` declares."""
        return (self.schemas.get(tool) or {}).get('properties') or {}

    def _k_shaped(self, tool: Any, args: dict[str, Any]) -> Any:
        """`args` fitted to the schema of `tool`; None when the schema rejects them."""
        faults: list[str] = []
        fitted = _shape(args, self.schemas.get(tool), 'args', faults)
        return None if faults else fitted

    def _k_step(self, tool: str, call: dict[str, Any]) -> dict[str, Any] | None:
        """One store call, with what it showed taken in; the reply, None when no call could be made."""
        reply = self.act(tool, call)
        if reply is not None:
            self.observe(tool, reply.get('observation'))
        return reply

    def _k_page_after(self, tool: str, call: dict[str, Any], seen: Any) -> Any:
        """The same call moved on to the page after the one `seen` shows; None when there is none or it does not fit."""
        page = self.paging(tool, seen)
        if not page or page[3] is None:
            return None
        return self._k_shaped(tool, {**call, page[2]: page[3]})

    def _k_may_page(self, mark: str, args: dict[str, Any], words: set[str], number: int, cat_key: Any) -> bool:
        """Whether a kept search shares a word with the item's values and shows rows this item may own."""
        query = ' '.join(value for value in args.values() if isinstance(value, str)).lower()
        if not words & _text_words(query):
            return False
        if not cat_key:
            return True
        owners = {self.owner_of(str(record.get(cat_key) or '')) for _, record in self.records_in(self.read_calls[mark])}
        return not owners or 0 in owners or number in owners

    def _k_walk_on(self, tool: str, mark: str, args: dict[str, Any], used: int, stop: int) -> tuple[int, bool]:
        """Follow a kept search page by page while fewer than `stop` steps are used, reusing pages already read;
        the steps used so far and whether the store stopped answering."""
        seen = self.read_calls[mark]
        call: Any = args
        visited = {mark}
        while used < stop and not self.finished:
            call = self._k_page_after(tool, call, seen)
            if call is None:
                break
            spelled = self.call_mark(tool, call)
            if spelled in visited:
                break
            visited.add(spelled)
            if spelled in self.read_calls:
                seen = self.read_calls[spelled]
                continue
            reply = self._k_step(tool, call)
            used += 1
            if reply is None:
                return used, True
            seen = reply.get('observation')
        return used, False

    def _k_at_start(self, name: str, key: str, value: Any, props: dict[str, Any]) -> bool:
        """A whole number at its schema minimum or default, or 1 on the tool's known page argument without either."""
        if not _k_whole(value):
            return False
        spec = props.get(key)
        if not isinstance(spec, dict):
            spec = {}
        least, default = spec.get('minimum'), spec.get('default')
        if value in (least, default):
            return True
        return value == 1 and key == self.page_args.get(name) and least is None and default is None

    def _k_short_last_page(self, name: str, observation: dict[str, Any]) -> tuple[Any, int] | None:
        """For a reply with no next page that counts more rows than it lists: its rows field and how many it lists."""
        shape = self.read_shape(name, observation)
        if not shape or not shape.get('count') or shape.get('next') is None:
            return None
        if observation.get(shape['next']) is not None:
            return None
        rows, count = observation.get(shape['rows']), observation.get(shape['count'])
        if isinstance(rows, list) and isinstance(count, int) and count > len(rows):
            return shape['rows'], len(rows)
        return None

    def _k_missed_words(self, plan: Any, name: str, args: dict[str, Any], observation: Any) -> str | None:
        """The text of a words search of three or more words whose reply held nothing at all, when there are steps to retry it."""
        if not plan or plan[0] != name:
            return None
        if self.left() <= self.tail() + SHORT_KEEP or self.finished or not isinstance(observation, dict):
            return None
        text = args.get(plan[2])
        if not isinstance(text, str) or self.records_in(observation) or len(text.split()) < 3:
            return None
        if any(_k_has_content(value) for value in observation.values()):
            return None
        return text

    def _k_opening_queries(self, item: Any) -> list[str]:
        """An item's opening searches: up to two short required values with its noun, then the noun alone."""
        values: list[str] = []
        for need in item.musts:
            if ':' not in need or need in [ref for _, ref in self.refs]:
                continue
            value = need.split(':', 1)[1].strip()
            if value and len(value.split()) <= 3 and not re.search(OPEN_NOT, value, re.I):
                values.append(value)
        name_words = item.name.split()
        noun = item.name if len(name_words) <= 3 else ' '.join(name_words[-2:])
        queries = [' '.join(values[:2] + [noun])] if values else []
        return list(dict.fromkeys(queries + [noun]))

    def _k_open_item(self, plan: tuple[str, dict[str, Any], str], number: int, item: Any, stop_at: int,
                     shown: list[str]) -> bool:
        """Run one item's opening searches, adding what each showed to `shown`; False when the store stopped answering."""
        tool, base, field = plan
        for query in self._k_opening_queries(item):
            args = self._k_shaped(tool, {**base, field: query})
            if args is None or self.finished or self.step >= stop_at:
                break
            reply = self._k_step(tool, args)
            if reply is None:
                return False
            seen = reply.get('observation')
            if not self.records_in(seen):
                verdict = 'the store refused the call: ' + self.show(seen) if _failed(seen) else 'no rows'
                shown.append(f'{query}: {verdict}')
                continue
            self.searched = [*self.searched, dict(args)][-8:]
            shown.append(f'{query}: ' + self.show(seen))
            shown += self.more_rows_for_keys(tool, args, seen, stop_at, number)
            if self.rows_of_item(number) >= OPEN_EACH:
                break
        return True

    def _k_open_more(self, stop_at: int) -> None:
        """Read more listings in full, round by round, until every item with rows has OPEN_EACH of them read, the opening's
        share of steps is used, or a round reads nothing new."""
        until = self.step + self.max_steps // OPEN_SHARE
        found = [n for n in range(1, len(self.items) + 1) if self.rows_of_item(n)]
        while self.step < until and self.step < stop_at and not self.finished:
            if all(self.records_of_item(n) >= OPEN_EACH for n in found):
                break
            had = len(self.full)
            self.widen()
            if len(self.full) == had:
                break

    def _k_batch_compare(self, tool: Any, schema: dict[str, Any], target: Any) -> bool:
        """Whether the compare tool needs nothing but its list of ids and has not shown itself to be bare."""
        if not tool or not target:
            return False
        if any(key != target for key in schema.get('required') or []):
            return False
        return not self.compare_bare()

    def _k_round_robin(self, cap: int) -> list[str]:
        """Up to `cap` unread ids, one from each item in turn, items with the fewest listings read first."""
        counted = [(sum(1 for i in ids if i in self.full), [i for i in ids if i not in self.full]) for ids in self.item_groups()]
        counted.sort(key=lambda pair: pair[0])
        lanes = [lane for _, lane in counted]
        picks = [lane[turn] for turn in range(cap) for lane in lanes if turn < len(lane)]
        return picks[:cap]

    def _k_take_compare(self, tool: str, seen: Any, picks: list[str], before: int) -> Any:
        """Take in a compare reply and learn what the tool shows; its rows."""
        self.observe(tool, seen)
        self.note_view(tool, seen)
        rows = self.records_in(seen)
        if not rows:
            self.compare_bad = min(self.compare_bad or 999, len(picks))
        else:
            self.facts_from[tool] = self.facts_from.get(tool, False) or any(_fact_block(record) for _, record in rows)
            self.compare_ok = max(self.compare_ok, len(rows))
        self.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(self.full) - before})
        return rows

    def _k_compare_pair(self, tool: str, target: str, picks: list[str]) -> None:
        """Compare only the first two picks, after a longer list read nothing new."""
        args = self._k_shaped(tool, {target: picks[:2]})
        if args is None:
            return
        reply = self._k_step(tool, args)
        if reply is not None:
            self.compare_ok = max(self.compare_ok, len(self.records_in(reply.get('observation'))))

    def _k_message_slot(self, props: dict[str, Any], required: list[Any], text: str) -> dict[str, Any]:
        """Where the message goes: the most message-like plain text field (required ones first), else the first list
        argument or object argument with a text field; {} when none."""
        plain = [key for key, spec in props.items()
                 if isinstance(spec, dict) and 'string' in _kinds(spec) and 'enum' not in spec and 'const' not in spec]
        pool = [key for key in required if key in plain] or plain
        if pool:
            return {max(pool, key=lambda key: _k_message_rank(key, props[key])): text}
        for key, spec in props.items():
            holder = _k_text_holder(spec)
            if isinstance(spec, dict) and 'array' in _kinds(spec):
                return {key: [text]}
            if holder:
                return {key: {holder: text}}
        return {}

    def _k_fresh_lanes(self, width: int) -> None:
        """Plan the question lanes and clear their tallies."""
        self.lanes = self.plan_lanes(width)
        self.lane_found, self.lane_spent = dict.fromkeys(self.lanes, 0), dict.fromkeys(self.lanes, 0)
        self.lane_used = {n: set() for n in self.lanes}

    def _k_pointed(self, told: list[str], ending: str) -> str:
        """The note for new requirements whose value sits on another listing; '' when none does."""
        pointed = [need for _, need in self.refs if any(need in line for line in told)]
        if not pointed:
            return ''
        return _K_POINTED_HEAD + ' | '.join(pointed) + _K_POINTED_STEPS + ending

    def _k_may_ask_more(self) -> bool:
        return self.msgs_free() > 0 and not self.finished and self.left() > 6 and self.clock() < QUESTION_BY

    def _k_next_batch(self, groups: list[Any], told: list[str]) -> Any:
        """The next questions sent: from the planned groups while a listing task has some, else the next lane."""
        while groups and self.listing:
            batch = self.ask_group(groups.pop(0), told)
            if batch:
                return batch
        return self.ask_next(told)

    def _k_interview_summary(self, sent: list[tuple[int, str, str]], told: list[str], chain: str, searched: str) -> str:
        parts = ["K2's interviewer asked the shopper (each question used one step):\n"
                 + '\n'.join(f'Q: {question}\nA: {answer}' for _, question, answer in sent)]
        parts.append('\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none'))
        parts.append('. Apply them to every pick' + _K_SEARCH_AGAIN + '.')
        parts.append(self._k_pointed(told, ', and the order counts only after all of them are opened.'))
        parts += ['\n' + extra for extra in (chain, searched) if extra]
        if self.queued:
            parts.append('\n[notes seen meanwhile] ' + ' | '.join(self.queued))
            self.queued = []
        return ''.join(parts)

    def _k_late_allowed(self) -> bool:
        """A late question needs a free message no order step keeps, steps, time and a model to judge the answer."""
        if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit:
            return False
        if any(p['when'] == 'order' and p['messages'] > 0 for p in self.process):
            return False
        return self.left() >= LATE_STEPS and self.clock() <= LATE_BY and bool(self.judges())

    def _k_told_template(self, tool: str) -> tuple[str, dict[str, Any], str] | None:
        """The words search the brief describes (tool, text field, fixed arguments), when its field takes free text and
        the fixed arguments fit; else None."""
        hint = self.brief.get('search_words')
        if not isinstance(hint, dict):
            hint = {}
        field = hint.get('field') if isinstance(hint.get('field'), str) else None
        fixed = hint.get('args') if isinstance(hint.get('args'), dict) else {}
        named_tool = hint.get('tool')
        if not isinstance(named_tool, str) or named_tool not in self.schemas:
            named_tool = tool
        spec = self._k_props(named_tool)
        if field not in spec or 'string' not in _kinds(spec[field]) or 'enum' in spec[field]:
            return None
        base = self._k_shaped(named_tool, {**fixed, field: 'x'})
        if base is None:
            return None
        base.pop(field, None)
        return (named_tool, base, field)

    def _k_value_tries(self, pairs: list[tuple[str, bool]], item: Any) -> list[list[str]]:
        """The word lists to try, at most two: the answered short values first then the rest, then the same without the
        last value the request itself stated; a single answered value is tried with the item's name, then alone."""
        answered = [value for value, told in pairs if told]
        lead = [value for value in answered if len(value.split()) == 1 or re.fullmatch(AMOUNT, value, re.I)]
        ordered = lead + [value for value, _ in pairs if value not in lead]
        if len(ordered) == 1:
            return [[ordered[0], item.name], ordered] if lead else [ordered]
        stated = [place for place, value in enumerate(ordered) if value not in answered]
        drop = stated[-1] if stated else 0
        return [ordered, ordered[:drop] + ordered[drop + 1:]]

    def _k_search_values(self, plan: tuple[str, dict[str, Any], str], number: int, item: Any,
                         pairs: list[tuple[str, bool]], room: int, notes: list[str]) -> int:
        """Run an item's value searches until one shows rows of its kind, then read up to three of them; the room left."""
        tool, base, field = plan
        for words in self._k_value_tries(pairs, item):
            if not words:
                continue
            args = self._k_shaped(tool, {**base, field: ' '.join(words)})
            if args is None or self.finished:
                break
            reply = self._k_step(tool, args)
            room -= 1
            if reply is None:
                break
            rows = self._k_kind_rows(number, item, reply.get('observation'))
            notes.append(f'{_dump(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
            if not rows:
                continue
            unread = [ident for ident in rows if ident not in self.full]
            opened = self.fill_facts(unread[:min(3, room)], self.tail() + VALUE_KEEP)
            room -= len(opened)
            if opened:
                notes.append('opened: ' + ' | '.join(opened)[:2400])
            break
        return room

    def _k_label(self, ident: str, cat_key: Any) -> str:
        """The category label a search row carries."""
        return str((self.rows.get(ident) or {}).get(cat_key) or '')

    def _k_price_cap(self, found: list[str]) -> Any:
        """The budget as a price limit, unless every row found costs over ten times it (another currency or unit)."""
        cap = self.budget
        if not isinstance(cap, (int, float)):
            return None
        prices = [self.price_of(ident) for ident in found]
        if prices and all(price is not None and price > 10 * cap for price in prices):
            return None
        return cap

    def _k_kind_rows(self, number: int, item: Any, observation: Any) -> list[str]:
        """The buyable rows a reply showed of the item's kind within the budget, the item's own category first."""
        cat_key = self.category_key()
        found = [ident for ident, _ in self.records_in(observation)]
        cap = self._k_price_cap(found)
        rows = [ident for ident in found if self.buyable(ident)
                and (not cat_key or self.owns(item, self._k_label(ident, cat_key)))
                and (cap is None or (self.price_of(ident) or 0) <= cap)]
        home = self.home_label(number)
        if cat_key and home:
            rows.sort(key=lambda ident: _words(self._k_label(ident, cat_key)) != home)
        return rows

    def _m_open_labels(self, number: int) -> list[str]:
        """The item's planned labels not yet asked and not yet answered."""
        lane = self.lanes.get(number) or []
        if not lane:
            return []
        asked = self.lane_used.setdefault(number, set())
        return [label for label in lane if _words(label) not in asked and not self.told_about(number, label)]

    def _m_group_share(self, size: int) -> int:
        """Labels per item in a message covering `size` items."""
        cap = self.list_cap()
        now = self.list_now()
        if now >= cap:
            return max(MIX_LABELS, cap // size)
        return max(1, now // size)

    def _m_fits(self, text: str) -> bool:
        """Whether the message tool's arguments carry this question whole."""
        args = self.message_args(text)
        return bool(args) and _says(args, text)

    def _m_mixed_text(self, parts: list[tuple[int, list[str]]], size: int) -> str:
        """One question over several items, each naming its first `size` labels."""
        clauses = []
        for number, labels in parts:
            head = labels[:size]
            named = _m_either(head) if len(head) > 1 else labels[0]
            clauses.append(f'what {named} for the {self.item_said(number)}')
        joined = ', and '.join(clauses)
        return 'W' + joined[1:].replace(' for the ', ' would you like for the ', 1) + '?'

    def _m_fit_mixed(self, parts: list[tuple[int, list[str]]], share: int) -> tuple[list[tuple[int, list[str]]], str] | None:
        """The widest mixed question the message tool takes whole: fewer labels per item first, then fewer items
        (dropping from the end); None when even one label of the first item does not fit."""
        while parts:
            for size in range(share, 0, -1):
                text = self._m_mixed_text(parts, size)
                if self._m_fits(text):
                    return [(number, labels[:size]) for number, labels in parts], text
            parts = parts[:-1]
        return None

    def _m_send_one(self, number: int, labels: list[str], shown: bool, text: str, floor: int, told: list[str]) -> list[tuple[int, str, str]]:
        """Send a question about one item and, when it went out, read and settle its answer."""
        batch = self.send([(number, labels[0], text)], floor)
        if batch:
            self.finish_ask(number, labels, shown, batch, told)
        return batch

    def _m_read_reply(self, sent: list[tuple[int, str, str]], drop_free: bool, told: list[str]) -> tuple[list[list[str]], dict[str, Any]]:
        """Have the answer read, take its needs, and return each item's new needs (in item order) with the reading."""
        before = [len(item.told) for item in self.items]
        heard = self.hear(sent, (LIGHT, HEAVY))
        if drop_free:
            heard.pop('free', None)
        self.take_heard(heard, told)
        return [item.told[size:] for item, size in zip(self.items, before)], heard

    def _m_lists_off(self, number: int) -> None:
        self.listing = False
        self.trace.append({'kind': 'lists_off', 'item': number})

    def _m_key_words(self, number: int) -> dict[str, set[str]]:
        """For each key on the item's opened listings, the value words seen under it."""
        words: dict[str, set[str]] = {}
        for ident in self.item_ids(number):
            block = _fact_block(self.full.get(ident) or {}) or {}
            for key, value in block.items():
                words.setdefault(_words(key), set()).update(_roots(str(value)))
        return words

    def _m_tail(self, number: int) -> str:
        return f' would you like for the {self.item_said(number)}?'

    def _m_take_next(self) -> tuple[int, list[str], bool]:
        """The next question to try: a pending follow-up still unanswered, else the next planned labels."""
        self.follow_up = [f for f in self.follow_up if not self.told_about(f[0], f[1][0])]
        if not self.follow_up:
            number, labels = self.next_ask(self.lanes, self.lane_found, self.lane_spent, self.lane_used)
            return number, labels, False
        return self.follow_up.pop(0)

    def _m_fitted_ask(self) -> tuple[int, list[str], bool, str] | None:
        """The first candidate question that fits the message tool, setting aside the labels that never fit."""
        for _ in range(LIST_SIZE):
            number, labels, shown = self._m_take_next()
            if not labels:
                return None
            fitted, text = self.fit_ask(number, labels, shown)
            if shown and not fitted:
                fitted, text = self.fit_ask(number, labels, False)
            if fitted:
                return number, fitted, shown, text
            self.lane_used.setdefault(number, set()).add(_words(labels[0]))
            self.trace.append({'kind': 'ask_too_long', 'item': number, 'label': labels[0]})
        return None

    def _m_settle_single(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool, used: set[str]) -> None:
        """A question naming one label: the label counts as asked; a reply asking for a narrower question queues the
        follow-up (one product of a joined item name, or the label set apart from look-alike keys)."""
        lead = labels[0]
        used.add(_words(lead))
        if crowded and self.listing:
            self._m_lists_off(number)
        if new or not crowded:
            return
        if number not in self.one_product and self.name_parts(number):
            self.one_product.add(number)
            self.trace.append({'kind': 'one_product', 'item': number, 'said': self.item_said(number)})
            self.follow_up.insert(0, (number, labels, shown))
        elif not self.apart:
            alike = self.look_alikes(number, lead)
            if alike:
                self.apart.setdefault(lead, {})[number] = alike
                self.trace.append({'kind': 'apart', 'item': number, 'label': lead, 'from': alike})
                self.follow_up.insert(0, (number, labels, False))

    def _m_hear_prompt(self, sent: list[tuple[int, str, str]]) -> str:
        listed = '\n'.join(f'item {n}: {item.name}' for n, item in enumerate(self.items, 1))
        talk = []
        for index, (number, question, answer) in enumerate(sent, 1):
            about = f'about item {number}' if number else 'about several items'
            talk.append(f'{index}. ({about}) Q: {question}\nA: {answer}')
        return ''.join((HEAR, '\n\nITEMS:\n', listed, '\n\nQUESTIONS AND ANSWERS:\n', '\n'.join(talk)))

    def _m_known_words(self) -> set[str]:
        """Every word the episode already knows: item names and shelves, the properties of their needs, the listing
        keys and their categories, and the longer words of the opened listings' values."""
        known: set[str] = set()
        for item in self.items:
            known.update(_spelled(item.name) | _spelled(item.shelf or ''))
            for need in item.musts + item.told:
                known.update(_m_spelled_words(need.split(':', 1)[0]))
        for label, name, _, _ in self.key_table():
            known.update(_spelled(name) | _spelled(label))
        for record in self.full.values():
            for value in (_fact_block(record) or {}).values():
                known.update(w for w in _m_spelled_words(str(value)) if len(w) >= QUIET_WORD)
        return known

    def _m_gate_tools(self, ending: set[str]) -> list[str]:
        lines = []
        for tool in self.tools:
            spec = tool['function']
            if spec['name'] in ending:
                continue
            lines.append(f"- {spec['name']}: {spec.get('description', '')} | parameters: {_dump(spec.get('parameters') or {})}")
        return lines

    def _m_gate_prompt(self, ordering: bool, due: list[dict[str, Any]], tools: list[str]) -> str:
        pieces = [GATE.replace('STAGE', 'place the order' if ordering else 'end without a purchase'),
                  '\n\nREQUEST:\n', self.request,
                  '\n\nSTEPS DUE NOW:\n', '\n'.join(f"- {p['step']}" for p in due),
                  '\n\nLEDGER:\n', self.ledger(),
                  '\n\nCART NOW: ', self.cart_text(),
                  '\n\nRECORD (messages and step calls so far):\n',
                  '\n'.join(f'{i}. {line}' for i, line in enumerate(self.done_log, 1)) or '(nothing yet)']
        if self.notes:
            pieces.append('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]))
        pieces.append(f'\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()}')
        pieces.append(f" (the {'order' if ordering else 'ending'} itself takes one)")
        pieces.append('\n\nTOOLS:\n' + '\n'.join(tools))
        return ''.join(pieces)

    def _m_gate_heard(self, got: dict[str, Any], said: list[str]) -> tuple[list[str], str]:
        """Needs and a hold the gate's reading found in replies to the gate's own messages (only once it sent one)."""
        told: list[str] = []
        if not said:
            return told, ''
        self.take_heard({'needs': got.get('needs')}, told)
        return told, str(got.get('hold') or '').strip()

    def _m_gate_unproven(self, got: dict[str, Any], due: list[dict[str, Any]], ordering: bool) -> list[dict[str, Any]]:
        """The steps the reading lists that no record line shows done; at the order, when a due step needs a message,
        only lines made after the cart last changed count."""
        listed = [entry for entry in _items(got.get('steps')) if isinstance(entry, dict)]
        talk = ordering and any(p['messages'] > 0 for p in due)
        count = len(self.done_log)
        if talk:
            proof = {i for i, at in enumerate(self.done_steps[:count], 1) if at > self.cart_at}
        else:
            proof = set(range(1, count + 1))

        def proven(entry: dict[str, Any]) -> bool:
            return entry.get('done') is True and any((_count(n) or 0) in proof for n in _items(entry.get('lines')))
        return [entry for entry in listed if not proven(entry)]

    def _m_may_nudge(self, said: list[str], nudged: bool) -> bool:
        if said or nudged or not self.roles.get('ask'):
            return False
        return self.msg_used < self.msg_limit and self.left() >= 2

    def _m_gate_nudge(self, unproven: list[dict[str, Any]]) -> None:
        """Note in the record that the listed steps are not shown done, so the next round gives a call for them."""
        self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
        missing = '; '.join(str(entry.get('step') or '') for entry in unproven)[:200]
        self.done_log.append(''.join((f'(K2: no line made since the cart last changed at step {self.cart_at} shows ',
                                      missing,
                                      f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps',
                                      ' are left: give the call that does it)')))
        self.done_steps.append(-1)

    def _m_gate_message(self, args: dict[str, Any]) -> str | None:
        """Send the gate's message; the exchange for the record, or None when nothing went out."""
        text = max(_strings(args), key=len, default='')
        if not text:
            return None
        batch = self.send([(1, '', text)], 1)
        if not batch:
            return None
        self.gate_msgs += 1
        return f'Q: {text}\nA: {batch[0][2]}'

    def _m_gate_call(self, name: str, args: dict[str, Any]) -> str | None:
        """Make the gate's store call; the call and its result for the record, or None when it failed."""
        reply = self.act(name, args)
        if reply is None:
            return None
        seen = reply.get('observation')
        self.absorb(seen)
        shown = self.show(seen)
        call = f'{name} {_dump(args)} -> '
        self.done_log.append(f'step {self.step}: {call}{shown[:300]} (cart then: {self.cart_mark()})')
        self.done_steps.append(self.step)
        return call + shown[:400]

    def _m_record_ids(self, observation: Any) -> list[str]:
        """Ids of the full records an observation holds, in depth-first order, each once; a record's own children are
        not searched once it counted, and nothing deeper than six levels is."""
        found: list[str] = []
        pending: list[tuple[Any, int]] = [(observation, 0)]
        while pending:
            node, depth = pending.pop()
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
            pending.extend((child, depth + 1) for child in reversed(children))
        return found

    def _m_evidence_line(self, ident: str, needs: list[tuple[int, str]], many: bool) -> str | None:
        """The keys of one record whose names hold a requirement's property, or None when no key does."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        parts = []
        for number, need in needs[:8]:
            prop = need.split(':', 1)[0].strip()
            wanted = _stem(prop)
            hits = [f'{k}={str(v)[:80]}' for k, v in block.items() if _names(wanted, _stem(k))]
            if not hits:
                continue
            tag = f'(item {number}) ' if many else ''
            parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]))
        return f'{ident}: ' + '; '.join(parts) if parts else None

    def _p_series_of(self, carrier: int, index: dict[str, Any]) -> set[str] | None:
        """The series an item's rows carry, when enough of them carry one and the series are few; None otherwise."""
        rows = self.item_ids(carrier)
        tagged = [ident for ident in rows if _p_mark(index, ident).get('fam')]
        if len(tagged) < LINK_ROWS or 2 * len(tagged) < len(rows):
            return None
        families: set[str] = set()
        for ident in tagged:
            families.update(index['marks'][ident]['fam'])
        return None if len(families) > LINK_FAMILIES else families

    def _p_series_fault(self, carrier: int, carried: str, namer: str, families: set[str], main: int,
                        index: dict[str, Any]) -> str:
        """The note for one cart pair whose series do not match, with up to four seen rows of the wanted series."""
        wanted = _p_tags(index, namer, 'named') & families
        spare = [ident for ident in self.item_ids(carrier)
                 if _p_tags(index, ident, 'fam') & wanted and not self.sold_out(ident) and ident != carried][:4]
        holds = ', '.join(sorted(_p_tags(index, carried, 'fam') & families))
        head = f'{carried} is of series {holds}, while {namer} takes ' + ', '.join(sorted(wanted))
        if spare:
            return head + '; seen listings of that series: ' + ', '.join(spare)
        return head + f"; search that series with the item's words, or take another {self.items[main - 1].name} whose series the wanted lines carry"

    def _p_linked_hits(self, number: int, index: dict[str, Any]) -> Any:
        """(row, series) for each series of a system that a row of either tied item carries or names."""
        for part, main in self.bound:
            if number not in (part, main):
                continue
            system = self.link_system(part, main, index)
            if not system:
                continue
            for member in (part, main):
                for ident in self.item_ids(member):
                    marked = _p_tags(index, ident, 'fam') | _p_tags(index, ident, 'named')
                    for family in marked & system[2]:
                        yield ident, family

    def _p_pair_faults(self, part: int, main: int, cart: list[str], index: dict[str, Any]) -> list[str]:
        """Notes for the cart's part lines that do not fit its one device line of a bound pair of items."""
        device_rows, part_rows = self.item_ids(main), self.item_ids(part)
        devices = [ident for ident in cart if ident in device_rows]
        lines = [ident for ident in cart if ident in part_rows]
        if len(devices) != 1 or not lines:
            return []
        device = devices[0]
        linked = self.link_faults(part, main, device, lines, index)
        if linked:
            return linked
        codes = set(_p_mark(index, device).get('codes') or [])
        if not codes:
            return []

        def covers(ident: str) -> bool:
            return _p_codes_fit(self.fit_set(ident, index), _p_tags(index, ident, 'bars'), codes)
        part_name = self.items[part - 1].name
        device_codes = ', '.join(sorted(codes))
        found = []
        for line in lines:
            fits = self.fit_set(line, index)
            if not fits or covers(line):
                continue
            good = [ident for ident in part_rows if covers(ident) and not self.sold_out(ident) and ident not in cart][:4]
            if good:
                advice = "; seen parts of that kind that work with it: " + ', '.join(good)
            else:
                advice = '; no seen part of that kind works with it, so look for one or switch the device: ' + self.fit_plan(short=True)
            found.append(f"{line} ({part_name}) works with {', '.join(sorted(fits)[:4])}, not with {device} ({device_codes})" + advice)
        return found

    def _p_plan_lines(self, main: int, index: dict[str, Any]) -> list[str]:
        """The plan lines of one device item: the codes its parts name, ranked, and (cheapest pick) the lowest sets."""
        parts = [p for p, m in self.bound if m == main]
        tally: dict[str, dict[int, int]] = {}
        for code, part in self._p_part_codes(main, parts, index):
            per = tally.setdefault(code, {})
            per[part] = per.get(part, 0) + 1
        ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:6]
        out = []
        text = '; '.join(self._p_code_entry(main, code, per, index) for code, per in ranked)
        if text:
            out.append(f"codes the {self.items[main - 1].name} candidates' parts name -> {text}")
        sets = self.cheapest_sets(main, parts, index) if self.brief.get('pick') == 'cheapest' else ''
        if sets:
            out.append(sets)
        return out

    def _p_part_codes(self, main: int, parts: list[int], index: dict[str, Any]) -> Any:
        """(code, part item) once per buyable-looking part row and code: the codes the rows work with, then, for a part
        item that carries a series system, the series of each row."""
        for part in parts:
            for ident in self.item_ids(part):
                if not self.sold_out(ident):
                    yield from ((code, part) for code in self.fit_set(ident, index))
        for part in parts:
            system = self.link_system(part, main, index)
            if not system or system[0] != part:
                continue
            for ident in self.item_ids(part):
                if self.sold_out(ident):
                    continue
                for code in _p_tags(index, ident, 'fam') & system[2]:
                    yield code, part

    def _p_code_entry(self, main: int, code: str, per: dict[int, int], index: dict[str, Any]) -> str:
        """One ranked code: how many rows of each part item name it, and up to two devices seen that it reaches."""
        counted = ', '.join(f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))
        reached = [ident for ident in self.item_ids(main)
                   if code in _p_tags(index, ident, 'named') or _p_hits_any([code], _p_mark(index, ident).get('codes') or [])][:2]
        where = f" (device seen: {', '.join(reached)})" if reached else ' (device not seen yet)'
        return f'{code}: ' + counted + where

    def _p_priced_shelf(self, part: int, index: dict[str, Any]) -> list[tuple[float, str, set[str], list[str]]]:
        """Buyable priced rows of a part item as (price, id, codes it works with, codes it rules out), cheapest first."""
        shelf = []
        for ident in self.item_ids(part):
            price = self.price_of(ident)
            if price is None or not self.buyable(ident):
                continue
            shelf.append((price, ident, self.fit_set(ident, index), _p_mark(index, ident).get('bars') or []))
        shelf.sort(key=lambda o: (o[0], o[1]))
        return shelf

    def _p_whole_set(self, device: str, main: int, parts: list[int], systems: dict[int, Any], shelves: dict[int, Any],
                     index: dict[str, Any]) -> tuple[float, list[str]] | None:
        """(total, ids) of a device row with the cheapest fitting row of each part item, or None when one is missing."""
        mark = _p_mark(index, device)
        codes = mark.get('codes') or []
        price = self.price_of(device)
        if not (codes or mark.get('named')) or price is None or not self.buyable(device):
            return None
        total = price * self.items[main - 1].qty
        picks = [device]
        for part in parts:
            fitting = self._p_set_part(shelves[part], systems[part], part, device, codes, index)
            if not fitting:
                return None
            picks.append(fitting)
            total += (self.price_of(fitting) or 0.0) * self.items[part - 1].qty
        return (total, picks)

    def _p_set_part(self, shelf: list[Any], system: Any, part: int, device: str, codes: list[str],
                    index: dict[str, Any]) -> str | None:
        """The first (cheapest) shelf row that fits the device by its codes or by the series system."""
        by_series = bool(system) and system[0] == part
        for _, ident, fits, bars in shelf:
            if _p_codes_fit(fits, bars, codes):
                return ident
            if by_series and self.link_holds(ident, device, system[2], index) is True:
                return ident
        return None

    def _p_store_labels(self, cat_key: str | None) -> list[str]:
        """The distinct non-empty labels the seen rows carry under the category key, sorted."""
        if not cat_key:
            return []
        found = {str(row.get(cat_key) or '').strip() for row in self.rows.values()}
        found.discard('')
        return sorted(found)

    def _p_own_shelf(self, item: Any, labels: list[str]) -> str | None:
        """The label an item should move to, or None when it keeps its shelf."""
        name = _roots(item.name)
        if not name or not item.shelf.strip():
            return None
        others = [other for other in self.items if other is not item]
        shelf_words = _words(item.shelf)
        if not any(_words(other.shelf) == shelf_words for other in others):
            return None
        if any(_roots(other.name) == name for other in others):
            return None
        own = [label for label in labels if _roots(label) == name and _words(label) != shelf_words]
        if len(own) != 1 or any(_words(other.shelf) == _words(own[0]) for other in self.items):
            return None
        return own[0]

    def _p_listed_items(self, cat_key: str) -> set[int]:
        """Numbers of the items that some seen row is labelled for."""
        labels = [str(row.get(cat_key) or '') for row in self.rows.values()]
        return {number for number, item in enumerate(self.items, 1) if any(self.owns(item, text) for text in labels)}

    def _p_owners(self, label: str) -> list[int]:
        """Numbers of the items a label belongs to."""
        return [number for number, item in enumerate(self.items, 1) if self.owns(item, label)]

    def _p_item_faults(self, number: int, item: Any, ids: list[str], owner: dict[str, list[int]],
                       label: dict[str, str]) -> list[str]:
        """Cart problems of one item: no line, several lines, a wrong quantity, or a line from a less exact label."""
        tag = f'item {number} ({item.name})'
        lines = [i for i in ids if number in owner[i]]
        out = []
        if not lines:
            out.append(tag + ' has no line in the cart')
        elif sum(1 for i in lines if owner[i] == [number]) > 1:
            out.append(f"{tag} has {len(lines)} lines ({', '.join(lines)}); keep one")
        for i in lines:
            have = self.held_units(i)
            if owner[i] == [number] and have != item.qty:
                out.append(f'{i} is held at {have} unit(s); {tag} wants {item.qty}')
            better = self.fuller_label(number, label[i]) or self.named_label(number, label[i])
            if better:
                out.append(f'{i} is from {label[i]}, but the store has the label {better}, which names {tag} in full: take the line from there')
        return out

    def _p_budget_faults(self, ids: list[str]) -> list[str]:
        """Budget breaches of the cart (only judged when every line has a price and the overrun is under tenfold)."""
        prices = {i: self.price_of(i) for i in ids}
        if not isinstance(self.budget, (int, float)) or any(p is None for p in prices.values()):
            return []
        cap = self.budget
        if self.brief.get('budget_scope') == 'each':
            return [f'{i} costs {prices[i]:g}, over the {cap:g} limit for each item' for i in ids if cap < prices[i] <= 10 * cap]
        total = sum(prices[i] * self.held_units(i) for i in ids)
        if cap + 1e-09 < total <= 10 * cap:
            return [f'the total is {total:g}, over the budget of {cap:g}']
        return []

    def _p_chains_waiting(self) -> bool:
        """Whether some live requirement still points at a value on another listing, with steps and time left to follow it."""
        live = {need for _, need in self.needs_now()}
        if self.left() < TRAIL_ROOM or self.clock() > HUNT_BY:
            return False
        return any(need in live and need not in self.ref_values for _, need in self.refs)

    def _p_may_trail(self, stop: float | None) -> bool:
        """Room for another chain round: rounds under the cap, spend under the hard limit, and the clock before `stop`."""
        rounds = min(6 if self.left() > 15 else 4, self.trail_room or 6)
        if not (self.trails < rounds and self.cost < SPEND_HARD):
            return False
        return stop is None or self.clock() <= stop

    def _p_trail_models(self) -> list[str]:
        """The light model, and the heavy one too while the spend leaves room for it."""
        models = [LIGHT]
        if self.cost + HEAVY_ROOM <= SPEND_HARD:
            models.append(HEAVY)
        return models

    def _p_read_record(self, ident: str) -> Any:
        """The record of a listing whose page was read, else None."""
        if not self.page_read(ident):
            return None
        return self.full.get(ident) or self.rows.get(ident)

    def _p_trail_prompt(self, pending: list[str]) -> str:
        """The chain prompt: the pending requirements, then up to 45 read pages and the other rows, newest first, 60 in all."""
        newest = list(reversed(self.seen_order))
        pages = [record for record in map(self._p_read_record, newest) if record][:45]
        records = ['OPENED ' + _p_snippet(record, 300, 1400) for record in pages]
        unread = [ident for ident in newest if not self._p_read_record(ident)][:60 - len(records)]
        for ident in unread:
            if ident in self.full:
                records.append('ROW ' + _p_snippet(self.full[ident], 120, 700))
            else:
                records.append('ROW ' + _p_snippet(self.rows.get(ident), 80, 300))
        return ''.join([TRAIL, '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n', '\n'.join(pending),
                        '\n\nRECORDS (newest first):\n', '\n'.join(records)])

    def _p_chain_need(self, chain: dict[str, Any], pending: list[str]) -> str | None:
        """The pending requirement a chain answers: the one it quotes, or the only one pending."""
        quoted = str(chain.get('need') or '').strip()
        match = quoted if quoted in pending else None
        return match or (pending[0] if len(pending) == 1 else None)

    def _p_chain_settles(self, chain: dict[str, Any], links: list[dict[str, Any]], ids: list[str]) -> bool:
        """A chain is settled when every page on it was read and either it says it is done with a value, or its last
        link asks for nothing further."""
        if not links or not all(self.page_read(ident) for ident in ids):
            return False
        if chain.get('done') is True and str(chain.get('value') or '').strip():
            return True
        return not str(links[-1].get('find') or '').strip()

    def _p_take_chains(self, got: dict[str, Any], pending: list[str]) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """Apply the model's chain answers: record the settled values, and return the seen rows still to open, the notes
        for links not seen at all, and (description, search words) for each of those."""
        reach: list[str] = []
        lost: list[str] = []
        gaps: list[tuple[str, str]] = []
        for chain in _items(got.get('chains')):
            if not isinstance(chain, dict):
                continue
            need = self._p_chain_need(chain, pending)
            if not need:
                continue
            links = [link for link in _items(chain.get('links')) if isinstance(link, dict)]
            ids = [str(link.get('id') or '').strip() for link in links]
            if self._p_chain_settles(chain, links, ids):
                self.ref_values[need] = str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}'
                self.version += 1
                continue
            closed = [ident for ident in ids if ident in self.rows and not self.page_read(ident)]
            if closed:
                fresh = [ident for ident in closed if ident not in reach]
                reach.extend(fresh)
                continue
            gap = next((link for link, ident in zip(links, ids) if ident not in self.rows), {})
            about = str(gap.get('about') or '') or 'the listing it points at'
            words = str(gap.get('find') or '').strip()
            lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
            gaps.append((about, words))
        return reach, lost, gaps

    def _p_open_one(self, tool: str, target: str, ident: str) -> dict[str, Any] | None:
        """Open one listing with a one-target tool; None when the call cannot be made, fails, or ends the episode."""
        faults: list[str] = []
        args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
        if faults or self.rushed():
            return None
        reply = self.act(tool, args)
        if reply is None or self.finished:
            return None
        return reply

    def _p_open_pages(self, tool: str, target: str, picks: list[str], seen: list[str]) -> None:
        """Open the chosen pages in turn, adding what each shows to `seen`; stop at the first that cannot be opened."""
        for ident in picks:
            reply = self._p_open_one(tool, target, ident)
            if reply is None:
                return
            self.observe(tool, reply.get('observation'))
            seen.append(f'{ident}: ' + self.show(reply.get('observation'))[:1500])

    def _p_trail_report(self, seen: list[str], lost: list[str]) -> tuple[str, bool]:
        """The text a chain follow hands back, and whether the order must wait for pages still missing."""
        parts = []
        if seen:
            parts.append('[K2 opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n')
        if self.ref_values:
            values = '; '.join(f'{k} -> {v}' for k, v in self.ref_values.items())
            parts.append('[values read from the pages requirements point at] ' + values + '\n')
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            parts.append(_P_HOLD_HEAD + ' | '.join(lost[:3]) + _P_HOLD_TAIL)
        return (''.join(parts), hold)

    def _p_shop_named(self, about: str, idents: Any, unopened_only: bool) -> list[str]:
        """The rows among `idents` with a short text field that the description names as a whole phrase."""
        skip = self.not_maker_keys()
        lowered = about.lower()
        hits = []
        for ident in idents:
            if unopened_only and ident in self.opened:
                continue
            names = _p_name_fields(self.rows.get(ident) or {}, skip, 2)
            if any(_p_word_in(name.lower(), lowered) for name in names):
                hits.append(ident)
        return hits

    def _p_shop_words(self, about: str, skip: tuple[Any, ...]) -> set[str]:
        """Words of the name fields of the rows of the shop the description names (they say nothing about the listing)."""
        shop = self.shop_rows(about, list(self.rows)) + self.opened_shop(about)
        found: set[str] = set()
        for ident in shop:
            for field in _p_name_fields(self.rows.get(ident) or {}, skip, 0):
                found |= _p_tokens3(field.lower())
        return found

    def _p_shop_page_shows(self, about: str, values: set[str]) -> bool:
        """Whether an opened page of the named shop already shows every value the description gives (traced as known)."""
        if not values:
            return False
        for ident in self.opened_shop(about):
            text = ' '.join(_p_scalars(self.full.get(ident) or self.rows.get(ident) or {})).lower()
            if all(_p_word_in(value, text) for value in values):
                self.trace.append({'kind': 'link_known', 'id': ident})
                return True
        return False

    def _p_link_search(self, plan: tuple[str, dict[str, Any], str], words: str, about: str, memo: str) -> list[str]:
        """Search the store for the linked listing by its words (then by their first two), stopping once a search lists
        rows; returns the listed rows of the named shop."""
        tool, base, field = plan
        known: list[str] = []
        for query in dict.fromkeys((words, ' '.join(words.split()[:2]))):
            faults: list[str] = []
            args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            self.link_searches += 1
            reply = self.act(tool, args)
            if reply is None:
                break
            observation = reply.get('observation')
            self.observe(tool, observation)
            listed = [ident for ident, _ in self.records_in(observation)]
            self.link_seen[memo] = listed
            known = self.shop_rows(about, listed)
            self.trace.append({'kind': 'link_search', 'words': query[:80], 'rows': known[:4]})
            if listed:
                break
        return known

    def _p_of_kind(self, ids: list[str], kind: set[str]) -> list[str]:
        """The rows whose record shares a word with the kind the description asks for (all of them when it gives none)."""
        if not kind:
            return list(ids)
        return [ident for ident in ids if kind & _p_tokens3(_dump(self.rows.get(ident) or {}).lower())]

    def _p_audit_record(self, ident: str) -> str:
        """One audit record: the opened page when there is one, else the search row."""
        page = self.full.get(ident)
        if page:
            return 'OPENED ' + _p_snippet(page, 200, 1200)
        return 'ROW ' + _p_snippet(self.rows.get(ident), 120, 500)

    def _p_wants_heavy(self, judges: list[str], got: dict[str, Any] | None) -> bool:
        """A second, heavy audit is worth it when the heavy model may judge, credit lasts, and the light one named nothing."""
        if HEAVY not in judges or self.no_credit:
            return False
        return not self.audit_fits(got)

    def _p_open_plan(self, rows: list[str]) -> list[tuple[str, dict[str, Any]]] | None:
        """The calls that open the rows: one compare call when the store has one, else the first two by the facts tool."""
        many, joiner = self.compare_target(), self.roles.get('compare')
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            return [(joiner, {many: rows[:self.compare_room()]})]
        if tool and target:
            return [(tool, {target: ident}) for ident in rows[:2]]
        return None

    def _q_end_window(self) -> set[Any] | None:
        """The tool names still offered as the steps run out, or None while every tool is."""
        finishing = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        if not finishing:
            return None
        spare = self.left() - self.tail()
        if spare <= 0:
            jobs: tuple[str, ...] = ()
        elif spare <= 3:
            jobs = ('cart', 'cart_view')
        elif self.left() <= len(self.items) + 5 and self.buys == 0:
            jobs = ('cart', 'cart_view', 'offer')
        else:
            return None
        return finishing | set(self._q_role_tools(*jobs))

    def _q_interviewer_speaks(self) -> bool:
        """Whether K2's interviewer can word messages to the shopper itself (then the model never sends one)."""
        return self.message_args('x' * 200) is not None

    def _q_role_tools(self, *jobs: str) -> tuple[Any, ...]:
        return tuple(self.roles.get(job) for job in jobs)

    def _q_cart_now(self) -> str:
        return _Q_CART_HEAD + self.cart_text()

    def _q_cart_fresh(self) -> bool:
        return self.cart_view is not None and self.cart_seen_at >= self.cart_at

    def _q_plain_tools(self) -> set[Any]:
        """Tools with a job other than searching: their results are never paged or searched again."""
        return {tool for job, tool in self.roles.items() if job != 'search'}

    def _q_say(self, content: str) -> None:
        self.chat.append({'role': 'user', 'content': content})

    def _q_tool_reply(self, cid: str, content: str) -> None:
        self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': content})

    def _q_settle(self, call: dict[str, Any], batch: int) -> tuple[str, str]:
        allowed = {t['function']['name'] for t in self.offered()}
        cid = call['id']
        if call.get('joined'):
            return (cid, _Q_JOINED)
        fn = call.get('function') or {}
        step = _q_Step(fn.get('name'))
        # each stage either holds the call back with a reply, or returns None and hands the call on
        held = (self._q_refuse_early(step.name, allowed)
                or self._q_read_args(step, fn.get('arguments'))
                or self._q_shape_call(step, batch)
                or self._q_skip_repeat(step)
                or self._q_hold_ending(step))
        return (cid, held or self._q_carry_out(step))

    def _q_refuse_early(self, name: Any, allowed: set[str]) -> str | None:
        if self.finished:
            return 'Not run: the episode is over.'
        if name == self.roles.get('ask') and self._q_interviewer_speaks():
            return _Q_SKIPPED + "K2's interviewer handles every message to the shopper."
        if name not in self.schemas:
            return _Q_SKIPPED + f'unknown tool. Tools: {_q_tool_names(allowed)}.'
        if name not in allowed:
            return _Q_SKIPPED + 'too few steps are left for that; use: ' + _q_tool_names(allowed) + '.'
        return None

    def _q_read_args(self, step: _q_Step, raw: Any) -> str | None:
        if isinstance(raw, dict):
            args = raw
        elif (raw or '').strip():
            args = _json_obj(raw)
            if args is None:
                return _Q_SKIPPED + 'the arguments were not one JSON object.'
        else:
            args = {}
        problems: list[str] = []
        args = _shape(args, self.schemas[step.name], 'args', problems)
        if problems:
            return _Q_SKIPPED + '; '.join(problems[:6])
        args, wrong = self.resolve_ids(step.name, args)
        if wrong:
            return _Q_SKIPPED + wrong + '.'
        step.args = args
        return None

    def _q_shape_call(self, step: _q_Step, batch: int) -> str | None:
        name = step.name
        if name == self.roles.get('cart'):
            step.args = self.whole_removal(name, step.args)
        self._q_fit_compare_room(step)
        step.target = self.single_target(name)
        told = self._q_gone_target(step) or self._q_nothing_to_remove(step)
        if told:
            return told
        step.ending = self.is_buy(name, step.args) or self.is_refuse(name, step.args)
        if step.ending and batch > 1:
            return _Q_SKIPPED + 'send the finishing call on its own, after reading your other results.'
        if not step.ending and self.left() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')):
            return _Q_SKIPPED + ("the last steps are kept for checking out (with the shopper's approval of the basket"
                                 + ' when it is due) or ending without a purchase.')
        return None

    def _q_fit_compare_room(self, step: _q_Step) -> None:
        """A side-by-side call longer than this store has shown it can take is cut to the first listings that fit."""
        step.many = self.compare_target() if step.name == self.roles.get('compare') else None
        if not step.many:
            return
        listed = step.args.get(step.many)
        if not isinstance(listed, list) or not self.compare_bad:
            return
        room = self.compare_room()
        if len(listed) <= room:
            return
        unsent = ', '.join(map(str, listed[room:]))
        step.notes.append(f'{_Q_TAG}Sent the first {room} listings only: this store shows at most {room} side by side.'
                          + f' Not shown yet: {unsent}.')
        step.args = {**step.args, step.many: listed[:room]}

    def _q_gone_target(self, step: _q_Step) -> str | None:
        target = step.target
        if not target or step.name not in self._q_role_tools('open', 'offer'):
            return None
        if str(step.args.get(target)) not in self.gone:
            return None
        ident = step.args[target]
        return _Q_SKIPPED + (f'{ident} sold out at step {self.gone[str(ident)]} and will not come back.'
                             + ' Replace its cart line with another fitting listing.')

    def _q_nothing_to_remove(self, step: _q_Step) -> str | None:
        """A cart call that takes units off a listing the cart has never held is answered without a step."""
        if step.name != self.roles.get('cart'):
            return None
        values = step.args.values()
        if not any(((_count(v) or 0) < 0 for v in values if not isinstance(v, str))):
            return None
        view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
        absent = next((v for v in values if isinstance(v, str) and v in self.rows and v not in self.cart_lines
                       and v not in view and v not in self.cart_said), None)
        if absent is None:
            return None
        return _Q_SKIPPED + f'{absent} is not in the cart, so there is nothing to take off. Cart now: ' + self.cart_text()

    def _q_skip_repeat(self, step: _q_Step) -> str | None:
        name = step.name
        step.fresh = name not in self._q_plain_tools() and self.query_is_new(name, step.args)
        sent = step.args
        step.args = self.first_look(name, sent, step.fresh)
        if step.args != sent:
            step.notes.append(f'{_Q_TAG}This is a new search, so K2 asked for its first page ({_dump(sent)} was the call you sent).')
        step.args = self.row_ids(name, step.args)
        if name == self.roles.get('search') and self.codes_dead and self.names_code(step.args):
            return _Q_SKIPPED + ("this store's search found nothing for a model or part code."
                                 + ' Search by words and values, list the category, and read the codes on the rows.')
        step.mark = name + _dump(step.args)
        if not step.ending and name != self.roles.get('cart') and step.mark in self.recent:
            return _Q_SKIPPED + 'you just made this same call and nothing changed; its result is above.'
        step.readers = set(self._q_role_tools('search', 'open', 'offer', 'compare')) - {None}
        return self._q_seen_before(step)

    def _q_seen_before(self, step: _q_Step) -> str | None:
        name, args = step.name, step.args
        reading = name in step.readers
        cached = self.read_cache.get(step.mark)
        if reading and cached and cached[0] == self.version:
            return _Q_SKIPPED + f'the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]
        target = step.target
        if target and reading and name != self.roles.get('compare'):
            ident = str(args.get(target))
            if self.shown_already(name, ident) or (name, ident) in self.factless:
                return _Q_SKIPPED + (f'{args[target]} has shown everything this view gives since the last change.'
                                     + ' Its record: ' + self.record_of(str(args[target])))
        return self._q_drop_shown(step)

    def _q_drop_shown(self, step: _q_Step) -> str | None:
        """Listings of a side-by-side call that have shown all this view gives are left out (all of them: no call)."""
        listed = step.args.get(step.many) if step.many else None
        if not isinstance(listed, list) or not listed:
            return None
        unseen = [i for i in listed if not self.shown_already(step.name, str(i))]
        if not unseen:
            records = '\n'.join((f'{i}: {self.record_of(str(i))}' for i in listed))
            return _Q_SKIPPED + ('these listings have shown everything this view gives since the last change.'
                                 + ' Their records:\n' + records[:4000])
        if len(unseen) < len(listed):
            skipped = ', '.join((str(i) for i in listed if i not in unseen))
            step.notes.append(f'{_Q_TAG}Left out {skipped}: shown in full already, nothing changed since.')
            step.args = {**step.args, step.many: unseen}
            step.mark = step.name + _dump(step.args)
        return None

    def _q_hold_ending(self, step: _q_Step) -> str | None:
        name = step.name
        if step.ending and self.left() > 2:
            if self.is_buy(name, step.args):
                held = self._q_checkout_hold()
            else:
                held = self._q_decline_hold() if self.left() > 5 else None
            if held:
                return held
        if self.is_refuse(name, step.args):
            told = self._q_late_decline(step) or self._q_push_search()
            if told:
                return told
            done = self.complete_cart()
            if done:
                return done + (' [episode finished]' if self.finished else '')
            step.args = self.quiet(name, step.args)
        if self.is_buy(name, step.args):
            late = self.late_question()
            if late:
                return _Q_PENDING + late + self._q_cart_now()
        return self._q_pass_gate(step)

    def _q_pass_gate(self, step: _q_Step) -> str | None:
        """The steps the request asks for at the order or the ending run now; a gate that says no holds the call."""
        if not step.ending or self.left() < 2:
            return None
        go, said = self.gate('order' if self.is_buy(step.name, step.args) else 'end')
        if not go:
            return _Q_PENDING + said + self._q_cart_now()
        if said:
            step.notes.insert(0, _Q_TAG + said)
        return None

    def _q_checkout_hold(self) -> str | None:
        """K2's last look before a checkout goes out; a reply when the order should wait, else None."""
        hunted = self.hunt_cheaper()
        wider = self.wider_hold()
        if wider:
            return wider + self._q_cart_now()
        reread = self.reread_offers()
        if self.note_sold:
            self.find_sold()
        told = self._q_cart_change() or self._q_sold_hold()
        if told:
            return told
        chain, hold = self.trail()
        if hold:
            return _Q_WAITING + chain + self._q_cart_now()
        return (self._q_fit_hold() or self._q_count_hold() or self._q_verdict_hold(hunted, chain)
                or self._q_spend_hold(reread))

    def _q_cart_change(self) -> str | None:
        """When the cart changed since K2 last saw it, look at it once; a change it shows holds the order."""
        viewer = self.roles.get('cart_view')
        if not (self.cart_touched and self.cart_seen_at < self.cart_at and viewer and self.left() > 5) or self.rushed():
            return None
        problems: list[str] = []
        view_args = _shape({}, self.schemas.get(viewer), 'args', problems)
        if problems:
            return None
        view = self.act(viewer, view_args)
        if view is None:
            return None
        self.cart_view = view.get('observation')
        self.cart_seen_at = self.step
        change = self.observe(viewer, self.cart_view)
        if not change:
            return None
        self.queued = []
        return (_Q_WAITING + 'Looking at the cart showed a change: ' + change + self._q_cart_now()
                + '\nApply the change, fix the cart if needed, then check out.')

    def _q_sold_hold(self) -> str | None:
        swap = self.swap_hint()
        lost = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
        if not swap and not lost:
            return None
        if not swap:
            self.held += 1
        why = swap or f"{', '.join(lost)} in the cart is sold out: replace that line first."
        return _Q_WAITING + why + self._q_cart_now()

    def _q_fit_hold(self) -> str | None:
        if self.fit_holds >= 2 or self.left() <= self.tail() + 3:
            return None
        misfit = self.fit_faults()
        if not misfit:
            return None
        self.fit_holds += 1
        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
        return (_Q_WAITING + 'A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3])
                + '. Swap the part (or the device), then check out.' + self._q_cart_now())

    def _q_count_hold(self) -> str | None:
        """Lines held at the wrong count: K2 sets the counts itself when it can, otherwise the model is told."""
        if self.line_holds >= 2 or self.left() <= self.tail() + 2:
            return None
        faults = self.line_faults()
        if faults and all((' is held at ' in f for f in faults)):
            fixed = self.fix_counts(self.count_fixes())
            if fixed:
                faults = self.line_faults()
                if not faults:
                    self.line_holds += 1
                    return (_Q_WAITING + "K2 set the cart's quantities itself: " + '; '.join(fixed)
                            + '. If the store refused any of these, fix those lines; then check out.' + self._q_cart_now())
        if not faults:
            return None
        self.line_holds += 1
        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
        return (_Q_WAITING + 'The cart does not match the request yet: ' + '; '.join(faults[:6])
                + ". Fix those lines (the cart tool's own way to add or take off units), then check out." + self._q_cart_now())

    def _q_verdict_hold(self, hunted: str, chain: str) -> str | None:
        """Read what the check needs about the cart's lines, then let the independent check judge the cart."""
        lines = self.cart_ids()
        read = self.fill_facts(lines, self.tail() + (1 if len(lines) >= 2 else 3))
        self.show_links(lines, self.tail() + 1)
        read += self.cheaper_rows()
        verdict = self.check_cart()
        if not verdict:
            return None
        seen = ''
        if read:
            seen = "K2 read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n'
        if hunted:
            seen = hunted + ' ' + self.below_cart() + '\n' + seen
        return _Q_PENDING + chain + seen + verdict + self._q_cart_now()

    def _q_spend_hold(self, reread: str) -> str | None:
        if not reread or self.cost <= SPEND_SOFT:
            return None
        return (_Q_WAITING + "K2 re-read the cart's listings just now:\n" + reread
                + '\nCheck out again when every line still shows stock and a price inside the budget;'
                + ' otherwise replace that line first.')

    def _q_decline_hold(self) -> str | None:
        """K2's review before ending without a purchase; a reply when the ending should wait, else None."""
        pointed = self.refuse_trail()
        if pointed:
            return _Q_WAITING + pointed
        self.hunt_fits()
        review = self.audit()
        if review:
            return _Q_WAITING + review
        if not self.leads or self.left() <= 4:
            return None
        # the first refusal of a decline names the leads, the second has the first lead checked independently
        stage = {0: self._q_name_leads, 1: self._q_confirm_lead}.get(self.refuse_blocks)
        return stage() if stage else None

    def _q_name_leads(self) -> str:
        self.refuse_blocks = 1
        self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
        return _Q_WAITING + 'The review found listings that appear to fit: ' + ', '.join(self.leads) + _Q_LEADS_END

    def _q_confirm_lead(self) -> str | None:
        self.refuse_blocks = 2
        first = self.leads[0]
        self.check_cart([first], SPEND_HARD)
        if not self.verdicts.get(self.check_key([first])) or first not in self.leads:
            return None
        self.confirmed = first
        self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
        return _Q_WAITING + (f"An independent check of {first}'s records says it meets every requirement."
                             + ' Add it to the cart at the requested quantity and check out;'
                             + ' end without a purchase only if its record clearly breaks a requirement.')

    def _q_late_decline(self, step: _q_Step) -> str | None:
        """In the last steps, a decline while a full basket was built once is turned back (once)."""
        if self.left() > 5 or self.late_held:
            return None
        lines = [i for i in self.built if not self.sold_out(i)]
        # K2 keeps this cart freshness in the same flag as the new-search test; later page checks read it
        step.fresh = self._q_cart_fresh()
        holding = self.cart_ids()
        missing = [i for i in lines if i not in holding or (i in self.dropped and not step.fresh)]
        if len(lines) < len(self.items) or len(missing) + 1 > self.left():
            return None
        self.late_held = True
        self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
        if missing:
            redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out."
        else:
            redo = 'Check out the cart as it is.'
        return _Q_WAITING + _Q_LATE_DECLINE + redo + _Q_LATE_DECLINE_END

    def _q_push_search(self) -> str | None:
        if self.leads or self.pushed or self.left() <= self.max_steps // 3:
            return None
        self.pushed = True
        return _Q_WAITING + (f'{self.left()} steps are still left, so search in ways you have not tried before ending:'
                             + ' browse the category of each requested item and page through it with the token or offset'
                             + " the store returns, try single-word value queries and the property's other names,"
                             + ' and open the listings that might fit. End without a purchase only after that.')

    def _q_carry_out(self, step: _q_Step) -> str:
        """Send the call to the store, learn from its result, and word the reply the model reads."""
        name = step.name
        reply = self.act(name, step.args)
        if reply is None:
            return 'The call failed (network or server). You may try again.'
        observation = reply.get('observation')
        self._q_learn_compare_width(step, observation)
        self._q_remember_call(step)
        self.absorb(observation)
        self._q_note_records(step, observation)
        plain = self._q_plain_tools()
        paged = self._q_more_rows(step, observation, plain)
        self._q_learn_roles(name, observation)
        if name == self.roles.get('cart'):
            self._q_after_cart(name, step.args, observation)
        if name == self.roles.get('search'):
            self._q_after_search(step.args, observation)
        if name == self.roles.get('cart_view'):
            self.cart_view = observation
            self.cart_seen_at = self.step
        if self.is_buy(name, step.args):
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
        return self._q_compose(step, reply, observation, plain, paged)

    def _q_learn_compare_width(self, step: _q_Step, observation: Any) -> None:
        """A side-by-side result that names none of the listings sent marks that many as too many for this store."""
        listed = step.args.get(step.many) if step.many else None
        if not isinstance(listed, list) or len(listed) <= 1:
            return
        shown = _dump(observation)
        if any((str(i) in shown for i in listed)):
            self.compare_ok = max(self.compare_ok, len(listed))
            return
        self.compare_bad = min(self.compare_bad or 999, len(listed))
        step.notes.append(f'{_Q_TAG}The side-by-side view showed nothing for {len(listed)} listings:'
                          + f' send at most {self.compare_room()} per call.')

    def _q_remember_call(self, step: _q_Step) -> None:
        self.recent = (self.recent + [step.mark])[-3:]
        for key, value in step.args.items():
            values = value if isinstance(value, list) else [value]
            if values and all((isinstance(v, str) and v in self.rows for v in values)):
                self.id_args.add((step.name, key))

    def _q_note_records(self, step: _q_Step, observation: Any) -> None:
        name = step.name
        if name == self.roles.get('compare'):
            rows = self.records_in(observation)
            if rows:
                self.facts_from[name] = self.facts_from.get(name, False) or any((_fact_block(r) for _, r in rows))
        if name in step.readers - {self.roles.get('search')}:
            self.note_view(name, observation)

    def _q_more_rows(self, step: _q_Step, observation: Any, plain: set[Any]) -> str:
        """Further pages, a first page and a shorter search that K2 runs itself after a search result."""
        if step.ending:
            return ''
        searching = step.name not in plain
        found = self.more_pages(step.name, step.args, observation) if searching else ''
        if step.fresh:
            found += self.last_page_first(step.name, step.args, observation)
        if searching:
            found += self.shorter_search(step.name, step.args, observation)
        return found

    def _q_learn_roles(self, name: str, observation: Any) -> None:
        """A one-listing view whose pages carry no attributes hands the open job on, and may serve as the offer view."""
        if not self.single_target(name) or name in self._q_role_tools('cart', 'buy', 'refuse', 'ask'):
            return
        listing = self.ident(observation) if isinstance(observation, dict) else None
        if not listing:
            return
        self.note_facts(name, bool(_fact_block(observation)))
        if name == self.roles.get('open') and self.facts_from.get(name) is False:
            other = next((t for t in self.schemas if self._q_may_open(t, name)), None)
            if other:
                self.roles['open'] = other
        if self.facts_from.get(name) is False and name != self.roles.get('open') and not self.roles.get('offer'):
            self.roles['offer'] = name
        if name == self.offer_tool():
            self.offer_seen[listing] = self.step

    def _q_may_open(self, tool: str, current: str) -> bool:
        return (tool != current and self.single_target(tool) and self.facts_from.get(tool) is not False
                and tool not in self._q_role_tools('cart', 'buy', 'refuse', 'ask', 'search') and not self.lists_things(tool))

    def _q_after_cart(self, name: str, args: dict[str, Any], observation: Any) -> None:
        self.note_cart(name, args)
        self.cart_said = _dump(observation)
        self.recent = []
        counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None]
        if counts:
            first = counts[0]
            for value in args.values():
                if not isinstance(value, str) or value not in self.rows:
                    continue
                if first > 0:
                    self.dropped.discard(value)
                elif not _in_rows(observation, value):
                    self.dropped.add(value)
        kept = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
        if len(kept) >= len(self.built):
            self.built = kept

    def _q_after_search(self, args: dict[str, Any], observation: Any) -> None:
        if self.names_code(args):
            self.code_tries.append(bool(self.records_in(observation)))
            self.codes_dead = len(self.code_tries) >= 2 and not any(self.code_tries)
        if self.records_in(observation):
            self.searched = (self.searched + [dict(args)])[-8:]

    def _q_compose(self, step: _q_Step, reply: dict[str, Any], observation: Any, plain: set[Any], paged: str) -> str:
        name = step.name
        body = self.show(observation) + paged + ''.join(step.notes)
        if name not in plain and not step.ending:
            body += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, step.args, observation)
        extra = self._q_side_notes(step, reply, observation)
        if name in step.readers:
            # the cache keeps the reply without the notes that only held at this moment
            self.read_cache[step.mark] = (self.version, self.step, body)
        return body + extra + (' [episode finished]' if reply.get('finished') else '')

    def _q_side_notes(self, step: _q_Step, reply: dict[str, Any], observation: Any) -> str:
        parts: list[str] = []
        proof = self.evidence(observation)
        if proof:
            parts.append('\n' + proof)
        if self._q_zero_kept(step, observation):
            parts.append(_Q_TAG + 'A count of 0 left that line in the cart: this tool adds the count to the line,'
                         + ' so a negative count takes units off.')
        self.note_sold = False
        parts.append(self._q_read_note(observation))
        parts.append(self._q_recheck_sold(step, reply, observation))
        self.note_sold = False
        swap = self.swap_hint()
        if swap:
            parts.append('\n' + swap)
        parts.append(self._q_cart_misfit(step.name))
        if self.queued:
            parts.append("\n[notes seen on K2's own calls] " + ' | '.join(self.queued))
            self.queued = []
        return ''.join(parts)

    def _q_zero_kept(self, step: _q_Step, observation: Any) -> bool:
        """A cart call with a count of 0 whose listing is still in the result: this cart tool adds counts."""
        if step.name != self.roles.get('cart'):
            return False
        values = step.args.values()
        if not any((_count(v) == 0 for v in values if not isinstance(v, str))):
            return False
        return any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in values))

    def _q_read_note(self, observation: Any) -> str:
        lines = self.loose_lines(observation)
        if not lines:
            return ''
        meaning = self.take_note(lines)
        if not meaning:
            return ''
        return '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'

    def _q_recheck_sold(self, step: _q_Step, reply: dict[str, Any], observation: Any) -> str:
        """After a refused checkout or a loose note about sold-out lines, K2 re-reads the cart's lines."""
        bought = self.is_buy(step.name, step.args)
        refused = bought and not reply.get('finished')
        repeated = False
        if refused and self.sold_lines:
            said = self.spoken(observation)
            repeated = any((line in said for line in self.sold_lines))
        loose = self.note_sold and (refused or (not self.note_named and not bought))
        if not (loose or repeated) or not self.cart_ids() or self.finished:
            return ''
        seen = self.find_sold()
        return "\n[K2 re-read the cart's lines after the refusal]\n" + seen if seen else ''

    def _q_cart_misfit(self, name: str) -> str:
        if name != self.roles.get('cart') or not self.bound or self.finished:
            return ''
        misfit = self.fit_faults()
        return _Q_TAG + 'These lines do not work together yet: ' + '; '.join(misfit) if misfit else ''

    def _q_take_opening(self) -> Any:
        """What the store showed before the first step, taken into K2's records (None when it showed nothing)."""
        opening = self.view.get('opening')
        if opening is None:
            return None
        opening = _restore(opening)
        self.absorb(opening)
        return opening

    def _q_intro(self, opening: Any, moved: str) -> str:
        parts = ["SHOPPER'S REQUEST:\n" + self.request]
        if opening is not None:
            parts.append('\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening))
        parts.append(_Q_MOVED_HEAD + moved + _Q_MOVED_TAIL if moved else _Q_START)
        parts.append(self.status())
        return ''.join(parts)

    def _q_close_out(self) -> None:
        """Finish an episode the model left open; a refused finish gives the model one more stretch of turns."""
        turned = self.finish(again=True)
        if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
            self._q_say('[K2] K2 tried to end the episode and ' + turned
                        + '\nApply what the store says, then finish.\n' + self.status())
            self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
        if not self.finished:
            self.finish()

    def _q_may_go_on(self, loop_cap: int, spend_cap: float) -> bool:
        if self.finished or self.broke or self.left() <= 0:
            return False
        if self.loop_calls >= loop_cap or self.cost >= spend_cap:
            return False
        if self.clock() < LAST_TURN:
            return True
        # past the last-turn mark only an empty cart with model time to spare keeps the loop going
        return not self.cart_ids() and self.model_room() > EMPTY_ROOM

    def _q_after_miss(self) -> bool:
        """Count a model turn that brought no message; True when the loop should stop."""
        self.misses += 1
        if self.no_credit or self.calls >= ALL_CALLS:
            return True
        if self.misses <= 2:
            return False
        if self.clock() >= OUTAGE_BY:
            return True
        sleep(BRIEF_GAP)
        return False

    def _q_clean_call(self, number: int, call: dict[str, Any]) -> dict[str, Any]:
        """A model tool call in the form the chat keeps: an id, a name, and its arguments as one JSON object."""
        fn = call.get('function')
        if not isinstance(fn, dict):
            fn = {}
        raw = fn.get('arguments')
        if isinstance(raw, dict):
            parsed = raw
        elif isinstance(raw, str) and raw.strip():
            parsed = _json_obj(raw)
        else:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        ident = call.get('id') or f'k{self.loop_calls}_{number}'
        return {'id': str(ident), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _dump(parsed)}}

    def _q_take_turn(self, content: str, calls: list[dict[str, Any]]) -> None:
        asking = self._q_ask_first(calls)
        self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
        if asking:
            for call in calls:
                self._q_tool_reply(call['id'], _Q_SKIPPED + 'the shopper is asked first.')
            summary = self.interview()
            self._q_say((summary or 'The interviewer had nothing to ask.') + '\n' + self.status())
            return
        self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
        replies = self.run_calls(calls)
        last = len(replies) - 1
        for cid, text in replies[:last]:
            self._q_tool_reply(cid, text)
        for cid, text in replies[last:]:
            self._q_tool_reply(cid, text + '\n' + self.status())

    def _q_ask_first(self, calls: list[dict[str, Any]]) -> bool:
        """A first turn that would fill the cart or finish waits for the interviewer's questions to the shopper."""
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
        if not any((c['function']['name'] in committing for c in calls)):
            return False
        return self.left() > 8 and self.clock() <= ASK_BY

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = K2(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return list(episode.run())
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return list(episode.trace)
__all__ = ['agent_main']
