"""RYU: a buying agent for one shopper in an ORO store world (a fork of the public Halo v2 agent, itself built on Know
Thyself).

A light model walks the store with the world's own tools. The code keeps the books (the requested items, their firm
needs, the budget, every listing seen), plans the questions to the shopper, checks the cart before an order and the
evidence before ending without one, and keeps every episode inside the validator's clock.
"""
from __future__ import annotations

import json
import random
import re
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

# ---- models: the light one drives the store, the heavy one reads the request and judges, the spare one stands in
LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'

# ---- model calls per episode: tool-loop turns, and all calls together
LOOP_CALLS = 42
ALL_CALLS = 60

# ---- spend per episode in dollars (one run key pays for every task of a run)
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
HEAVY_ROOM = 0.02           # spend that must remain before a heavy-model judgement is started

# ---- retries: waits before a failed model call goes out again, and the most an episode spends waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45

# ---- sizes of what the model gets to see
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000

# ---- store-call allowances of the code's own work
GATE_ROUNDS = 3             # calls the process gate may spend before one checkout or ending
AUTO_PAGES = 6              # further result pages read for a cheapest / most / least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind opened before a cheapest checkout

# ---- clock (seconds from the start of agent_main; the validator stops a task at TASK_SECONDS)
TASK_SECONDS = 300.0
TIME_MARGIN = 20.0
CALL_FLOOR = 6.0
ASK_BY = 165.0              # the interview starts only before this
QUESTION_BY = 210.0         # no question to the shopper after this
WIND_DOWN = 200.0
LAST_TURN = 255.0
EMPTY_ROOM = 12.0
LATE_BY = 190.0             # the late question is sent only before this ...
LATE_STEPS = 8              # ... and while this many steps remain
SCARCE_MESSAGES = 2
LIST_SIZE = 8               # planned properties one message to the shopper may name

# ---- argument and request patterns (tool-schema words, not product vocabulary)
CAP_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
CAP_NOT = 'result|row|page|count|size|num|item|step'
SHELF_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MODE = 'field|filter|facet|brows|categor|list|structured'
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'

# ---- clients: model calls get one attempt each (the code retries itself); store calls may retry
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
        'You are RYU, a careful buyer working for one shopper in an online store.',
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
            'Keyword queries work best short and value-first (the wanted value plus the product noun).',
            'A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.',
        ),
        _sentences(
            'Search rows are summaries.',
            "Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either.",
            'The ledger names the tool that shows attributes once RYU knows it.',
            'Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need.',
            'Never open a listing again when nothing has changed since you read it.',
        ),
        _sentences(
            "Questions to the shopper are handled for you by RYU's interviewer, so you never message the shopper yourself.",
            'Its answers arrive in the conversation and are added to the ledger.',
        ),
        _sentences(
            'The request can also ask for steps along the way: an approval, a message, something to check or read first.',
            'The ledger lists them.',
            "RYU carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through RYU's interviewer).",
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
            "A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement.",
            'RYU adds [evidence] lines after opened listings with the keys whose names match a requirement.',
        ),
        _sentences(
            'When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts.',
            'A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.',
        ),
        "When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.",
        _sentences(
            'A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ...").',
            'Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value.',
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
            'Find the whole fitting set (page through it: result rows are often in no price order, and RYU reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits.',
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
        "Re-read a chosen listing's record shortly before checking out, because price and stock can change silently.",
    )),
    _lines('FINISHING', _bullets(
        'Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.',
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
        'Reading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value.',
        'Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget.',
        'The newest record of a listing wins.',
    ),
    "When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.",
    _sentences(
        'Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}.',
        'Use ok=false only for a problem the records clearly show or a requirement no record addresses.',
    ),
)


READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'


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
        'ROW: seen only as a search result row, its page not read).',
    ),
    'Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page, or ""}]}',
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



# ---- guards and key screens
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5} ?\\d{3,6}'
PART_CODE = '[A-Z]{1,4}\\d{3,6}'


# ---- fit codes: model numbers, "also sold as" names and what a listing says it works with
def _codes(text: Any) -> list[str]:
    found = re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))
    return [re.sub('\\s+', ' ', code).strip().lower() for code in found]


def _codes_in_hits(pattern: str, text: str) -> list[str]:
    """Every fit code inside the first group of each case-blind match of `pattern`."""
    gathered: list[str] = []
    for hit in re.finditer(pattern, text, re.I):
        gathered.extend(_codes(hit.group(1)))
    return gathered


def _upper_group(hit: Any) -> str:
    return hit.group(1).upper() if hit else ''


def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own_part = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    named = _codes_in_hits('\\bmodel\\s*(?:no\\.?|number|#)\\s*[:.]?\\s*(' + FIT_CODE + ')\\b', text)
    named += _codes_in_hits('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text)
    suits: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', text, re.I):
        lead_in = text[max(0, hit.start() - 40):hit.start()]
        # "part X also fits ..." is about another part, read below
        if re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', lead_in, re.I):
            continue
        suits.extend(_codes(hit.group(1)))
    other_parts = [(hit.group(1).upper(), _codes(hit.group(2)))
                   for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    pointer = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    return {'own': _upper_group(own_part), 'codes': named, 'works': suits, 'also': other_parts, 'more': _upper_group(pointer)}


# ---- text and JSON
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


_FENCED = re.compile('^```[A-Za-z]*\\s*(.*?)\\s*```$', re.S)


def _parsed_object(text: str) -> tuple[bool, dict[str, Any] | None]:
    """(whether `text` is valid JSON, the object it holds or None)."""
    try:
        got = json.loads(text)
    except ValueError:
        return False, None
    return True, got if isinstance(got, dict) else None


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


# ---- names: words of a key, folded plurals
def _words(key: str) -> str:
    spaced = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', spaced).strip().lower()


def _snake(key: str) -> str:
    return '_'.join(_words(key).split(' '))


def _stem(text: str) -> frozenset[str]:
    return frozenset(word[:-1] if len(word) > 3 and word.endswith('s') else word for word in _words(text).split())


def _singular(word: str) -> str:
    """A plural -s dropped (not from -ss words or short ones)."""
    return word[:-1] if len(word) > 3 and word.endswith('s') and not word.endswith('ss') else word


_ROOT_SKIP = ('and', 'for', 'the', 'with')


def _root(word: str) -> str:
    return word[:-2] if re.search('(?:ch|sh|ss|x|z)es$', word) else _singular(word)


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


# ---- tool schemas: kinds, and shaping arguments to fit them
def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [entry for entry in kind or [] if isinstance(entry, str)]


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


def _fits_kind(value: Any, kind: str) -> bool:
    test = _KIND_TEST.get(kind)
    return True if test is None else test(value)


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


# ---- small readers of model and store values
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
    """The key that identifies listing rows: id-like on every row and unique; keys named "id" first, then longer."""
    chosen, chosen_rank = None, None
    for key in rows[0]:
        named = 'id' in str(key).lower()
        raw = [row.get(key) for row in rows]
        if not all(_id_like(value, named) for value in raw):
            continue
        cleaned = [value.strip() for value in raw]
        if len(rows) > 1 and len(set(cleaned)) < len(cleaned):
            continue
        rank = (named, sum(map(len, cleaned)) / len(cleaned))
        if chosen_rank is None or rank > chosen_rank:
            chosen, chosen_rank = key, rank
    return chosen


def _row_holds(row: Any, ident: str) -> bool:
    return isinstance(row, dict) and ident in [str(text).strip() for text in row.values() if isinstance(text, str)]


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


def _scalars(block: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in block.items() if isinstance(value, (str, int, float)) and not isinstance(value, bool)}


def _fact_block(node: Any) -> dict[str, Any] | None:
    """The plain fields of the nested object with the most of them (two at least): a listing's attribute block."""
    if not isinstance(node, dict):
        return None
    blocks = [_scalars(value) for value in node.values() if isinstance(value, dict) and len(value) >= 2]
    blocks = [block for block in blocks if len(block) >= 2]
    return max(blocks, key=len) if blocks else None


# ---- store records sent as text (T1): "key: value | key: value | ... | facts: a=b; c=d" read as objects
_TEXT_PART = re.compile('([A-Za-z_][A-Za-z0-9_ ]{0,40}): (.*)', re.S)
_TEXT_PAIR = re.compile('([A-Za-z_][A-Za-z0-9_ ]{0,40})=(.*)', re.S)


def _typed_text(text: str) -> Any:
    """A text value typed the way JSON would type it: None, true/false, an integer or a decimal; else the text."""
    word = text.strip()
    if word in ('None', 'null'):
        return None
    if word in ('True', 'False'):
        return word == 'True'
    if re.fullmatch('-?\\d{1,15}', word):
        return int(word)
    if re.fullmatch('-?\\d{1,15}\\.\\d+', word):
        return float(word)
    return word


def _named_parts(text: str, joint: str, pattern: Any, least: int) -> list[tuple[str, str]] | None:
    """The (name, value) parts of a joined text, or None when it is not one. A piece that does not start with a name
    belongs to the value before it (the joint occurred inside that value)."""
    pieces = text.split(joint)
    if len(pieces) < least:
        return None
    parts: list[tuple[str, str]] = []
    for piece in pieces:
        hit = pattern.fullmatch(piece.strip())
        if hit:
            parts.append((hit.group(1).strip(), hit.group(2)))
        elif parts:
            parts[-1] = (parts[-1][0], parts[-1][1] + joint + piece)
        else:
            return None
    names = [name for name, _ in parts]
    if len(parts) < least or len(set(names)) < len(names):
        return None
    return parts


def _text_record(text: str) -> dict[str, Any] | None:
    """A listing record sent as one text line ("key: value | key: value | ..."), as an object; None for any other text
    (a record carries at least three fields, one of them an id-like value)."""
    if text.count(' | ') < 2:
        return None
    parts = _named_parts(text.strip(), ' | ', _TEXT_PART, 3)
    if not parts or not any(_id_like(value.strip()) for _, value in parts):
        return None
    record: dict[str, Any] = {}
    for name, value in parts:
        block = _named_parts(value, '; ', _TEXT_PAIR, 2) if '=' in value else None
        record[name] = {key: _typed_text(part) for key, part in block} if block else _typed_text(value)
    return record


def _readable(node: Any, depth: int = 0) -> Any:
    """A store reply with its text records turned into objects; everything else unchanged."""
    if depth > 8:
        return node
    if isinstance(node, str):
        record = _text_record(node)
        return node if record is None else record
    if isinstance(node, list):
        return [_readable(child, depth + 1) for child in node]
    if isinstance(node, dict):
        return {key: _readable(child, depth + 1) for key, child in node.items()}
    return node


class Item:
    """One requested product: its name, store shelf, quantity and firm needs, plus what the shopper said later."""

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name, self.shelf, self.qty, self.musts = name, shelf, qty, musts
        self.told: list[str] = []          # needs the shopper stated in answers
        self.dropped: list[str] = []       # needs the shopper gave up
        self.free: list[str] = []          # properties the shopper does not mind about


_B_WHEN_TEXT = {
    'choose': 'before choosing',
    'order': 'at the order (RYU does it when you check out)',
    'end': 'at an ending without a purchase (RYU does it then)',
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


def _b_named_tool(entry: Any) -> bool:
    """A tool entry the agent can call: a dict whose 'function' part carries a string name."""
    if not isinstance(entry, dict):
        return False
    function = entry.get('function')
    return isinstance(function, dict) and isinstance(function.get('name'), str)


def _b_params(function: dict[str, Any]) -> dict[str, Any]:
    """The tool's parameter schema, or an empty object schema when the tool gives none."""
    params = function.get('parameters')
    if isinstance(params, dict):
        return params
    return {'type': 'object', 'properties': {}}


def _b_positive(counted: int | None, fallback: int) -> int:
    """A counted value when it is a positive whole number, else the fallback."""
    if counted and counted > 0:
        return counted
    return fallback


def _b_cache_marked(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies of the messages with the first and the last text turned into cacheable blocks (Anthropic models)."""
    copied = [dict(m) for m in messages]
    for entry in (copied[0], copied[-1]):
        text = entry.get('content')
        if isinstance(text, str) and text:
            entry['content'] = [{'type': 'text', 'text': text, 'cache_control': {'type': 'ephemeral'}}]
    return copied


def _b_usable_brief(brief: dict[str, Any]) -> bool:
    """A brief names at least one item and a checkout tool."""
    if not _items(brief.get('items')):
        return False
    roles = brief.get('roles')
    return isinstance(roles, dict) and bool(roles.get('buy'))


def _b_item_from(entry: dict[str, Any]) -> Item:
    quantity = _b_positive(_count(entry.get('quantity')), 1)
    musts = [str(m) for m in _items(entry.get('musts')) if m]
    return Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), quantity, musts)


def _b_clash(first: str, second: str) -> bool:
    """Two requirements on the same property that ask for different values."""
    first_prop, _, first_value = first.partition(':')
    second_prop, _, second_value = second.partition(':')
    return _words(first_prop) == _words(second_prop) and _words(first_value) != _words(second_value)


def _b_money(value: Any) -> Any:
    """A positive budget figure from the brief (a numeric string becomes a float), else None."""
    if isinstance(value, str):
        if not re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
            return None
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if value > 0 else None


def _b_process_step(entry: Any) -> dict[str, Any] | None:
    """One process step of the brief in RYU's form, or None when it names no step."""
    if not isinstance(entry, dict):
        return None
    label = str(entry.get('step') or '').strip()
    if not label:
        return None
    stage = str(entry.get('when') or '').strip().lower()
    if stage not in ('choose', 'order', 'end'):
        stage = 'order'
    return {'step': label, 'when': stage, 'messages': max(0, _count(entry.get('messages')) or 0)}


def _b_flipped(before: dict[str, Any], after: dict[str, Any], was: bool, now: bool) -> bool:
    """Some flag that was `was` in the stored row is `now` in the new copy of it."""
    return any(before.get(key) is was and value is now for key, value in after.items())


_C_RANGE_WORDS = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b'


def _c_is_number(value: Any) -> bool:
    """True for an int or float; a bool is not a quantity."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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


def _c_echo_count(values: list[str]) -> int:
    """How many values share a 3+ letter word with at least one of the other values."""
    vocab = [set(re.findall('[a-z]{3,}', v)) for v in values]
    shared = 0
    for v, words in zip(values, vocab):
        rest: set[str] = set()
        for other, other_words in zip(values, vocab):
            if other is not v:
                rest |= other_words
        if words & rest:
            shared += 1
    return shared


def _c_noisy_values(values: list[str]) -> bool:
    """Values that make a poor attribute column: long text, mostly 'a / b' lists, or a free-text field whose values
    are all different and mostly share no words."""
    mean = sum(map(len, values)) / len(values)
    if mean > 40:
        return True
    if 2 * sum(' / ' in v for v in values) >= len(values):
        return True
    if len(values) < 4 or mean <= 12 or len(set(values)) != len(values):
        return False
    return 2 * _c_echo_count(values) < len(values)


def _c_fold_slot(into: dict[str, Any], extra: dict[str, Any]) -> None:
    """Merge a near-duplicate attribute column into the column already kept."""
    per, listed = into['per'], into['values']
    for ident, text in extra['per'].items():
        if ident not in per:
            per[ident] = text
        if text not in listed:
            listed.append(text)
    into['n'] = len(per)


def _c_values_agree(first: str, second: str) -> bool:
    """Two values agree when equal, or when the shorter one's words all occur in a not much longer other one."""
    if first == second:
        return True
    short = re.findall('[a-z0-9]+', first.lower())
    long = re.findall('[a-z0-9]+', second.lower())
    if len(long) < len(short):
        short, long = long, short
    return bool(short) and set(short) <= set(long) and len(long) <= 2 * len(short) + 2


def _c_price_cap(props: dict[str, Any], key: str, value: Any) -> bool:
    """A numeric argument the call sent as an upper price bound."""
    if key not in props or not _c_is_number(value):
        return False
    if not re.search('max|most|under|below|limit|cap|upto|up_to', key, re.I):
        return False
    spec = props.get(key) if isinstance(props.get(key), dict) else {}
    return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))


def _c_mode_hint(props: dict[str, Any], args: dict[str, Any], cap_key: str) -> str:
    """Suggest switching an enum mode argument (keeping the cap), so the tool lists only rows under the cap."""
    switches = {k: [o for o in props[k]['enum'] if o != v] for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
    if not any(switches.values()):
        return ''
    settings = ', '.join(f'{k}={_dump(rest[0])}' for k, rest in switches.items() if rest)
    return '; list the category with ' + settings + f' and the same {cap_key} to see only cheaper listings'


def _c_past_first_page(given: Any) -> bool:
    """The call already asked for a later page (a page number past 1, or any cursor text)."""
    if isinstance(given, int) and not isinstance(given, bool):
        return given not in (0, 1)
    return bool(str(given or '').strip())


def _c_cap_and_shelf(props: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The numeric max-price arguments and the free-text shelf arguments of a tool."""
    caps: list[str] = []
    shelves: list[str] = []
    for key, spec in props.items():
        if not isinstance(spec, dict):
            continue
        kinds = _kinds(spec)
        snake = _snake(key)
        if {'integer', 'number'} & set(kinds) and re.search(CAP_ARG, snake, re.I) and not re.search(CAP_NOT, key, re.I):
            caps.append(key)
        if 'string' in kinds and 'enum' not in spec and re.search(SHELF_ARG, snake, re.I):
            shelves.append(key)
    return caps, shelves


def _c_free_query(call: dict[str, Any], shelf: str, props: dict[str, Any]) -> bool:
    """The earlier call also carried free search words besides the shelf (so it was a search, not a listing)."""
    for key, value in call.items():
        if key == shelf or not isinstance(value, str) or not value.strip():
            continue
        spec = props.get(key)
        if isinstance(spec, dict) and 'enum' in spec:
            continue
        if re.search(QUERY_WORDS, _snake(key), re.I):
            return True
    return False


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


_d_recheck_clauses = ('The cart was not checked out:', 'check each line against them',
                      '(search again with the new values where needed),', 'fix the cart, then check out.')


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


def _d_label_fit(words, roots, item):
    """How well a category label names an item: exact shelf 3, exact name 2, else the overlap of their root words."""
    if words == _words(item.shelf or ''):
        return 3.0
    if words == _words(item.name):
        return 2.0
    theirs = _roots(item.shelf or '') | _roots(item.name)
    return len(roots & theirs) / max(1, len(roots | theirs))


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


_E_DID_STEPS = 'Before going on, RYU did the steps the request asks for:\n'


_E_SOLD_TAG = 'SOLD OUT (the store says it can no longer be bought). '


_E_LIMIT_WORDS = frozenset({'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max',
                            'minimum', 'maximum'})


_E_EVIDENCE_NOTE = ''.join((
    '[evidence: ',
    "keys whose names contain a requirement's property; ",
    'one naming a different property (another part or aspect) decides nothing, ',
    'and a key with another name may still hold the same property] ',
))


_E_FIT_NOTE = ''.join((
    '[fit plan: ',
    "a device fits a part when the part's text names the device's model code or another name it is sold as; ",
    "codes are not searchable, so list the device's category and read the codes on its rows] ",
))


_E_KNOWN_LINK = ': an opened page of the shop this chain names that shows the values its description gives; read the value from it'


_E_TRAIL_HOLD = (
    ' '.join((
        'The order counts only after every page a requirement points at has been opened in full',
        '(a search row is not enough), following each page that refers to a further listing.',
        'Still to find and open: ',
    )),
    ' '.join((
        ". Search for it by its shop's name together with the values its description gives",
        "(the shop's other listings of that kind are not it), open the row of that shop whose page shows those values,",
        'apply the value its page gives, then check out.',
    )),
)


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


def _e_tally_rank(entry: tuple[str, dict[int, int]]) -> tuple[int, int]:
    per_part = entry[1]
    return (-len(per_part), -sum(per_part.values()))


def _e_need_entries(value: Any) -> list[dict[str, Any]]:
    return [e for e in _items(value) if isinstance(e, dict) and str(e.get('need') or '').strip()]


def _e_first_key(props: dict[str, Any], test) -> str | None:
    return next((k for k, v in props.items() if isinstance(v, dict) and test(v)), None)


def _e_switch_modes(options: list[str]) -> tuple[str, str]:
    """The enum value that takes units off a line, and the one that puts them on."""
    off = next((o for o in options if re.search('remov|delet|drop|clear|take', o, re.I)), options[1])
    on = next((o for o in options if o != off), options[0])
    return (off, on)


def _e_chain_gap(links: list[dict[str, Any]], ids: list[str], rows: dict[str, Any]) -> tuple[str, str]:
    """What the first link of a chain that no row matches is about, and the words to search it by."""
    gap = next((link for link, ident in zip(links, ids) if ident not in rows), {})
    about = str(gap.get('about') or '') or 'the listing it points at'
    return (about, str(gap.get('find') or '').strip())


def _e_tool_line(spec: dict[str, Any]) -> str:
    params = _dump(spec.get('parameters') or {})
    return f"- {spec['name']}: {spec.get('description', '')} | parameters: {params}"


_f_seller_field = 'seller|shop|store|vendor|brand|maker|merchant'


_f_link_filler = frozenset({'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'})


_f_decline_text = 'Sorry, nothing in the store meets all of your requirements, ' + 'so I am not buying anything.'


def _f_word_in(word: str, text: str) -> bool:
    """`word` occurs in `text` with no letter or digit touching it on either side."""
    return re.search('(?<![a-z0-9])' + re.escape(word) + '(?![a-z0-9])', text) is not None


def _f_leaves(node: Any):
    """Every scalar inside a nested record, as text."""
    if isinstance(node, dict):
        node = list(node.values())
    if not isinstance(node, list):
        yield str(node)
        return
    for child in node:
        yield from _f_leaves(child)


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


def _g_skipped(reason):
    """Reply for a call that was turned away before it reached the store (no step spent)."""
    return 'Not run (no step used): ' + reason


def _g_held(reason):
    """Reply for an ending call that RYU holds back for now (no step spent)."""
    return 'Not run yet (no step used). ' + reason


def _g_tool_list(names):
    return ', '.join(sorted(names))


class Ryu:

    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self._b_read_view()
        self.started = monotonic()
        self._b_blank_state()

    def left(self) -> int:
        return self.max_steps - self.step

    def clock(self) -> float:
        return monotonic() - self.started

    def model_room(self) -> float:
        usable = TASK_SECONDS - TIME_MARGIN
        return usable - self.clock()

    def msgs_free(self) -> int:
        held_back = max(0, self.reserve - self.gate_msgs)
        return self.msg_limit - self.msg_used - held_back

    def cart_mark(self) -> str:
        lines = sorted(self.cart_lines.items())
        return _dump(lines)

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
        for _attempt in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                return self._b_landed(name, args, _readable(result.data))
            if not self._b_keep_trying(result, payload):
                return None
        return None

    def read_request(self) -> None:
        brief = self._b_fetch_brief(self._b_brief_prompt())
        self.brief = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.roles = {job: tool for job, tool in roles.items() if isinstance(tool, str) and tool in self.schemas}
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        self._b_split_modes(brief)
        self.items.extend(_b_item_from(entry) for entry in _items(brief.get('items')) if isinstance(entry, dict))
        became = self._b_merge_items()
        self._b_bind_fits(brief, became)
        for lookup in _items(brief.get('lookups')):
            entry = self._b_locate_lookup(lookup)
            if entry[1] and entry not in self.refs:
                self.refs.append(entry)
        self.unmix()
        self._b_take_limits(brief)
        self.process.extend(step for step in map(_b_process_step, _items(brief.get('process'))) if step)
        if self.roles.get('ask'):
            late = [entry['messages'] for entry in self.process if entry['when'] in ('order', 'end') and entry['messages']]
            self.reserve = min(max(late), self.msg_limit) if late else 0

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
        pieces = [f'[RYU] step {self.step} of {self.max_steps} used, {left} left.', self._b_pace_note(left)]
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
        groups = self._c_fact_groups(self.category_key())
        item_keys = self._c_item_keys()
        chosen = {label: group for label, group in groups.items() if self._c_label_wanted(label, item_keys)}
        table: list[tuple[str, str, int, list[str]]] = []
        for label, group in sorted((chosen or groups).items(), key=lambda entry: -entry[1]['n']):
            for name, slot in self._c_kept_columns(group['keys']):
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
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
        mine, theirs = set(name.split()), set(other.split())
        if not mine & theirs:
            return False
        left, right = spec['per'], slot['per']
        common = [i for i in left if i in right and left[i].strip() and right[i].strip()]
        agreeing = sum(_c_values_agree(left[i], right[i]) for i in common)
        if mine <= theirs or theirs <= mine:
            if common:
                return agreeing * 2 > len(common)
            return bool((set(spec['values']) & set(slot['values'])) - {''})
        return len(common) >= 2 and agreeing == len(common)

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
        raw = (self.rows.get(ident) or {}).get(self.price_key)
        if isinstance(raw, dict):
            raw = next((v for v in raw.values() if _c_is_number(v)), None)
        elif isinstance(raw, str):
            return float(raw) if re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', raw) else None
        return float(raw) if _c_is_number(raw) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        if not self.stock_key:
            return True
        stock = (self.rows.get(ident) or {}).get(self.stock_key)
        if stock is False:
            return False
        return not (_c_is_number(stock) and stock <= 0)

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, list):
            return self._c_rows_of(observation)
        if not isinstance(observation, dict):
            return []
        own = self.ident(observation)
        if own:
            return [(own, observation)]
        found: list[tuple[str, dict[str, Any]]] = []
        for value in observation.values():
            if isinstance(value, list):
                found.extend(self._c_rows_of(value))
                continue
            if isinstance(value, dict):
                for inner in value.values():
                    if isinstance(inner, list):
                        found.extend(self._c_rows_of(inner))
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
            return ' '.join((f'\n[RYU] This search matched {count} listings and showed {shown}, in no price order.',
                             "The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with " + f'{param}={_dump(cursor)}.'))
        texts, cursor = self._c_read_pages(name, args, param, cursor, pages)
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        if not texts:
            return ''
        report = '\n[RYU read the next pages of this search itself, one step a page]\n' + '\n'.join(texts)
        if cursor is not None:
            report += f'\n[RYU] {count} listings matched and RYU has read {shown * (len(texts) + 1)} rows of them; ' + "narrow the search with the requirement's value or page on to see the rest."
        return report

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.schemas.get(name) or {}).get('properties') or {}
        caps = [k for k, v in args.items() if _c_price_cap(props, k, v)]
        if not caps or not self.price_key:
            return ''
        tightest = sorted(caps, key=lambda k: args[k])[0]
        limit = args[tightest]
        over = [i for i, _ in self.records_in(observation) if (self.price_of(i) or 0) > limit]
        if not over:
            return ''
        return f'\n[RYU] This search did not apply {tightest}={limit:g}: {len(over)} of its rows cost more' + _c_mode_hint(props, args, tightest) + '.'

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
        return '\n[RYU] buyable rows here, cheapest first: ' + listed

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

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        queue = [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i)]
        texts: list[str] = []
        for ident in queue:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False:
                break
            page = self._c_open_page(tool, target, ident)
            if page is None:
                break
            texts.append(page)
        if texts:
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def show_links(self, ids: list[str], keep: int) -> None:
        # For carts of several lines: search each part code a cart line points to (once per code) and open the
        # pages that search finds while steps last. Only the ledger uses what comes back.
        plan = self.value_template()
        if not plan or (len(self.items) < 2 and len(ids) < 2):
            return
        tool, base, field = plan
        for ident in ids:
            code = self._c_link_code(ident)
            if code is None:
                continue
            if self.left() <= keep or self.finished:
                return
            self.link_tried.add(code)
            if not self._c_follow_link(tool, base, field, ident, code, keep):
                return

    def cap_template(self) -> tuple[str, dict[str, Any], str, str] | None:
        # The first tool with exactly one max-price argument and one shelf argument, plus the enum settings it
        # needs: the setting of an earlier pure listing by shelf, else the single option that names a listing mode.
        for tool, schema in self.schemas.items():
            props = (schema or {}).get('properties') or {}
            caps, shelves = _c_cap_and_shelf(props)
            if len(caps) == 1 and len(shelves) == 1:
                base = self._c_list_modes(props, shelves[0])
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
            label = self._c_label_of(line, cat_key)
            if price is None or price < 2 or not label or (line, price) in self.capped:
                continue
            if self.left() <= self.tail() + 5 or self.finished:
                return
            self.capped.add((line, price))
            ceiling = int(price) - 1 if float(price).is_integer() else int(price)
            call = _c_shaped(self.schemas.get(tool), {**base, shelf: label, cap: ceiling})
            if call is None:
                return
            reply = self.act(tool, call)
            if reply is None:
                return
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.trace.append({'kind': 'list_below', 'line': line, 'price': price})
            self._c_second_capped_page(tool, call, seen)

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
            fresh = [i for i in self._c_cheaper_than(line, cart, cat_key) if i not in picks]
            picks.extend(fresh)
        limit = min(6, CHEAPER_OPENS * max(1, len(cart)))
        return self.fill_facts(picks[:limit], self.tail() + 4)

    def observe(self, name: str, observation: Any) -> str:
        self.absorb(observation)
        lines = self.loose_lines(observation)
        meaning = self.take_note(lines) if lines else ''
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def widen(self) -> None:
        slot = self._c_compare_slot()
        if slot is None:
            self.widen_singly()
            return
        if self.left() <= 12:
            return
        tool, schema, target, spec = slot
        cap = spec.get('maxItems') if isinstance(spec.get('maxItems'), int) else 4
        ranked = self.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            unopened = [i for i in ids if i not in self.full]
            picks.extend(unopened[:share])
        picks = picks[:cap]
        if len(picks) < 2:
            return
        call = _c_shaped(schema, {target: picks})
        if call is None:
            return
        before = len(self.full)
        reply = self.act(tool, call)
        if reply is None:
            return
        seen = reply.get('observation')
        self.observe(tool, seen)
        self.note_view(tool, seen)
        rows = self.records_in(seen)
        if rows:
            self._c_mark_facts(tool, [record for _, record in rows])
        self.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(self.full) - before})
        if rows and self.facts_from.get(tool) is False:
            self.widen_singly()
        elif len(self.full) == before and len(picks) > 2 and self.left() > 12:
            self._c_try_call(tool, schema, {target: picks[:2]})

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
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0 or self.left() <= 8 or self.clock() > ASK_BY:
            return False
        return len(self.full) >= 3 * len(self.items) + 1 or self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        schema = self.schemas.get(self.roles.get('ask') or '') or {}
        target = self._c_text_slot(schema, False)
        if not target:
            return None
        props = schema.get('properties') or {}
        args: dict[str, Any] = {target: text}
        for key in schema.get('required') or []:
            if key in args:
                continue
            options = (props.get(key) or {}).get('enum')
            if not (isinstance(options, list) and options):
                return None
            args[key] = options[0]
        return _c_shaped(schema, args)

    def interview(self) -> str:
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.widen()
        self._c_open_lanes()
        told: list[str] = []
        sent = self._c_question_loop(told)
        answered = [n for n, found in self.lane_found.items() if found > 0]
        searched = self.value_search(answered) if answered else ''
        if told:
            self.version += 1
        return self._c_debrief(sent, told, searched)

    def late_question(self) -> str:
        """Use a message the checkout does not need for one more planned question just before buying.

        Messages the request keeps for a process step are left out of the interview. When the step belongs to an
        ending without a purchase and RYU is about to buy, those messages are spare. Returns a note for the loop
        model when the answer adds a firm requirement, otherwise ''."""
        if not self._d_late_allowed():
            return ''
        self.late_asked = True
        if not self.lanes:
            self._d_open_lanes()
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
        note = self._d_late_note(question, answer, told)
        extra = self.value_search([number])
        if extra:
            note += '\n' + extra
        return note

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
        """The search a value search copies: search_template() when it finds one; else, after category-only searches,
        the schema's single query field with at most one option field switched to its query mode. None on doubt."""
        plan = self.search_template()
        tool = self.roles.get('search')
        if plan or not tool or not self.searched:
            return plan
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        _, named = _d_text_fields(props)
        last = self.searched[-1]
        options = {k: v['enum'] for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and len(v['enum']) >= 2}
        modes = [mode for mode in (self._d_query_mode(k, opts, k in required, last) for k, opts in options.items()) if mode]
        if len(named) != 1 or len(modes) > 1 or (options and not modes):
            return None
        query = named[0]
        copied = [(k, last[k]) for k in required if k in last and k != query]
        return (tool, dict(copied + modes), query)

    def value_search(self, numbers: list[int]) -> str:
        plan = self.value_template()
        if not plan or self.left() < 22:
            return ''
        tool, base, field = plan
        seen: list[str] = []
        budget = 8
        for number in numbers:
            pairs = self.value_words(number)
            fresh = [v for v, told in pairs if told]
            if not fresh or budget <= 2:
                continue
            item = self.items[number - 1]
            for words in self._d_word_tries(pairs, fresh, item.name):
                faults: list[str] = []
                args = _shape({**base, field: ' '.join(words)}, self.schemas.get(tool), 'args', faults)
                if faults or self.finished:
                    break
                reply = self.act(tool, args)
                budget -= 1
                if reply is None:
                    break
                rows = self._d_fitting_rows(tool, reply.get('observation'), item, number)
                seen.append(f'{_dump(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                if not rows:
                    continue
                unread = [i for i in rows if i not in self.full]
                opened = self.fill_facts(unread[:min(3, budget)], self.tail() + 16)
                budget -= len(opened)
                if opened:
                    seen.append('opened: ' + ' | '.join(opened)[:2400])
                break
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        head = "[RYU searched each such item's requirement values together and read the first fitting-kind rows] "
        return head + '\n'.join(seen)

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

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        rank = {n: i for i, n in enumerate(self.plan_order)}
        even = self.msg_limit <= SCARCE_MESSAGES and not self.listing

        def priority(n: int) -> tuple[bool, int, int]:
            # items with no disclosed need first, then the least asked (unless messages are scarce), then plan order
            return (found.get(n, 0) > 0, 0 if even else spent.get(n, 0), rank.get(n, len(rank) + n))
        for number in sorted(lanes, key=priority):
            open_labels = [label for label in lanes[number] if self._d_unasked(number, label, used)]
            if open_labels:
                return (number, open_labels[:LIST_SIZE if self.listing else 1])
        return (0, [])

    def told_about(self, number: int, label: str) -> bool:
        """True when an answer already named this property of the item: the label and a told property name contain
        one another (a one-word told name only covers a one-word label), with plurals and colour/color folded."""
        asked = _spelled(label)
        props = (_spelled(t.partition(':')[0]) for t in self.items[number - 1].told)
        return any(_names(asked, prop) for prop in props)

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) > 1:
            return self.question(number, ', '.join(labels[:-1]) + ' or ' + labels[-1])
        only = labels[0]
        return self.question(number, only, self.samples(number, only) if shown else None)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """Drop labels from the end until the question passes the message schema unchanged (its length cap would
        otherwise cut the text); ([], '') when even one label does not fit."""
        kept = list(labels)
        while kept:
            text = self.ask_text(number, kept, shown)
            args = self.message_args(text)
            if args and text in args.values():
                return (kept, text)
            kept = kept[:-1]
        return ([], '')

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Ask the next question (a queued follow-up before the planned labels), take its answer into the items
        (each new need goes onto `told`) and settle the labels it named. Returns the sent batch, [] if none."""
        picked = self._d_pick_question()
        if picked is None:
            return []
        number, labels, text, shown = picked
        batch = self.send([(number, labels[0], text)], floor)
        if not batch:
            return []
        self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
        before = [len(item.told) for item in self.items]
        heard = self.hear(batch, (LIGHT, HEAVY))
        if len(labels) > 1:
            heard.pop('free', None)
        self.take_heard(heard, told)
        new = self._d_new_needs(before)
        crowded = 1 in [_count(x) for x in _items(heard.get('crowded'))]
        self.settle(number, labels, shown, new, crowded)
        return batch

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        """Record what the answer showed about the labels the question named.

        A single label is done; if the shopper asked for something narrower, a follow-up is queued (one product of a
        joined item name, the label's head word, or example values). For a list, the shopper answers about the first
        property it cares about, so a disclosed need settles that label and those before it and the later ones lead
        next time. A reply with nothing settles the first label only and queues the rest reversed, and every second
        such reply in a row the untried labels go first. A one-at-a-time reply turns lists off for the episode."""
        used = self.lane_used.setdefault(number, set())
        if len(labels) == 1:
            used.add(_words(labels[0]))
            if crowded and not new:
                retry = self._d_narrower(number, labels, shown, used)
                if retry:
                    self.follow_up.insert(0, retry)
            return
        if crowded and not new:
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': number})
            self.follow_up.insert(0, (number, labels[:1], False))
            return
        hit = self.answered_label(labels, new) if new else -1
        if hit < 0:
            ahead = labels[:0:-1]
            misses = self.lane_misses.get(number, 0) + 1
        else:
            ahead = labels[hit + 1:]
            misses = 0
        used.update(_words(label) for label in labels[:max(hit, 0) + 1])
        rest = [label for label in self.lanes.get(number) or [] if label not in labels]
        self.lane_misses[number] = misses
        self.lanes[number] = rest + ahead if misses and not misses % 2 else ahead + rest
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
        settled = [_stem(m.split(':', 1)[0]) for m in self.items[number - 1].musts]
        mine = [row for row in table if self.owner_of(row[0]) == number]
        if not mine and len(self.items) == 1:
            mine = list(table)
        numeric = self._d_numeric_keys(table)
        # most carried first, worded keys before measurement keys, then the keys with the most distinct values
        mine.sort(key=lambda row: (-row[2], numeric.get(row[1], False), -len(row[3])))
        keys = [row[1] for row in mine if not re.search(NOT_PROPERTY, row[1]) and not any(_names(prop, _stem(row[1])) for prop in settled)]
        return _d_catch_all_last(keys)

    def adopt_labels(self) -> None:
        cat_key = self.category_key()
        if len(self.items) < 2 or not cat_key:
            return
        labels = {_d_label(record, cat_key) for record in self.full.values()} - {''}
        owners = {label: self.owner_of(label) for label in labels}
        strays = [label for label, owner in owners.items() if not owner]
        claimed = {owner for label, owner in owners.items() if label not in self.adopted}
        orphans = [n for n in range(1, len(self.items) + 1) if n not in claimed]
        if not strays or len(orphans) != 1:
            return
        heir = orphans[0]
        self.adopted.update(dict.fromkeys(strays, heir))
        self.trace.append({'kind': 'adopted', 'item': heir, 'labels': strays})

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        if label in self.adopted:
            return self.adopted[label]
        words, roots = _words(label), _roots(label)
        fits = [_d_label_fit(words, roots, item) for item in self.items]
        best = max(fits, default=0.0)
        return fits.index(best) + 1 if best > 0 else 0

    def owns(self, item: Item, label: str) -> bool:
        if not label or item not in self.items:
            return False
        return self.owner_of(label) == self.items.index(item) + 1

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
        names = [f'item {pos}: {item.name}' for pos, item in enumerate(self.items, 1)]
        talk = [f'{k}. (about item {num}) Q: {question}\nA: {answer}' for k, (num, question, answer) in enumerate(sent, 1)]
        text = _e_sections(HEAR, [('ITEMS:\n', '\n'.join(names)), ('QUESTIONS AND ANSWERS:\n', '\n'.join(talk))])
        return self.ask_json(text, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        due = [p for p in self.process if p['when'] == stage]
        if not (due and not self.finished and self.gate_due(stage)):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [_e_tool_line(t['function']) for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            models = [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY]
            got = self.ask_json(self._e_gate_prompt(stage, due, tools), models, 3500)
            if got is None:
                break
            told, hold = self._e_gate_heard(got, said)
            call = got.get('call')
            if not isinstance(call, dict):
                call = None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.version += 1
                return (False, self._e_gate_changed(said, told, hold))
            name = str((call or {}).get('tool') or '')
            unproven = self._e_unproven(got, stage, due)
            if not call and self._e_may_nudge(unproven, said, nudged):
                nudged = True
                self._e_gate_nudge(unproven)
                continue
            if not call:
                self.gate_ok.add((stage, self._e_versioned(self.cart_mark())))
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
            went = self._e_gate_ask(args, said) if asking else self._e_gate_call(name, args, said)
            if not went:
                break
        return (True, self._e_did(said))

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
        ids = self._e_fact_ids(observation)
        needs = self.needs_now()
        if not ids or not needs:
            return ''
        many = len(self.items) > 1
        lines = [self._e_evidence_line(ident, needs[:8], many) for ident in ids[:12]]
        lines = [line for line in lines if line]
        return _E_EVIDENCE_NOTE + ' | '.join(lines) if lines else ''

    def cart_text(self) -> str:
        if not self._e_view_fresh():
            return "lines by RYU's record of the cart calls (listing: units): " + _dump(self.cart_lines)
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
        marks = {ident: _fit_marks(self._e_fit_text(ident)) for ident in self.seen_order}
        notes: dict[str, set[str]] = {}
        for mark in marks.values():
            for part, codes in mark['also']:
                notes.setdefault(part, set()).update(codes)
        return {'marks': marks, 'notes': notes}

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

    def fit_faults(self) -> list[str]:
        if not self.bound:
            return []
        index = self.fit_index()
        cart = self.cart_ids()
        faults: list[str] = []
        for part, main in self.bound:
            faults += self._e_pair_faults(part, main, index, cart)
        return faults

    def fit_plan(self, short: bool=False) -> str:
        if not self.bound:
            return ''
        index = self.fit_index()
        mains = sorted({m for _, m in self.bound})
        lines = [line for line in (self._e_main_plan(main, index) for main in mains) if line]
        if not lines:
            return ''
        joined = ' | '.join(lines)
        return joined if short else _E_FIT_NOTE + joined

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

    def line_faults(self) -> list[str]:
        self.adopt_labels()
        cat_key = self.category_key()
        ids = self._e_held_ids()
        if not ids or not cat_key:
            return []
        label = {i: self._e_label(i, cat_key) for i in ids}
        shelf = [str(r.get(cat_key) or '') for r in self.rows.values()]
        known = {n for n, item in enumerate(self.items, 1) if self._e_owns_any(item, shelf)}
        owner = {i: [n for n, item in enumerate(self.items, 1) if self.owns(item, label[i])] for i in ids}
        faults: list[str] = []
        for n, item in enumerate(self.items, 1):
            if n in known:
                faults += self._e_item_lines(n, item, ids, label, owner)
        if len(known) == len(self.items):
            faults += [f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out" for i in ids if not owner[i]]
        faults += [f'{i} is sold out' for i in ids if self.sold_out(i)]
        return faults + self._e_budget_faults(ids)

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

    def trail(self) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.single_target(tool)
        can_open = bool(tool and target)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self._e_trail_room():
            self.trails += 1
            records = self._e_trail_records()
            prompt = _e_sections(TRAIL, [('REQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n', '\n'.join(pending)), ('RECORDS (newest first):\n', '\n'.join(records))])
            got = self.ask_json(prompt, self._e_trail_models(), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach, lost, gaps = self._e_follow_chains(got, pending)
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and can_open and self.left() > 12 and self.link_searches < 3:
                reach, known = self._e_link_search(gaps[0], seen)
                if known:
                    lost = lost[1:]
            if not reach or not can_open or self.left() < len(reach) + 4 or self.finished:
                break
            self._e_open_pages(tool, target, reach[:max(0, min(3, 6 - len(seen)))], seen)
        return self._e_trail_text(seen, lost)

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        text = about.lower()
        return [ident for ident in ids if ident not in self.opened and self._f_sells_here(ident, text)]

    def opened_shop(self, about: str) -> list[str]:
        text = about.lower()
        return [ident for ident in self.opened if self._f_sells_here(ident, text)]

    def find_link(self, about: str, words: str) -> list[str]:
        clue = (words or about).lower()
        seller = self._f_shop_words(about)
        values = {w for w in re.findall('[a-z0-9]+', clue) if len(w) >= 3 or w.isdigit()} - seller - _f_link_filler
        if self._f_link_opened(about, values):
            return []
        found = self._f_link_lookup(about, words)
        kind = set(re.findall('[a-z0-9]{3,}', clue)) - seller - _f_link_filler
        return (self._f_of_kind(kind, found) or self._f_of_kind(kind, self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

    def offer_tool(self) -> str | None:
        page = self.roles.get('offer')
        return page if page else self.roles.get('open')

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not (tool and target):
            return ''
        stale = self._f_stale_offers()
        if not stale or self.left() < len(stale) + 3:
            return ''
        notes = []
        for ident in stale:
            note = self._f_reread_one(tool, target, ident)
            if note is None:
                break
            if note:
                notes.append(note)
        self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(notes)})
        if notes:
            self.version += 1
        return '\n'.join(notes)

    def sold_out(self, ident: str) -> bool:
        return any((ident in pool for pool in (self.unavailable, self.gone)))

    def swap_hint(self) -> str:
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ''
        self._f_retire(lost)
        picks = self.replacements(lost)
        head = '[RYU] Sold out now: ' + ', '.join(lost) + ' (the store shows it cannot be bought). It will not come back, so do not open it again.'
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
        records = [self._f_audit_record(i) for i in self._f_newest(30, self.sold_out)]
        if not records:
            return None
        prompt = self._f_prompt_head(AUDIT) + _f_section('LISTING RECORDS SEEN (newest first)', '\n'.join(records))
        judges = self.judges()
        got = self.ask_json(prompt, [LIGHT] if judges else [], 4000)
        second = HEAVY in judges and not self.no_credit and not self.audit_fits(got)
        if second:
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
        plan = self._f_open_plan(rows)
        if plan is None or self.left() < len(plan) + 5:
            return False
        before = len(self.full)
        if not self._f_run_opens(plan):
            return False
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
        return 'Not run (RYU acted instead): the independent check confirmed ' + self.confirmed + ', so RYU set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        return self._f_role_call('buy', 'buy_args', name, args)

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        return self._f_role_call('refuse', 'refuse_args', name, args)

    def offered(self) -> list[dict[str, Any]]:
        left = self.left()
        ask = self.roles.get('ask')
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        extra = self._f_closing_roles(left)
        tools = [t for t in self.tools if t['function']['name'] != ask]
        if extra is None or not ends:
            return tools
        allowed = ends | {self.roles.get(r) for r in extra}
        return [t for t in tools if t['function']['name'] in allowed] or tools

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
        rendered = _dump(_clip(observation, SHOW_CHARS))
        return rendered if len(rendered) <= OBS_LIMIT else rendered[:OBS_LIMIT] + '...(cut)'

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
        calls = self.join_opens(calls)
        return [(call['id'], self._g_answer(call, len(calls))) for call in calls]

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
        opening = _readable(self.view.get('opening'))
        if opening is not None:
            self.absorb(opening)
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': self._g_intro(opening)}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            self._g_wrap_up()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while self._g_may_continue(loop_cap, spend_cap):
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
            calls = [self._g_tidy_call(n, c) for n, c in enumerate(message.get('tool_calls') or []) if isinstance(c, dict)]
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
            asking = self._g_ask_first(calls)
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            if asking:
                self._g_interview_instead(calls)
            else:
                self._g_play_turn(calls, content)

    def _b_read_view(self) -> None:
        """The request text, the callable tools with their schemas and the step limit from the policy view."""
        view = self.view
        self.request = str(view.get('query') or '')
        self.tools = [entry for entry in view.get('tools') or [] if _b_named_tool(entry)]
        self.schemas = {entry['function']['name']: _b_params(entry['function']) for entry in self.tools}
        raw = view.get('max_steps')
        counted = _count(raw)
        if counted is None and isinstance(raw, float):
            counted = int(raw)
        self.max_steps = _b_positive(counted, 30)
        self.finished = bool(view.get('finished'))

    def _b_blank_state(self) -> None:
        # step, call and spend counters
        self.step = self.waited = self.misses = 0
        self.calls = self.loop_calls = 0
        self.cost = 0.0
        self.broke = self.no_credit = False
        self.trace, self.chat, self.recent = [], [], []
        # what the request asks for
        self.brief, self.roles = {}, {}
        self.items, self.process, self.refs, self.bound = [], [], [], []
        self.budget = None
        self.msg_limit = self.msg_used = self.reserve = self.gate_msgs = 0
        self.ref_values = {}
        # the shopper interview
        self.asked = self.late_asked = False
        self.qa, self.heard = [], set()
        self.plan_order, self.follow_up = [], []
        self.lanes, self.lane_found, self.lane_spent, self.lane_used, self.lane_misses = {}, {}, {}, {}, {}
        self.one_product = set()
        self.listing = True
        # process steps and their checks
        self.gate_ok = set()
        self.done_log, self.done_steps = [], []
        # listings seen and read
        self.rows, self.full, self.seen_order = {}, {}, []
        self.opened, self.unavailable, self.checked = set(), set(), set()
        self.id_key = self.price_key = self.stock_key = None
        self.facts_from, self.tool_keys, self.view_keys, self.read_cache = {}, {}, {}, {}
        self.shapes, self.adopted, self.offer_seen, self.gone = {}, {}, {}, {}
        self.pages_read = 0
        self.paged_obs, self.searched = [], []
        self.cheaper_marks, self.capped, self.id_args = set(), set(), set()
        self.wider_held = False
        self.link_tried, self.link_seen, self.link_searches = set(), {}, 0
        # notes from the store and the cart
        self.notes, self.built, self.queued, self.leads = [], [], [], []
        self.version = 0
        self.cart_touched = self.note_named = self.note_sold = False
        self.cart_at = self.cart_seen_at = self.refused_at = -1
        self.cart_view = None
        self.cart_lines = {}
        self.cart_said = ''
        self.sold_lines, self.dropped = set(), set()
        self.lead_items, self.verdicts = {}, {}
        self.confirmed = None
        # how the episode is ending
        self.held = self.refuse_blocks = self.buys = self.refusals = 0
        self.trails = self.trail_blocks = self.compare_ok = self.compare_bad = 0
        self.line_holds = self.fit_holds = 0
        self.audited = self.pushed = self.late_held = self.completed = False

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

    def _b_landed(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Record an accepted store call: the step counter, the end flag and the listings a page tool opened."""
        self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
        if data.get('finished'):
            self.finished = True
        seen = data.get('observation')
        if name in self.page_tools() and not (isinstance(seen, dict) and seen.get('error')):
            shown = [piece.strip() for value in args.values() for piece in (value if isinstance(value, list) else [value]) if isinstance(piece, str)]
            self.opened.update(shown)
        return data

    def _b_keep_trying(self, result: Any, payload: dict[str, Any]) -> bool:
        """After a refused store call: trace it and say whether sending it again can help."""
        fault = result.error or {}
        status = fault.get('status')
        body = str(fault.get('body') or '')
        self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
        if status in (400, 401, 402, 403, 404, 422):
            if status == 402:
                self.broke = True
            return False
        if status != 409:
            return True
        return self._b_resync(body, payload)

    def _b_resync(self, body: str, payload: dict[str, Any]) -> bool:
        """A step conflict: the episode is over, or the step number is put right for one more try."""
        low = body.lower()
        if any(word in low for word in ('finish', 'quarantin', 'finaliz')):
            self.finished = True
            return False
        expected = re.search('expected (\\d+)', body)
        if expected:
            self.step = int(expected.group(1)) - 1
        elif 'reuse' in low or 'different action' in low:
            self.step += 1
        else:
            return False
        payload['step'] = self.step + 1
        return True

    def _b_brief_prompt(self) -> str:
        described = []
        for entry in self.tools:
            function = entry['function']
            described.append(f"- {function['name']}: {function.get('description', '')}"
                             f" | parameters: {_dump(function.get('parameters') or {})}")
        parts = [BRIEF, '\nTOOLS:\n', '\n'.join(described), '\n\nREQUEST:\n', self.request]
        opening = _readable(self.view.get('opening'))
        if opening is not None:
            parts += ['\n\nOPENING NOTE FROM THE STORE:\n', _dump(_clip(opening, 1500))]
        return ''.join(parts)

    def _b_fetch_brief(self, prompt: str) -> dict[str, Any]:
        """The light model reads the request; the heavy one reads it again when that reading is unusable or puts
        two items on one shelf."""
        first = self.ask_json(prompt, [LIGHT], 4000) or {}
        if self.no_credit:
            return first
        if not _b_usable_brief(first):
            return self.ask_json(prompt, [HEAVY, LIGHT], 2500) or first
        if not self.shared_shelf(first):
            return first
        second = self.ask_json(prompt, [HEAVY], 2500) or {}
        return second if _b_usable_brief(second) else first

    def _b_split_modes(self, brief: dict[str, Any]) -> None:
        """When one tool both checks out and declines, make sure the brief knows which arguments pick each mode."""
        tool = self.roles.get('buy')
        if not tool or tool != self.roles.get('refuse'):
            return
        checkout, ending = brief['buy_args'], brief['refuse_args']
        if checkout and ending and checkout != ending:
            return
        question = (f'The store tool {tool} both checks out and ends without a purchase, depending on its arguments. Its schema: '
                    + _dump(self.schemas.get(tool))
                    + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).')
        answer = self.ask_json(question, [LIGHT, HEAVY], 1500) or {}
        checkout, ending = answer.get('buy_args'), answer.get('refuse_args')
        if isinstance(checkout, dict) and isinstance(ending, dict) and checkout != ending:
            brief['buy_args'], brief['refuse_args'] = checkout, ending

    def _b_merge_items(self) -> dict[int, int]:
        """Fold items that share a shelf and do not contradict each other; returns old number -> new number."""
        kept: list[Item] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.items, 1):
            slot = next((pos for pos, other in enumerate(kept, 1) if self._b_same_item(other, item)), 0)
            if not slot:
                kept.append(item)
                became[old] = len(kept)
                continue
            became[old] = slot
            host = kept[slot - 1]
            extra = [must for must in item.musts if must not in host.musts]
            host.musts.extend(extra)
            self.trace.append({'kind': 'items_merged', 'into': host.name, 'dropped': item.name})
        self.items = kept or [Item('the requested product', '', 1, [])]
        return became

    @staticmethod
    def _b_same_item(host: Item, item: Item) -> bool:
        if not item.shelf.strip() or _words(host.shelf) != _words(item.shelf):
            return False
        return not any(_b_clash(a, b) for a in host.musts for b in item.musts)

    def _b_bind_fits(self, brief: dict[str, Any], became: dict[int, int]) -> None:
        """Pairs (part, main) where one requested item must fit another, in merged item numbers."""
        total = len(self.items)
        for pair in _items(brief.get('fits')):
            if not isinstance(pair, dict):
                continue
            edge = (became.get(_count(pair.get('item'))), became.get(_count(pair.get('with'))))
            part, main = edge
            if not (part and main) or part == main or min(edge) < 1 or max(edge) > total:
                continue
            if edge not in self.bound:
                self.bound.append(edge)

    def _b_locate_lookup(self, raw: Any) -> tuple[int, str]:
        """A lookup of the brief as (item number, the must it is): an exact must, else the first must on the same
        property, else the text itself on item 1."""
        text = str(raw).strip()
        for number, item in enumerate(self.items, 1):
            if text in item.musts:
                return number, text
        prop = _words(text.split(':')[0])
        for number, item in enumerate(self.items, 1):
            same = [must for must in item.musts if _words(must.split(':')[0]) == prop]
            if same:
                return number, same[0]
        return 1, text

    def _b_take_limits(self, brief: dict[str, Any]) -> None:
        money = _b_money(brief.get('budget'))
        if money is not None:
            self.budget = money
        messages = _count(brief.get('messages'))
        if messages is None or messages < 0:
            if self.roles.get('ask'):
                self.msg_limit = 3
        else:
            self.msg_limit = messages
        steps = _count(brief.get('steps'))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps

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
        """Which tool shows listing attributes, once RYU knows it."""
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

    def _c_label_of(self, ident: str, cat_key: str) -> str:
        return str((self.rows.get(ident) or {}).get(cat_key) or '')

    def _c_fact_groups(self, cat_key: str | None) -> dict[str, dict[str, Any]]:
        # opened records grouped by category; per group the attribute columns with each record's value
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            self._c_add_facts(group['keys'], ident, _fact_block(record) or {})
        return groups

    def _c_add_facts(self, columns: dict[str, dict[str, Any]], ident: str, block: dict[str, Any]) -> None:
        taken: set[str] = set()
        for key, value in block.items():
            name = _words(key)
            if not name or name in taken:
                continue
            taken.add(name)
            column = columns.setdefault(name, {'n': 0, 'values': [], 'per': {}})
            column['n'] += 1
            text = str(value).strip().lower()
            column['per'][ident] = text
            if text not in column['values']:
                column['values'].append(text)

    def _c_kept_columns(self, columns: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        # the usable attribute columns of a group, near-duplicates folded together, most common first, at most 14
        kept: list[tuple[str, dict[str, Any]]] = []
        for name, column in sorted(columns.items(), key=lambda entry: (-entry[1]['n'], len(entry[0]))):
            if _c_noisy_values(list(column['per'].values())):
                continue
            twin = next((spec for other, spec in kept if self.twins(other, spec, name, column)), None)
            if twin is None:
                kept.append((name, column))
            else:
                _c_fold_slot(twin, column)
        kept.sort(key=lambda entry: -entry[1]['n'])
        return kept[:14]

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

    def _c_rows_of(self, seq: list[Any]) -> list[tuple[str, dict[str, Any]]]:
        found = []
        for record in seq:
            if isinstance(record, dict):
                ident = self.ident(record, True)
                if ident:
                    found.append((ident, record))
        return found

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

    def _c_mark_facts(self, tool: str, records: list[Any]) -> None:
        # remember whether this tool's pages carry attribute blocks (once True it stays True)
        self.facts_from[tool] = self.facts_from.get(tool, False) or any(_fact_block(r) for r in records)

    def _c_open_page(self, tool: str, target: str, ident: str) -> str | None:
        call = _c_shaped(self.schemas.get(tool), {target: ident})
        if call is None:
            return None
        reply = self.act(tool, call)
        if reply is None:
            return None
        seen = reply.get('observation')
        self.observe(tool, seen)
        self.note_view(tool, seen)
        if isinstance(seen, dict) and self.ident(seen):
            self._c_mark_facts(tool, [seen])
        return f'{ident}: ' + self.show(seen)[:1500]

    def _c_link_code(self, ident: str) -> Any:
        # the code a line refers to, when it is new and no seen listing already carries it as its own
        marks = self.fit_index()['marks']
        code = (marks.get(ident) or {}).get('more')
        if not code or code in self.link_tried:
            return None
        if any(mark.get('own') == code for mark in marks.values()):
            return None
        return code

    def _c_follow_link(self, tool: str, base: dict[str, Any], field: str, ident: str, code: Any, keep: int) -> bool:
        call = _c_shaped(self.schemas.get(tool), {**base, field: code})
        if call is None:
            return False
        reply = self.act(tool, call)
        if reply is None:
            return False
        self.observe(tool, reply.get('observation'))
        self.trace.append({'kind': 'show_link', 'line': ident, 'code': code})
        owners = [i for i, mark in self.fit_index()['marks'].items() if mark.get('own') == code]
        self.fill_facts(owners, max(keep, self.tail() + 6))
        return True

    def _c_list_modes(self, props: dict[str, Any], shelf: str) -> dict[str, Any] | None:
        settings: dict[str, Any] = {}
        for key, spec in props.items():
            options = spec.get('enum') if isinstance(spec, dict) else None
            if not isinstance(options, list) or len(options) < 2:
                continue
            mode = next((call[key] for call in reversed(self.searched) if shelf in call and key in call and not _c_free_query(call, shelf, props)), None)
            if mode is None:
                listing = [o for o in options if isinstance(o, str) and re.search(LIST_MODE, o, re.I)]
                if len(listing) != 1:
                    return None
                mode = listing[0]
            settings[key] = mode
        return settings

    def _c_try_call(self, tool: str, schema: Any, args: dict[str, Any]) -> None:
        # one extra call whose result only feeds what Halo has seen
        call = _c_shaped(schema, args)
        reply = None if call is None else self.act(tool, call)
        if reply is not None:
            self.observe(tool, reply.get('observation'))

    def _c_second_capped_page(self, tool: str, call: dict[str, Any], seen: Any) -> None:
        # a capped listing with just one more page left (and not already in price order): read that page too
        page = self.paging(tool, seen)
        if not page or page[3] is None:
            return
        total, shown, arg, cursor = page
        if not shown < total <= 2 * shown or self.left() <= self.tail() + 5 or self.price_sorted(seen):
            return
        self._c_try_call(tool, self.schemas.get(tool), {**call, arg: cursor})

    def _c_price_or_inf(self, ident: str) -> float:
        price = self.price_of(ident)
        return float('inf') if price is None else price

    def _c_cheaper_than(self, line: str, cart: list[str], cat_key: str) -> list[str]:
        # the two cheapest unopened buyable listings of the line's category priced below it
        price = self.price_of(line)
        kind = self._c_label_of(line, cat_key).strip().lower()
        if price is None or not kind:
            return []
        pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and self._c_label_of(i, cat_key).strip().lower() == kind and self._c_price_or_inf(i) < price]
        pool.sort(key=lambda i: self.price_of(i) or 0.0)
        return pool[:2]

    def _c_compare_slot(self) -> tuple[str, dict[str, Any], str, Any] | None:
        # the compare tool whose only required argument is a list of ids, unless comparing shows no attributes
        tool = self.roles.get('compare')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not tool or not target:
            return None
        if any(k != target for k in schema.get('required') or []) or self.compare_bare():
            return None
        return (tool, schema, target, props[target])

    def _c_lane(self, ids: list[str], wanted: set[str], cheapest: bool) -> list[str]:
        # unopened buyable listings of one group, those sharing the most requirement words first (cheapest
        # first among equals when the pick is the cheapest)
        pool = [i for i in ids if i not in self.full and self.buyable(i)]
        if cheapest:
            pool.sort(key=self._c_price_or_inf)
        pool.sort(key=lambda i: -len(wanted & _c_alnum3(_dump(self.rows.get(i) or {}))))
        return pool

    def _c_open_lanes(self) -> None:
        room = self.msg_limit - self.msg_used
        self.lanes = self.plan_lanes(max(room, LIST_SIZE) if self.listing else room)
        self.lane_found = dict.fromkeys(self.lanes, 0)
        self.lane_spent = dict.fromkeys(self.lanes, 0)
        self.lane_used = {n: set() for n in self.lanes}
        self.follow_up = []

    def _c_question_loop(self, told: list[str]) -> list[tuple[int, str, str]]:
        sent: list[tuple[int, str, str]] = []
        while self.msgs_free() > 0 and not self.finished and self.left() > 6 and self.clock() < QUESTION_BY:
            batch = self.ask_next(told)
            if not batch:
                break
            sent.extend(batch)
        return sent

    def _c_debrief(self, sent: list[tuple[int, str, str]], told: list[str], searched: str) -> str:
        # the interview report for the main model: questions and answers, new requirements, and follow-up notes
        exchange = '\n'.join(f'Q: {q}\nA: {a}' for _, q, a in sent)
        parts = ["RYU's interviewer asked the shopper (each question used one step):\n" + exchange,
                 '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none'),
                 '. Apply them to every pick (search again with the new values where needed).']
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            parts.append(' These take their value from another listing: ' + ' | '.join(pointed))
            parts.append('. Find that listing, open its page in full, and open every listing its page refers to; ')
            parts.append('the last page gives the value, and the order counts only after all of them are opened.')
        if searched:
            parts.append('\n' + searched)
        if self.queued:
            parts.append('\n[notes seen meanwhile] ' + ' | '.join(self.queued))
            self.queued = []
        return ''.join(parts)

    def _d_late_allowed(self) -> bool:
        if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit:
            return False
        if any(step['when'] == 'order' and step['messages'] > 0 for step in self.process):
            return False   # the order itself needs the spare messages
        return self.left() >= LATE_STEPS and self.clock() <= LATE_BY and bool(self.judges())

    def _d_open_lanes(self) -> None:
        self.lanes = self.plan_lanes(LIST_SIZE if self.listing else 1)
        self.lane_found = dict.fromkeys(self.lanes, 0)
        self.lane_spent = dict.fromkeys(self.lanes, 0)
        self.lane_used = {n: set() for n in self.lanes}

    def _d_late_note(self, question: str, answer: str, told: list[str]) -> str:
        found = 'New firm requirements from this answer: ' + '; '.join(told) + '. ' + ' '.join(_d_recheck_clauses)
        lines = ['RYU asked the shopper one more question before checking out (one step):', f'Q: {question}', f'A: {answer}', found]
        note = '\n'.join(lines)
        linked = [need for _, need in self.refs if any(need in t for t in told)]
        if linked:
            note += (' These take their value from another listing: ' + ' | '.join(linked)
                     + '. Find that listing, open its page in full, and open every listing its page refers to; '
                     + 'the last page gives the value.')
        return note

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

    def _d_query_mode(self, key: str, options: list[Any], needed: bool, last: dict[str, Any]) -> tuple[str, Any] | None:
        """The option that switches a mode field to reading words: the one option named like a query, else the other
        side of a required two-way switch."""
        reads = [o for o in options if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)]
        if len(reads) == 1:
            return (key, reads[0])
        rest = [o for o in options if o != last.get(key)]
        if needed and len(options) == 2 and len(rest) == 1:
            return (key, rest[0])
        return None

    def _d_word_tries(self, pairs: list[tuple[str, bool]], fresh: list[str], name: str) -> list[list[str]]:
        """The (at most two) word lists to search: all values with the short answered ones first, then the same
        without the last request value (or without the first value when every value came from answers). A single
        short value is first tried with the item name."""
        lead = [v for v in fresh if len(v.split()) == 1 or re.fullmatch(AMOUNT, v, re.I)]
        order = lead + [v for v, _ in pairs if v not in lead]
        if len(order) == 1:
            return [[order[0], name], order] if lead else [order]
        cut = max((i for i, v in enumerate(order) if v not in fresh), default=0)
        return [order, order[:cut] + order[cut + 1:]]

    def _d_fitting_rows(self, tool: str, observation: Any, item: Item, number: int) -> list[str]:
        """Read a value search's result and return its buyable rows of the item's kind within the budget, rows on the
        item's own category label first."""
        self.observe(tool, observation)
        cat_key = self.category_key()
        found = [i for i, _ in self.records_in(observation)]
        cap = self._d_price_cap(found)
        rows = [i for i in found if self.buyable(i)
                and (not cat_key or self.owns(item, _d_label(self.rows.get(i), cat_key)))
                and (cap is None or (self.price_of(i) or 0) <= cap)]
        home = self.home_label(number) if cat_key else ''
        if not home:
            return rows
        at_home = [i for i in rows if _words(_d_label(self.rows.get(i), cat_key)) == home]
        return at_home + [i for i in rows if i not in at_home]

    def _d_price_cap(self, found: list[str]) -> Any:
        cap = self.budget
        if not isinstance(cap, (int, float)):
            return None
        prices = [self.price_of(i) for i in found]
        if prices and all(p is not None and p > 10 * cap for p in prices):
            return None   # every price far above the budget: the budget is not in the listing's unit
        return cap

    def _d_tally(self, item: Item, pool: dict[str, Any], cat_key: str) -> dict[str, int]:
        tally: dict[str, int] = {}
        for label in (_d_label(record, cat_key) for record in pool.values()):
            if label and self.owns(item, label):
                key = _words(label)
                tally[key] = tally.get(key, 0) + 1
        return tally

    def _d_unasked(self, number: int, label: str, used: dict[int, set[str]]) -> bool:
        return _words(label) not in used[number] and not self.told_about(number, label)

    def _d_pick_question(self) -> tuple[int, list[str], str, bool] | None:
        """The next labels and their question text, skipping labels whose question cannot fit the message tool."""
        for _attempt in range(LIST_SIZE):
            number, labels, shown = self._d_next_labels()
            if not labels:
                return None
            fitted, text = self.fit_ask(number, labels, shown)
            if shown and not fitted:
                fitted, text = self.fit_ask(number, labels, False)
            if fitted:
                return (number, fitted, text, shown)
            self.lane_used.setdefault(number, set()).add(_words(labels[0]))
            self.trace.append({'kind': 'ask_too_long', 'item': number, 'label': labels[0]})
        return None

    def _d_next_labels(self) -> tuple[int, list[str], bool]:
        self.follow_up = [f for f in self.follow_up if not self.told_about(f[0], f[1][0])]
        if self.follow_up:
            return self.follow_up.pop(0)
        number, labels = self.next_ask(self.lanes, self.lane_found, self.lane_spent, self.lane_used)
        return (number, labels, False)

    def _d_new_needs(self, before: list[int]) -> list[str]:
        """Needs the last answer added, in item order; each item that gained one counts a found answer."""
        new: list[str] = []
        for n, (item, size) in enumerate(zip(self.items, before), 1):
            gained = item.told[size:]
            if gained:
                self.lane_found[n] = self.lane_found.get(n, 0) + 1
                new.extend(gained)
        return new

    def _d_narrower(self, number: int, labels: list[str], shown: bool, used: set[str]) -> tuple[int, list[str], bool] | None:
        """The follow-up for a single-label question the shopper found too broad, or None."""
        if number not in self.one_product and self.name_parts(number):
            self.one_product.add(number)
            self.trace.append({'kind': 'one_product', 'item': number, 'said': self.item_said(number)})
            return (number, labels, shown)
        words = labels[0].split()
        head = words[-1]
        if len(words) >= 2 and _words(head) not in used:
            return (number, [head], False)
        if not shown and len(self.samples(number, labels[0])) >= 2:
            return (number, labels, True)
        return None

    def _d_value_roots(self) -> dict[str, set[str]]:
        roots: dict[str, set[str]] = {}
        for _, key, _, values in self.key_table():
            bucket = roots.setdefault(_words(key), set())
            for value in values:
                bucket.update(_roots(value))
        return roots

    def _d_numeric_keys(self, table: list[tuple[str, str, int, list[str]]]) -> dict[str, bool]:
        """For each key, whether most of its values are a bare number with an optional short unit."""
        numeric: dict[str, bool] = {}
        for _, key, _, values in table:
            amounts = [v for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v)]
            numeric[key] = 2 * len(amounts) > len(values)
        return numeric

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

    def _e_gate_prompt(self, stage: str, due: list[dict[str, Any]], tools: list[str]) -> str:
        ordering = stage == 'order'
        record = '\n'.join(f'{k}. {line}' for k, line in enumerate(self.done_log, 1))
        parts = [
            ('REQUEST:\n', self.request),
            ('STEPS DUE NOW:\n', '\n'.join(f"- {p['step']}" for p in due)),
            ('LEDGER:\n', self.ledger()),
            ('CART NOW: ', self.cart_text()),
            ('RECORD (messages and step calls so far):\n', record or '(nothing yet)'),
        ]
        if self.notes:
            parts.append(('STORE NOTES SEEN:\n', '\n'.join(self.notes[-4:])))
        closing = 'order' if ordering else 'ending'
        room = max(0, self.msg_limit - self.msg_used)
        parts.append(('MESSAGES LEFT: ', f'{room}   STEPS LEFT: {self.left()} (the {closing} itself takes one)'))
        parts.append(('TOOLS:\n', '\n'.join(tools)))
        goal = 'place the order' if ordering else 'end without a purchase'
        return _e_sections(GATE.replace('STAGE', goal), parts)

    def _e_gate_heard(self, got: dict[str, Any], said: list[str]) -> tuple[list[str], str]:
        """Once the gate has spoken to the shopper, what the reply changes: new needs, and anything to wait for."""
        told: list[str] = []
        if not said:
            return (told, '')
        self.take_heard({'needs': got.get('needs')}, told)
        return (told, str(got.get('hold') or '').strip())

    def _e_did(self, said: list[str]) -> str:
        return _E_DID_STEPS + '\n'.join(said) if said else ''

    def _e_gate_changed(self, said: list[str], told: list[str], hold: str) -> str:
        done = self._e_did(said)
        if done:
            done += '\n'
        shifts = told + [hold] if hold else told
        return done + "The shopper's reply changes the task: " + '; '.join(shifts) + '. Fix the cart for it, then check out again.'

    def _e_unproven(self, got: dict[str, Any], stage: str, due: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Steps the model calls done that no record line backs (after the last cart change, when messages are due)."""
        entries = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
        talk = stage == 'order' and any(p['messages'] > 0 for p in due)
        total = len(self.done_log)
        if talk:
            upto = min(total, len(self.done_steps))
            proof = {k for k in range(1, upto + 1) if self.done_steps[k - 1] > self.cart_at}
        else:
            proof = set(range(1, total + 1))
        return [e for e in entries if not self._e_backed(e, proof)]

    def _e_backed(self, entry: dict[str, Any], proof: set[int]) -> bool:
        return entry.get('done') is True and any((_count(n) or 0) in proof for n in _items(entry.get('lines')))

    def _e_may_nudge(self, unproven: list[dict[str, Any]], said: list[str], nudged: bool) -> bool:
        if not unproven or said or nudged:
            return False
        return bool(self.roles.get('ask')) and self.msg_used < self.msg_limit and self.left() >= 2

    def _e_gate_nudge(self, unproven: list[dict[str, Any]]) -> None:
        self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
        missing = '; '.join(str(e.get('step') or '') for e in unproven)[:200]
        self.done_log.append(f'(RYU: no line made since the cart last changed at step {self.cart_at} shows {missing} done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
        self.done_steps.append(-1)

    def _e_gate_ask(self, args: dict[str, Any], said: list[str]) -> bool:
        text = next((v for v in args.values() if isinstance(v, str)), '')
        batch = self.send([(1, '', text)], 1)
        if not batch:
            return False
        self.gate_msgs += 1
        said.append(f'Q: {text}\nA: {batch[0][2]}')
        return True

    def _e_gate_call(self, name: str, args: dict[str, Any], said: list[str]) -> bool:
        reply = self.act(name, args)
        if reply is None:
            return False
        seen = reply.get('observation')
        self.absorb(seen)
        sent = f'{name} {_dump(args)} -> '
        shown = self.show(seen)
        self.done_log.append(f'step {self.step}: {sent}{shown[:300]} (cart then: {self.cart_mark()})')
        self.done_steps.append(self.step)
        said.append(sent + shown[:400])
        return True

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

    def _e_fact_ids(self, observation: Any) -> list[str]:
        """Ids of listings with an attribute block in a reply (the walk does not go inside such a listing)."""
        found: list[str] = []
        stack: list[tuple[Any, int]] = [(observation, 0)]
        while stack:
            node, depth = stack.pop()
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
            stack += [(child, depth + 1) for child in reversed(children)]
        return found

    def _e_evidence_line(self, ident: str, needs: list[tuple[int, str]], many: bool) -> str:
        block = _fact_block(self.full.get(ident) or {}) or {}
        parts: list[str] = []
        for number, need in needs:
            prop = need.split(':', 1)[0].strip()
            stem = _stem(prop)
            hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if _names(stem, _stem(k))]
            if not hits:
                continue
            tag = f'(item {number}) ' if many else ''
            parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]))
        return f'{ident}: ' + '; '.join(parts) if parts else ''

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

    def _e_fit_text(self, ident: str) -> str:
        record = self._e_record(ident) or {}
        flat = [str(v) for v in record.values() if isinstance(v, str)]
        nested = [f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)]
        return ' '.join(flat) + ' ' + '. '.join(nested)

    def _e_codes_of(self, index: dict[str, Any], ident: str) -> list[str]:
        return index['marks'].get(ident, {}).get('codes') or []

    def _e_pair_faults(self, part: int, main: int, index: dict[str, Any], cart: list[str]) -> list[str]:
        """Carted parts of one bound pair that name none of the codes of the single carted device."""
        main_ids = self.item_ids(main)
        part_ids = self.item_ids(part)
        devices = [i for i in cart if i in main_ids]
        held = [i for i in cart if i in part_ids]
        if len(devices) != 1 or not held:
            return []
        device = devices[0]
        codes = set(self._e_codes_of(index, device))
        if not codes:
            return []
        found: list[str] = []
        for line in held:
            fits = self.fit_set(line, index)
            if fits and not fits & codes:
                found.append(self._e_misfit(line, part, device, fits, codes, part_ids, index, cart))
        return found

    def _e_misfit(self, line: str, part: int, device: str, fits: set[str], codes: set[str], part_ids: list[str], index: dict[str, Any], cart: list[str]) -> str:
        swaps = [i for i in part_ids if self.fit_set(i, index) & codes and not self.sold_out(i) and i not in cart][:4]
        head = f"{line} ({self.items[part - 1].name}) works with {', '.join(sorted(fits)[:4])}, not with {device} ({', '.join(sorted(codes))})"
        if swaps:
            return f"{head}; seen parts of that kind that work with it: {', '.join(swaps)}"
        return head + '; no seen part of that kind works with it, so look for one or switch the device: ' + self.fit_plan(short=True)

    def _e_code_tally(self, main: int, index: dict[str, Any]) -> dict[str, dict[int, int]]:
        """For each code, how many unsold seen parts of each item bound to this device name it."""
        tally: dict[str, dict[int, int]] = {}
        for part in [p for p, m in self.bound if m == main]:
            for ident in self.item_ids(part):
                if self.sold_out(ident):
                    continue
                for code in self.fit_set(ident, index):
                    per_part = tally.setdefault(code, {})
                    per_part[part] = per_part.get(part, 0) + 1
        return tally

    def _e_main_plan(self, main: int, index: dict[str, Any]) -> str:
        ranked = sorted(self._e_code_tally(main, index).items(), key=_e_tally_rank)[:4]
        if not ranked:
            return ''
        main_ids = self.item_ids(main)
        pieces: list[str] = []
        for code, per_part in ranked:
            counts = ', '.join(f'{n} {self.items[p - 1].name}' for p, n in sorted(per_part.items()))
            seen = [i for i in main_ids if code in self._e_codes_of(index, i)][:2]
            where = f" (device seen: {', '.join(seen)})" if seen else ' (device not seen yet)'
            pieces.append(f'{code}: {counts}{where}')
        return f"codes the {self.items[main - 1].name} candidates' parts name -> " + '; '.join(pieces)

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

    def _e_item_lines(self, n: int, item: Any, ids: list[str], label: dict[str, str], owner: dict[str, list[int]]) -> list[str]:
        """Faults of one item's lines: none or several, a wrong unit count, or a line from a shorter label."""
        who = f'item {n} ({item.name})'
        lines = [i for i in ids if n in owner[i]]
        solo = [i for i in lines if owner[i] == [n]]
        found: list[str] = []
        if not lines:
            found.append(f'{who} has no line in the cart')
        elif len(solo) > 1:
            found.append(f"{who} has {len(lines)} lines ({', '.join(lines)}); keep one")
        for i in lines:
            if i in solo:
                have = self.held_units(i)
                if have != item.qty:
                    found.append(f'{i} is held at {have} unit(s); {who} wants {item.qty}')
            fuller = self.fuller_label(n, label[i])
            if fuller:
                found.append(f'{i} is from {label[i]}, but the store has the label {fuller}, which names {who} in full: take the line from there')
        return found

    def _e_budget_faults(self, ids: list[str]) -> list[str]:
        prices = {i: self.price_of(i) for i in ids}
        if not isinstance(self.budget, (int, float)) or any(p is None for p in prices.values()):
            return []
        if self.brief.get('budget_scope') == 'each':
            return [f'{i} costs {prices[i]:g}, over the {self.budget:g} limit for each item' for i in ids if self.budget < prices[i] <= 10 * self.budget]
        total = sum((prices[i] * self.held_units(i) for i in ids))
        if not self.budget + 1e-09 < total <= 10 * self.budget:
            return []
        return [f'the total is {total:g}, over the budget of {self.budget:g}']

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

    def _e_by_room(self, roomy: int, tight: int) -> int:
        return roomy if self.left() > 15 else tight

    def _e_trail_room(self) -> bool:
        return self.trails < self._e_by_room(6, 4) and self.cost < SPEND_HARD

    def _e_trail_models(self) -> list[str]:
        models = [LIGHT]
        if self.cost + HEAVY_ROOM <= SPEND_HARD:
            models.append(HEAVY)
        return models

    def _e_trail_records(self) -> list[str]:
        """Up to 45 opened records, then plain rows of the listings not opened, 60 lines at most, newest first."""
        newest = list(reversed(self.seen_order))
        opened = [record for record in (self._e_record(i) if i in self.opened else None for i in newest) if record][:45]
        records = ['OPENED ' + _e_dumped(record, 300, 1400) for record in opened]
        rows = ['ROW ' + _e_dumped(self.rows.get(i), 80, 300) for i in newest if i not in self.opened or not self._e_record(i)]
        return records + rows[:60 - len(records)]

    def _e_chain_need(self, chain: dict[str, Any], pending: list[str]) -> str | None:
        named = str(chain.get('need') or '').strip()
        if named and named in pending:
            return named
        return pending[0] if len(pending) == 1 else None

    def _e_chain_value(self, chain: dict[str, Any], links: list[dict[str, Any]], ids: list[str]) -> str | None:
        """The value a chain settles once every page on it is opened: the model's value when it calls the chain done,
        else (when the last page needs no further search) its value or a pointer to that page."""
        if not links or not all(i in self.opened for i in ids):
            return None
        value = str(chain.get('value') or '').strip()
        if chain.get('done') is True and value:
            return value
        if str(links[-1].get('find') or '').strip():
            return None
        return value or f'as stated on {ids[-1]}'

    def _e_follow_chains(self, got: dict[str, Any], pending: list[str]) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """Settle the chains the model traced; returns rows to open, lost links in words, and (about, search) gaps."""
        reach: list[str] = []
        lost: list[str] = []
        gaps: list[tuple[str, str]] = []
        for chain in _items(got.get('chains')):
            if not isinstance(chain, dict):
                continue
            need = self._e_chain_need(chain, pending)
            if not need:
                continue
            links = [link for link in _items(chain.get('links')) if isinstance(link, dict)]
            ids = [str(link.get('id') or '').strip() for link in links]
            value = self._e_chain_value(chain, links, ids)
            if value is not None:
                self.ref_values[need] = value
                self.version += 1
                continue
            closed = [i for i in ids if i in self.rows and i not in self.opened]
            if closed:
                reach += [i for i in closed if i not in reach]
                continue
            about, words = _e_chain_gap(links, ids, self.rows)
            lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
            gaps.append((about, words))
        return (reach, lost, gaps)

    def _e_link_search(self, gap: tuple[str, str], seen: list[str]) -> tuple[list[str], bool]:
        """Search the shop for a lost link; the flag says the search found a page that is already open."""
        found = self.find_link(*gap)
        last = self.trace[-1] if self.trace else {}
        if last.get('kind') != 'link_known':
            return (found, False)
        seen.append(f"{last['id']}{_E_KNOWN_LINK}")
        return (found, True)

    def _e_open_pages(self, tool: str, target: str, idents: list[str], seen: list[str]) -> None:
        for ident in idents:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                return
            reply = self.act(tool, args)
            if reply is None or self.finished:
                return
            page = reply.get('observation')
            self.observe(tool, page)
            seen.append(f'{ident}: ' + self.show(page)[:1500])

    def _e_trail_text(self, seen: list[str], lost: list[str]) -> tuple[str, bool]:
        pieces: list[str] = []
        if seen:
            pieces.append('[RYU opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n')
        if self.ref_values:
            values = '; '.join(f'{k} -> {v}' for k, v in self.ref_values.items())
            pieces.append(f'[values read from the pages requirements point at] {values}\n')
        hold = bool(lost) and self.trail_blocks < self._e_by_room(3, 2) and self.left() > 5
        if hold:
            self.trail_blocks += 1
            pieces.append(_E_TRAIL_HOLD[0] + ' | '.join(lost[:3]) + _E_TRAIL_HOLD[1])
        return (''.join(pieces), hold)

    def _f_seller_names(self, ident: str) -> list[str]:
        names = []
        for field, value in (self.rows.get(ident) or {}).items():
            if not isinstance(value, str) or not re.search(_f_seller_field, str(field), re.I):
                continue
            label = value.strip()
            if 2 < len(label) <= 40:
                names.append(label)
        return names

    def _f_sells_here(self, ident: str, text: str) -> bool:
        """The listing's seller is named (as a whole word) in the lower-cased `text`."""
        return any((_f_word_in(name.lower(), text) for name in self._f_seller_names(ident)))

    def _f_shop_words(self, about: str) -> set[str]:
        """Word pieces of the seller names of every listing whose seller `about` mentions."""
        pieces: set[str] = set()
        for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about):
            for field, value in (self.rows.get(ident) or {}).items():
                if isinstance(value, str) and len(value.strip()) <= 40 and re.search(_f_seller_field, str(field), re.I):
                    pieces.update(re.findall('[a-z0-9]{3,}', value.lower()))
        return pieces

    def _f_link_opened(self, about: str, values: set[str]) -> bool:
        """An opened page of that seller already shows every value of the description."""
        for ident in self.opened_shop(about):
            page = ' '.join(_f_leaves(self.full.get(ident) or self.rows.get(ident) or {})).lower()
            if values and all((_f_word_in(v, page) for v in values)):
                self.trace.append({'kind': 'link_known', 'id': ident})
                return True
        return False

    def _f_link_lookup(self, about: str, words: str) -> list[str]:
        """Seller rows for the description, from a search done earlier for the same words or a new one."""
        plan = self.search_template()
        memo = ' '.join(words.lower().split())
        if memo and memo in self.link_seen:
            self.link_searches += 1
            return self.shop_rows(about, self.link_seen[memo])
        if not (plan and words):
            return []
        tool, base, field = plan
        args, fits = self._f_shaped(tool, {**base, field: words})
        if not fits:
            return []
        self.link_searches += 1
        reply = self.act(tool, args)
        if reply is None:
            return []
        observation = reply.get('observation')
        self.observe(tool, observation)
        self.link_seen[memo] = [i for i, _ in self.records_in(observation)]
        rows = self.shop_rows(about, self.link_seen[memo])
        self.trace.append({'kind': 'link_search', 'words': words[:80], 'rows': rows[:4]})
        return rows

    def _f_of_kind(self, kind: set[str], ids: list[str]) -> list[str]:
        if not kind:
            return list(ids)
        return [i for i in ids if kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]

    def _f_shaped(self, tool: str, args: dict[str, Any]) -> tuple[Any, bool]:
        """Arguments fitted to the tool's schema, and whether they fitted without a fault."""
        problems: list[str] = []
        fitted = _shape(args, self.schemas.get(tool), 'args', problems)
        return fitted, not problems

    def _f_stale_offers(self) -> list[str]:
        """Cart lines whose offer page has not been read since the last turn or the last refused order."""
        since = max(self.step - 2, self.refused_at)
        return [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]

    def _f_reread_one(self, tool: str, target: str, ident: str) -> str | None:
        """Read one offer page again; None stops the round, '' means nothing moved."""
        args, fits = self._f_shaped(tool, {target: ident})
        if not fits:
            return None
        earlier = dict(self.rows.get(ident) or {})
        reply = self.act(tool, args)
        if reply is None or self.finished:
            return None
        observation = reply.get('observation')
        self.offer_seen[ident] = self.step
        self.observe(tool, observation)
        now = self.rows.get(ident) or {}
        moved = [k for k, v in now.items() if k in earlier and earlier[k] != v and (not isinstance(v, (dict, list)))]
        if not moved:
            return ''
        return f"{ident} changed ({', '.join(moved[:4])}): " + _dump(_clip(observation, 300))[:900]

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

    def _f_audit_record(self, ident: str) -> str:
        page = self.full.get(ident)
        if page:
            return 'OPENED ' + _dump(_clip(page, 200))[:1200]
        return 'ROW ' + _dump(_clip(self.rows.get(ident), 120))[:500]

    def _f_open_plan(self, rows: list[str]) -> list[tuple[str, dict[str, Any]]] | None:
        """One compare call over the rows when it shows facts, else up to two single opens; None if neither."""
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and (not self.compare_bare()):
            return [(joiner, {many: rows[:self.compare_room()]})]
        if tool and target:
            return [(tool, {target: i}) for i in rows[:2]]
        return None

    def _f_run_opens(self, plan: list[tuple[str, dict[str, Any]]]) -> bool:
        """Send the open calls; False when one does not fit its schema."""
        for name, args in plan:
            args, fits = self._f_shaped(name, args)
            if not fits:
                return False
            reply = self.act(name, args)
            if reply is None or self.finished:
                return True
            self.observe(name, reply.get('observation'))
        return True

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

    def _f_closing_roles(self, left: int) -> tuple[str, ...] | None:
        """Roles kept beside buy/refuse as the steps run out; None while every tool stays offered."""
        tail = self.tail()
        for limit, roles in ((tail, ()), (tail + 3, ('cart', 'cart_view'))):
            if left <= limit:
                return roles
        if self.buys == 0 and left <= len(self.items) + 5:
            return ('cart', 'cart_view', 'offer')
        return None

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

    def _g_answer(self, call: dict[str, Any], batch: int) -> str:
        """Check one model tool call, run it when it passes, and return the text the model gets back."""
        if call.get('joined'):
            return 'Shown in the side-by-side result above: RYU opened these listings together in one step.'
        fn = call.get('function') or {}
        name = fn.get('name')
        refusal = self._g_screen(name)
        if refusal:
            return refusal
        args, refusal = self._g_checked_args(name, fn.get('arguments'))
        if refusal:
            return refusal
        many = self.compare_target() if name == self.roles.get('compare') else None
        args, note = self._g_room_cut(args, many)
        ending = self.is_buy(name, args) or self.is_refuse(name, args)
        readers = {self.roles.get(job) for job in ('search', 'open', 'offer', 'compare')} - {None}
        refusal = self._g_premature(name, args, ending, batch) or self._g_repeat(name, args, ending, readers)
        if refusal:
            return refusal
        fresh = self._g_unseen(name, args, many)
        if fresh is not None:
            if not fresh:
                records = '\n'.join(f'{i}: {self.record_of(str(i))}' for i in args[many])
                return _g_skipped('these listings have shown everything this view gives since the last change. Their records:\n' + records[:4000])
            if len(fresh) < len(args[many]):
                left_out = ', '.join(str(i) for i in args[many] if i not in fresh)
                note += f'\n[RYU] Left out {left_out}: shown in full already, nothing changed since.'
                args = {**args, many: fresh}
        mark = name + _dump(args)
        if ending and self.left() > 2:
            hold = self._g_ending_hold(name, args)
            if hold:
                return hold
        hold = self._g_decline_push(name, args)
        if hold:
            return hold
        if self.is_refuse(name, args):
            done = self.complete_cart()
            if done:
                return done + (' [episode finished]' if self.finished else '')
            args = self.quiet(name, args)
        hold, said = self._g_order_gate(name, args, ending)
        if hold:
            return hold
        if said:
            note = '\n[RYU] ' + said + note
        return self._g_execute(name, args, many, ending, mark, readers, note)

    def _g_screen(self, name: Any) -> str | None:
        """Turn away calls that cannot run at all: episode over, shopper messages, unknown or withheld tools."""
        if self.finished:
            return 'Not run: the episode is over.'
        if name == self.roles.get('ask'):
            return _g_skipped("RYU's interviewer handles every message to the shopper.")
        usable = {tool['function']['name'] for tool in self.offered()}
        if name not in self.schemas:
            return _g_skipped(f'unknown tool. Tools: {_g_tool_list(usable)}.')
        if name not in usable:
            return _g_skipped(f'too few steps are left for that; use: {_g_tool_list(usable)}.')
        return None

    def _g_checked_args(self, name: str, raw: Any) -> tuple[dict[str, Any] | None, str | None]:
        """Parse, schema-shape and id-resolve the arguments; returns (args, None) or (None, refusal text)."""
        if isinstance(raw, dict):
            args = raw
        elif not (raw or '').strip():
            args = {}
        else:
            args = _json_obj(raw)
            if args is None:
                return None, _g_skipped('the arguments were not one JSON object.')
        problems: list[str] = []
        args = _shape(args, self.schemas[name], 'args', problems)
        if problems:
            return None, _g_skipped('; '.join(problems[:6]))
        args, wrong = self.resolve_ids(name, args)
        if wrong:
            return None, _g_skipped(wrong + '.')
        if name == self.roles.get('cart'):
            args = self.whole_removal(name, args)
        return args, None

    def _g_room_cut(self, args: dict[str, Any], many: str | None) -> tuple[dict[str, Any], str]:
        """Cut a side-by-side request down to the batch size this store is known to handle."""
        asked = args.get(many) if many else None
        if not isinstance(asked, list) or not self.compare_bad:
            return args, ''
        room = self.compare_room()
        if len(asked) <= room:
            return args, ''
        waiting = ', '.join(map(str, asked[room:]))
        note = f'\n[RYU] Sent the first {room} listings only: this store shows at most {room} side by side.'
        return {**args, many: asked[:room]}, note + f' Not shown yet: {waiting}.'

    def _g_premature(self, name: str, args: dict[str, Any], ending: bool, batch: int) -> str | None:
        """Calls that are wrong at this moment: sold-out targets, removing absent lines, badly timed endings."""
        field = self.single_target(name)
        if field and name in (self.roles.get('open'), self.roles.get('offer')):
            wanted = str(args.get(field))
            if wanted in self.gone:
                return _g_skipped(f'{args[field]} sold out at step {self.gone[wanted]} and will not come back. Replace its cart line with another fitting listing.')
        absent = self._g_absent_line(name, args)
        if absent is not None:
            return _g_skipped(f'{absent} is not in the cart, so there is nothing to take off. Cart now: ') + self.cart_text()
        if ending:
            if batch > 1:
                return _g_skipped('send the finishing call on its own, after reading your other results.')
            return None
        if self.left() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')):
            return _g_skipped("the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase.")
        return None

    def _g_absent_line(self, name: str, args: dict[str, Any]) -> str | None:
        """A listing a removing cart call names although the cart does not hold it."""
        if name != self.roles.get('cart'):
            return None
        if not any((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
            return None
        view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
        for value in args.values():
            if not isinstance(value, str) or value not in self.rows:
                continue
            if value not in self.cart_lines and value not in view and value not in self.cart_said:
                return value
        return None

    def _g_repeat(self, name: str, args: dict[str, Any], ending: bool, readers: set[str]) -> str | None:
        """Calls whose answer is already known: an immediate repeat, a cached read, or a fully shown listing."""
        mark = name + _dump(args)
        if not ending and name != self.roles.get('cart') and mark in self.recent:
            return _g_skipped('you just made this same call and nothing changed; its result is above.')
        if name not in readers:
            return None
        cached = self.read_cache.get(mark)
        if cached and cached[0] == self.version:
            return _g_skipped(f'the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000])
        field = self.single_target(name)
        if not field or name == self.roles.get('compare'):
            return None
        wanted = str(args.get(field))
        if not self.shown_already(name, wanted):
            return None
        return _g_skipped(f'{args[field]} has shown everything this view gives since the last change. Its record: ' + self.record_of(wanted))

    def _g_unseen(self, name: str, args: dict[str, Any], many: str | None) -> list[Any] | None:
        """Side-by-side ids that still have something to show; None when the call is not a side-by-side batch."""
        asked = args.get(many) if many else None
        if not isinstance(asked, list) or not asked:
            return None
        return [i for i in asked if not self.shown_already(name, str(i))]

    def _g_cart_fresh(self) -> bool:
        return self.cart_view is not None and self.cart_seen_at >= self.cart_at

    def _g_cart_note(self, text: str) -> str:
        return text + '\nCart now: ' + self.cart_text()

    def _g_ending_hold(self, name: str, args: dict[str, Any]) -> str | None:
        if self.is_buy(name, args):
            return self._g_checkout_hold()
        if self.left() > 5:
            return self._g_decline_hold()
        return None

    def _g_checkout_hold(self) -> str | None:
        """Pre-checkout review: every reason to hold the checkout back, in priority order."""
        wider = self.wider_hold()
        if wider:
            return self._g_cart_note(wider)
        reread = self.reread_offers()
        if self.note_sold:
            self.find_sold()
        change = self._g_cart_recheck()
        if change:
            return change
        swap = self.swap_hint()
        stuck = [] if self.held >= 2 else [i for i in self.cart_ids() if i in self.gone]
        if swap or stuck:
            if not swap:
                self.held += 1
            return self._g_cart_note(_g_held(swap or f"{', '.join(stuck)} in the cart is sold out: replace that line first."))
        chain, hold = self.trail()
        if hold:
            return self._g_cart_note(_g_held(chain))
        hold = self._g_fit_hold() or self._g_line_hold() or self._g_verdict_hold(chain)
        if hold:
            return hold
        if reread and self.cost > SPEND_SOFT:
            return _g_held("RYU re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.')
        return None

    def _g_cart_recheck(self) -> str | None:
        """Look at the cart once before checking out when RYU's picture of it is stale."""
        viewer = self.roles.get('cart_view')
        if not (self.cart_touched and self.cart_seen_at < self.cart_at and viewer and self.left() > 3):
            return None
        view_args, ok = self._g_fit_args(viewer, {})
        view = self.act(self.roles['cart_view'], view_args) if ok else None
        if view is None:
            return None
        self.cart_view = view.get('observation')
        self.cart_seen_at = self.step
        change = self.observe(self.roles['cart_view'], self.cart_view)
        if not change:
            return None
        self.queued = []
        return self._g_cart_note(_g_held('Looking at the cart showed a change: ' + change)) + '\nApply the change, fix the cart if needed, then check out.'

    def _g_fit_hold(self) -> str | None:
        if self.fit_holds >= 2 or self.left() <= self.tail() + 3:
            return None
        misfit = self.fit_faults()
        if not misfit:
            return None
        self.fit_holds += 1
        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
        reason = 'A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3])
        return self._g_cart_note(_g_held(reason + '. Swap the part (or the device), then check out.'))

    def _g_line_hold(self) -> str | None:
        """Cart lines that break the request; quantity-only faults RYU first tries to fix itself."""
        if self.line_holds >= 2 or self.left() <= self.tail() + 2:
            return None
        faults = self.line_faults()
        if faults and all(' is held at ' in f for f in faults):
            fixed = self.fix_counts(self.count_fixes())
            if fixed:
                faults = self.line_faults()
                if not faults:
                    self.line_holds += 1
                    return self._g_cart_note(_g_held("RYU set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.'))
        if not faults:
            return None
        self.line_holds += 1
        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
        return self._g_cart_note(_g_held('The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out."))

    def _g_verdict_hold(self, chain: str) -> str | None:
        """Read the carted listings, then let the independent check judge the basket."""
        lines = self.cart_ids()
        read = self.fill_facts(lines, self.tail() + (1 if len(lines) >= 2 else 3))
        self.show_links(lines, self.tail() + 1)
        read += self.cheaper_rows()
        verdict = self.check_cart()
        if not verdict:
            return None
        seen = ''
        if read:
            seen = "RYU read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n'
        return self._g_cart_note('Not run yet. ' + chain + seen + verdict)

    def _g_decline_hold(self) -> str | None:
        """Before ending without a purchase: the audit, then up to two pushes towards the fitting leads."""
        review = self.audit()
        if review:
            return _g_held(review)
        if not self.leads or self.left() <= 4:
            return None
        if self.refuse_blocks == 0:
            self.refuse_blocks = 1
            self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
            return _g_held('The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.')
        if self.refuse_blocks != 1:
            return None
        self.refuse_blocks = 2
        first = self.leads[0]
        self.check_cart([first], SPEND_HARD)
        if not (self.verdicts.get(self.check_key([first])) and first in self.leads):
            return None
        self.confirmed = first
        self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
        return _g_held(f"An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement.")

    def _g_decline_push(self, name: str, args: dict[str, Any]) -> str | None:
        """Late declines while the cart can still be completed, and early declines before the search is exhausted."""
        if not self.is_refuse(name, args):
            return None
        if self.left() <= 5 and not self.late_held:
            hold = self._g_keep_built_cart()
            if hold:
                return hold
        if self.leads or self.pushed or self.left() <= self.max_steps // 3:
            return None
        self.pushed = True
        advice = ('browse the category of each requested item and page through it with the token or offset the store returns',
                  "try single-word value queries and the property's other names",
                  'and open the listings that might fit')
        return _g_held(f'{self.left()} steps are still left, so search in ways you have not tried before ending: ' + ', '.join(advice) + '. End without a purchase only after that.')

    def _g_keep_built_cart(self) -> str | None:
        lines = [i for i in self.built if not self.sold_out(i)]
        stale = not self._g_cart_fresh()
        carted = self.cart_ids()
        missing = [i for i in lines if i not in carted or (i in self.dropped and stale)]
        if len(lines) < len(self.items) or len(missing) + 1 > self.left():
            return None
        self.late_held = True
        self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
        if missing:
            redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out."
        else:
            redo = 'Check out the cart as it is.'
        why = 'Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. '
        unless = ' End without a purchase only if your search showed that some requested item has no listing'
        return _g_held(why + redo + unless + ' that meets its requirements; then send the ending call again.')

    def _g_order_gate(self, name: str, args: dict[str, Any], ending: bool) -> tuple[str | None, str]:
        """Last questions and the shopper-approval gate; returns (hold text, gate remark)."""
        if self.is_buy(name, args):
            late = self.late_question()
            if late:
                return self._g_cart_note('Not run yet. ' + late), ''
        if not ending or self.left() < 2:
            return None, ''
        go, said = self.gate('order' if self.is_buy(name, args) else 'end')
        if not go:
            return self._g_cart_note('Not run yet. ' + said), ''
        return None, said

    def _g_execute(self, name: str, args: dict[str, Any], many: str | None, ending: bool, mark: str, readers: set[str], note: str) -> str:
        """Send the call to the store, learn from the result and build the reply text."""
        reply = self.act(name, args)
        if reply is None:
            return 'The call failed (network or server). You may try again.'
        observation = reply.get('observation')
        note += self._g_compare_outcome(args, many, observation)
        self.recent = (self.recent + [mark])[-3:]
        self.id_args.update(self._g_id_keys(name, args))
        self.absorb(observation)
        if name == self.roles.get('compare'):
            rows = self.records_in(observation)
            if rows and not self.facts_from.get(name, False):
                self.facts_from[name] = any(_fact_block(r) for _, r in rows)
        if name in readers and name != self.roles.get('search'):
            self.note_view(name, observation)
        plain = {tool for job, tool in self.roles.items() if job != 'search'}
        browsing = name not in plain and not ending
        paged = self.more_pages(name, args, observation) if browsing else ''
        self._g_learn_roles(name, observation)
        self._g_after_store(name, args, reply, observation)
        text = self.show(observation) + paged + note
        if browsing:
            text += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, args, observation)
        plain_text = text
        text += self._g_side_notes(name, args, reply, observation)
        if name in readers:
            self.read_cache[mark] = (self.version, self.step, plain_text)
        if reply.get('finished'):
            text += ' [episode finished]'
        return text

    def _g_compare_outcome(self, args: dict[str, Any], many: str | None, observation: Any) -> str:
        """Learn which side-by-side batch sizes the store answers."""
        asked = args.get(many) if many else None
        if not isinstance(asked, list) or len(asked) <= 1:
            return ''
        shown = _dump(observation)
        if any(str(i) in shown for i in asked):
            self.compare_ok = max(self.compare_ok, len(asked))
            return ''
        self.compare_bad = min(self.compare_bad or 999, len(asked))
        return f'\n[RYU] The side-by-side view showed nothing for {len(asked)} listings: send at most {self.compare_room()} per call.'

    def _g_id_keys(self, name: str, args: dict[str, Any]) -> list[tuple[str, str]]:
        """(tool, argument) pairs whose values are all known listing ids."""
        pairs = []
        for key, value in args.items():
            values = value if isinstance(value, list) else [value]
            if values and all(isinstance(x, str) and x in self.rows for x in values):
                pairs.append((name, key))
        return pairs

    def _g_learn_roles(self, name: str, observation: Any) -> None:
        """A one-listing view that lacks attributes is a price/stock page: re-assign the open and offer roles."""
        fixed_roles = (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'))
        if not self.single_target(name) or name in fixed_roles:
            return
        listing = self.ident(observation) if isinstance(observation, dict) else None
        if not listing:
            return
        if not self.facts_from.get(name, False):
            self.facts_from[name] = bool(_fact_block(observation))
        if name == self.roles.get('open') and not self.facts_from[name]:
            barred = fixed_roles + (self.roles.get('search'),)
            other = next((t for t in self.schemas if t != name and self.single_target(t) and self.facts_from.get(t) is not False and t not in barred), None)
            if other:
                self.roles['open'] = other
        if not self.facts_from[name] and name != self.roles.get('open') and not self.roles.get('offer'):
            self.roles['offer'] = name
        if name == self.offer_tool():
            self.offer_seen[listing] = self.step

    def _g_after_store(self, name: str, args: dict[str, Any], reply: dict[str, Any], observation: Any) -> None:
        """Bookkeeping for cart, search, cart view and checkout results."""
        if name == self.roles.get('cart'):
            self._g_after_cart(name, args, observation)
        if name == self.roles.get('search') and self.records_in(observation):
            self.searched = (self.searched + [dict(args)])[-8:]
        if name == self.roles.get('cart_view'):
            self.cart_view = observation
            self.cart_seen_at = self.step
        if self.is_buy(name, args):
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step

    def _g_after_cart(self, name: str, args: dict[str, Any], observation: Any) -> None:
        self.note_cart(name, args)
        self.cart_said = _dump(observation)
        self.recent = []
        counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None]
        if counts:
            amount = counts[0]
            for value in args.values():
                if not isinstance(value, str) or value not in self.rows:
                    continue
                if amount > 0:
                    self.dropped.discard(value)
                elif amount <= 0 and not _in_rows(observation, value):
                    self.dropped.add(value)
        lines = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
        if len(lines) >= len(self.built):
            self.built = lines

    def _g_side_notes(self, name: str, args: dict[str, Any], reply: dict[str, Any], observation: Any) -> str:
        """Everything appended after the store's own text: evidence, cart hints, notes and sell-out re-reads."""
        extra = ''
        proof = self.evidence(observation)
        if proof:
            extra += '\n' + proof
        if name == self.roles.get('cart') and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)) and any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
            extra += '\n[RYU] A count of 0 left that line in the cart: ' + 'this tool adds the count to the line, so a negative count takes units off.'
        self.note_sold = False
        loose_lines = self.loose_lines(observation)
        meaning = self.take_note(loose_lines) if loose_lines else None
        if meaning:
            extra += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
        buying = self.is_buy(name, args)
        refused = buying and not reply.get('finished')
        again = refused and any(line in self.spoken(observation) for line in self.sold_lines)
        loose = self.note_sold and (refused or (not self.note_named and not buying))
        if (loose or again) and self.cart_ids() and not self.finished:
            seen = self.find_sold()
            if seen:
                extra += "\n[RYU re-read the cart's lines after the refusal]\n" + seen
        self.note_sold = False
        swap = self.swap_hint()
        if swap:
            extra += '\n' + swap
        if self.queued:
            extra += "\n[notes seen on RYU's own calls] " + ' | '.join(self.queued)
            self.queued = []
        return extra

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
        """Check out on RYU's own; None means fall through to ending without a purchase."""
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

    def _g_intro(self, opening: Any) -> str:
        """First user message: the request, the store's opening view and how to begin."""
        parts = ["SHOPPER'S REQUEST:\n" + self.request]
        if opening is not None:
            parts.append('WHAT THE STORE SHOWS FIRST:\n' + self.show(opening))
        parts.append('Start by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.')
        parts.append(self.status())
        return '\n\n'.join(parts)

    def _g_wrap_up(self) -> None:
        """The loop stopped without an ending: end it RYU's way, with one more short loop if the store objects."""
        turned = self.finish(again=True)
        if turned and not (self.finished or self.broke or self.no_credit) and self.left() > 1:
            self.chat.append({'role': 'user', 'content': '[RYU] RYU tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
            self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
        if not self.finished:
            self.finish()

    def _g_may_continue(self, loop_cap: int, spend_cap: float) -> bool:
        if self.finished or self.broke or self.left() <= 0:
            return False
        if self.loop_calls >= loop_cap or self.cost >= spend_cap:
            return False
        return self.clock() < LAST_TURN or (not self.cart_ids() and self.model_room() > EMPTY_ROOM)

    def _g_tidy_call(self, number: int, call: dict[str, Any]) -> dict[str, Any]:
        """Normalise one model tool call: string id, string name, arguments re-dumped as a JSON object."""
        fn = call.get('function')
        if not isinstance(fn, dict):
            fn = {}
        packed = fn.get('arguments')
        if isinstance(packed, str) and packed.strip():
            parsed = _json_obj(packed)
        elif isinstance(packed, dict):
            parsed = packed
        else:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        return {'id': str(call.get('id') or f'k{self.loop_calls}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _dump(parsed)}}

    def _g_ask_first(self, calls: list[dict[str, Any]]) -> bool:
        """Committing calls before the shopper was ever asked: interview first."""
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
        if not any(c['function']['name'] in committing for c in calls):
            return False
        return self.left() > 8 and self.clock() <= ASK_BY

    def _g_interview_instead(self, calls: list[dict[str, Any]]) -> None:
        self.chat.extend({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'} for call in calls)
        summary = self.interview()
        self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.status()})

    def _g_play_turn(self, calls: list[dict[str, Any]], content: str) -> None:
        self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
        replies = self.run_calls(calls)
        closing = len(replies) - 1
        for position, (cid, text) in enumerate(replies):
            if position == closing:
                text += '\n' + self.status()
            self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})



def _crash_note(exc: BaseException) -> dict[str, Any]:
    return {'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Entry point: run one episode and return its event trace. A crash is recorded, and an unfinished episode still
    gets its finish so the store sees a checkout or an ending."""
    try:
        episode = Ryu(problem_data)
    except Exception as exc:
        return [_crash_note(exc)]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append(_crash_note(exc))
    try:
        if not episode.finished:
            episode.finish()
    except Exception as again:
        episode.trace.append(_crash_note(again))
    return episode.trace


__all__ = ['agent_main']
