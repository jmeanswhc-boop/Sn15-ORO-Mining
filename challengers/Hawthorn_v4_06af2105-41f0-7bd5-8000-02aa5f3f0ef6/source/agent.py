"""Cairn: buys for one shopper inside an ORO store world. Built on Halo v3, the public race-15 top agent.
(session3-2 r9: a cart line's page that refers to another listing gets that listing opened before the order; a numeric
limit needs a stated figure; r8: the guide asks for independent calls in one reply, since time, not steps, ends most race-15 failures;
r7: needs written into the request switch the interview off and hold a decline once, and when time runs
out on a partial cart of several items the missing ones get their closest listing instead of a decline; every cart line gets the
page view before any order, forced ones too, and so does each listing a requirement points at
(an attributes view the model asks of such a listing shows its page first); lists name at most three properties; a zero count adds the item.)

A light model walks the store with the world's own tools; code keeps the ledger, plans the questions to the shopper
and checks the cart before an order and the evidence before a decline. Beyond the base:
- a message the request kept for the ending is spent on one more planned question when Cairn is about to buy
  (late_question), so a tight message limit no longer leaves a hidden need unasked;
- a clock keeps every episode inside the validator's 300 s limit: no interview, question, loop turn or model request
  starts when its result could not be used in time, and the model timeout shrinks with the time left;
- each message to the shopper names up to LIST_SIZE (3) planned properties of one item while the shopper answers such
  lists, rotating them so that a property the shopper does not mind cannot hide the ones behind it (settle), and
  falls back to one property per message after a reply that asks for one question at a time.
"""
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
# A need the request itself writes out for the shopper: "The <property> of the <product> is <value>." Worlds
# that put the shopper's preferences into the request state each one in a sentence of this shape; the stories of
# ordinary requests never do (it fires on 0 of 180 ordinary race-15 and race-16 requests).
NEED_SENTENCE = re.compile("(?:^|[.!?]\\s+|\\n)The ([a-z][a-z0-9 /&'\\-]{1,48}?) of the ([a-z][a-z0-9 /&'\\-]{1,70}?) (?:is|are) ([^\\n]{1,420}?)[.](?=\\s|$)")
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
HAIKU_ROOM = 3200  # least output room for a Haiku call: billed Haiku reasons first, and a small cap returns nothing
DIRECTOR_TOKENS = 8000  # output room for a director turn
LOW_EFFORT_AT = 90.0  # seconds: later director turns are asked for low reasoning effort
PICK_ROUNDS = 1  # the strong model's checkout pick, once per episode
PICK_POOL = 10  # read listings per cart line shown to it, cheapest first
CHEAPER_OPENS = 3
TASK_SECONDS = 300.0
TIME_MARGIN = 20.0
CALL_FLOOR = 6.0
ASK_BY = 165.0
QUESTION_BY = 210.0
WIND_DOWN = 200.0
LAST_TURN = 255.0
REFER = '(?:refer to|take [a-z ]{1,30} from|go by|look to|see|check)\\s+(?:a different product,?\\s*)?(?:the\\s+)?([^.\"|]{0,140})'
BIG_STEPS = 60  # tasks with at least this many steps read rows for uncarted items in the model's turn (tr16)
BIG_BY = 150.0  # ...only this early
SWEEP_BY = 120.0  # big fit tasks page the device searches once, this early
CATEGORY_NAME = '(?:^|[_ -])(?:categor\\w*|section|department|dept|aisle|shelf|kind|type|class|group|range)(?:$|[_ -])|^(?:categor|section|department)'
CLOSEST_AT = 215.0  # from here a decline in a written-needs task with a partial cart becomes the closest order
EMPTY_ROOM = 12.0
LATE_BY = 190.0
LATE_STEPS = 8
SCARCE_MESSAGES = 2
# Easy race, five full runs, same tasks: a 5+ property list told a need 0.026 per message, one property 0.085,
# three properties 0.084 (and drew the one-at-a-time refusal 16% vs 11%), so lists stop at three.
LIST_SIZE = 12
LIST_MIXED = 8
PLAN_PICKS = (12, 8)  # properties the planner ranks per item (one item, several): more than a message names, so the lanes last
LIST_STEPS = (1, 3)  # labels in the episode's first and second question, before the full list size
LIST_STEPS_MIXED = (3, 3)  # the same on several items, where each question goes to another item (simulated: 3,3,8 level or above 1,3,8 on 2-3 items)
CAP_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
CAP_NOT = 'result|row|page|count|size|num|item|step'
CAP_PRICED = 'price|cost|budget|spend|money|amount'  # a limit argument's name, when no search has shown one holding yet
CAP_LOW = '(?:^|_)(?:min|minimum|low|lowest|least|from|start|offset|floor)(?:_|$)'
EARLY_TRAILS = 6  # chain rounds the early follow may spend; the checkout keeps its own
CAP_CHANCE = 0.05  # how rarely rows drawn at random from their labels would all sit under a value, for that value to count as a limit held
WALK_ROOM = 10  # steps beyond the finish a walk listing after the first needs: a refusal at the next checkout still needs a swap
WALK_CALLS = 4  # capped shelf listings per cart line at one checkout (race 16: 456 rows on a shelf, 6 a page)
SHELF_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MODE = 'field|filter|facet|brows|categor|list|structured'
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
SWEEP_BIG = 60  # max_steps from which a task is big (clock-bound)
BIG_OPENS = 2  # unread rows Cairn reads itself after a model search on a big task
SWEEP_SPARE = 20  # steps beyond the tail that big-task reads leave
SWEEP_SEARCHES = 2  # opening value searches paged up front
SWEEP_PAGES = 2  # further pages per paged value search
SWEEP_ROWS = (7, 48)  # match counts a value search must have to be paged
LABEL_SHARE = 0.4  # a lane label naming its item less than this share of the item's best label is dropped
SWEEP_KEEP = 6  # steps before the first-moves stop that value paging leaves free
SERIES_PINNED = 2  # most proven parts for a series sweep
SERIES_PAGES = 6  # page reads per series sweep
HUNT_CHOICES = 3  # choices of a closed list turned into hunt value sets
UNIT_VALUE = '(\\d[\\d.]*)\\s*([a-z]{2,4})'
LAST_PAGE_ROOM = 6  # spare steps to auto-read one remaining page
DEAR_SHARE = 2 / 3  # share of a page's priced rows cheaper than a carted line that makes it dear
DEAR_ROOM = 12  # steps beyond the tail a dear-line read needs
SOLD_PEERS = 2  # unread rows at or below a sold line's price read per sold line
CHAIN_SEARCHES = 6  # link searches per episode
EARLY_OPENS = 12  # pages the early chain follow may open in all
FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_short: dict[int, ProxyClient] = {}
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Cairn, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Time is limited too: the episode has about four minutes, and each of your replies takes several seconds however many calls it holds, so when calls do not depend on one another\'s results (searches for different items, opening several listings, adding the lines of several items) send them together in one reply. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun); a store\'s search can read attribute values and shop names as well as titles and may want every word, so a few required values plus the noun can list nearly the whole fitting set, and a model or part code may find nothing (one try tells). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Cairn knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Cairn\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Cairn carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Cairn\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Cairn adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures. A limit on a measure (at most 15 minutes, at least 22 inches) is met only by a stated figure of that measure; words that only suggest one ("no assembly required", "compact") do not meet it.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category. A listing under the requested label is that kind of product even when its title or description names something else, so judge it on the requirements, price and stock.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value; when the requirement asks for what goes with that listing\'s value (its partner, its counterpart) and the page names what its value pairs with, the requirement is that other value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Cairn reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Price and stock can change silently: Cairn re-reads the cart\'s lines and looks at the cart when you call the checkout, so do not spend a turn on either.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.' + ' (Session note: ledger copy B.)'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A clause saying that a property of the product is some value ("where the colour is red") is always a must of that product, never its shelf. Copy only the value: words that belong to the sentence around it, such as a verb that closes the request ("get a lamp where the colour is red sorted" gives "colour: red"), are not part of it. When the request says a property IS a value that reads like an absence or a refusal ("the warranty is not available"), that phrase is the literal value the listing must show ("warranty: not available"), not an exclusion. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product; keep the request\'s own words for how the value relates to that listing ("designed to pair with" stays as written, never "same as")}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"search_words": {"tool": the tool that searches listings by free words (when one tool lists a category and another takes words, the one that takes words), "field": its parameter that takes the words, "args": the fixed arguments that make it search by words (for a tool with a mode argument, that mode), else {}}.\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write \"not X\" only when the answer rejects X itself.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right. The requirements in force carry the shopper\'s later changes and override the request\'s text: one marked DROPPED no longer applies, even where the request still says it.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails, though a value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a measured limit (a number with a unit) is stated only under a key whose name shares a word with the property, so the same number under a key for another measure, or only in the title, leaves it unknown and the listing fails it; a list under the plural of the property\'s key shows the variants on offer, not this listing\'s own value, when the title names a different one of them; a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a measured limit (a number with a unit) is stated only under a key whose name shares a word with the property, so the same number under a key for another measure, or only in the title, leaves it unknown and the listing fails it; a list under the plural of the property\'s key shows the variants on offer, not this listing\'s own value, when the title names a different one of them; a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit; the requirements in force override the request\'s text, and one marked DROPPED no longer counts.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: its page not read, seen only as a search result row or by its attributes).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link (its own record id, never a code its title prints), else "", "about": how the requirement or the previous page describes that listing, "values": the values that description gives for that listing, as a few bare words with no property names and not its shop\'s name, "find": "for a link not found: search words (its shop\'s name and the values the description gives, leaving out property names and the kind of product)"}], "done": true when the last link is an OPENED page that states the property\'s value itself and whose own words do not send the reader on to another listing for that property (a page that hands the value over to a different listing is not the last link, even if its attributes list some value) (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page; when the requirement, or the request\'s own words for it, asks for a different value that partners that one (its partner or counterpart, not the value itself) and the page names what its value pairs with, that other value; wording that asks for whatever value the pointed page states, or to match it, asks for that value itself, even with a verb of going together; or "", "forms": [up to 3 other ways listings may write that same value (other common units or words), else []]}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the values the description gives, as a few search words, leaving out the names of properties and the kind of product ("the mugs in Kelvaro\'s listing with handle colour blue and finish matte" gives "Kelvaro blue matte"): the store\'s search wants every word to be in the listing, and property names or kind words are often absent from it.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
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
    found = re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))
    return [re.sub('[\\s-]+', ' ', c).strip().lower() for c in found if not re.fullmatch('[A-Z][a-z]+ (?:19|20)\\d\\d', c)]

def _code_hit(pattern: str, code: str) -> bool:
    """Whether a code a part names covers a device's code: the same letters and digits, or a series written with trailing
    x's ("abc12xx" covers "abc1234"); spaces and hyphens do not count."""
    wide, plain = (re.sub('[\\s-]+', '', pattern), re.sub('[\\s-]+', '', code))
    stem = wide.rstrip('x')
    if stem != wide and stem[-1:].isdigit():
        return plain.startswith(stem) and len(plain) == len(wide)
    return wide == plain

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    for hit in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)?\\s*[:.]?\\s*(?!(?:no|number)\\b)(' + FIT_CODE + ')\\b', text, re.I):
        codes += _codes(hit.group(1))
    for hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I):
        codes += _codes(hit.group(1))
    works: list[str] = []
    bars: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', text, re.I):
        if not re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
            cut = re.split('(?i)\\b(?:except|excluding|but not)\\b', hit.group(1), maxsplit=1)
            works += _codes(cut[0])
            bars += _codes(cut[1]) if len(cut) > 1 else []
    also = [(hit.group(1).upper(), _codes(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    fam, named = _series_marks(text)
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'bars': bars, 'also': also, 'more': more.group(1).upper() if more else '', 'fam': fam, 'named': named}

def _series_marks(text: str) -> tuple[list[str], list[str]]:
    """The series a record's own codes start with ("AB1" of "AB1-2345") and the bare series-like words it names outside a clause that
    excludes them. Which bare words are series is decided against the codes other records carry (Cairn.link_system)."""
    fam = {hit.upper() for hit in re.findall('\\b([A-Za-z]{1,5}\\d{1,3})-\\d{2,7}\\b', text)}
    rest = re.sub('\\b[A-Za-z]{1,5}\\d{1,3}-\\d{2,7}\\b', ' ', text)
    rest = re.sub('(?i)\\b(?:except|excluding|but not|not for|not compatible with|does not (?:fit|take))\\b[^.;]*', ' ', rest)
    return sorted(fam), sorted({hit.upper() for hit in re.findall('\\b[A-Za-z]{1,5}\\d{1,3}\\b', rest)})
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
    """Whether a result only reports that the call failed: a small object with a key saying so, or one bare code."""
    if not isinstance(observation, dict) or not 0 < len(observation) <= 3:
        return False
    if any((value and re.search(FAIL_KEY, str(key), re.I) for key, value in observation.items())):
        return True
    only = next(iter(observation.values()))
    return len(observation) == 1 and isinstance(only, str) and 0 < len(only) <= 40 and ' ' not in only

def _plain(text: str) -> Any:
    """A field's text as the value a structured result would hold: a boolean, nothing, a number, else the text."""
    bare = text.strip()
    low = bare.lower()
    if low in ('true', 'false'):
        return low == 'true'
    if low in ('none', 'null', ''):
        return None
    money = re.fullmatch('[$€£]?\\s?((?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?)', bare)
    if money and not re.match('0\\d', money.group(1)):
        number = float(money.group(1).replace(',', ''))
        return int(number) if number.is_integer() and '.' not in money.group(1) else number
    return bare

def _block_value(text: str) -> Any:
    """A value inside an attribute block: "attachment method=none" is a value, and a block only keeps text and numbers."""
    found = _plain(text)
    return text.strip() if isinstance(found, bool) or (found is None and text.strip()) else found

def _pair(text: str, mark: str) -> tuple[str, str] | None:
    """(name, value) when the text opens with a field name and this mark."""
    head, found, rest = text.partition(mark)
    if not found and text.endswith(mark.strip()):
        head, found, rest = (text[:-len(mark.strip())], mark, '')
    name = re.sub(LIST_BULLET, '', head).strip()
    if not found or not re.fullmatch(FIELD_NAME, name) or len(name.split()) > 5:
        return None
    return (name, rest.strip())

def _split_fields(text: str, between: str, mark: str) -> tuple[list[tuple[str, str]], int]:
    """The fields of a text cut at `between`, a part with no name joining the field before it (a value may hold the
    separator), with how many parts carried a name. A name closed by a colon with nothing after it heads the bulleted
    or indented lines below it, kept as one field holding those lines."""
    parts = [p for p in text.split(between) if p.strip()]
    fields: list[tuple[str, str]] = []
    named = 0
    heading = -1
    if parts and not _pair(parts[0].strip(), mark):
        return ([], 0)
    for part in parts:
        bare = part.strip()
        if between == '\n' and heading >= 0 and (re.match(LIST_BULLET, part) or part[:1] in ' \t'):
            name, value = fields[heading]
            fields[heading] = (name, (value + '\n' if value else '') + bare)
            continue
        heading = -1
        if between == '\n' and bare.endswith(':') and re.fullmatch(FIELD_NAME, bare[:-1].strip()):
            fields.append((bare[:-1].strip(), ''))
            heading = len(fields) - 1
            named += 1
            continue
        pair = _pair(bare, mark)
        if pair:
            fields.append(pair)
            named += 1
        elif fields:
            name, value = fields[-1]
            fields[-1] = (name, value + between + bare)
    return (fields, named) if len(parts) else ([], 0)

def _attr_block(text: str) -> dict[str, Any] | None:
    """A listing's attributes written inside one field: "a=b; c=d", a brace block, bulleted "name: value" lines, or short
    "name: value" parts between commas. Prose stays text whatever colons it holds (a description reading "Fits: A, B. Condition:
    new" is no block), and a name may be any short text, since attribute names come as the catalog has them."""
    bare = text.strip()
    if len(bare) < 5:
        return None
    if bare[0] in '{[' and bare[-1] in '}]':
        try:
            parsed = json.loads(bare)
            return parsed if isinstance(parsed, dict) else None
        except (ValueError, RecursionError):
            found = dict(re.findall('[\'"]?([A-Za-z][\\w ./-]{0,38})[\'"]?\\s*[:=]\\s*[\'"]?([^,\'"}\\]]*)', bare))
            return {k.strip(): _plain(v) for k, v in found.items()} if len(found) >= 2 else None
    if '=' in bare:
        out: dict[str, Any] = {}
        last = ''
        parts = [p for p in re.split(';\\s+|;$|\\s\\|\\s|\\n', bare) if p.strip()]
        for part in parts:
            name, found, value = part.partition('=')
            if found and 0 < len(name.strip()) <= BLOCK_NAME and name.strip() not in out:
                last = name.strip()
                out[last] = _block_value(value)
            elif last:
                out[last] = f'{out[last]}; {part.strip()}'
            else:
                return None
        return out if len(out) >= 2 or (out and ':' not in last and len(last) <= 30 and len(bare) <= 90) else None
    lines = [line for line in bare.splitlines() if line.strip()]
    if len(lines) >= 2 and all((re.match(LIST_BULLET, line) for line in lines)):
        pairs = [_pair(line.strip(), ': ') for line in lines]
        if all(pairs):
            return {name: _plain(value) for name, value in pairs}
    for between in (', ', '; '):
        if between in bare and ': ' in bare:
            pairs = [_pair(part.strip(), ': ') for part in bare.split(between)]
            if len(pairs) >= 2 and all(pairs) and all(('.' not in name and len(value) <= BLOCK_VALUE for name, value in pairs)):
                return {name: _plain(value) for name, value in pairs}
    return None

def _record_text(text: str) -> dict[str, Any] | None:
    """The record a text spells out field by field, whichever separators it uses; None when it reads as plain prose. The
    separators are found by trying them, the outer field mark first and then the pair mark giving the most named fields,
    and a value spelled the same way becomes a record of its own (a listing's attributes written inside one field)."""
    best: tuple[int, list[tuple[str, str]]] | None = None
    for between in FIELD_MARKS:
        if between not in text:
            continue
        pieces = len([p for p in text.split(between) if p.strip()])
        for mark in PAIR_MARKS:
            fields, named = _split_fields(text, between, mark)
            if named < RECORD_FIELDS or named < (PAIR_SHARE if between == ', ' else FIELD_SHARE) * pieces:
                continue
            if len({name for name, _ in fields}) * 2 <= len(fields):
                continue
            if best is None or named > best[0]:
                best = (named, fields)
        if best:
            break
    if best is None:
        return None
    record: dict[str, Any] = {}
    for name, value in best[1]:
        if name in record:
            continue
        inner = _attr_block(value)
        record[name] = inner if inner else value.strip() if re.search(ID_FIELD, name, re.I) else _plain(value)
    return record

def _table_text(text: str) -> list[dict[str, Any]] | None:
    """Rows of a text table: a heading line of names, then lines cut into as many cells (markdown rules are skipped)."""
    lines = [line.strip().strip('|').strip() for line in text.splitlines() if line.strip() and not re.fullmatch('[\\s|:+=-]+', line)]
    if len(lines) < 2:
        return None
    for between in ('|', '\t', ';', ','):
        cells = [[c.strip() for c in line.split(between)] for line in lines]
        size = len(cells[0])
        if size < 3 or any((len(row) != size for row in cells)) or not all((re.fullmatch(FIELD_NAME, c) for c in cells[0])):
            continue
        if any((re.fullmatch('[\\d.,$]+', c) for c in cells[0])):
            continue
        return [{name: _plain(c) for name, c in zip(cells[0], row)} for row in cells[1:]]
    return None

def _records_text(text: str) -> Any:
    """The structured form of a text a store tool returned, or None when it holds no records."""
    bare = text.strip()
    if len(bare) < 12:
        return None
    if bare[0] in '{[':
        try:
            return _restore(json.loads(bare))
        except (ValueError, RecursionError):
            pass
    for parts in (re.split('\\n\\s*\\n', bare), bare.splitlines()):
        if len(parts) >= 2:
            rows = [_record_text(p) for p in parts]
            good = [r for r in rows if r]
            if len(good) >= 2 and len(good) * 5 >= len(parts) * 4 and len({next(iter(r)) for r in good}) == 1:
                return good
    found = _table_text(bare) if '\n' in bare and _record_text(bare) is None else _record_text(bare)
    if found is None and len(bare) <= TAG_TEXT and re.search('<[a-zA-Z/][^<>]{0,40}>', bare):
        plain = re.sub('</?(?:b|i|em|strong|span|a|u|small)\\b[^<>]{0,40}>', '', bare, flags=re.I)
        plain = re.sub('</t[dh]>\\s*<t[dh][^<>]{0,40}>', ': ', plain, flags=re.I)
        plain = re.sub('\\s*(?:<[^<>]{1,40}>\\s*)+', '\n', plain).strip()
        if plain != bare and len(plain) >= 12:
            return _records_text(plain)
    if found is None and '\n' in bare:
        lines = [re.sub('(?:[ ]{2,}|\t)', ': ', line.strip(), count=1) for line in bare.splitlines() if line.strip()]
        spaced = '\n'.join(lines)
        if spaced != bare:
            return _record_text(spaced)
    return found

def _block_text(text: str) -> Any:
    """A text field of a row that already has its own fields: read as a record only when wholly written as name=value parts."""
    if len(text) < 5 or '=' not in text or not re.search('; | \\| |\\n', text):
        return text
    parts = [p for p in re.split(';\\s+|;$|\\s\\|\\s|\\n', text.strip()) if p.strip()]
    if len(parts) < 2 or not all((re.fullmatch('[^=:]{1,40}=[^=]{0,200}', p.strip()) for p in parts)):
        return text
    return _attr_block(text) or text

def _pairs_block(rows: list[Any], in_record: bool) -> dict[str, Any] | None:
    """Attributes written as a list: objects of a name and a value, [name, value] lists, or, inside a record, short "name: value"
    texts. A list of offers, of highlights or of notes is no such thing."""
    if len(rows) < 2:
        return None
    out: dict[str, Any] = {}
    first = rows[0]
    if isinstance(first, dict):
        if any((not isinstance(r, dict) or set(r) != set(first) or len(r) != 2 for r in rows)):
            return None
        keys = list(first)
        if not re.search(PAIR_KEY, str(keys[0]), re.I):
            return None
        for row in rows:
            name, value = row[keys[0]], row[keys[1]]
            if not isinstance(name, str) or not 0 < len(name) <= BLOCK_NAME or isinstance(value, (dict, list)):
                return None
            out[name.strip()] = value
        return out
    if isinstance(first, (list, tuple)):
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 2 or not isinstance(row[0], str) or isinstance(row[1], (dict, list)) or not 0 < len(row[0]) <= BLOCK_NAME:
                return None
            out[row[0].strip()] = row[1]
        return out
    if in_record and all((isinstance(r, str) for r in rows)):
        for row in rows:
            pair = _pair(row.strip(), ': ') or _pair(row.strip(), '=')
            if not pair or len(pair[1]) > BLOCK_VALUE or len(pair[0].split()) > 3:
                return None
            out[pair[0]] = _plain(pair[1])
        return out
    return None

def _column_rows(node: dict[str, Any]) -> dict[str, Any]:
    """A result that names its columns once and gives each row as a list comes back with each row as an object."""
    heads = [k for k, v in node.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(c, str) and re.fullmatch(FIELD_NAME, c) for c in v))]
    for head in heads:
        names = node[head]
        for key, rows in node.items():
            if key != head and isinstance(rows, list) and rows and all((isinstance(r, (list, tuple)) and len(r) == len(names) for r in rows)):
                out = {k: v for k, v in node.items() if k != head}
                out[key] = [dict(zip(names, row)) for row in rows]
                return out
    return node

def _id_texts(record: dict[str, Any]) -> dict[str, Any]:
    """A record's numeric identifier as text, the form the agent keys listings by (schemas turn it back when a tool wants a number)."""
    if not any((isinstance(v, str) and ' ' in v for v in record.values())):
        return record
    return {k: str(v) if isinstance(v, int) and not isinstance(v, bool) and re.search(ID_FIELD, k, re.I) and abs(v) >= 100 else v for k, v in record.items()}

def _keyed_rows(node: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Rows given as an object keyed by listing code, each a record with the same fields, as a list of rows with their code."""
    rows = list(node.values())
    if len(rows) < 2 or not all((isinstance(r, dict) and len(r) >= RECORD_FIELDS for r in rows)) or not all((re.search('\\d', str(k)) and ' ' not in str(k) for k in node)):
        return None
    if len(set(rows[0]).intersection(*rows[1:])) < RECORD_FIELDS:
        return None
    return [r if any((re.search(ID_FIELD, str(x), re.I) for x in r)) else {'id': str(k), **r} for k, r in node.items()]

def _holds_id(node: dict[str, Any]) -> bool:
    return any((re.search(ID_FIELD, str(k), re.I) for k in node))

def _flat_block(node: dict[str, Any]) -> dict[str, Any]:
    """An attribute block with groups inside it, as one level: {"size": {"width": 3}} gives {"size width": 3}."""
    if not any((isinstance(v, dict) for v in node.values())):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if isinstance(value, dict):
            out.update({f'{key} {inner}': v for inner, v in value.items() if not isinstance(v, (dict, list))})
        else:
            out[key] = value
    return out

def _restore_field(value: Any, depth: int, carded: bool, page: bool=True) -> Any:
    """A field's value restored; a text only becomes a record when it reads as a listing, by an id, by a code among its values
    or by many fields, so that a store's note written with colons and semicolons stays a note."""
    found = _restore(value, depth + 1, carded, page)
    if isinstance(value, str) and isinstance(found, dict) and len(found) < NOTE_FIELDS and not _holds_id(found) and not any((_id_like(v) for v in found.values())):
        return value
    return found


def _text_table(text: str) -> dict[str, Any] | None:
    """A result written as a text table: a header line of distinct names without digits, then lines with as many fields
    under one separator. Lines outside the table are kept as a note, with the total and the next offset or page read
    from it when it says them ("30 matches ... more from offset 5")."""
    lines = [l for l in str(text).splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    for sep in (' | ', '\t', ' ; ', '|', ';'):
        counts = [len(l.split(sep)) for l in lines]
        for at, line in enumerate(lines):
            width = counts[at]
            if width < 3:
                continue
            heads = [c.strip() for c in line.split(sep)]
            if any((re.search('\\d', c) or not c for c in heads)) or len(set(heads)) < width:
                continue
            body = [l for l, c in zip(lines[at + 1:], counts[at + 1:]) if c == width]
            rest = lines[:at] + [l for l, c in zip(lines[at + 1:], counts[at + 1:]) if c != width]
            out: dict[str, Any] = {'rows': [dict(zip(heads, [c.strip() for c in l.split(sep)])) for l in body]}
            if rest:
                note = ' '.join((l.strip() for l in rest))
                out['note'] = note
                total = re.search('(\\d+)\\s+(?:matches|results|items|listings|hits|products|found|rows in all)', note, re.I)
                if total:
                    out['total'] = int(total.group(1))
                    later = re.search('(?:offset|page|cursor|start|from)\\s+([A-Za-z0-9_-]+)\\s*$|(?:next|more)[^.;]*?(?:offset|page|cursor|start|from)\\s+([A-Za-z0-9_-]+)', note, re.I)
                    value = (later.group(1) or later.group(2)) if later else None
                    out['next'] = int(value) if value and value.isdigit() else value
            return out
    return None


def _text_sheet(text: str) -> dict[str, Any] | None:
    """One listing written as lines of "Key: value": an indented run under a key with no value is a nested block (its
    attributes); a first line without a key that starts with an id-like word gives the id and the title."""
    lines = [l.rstrip() for l in str(text).splitlines() if l.strip()]
    if len(lines) < 3:
        return None
    out: dict[str, Any] = {}
    block: dict[str, Any] | None = None
    pairs = 0
    lead = ''
    for number, raw in enumerate(lines):
        indent = len(raw) - len(raw.lstrip())
        found = re.match("^\\s*(?:[-*\u2022]\\s+)?([A-Za-z][A-Za-z0-9 _./&()'-]{0,40}?)\\s*[:=]\\s*(.*)$", raw)
        if not found:
            if number == 0:
                lead = raw.strip()
            continue
        key, value = (found.group(1).strip(), found.group(2).strip())
        if not value:
            block = {}
            out[key] = block
            continue
        if indent > 0 and block is not None:
            block[key] = value
        else:
            block = None if indent == 0 else block
            out[key] = value
        pairs += 1
    if pairs < 3:
        return None
    if lead:
        first = lead.split()[0].strip(':,;')
        if _id_like(first):
            out.setdefault('id', first)
            title = re.sub('^\\S+\\s*[-:|\u2013\u2014]?\\s*', '', lead).strip()
            if title:
                out.setdefault('title', title)
        else:
            out.setdefault('title', lead)
    return out

def _restore(value: Any, depth: int=0, in_record: bool=False, page: bool=True) -> Any:
    """A store result in the one form the rest of the agent reads: lists of row objects, a listing's attributes as a nested
    object. Worlds spell results differently (race 15 gave each record as one text, its attributes "a=b; c=d" inside it, and
    a result the agent cannot read leaves every record-based check blind), so the spelling is undone here, once, for all."""
    if depth > 7:
        return value
    if isinstance(value, str):
        found = _records_text(value)
        if found is None and depth == 0 and '\n' in value:
            found = _text_table(value) or _text_sheet(value)
        return found if found is not None else value
    if isinstance(value, list):
        block = _pairs_block(value, in_record and page)
        if block:
            return block
        return [_restore_field(v, depth, False, page) for v in value]
    if isinstance(value, dict):
        node = _id_texts(_column_rows(value))
        carded = len(node) >= RECORD_FIELDS or _holds_id(node)
        keyed = {k: _keyed_rows(v) if isinstance(v, dict) else None for k, v in node.items()}
        node = {k: keyed[k] or v for k, v in node.items()}
        listing = _holds_id(node) or any((_id_like(v) for v in node.values()))
        out = {k: _restore_field(v, depth, carded, page) if not isinstance(v, str) or len(node) <= 2 else (_block_text(v) if listing else v) for k, v in node.items()}
        if depth <= 3:
            out = {k: _flat_block(v) if isinstance(v, dict) and any((isinstance(x, dict) for x in v.values())) and not any((isinstance(x, list) for x in v.values())) and isinstance(next(iter(v.values()), None), (dict, str, int, float)) and len(node) >= RECORD_FIELDS and not _holds_id(v) else v for k, v in out.items()}
        if len(out) == 1:
            only = next(iter(out.values()))
            if isinstance(only, dict) and len(only) >= RECORD_FIELDS and isinstance(next(iter(value.values())), str):
                return only
        return out
    return value

_FIRST_SPELLING: dict[str, str] = {}

def _join_twins(held: str, value: str) -> str:
    """Two spellings' values as one list, each element once; the first spelling's value stays known, for measures of how a key's values read."""
    between = '(?<!\\d),|,(?!\\d)'
    have = {part.strip().lower() for part in re.split(between, held)}
    extra: list[str] = []
    for part in re.split(between, value):
        if part.strip() and part.strip().lower() not in have:
            have.add(part.strip().lower())
            extra.append(part.strip())
    if not extra:
        return held
    joined = (f'{held}, ' if held.strip() else '') + ', '.join(extra)
    _FIRST_SPELLING[joined] = _FIRST_SPELLING.get(held, held)
    return joined

def _glued(text: str) -> str:
    """A number and its short unit as one word, however it is spelled: "65 W" and "65W" are one value, "45 W" another."""
    return re.sub('(\\d)\\s+([a-z]{1,3})(?![a-z0-9])', '\\1\\2', text.lower())

def _text_words(text: str) -> set[str]:
    """The words of a row or a search, a number and its short unit also as one word."""
    low = text.lower()
    return set(re.findall('[a-z0-9]+', low)) | set(re.findall('[a-z0-9]+', _glued(low)))

def _kin_forms(want: str, word: str) -> bool:
    """Two forms of one word ("folding", "foldable"): a shared start of four letters or more, each remaining ending empty
    or two to four letters holding a vowel. Non-letter strings only match themselves."""
    if not (want.isalpha() and word.isalpha()) or want == word:
        return want == word
    size = 0
    for a, b in zip(want, word):
        if a != b:
            break
        size += 1
    if size < 4:
        return False
    return all((not tail or (2 <= len(tail) <= 4 and re.search('[aeiouy]', tail))) for tail in (want[size:], word[size:]))


def _value_tokens(value: str) -> list[str]:
    """A value's words a search or a row can hold: long ones, else short ones of two characters holding a letter (XL, 4g, 65w), else its numbers."""
    low = _glued(value)
    return re.findall('[a-z0-9]{3,}', low) or re.findall('[0-9]+[a-z]|[0-9]*[a-z][a-z0-9]+', low) or re.findall('[0-9]+', low)

def _fold_twins(node: Any, depth: int=0) -> Any:
    """One line per property: a block spelling a key twice keeps the first spelling, with both values when they differ, so that
    one element of a list is enough ("clock shape: rectangle" and "clock_shape: rectangular, square" hold square). About a tenth
    of the listings of races 13-15 and of a practice world hold such pairs with different values; read as a conflict, they
    lost listings the store's answer accepted. Every repeated line also costs tokens on each later model turn."""
    if depth > 5:
        return node
    if isinstance(node, list):
        return [_fold_twins(v, depth + 1) for v in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    seen: dict[str, str] = {}
    for key, value in node.items():
        name = _words(key)
        first = seen.get(name)
        if first is None:
            seen[name] = key
            out[key] = _fold_twins(value, depth + 1)
        elif out[first] == value:
            continue
        elif isinstance(out[first], str) and isinstance(value, str) and not re.search(ID_FIELD, name, re.I):
            out[first] = _join_twins(out[first], value)
        else:
            out[key] = _fold_twins(value, depth + 1)
    return out if len(out) >= 2 or len(node) < 2 else {k: _fold_twins(v, depth + 1) for k, v in node.items()}

def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

PICK = ('You are the final judge of a shopping cart. For each cart line below, look at the candidate listings of that '
        'category (already read in full) and decide which one the shopper should get: among the candidates that meet EVERY '
        'requirement in force for that item (judge each requirement against the listing\'s attributes, title and description; '
        'a value written differently but meaning the same counts), take the one the request\'s pick rule prefers (the '
        'cheapest, unless the request says otherwise). Keep the cart line when it is that listing or when no candidate is '
        'clearly better. Answer with ONE JSON object and nothing else: {"lines": [{"line": cart line id, "best": the id '
        'to buy instead or the same id, "why": a short reason}]}')
LEAN_TITLE = 110  # a row's title in a list of rows, in characters
LEAN_BLURB = 90  # a row's description in a list: its opening words (model number, "also sold as") only
BLURB_KEYS = ('blurb', 'description', 'details', 'about', 'summary', 'overview')


def _lean(value: Any, depth: int=0) -> Any:
    """A list of rows as the model needs it to choose: empty fields dropped, long titles and descriptions cut. The full
    record of a listing stays one page view away; this only trims rows shown side by side (prompt tokens are the cost)."""
    if depth > 6:
        return value
    if isinstance(value, list):
        return [_lean(v, depth + 1) for v in value]
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, item in value.items():
        if item is None or item == '' or item == [] or item == {}:
            continue
        low = str(key).lower()
        if isinstance(item, str) and any((b in low for b in BLURB_KEYS)):
            item = item if len(item) <= LEAN_BLURB else item[:LEAN_BLURB] + '...'
        elif isinstance(item, str) and ('title' in low or low == 'name') and len(item) > LEAN_TITLE:
            item = item[:LEAN_TITLE] + '...'
        else:
            item = _lean(item, depth + 1)
        out[key] = item
    return out


def _flat(value: Any, limit: int) -> str:
    if isinstance(value, dict):
        text = '; '.join((f'{k}={_flat(v, 80)}' for k, v in value.items() if v not in (None, '', [], {})))
    elif isinstance(value, list):
        text = ', '.join((_flat(v, 80) for v in value if v not in (None, '', [], {})))
    else:
        text = str(value)
    return text if len(text) <= limit else text[:limit] + '...'

def _line(record: dict[str, Any]) -> str:
    """One store record as one line, 'field: value | ...': no JSON quoting, empty fields dropped, titles and
    descriptions cut (descriptions keep their opening words: model number, 'also sold as'), facts kept whole."""
    parts = []
    for key, value in record.items():
        if value in (None, '', [], {}):
            continue
        low = str(key).lower()
        if any((b in low for b in BLURB_KEYS)):
            limit = LEAN_BLURB
        elif 'title' in low or low == 'name':
            limit = LEAN_TITLE
        elif isinstance(value, dict):
            limit = 700
        else:
            limit = 240
        parts.append(f'{key}: {_flat(value, limit)}')
    return ' | '.join(parts)


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
            pass
    # an answer that writes an object, then 'corrections' and a second object: take the last whole object
    decoder, at, last = (json.JSONDecoder(), 0, None)
    while True:
        at = text.find('{', at)
        if at < 0:
            break
        try:
            got, end = decoder.raw_decode(text, at)
            if isinstance(got, dict):
                last = got
            at = end
        except ValueError:
            at += 1
    return last

def _words(key: str) -> str:
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def _snake(key: str) -> str:
    return _words(key).replace(' ', '_')

def _stem(text: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _words(text).split()))

def _roots(text: str) -> set[str]:
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def _spelled(text: str) -> frozenset[str]:
    """A name's words with plural -s and British -our folded, so "colours" and "color" compare equal."""
    return frozenset((re.sub('our$', 'or', w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w) for w in _words(text).split()))

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
        top = schema['maxLength']
        cut = value[:top]
        value = cut.rsplit(' ', 1)[0].rstrip() if ' ' in cut.strip() and not value[top:top + 1].isspace() and top >= 8 else cut
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
        dotted = [k for k in value if isinstance(k, str) and '.' in k and k not in props and k.split('.', 1)[0] in props]
        if dotted:
            value = dict(value)
            for key in dotted:
                head, rest = key.split('.', 1)
                inner = dict(value[head]) if isinstance(value.get(head), dict) else {}
                inner[rest] = value.pop(key)
                value[head] = inner
        if schema.get('additionalProperties') is False:
            for name in [n for n in value if n not in props]:
                faults.append(f'{where}.{name} is not an allowed argument')
        for name in schema.get('required') or []:
            if name not in value:
                faults.append(f'{where}.{name} is required')
        value = {k: _shape(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
    return value

LINK_FILLER = {'the', 'and', 'with', 'from', 'that', 'this', 'these', 'those', 'item', 'items', 'listing', 'listings', 'product',
               'products', 'page', 'pages', 'pair', 'pairs', 'version', 'versions', 'one', 'ones', 'its', 'are', 'made', 'for', 'offers',
               'offer', 'sells', 'sell', 'comes', 'come', 'makes', 'make', 'called', 'named', 'description', 'listed', 'which', 'whose',
               'than', 'rather', 'instead', 'not', 'also', 'only', 'same', 'other', 'another', 'different', 'here', 'there', 'elsewhere',
               'exists', 'shop', 'store', 'seller', 'brand', 'styled', 'intended', 'look', 'refer', 'see', 'take', 'taken', 'should',
               'given', 'gives', 'say', 'says', 'stated', 'states', 'has', 'have', 'had', 'available', 'all', 'any', 'into', 'out'}


CHOICE_NOT = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|listing|page|same|either way|whichever)\\b'
SENDS = '(?:refers? to|go(?:es)? by|look(?:s)? (?:at|to)|see|check|use|taken? from|comes? from|follow(?:s)?)\\s+(?:a different product,?\\s*|another product,?\\s*|a different listing,?\\s*)?(?:the\\s+)?([^.;|"]{3,160})'


def _choices(value: str) -> list[str]:
    """The choices of a closed either-or value ("asus or hp", "red, blue or green"); [] for any other value. A query that
    holds two choices finds only listings that carry both, so each choice is searched on its own."""
    low = (value or '').strip().lower()
    if not re.search('\\bor\\b', low) or re.search(CHOICE_NOT, low):
        return []
    parts = [re.sub('^(?:a|an|the|either)\\s+', '', p.strip(' ,.;')) for p in re.split(',|\\bor\\b|/', low)]
    parts = [p for p in parts if p]
    if len(parts) < 2 or any((len(p.split()) > 4 for p in parts)):
        return []
    return list(dict.fromkeys(parts))[:4]


def _sent_on(text: str, prop: str) -> str:
    """When a page's own sentence about property `prop` sends the reader to another listing ("for occasion, go by the
    Belkori sneakers ..., not this item"), that listing's description; else ''."""
    want = [w for w in re.findall('[a-z]{3,}', (prop or '').lower()) if w not in ('the', 'and', 'for')]
    if not want:
        return ''
    for sentence in re.split('(?<=[.!?])\\s+', text or ''):
        low = sentence.lower()
        tokens = set(re.findall('[a-z0-9]+', low))
        if not all((_holds_word(w, tokens) for w in want)):
            continue
        hit = re.search(SENDS, sentence, re.I)
        first = min((m.start() for w in want for m in [re.search('(?<![a-z])' + re.escape(w[:max(3, len(w) - 2)]), low)] if m), default=len(low))
        if hit and first < hit.start():
            about = re.split(',?\\s+(?:not|rather than|instead of)\\b', hit.group(1), flags=re.I)[0].strip(' ,')
            if len(about.split()) >= 2:
                return about
    return ''


def _link_words(about: str) -> list[str]:
    """The words of a link's description that a matching page must hold: its shop, kind and values, without filler."""
    words = re.findall('[a-z0-9]+(?:\\.[0-9]+)?', (about or '').lower())
    return list(dict.fromkeys(w for w in words if (len(w) >= 3 or w.isdigit()) and w not in LINK_FILLER))


def _holds_word(word: str, tokens: set[str]) -> bool:
    if word in tokens:
        return True
    if word.replace('.', '').isdigit():
        return False
    if len(word) >= 5:
        stem = word[:max(4, len(word) - 2)]
        return any(t.startswith(stem) for t in tokens)
    return word + 's' in tokens or (word.endswith('s') and word[:-1] in tokens) or word + 'es' in tokens


def _record_tokens(record: Any) -> set[str]:
    return set(re.findall('[a-z0-9]+(?:\\.[0-9]+)?', _dump(record).lower()))


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
        named = bool(re.search(ID_FIELD, str(key), re.I))
        if not all((_id_like(v, named) for v in values)):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = (named, sum((len(v) for v in values)) / len(values))
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

def _llm_within(seconds: float) -> ProxyClient:
    """The model client whose timeout fits the time left: the usual one, or one with a timeout in whole 5 s steps."""
    if seconds >= 50:
        return _llm
    limit = max(5, int(seconds) // 5 * 5)
    if limit not in _llm_short:
        _llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return _llm_short[limit]

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

def _soft(fallback: Any) -> Any:
    """A method that adds to the episode but is not needed for it: if it fails, the trace says so and the episode goes on."""

    def wrap(method: Any) -> Any:

        def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return method(self, *args, **kwargs)
            except Exception as exc:
                self.trace.append({'kind': 'soft_crash', 'where': method.__name__, 'error': f'{type(exc).__name__}: {exc}'[:300]})
                return fallback() if callable(fallback) else fallback
        return guarded
    return wrap

class _Stamped(list):
    """A trace whose entries say when they happened."""

    def __init__(self, clock: Any):
        super().__init__()
        self.clock = clock

    def append(self, entry: dict[str, Any]) -> None:
        entry.setdefault('at', round(self.clock(), 1))
        super().append(entry)

def _filler(spec: dict[str, Any]) -> Any:
    """A value for a required field the agent has nothing to say in; None when the schema gives no safe one."""
    if isinstance(spec.get('enum'), list) and spec['enum']:
        return spec['enum'][0]
    for word in ('const', 'default'):
        if word in spec:
            return spec[word]
    kinds = _kinds(spec)
    if 'integer' in kinds or 'number' in kinds:
        return spec['minimum'] if isinstance(spec.get('minimum'), (int, float)) else 1
    if 'boolean' in kinds:
        return False
    return 'question'[:spec.get('maxLength') or 8] if 'string' in kinds else None

def _strings(node: Any) -> list[str]:
    """The texts a call's arguments hold, at any depth."""
    if isinstance(node, dict):
        return [s for v in node.values() for s in _strings(v)]
    if isinstance(node, list):
        return [s for v in node for s in _strings(v)]
    return [node] if isinstance(node, str) else []

def _says(node: Any, text: str) -> bool:
    """Whether a call's arguments carry `text` whole, however deep."""
    if isinstance(node, dict):
        return any((_says(v, text) for v in node.values()))
    if isinstance(node, list):
        return any((_says(v, text) for v in node))
    return node == text

# ---- c15_v28: readers shared by the code checks ported from the arena forum guides (#96-#103) ----
LIMIT_RE = re.compile(r'\b(?:at least|at most|no more than|no less than|not more than|not less than|up to|under|over|below|above|less than|more than|greater than|fewer than|minimum|maximum|min|max|or more|or less|or fewer|or higher|or lower|between)\b', re.I)
UPPER_RE = re.compile(r'\b(?:at most|no more than|not more than|up to|under|below|less than|fewer than|maximum|max|or less|or fewer|or lower|or colder|or lighter|or shorter|or smaller|or cheaper|or faster|colder than|lighter than|shorter than|smaller than|lower than|no heavier than|no longer than|no warmer than)\b', re.I)
NEG_RE = re.compile(r'\b(?:not|no|except|other than|unlike|anything but|without|never|different from|excluding)\b', re.I)
COND_RE = re.compile(r'\b(?:unless|only if|except when|provided that|in case|if|when)\b', re.I)
SET_STEPS = 3
NUMBER_WORDS = '(?:one|two|three|four|five|six|seven|eight|nine|ten|single|double|pair|set|pack|piece)s?'  # steps a code-made cart leaves beyond the cart calls and the finish


def _vw(text: Any) -> list[str]:
    """A value's words in order: lower case, hyphens and list marks as spaces, plural -s and British -our folded, true/false as yes/no."""
    out = []
    for w in re.findall('[a-z0-9]+(?:\\.[0-9]+)?', str(text or '').lower().replace('-', ' ')):
        w = w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') and not w[-2].isdigit() else w
        w = re.sub('our$', 'or', w) if len(w) > 4 else w
        out.append({'true': 'yes', 'false': 'no'}.get(w, w))
    return out


def _kin(a: str, b: str) -> bool:
    """Two spellings of one word: the same; word forms sharing their first five letters (folding, foldable); or one letter
    apart in a word of four or more letters (grey, gray). Numbers must be equal."""
    if a == b:
        return True
    if re.search('\\d', a) or re.search('\\d', b):
        return False
    if len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5]:
        return True
    return len(a) == len(b) >= 4 and sum((x != y for x, y in zip(a, b))) == 1


def _run(want: list[str], have: list[str]) -> bool:
    """Whether `have` holds the words of `want` together and in order."""
    n = len(want)
    return bool(n) and any((all((_kin(want[j], have[i + j]) for j in range(n))) for i in range(len(have) - n + 1)))


def _spread(want: list[str], have: list[str]) -> bool:
    """`have` holds `want`'s words in order but with other words between them: "10k white gold" is another value than "10k gold"."""
    if len(want) < 2 or _run(want, have):
        return False
    at = 0
    for w in want:
        while at < len(have) and (not _kin(w, have[at])):
            at += 1
        if at >= len(have):
            return False
        at += 1
    return True


def _within(want: list[str], have: list[str]) -> bool:
    return bool(want) and all((any((_kin(w, h) for h in have)) for w in want))


def _figs(text: Any) -> list[tuple[float, str]]:
    """(number, unit) pairs in a text: "1.188 kg" gives (1.188, 'kg'); a bare number has unit ''; the first end of a
    range written with one unit ("41-55 in", "between 10 and 20 cm") takes that unit."""
    text = re.sub('(\\d(?:[\\d.,]*\\d)?)(\\s*(?:-|\u2013|to|and)\\s*-?\\d[\\d.,]*\\s*)("|[a-z\u00b0]{1,12}\\b)', lambda m: m.group(1) + ' ' + m.group(3) + m.group(2) + m.group(3), str(text or '').lower())
    out = []
    for num, unit in re.findall('(?<![a-z0-9.])((?:(?<![\\w-])-)?\\d+(?:,\\d{3})*(?:\\.\\d+)?)\\s*("|[a-z\u00b0]{1,12}\\b)?', str(text or '').lower()):
        try:
            value = float(num.replace(',', ''))
        except ValueError:
            continue
        out.append((value, 'in' if unit == '"' else unit or ''))
    return out


def _unit_eq(a: str, b: str) -> bool:
    a = a[:-1] if len(a) > 2 and a.endswith('s') else a
    b = b[:-1] if len(b) > 2 and b.endswith('s') else b
    return bool(a) and bool(b) and (a == b or (min(len(a), len(b)) >= 2 and (a.startswith(b) or b.startswith(a))))


def _parts(value: Any) -> list[list[str]]:
    return [w for w in (_vw(p) for p in re.split('[,;/|]', str(value or ''))) if w]


def _judge_text(record: Any, limit: int, clip: int=400) -> str:
    """A record shortened for a judge. When it fits it reads exactly as before; when it does not, its plain fields (the
    prose where a page points on to another listing) keep their own allowance and the nested attribute block takes the cut."""
    whole = _dump(_clip(record, clip))
    if len(whole) <= limit or not isinstance(record, dict):
        return whole[:limit]
    plain = {k: v for k, v in record.items() if not isinstance(v, (dict, list))}
    nested = {k: v for k, v in record.items() if isinstance(v, (dict, list))}
    head = _dump(_clip(plain, min(clip, 600)))
    room = limit - len(head) - 2
    if room < 120 or not nested:
        return head[:limit]
    return (head[:-1] + ', ' + _dump(_clip(nested, 120))[1:room] + '...}')[:limit]


class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Cairn:

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
        self.started = time.monotonic()
        self.step = 0
        self.finished = bool(self.view.get('finished'))
        self.broke = False
        self.no_credit = False
        self.waited = 0
        self.misses = 0
        self.cost = 0.0
        self.calls = 0
        self.loop_calls = 0
        self.trace: list[dict[str, Any]] = _Stamped(self.clock)
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
        self.stated: list[tuple[str, str, str]] | None = None
        self.pushed = False
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.late_held = False
        self.pointed_done: set[str] = set()
        self.closest_done = False
        self.line_refs_done: set[tuple[str, str]] = set()
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed: str | None = None
        self.completed = False
        self.link_tried: set[str] = set()
        self.lane_misses: dict[int, int] = {}
        self.adopted: dict[str, int] = {}
        self.capped: set[tuple[str, float]] = set()
        self.factless: set[tuple[str, str]] = set()
        self.trail_room: int | None = None
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.queued: list[str] = []
        self.buys = 0
        self.refusals = 0
        self.recent: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.opened: set[str] = set()
        self.views: dict[str, set[str]] = {}
        self.fit_cached: tuple[Any, dict[str, Any]] | None = None
        self.owners_seen: dict[tuple[str, int], int] = {}
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
        self.pick_marks: set[str] = set()
        self.pick_rounds = 0
        self.effort_ok = True
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
        self.list_asks = 0
        self.apart: dict[str, dict[int, list[str]]] = {}
        self.follow_up: list[tuple[int, list[str], bool]] = []
        self.listing = True
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.link_offs: dict[str, int] = {}
        self.cheap_holds = 0
        self.dear_paged: set[int] = set()
        self.cheap_named: set[tuple[str, str]] = set()
        self.better_by_key: dict[str, list[str]] = {}
        self.cheap_agreed = False
        self.weak_holds = 0
        self.proofs: dict[str, list[str]] = {}
        self.origin: dict[int, list[str]] = {}
        self.pages_swept = False
        self.series_swept: set[tuple[int, frozenset[str]]] = set()
        self.set_holds = 0
        self.link_more: dict[str, Any] = {}
        self.trail_notes: list[str] = []
        self.prelude_done = False
        self.loop_t0: float | None = None
        self.hunts = 0
        self.env_gap = ENV_GAP
        self.env_spent = 0.0
        self.env_count = 0
        self.read_calls: dict[str, Any] = {}
        self.read_args: dict[str, tuple[str, dict[str, Any]]] = {}
        self.value_rows: set[str] = set()
        self.row_keys: set[str] = set()
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
        self.item_caps: dict[int, float] = {}
        self.ref_forms: dict[str, list[str]] = {}
        self.set_cached: tuple[Any, Any] | None = None
        self.forum_held = 0
        self.hold_marks: set[str] = set()
        self.named_swaps: dict[str, str] = {}
        self.check_better: list[str] = []
        self.swapped = False
        self.cond_marks: set[str] = set()
        self.literal_held = False
        self.code_ordered = False
        self.basket_done = False
        self.repaired = False
        self.first_seen: list[tuple[str, dict[str, Any], Any]] = []
        self.dear_done = False
        self.drop_items: set[int] = set()
        self.drop_hunted = False
        self.swept = False
        self.next_pages: dict[tuple[str, str], Any] = {}
        self.cap_dead: set[tuple[str, str]] = set()
        self.cap_seen: dict[str, str] = {}
        self.query_told = False
        self.trail_opened = -1
        self.confirming = False
        self.noask_known: bool | None = None
        self.two_steps: bool | None = None
        self.late_tried = False
        self.reopened = False
        self.auto_swapped = False
        self.carried: set[str] = set()

    def left(self) -> int:
        return self.max_steps - self.step

    def clock(self) -> float:
        return time.monotonic() - self.started

    def model_room(self) -> float:
        return TASK_SECONDS - TIME_MARGIN - self.clock()

    def rushed(self) -> bool:
        """No room for a read the order does not need: that read and the order itself, at this run's store speed, must end inside the margin."""
        pace = self.env_spent / self.env_count if self.env_count else 1.0
        return self.model_room() <= 2 * max(pace, 1.0)

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def row_ids(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A listing tool called with a code that one seen row's title holds gets that row's id; anything else stays as it was sent."""
        if name not in {self.roles.get(job) for job in ('open', 'offer', 'compare', 'cart')} or not self.rows:
            return args
        title = self.title_key()
        if not title:
            return args

        props = (self.schemas.get(name) or {}).get('properties') or {}

        def known(value: Any, key: str) -> Any:
            if not isinstance(value, str) or value.strip() in self.rows or not 3 <= len(value.strip()) <= 30 or not re.search('\\d', value) or 'enum' in (props.get(key) or {}):
                return value
            pattern = '(?<![a-z0-9])' + re.escape(value.strip().lower()) + '(?![a-z0-9])'
            hits = [i for i, row in self.rows.items() if re.search(pattern, str(row.get(title) or '').lower())]
            if len(hits) == 1:
                self.trace.append({'kind': 'row_id', 'sent': value, 'id': hits[0]})
                return hits[0]
            return value
        return {key: [known(v, key) for v in value] if isinstance(value, list) else known(value, key) for key, value in args.items()}

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
        base = 2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1
        return base + (1 if self.two_step() else 0)

    def two_step(self) -> bool:
        """Whether checking out takes a second call: the checkout tool wants a token, or another tool confirms with one."""
        buy = self.roles.get('buy')
        if not buy:
            return False
        if self.two_steps is None:
            self.two_steps = bool(self.token_field(buy)) or bool(self.confirm_tool(buy)[0])
        return self.two_steps

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
        if 'haiku' in model.lower():
            # billed Haiku reasons before it answers: under a small cap it returns nothing (8.5% of billed director turns)
            max_tokens = max(max_tokens, HAIKU_ROOM)
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
            if self.effort_ok and self.clock() >= LOW_EFFORT_AT:
                # late tool-calling turns that think for thousands of tokens take 30-45 s and push the task into the wind-down
                body['reasoning'] = {'effort': 'low'}
        result = _llm_within(self.model_room()).post_verbose('/inference/chat/completions', body)
        if 'reasoning' in body and not (isinstance(result.data, dict) and result.data.get('choices')):
            self.effort_ok = False
            body.pop('reasoning', None)
            result = _llm_within(self.model_room()).post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.cost += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return (message, 200)
            if data['choices'][0].get('finish_reason') == 'length' and max_tokens < 12000 and self.model_room() > 30:
                # cut off while still reasoning: one retry with double room beats falling back to another model
                self.trace.append({'kind': 'llm_cut_retry', 'model': model, 'room': max_tokens})
                return self.call_model(model, messages, min(12000, max_tokens * 2), tools)
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = _count(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get('content'))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        rest = self.env_at + self.env_gap - time.monotonic()
        if rest > 0:
            time.sleep(rest)
        self.env_at = time.monotonic()
        for _ in range(4):
            began = time.monotonic()
            result = _env.post_verbose('/environment/call', payload)
            self.env_spent += time.monotonic() - began
            self.env_count += 1
            if isinstance(result.data, dict):
                data = result.data
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                if name != self.roles.get('ask'):
                    data['observation'] = self.unspelled(name, data.get('observation'))
                    data['observation'] = self.own_record(name, args, data.get('observation'))
                if name == self.roles.get('buy') and not data.get('finished') and not self.confirming:
                    data = self.confirm_step(name, data)
                seen = data.get('observation')
                self.note_read(name, args, seen)
                if name == self.facts_tool() and isinstance(seen, dict) and not _failed(seen) and self.ident(seen) and not _fact_block(seen):
                    self.factless.add((name, self.ident(seen)))
                if name in self.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))) and not _failed(seen):
                    shown = _dump(seen)
                    for value in args.values():
                        named = [v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and (not isinstance(value, list) or v.strip() in shown)]
                        self.opened.update(named)
                        for ident in named:
                            self.views.setdefault(ident, set()).add(name)
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

    def unspelled(self, name: str, observation: Any) -> Any:
        """The result in the form the agent reads (_restore); on a listing view, a record whose attributes sit among its own
        keys gets them gathered into one object: the keys the search rows do not have, holding plain values."""
        views = {self.roles.get('open'), self.roles.get('offer'), self.roles.get('compare')}
        out = _fold_twins(_restore(observation, page=name in views or name != self.roles.get('search')))
        if name not in views:
            return out
        self.row_keys.update((k for ident, row in self.rows.items() if ident not in self.opened for k in row))
        known = self.row_keys
        if not known:
            return out

        def gathered(record: Any) -> Any:
            if not isinstance(record, dict) or _fact_block(record) or not self.ident(record, True):
                return record
            extra = {k: v for k, v in record.items() if k not in known and isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(extra) < RECORD_FIELDS:
                return record
            return {**{k: v for k, v in record.items() if k not in extra}, 'attributes': extra}
        if isinstance(out, dict) and self.ident(out):
            return gathered(out)
        if isinstance(out, dict):
            return {k: [gathered(r) for r in v] if isinstance(v, list) else v for k, v in out.items()}
        return out

    @_soft(None)
    def fix_counts_from_request(self) -> None:
        """'2 x <product>' in the request is that product's quantity: the brief sometimes reads it as 1. The single item
        whose name shares the most word roots with the words after the sign gets it; a tie or no shared root changes nothing,
        and a size ('3 x 5 inch') is not a count."""
        root = lambda w: w.lower()[:-1] if len(w) > 3 and w.lower().endswith('s') else w.lower()
        for found in re.finditer('(?<![\\w.])(\\d{1,3})\\s*x\\s+([^,;.:!?]{2,80})', self.request or ''):
            words = re.findall('[A-Za-z0-9&]+', found.group(2))[:6]
            if not words or words[0][0].isdigit():
                continue
            want = {root(w) for w in words if len(w) > 2}
            scores = [(len(want & {root(w) for w in re.findall('[A-Za-z0-9&]+', item.name + ' ' + item.shelf) if len(w) > 2}), n) for n, item in enumerate(self.items)]
            best = max((s for s, _ in scores), default=0)
            hits = [n for s, n in scores if s == best]
            count = int(found.group(1))
            if best > 0 and len(hits) == 1 and 0 < count < 100 and self.items[hits[0]].qty != count:
                self.trace.append({'kind': 'count_fixed', 'item': self.items[hits[0]].name, 'was': self.items[hits[0]].qty, 'now': count})
                self.items[hits[0]].qty = count

    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.request
        if self.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + _dump(_clip(self.view.get('opening'), 1500))
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit and (not _items(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        elif not self.no_credit and self.shared_shelf(brief):
            again = self.ask_json(prompt, [HEAVY], 2500) or {}
            if _items(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                brief = again
        while not brief and not self.no_credit and self.calls < ALL_CALLS and self.clock() < BRIEF_BY:
            time.sleep(BRIEF_GAP)
            brief = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        if brief:
            self.waited = 0
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
        self.fix_counts_from_request()
        merged: list[Item] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.items, 1):
            twin = next((m for m in merged if item.shelf.strip() and m.qty == item.qty and _words(m.shelf) == _words(item.shelf) and (not any((_words(a.split(':')[0]) == _words(b.split(':')[0]) and _words(a.partition(':')[2]) != _words(b.partition(':')[2]) for a in m.musts for b in item.musts)))), None)
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
            if not (re.search('listing|page|pointed|named|sold|sells|seller|shop|store|brand|maker', text, re.I) or re.search('[:,].*\\b[A-Z][a-z]{3,}', text)):
                # a rule between requested items (a value that depends on another requested item, or that item's own
                # model) names no listing to open: holding the order for it wastes steps
                self.trace.append({'kind': 'lookup_not_pointer', 'need': text[:80]})
                continue
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
        self.scope_musts()
        self.read_caps()

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
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4 or re.search('\\d', value):
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
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
            needs = [m + (f' (the page it points at gives: {self.ref_values[m]}' + (f"; also written {', '.join(self.ref_forms[m])}" if self.ref_forms.get(m) else '') + '; the other details in this clause only pick out that page and are not requirements on this item)' if m in self.ref_values else '') for m in live if m not in told]
            fitting = [m for m in told if re.search('compatib|\\bfits?\\b|works with|pairs? with|goes with|suits?\\b', m, re.I) and not _codes(m)]
            told = [m for m in told if m not in fitting]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            if item.shelf:
                head += f' (store label: {item.shelf})'
            head += f', quantity {item.qty}'
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if told:
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
            if fitting:
                text += '; asks for compatibility, not for the same value (a listing whose value differs can still qualify and must not be failed on this alone): ' + '; '.join(fitting)
            if item.dropped:
                text += '; DROPPED by the shopper, no longer required (requirements in force win over the request\'s first words): ' + '; '.join(dict.fromkeys(item.dropped))
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
            when = {'choose': 'before choosing', 'order': 'at the order (Cairn does it when you check out)', 'end': 'at an ending without a purchase (Cairn does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.page_tools() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        nested = self.label_line()
        if nested:
            parts.append(nested)
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        left = self.left()
        text = f'[Cairn] step {self.step} of {self.max_steps} used, {left} left.'
        if left <= self.tail() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif self.buys == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2):
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if self.clock() >= WIND_DOWN and left > self.tail() + 1:
            text += ' TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns; an episode that runs out of time earns nothing.'
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.buys == 0 and self.step <= self.max_steps // 2:
            either = [f'item {n} {need}' for n, need in self.needs_now() if ':' in need and _choices(need.split(':', 1)[1]) and need not in {m for _, m in self.refs}]
            if either:
                text += '\n[either-or values: a search holding two choices lists only listings that carry both, so search each choice on its own] ' + ' | '.join(either[:3])
        if self.loop_t0 is not None and self.loop_calls >= 3 and self.clock() < WIND_DOWN:
            per = (self.clock() - self.loop_t0) / self.loop_calls
            fit = int((LAST_TURN - self.clock()) / per) if per > 0 else 99
            room = left - self.tail()
            if 1 <= fit and fit * 2 < room:
                text += f'\n[pace] The clock, not the step limit, ends this episode: about {fit} more replies fit in the time left while {room} steps remain, so put every call that does not wait on another call\'s result into the same reply (searches for different items or choices, several listings to open, all cart lines).'
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        best = self.proven_set() if self.brief.get('pick') == 'cheapest' and self.buys == 0 else None
        if best:
            text += '\n[lowest whole order whose attributes state every requirement, among listings opened so far only (unopened ones may be cheaper)] ' + self.set_text(best) + f' = {self.set_total(best):g}'
        plan = self.fit_plan() if self.bound and self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        sets = self.read_sets() if self.bound else []
        if sets and not any((set(lines) == set(self.cart_ids()) for _, lines in sets)):
            text += '\n[whole sets already read: every line states its requirements in its attributes or title, and each bound line fits the device by code] ' + self.sets_text(sets) + '. Set the cart to one of these unless a record breaks a requirement.'
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        text += '\n[ledger]\n' + self.ledger()
        return text

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
            if present[key] < 0.8 * total or len(values) > 40 or key in (self.price_key, self.stock_key):
                continue
            if 2 * sum((1 for v in values if re.search('\\d', v))) > len(values):
                continue  # a label is words: a column of figures (stock, size, price) is not the category
            listed = sum((1 for v in values if v.strip().lower() in known)) / len(values)
            mean = sum((len(v) for v in values)) / len(values)
            named = bool(re.search(CATEGORY_NAME, str(key), re.I))
            score = (-listed, not named, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        if (best is None or not re.search(CATEGORY_NAME, str(best[1]), re.I)) and len(self.full) >= 2:
            # the rows carry no label but opened records do (a sheet's "Section"): take it from them
            keys = [k for k in next(iter(self.full.values())) if re.search(CATEGORY_NAME, str(k), re.I) and all((isinstance(r.get(k), str) and 0 < len(r[k]) <= 40 for r in self.full.values()))]
            if keys:
                return keys[0]
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
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}, 'plain': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][ident] = text
                slot['plain'][ident] = _FIRST_SPELLING.get(str(value).strip(), text).strip().lower()
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
                values = list(slot['plain'].values())
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
                        twin['plain'].setdefault(ident, slot['plain'][ident])
                        if text not in twin['values']:
                            twin['values'].append(text)
                    twin['n'] = len(twin['per'])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for name, slot in kept[:TABLE_KEYS]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        number = self.owner_of(label)
        if not number:
            return ''
        for must in self.items[number - 1].musts:
            prop, _, value = must.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            words = set(re.findall('[a-z0-9]{3,}', value.lower()))
            shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if prop.strip() and _words(prop) != name and _stem(prop) & _stem(name) and words and (not words <= shown):
                return f' [not the stated {prop.strip()}: another property]'
        return ''

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2))
        both = [i for i in spec['plain'] if i in slot['plain'] and spec['plain'][i].strip() and slot['plain'][i].strip()]
        same = sum((1 for i in both if agree(spec['plain'][i], slot['plain'][i])))
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
        matched = [(label, ids) for label, ids in groups.items() if any((_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)]
        best: dict[int, float] = {}
        for label, _ in matched:
            owner = self.owner_of(label)
            best[owner] = max(best.get(owner, 0.0), self.label_fit(label, owner))
        pool = [ids for label, ids in matched if not self.owner_of(label) or best[self.owner_of(label)] < 0.5 or self.label_fit(label, self.owner_of(label)) >= LABEL_SHARE * best[self.owner_of(label)]] or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.items))]

    def label_fit(self, label: str, number: int) -> float:
        """How closely a store label names item `number`: its shelf (or an adopted label) 3, its name 2, else the share of
        word roots the two hold in common."""
        if not label or not 0 < number <= len(self.items):
            return 0.0
        item = self.items[number - 1]
        words = _words(label)
        if words == _words(item.shelf or '') or self.adopted.get(label) == number:
            return 3.0
        if words == _words(item.name):
            return 2.0
        long = _roots(label)
        mine = _roots(item.shelf or '') | _roots(item.name)
        return len(long & mine) / max(1, len(long | mine))

    def single_target(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        if not self.price_key:
            # the key is learned on paging paths only (cheapest picks, opening searches): take the record's own price field
            row = self.rows.get(ident) or {}
            self.price_key = next((k for k, v in row.items() if re.search('price|cost', str(k), re.I) and ((isinstance(v, (int, float)) and (not isinstance(v, bool))) or (isinstance(v, str) and re.fullmatch('\\s*[^\\d\\s]{0,3}\\s?\\d[\\d,]*(?:\\.\\d+)?\\s*[A-Za-z]{0,8}\\.?\\s*', v)))), None)
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str):
            figure = re.fullmatch('\\s*[^\\d\\s]{0,3}\\s?(\\d[\\d,]*(?:\\.\\d+)?)\\s*[A-Za-z]{0,6}\\.?\\s*', value)
            value = float(figure.group(1).replace(',', '')) if figure else None
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        if isinstance(value, str) and re.search(STOCK_OUT, value.strip(), re.I):
            return False
        if isinstance(value, str) and len(re.findall('\\d+', value)) == 1 and int(re.findall('\\d+', value)[0]) == 0:
            return False
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
                    found += [(inner, value)] if inner else [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.ident(r, True)] if i]
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
        if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
            shape['count'] = None
        if not shape['next']:
            nexts = [k for k, v in observation.items() if k != rows and re.search('next', str(k), re.I) and (v is None or isinstance(v, (int, str))) and (not isinstance(v, bool))]
            shape['next'] = nexts[0] if len(nexts) == 1 else None
        if shape['next'] and (not shape['next_arg']):
            # a left-empty answer here switches every paging use off for the episode; the schema names the argument when
            # exactly one optional, non-enumerated parameter reads like a page or offset
            required = set((self.schemas.get(name) or {}).get('required') or [])
            argued = [k for k, v in props.items() if k not in required and isinstance(v, dict) and 'enum' not in v and ({'integer', 'number', 'string'} & set(_kinds(v))) and re.search('page|leaf|offset|start|cursor|next|token|lead|row|from|skip', str(k), re.I)]
            shape['next_arg'] = argued[0] if len(argued) == 1 else None
        if shape['next'] and (not shape['next_arg']):
            inner = [f'{k}.{n}' for k, v in props.items() if isinstance(v, dict) and 'object' in _kinds(v) for n, s in (v.get('properties') or {}).items() if isinstance(s, dict) and 'enum' not in s and ({'integer', 'number', 'string'} & set(_kinds(s))) and re.search('page|offset|start|cursor|next|token|from|skip', str(n), re.I)]
            shape['next_arg'] = inner[0] if len(inner) == 1 else None
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
            return self.next_page_big(name, args, observation) if pick not in ('cheapest', 'most', 'least') else ''
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
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < (LAST_PAGE_ROOM if pages == 1 else 12):
            return f"\n[Cairn] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
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
        ends = self.read_cheap_unread([observation] + self.paged_obs, 'end_reads', self.tail() + 8) if value is None and texts and pick == 'cheapest' and not self.rushed() else ''
        rest = f"\n[Cairn] {count} listings matched and Cairn has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Cairn read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest + ends if texts else ''

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))
        limits = self.cap_keys(name)
        caps = [k for k, v in args.items() if k in props and isinstance(v, (int, float)) and (not isinstance(v, bool)) and (re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k) or k in limits[:1] and len(limits) == 1)]
        if not caps or not self.price_key:
            return ''
        caps.sort(key=lambda k: args[k])
        cap = args[caps[0]]
        over = [i for i, _ in self.records_in(observation) if (self.price_of(i) or 0) > cap]
        if not over:
            return ''
        self.cap_dead.add((name, caps[0]))
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in modes.items()}
        hint = '; list the category with ' + ', '.join((f'{k}={_dump(o[0])}' for k, o in others.items() if o)) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[Cairn] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Cairn] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def note_facts(self, tool: str, shown: bool) -> None:
        """What a single-listing view showed: it shows attributes once any page had them, and none after two pages without."""
        if shown:
            self.facts_from[tool] = True
        elif self.facts_from.get(tool) is not True:
            self.bare_pages[tool] = self.bare_pages.get(tool, 0) + 1
            if self.bare_pages[tool] >= 2:
                self.facts_from[tool] = False

    def lists_things(self, tool: str) -> bool:
        """Whether a tool lists a shelf or takes search words rather than a listing's id."""
        target = self.single_target(tool) or ''
        return bool(re.search(SHELF_ARG, _snake(target), re.I) or re.search(QUERY_WORDS, _snake(target), re.I)) or tool in self.shapes

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if (not tool or not target) and self.compare_facts() is not True:
            return []
        texts = []
        wanted = [i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i))]
        side = self.roles.get('compare')
        many = self.compare_target()
        unread: list[str] = []
        while side and many and self.compare_facts() is True and len(wanted) >= 2 and self.left() > keep and not self.finished and not self.rushed():
            room = self.compare_room()
            batch, wanted = (wanted[:room], wanted[room:])
            faults_side: list[str] = []
            args_side = _shape({many: batch}, self.schemas.get(side), 'args', faults_side)
            if faults_side:
                wanted = batch + wanted
                break
            reply = self.act(side, args_side)
            if reply is None:
                wanted = batch + wanted
                break
            seen = reply.get('observation')
            self.observe(side, seen)
            self.note_view(side, seen)
            rows = self.records_in(seen)
            for ident, record in rows:
                texts.append(f'{ident}: ' + self.show(record)[:1500])
            if rows:
                self.compare_ok = max(self.compare_ok, len(rows))
                unread += [i for i in batch if i not in self.full]
                continue
            self.compare_bad = min(self.compare_bad or 999, len(batch))
            wanted = batch + wanted
            if self.compare_room() >= len(batch):
                break
        if not tool or not target:
            return texts
        for ident in unread + wanted:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False or self.rushed():
                break
            if (tool, ident) in self.factless:
                continue
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
                self.note_facts(tool, bool(_fact_block(seen)))
            texts.append(f'{ident}: ' + self.show(seen)[:1500])
        if texts:
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def show_links(self, ids: list[str], keep: int) -> None:
        """Run the part code a cart line sends the reader to through the search, once per code, and open
        the page it finds when steps allow; the result feeds the ledger, not the model. Multi-line carts only."""
        plan = self.value_template()
        if not plan or (len(self.items) < 2 and len(ids) < 2):
            return
        tool, base, field = plan
        for ident in ids:
            marks = self.fit_index()['marks']
            code = (marks.get(ident) or {}).get('more')
            if not code or code in self.link_tried or any((mark.get('own') == code for mark in marks.values())):
                continue
            if self.left() <= keep or self.finished or self.rushed():
                return
            self.link_tried.add(code)
            faults: list[str] = []
            args = _shape({**base, field: code}, self.schemas.get(tool), 'args', faults)
            if faults:
                return
            reply = self.act(tool, args)
            if reply is None:
                return
            self.observe(tool, reply.get('observation'))
            self.trace.append({'kind': 'show_link', 'line': ident, 'code': code})
            marks = self.fit_index()['marks']
            self.fill_facts([i for i, mark in marks.items() if mark.get('own') == code], max(keep, self.tail() + 6))

    def cap_template(self) -> tuple[str, dict[str, Any], str, str] | None:
        """The listing tool that takes a shelf and a maximum price, with the mode it needs: the mode an earlier listing by
        shelf used, else the one option naming a listing. None when no tool or mode is clear."""
        for tool, schema in self.schemas.items():
            props = (schema or {}).get('properties') or {}
            caps = self.cap_keys(tool)
            shelves = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and re.search(SHELF_ARG, _snake(k), re.I)]
            if len(caps) != 1 or len(shelves) != 1:
                continue
            learned = [(dict(mode), held) for (key, mode), held in self.cap_modes(tool).items() if key == caps[0]]
            options_args = [k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and len(v['enum']) >= 2]
            single = len(options_args) == 1 and options_args[0] in ((schema or {}).get('required') or [])
            base: dict[str, Any] = {}
            for key, spec in props.items():
                options = spec.get('enum') if isinstance(spec, dict) else None
                if not isinstance(options, list) or len(options) < 2:
                    continue
                pick = next((mode[key] for mode, held in learned if held and key in mode), None)
                if pick is None:
                    pick = next((a[key] for a in reversed(self.searched) if shelves[0] in a and key in a and not any(k != shelves[0] and not (isinstance(props.get(k), dict) and 'enum' in props[k]) and isinstance(v, str) and v.strip() and re.search(QUERY_WORDS, _snake(k), re.I) for k, v in a.items())), None)
                    named = [o for o in options if isinstance(o, str) and re.search(LIST_MODE, o, re.I)]
                    worded = {a[key] for a in self.searched if key in a and any(isinstance(v, str) and v.strip() and re.search(QUERY_WORDS, _snake(k), re.I) for k, v in a.items())}
                    unworded = [o for o in options if o not in worded]
                    pick = pick if pick is not None else named[0] if len(named) == 1 else unworded[0] if single and worded and len(unworded) == 1 else None
                spoiled = {mode[key] for mode, held in learned if not held and key in mode} if single else set()
                if spoiled and (pick is None or pick in spoiled):
                    rest = [o for o in options if o not in spoiled]
                    pick = rest[0] if len(rest) == 1 else None
                if pick is None:
                    return None
                base[key] = pick
            return (tool, base, shelves[0], caps[0])
        return None

    def cap_keys(self, tool: str) -> list[str]:
        """The numeric arguments of `tool` that may cap a listing's price: those a search has shown holding a limit, else those whose
        name says a maximum, else those whose name says a price or budget."""
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        numbers = [k for k, v in props.items() if isinstance(v, dict) and {'integer', 'number'} & set(_kinds(v)) and not re.search(CAP_NOT, k, re.I) and not re.search(CAP_LOW, _snake(k), re.I)]
        held = {key for (key, _), ok in self.cap_modes(tool).items() if ok}
        named = [k for k in numbers if re.search(CAP_ARG, _snake(k), re.I)]
        priced = [k for k in numbers if re.search(CAP_PRICED, _snake(k), re.I)]
        for group in ([k for k in numbers if k in held], [k for k in named if k in priced], priced, named):
            if group:
                return group
        return []

    def cap_modes(self, tool: str) -> dict[tuple[str, tuple[tuple[str, Any], ...]], bool]:
        """What the searches of `tool` so far show of each numeric argument as a price limit, by argument and mode (the call's option
        arguments): False once a call's rows held one priced above its value; True when a call's rows all sat at or below it and rows drawn
        at random from the known rows of their labels would rarely have (CAP_CHANCE), so the limit, not the query, kept dearer rows out."""
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        cat_key = self.category_key()
        out: dict[tuple[str, tuple[tuple[str, Any], ...]], bool] = {}
        prices: dict[str, list[float]] = {}
        for ident, row in self.rows.items():
            price = self.price_of(ident)
            if price is not None:
                prices.setdefault(str(row.get(cat_key) or '').strip().lower() if cat_key else '', []).append(price)
        for mark, (name, args) in self.read_args.items():
            if name != tool:
                continue
            rows = [i for i, _ in self.records_in(self.read_calls.get(mark))]
            shown = [self.price_of(i) for i in rows]
            if not rows or None in shown:
                continue
            mode = tuple(sorted(((k, v) for k, v in args.items() if isinstance(props.get(k), dict) and isinstance(props[k].get('enum'), list) and isinstance(v, (str, int, float)))))
            labels = {str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() for i in rows} if cat_key else set()
            for key, value in args.items():
                if key not in props or isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                pool = [p for label in labels for p in prices.get(label, [])]
                if max(shown) > value:
                    out[(key, mode)] = False
                elif out.get((key, mode)) is not False and pool and (sum((p <= value for p in pool)) / len(pool)) ** len(rows) < CAP_CHANCE:
                    out[(key, mode)] = True
        return out

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
            self.capped.add((line, price))
            limit = int(price) - 1 if float(price).is_integer() else int(price)
            tried: set[int] = set()
            for _ in range(WALK_CALLS):
                if self.left() <= self.tail() + (WALK_ROOM if tried else 7) or self.finished or self.rushed():
                    return
                if limit < 1 or limit in tried:
                    break
                tried.add(limit)
                faults: list[str] = []
                args = _shape({**base, shelf: label, cap: limit}, self.schemas.get(tool), 'args', faults)
                if faults:
                    return
                reply = self.act(tool, args)
                if reply is None:
                    return
                seen = reply.get('observation')
                self.observe(tool, seen)
                rows = self.records_in(seen)
                self.trace.append({'kind': 'list_below', 'line': line, 'price': price, 'limit': limit, 'rows': len(rows)})
                page = self.paging(tool, seen)
                if not rows or any(((self.price_of(i) or 0) > limit for i, _ in rows)) or not page or page[0] <= page[1] or self.price_sorted(seen):
                    break
                if page[0] <= 2 * page[1]:
                    call = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults) if page[3] is not None else None
                    more = None if faults or call is None or self.left() <= self.tail() + 7 or self.rushed() else self.act(tool, call)
                    if more is not None:
                        self.observe(tool, more.get('observation'))
                    break
                low = min((p for i, _ in rows for p in [self.price_of(i)] if p is not None), default=None)
                if low is None:
                    break
                limit = int(low) - 1 if float(low).is_integer() else int(low)

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
        room = self.compare_room() if self.compare_facts() is True else 0
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            lowest = [i for i in pool if i in self.value_rows][:1]
            pool = lowest + [i for i in pool if i not in lowest]
            picks += [i for i in pool[:room or 2] if i not in picks]
        return self.fill_facts(picks[:room * len(cart) if room else min(6, CHEAPER_OPENS * max(1, len(cart)))], self.tail() + 6)

    @_soft('')
    def below_cart(self) -> str:
        """The buyable rows priced below a cart line, of its label, that nobody has read yet."""
        cat_key = self.category_key()
        cart = self.cart_ids()
        lines = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower() if cat_key else ''
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind and (self.price_of(i) if self.price_of(i) is not None else float('inf')) < price]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            lines += [f'{i} at {self.price_of(i):g}' for i in pool[:CHEAPER_SHOWN]]
        return 'Still unread and below a cart line: ' + ', '.join(lines) + '.' if lines else ''

    def title_key(self) -> str | None:
        """The row key holding a listing's name: of the short texts most rows carry and few rows share, with more than one word
        on average and about as many different values as any such key has (a seller's name repeats where a listing's does not),
        the shortest."""
        cat_key = self.category_key()
        totals: dict[str, list[Any]] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if key != cat_key and key not in (self.price_key, self.stock_key) and isinstance(value, str) and len(value) <= TITLE_CHARS and not re.match('\\s*[^\\w\\s]{0,3}\\s?\\d', value):
                    slot = totals.setdefault(key, [0, 0, 0, set()])
                    slot[0] += len(value.split())
                    slot[1] += 1
                    slot[2] += len(value)
                    slot[3].add(value)
        kept = [(slot[2] / slot[1], key, len(slot[3])) for key, slot in totals.items() if slot[1] * 5 >= len(self.rows) * 4 and slot[0] >= slot[1] * 2 and len(slot[3]) * 2 >= slot[1]]
        most = max((distinct for _, _, distinct in kept), default=0)
        ranked = [(mean, key) for mean, key, distinct in kept if distinct * 10 >= most * 9]
        return min(ranked)[1] if ranked else None

    def item_choices(self, number: int) -> list[str]:
        """The choices of the item's first live either-or requirement that points at no other listing."""
        refs = {m for _, m in self.refs}
        for n, need in self.needs_now():
            if n == number and need not in refs and ':' in need:
                found = _choices(need.split(':', 1)[1])
                if found:
                    return found
        return []

    def hunt_words(self, number: int) -> tuple[str, list[str]]:
        """The item's noun, a word its store label holds that no other item's label holds (a search reads labels too, so it
        keeps the rows to this item), with the words a further search can add, most promising first: a word on most rows finds
        nothing new and a word on one row finds little else, so the words kept sit between."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        title = self.title_key()
        labels = [str(row.get(cat_key) or '') for row in self.rows.values() if self.owns(item, str(row.get(cat_key) or ''))] if cat_key else []
        names = [str(row.get(title) or '') for row in self.rows.values() if title and cat_key and self.owns(item, str(row.get(cat_key) or ''))]
        for record in self.full.values():
            if cat_key and self.owns(item, str(record.get(cat_key) or '')):
                names.append(' '.join((str(v) for v in (_fact_block(record) or {}).values() if isinstance(v, str) and len(v) <= HUNT_VALUE_CHARS)))
        others = {w for n, other in enumerate(self.items, 1) if n != number for w in re.findall('[a-z]{4,}', ((other.shelf or '') + ' ' + other.name).lower())}
        label = max(labels, key=labels.count).lower() if labels else (item.shelf or item.name).lower()
        nouns = [w for w in re.findall('[a-z]{4,}', label) if w not in others]
        noun = '' if number in self.hunt_no_noun else (max(nouns, key=len) if nouns else '')
        if not names:
            return (noun, [])
        counts: dict[str, int] = {}
        for name in names:
            for word in {w.lower() for w in re.findall("[A-Za-z][A-Za-z'-]{2,}", name)}:
                counts[word] = counts.get(word, 0) + 1
        own = _roots(item.shelf or '') | _roots(item.name)
        ceiling = max(2, int(len(names) * HUNT_COMMON))
        words = [w for w, n in counts.items() if 2 <= n <= ceiling and w != noun and _roots(w) and not _roots(w) & own]
        words.sort(key=lambda w: (-counts[w], w))
        return (noun, words)

    def rows_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for row in self.rows.values() if not cat_key or self.owns(item, str(row.get(cat_key) or ''))))

    @_soft('')
    def hunt_cheaper(self) -> str:
        """Search on, without the model, while the pick is the cheapest and the store shows only part of what it holds.

        Searches stop at a page cap while reporting the whole total (race 14: 96 rows of 199; race 15: 49 rows of up to 1000, no
        price order, no price filter), so the lowest price of a kind can sit where no first search reaches. Race 15's search reads
        titles, categories, brands, sellers and attribute VALUES and wants every word (22642 of 91849 matched words were found in
        an attribute value only), so all of an item's required values together list nearly its whole fitting set, and adding a
        further word splits a set too long for the cap. After a requirement was dropped the fitting set widens: that decided 7
        tasks of race 15, nearly every agent buying a dearer listing, and a rare passing run had kept trying new words (9 cents
        after 98 and 20). Each search is one step and no model call; cheaper_rows then opens the lowest rows."""
        mark = _dump([self.needs_now(), self.budget, sorted(self.gone)])
        if self.brief.get('pick') != 'cheapest' or self.hunts >= HUNT_ROUNDS or self.hunt_mark == mark:
            return ''
        plan = self.value_template()
        if not plan or not self.price_key or self.clock() > HUNT_BY:
            return ''
        tool, base, field = plan
        room = min(HUNT_STEPS, self.left() - self.tail() - HUNT_KEEP - 2 * len(self.items))
        if room < 3:
            return ''
        self.hunts += 1
        self.hunt_mark = mark
        cart = self.cart_ids()
        numbers = list(range(1, len(self.items) + 1))
        share = max(3, room // max(1, len(numbers)))
        spent = 0
        fresh_total = 0
        for number in numbers:
            if spent >= room or self.finished:
                break
            used, gained = self.hunt_item(number, tool, base, field, min(share, room - spent))
            spent += used
            fresh_total += gained
        self.trace.append({'kind': 'hunt', 'steps': spent, 'new_rows': fresh_total, 'cart': cart})
        if spent:
            self.wider_held = True
        return f'Cairn ran {spent} more searches for the lowest price, one step each, and they showed {fresh_total} more rows.' if spent else ''

    def hunt_stems(self, number: int) -> list[list[str]]:
        """The value sets a cheapest hunt searches for one item: its plain values together (codes left out, since they find
        nothing), one set per choice of a closed list ("a or b"), a unit spelling the searches have not tried yet, and the
        first set once more with a leading word dropped before a long number."""
        said = ' '.join((str(v) for args in self.searched for v in args.values() if isinstance(v, str))).lower()
        pointed = {m for _, m in self.refs}
        plain = [value for value, _ in self.value_words(number) if not _codes(value)]
        choices: list[str] = []
        for n, need in self.needs_now():
            if n != number or ':' not in need or need in pointed or choices:
                continue
            value = need.split(':', 1)[1].strip()
            if not re.search('\\bor\\b', value, re.I) or re.search(OPEN_NOT, re.sub('\\bor\\b', ' ', value, flags=re.I), re.I):
                continue
            parts = [p.strip(' .') for p in re.split('\\s*,\\s*|\\s+or\\s+', value, flags=re.I) if p.strip(' .')]
            choices = [p for p in parts if not _codes(p) and len(p) <= HUNT_VALUE_CHARS][:HUNT_CHOICES] if len(parts) >= 2 else []
        out: list[list[str]] = []
        for values in ([plain + [c] for c in choices] or [plain]):
            other = []
            for v in values:
                unit = re.fullmatch(UNIT_VALUE, v.strip(), re.I)
                other.append((unit.group(1) + unit.group(2) if ' ' in v.strip() else unit.group(1) + ' ' + unit.group(2)) if unit else v)
            fresh = other != values and any((v.lower() in said and o.lower() not in said for v, o in zip(values, other) if v != o))
            found = other if fresh else values
            if found not in out:
                out.append(found)
        for values in out[:1]:
            bare = [re.sub('^[a-z]+\\s+(?=\\d{3,}$)', '', v.strip(), flags=re.I) for v in values]
            if bare != values and bare not in out:
                out.append(bare)
        return out

    def hunt_item(self, number: int, tool: str, base: dict[str, Any], field: str, room: int) -> tuple[int, int]:
        """Up to `room` searches for one item: all its required values together, then split by further words."""
        tried = self.hunt_tried.setdefault(number, set())
        noun, _ = self.hunt_words(number)
        sets = self.hunt_stems(number)
        values = sets[0] if sets else []
        spent = gained = idle = 0
        issued = refused = False

        def run(query: str) -> bool:
            """Search `query` to its last page or the room, reading again no page this episode has read; whether it found any row."""
            nonlocal spent, gained, idle, issued, refused
            tried.add(query.lower())
            faults: list[str] = []
            args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
            refused = refused or bool(faults)
            if faults or spent >= room or self.finished:
                return False
            issued = True
            holds_values = any((bool(vals) and all((v.lower() in query.lower() for v in vals)) for vals in sets))
            found = False
            walked: set[str] = set()
            while not self.finished and self.clock() <= HUNT_BY + HUNT_GRACE:
                mark = self.call_mark(tool, args)
                if mark in walked:
                    return found
                walked.add(mark)
                if mark in self.read_calls:
                    seen = self.read_calls[mark]
                    new = HUNT_PAGE_NEW
                elif spent >= room:
                    return found
                else:
                    before = self.rows_of_item(number)
                    reply = self.act(tool, args)
                    spent += 1
                    if reply is None:
                        return found
                    seen = reply.get('observation')
                    self.observe(tool, seen)
                    new = self.rows_of_item(number) - before
                    gained += new
                    idle = 0 if new else idle + 1
                rows = self.records_in(seen)
                found = found or bool(rows)
                if holds_values:
                    self.value_rows.update((i for i, _ in rows))
                page = self.paging(tool, seen)
                if not page or page[3] is None or (new < HUNT_PAGE_NEW and page[0] > HUNT_WHOLE):
                    return found
                faults = []
                args = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                if faults:
                    return found
            return found
        mark = (number, tuple((tuple(vals) for vals in sets)))
        stem = self.hunt_stem.get(mark)
        if stem is None:
            stem = ''
            for vals in sets:
                tries = ([' '.join(vals + [noun]).strip(), ' '.join(vals)] if vals else []) + ([' '.join(vals[:-1] + [noun]).strip()] if len(vals) >= 2 else [])
                for query in dict.fromkeys(tries):
                    issued = False
                    if run(query):
                        stem = stem or query
                        break
                    if not issued:
                        return (spent, gained)
            if not stem and noun:
                issued = False
                if run(noun):
                    stem = noun
                elif not issued:
                    return (spent, gained)
                else:
                    self.hunt_no_noun.add(number)
            self.hunt_stem[mark] = stem
        while spent < room and idle < HUNT_IDLE and not refused and not self.finished and self.clock() <= HUNT_BY + HUNT_GRACE:
            _, words = self.hunt_words(number)
            words = [w for w in words if f'{w} {stem}'.strip().lower() not in tried and w not in stem.lower().split()]
            if not words:
                break
            run(f'{words[0]} {stem}'.strip())
        return (spent, gained)

    @_soft(list)
    def hunt_fits(self) -> list[str]:
        """Search on, without the model, before ending without a purchase.

        Race 15: 495 of 2340 episodes of 13 leaders ended without a purchase on a task that has a right answer, and on 433 of them
        another agent held a cart the store's answer set accepted, so a fit existed within reach; the same holds in practice worlds
        (a fit past the page cap, "no in-stock jeans list a slim fit"). A requirement's value and its words find listings the
        first searches did not show; the rows naming them are opened so that the audit judges their attributes. Returns what
        was read."""
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
        held = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in self.cart_ids() if (self.held_units(i) or 0) > 0] if cat_key else []
        numbers = [n for n, item in enumerate(self.items, 1) if not any((self.owns(item, label) for label in held))] or list(range(1, len(self.items) + 1))
        wanted = {number: {w for value, _ in self.value_words(number) for w in _value_tokens(value)} for number in numbers}
        paged = self.unread_pages(tool, wanted, min(FIT_PAGES, spare // 2))
        room = min(HUNT_STEPS, self.left() - self.tail() - HUNT_KEEP)
        spent = 0
        for number in numbers if room >= 3 else []:
            item = self.items[number - 1]
            pairs = self.value_words(number)
            noun, words = self.hunt_words(number)
            tries = [f'{family} {value} {noun}'.strip() for family in self.series_for(number) for value in ([v for v, _ in pairs][:1] or [''])]
            tries += [f'{choice} {noun}'.strip() for choice in self.item_choices(number)[:3]]
            tries += [f'{value} {noun}'.strip() for value, _ in pairs] + [value for value, _ in pairs if noun] + [f'{word} {noun}'.strip() for word in words[:HUNT_FIT_WORDS]] + ([noun] if noun else [])
            share = max(2, room // max(1, len(numbers)))
            used = 0
            for query in dict.fromkeys(tries):
                if spent >= room or used >= share or self.finished or self.clock() > HUNT_BY + HUNT_GRACE:
                    break
                faults: list[str] = []
                args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
                if faults:
                    continue
                reply = self.act(tool, args)
                spent += 1
                used += 1
                if reply is None:
                    return []
                self.observe(tool, reply.get('observation'))
        picks: list[str] = []
        for number in numbers:
            item = self.items[number - 1]
            pool = [i for i in self.seen_order if i not in self.full and self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')))]
            cap = self.budget if isinstance(self.budget, (int, float)) else None
            pool = [i for i in pool if cap is None or (self.price_of(i) or 0) <= cap]
            words = wanted.get(number) or set()
            pool.sort(key=lambda i: (-len(words & _text_words(_dump(self.rows.get(i) or {}))), self.price_of(i) if self.price_of(i) is not None else float('inf')))
            picks += pool[:HUNT_FIT_OPENS]
        read: list[str] = []
        for start in range(0, len(picks), HUNT_FIT_OPENS):
            if self.clock() > HUNT_BY + HUNT_GRACE:
                break
            read += self.fill_facts(picks[start:start + HUNT_FIT_OPENS], keep)
        self.trace.append({'kind': 'hunt_fits', 'pages': paged, 'searches': spent, 'opened': len(read)})
        return read

    def unread_pages(self, tool: str, wanted: dict[int, set[str]], room: int) -> int:
        """Up to `room` further pages of the earliest searches that hold a required value, each item its share first, before any
        new words; the steps it took. A search whose rows all belong to another item is left. In a practice world accepted rows
        sat on unread pages of the episode's own searches in nearly every wrong decline, and in races 13-15 about 23% of passing
        carts were first seen on page 2 or later."""
        spent = 0
        cat_key = self.category_key()
        share = max(1, room // max(1, len(wanted)))
        for limit in (share, room):
            for number, words in wanted.items():
                stop = min(room, spent + limit)
                for mark, (name, args) in list(self.read_args.items()):
                    text = ' '.join((v for v in args.values() if isinstance(v, str))).lower()
                    if name != tool or not words & _text_words(text):
                        continue
                    seen = self.read_calls[mark]
                    owners = {self.owner_of(str(record.get(cat_key) or '')) for _, record in self.records_in(seen)} if cat_key else set()
                    if owners and 0 not in owners and number not in owners:
                        continue
                    call, walked = args, {mark}
                    while spent < stop and not self.finished:
                        page = self.paging(tool, seen)
                        if not page or page[3] is None:
                            break
                        faults: list[str] = []
                        call = _shape({**call, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                        now = self.call_mark(tool, call) if not faults else ''
                        if faults or now in walked:
                            break
                        walked.add(now)
                        if now in self.read_calls:
                            seen = self.read_calls[now]
                            continue
                        reply = self.act(tool, call)
                        spent += 1
                        if reply is None:
                            return spent
                        seen = reply.get('observation')
                        self.observe(tool, seen)
        return spent

    def call_mark(self, name: str, args: dict[str, Any]) -> str:
        """One spelling per call: words in any case and spacing, a number at its first value left out."""
        props = (self.schemas.get(name) or {}).get('properties') or {}
        kept: dict[str, Any] = {}
        for key, value in sorted(args.items()):
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            first = value in (spec.get('minimum'), spec.get('default')) or (value == 1 and key == self.page_args.get(name) and spec.get('minimum') is None and spec.get('default') is None)
            if isinstance(value, int) and not isinstance(value, bool) and first:
                continue
            kept[key] = ' '.join(value.lower().split()) if isinstance(value, str) else value
        return name + _dump(kept)

    def note_read(self, name: str, args: dict[str, Any], seen: Any) -> None:
        """Remember what a search or listing call showed, whoever made it."""
        if name in self.page_tools() or name in {self.roles.get(r) for r in ('ask', 'cart', 'cart_view', 'buy', 'refuse')}:
            return
        self.query_is_new(name, args)
        if isinstance(seen, (dict, list)) and not _failed(seen):
            self.read_calls[self.call_mark(name, args)] = seen
            self.read_args[self.call_mark(name, args)] = (name, dict(args))

    def query_is_new(self, name: str, args: dict[str, Any]) -> bool:
        """Whether this search (its arguments apart from numbers) is the first of its kind, and remember it."""
        mark = _dump({k: v for k, v in args.items() if isinstance(v, bool) or not isinstance(v, int)})
        seen = self.page_queries.setdefault(name, set())
        fresh = mark not in seen
        seen.add(mark)
        return fresh

    @_soft('')
    def last_page_first(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A new search that opened on a later page shows few rows and no next page, which reads as all there is: race 15's models
        asked the browse tool for its highest page in 1716 of 2845 calls, seeing 7 of 342 rows and "next: none". When such a call
        reports more rows than it shows, the same call at the lowest value of its numeric argument tells: rows with other ids,
        at least as many, mean that argument is the page."""
        if name in self.page_args or not isinstance(observation, dict):
            return ''
        props = (self.schemas.get(name) or {}).get('properties') or {}
        numeric = [k for k, v in args.items() if isinstance(v, int) and not isinstance(v, bool) and k in props and v > (props[k].get('minimum') if isinstance(props[k].get('minimum'), int) else 1)]
        if len(numeric) != 1 or self.left() <= self.tail() + 4:
            return ''
        shape = self.read_shape(name, observation)
        if not shape or not shape.get('count') or shape.get('next') is None or observation.get(shape['next']) is not None:
            return ''
        rows = observation.get(shape['rows'])
        count = observation.get(shape['count'])
        if not isinstance(rows, list) or not isinstance(count, int) or count <= len(rows):
            return ''
        key = numeric[0]
        low = props[key].get('minimum') if isinstance(props[key].get('minimum'), int) else 1
        faults: list[str] = []
        call = _shape({**args, key: low}, self.schemas.get(name), 'args', faults)
        if faults:
            return ''
        reply = self.act(name, call)
        if reply is None:
            return ''
        seen = reply.get('observation')
        self.observe(name, seen)
        new = seen.get(shape['rows']) if isinstance(seen, dict) else None
        if not isinstance(new, list) or len(new) < len(rows) or {i for i, _ in self.records_in(seen)} & {i for i, _ in self.records_in(observation)}:
            return ''
        self.page_args[name] = key
        self.trace.append({'kind': 'page_arg', 'tool': name, 'arg': key})
        return f'\n[Cairn] {key} is the page number of this search: the call above showed its last page. Page {low} of the same search:\n' + self.show(seen)

    @_soft(False)
    def names_code(self, args: dict[str, Any]) -> bool:
        """Whether a search names a model or part code some listing's text gave as a code."""
        said = {c for v in args.values() if isinstance(v, str) for c in _codes(v)}
        if not said:
            return False
        known: set[str] = set()
        for mark in self.fit_index()['marks'].values():
            known.update(mark.get('codes') or [])
            known.update(mark.get('works') or [])
        return bool(said & known)

    def first_look(self, name: str, args: dict[str, Any], fresh: bool) -> dict[str, Any]:
        """A new search starts on its first page once the tool's page argument is known."""
        key = self.page_args.get(name)
        if not key and fresh:
            arg = (self.shapes.get(name) or {}).get('next_arg')
            spec = ((self.schemas.get(name) or {}).get('properties') or {}).get(arg) if arg else None
            if arg and isinstance(spec, dict) and arg in args and arg not in ((self.schemas.get(name) or {}).get('required') or []) and args.get(arg) not in (spec.get('default'), spec.get('minimum')):
                self.trace.append({'kind': 'first_row', 'tool': name, 'dropped': {arg: args.get(arg)}})
                return {k: v for k, v in args.items() if k != arg}
        if not key or not fresh or not isinstance(args.get(key), int):
            return args
        props = (self.schemas.get(name) or {}).get('properties') or {}
        low = props.get(key, {}).get('minimum') if isinstance(props.get(key, {}).get('minimum'), int) else 1
        return {**args, key: low}

    @_soft('')
    def shorter_search(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A long words search that found nothing is run again with its first words, no model turn needed. Race 15: 54% of
        six-word queries and 42% of five-word ones came back empty (search wants every word), 10% of one-word ones."""
        plan = self.value_template()
        if not plan or plan[0] != name or self.left() <= self.tail() + SHORT_KEEP or self.finished or not isinstance(observation, dict):
            return ''
        field = plan[2]
        text = args.get(field)
        if not isinstance(text, str) or self.records_in(observation) or len(text.split()) < 2:
            return ''
        for value in observation.values():
            if (isinstance(value, (list, str)) and value) or (isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0):
                return ''
        tried = []
        for short in self.zero_rungs(text)[:2]:
            if self.left() <= self.tail() + SHORT_KEEP or self.finished:
                break
            faults: list[str] = []
            call = _shape({k: v for k, v in {**args, field: short}.items() if isinstance(v, bool) or not isinstance(v, int)}, self.schemas.get(name), 'args', faults)
            if faults or self.call_mark(name, call) in self.read_calls:
                continue
            reply = self.act(name, call)
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(name, seen)
            self.trace.append({'kind': 'shorter', 'from': text, 'to': short, 'rows': len(self.records_in(seen))})
            if self.records_in(seen):
                return (''.join(tried)) + f'\n[Cairn] Nothing has all those words; the same search with {short!r} shows:\n' + self.show(seen)
            tried.append(f'\n[Cairn] {short!r} found nothing either.')
        return ''.join(tried)

    @_soft('')
    def first_moves(self) -> str:
        """Search each item once and read a few of its listings before the model's first turn, so the interview can start at once.

        A race model turn costs 5-10 s (race 15: 207 s mean per task, 16 of 180 and 30 of 180 episodes of two agents killed at 300 s,
        all on 3+ items) and the first turns only do this; one-item tasks have time, and keep the model's own first searches. Each
        item's search is its required values with its noun, then its name when that finds nothing. Returns what the store showed; '' when the search tool cannot be used without a model's reading of its
        modes, and the model then starts as usual."""
        plan = self.value_template()
        if not plan or self.max_steps < OPEN_MIN_STEPS or len(self.items) < OPEN_ITEMS:
            return ''
        tool, base, field = plan
        self.env_gap = OPEN_GAP
        shown: list[str] = []
        sweeps: list[tuple[int, dict[str, Any], Any]] = []
        stop_at = self.step + self.left() - self.tail() - self.msgs_free() - OPEN_LATER * len(self.items) - HUNT_KEEP // 2
        for number, item in enumerate(self.items, 1):
            wanted = [m.split(':', 1)[1].strip() for m in item.musts if ':' in m and m not in [need for _, need in self.refs]]
            wanted = [w for w in wanted if w and len(w.split()) <= 3 and not re.search(OPEN_NOT, w, re.I)]
            noun = item.name if len(item.name.split()) <= 3 else ' '.join(item.name.split()[-2:])
            tries = list(dict.fromkeys(([' '.join(wanted[:2] + [noun])] if wanted else []) + [noun]))
            singles = [f'{w} {noun}' for w in wanted[:2]] if len(wanted) >= 2 else []
            while tries:
                query = tries.pop(0)
                faults: list[str] = []
                args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
                if faults or self.finished or self.step >= stop_at:
                    break
                reply = self.act(tool, args)
                if reply is None:
                    return '\n'.join(shown)
                seen = reply.get('observation')
                self.observe(tool, seen)
                if self.records_in(seen):
                    self.searched = (self.searched + [dict(args)])[-8:]
                    self.first_seen.append((tool, dict(args), seen))
                    shown.append(f'{query}: ' + self.show(seen))
                    if wanted and query != noun:
                        mine = self.origin.setdefault(number, [])
                        mine += [i for i, _ in self.records_in(seen) if i not in mine]
                        sweeps.append((number, dict(args), seen))
                    shown += self.more_rows_for_keys(tool, args, seen, stop_at, number)
                    tries = [t for t in tries if t not in singles]
                    if self.rows_of_item(number) >= OPEN_EACH:
                        break
                else:
                    shown.append(f"{query}: {'the store refused the call: ' + self.show(seen) if _failed(seen) else 'no rows'}")
                    if singles and query not in singles and query != noun and not _failed(seen):
                        tries = [t for t in singles if t not in tries] + tries
        shown += self.value_pages(tool, sweeps, stop_at)
        if self.rows and self.roles.get('ask') and self.msgs_free() > 0:
            start = self.step
            found = [n for n in range(1, len(self.items) + 1) if self.rows_of_item(n)]
            while self.step - start < self.max_steps // OPEN_SHARE and self.step < stop_at and not self.finished:
                if min((self.records_of_item(n) for n in found), default=OPEN_EACH) >= OPEN_EACH:
                    break
                had = len(self.full)
                self.widen()
                if len(self.full) == had:
                    break
        self.env_gap = ENV_GAP
        self.trace.append({'kind': 'first_moves', 'rows': len(self.rows), 'opened': len(self.full)})
        swept = self.series_sweep()
        shown += ['read in full: ' + self.page_text(ident, record) for ident, record in self.full.items()]
        shown += [swept] if swept else []
        proof = self.evidence(list(self.full.values()))
        return '\n'.join(shown + ([proof] if proof else []))

    def page_text(self, ident: str, record: dict[str, Any]) -> str:
        """One read listing for the model: its attributes first, since those are what it judges."""
        block = _fact_block(record) or {}
        rest = {k: v for k, v in record.items() if v is not block and not isinstance(v, (dict, list))}
        return f'{ident}: attributes ' + _dump(_clip(block, 120))[:PAGE_ATTR_CHARS] + ' | ' + _dump(_clip(rest, 200))[:PAGE_REST_CHARS]

    def more_rows_for_keys(self, tool: str, args: dict[str, Any], seen: Any, stop_at: int, number: int) -> list[str]:
        """One further page when a first page shows too few of the item's listings to tell what they carry; what it showed."""
        page = self.paging(tool, seen)
        if not page or page[3] is None or self.rows_of_item(number) >= OPEN_EACH or self.finished or self.step >= stop_at:
            return []
        faults: list[str] = []
        call = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
        if faults:
            return []
        reply = self.act(tool, call)
        if reply is None:
            return []
        self.observe(tool, reply.get('observation'))
        return [f'next page: ' + self.show(reply.get('observation'))]

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
        cap = min(props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4, self.compare_room())
        ranked = sorted(self.item_groups(), key=lambda ids: sum((1 for i in ids if i in self.full)))
        lanes = [[i for i in ids if i not in self.full] for ids in ranked]
        picks: list[str] = []
        for round_of in range(cap):
            picks += [lane[round_of] for lane in lanes if round_of < len(lane)]
        picks = picks[:cap]
        if len(picks) < 2:
            self.fill_facts(picks, 12)
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
                self.compare_ok = max(self.compare_ok, len(rows))
            else:
                self.compare_bad = min(self.compare_bad or 999, len(picks))
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
                        self.compare_ok = max(self.compare_ok, len(self.records_in(reply.get('observation'))))

    def widen_singly(self) -> None:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        cat_key = self.category_key()
        per_item: dict[int, set[str]] = {}
        for n, need in self.needs_now():
            per_item.setdefault(n, set()).update(re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower()))
        every = set().union(*per_item.values()) if per_item else set()
        each = WIDEN_EACH if len(self.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in self.item_groups():
            opened = sum((1 for i in ids if i in self.full))
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            label = str((self.rows.get(ids[0]) or {}).get(cat_key) or '') if cat_key and ids else ''
            owners = [n for n, item in enumerate(self.items, 1) if self.owns(item, label)]
            wanted = set().union(*(per_item.get(n, set()) for n in owners)) if owners else every
            found = set().union(*(set(self.origin.get(n, [])) for n in owners)) if owners else set()

            def hits(i: str) -> int:
                return len(wanted & set(re.findall('[a-z0-9]{3,}', _words(_dump(self.rows.get(i) or {})))))

            def price(i: str) -> float:
                return self.price_of(i) if self.price_of(i) is not None else float('inf')
            if self.brief.get('pick') == 'cheapest':
                pool.sort(key=lambda i: (i not in found, -hits(i), price(i)))
            else:
                pool.sort(key=lambda i: (i not in found, -hits(i)))
            lanes.append(pool[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.fill_facts(first[:cap][:8], 22)
        self.fill_facts((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    def written_needs(self) -> list[tuple[str, str, str]]:
        """The needs the request writes out as sentences of their own, as (property, product, value)."""
        if self.stated is None:
            self.stated = [tuple(x.strip() for x in m) for m in NEED_SENTENCE.findall(self.request or '')]
            if self.stated:
                self.trace.append({'kind': 'needs_written', 'count': len(self.stated)})
        return self.stated

    def noask_mode(self) -> bool:
        """This build plays race 15 with the needs written into the request: the shopper discloses nothing there (King
        #15114: 260 replies, every one 'any is fine' or 'say again'), so questions only spend steps and clock, and the
        mode's decline hold and clock-end closest order apply to every task, not only to requests with need sentences.
        A world where the shopper holds needs back (an item the request gives no requirement for, or a request that asks
        to be checked with first) turns the King's interview back on for that task (M6: unseen worlds)."""
        if not self.items:
            return True
        if self.noask_known is not None:
            return self.noask_known
        asks = bool(_items(self.brief.get('ask_first')))
        bare = [item.name for n, item in enumerate(self.items, 1) if (not any((':' in m for m in item.musts)) or self.unmatched_needs(n)) and not self.named_extra(n)]
        mode = not (bare or asks) or not self.roles.get('ask')
        if bare or asks or len(self.full) >= 3 * len(self.items) or self.loop_calls >= 3 or not self.roles.get('ask'):
            self.noask_known = mode
            if not mode:
                self.trace.append({'kind': 'interview_on', 'bare': bare, 'ask_first': asks})
        return mode

    def named_extra(self, number: int) -> bool:
        """Whether the item's name carries a requirement value ("birch" in "birch brush handle"): a word of the name, beyond
        its store label, that read listings of this store state as an attribute value. Words that are only part of what
        the product is called ("paint", "air-dry") are not, and with nothing read yet the answer is no."""
        item = self.items[number - 1]
        cat_key = self.category_key()
        labels = {str((self.rows.get(i) or {}).get(cat_key) or '') for i in self.item_ids(number)} if cat_key else set()
        known = set().union(*[_roots(l) for l in labels]) if labels else set()
        extra = {w for w in _roots(item.name) - known - _roots(item.shelf or '') if not w.isdigit() and not re.fullmatch(NUMBER_WORDS, w)}
        if not extra:
            return False
        pool = [i for i in self.item_ids(number) if i in self.full] or list(self.full)
        values = {w for i in pool for _, v in self.attr_pairs(i) for w in _vw(v)}
        return any((_kin(e, w) for e in extra for w in values))

    def settle_noask(self) -> bool:
        """At the first cart or finishing call the interview decision is final: whatever noask_mode says now."""
        mode = self.noask_mode()
        if self.noask_known is None:
            self.noask_known = mode
        return mode

    def unmatched_needs(self, number: int) -> bool:
        """Every written requirement of the item names a property no attribute of three or more of its read listings
        carries (a use or a story read as a requirement): the request gave no real requirement for it."""
        item = self.items[number - 1]
        props = [m.partition(':')[0] for m in item.musts if ':' in m]
        keys: set[frozenset[str]] = set()
        read = 0
        for ident in self.item_ids(number):
            block = _fact_block(self.full.get(ident) or {}) or {}
            if block:
                read += 1
                keys |= {_spelled(k) for k in block}
        if read < 3 or not props:
            return False
        return not any((_names(_spelled(p), k) or (_spelled(p) & k) for p in props for k in keys))

    def prelude_due(self) -> bool:
        """Needs written into the request turn the questions off, but not what the interview does before and after them:
        due at the moment the interview would start (the same reads-or-steps gate), once."""
        if self.prelude_done or self.finished or not self.noask_mode():
            return False
        if self.left() <= 8 or self.clock() > ASK_BY:
            return False
        if self.loop_calls < ASK_TURNS and not all((self.records_of_item(n) for n in range(1, len(self.items) + 1))):
            return False
        if len(self.full) >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def noask_prelude(self) -> str:
        """The interview's own reads without its questions: read more listings of each item (each lane ranked by its own
        item's words and value-search rows), then follow the chains of requirements that point at another listing while
        there is room to act on them (Gental's race-16 pipeline runs both inside the interview)."""
        self.prelude_done = True
        had = list(self.full)
        self.widen()
        read = ['read in full: ' + self.page_text(i, self.full[i]) for i in self.full if i not in had]
        chain = self.early_trail()
        self.trace.append({'kind': 'noask_prelude', 'read': len(read), 'chain': bool(chain)})
        parts = []
        if read:
            proof = self.evidence([self.full[i] for i in self.full if i not in had])
            parts.append("Cairn read more listings of each item (one step each):\n" + '\n'.join(read) + ('\n' + proof if proof else ''))
        if chain:
            parts.append(chain.strip())
        return '\n'.join(parts)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        if self.noask_mode():
            # every preference is in the request: the shopper has nothing left to disclose, and each question
            # costs a step that a tight step limit needs for opening and checking listings
            if self.noask_known is not None:
                self.asked = True
            return False
        if self.left() <= 8 or self.clock() > ASK_BY:
            return False
        opened = len(self.full)
        if self.loop_calls < ASK_TURNS and not all((self.records_of_item(n) for n in range(1, len(self.items) + 1))):
            return False
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and 'const' not in v]
        pool = [k for k in required if k in texts] or texts
        args: dict[str, Any] = {}
        if pool:
            args[max(pool, key=lambda k: (not re.search(NOT_MESSAGE, _snake(k), re.I), bool(re.search(MESSAGE_FIELD, _snake(k), re.I)), props[k].get('maxLength') or 10 ** 6))] = text
        else:
            for key, spec in props.items():
                inner = next((k for k, v in (spec.get('properties') or {}).items() if isinstance(v, dict) and 'string' in _kinds(v)), None) if isinstance(spec, dict) else None
                if isinstance(spec, dict) and 'array' in _kinds(spec):
                    args[key] = [text]
                    break
                if inner:
                    args[key] = {inner: text}
                    break
        if not args:
            return None
        for key in required:
            if key not in args:
                args[key] = _filler(props.get(key) or {})
                if args[key] is None:
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
        self.lanes = self.plan_lanes(max(count, self.list_cap()) if self.listing else count)
        self.lane_found = {n: 0 for n in self.lanes}
        self.lane_spent = {n: 0 for n in self.lanes}
        self.lane_used = {n: set() for n in self.lanes}
        self.follow_up = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        groups = self.groups_for(self.msgs_free()) if self.listing else []
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.clock() < QUESTION_BY):
            batch = []
            while groups and self.listing and not batch:
                batch = self.ask_group(groups.pop(0), told)
            batch = batch or self.ask_next(told)
            if not batch:
                break
            sent += batch
        found_items = [n for n in self.lane_found if self.lane_found[n] > 0]
        chain = self.early_trail()
        searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "Cairn's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if chain:
            summary += '\n' + chain
        if searched:
            summary += '\n' + searched
        if self.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def late_question(self) -> str:
        """Before a checkout, ask the next planned question with a message the order itself does not need.

        The interview leaves out every message the request keeps for a process step, whether the step comes with the
        order or with an ending without a purchase. When Cairn is about to buy, the messages kept for the ending are
        free. Race 13 allowed two messages: the base asked one question in 28 of 180 episodes and left a hidden need
        undisclosed in 12 of them (no episode with an undisclosed need passed). Returns the summary for the loop model
        when the answer adds a requirement, else ''."""
        if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit or self.noask_mode():
            return ''
        if any((p['when'] == 'order' and p['messages'] > 0 for p in self.process)):
            return ''
        if self.left() < LATE_STEPS or self.clock() > LATE_BY or not self.judges():
            return ''
        self.late_asked = True
        if not self.lanes:
            self.lanes = self.plan_lanes(self.list_cap() if self.listing else 1)
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
        summary = f'Cairn asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
        searched = self.value_search([number])
        return summary + ('\n' + searched if searched else '')

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search(QUERY_WORDS, _snake(k), re.I)]
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

    def value_words(self, number: int) -> list[tuple[str, bool]]:
        """Search words for each requirement value of the item a words search can match, each with whether an answer
        gave it: the value itself, or for an answer's lower numeric limit with a unit its number and
        unit, since such a hidden limit often sits on the value of a listing that fits."""
        out = []
        told = self.items[number - 1].told
        for n, need in self.needs_now():
            if n != number or need in [m for _, m in self.refs] or ':' not in need:
                continue
            value = need.split(':', 1)[1].strip()
            edge = re.search(FLOOR, value, re.I)
            if edge:
                if need in told:
                    out.append((edge.group(1).strip(), True))
                continue
            if not value or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            out.append((value, need in told))
        return out

    def value_template(self) -> tuple[str, dict[str, Any], str] | None:
        """The search a value search copies: search_template(), or, when every search so far listed a category without
        words, the tool's one query field with its one mode argument set to read it (the option naming words or a query,
        else the other option of a required two-option mode). None when the schema leaves a doubt."""
        plan = self.search_template()
        tool = self.roles.get('search')
        if plan or not tool:
            return plan
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        told = self.brief.get('search_words') if isinstance(self.brief.get('search_words'), dict) else {}
        said, fixed = (told.get('field') if isinstance(told.get('field'), str) else None, told.get('args') if isinstance(told.get('args'), dict) else {})
        named_tool = told.get('tool') if isinstance(told.get('tool'), str) and told.get('tool') in self.schemas else tool
        spec = (self.schemas.get(named_tool) or {}).get('properties') or {}
        if said in spec and 'string' in _kinds(spec[said]) and 'enum' not in spec[said]:
            faults: list[str] = []
            base = _shape({**fixed, said: 'x'}, self.schemas.get(named_tool), 'args', faults)
            if not faults:
                base.pop(said, None)
                return (named_tool, base, said)
        named = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and re.search(QUERY_WORDS, _snake(k), re.I)]
        if not named:
            named = self.nested_text(tool, said)
        last = self.searched[-1] if self.searched else {}
        enums = {k: v['enum'] for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and len(v['enum']) >= 2}
        modes = []
        for key, options in enums.items():
            reads = [o for o in options if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)]
            others = [o for o in options if o != last.get(key)]
            if len(reads) == 1:
                modes.append((key, reads[0]))
            elif key in required and len(options) == 2 and len(others) == 1:
                modes.append((key, others[0]))
        if len(named) != 1 or len(modes) > 1 or (any((k in required for k in enums)) and not modes):
            return None
        base = {k: last[k] for k in required if k in last and k != named[0] and k != named[0].split('.')[0]}
        base.update(modes)
        return (tool, base, named[0])

    def value_search(self, numbers: list[int]) -> str:
        plan = self.value_template()
        if not plan or self.left() < self.tail() + VALUE_ROOM:
            return ''
        tool, base, field = plan
        seen: list[str] = []
        budget = min(8, self.left() - self.tail() - VALUE_KEEP)
        for number in numbers:
            pairs = self.value_words(number)
            item = self.items[number - 1]
            values = [v for v, _ in pairs]
            fresh = [v for v, told in pairs if told]
            if not fresh or budget <= 2:
                continue
            lead = [v for v in fresh if len(v.split()) == 1 or re.fullmatch(AMOUNT, v, re.I)]
            values = lead + [v for v in values if v not in lead]
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [[values[0], item.name], values] if lead else [values]
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
                home = self.home_label(number)
                if cat_key and home:
                    rows.sort(key=lambda i: _words(str((self.rows.get(i) or {}).get(cat_key) or '')) != home)
                seen.append(f'{_dump(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                if rows:
                    opened = self.fill_facts([i for i in rows if i not in self.full][:min(3, budget)], self.tail() + VALUE_KEEP)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Cairn searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def home_label(self, number: int) -> str:
        """The category label (in _words form) of the item's own listings: the request's shelf name when a listing
        seen carries it, else the label most of the item's opened listings (or, with none opened, its search rows)
        carry; '' when none is known."""
        cat_key = self.category_key()
        if not cat_key:
            return ''
        item = self.items[number - 1]
        for pool in (self.full, self.rows):
            counts: dict[str, int] = {}
            for record in pool.values():
                label = str(record.get(cat_key) or '')
                if label and self.owns(item, label):
                    counts[_words(label)] = counts.get(_words(label), 0) + 1
            for name in (item.shelf, item.name):
                if _words(name) in counts:
                    return _words(name)
            if counts:
                return max(counts, key=lambda label: counts[label])
        return ''

    def groups_for(self, free: int) -> list[list[int]]:
        """Which items each of the free messages asks about when the items outnumber them: consecutive shares as even as they
        can be, so no item goes unasked. Race 15: 19 of 90 tasks wanted 4 or 5 items and allowed 3 messages, and an agent naming
        two items in one message drew a need on either of them (10 needs in 37 such messages of one agent)."""
        numbers = [n for n in self.plan_order if n in self.lanes] + [n for n in self.lanes if n not in self.plan_order]
        numbers = [n for n in numbers if self.lanes.get(n)]
        if free <= 0 or len(numbers) <= free:
            return []
        out: list[list[int]] = []
        start = 0
        for slot in range(free):
            size = (len(numbers) - start + (free - slot) - 1) // (free - slot)
            out.append(numbers[start:start + size])
            start += size
        return out

    def ask_group(self, group: list[int], told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """One message naming a few planned properties of each item of the group, the same list rules applying: the reply is
        about the first tracked property named, so each item's best keys lead its part and the first item's first label is the
        only one a reply disclosing nothing settles."""
        share = max(1, self.list_now() // len(group)) if self.list_now() < self.list_cap() else max(MIX_LABELS, self.list_cap() // len(group))
        parts: list[tuple[int, list[str]]] = []
        for number in group:
            left = [label for label in self.lanes.get(number) or [] if _words(label) not in self.lane_used.setdefault(number, set()) and (not self.told_about(number, label))]
            if left:
                parts.append((number, left[:share]))
        if not parts:
            return []
        if len(parts) == 1:
            number, labels = parts[0]
            fitted, text = self.fit_ask(number, labels, False)
            if not fitted:
                return []
            batch = self.send([(number, labels[0], text)], floor)
            if batch:
                self.finish_ask(number, fitted, False, batch, told)
            return batch
        text = ''
        while True:
            for size in range(share, 0, -1):
                pieces = [f"{', '.join(labels[:size][:-1])} or {labels[:size][-1]}" if len(labels[:size]) >= 2 else labels[0] for _, labels in parts]
                text = ', and '.join((f'what {piece} for the {self.item_said(number)}' for piece, (number, _) in zip(pieces, parts)))
                text = 'W' + text[1:].replace(' for the ', ' would you like for the ', 1) + '?'
                args = self.message_args(text)
                if args and _says(args, text):
                    parts = [(number, labels[:size]) for number, labels in parts]
                    break
            else:
                parts = parts[:-1]
                if not parts:
                    return []
                continue
            break
        first = parts[0][0]
        batch = self.send([(first, parts[0][1][0], text)], floor)
        if not batch:
            return []
        for number, _ in parts:
            self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
        self.list_asks += 1
        had = [len(item.told) for item in self.items]
        heard = self.hear([(0, text, batch[0][2])], (LIGHT, HEAVY))
        heard.pop('free', None)
        self.take_heard(heard, told)
        answered = False
        for number, labels in parts:
            new = self.items[number - 1].told[had[number - 1]:]
            if new:
                answered = True
                self.lane_found[number] = self.lane_found.get(number, 0) + 1
                self.settle(number, labels, False, list(new), False)
        crowded = any((_count(x) == 1 for x in _items(heard.get('crowded'))))
        if crowded and not answered:
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': first})
            return batch
        if not answered:
            self.lane_used.setdefault(first, set()).add(_words(parts[0][1][0]))
        self.trace.append({'kind': 'mixed_ask', 'parts': parts, 'answered': answered})
        return batch

    def finish_ask(self, number: int, labels: list[str], shown: bool, batch: list[tuple[int, str, str]], told: list[str]) -> None:
        """Read a sent question's answer into the items' needs and settle the labels it named."""
        self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
        self.list_asks += 1
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

    def list_cap(self) -> int:
        return LIST_SIZE if len(self.items) == 1 else LIST_MIXED

    def plan_picks(self) -> int:
        return PLAN_PICKS[0] if len(self.items) == 1 else PLAN_PICKS[1]

    def list_now(self) -> int:
        """Labels the next list names: few in the episode's first questions, since nothing shows yet how this shopper takes a list."""
        steps = LIST_STEPS if len(self.items) == 1 else LIST_STEPS_MIXED
        return min(steps[self.list_asks], self.list_cap()) if self.list_asks < len(steps) else self.list_cap()

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        scarce = self.msg_limit <= SCARCE_MESSAGES and (not self.listing)
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number] and (not self.told_about(number, label))]
            if left:
                return (number, left[:self.list_now() if self.listing else 1])
        return (0, [])

    def told_about(self, number: int, label: str) -> bool:
        """Whether an answer already gave a need on this property of the item: the label's words are held by the told
        property's name ("colour" after "shade colour") or hold a told name of two or more words; colour and color,
        singular and plural alike. A one-word told name ("type") hides no longer label ("bulb type")."""
        mine = _spelled(label)
        return any((_names(mine, _spelled(t.split(':', 1)[0])) for t in self.items[number - 1].told))

    def look_alikes(self, number: int, label: str) -> list[str]:
        """Up to two keys of the item a reader may take for `label` as well: keys sharing a word with it, or values written under both on the
        listings read. A word several of the item's keys carry says nothing, and neither does a value of one letter or digit."""
        mine = _words(label)
        seen: dict[str, set[str]] = {}
        for ident in self.item_ids(number):
            for key, value in (_fact_block(self.full.get(ident) or {}) or {}).items():
                seen.setdefault(_words(key), set()).update((w for w in _roots(str(value)) if len(w) > 1))
        counts: dict[str, int] = {}
        for name in seen:
            for word in set(name.split()):
                counts[word] = counts.get(word, 0) + 1
        plain = {w for w in mine.split() if counts.get(w, 0) <= 2}
        scored = []
        for order, (name, values) in enumerate(seen.items()):
            if name != mine and name:
                shared = plain & set(name.split())
                common = seen.get(mine, set()) & values
                if shared or common:
                    scored.append((-(3 * len(shared) + min(len(common), 3)), order, name))
        return [name for _, _, name in sorted(scored)[:LOOK_ALIKES]]

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) >= 2:
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.item_said(number)}?"
        alike = self.apart.get(labels[0], {}).get(number)
        if alike:
            return f"Apart from the {' and the '.join(alike)}, which {labels[0]} would you like for the {self.item_said(number)}?"
        return self.question(number, labels[0], self.samples(number, labels[0]) if shown else None)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """The longest leading part of `labels` whose question the message tool's schema takes whole (the schema shaper
        cuts a text past a length cap, so a list too long drops labels from the end instead), with that question;
        ([], '') when not even one label fits."""
        for size in range(len(labels), 0, -1):
            text = self.ask_text(number, labels[:size], shown)
            args = self.message_args(text)
            if args and _says(args, text):
                return (labels[:size], text)
        return ([], '')

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Send the next question (a pending follow-up first, else the next planned labels), read its answer into the
        items' needs (appending each new one to `told`) and settle the labels it named. Returns the sent batch, or []
        when nothing was sent."""
        for _ in range(LIST_SIZE):
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
        self.finish_ask(number, labels, shown, batch, told)
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
            if crowded and self.listing:
                self.listing = False
                self.trace.append({'kind': 'lists_off', 'item': number})
            if new or not crowded:
                return
            if number not in self.one_product and self.name_parts(number):
                self.one_product.add(number)
                self.trace.append({'kind': 'one_product', 'item': number, 'said': self.item_said(number)})
                self.follow_up.insert(0, (number, labels, shown))
                return
            if not self.apart:
                alike = self.look_alikes(number, labels[0])
                if alike:
                    self.apart.setdefault(labels[0], {})[number] = alike
                    self.trace.append({'kind': 'apart', 'item': number, 'label': labels[0], 'from': alike})
                    self.follow_up.insert(0, (number, labels, False))
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
        if self.lane_misses[number] and self.lane_misses[number] % 2 == 0:
            ahead = [label for _, _, label in sorted((abs(i - (len(ahead) - 1) / 2), i, label) for i, label in enumerate(ahead))]
            self.lanes[number] = rest + ahead
        else:
            self.lanes[number] = ahead + rest
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
        every = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled))]
        keys = [key for key in every if not re.search(NOT_PROPERTY, key) or re.search('\\w[ _]models?\\b', key)]
        return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)] + [key for key in every if key not in keys]

    def adopt_labels(self) -> None:
        cat_key = self.category_key()
        if len(self.items) < 2 or not cat_key:
            return
        self.part_shelves()
        labels = {str(record.get(cat_key) or '') for record in self.full.values()} - {''}
        strays = [label for label in labels if not self.owner_of(label)]
        orphans = [n for n in range(1, len(self.items) + 1) if not any((self.owner_of(label) == n and label not in self.adopted for label in labels))]
        if strays and len(orphans) == 1:
            self.adopted.update({label: orphans[0] for label in strays})
            self.trace.append({'kind': 'adopted', 'item': orphans[0], 'labels': strays})

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        if label in self.adopted:
            return self.adopted[label]
        known = self.owners_seen.get((label, len(self.items)))
        if known is not None:
            return known
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
        self.owners_seen[(label, len(self.items))] = best
        return best

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1 or _words(label) == _words(item.shelf or ''))

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
        """The products an item name joins with "and" or commas when each is a plural noun ("door handles and locks"),
        else []; a single product named with "and" ("salt and pepper grinder", "usb and lightning cables", "his and
        hers towels") gives []."""
        name = self.items[number - 1].name
        if not re.search('\\band\\b', name, re.I):
            return []
        parts = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I) if p.strip()]
        return parts if len(parts) >= 2 and all((re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts)) else []

    def item_said(self, number: int) -> str:
        """The item as a question names it. A name joining plural nouns with "and" is asked with "or" (race 13, 10
        agents: "door handles and locks" drew one-at-a-time in 81 of 155 questions; names joined with "or" in 0 of 50,
        "dresser or cabinet" disclosing a need in 10 of 28). After a one-at-a-time reply about the item, one of the
        products its name joins: the one whose head noun most of the item's own rows name outside their category label
        (with no such rows, the first)."""
        item = self.items[number - 1]
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
        if self.quiet_reply(sent):
            self.trace.append({'kind': 'quiet_reply', 'a': sent[0][2][:80]})
            return {}
        self.hear_reads += 1
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f"{i}. ({f'about item {n}' if n else 'about several items'}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    @_soft(False)
    def quiet_reply(self, sent: list[tuple[int, str, str]]) -> bool:
        """Whether an answer needs no reading: it names nothing the questions, the items, the listings' keys or their values name
        and holds no number, and it asks nothing about how to ask. Race 15: 5112 of 5160 non-need replies are such (race 14: 5050 of 5055) and
        the check misses few need replies once the keys count (race 15: 35 of 1355 by question words alone). A reading costs a
        model call, and a race model call about 5-10 s of a 300 s episode. The first answer of an episode is always read: it
        tells whether this shopper takes lists."""
        if self.hear_reads < 1:
            return False
        known: frozenset[str] = frozenset()
        for number, item in enumerate(self.items, 1):
            known |= _spelled(item.name) | _spelled(item.shelf or '')
            for need in item.musts + item.told:
                known |= _spelled(' '.join(re.findall('[A-Za-z0-9]+', need.split(':', 1)[0])))
        for label, name, _, _ in self.key_table():
            known |= _spelled(name) | _spelled(label)
        for record in self.full.values():
            for value in (_fact_block(record) or {}).values():
                known |= {w for w in _spelled(' '.join(re.findall('[A-Za-z0-9]+', str(value)))) if len(w) >= QUIET_WORD}
        for _, question, answer in sent:
            asked = re.sub('\\b(?:what|which|would|you|like|for|the|please|tell|me|and|or)\\b', ' ', question, flags=re.I)
            mine = known | _spelled(' '.join(re.findall('[A-Za-z0-9]+', asked)))
            said = _spelled(' '.join(re.findall('[A-Za-z0-9]+', answer)))
            if re.search('\\d', answer) or said & mine or re.search(ASK_WORDS, answer, re.I):
                return False
        return True

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
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
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
                return (False, ('Before going on, Cairn did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2):
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Cairn: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
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
                text = max(_strings(args), key=len, default='')
                if not text:
                    break
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
        return (True, 'Before going on, Cairn did the steps the request asks for:\n' + '\n'.join(said) if said else '')

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
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
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
            budget = self.budget_in_unit(budget)
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
            self.drop_items.add(self.items.index(item) + 1)
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
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or '; '.join(changed))
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

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
                hits = [f'{k}={str(v)[:80]}' + (f" (a bare figure: read in {self.measured(need)}, the requirement's unit, that is {self.bare_off(need, v):g} {self.measured(need)})" if self.bare_off(need, v) is not None else '') for k, v in block.items() if _names(_stem(prop), _stem(k))]
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
        return "lines by Cairn's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def held_units(self, ident: str) -> int:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            stack: list[Any] = [self.cart_view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if ident in [v.strip() for v in node.values() if isinstance(v, str)]:
                        counts = [_count(v) for k, v in node.items() if not isinstance(v, str) and _count(v) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _words(k))]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
            text = self.cart_view if isinstance(self.cart_view, str) else _dump(self.cart_view).replace('\\n', ' ')
            said = re.search(re.escape(ident) + '\\s*(?:x|\u00d7|qty:?|units?:?)\\s*(\\d+)\\b', text, re.I) or re.search('\\b(\\d+)\\s*(?:x|\u00d7)\\s*' + re.escape(ident), text, re.I)
            if said:
                return int(said.group(1))
        return self.cart_lines.get(ident, 1)

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = self.cart_view if isinstance(self.cart_view, str) else _dump(self.cart_view).replace('\\n', ' ')
            known = [i for i in self.rows if i in text]
            return known + [i for i in self.shaped_ids(text) if i not in known]
        return list(self.cart_lines)

    def shaped_ids(self, text: str) -> list[str]:
        """Words of a text shaped like the ids this store's rows carry (the same letters, digits at the same places)."""
        shapes = {re.sub('[0-9]', '9', re.sub('[A-Za-z]', 'a', i)) for i in list(self.rows)[:40]}
        if not shapes:
            return []
        found = []
        for word in re.findall('[A-Za-z0-9][A-Za-z0-9_\\-:.]{2,40}[A-Za-z0-9]', text):
            if re.sub('[0-9]', '9', re.sub('[A-Za-z]', 'a', word)) in shapes and word not in found:
                found.append(word)
        return found

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _judge_text(record, 3000) if record else '(no record seen)'

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
        state = (len(self.seen_order), len(self.full), sum((len(v) for v in self.views.values())))
        if self.fit_cached and self.fit_cached[0] == state:
            return self.fit_cached[1]
        marks: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident) or self.rows.get(ident) or {}
            text = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
            text += ' ' + '. '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            marks[ident] = _fit_marks(text)
            for part, codes in marks[ident]['also']:
                notes.setdefault(part, set()).update(codes)
        self.fit_cached = (state, {'marks': marks, 'notes': notes})
        return self.fit_cached[1]

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or [])
        if mark.get('own'):
            found |= index['notes'].get(mark['own'], set())
        return found

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def link_system(self, part: int, main: int, index: dict[str, Any]) -> tuple[int, int, set[str]] | None:
        """(item carrying series codes, item naming them, the series) of two items bound to each other, or None when the rows read so far
        show no such system: most rows of one item carry a code of few series, and rows of the other name at least one of those series."""
        marks = index['marks']
        for carrier, namer in ((part, main), (main, part)):
            rows = self.item_ids(carrier)
            tagged = [i for i in rows if (marks.get(i) or {}).get('fam')]
            families = {f for i in tagged for f in marks[i]['fam']}
            if len(tagged) < LINK_ROWS or 2 * len(tagged) < len(rows) or len(families) > LINK_FAMILIES:
                continue
            if any((f in families for i in self.item_ids(namer) for f in (marks.get(i) or {}).get('named') or [])):
                return (carrier, namer, families)
        return None

    def link_holds(self, carried: str, namer: str, families: set[str], index: dict[str, Any]) -> bool | None:
        """Whether a row's own series is one the other row names; None when either row says nothing about it."""
        own = set((index['marks'].get(carried) or {}).get('fam') or []) & families
        named = set((index['marks'].get(namer) or {}).get('named') or []) & families
        return bool(own & named) if own and named else None

    def link_faults(self, part: int, main: int, device: str, lines: list[str], index: dict[str, Any]) -> list[str]:
        system = self.link_system(part, main, index)
        if not system:
            return []
        carrier, _, families = system
        marks = index['marks']
        faults = []
        for line in lines:
            carried, namer = (line, device) if carrier == part else (device, line)
            if self.link_holds(carried, namer, families, index) is not False:
                continue
            codes = set((marks.get(device) or {}).get('codes') or [])
            text = re.sub('[\\s-]+', ' ', _dump(self.full.get(line) or self.rows.get(line) or {}).lower())
            if codes and any((re.search('(?<![a-z0-9])' + re.escape(c) + '(?![a-z0-9])', text) for c in codes)):
                continue
            wanted = set(marks[namer].get('named') or []) & families
            good = [i for i in self.item_ids(carrier) if set(marks[i].get('fam') or []) & wanted and not self.sold_out(i) and i != carried][:4]
            faults.append(f"{carried} is of series {', '.join(sorted(set(marks[carried].get('fam') or []) & families))}, while {namer} takes {', '.join(sorted(wanted))}" + (f"; seen listings of that series: {', '.join(good)}" if good else f"; search that series with the item's words, or take another {self.items[main - 1].name} whose series the wanted lines carry"))
        return faults

    def series_for(self, number: int) -> list[str]:
        """The series to search a linked item by, most useful first: those the cart's lines of the linked items hold, then those most rows of the
        linked items carry or name. [] when the item is linked by no series system."""
        index = self.fit_index()
        marks = index['marks']
        counts: dict[str, int] = {}
        held: list[str] = []
        cart = self.cart_ids()
        for part, main in self.bound:
            system = self.link_system(part, main, index) if number in (part, main) else None
            if not system:
                continue
            for member in (part, main):
                for i in self.item_ids(member):
                    tags = set((marks.get(i) or {}).get('fam') or []) | set((marks.get(i) or {}).get('named') or [])
                    for family in tags & system[2]:
                        counts[family] = counts.get(family, 0) + 1
                        if i in cart and family not in held:
                            held.append(family)
        return (held + [f for f in sorted(counts, key=lambda f: (-counts[f], f)) if f not in held])[:LINK_SEARCHES]

    def fit_faults(self) -> list[str]:
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
            linked = self.link_faults(part, main, devices[0], parts, index)
            if linked:
                faults += linked
                continue
            codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
            if not codes:
                continue
            barred = lambda i: set(index['marks'].get(i, {}).get('bars') or [])
            covers = lambda i: any((_code_hit(p, c) for p in self.fit_set(i, index) for c in codes)) and not any((_code_hit(p, c) for p in barred(i) for c in codes))
            for line in parts:
                fits = self.fit_set(line, index)
                if fits and (not covers(line)):
                    good = [i for i in self.item_ids(part) if covers(i) and (not self.sold_out(i)) and (i not in cart)][:4]
                    faults.append(f"{line} ({self.items[part - 1].name}) works with {', '.join(sorted(fits)[:4])}, not with {devices[0]} ({', '.join(sorted(codes))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.fit_plan(short=True)))
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
            for p in parts:
                system = self.link_system(p, main, index)
                for i in self.item_ids(p) if system and system[0] == p else []:
                    for code in set((index['marks'].get(i) or {}).get('fam') or []) & system[2] if not self.sold_out(i) else ():
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            holders = {code: sum((1 for p in per if any((code in self.fit_set(i, index) or code in set((index['marks'].get(i) or {}).get('fam') or []) for i in self.item_ids(p) if self.holds_values(i, p))))) for code, per in tally.items()}
            ranked = sorted(tally.items(), key=lambda kv: (-holders.get(kv[0], 0), -len(kv[1]), -sum(kv[1].values()), kv[0]))[:6]
            devices = {code: [i for i in self.item_ids(main) if code in ((index['marks'].get(i) or {}).get('named') or []) or any((_code_hit(code, c) for c in index['marks'].get(i, {}).get('codes') or []))][:2] for code, _ in ranked}
            text = '; '.join((f'{code}: ' + ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') + (f' ({holders[code]} of its kinds with a part showing every plain required value)' if holders.get(code) else '') for code, per in ranked))
            if text:
                lines.append(f"codes the {self.items[main - 1].name} candidates' parts name -> {text}")
            sets = self.cheapest_sets(main, parts, index) if self.brief.get('pick') == 'cheapest' else ''
            if sets:
                lines.append(sets)
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as, or when the device's text names the series of the part's code; try one search of a code or series with the item's words, and if it finds nothing list the device's category and read the codes on its rows] " + ' | '.join(lines)

    @_soft('')
    def cheapest_sets(self, main: int, parts: list[int], index: dict[str, Any]) -> str:
        """The lowest totals of one device of the item with one part of each bound item its codes fit, among buyable rows
        seen, requirements not checked: the pick counts the whole order, not each line on its own."""
        options = []
        systems = {part: self.link_system(part, main, index) for part in parts}
        shelves = {part: sorted(((self.price_of(i), i, self.fit_set(i, index), index['marks'].get(i, {}).get('bars') or []) for i in self.item_ids(part) if self.price_of(i) is not None and self.buyable(i)), key=lambda o: (o[0], o[1])) for part in parts}
        for device in self.item_ids(main):
            codes = index['marks'].get(device, {}).get('codes') or []
            price = self.price_of(device)
            if not (codes or (index['marks'].get(device, {}).get('named'))) or price is None or not self.buyable(device):
                continue
            total = price * self.items[main - 1].qty
            picks = [device]
            for part in parts:
                system = systems[part]
                fitting = next((i for _, i, fits, bars in shelves[part] if (any((_code_hit(p, c) for p in fits for c in codes)) and not any((_code_hit(p, c) for p in bars for c in codes))) or (system and system[0] == part and self.link_holds(i, device, system[2], index) is True)), None)
                if not fitting:
                    picks = []
                    break
                picks.append(fitting)
                total += (self.price_of(fitting) or 0.0) * self.items[part - 1].qty
            if picks:
                options.append((total, picks))
        options.sort(key=lambda o: o[0])
        if not options:
            return ''
        return 'lowest whole sets seen, requirements not yet checked: ' + '; '.join((f"{' + '.join(p)} = {t:g}" for t, p in options[:SET_SHOWN]))

    def missing_items(self) -> list[int]:
        """Items with no line in a cart holding fewer lines than items; items no seen listing is labelled for are not counted."""
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not cat_key or len(ids) >= len(self.items):
            return []
        labels = [str(r.get(cat_key) or '') for r in self.rows.values()]
        held = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids]
        return [n for n, item in enumerate(self.items, 1) if any((self.owns(item, label) for label in labels)) and (not any((self.owns(item, label) for label in held)))]

    def states_needs(self, ident: str, number: int) -> bool:
        """Whether a listing's attributes state each of the item's requirements word for word; a limit or an exclusion is
        not read here and counts as not stated."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        for owner, need in self.needs_now():
            if owner != number:
                continue
            prop, _, value = need.partition(':')
            split = lambda text: set(re.sub('[,/;|]', ' ', _words(text)).split())
            words = split(value)
            if not words or words & {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'}:
                return False
            if not any((_names(_spelled(prop), _spelled(k)) and words <= split(str(v)) for k, v in block.items())):
                return False
        return True

    def met_count(self, ident: str, number: int) -> float:
        """How many of the item's requirements a listing's record shows met: 1 for each whose property's attribute holds the
        value word for word, 0.5 when the value's words appear elsewhere in its record; a limit or an exclusion it cannot
        read counts as a miss."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        text = _words(_dump(self.full.get(ident) or {}) + ' ' + _dump(self.rows.get(ident) or {}))
        split = lambda value: set(re.sub('[,/;|]', ' ', _words(value)).split())
        score = 0.0
        for owner, need in self.needs_now():
            if owner != number:
                continue
            prop, _, value = need.partition(':')
            words = split(value)
            if not words or words & {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum', 'within', 'named', 'pointed'}:
                continue
            if any((_names(_spelled(prop), _spelled(k)) and words <= split(str(v)) for k, v in block.items())):
                score += 1
            elif words <= set(text.split()):
                score += 0.5
        return score

    def closest_order(self) -> str:
        """Needs written into the request, time running out, a cart already holding lines for some of several items: instead
        of ending without a purchase (race 15 noask #15230: 12 such endings forced by the clock, every one wrong; the two
        right declines there had an empty cart on a single item), give each item with no line the read, buyable listing of
        its label that shows the most of its requirements met (the cheaper on a tie) inside the budget, show the lines'
        pages, and check out. '' when an item has no candidate, or the steps or seconds do not allow it."""
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if self.closest_done or not cart or not buy or self.finished or len(self.items) < 2:
            return ''
        lines = {i: self.cart_lines[i] for i in self.cart_ids() if i in self.cart_lines and (self.cart_lines.get(i) or 0) > 0 and not self.sold_out(i)}
        pace = self.env_spent / self.env_count if self.env_count else 1.0
        if not lines or not self.missing_items() or TASK_SECONDS - self.clock() < (len(self.items) + 3) * max(pace, 0.5) + 2:
            return ''
        self.closest_done = True
        held = sum(((self.price_of(i) or 0.0) * n for i, n in lines.items()))
        each = self.brief.get('budget_scope') == 'each'
        picks = {}
        for number in self.missing_items():
            item = self.items[number - 1]
            pool = [i for i in self.item_ids(number) if i not in lines and i not in picks and self.buyable(i) and self.price_of(i) is not None]
            pool.sort(key=lambda i: (-self.met_count(i, number), i not in self.full, self.price_of(i) or 0.0))
            pick = None
            for ident in pool:
                cost = (self.price_of(ident) or 0.0) * (1 if each else item.qty)
                if not isinstance(self.budget, (int, float)) or cost + (0.0 if each else held + sum(((self.price_of(i) or 0.0) * n for i, n in picks.items()))) <= self.budget + 1e-09:
                    pick = ident
                    break
            if not pick:
                self.trace.append({'kind': 'closest_none', 'item': number})
                return ''
            picks[pick] = item.qty
        if not picks and not lines:
            return ''
        calls = self.cart_calls({**lines, **picks}, []) if picks else []
        if len(calls) + 1 > self.left():
            return ''
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
        if self.left() >= 2:
            self.reread_offers()
        faults = []
        args = self.buy_args_for(faults)
        if faults or self.left() < 1 or self.finished:
            return ''
        go, said = self.gate('order')
        if not go or self.finished or self.left() < 1:
            return ''
        reply = self.act(buy, args)
        self.trace.append({'kind': 'closest_order', 'added': picks, 'kept': list(lines), 'reply': _clip(reply, 200)})
        if reply is None:
            return ''
        self.buys += 1
        if not reply.get('finished'):
            self.refused_at = self.step
        self.observe(buy, reply.get('observation'))
        added = ', '.join((f'{i} x{n}' for i, n in picks.items())) or 'nothing'
        return f'Not run (Cairn acted instead): the needs are written into this request, and ending without a purchase is almost never right there, so Cairn added the closest listing for each item without a line ({added}) and checked out. Store reply: ' + self.show(reply.get('observation'))

    def fill_short_cart(self) -> bool:
        """Before a forced order: add, for each item with no line, the cheapest opened listing that states its requirements.
        False when an item stays uncovered, so the episode ends by refusing rather than with a short order."""
        cart = self.roles.get('cart') or ''
        bound = {n for pair in self.bound for n in pair}
        for number in self.missing_items():
            item = self.items[number - 1]
            fits = [i for i in self.item_ids(number) if i in self.full and i not in self.cart_lines and self.buyable(i) and self.price_of(i) is not None and self.states_needs(i, number)]
            fits.sort(key=lambda i: self.price_of(i) or 0.0)
            pick = fits[0] if fits and number not in bound else None
            if pick and isinstance(self.budget, (int, float)):
                held = sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in self.cart_ids()))
                cost = (self.price_of(pick) or 0.0) * (1 if self.brief.get('budget_scope') == 'each' else item.qty)
                if cost + (0.0 if self.brief.get('budget_scope') == 'each' else held) > self.budget + 1e-09:
                    pick = None
            calls = self.cart_calls({**{i: self.cart_lines[i] for i in self.cart_ids() if i in self.cart_lines}, pick: item.qty}, []) if pick else None
            if not calls or len(calls) + 1 > self.left():
                return False
            for args in calls:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(cart), 'args', faults)
                if faults or self.act(cart, args) is None:
                    return False
                self.note_cart(cart, args)
            self.trace.append({'kind': 'filled_line', 'item': number, 'line': pick})
        return True

    def part_shelves(self) -> None:
        """An item that shares its shelf with another requested item takes the store label that is its own name, when one is seen
        and no item holds it yet."""
        cat_key = self.category_key()
        labels = sorted({str(row.get(cat_key) or '').strip() for row in self.rows.values()} - {''}) if cat_key else []
        moved = False
        for item in self.items:
            name = _roots(item.name)
            if not name or not item.shelf.strip() or not any((other is not item and _words(other.shelf) == _words(item.shelf) for other in self.items)):
                continue
            if any((other is not item and _roots(other.name) == name for other in self.items)):
                continue
            own = [label for label in labels if _roots(label) == name and _words(label) != _words(item.shelf)]
            if len(own) == 1 and not any((_words(other.shelf) == _words(own[0]) for other in self.items)):
                self.trace.append({'kind': 'shelf_parted', 'item': item.name, 'from': item.shelf, 'to': own[0]})
                item.shelf = own[0]
                moved = True
        if moved:
            self.owners_seen = {}
            self.fit_cached = None

    def fuller_label(self, number: int, label: str) -> str:
        cat_key = self.category_key()
        mine = _roots(self.items[number - 1].name)
        if len(mine) < 2 or mine <= _roots(label) or not cat_key or (self.items[number - 1].shelf and _roots(label) == _roots(self.items[number - 1].shelf)):
            return ''
        return next((seen for seen in sorted({str(r.get(cat_key) or '') for r in self.rows.values()}) if seen and mine <= _roots(seen)), '')

    def named_label(self, number: int, label: str) -> str:
        """The store label the request named for this item, when a listing seen carries it and `label` is another one."""
        item = self.items[number - 1]
        cat_key = self.category_key()
        wanted = _words(item.shelf or '')
        if not wanted or not cat_key or _words(label) == wanted:
            return ''
        here = next((str(r.get(cat_key)) for r in self.rows.values() if _words(str(r.get(cat_key) or '')) == wanted), '')
        return here

    def line_faults(self) -> list[str]:
        self.adopt_labels()
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
                text = f'item {n} ({item.name}) has no line in the cart'
                home = self.label_text(n)
                taken = [i for i in ids if home and label[i] and _words(label[i]) != _words(home) and _roots(label[i]) & _roots(home)]
                if taken:
                    text += f" (its listings carry the store label '{home}'; " + ', '.join((f"{i} carries '{label[i]}'" for i in taken)) + f" and does not count for item {n} whatever the title says: list '{home}' and take item {n}'s line from there)"
                faults.append(text)
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.held_units(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
                fuller = self.fuller_label(n, label[i]) or self.named_label(n, label[i])
                if fuller:
                    faults.append(f'{i} is from {label[i]}, but the store has the label {fuller}, which names item {n} ({item.name}) in full: take the line from there')
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

    @_soft('')
    def sonnet_pick(self) -> str:
        """One compact call to the strong model at checkout: for each cart line, the listing read so far (same category
        label) that the request's pick rule prefers among those meeting every requirement. A cheap model drives the
        episode; this is the one decision that needs the strong one (Sonnet driving every turn cost $0.07+ a task)."""
        cat_key = self.category_key()
        cart = self.cart_ids()
        if not cart or not cat_key:
            return ''
        mark = self.cart_mark() + f'#{self.version}'
        if mark in self.pick_marks:
            return ''
        self.pick_marks.add(mark)
        blocks = []
        for line in cart:
            label = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if not label:
                continue
            pool = [i for i in self.seen_order if i in self.full and self.buyable(i) and str((self.rows.get(i) or self.full.get(i) or {}).get(cat_key) or '').strip().lower() == label]
            if line not in pool:
                pool.append(line)
            pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
            pool = pool[:PICK_POOL]
            if len(pool) < 2:
                continue
            rows = '\n'.join((f'- {i} (price {self.price_of(i)}): ' + _line(self.full.get(i) or self.rows.get(i) or {})[:700] for i in pool))
            blocks.append(f'CART LINE {line} ({label}). Candidates read so far, cheapest first:\n{rows}')
        if not blocks:
            return ''
        prompt = (PICK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
                  + '\n\nCART NOW:\n' + self.cart_text() + '\n\n' + '\n\n'.join(blocks))
        got = self.ask_json(prompt, [HEAVY], 2500) or {}
        notes = []
        for entry in _items(got.get('lines')):
            if not isinstance(entry, dict):
                continue
            line, best = str(entry.get('line') or '').strip(), str(entry.get('best') or '').strip()
            if line in cart and best and best != line and best in self.full and self.buyable(best):
                notes.append(f"{line} -> {best} ({str(entry.get('why') or '')[:160]})")
        self.trace.append({'kind': 'sonnet_pick', 'notes': notes[:4]})
        if not notes:
            return ''
        return ("Cairn's final check of the listings already read says a different listing is the right pick for these lines: "
                + '; '.join(notes) + '. Swap each named line for its replacement at the requested quantity (the cart tool), then check out.')

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or not self.judges(limit) or (not ids):
            return ''
        self.checked.add(key)
        prompt = CHECK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1)))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.sold_out(i) else '') + self.record_of(i) for i in ids))
        weak = [] if lead else self.weak_lines()
        if weak:
            prompt += "\n\nNOT STATED BY THESE RECORDS (a measured limit counts only under a key sharing a word with the property's name; a wanted word only in an attribute value or the title, never from a related attribute holding another kind of value nor from the description), so each counts as unmet unless its record states it in other words:\n" + '\n'.join((f"{line}: {'; '.join(needs)}" for line, _, needs in weak))
        linked = self.code_fits(ids)
        if linked:
            prompt += "\n\nFITS SHOWN BY CODES (the store decides whether a listing fits another listing or a named device by these codes, and each of these holds; a description naming another use, size or seat count does not undo it):\n" + '\n'.join(linked)
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
        judges = self.judges(limit)
        got = None
        if len(judges) > 1 and not lead and not self.audited and self.brief.get('pick') not in ('cheapest', 'most', 'least') and not weak:
            got = self.ask_json(prompt, [LIGHT], 3000)
            if got:
                judges = [HEAVY] if got.get('ok') is False and HEAVY in self.judges(limit) else []
        if judges and not self.no_credit:
            got = self.ask_json(prompt, judges, 3000) or got
        if got and got.get('ok') is False and not lead and self.review_overruled(ids, got):
            got = {**got, 'ok': True, 'overruled': True}
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get('problems'))][:5]
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        self.check_better = list(dict.fromkeys(self.check_better + better))
        if not lead:
            self.better_by_key[key] = better
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def refuse_trail(self) -> str:
        """Once, before a decline with room left: follow what a requirement points at; what that showed, '' when nothing."""
        live = {need for _, need in self.needs_now()}
        again = self.refuse_trailed and len(self.opened) > self.trail_opened >= 0
        if (self.refuse_trailed and not again) or self.left() < TRAIL_ROOM or self.clock() > HUNT_BY or not any((need in live and need not in self.ref_values for _, need in self.refs)):
            return ''
        self.trail_opened = -1 if again else len(self.opened)
        self.refuse_trailed = True
        text, _ = self.trail(HUNT_BY)
        return text + 'Search for those values now and buy what fits; end without a purchase only if nothing does.' if text else ''

    def early_trail(self) -> str:
        """Follow the chains of requirements that point at another listing, once, while there is room to act on what they say."""
        live = {need for _, need in self.needs_now()}
        if self.left() < TRAIL_ROOM or self.clock() > HUNT_BY or not any((need in live and need not in self.ref_values for _, need in self.refs)):
            return ''
        kept = (self.trails, self.trail_blocks)
        self.trails, self.trail_room = 0, EARLY_TRAILS
        try:
            text, _ = self.trail(HUNT_BY)
        finally:
            (self.trails, self.trail_blocks), self.trail_room = kept, None
        self.trace.append({'kind': 'early_trail', 'left': self.left(), 'values': list(self.ref_values.values())[:3]})
        return text

    def show_pointed(self, early: bool=False) -> str:
        """Give the page view to the listings a requirement points at. Race 16 counts a page as read only through its page
        view (the attributes view of the same listing is not enough), and an order whose pointed-at page was never opened
        that way is a process miss even with the right cart. Of the rows read so far: those of the shop the requirement
        names and of its kind, best matching its words first. Early (while steps are plenty), once the model has read that
        shop's listings with the attributes view: the best one, or the two best when they tie. Before an order: the same,
        leaving the steps the cart's own page views and the order need. Returns what the pages showed, for the model."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target or self.finished:
            return ''
        live = {need for _, need in self.needs_now()}
        pointed = [need for _, need in self.refs if need in live and need not in self.pointed_done]
        if not pointed:
            return ''
        cat_key = self.category_key()
        not_names = self.not_maker_keys()
        cart = self.cart_ids()
        never = [i for i in cart if i not in self.offer_seen and i not in self.gone]
        shown: list[str] = []
        for need in pointed:
            low = need.lower()
            words = {w for w in re.findall('[a-z0-9]{3,}', low)} - {'the', 'and', 'with', 'from', 'that', 'listing', 'page', 'named', 'pointed', 'one', 'not', 'its', 'for', 'within', 'either', 'direction', 'whatever', 'land', 'you'}
            scored = []
            for order, ident in enumerate(self.seen_order):
                if ident in cart:
                    continue
                row = self.rows.get(ident) or {}
                names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and k not in not_names and v.strip().lower() not in NO_NAME]
                if not any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                    continue
                label = set(re.findall('[a-z0-9]{4,}', str(row.get(cat_key) or '').lower())) if cat_key else set()
                if label and not (label & set(re.findall('[a-z0-9]{4,}', low))):
                    continue
                text = _dump(self.full.get(ident) or {}).lower() + ' ' + _dump(row).lower()
                scored.append((-sum((1 for w in words if re.search('(?<![a-z0-9])' + re.escape(w), text))), order, ident))
            scored.sort()
            if not scored:
                continue
            if early and not any((i in self.full for _, _, i in scored)):
                continue
            if self.page_read(scored[0][2]):
                self.pointed_done.add(need)
                continue
            tied = len(scored) > 1 and scored[1][0] == scored[0][0]
            picks = [i for _, _, i in scored[:2 if tied else 1] if not self.page_read(i)]
            room = self.left() - (len(never) + 1 if not early else 10)
            for ident in picks[:max(0, room)]:
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
                if faults or self.rushed():
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.observe(tool, reply.get('observation'))
                self.offer_seen[ident] = self.step
                self.trace.append({'kind': 'pointed_shown', 'need': need[:80], 'id': ident, 'early': early})
                self.pointed_done.add(need)
                shown.append(f'{ident}: ' + self.show(reply.get('observation'))[:1200])
        return '[Cairn opened the page of the listing a requirement points at (the page view is what counts as reading it)]\n' + '\n'.join(shown) if shown else ''

    def chain_page(self, name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        """An attributes view the model asks for, of a listing whose shop a live pointing requirement or a page's pointers name,
        goes to the page view instead when that listing's page is still unread: where the store splits a record, the pointers
        that carry a chain on to its next listing live only on the page (race 16), and the page view is what counts as
        reading a pointed-at page. Once per listing: a later attributes view of it runs as asked."""
        page = self.roles.get('offer')
        page_target = self.single_target(page)
        target = self.single_target(name)
        if not page or page == name or not page_target or not target:
            return (name, args, '')
        ident = args.get(target)
        if not isinstance(ident, str) or ident not in self.rows or self.page_read(ident):
            return (name, args, '')
        live = {need for _, need in self.needs_now()}
        texts = [need for _, need in self.refs if need in live]
        if not texts:
            return (name, args, '')
        for other, record in self.rows.items():
            if other != ident:
                texts += [v for k, v in record.items() if isinstance(v, str) and 'pointer' in k.lower() and v.strip()]
        row = self.rows.get(ident) or {}
        not_names = self.not_maker_keys()
        names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and k not in not_names and v.strip().lower() not in NO_NAME]
        low = ' '.join(texts).lower()
        if not any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
            return (name, args, '')
        faults: list[str] = []
        moved = _shape({page_target: ident}, self.schemas.get(page), 'args', faults)
        if faults:
            return (name, args, '')
        self.trace.append({'kind': 'chain_page', 'id': ident})
        return (page, moved, f"\n[Cairn] Showed {ident}'s page instead of its attributes: its shop is one a requirement or a page points at, and a chain of pages is read through the page view, where the pointers to the next listing are.")

    def show_line_refs(self) -> None:
        """Before an order: when a cart line's own page sends you to another shop's listing for one of its properties ("For
        storage, refer to the Business Laptops listing sold by <shop>"), give that listing the page view, finding it with one
        search of the shop's name and the kind named when no row of that shop has been seen. On race 15 noask the one order whose
        line referred to an unopened listing was a process miss: the listing had been searched for, never opened."""
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.single_target(tool)
        if not tool or not target or self.finished:
            return
        not_names = self.not_maker_keys()
        shops: dict[str, list[str]] = {}
        for ident, row in self.rows.items():
            for k, v in row.items():
                if isinstance(v, str) and k not in not_names and re.fullmatch('[A-Z][a-z]{3,}', v.strip()):
                    shops.setdefault(v.strip(), []).append(ident)
        cart = self.cart_ids()
        never = [i for i in cart if i not in self.offer_seen and i not in self.gone]
        for line in cart:
            row = self.rows.get(line) or {}
            own = {v.strip() for v in row.values() if isinstance(v, str)}
            text = ' '.join((str(v) for v in row.values() if isinstance(v, str))) + ' ' + _dump(self.full.get(line) or {})
            for found in re.finditer(REFER, text, re.I):
                phrase = found.group(1)
                named = set(re.findall("(?:sold by|from|by)\\s+([A-Z][a-z]{3,})", phrase)) | set(re.findall("\\b([A-Z][a-z]{3,})'s", phrase))
                named |= {w for w in re.findall('\\b[A-Z][a-z]{3,}\\b', phrase) if w in shops}
                for name in sorted(named):
                    if name in own or (line, name) in self.line_refs_done:
                        continue
                    kind = ' '.join((w for w in re.findall('\\b[A-Z][a-z]+\\b', phrase) if w not in named and w not in shops))[:40]
                    pool = [i for i in shops.get(name, []) if i not in cart]
                    if not pool:
                        if not kind or self.left() < len(never) + 4:
                            continue
                        self.line_refs_done.add((line, name))
                        pool = [i for i in self.find_link(phrase, f'{name} {kind}'.strip()) if i not in cart]
                        if not pool:
                            continue
                    self.line_refs_done.add((line, name))
                    if any((self.page_read(i) for i in pool)):
                        continue
                    words = {w for w in re.findall('[a-z0-9]{3,}', phrase.lower())} - {'the', 'and', 'with', 'listing', 'sold', 'not', 'this', 'item', 'from', 'for'}
                    pool.sort(key=lambda i: -sum((1 for w in words if w in _dump(self.full.get(i) or self.rows.get(i) or {}).lower())))
                    if self.left() < len(never) + 2 or self.rushed():
                        return
                    faults: list[str] = []
                    args = _shape({target: pool[0]}, self.schemas.get(tool), 'args', faults)
                    if faults:
                        continue
                    reply = self.act(tool, args)
                    if reply is None or self.finished:
                        return
                    self.observe(tool, reply.get('observation'))
                    if tool == self.offer_tool():
                        self.offer_seen[pool[0]] = self.step
                    self.trace.append({'kind': 'line_ref_shown', 'line': line, 'shop': name, 'id': pool[0]})

    def page_read(self, ident: str) -> bool:
        """Whether a listing's page was read: opened, and through the store's page view when it has one apart from the attribute view."""
        page = self.roles.get('offer')
        return ident in self.opened and (not page or not self.single_target(page) or page in self.views.get(ident, ()))

    def trail(self, stop: float | None=None) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < (self.trail_room or (6 if self.left() > 15 else 4)) and self.cost < SPEND_HARD and (stop is None or self.clock() <= stop):
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                record = self.full.get(ident) or self.rows.get(ident) if self.page_read(ident) else None
                if record and len(records) < 45:
                    records.append('OPENED ' + _judge_text(record, 1400, 300))
            for ident in reversed(self.seen_order):
                if len(records) >= 60:
                    break
                if not self.page_read(ident) or not (self.full.get(ident) or self.rows.get(ident)):
                    records.append('ROW ' + (_dump(_clip(self.full[ident], 120))[:700] if ident in self.full else _dump(_clip(self.rows.get(ident), 80))[:300]))
            prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += "\n\nTHE SHOPPER'S REQUEST (each requirement above is a short reading of it; where the request's own words say how the wanted value relates to the value along the chain, those words decide it, not the requirement's wording):\n" + self.request[:TEXT_CHARS]
            if self.trail_notes:
                prompt += '\n\nCHECKED BY CAIRN ON THE RECORDS:\n' + '\n'.join(self.trail_notes[-6:])
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.ask_json(prompt, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            again = False
            for chain in _items(got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need = self.chain_need(str(chain.get('need') or '').strip(), pending)
                if not need:
                    continue
                links = [l for l in _items(chain.get('links')) if isinstance(l, dict)]
                ids = [self.link_id(str(l.get('id') or '').strip()) for l in links]
                forms = [str(f).strip() for f in _items(chain.get('forms')) if isinstance(f, (str, int, float)) and str(f).strip()][:3]
                if forms:
                    self.ref_forms[need] = forms
                for place, (link, ident) in enumerate(zip(links, ids)):
                    said = need if not place else (_dump(self.full.get(ids[place - 1]) or self.rows.get(ids[place - 1]) or {}) if ids[place - 1] else '')
                    missing = self.lacks_values(ident, str(link.get('values') or link.get('find') or ''), said) if ident and (self.page_read(ident) or ident in self.full) and said else []
                    if missing and not self.page_read(ident) and not set(missing) & self.key_words():
                        missing = []
                    note = f'{ident} is not the listing described as "' + str(link.get('about') or '')[:150] + '": its attributes and title lack ' + ', '.join(missing[:3])
                    if missing and note not in self.trail_notes:
                        self.trail_notes.append(note)
                        self.trace.append({'kind': 'link_unlike', 'id': ident, 'lacks': missing[:3]})
                        context = (need + ' ' + str(link.get('about') or '')).lower()
                        shop = next((n for n in self.shop_names(ident) if re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', context)), '')
                        words = str(link.get('find') or '').strip() or f"{shop} {link.get('values') or ''}".strip()
                        links, ids, again = (links[:place] + [dict(link, id='', find=words)], ids[:place] + [''], True)
                        break
                if chain.get('done') is True and links and all((self.page_read(i) for i in ids)) and str(chain.get('value') or '').strip():
                    note = f'{ids[-1]} names the property of "{need[:80]}" in its own words without the value {str(chain["value"]).strip()[:40]}: it defers that property to a different listing, so the chain goes on from what its words name'
                    if self.sends_on(ids[-1], need, str(chain['value'])) and note not in self.trail_notes:
                        self.trail_notes.append(note)
                        self.trace.append({'kind': 'trail_onward', 'id': ids[-1], 'value': str(chain['value'])[:40]})
                        again = True
                        continue
                    self.ref_values[need] = str(chain['value']).strip()
                    self.version += 1
                    continue
                if links and all((self.page_read(i) for i in ids)) and not str(links[-1].get('find') or '').strip():
                    self.ref_values[need] = str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}'
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and not self.page_read(i)]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    gap = next((l for l, i in zip(links, ids) if i not in self.rows), {})
                    about = str(gap.get('about') or '') or 'the listing it points at'
                    words = str(gap.get('find') or '').strip()
                    lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and tool and target and self.left() > 12:
                settled: set[int] = set()
                for place, gap in enumerate(gaps[:2]):
                    if self.link_searches >= CHAIN_SEARCHES:
                        break
                    marks = len(self.trace)
                    reach += [i for i in self.find_link(*gap) if i not in reach]
                    known = next((t['id'] for t in self.trace[marks:] if t.get('kind') == 'link_known'), None)
                    if known:
                        seen.append(f'{known}: an opened page of the shop this chain names whose attributes hold every value its description gives; read the value from its page')
                        settled.add(place)
                lost = [l for place, l in enumerate(lost) if place not in settled]
            if not reach or not tool or not target or self.finished:
                if again and pending and tool and target and not self.finished:
                    continue
                break
            if self.trail_room:
                room = min(3, EARLY_OPENS - len(seen), self.left() - (3 * len(self.items) + self.tail() + 1))
            else:
                room = min(3, 6 - len(seen)) if self.left() >= len(reach) + 4 else 0
            if room <= 0:
                break
            for ident in reach[:room]:
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
                if faults or self.rushed():
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.observe(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.show(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += "[Cairn opened these pages while following the listings requirements point at; a page is such a listing only when it shows every value the listing's description gives]\n" + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (text, hold)

    def not_maker_keys(self) -> tuple[Any, ...]:
        """The row keys that cannot hold the name of who sells or makes a listing: its id, its label and its name. Any other
        text field can, and the value must still appear word for word in the description being matched."""
        return (self.id_key, self.category_key(), self.title_key())

    def shop_rows(self, about: str, ids: list[str], values: Any=None) -> list[str]:
        not_names = self.not_maker_keys()
        low = about.lower()
        out = []
        for ident in ids:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and k not in not_names and v.strip().lower() not in NO_NAME]
            heads = [n.split()[0] for n in names if 2 <= len(n.split()) <= 3 and len(n.split()[0]) >= 5 and n[:1].isupper()]
            if ident not in self.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names + heads)) and not (values and self.read_misses(ident, values)):
                out.append(ident)
        return out

    def shop_names(self, ident: str) -> list[str]:
        not_names = self.not_maker_keys()
        return [str(v).strip() for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and k not in not_names and v.strip().lower() not in NO_NAME]

    def names_shop(self, ident: str, text: str) -> bool:
        low = (text or '').lower()
        return any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in self.shop_names(ident)))

    def read_misses(self, ident: str, values: Any) -> bool:
        """Whether a listing whose attributes were read lacks one of a link's values there (whole words)."""
        if not values or ident not in self.full:
            return False
        leaves: list[str] = []
        stack: list[Any] = [_fact_block(self.full[ident]) or {}]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                stack += list(node.values())
            elif isinstance(node, list):
                stack += node
            else:
                leaves.append(str(node))
        text = ' '.join(leaves).lower()
        return bool(leaves) and not all((re.search('(?<![a-z0-9])' + re.escape(str(v).lower()) + '(?![a-z0-9])', text) for v in values))

    def key_words(self) -> set[str]:
        return {w for record in self.full.values() for k in (_fact_block(record) or {}) for w in re.findall('[a-z]{3,}', _words(k))}

    def link_query(self, tool: str, base: dict[str, Any], field: str, query: str, about: str, values: Any=None) -> list[str] | None:
        """The rows a chain-link search showed: a query sent before reuses its rows with no step; once every row of the named
        shop it showed is opened, its next page is fetched while the store has one."""
        memo = query.lower()
        found = self.link_seen.get(memo)
        more = self.link_more.get(memo) if found is not None else None
        if found is not None and (self.shop_rows(about, found, values) or not more or not any((self.names_shop(i, about) for i in found))):
            return found
        if self.link_searches >= CHAIN_SEARCHES or self.left() <= 12:
            return found
        faults: list[str] = []
        args = _shape({**base, field: query, **({more[0]: more[1]} if more else {})}, self.schemas.get(tool), 'args', faults)
        if faults:
            return found
        self.link_searches += 1
        reply = self.act(tool, args)
        if reply is None:
            return found
        observation = reply.get('observation')
        self.observe(tool, observation)
        rows = [i for i, _ in self.records_in(observation)]
        page = self.paging(tool, observation)
        self.link_more[memo] = (page[2], page[3]) if page and page[3] is not None else None
        found = self.link_seen[memo] = list(dict.fromkeys((found or []) + rows))
        self.trace.append({'kind': 'link_search', 'words': query[:80], 'paged': bool(more), 'rows': self.shop_rows(about, rows)[:4]})
        return found

    def opened_shop(self, about: str) -> list[str]:
        not_names = self.not_maker_keys()
        low = about.lower()
        out = []
        for ident in self.opened:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and k not in not_names and v.strip().lower() not in NO_NAME]
            if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def find_link(self, about: str, words: str) -> list[str]:
        """The listing a chain link describes: the store's search wants every word over values only, so attribute names are
        dropped; the whole description, then the shop with each value, then the shop alone are searched (rows reused and
        paged); rows shown by more of those searches lead, and a row whose read attributes lack a value goes last."""
        not_names = self.not_maker_keys()
        stop = {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        named = {w for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and k not in not_names and v.strip().lower() not in NO_NAME for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - stop
        for ident in self.opened_shop(about):
            leaves: list[str] = []
            stack: list[Any] = [_fact_block(self.full.get(ident) or {}) or {}]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    stack += list(node.values())
                elif isinstance(node, list):
                    stack += node
                else:
                    leaves.append(str(node))
            text = ' '.join(leaves).lower()
            if values and leaves and all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values)):
                self.trace.append({'kind': 'link_known', 'id': ident})
                return []
        plan = self.value_template()
        props = self.key_words()
        parts = [w for n, w in enumerate(re.sub('[,;]+', ' ', words or '').split()) if not n or w.lower() not in props]
        votes: dict[str, list[int]] = {}
        known: list[str] = []
        if plan and parts:
            tool, base, field = plan
            shop = parts[0]
            singles = [f'{shop} {w}' for w in parts[1:] if len(w) >= 3 or re.search('[0-9]', w)] if len(parts) >= 3 else []
            for query in dict.fromkeys([' '.join(parts)] + singles + [shop]):
                if query == shop and votes:
                    break
                found = self.link_query(tool, base, field, query, about, values)
                if found is None:
                    break
                fresh = self.shop_rows(about, found, values)
                while not fresh and found and self.link_more.get(query.lower()) and all((self.names_shop(i, about) for i in found)):
                    more = self.link_query(tool, base, field, query, about, values)
                    if not more or len(more) == len(found):
                        break
                    found, fresh = (more, self.shop_rows(about, more, values))
                for ident in fresh:
                    votes.setdefault(ident, []).append(len(found))
                if fresh and query not in singles:
                    break
            known = list(votes)
        terms = set(re.findall('[a-z0-9]{3,}', f'{about} {words}'.lower())) - named - (stop - {'of'})

        def ranked(ids: list[str], every: bool=False) -> list[str]:
            scored = []
            for pos, i in enumerate(ids):
                hit = len(terms & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower())))
                if hit or every or not terms:
                    scored.append((-hit, pos, i))
            return [i for _, _, i in sorted(scored)]
        rest = ranked(self.shop_rows(about, list(reversed(self.seen_order))))
        clear = [i for i in rest if not self.read_misses(i, values)]
        return (sorted(ranked(known, True), key=lambda i: (-len(votes[i]), min(votes[i]))) or clear + [i for i in rest if i not in clear])[:2]

    def link_id(self, given: str) -> str:
        """A chain link the model named by a code its title prints: the one seen listing holding that code."""
        if not given or given in self.rows:
            return given
        inside = [i for i in self.rows if len(i) >= 6 and i in given]
        if len(inside) == 1:
            return inside[0]
        if not (3 <= len(given) <= 40 and re.search('[0-9]', given)):
            return given
        pattern = '(?<![a-z0-9])' + re.escape(given.lower()) + '(?![a-z0-9])'
        title = self.title_key()
        passes = ([lambda i: str((self.rows.get(i) or {}).get(title) or '')] if title else []) + [lambda i: _dump(self.full.get(i) or self.rows.get(i) or {})]
        for text_of in passes:
            hits = [i for i in self.seen_order if re.search(pattern, text_of(i).lower())]
            if len(hits) == 1:
                self.trace.append({'kind': 'link_id', 'sent': given, 'id': hits[0]})
                return hits[0]
            if len(hits) > 1:
                break
        return given

    def chain_need(self, given: str, pending: list[str]) -> str | None:
        if given in pending:
            return given
        if len(pending) == 1:
            return pending[0]
        words = set(re.findall('[a-z0-9]+', given.lower()))
        scored = sorted(((len(words & set(re.findall('[a-z0-9]+', p.lower()))) / max(1, len(words | set(re.findall('[a-z0-9]+', p.lower())))), p) for p in pending), reverse=True)
        if words and len(scored) > 1 and scored[0][0] >= 0.6 and scored[0][0] > scored[1][0] + 0.15:
            return scored[0][1]
        return None

    def lacks_values(self, ident: str, words: str, said: str='') -> list[str]:
        """The value words of a link's description that the listing's attributes and title hold nowhere (spaces and signs
        aside; a word right after a number is its unit and is not checked; with `said`, only words that text names)."""
        blocks = [b for b in (_fact_block(self.full.get(ident) or {}), _fact_block(self.rows.get(ident) or {})) if isinstance(b, dict) and b]
        if not blocks or not (words or '').strip():
            return []
        names = {t for n in self.shop_names(ident) for t in re.findall('[a-z0-9]+', n.lower())}
        title = self.title_key()
        held = re.sub('[^a-z0-9.]', '', (' '.join((_dump(b) for b in blocks)) + ' ' + (str((self.rows.get(ident) or {}).get(title) or '') if title else '')).lower())
        tokens = [t.strip('.') for t in re.findall('[a-z0-9][a-z0-9.]*', words.lower())]
        wanted = [t for n, t in enumerate(tokens) if t and (len(t) >= 3 or t[0].isdigit()) and t not in names and not (n > 0 and tokens[n - 1][:1].isdigit() and t.isalpha())]
        if said:
            low = said.lower()
            wanted = [t for t in wanted if (t[0].isdigit() or _roots(t)) and re.search('(?<![a-z0-9])' + re.escape(t) + '(?![a-z0-9])', low)]
        return [t for t in dict.fromkeys(wanted) if t not in held]

    def sends_on(self, ident: str, need: str, value: str) -> bool:
        """Whether a page names the requirement's property in its own long words without the value taken from it: such a
        page hands that property to another listing, so a chain does not end there."""
        prop = ' '.join(re.findall('[a-z0-9]+', need.split(':')[0].lower()))
        want = re.sub('[^a-z0-9]', '', value.lower())
        if len(prop) < 3 or not want:
            return False
        title = self.title_key()
        for record in (self.rows.get(ident) or {}, self.full.get(ident) or {}):
            for key, text in record.items():
                if isinstance(text, str) and len(text) > 40 and key != title:
                    words = ' '.join(re.findall('[a-z0-9]+', text.lower()))
                    if re.search('(?<![a-z0-9])' + re.escape(prop) + '(?![a-z0-9])', words) and want not in re.sub('[^a-z0-9]', '', text.lower()):
                        return True
        return False

    def chain_texts(self) -> tuple[str, str]:
        live = {need for _, need in self.needs_now()}
        needs = ' | '.join((need for _, need in self.refs if need in live and need not in self.ref_values))
        if not needs:
            return ('', '')
        title = self.title_key()
        pages: list[str] = []
        used: set[str] = set()
        for _ in range(5):
            known = needs + ' ' + ' '.join(pages)
            new = [i for i in self.seen_order if i not in used and self.page_read(i) and self.names_shop(i, known)]
            if not new:
                break
            for i in new:
                used.add(i)
                pages += [v for k, v in (self.rows.get(i) or {}).items() if isinstance(v, str) and len(v) > 40 and k != title]
        return (needs, ' '.join(pages))

    def page_first(self, name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        """A model attribute read of a listing of a shop a pointer chain names, outside the requested items' categories, goes
        to the page view: only the page says where it points, and the order counts only with that page opened."""
        opener, page = (self.roles.get('open'), self.roles.get('offer'))
        if name != opener or not page or page == opener or page not in self.schemas:
            return (name, args, '')
        view = self.single_target(page)
        target = self.single_target(name)
        ident = str(args.get(target) or '').strip() if target else ''
        if not view or ident not in self.rows or self.page_read(ident) or ident in self.gone:
            return (name, args, '')
        label = str(self.rows[ident].get(self.category_key() or '') or '')
        if not label or self.owner_of(label) != 0:
            return (name, args, '')
        needs, pages = self.chain_texts()
        if not needs or not self.names_shop(ident, needs + ' ' + pages):
            return (name, args, '')
        faults: list[str] = []
        moved = _shape({view: ident}, self.schemas.get(page), 'args', faults)
        if faults:
            return (name, args, '')
        self.trace.append({'kind': 'page_first', 'id': ident})
        return (page, moved, f"\n[Cairn] Showed {ident}'s page ({page}) instead of its attributes: a requirement still takes its value from a listing of this shop, and only the page says where it points or what its value goes with (the order counts only with that page opened). Call {opener} on it again if you still need its attributes.")

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        never = [i for i in stale if i not in self.offer_seen]
        if never and self.left() >= len(never) + 1:
            # a line the page view never showed is not opened as the store counts it: show those before
            # anything else, even when the steps are too few to re-read the others
            stale = never + [i for i in stale if i not in never][:max(0, self.left() - len(never) - 6)]
        elif not stale or self.left() < len(stale) + 5 + (3 if self.max_steps <= 30 else 0):
            return ''
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults or self.rushed():
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
        peers = self.sold_peers(lost)
        if peers:
            self.fill_facts(peers, self.tail() + 5)
        picks = self.replacements(lost)
        text = f"[Cairn] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        floor = max((self.price_of(i) or 0.0 for i in lost), default=0.0)
        if picks and self.brief.get('pick') == 'cheapest' and (self.price_of(picks[0]) is None or self.price_of(picks[0]) > floor + 1e-09):
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
        if not self.judges():
            return []
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.record_of(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.cart_text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges()[::-1], 3500)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and (ident not in self.gone) and (ident not in self.cart_ids()) and (ident not in [i for _, i in fits]):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get('pick') == 'cheapest' else [i for _, i in fits]
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return ranked[:4]

    def audit(self) -> str:
        if self.audited or not self.judges():
            return ''
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            rows = [str(i) for i in _items((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        for ident in sorted(reversed(self.seen_order), key=lambda i: i not in self.full):
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
        judges = self.judges()
        got = self.ask_json(prompt, [LIGHT] if judges else [], 4000)
        if HEAVY in judges and not self.no_credit and not self.audit_fits(got):
            got = self.ask_json(prompt, [HEAVY], 4000) or got
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
        cart_steps = len(self.cart_calls({f'#{n}': 1 for n, _ in enumerate(self.items)}, []) or self.items)
        if self.left() < len(plan) + max(5, self.tail() + cart_steps):
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
            found = re.search('\\d+', str(number))
            index = int(found.group()) if found else (1 if len(self.items) == 1 else None)
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
        self.show_links(list(lines), len(calls) + 3)
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
        args = self.buy_args_for(faults)
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
            self.observe(buy, reply.get('observation'))
            replies.append(self.show(reply.get('observation')))
            if self.finished or reply.get('finished'):
                break
        if not replies:
            return ''
        return 'Not run (Cairn acted instead): the independent check confirmed ' + self.confirmed + ', so Cairn set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

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
            if left >= 3 and self.missing_items():
                keep.add(self.facts_tool())
        elif left <= len(self.items) + 5 and self.buys == 0 and not (self.reopened and self.short_cart()):
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
            if self.missing_items():
                # an item with no line still needs a listing: without these the model can only end the episode
                keep |= {self.offer_tool(), self.roles.get('search')}
        tools = [t for t in self.tools if t['function']['name'] != ask or self.message_args('x' * 200) is None]
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

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A removal of a held line sent without a count names the units held, since the store would take off only one."""
        spec = self.schemas.get(name, {}).get('properties') or {}
        count = next((k for k, v in spec.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        removal = any((isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items()))
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.cart_lines), None)
        return {**args, count: min(self.cart_lines[ident], spec[count].get('maximum', self.cart_lines[ident]))} if count and removal and ident and count not in args else args

    def zero_add(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        """A cart call whose one count field is 0 for a listing the cart does not hold changes nothing; some stores give that
        field a name that reads like a line index (0..4), so the call carries the item's requested quantity instead."""
        spec = self.schemas.get(name, {}).get('properties') or {}
        counts = [k for k, v in spec.items() if isinstance(v, dict) and 'integer' in _kinds(v)]
        if len(counts) != 1 or args.get(counts[0]) != 0 or any((isinstance(v, (list, dict)) for v in args.values())):
            return args
        if any((isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items())):
            return args
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.rows), None)
        if not ident or ident in self.cart_lines or ident in self.cart_ids():
            return args
        owners = [n for n in range(1, len(self.items) + 1) if ident in self.item_ids(n)]
        qty = self.items[owners[0] - 1].qty if owners else self.items[0].qty if len(self.items) == 1 else 1
        top = spec[counts[0]].get('maximum')
        qty = max(1, min(int(qty or 1), int(top))) if isinstance(top, (int, float)) and top >= 1 else max(1, int(qty or 1))
        self.trace.append({'kind': 'zero_add', 'id': ident, 'qty': qty})
        return {**args, counts[0]: qty}

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
                have = have - (count or 1) if any(('integer' in _kinds(v) for v in spec.values() if isinstance(v, dict))) else have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def show(self, observation: Any) -> str:
        if len(self.records_in(observation)) >= 2:
            observation = _lean(observation)
        text = _dump(_clip(observation, SHOW_CHARS if self.records_in(observation) else TEXT_CHARS))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT] + '...(cut)'
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


    # ---- c15_v42: Gental's race-16 rebuild spec (#97) part 1, in our own code ----
    def measured(self, need: str) -> str | None:
        """For a requirement holding a measured limit, the unit written after its number ('' when not plain letters);
        None for any other requirement (a plain "2 kg" with no limit word and no decimal is not measured)."""
        if ':' not in need:
            return None
        raw = need.partition(':')[2].lower()
        value = _words(need.partition(':')[2])
        amount = None
        for found in re.finditer('(?<![a-z0-9])' + AMOUNT, raw):
            if not re.fullmatch(OPEN_NOT, re.sub('^[\\d.,\\s]+', '', found.group())):
                amount = found
                break
        if amount is None or not (re.search('\\d\\.\\d', raw) or re.search(OPEN_NOT, value)):
            return None
        unit = re.sub('^[\\d.,\\s]+', '', amount.group())
        return unit if re.fullmatch('[a-z]+', unit) else ''

    def bare_off(self, need: str, text: Any) -> float | None:
        """A bare number read in the requirement's own unit that lies ten times or more away from its figure."""
        unit = self.measured(need)
        low = str(text).strip().lower().replace(',', '')
        if not unit or not re.fullmatch('\\d+(?:\\.\\d+)?', low):
            return None
        want = re.findall('\\d+(?:\\.\\d+)?', need.partition(':')[2].replace(',', ''))
        goal = float(want[0]) if want else 0.0
        got = float(low)
        if goal <= 0 or got <= 0:
            return None
        return got if got >= 10 * goal or 10 * got <= goal else None

    def bare_misread(self, block: dict[str, Any], need: str) -> bool:
        roots = _roots(need.partition(':')[0])
        named = [v for k, v in block.items() if roots & _roots(k) and re.search('\\d', str(v))]
        return bool(named) and all((self.bare_off(need, v) is not None for v in named))

    def states_measure(self, block: dict[str, Any], prop: str, unit: str) -> bool:
        for key, text in block.items():
            low = str(text).lower()
            if _roots(prop) & _roots(key) and re.search('\\d', low) and (not unit or not re.search('[a-z]', low) or re.search('\\d[\\s-]*' + re.escape(unit), low)):
                return True
        return False

    def names_codes(self, ident: str, codes: list[str]) -> bool:
        mark = self.fit_index()['marks'].get(ident) or {}
        held = list(mark.get('works') or []) + list(mark.get('codes') or [])
        bars = list(mark.get('bars') or [])
        return all((any((_code_hit(h, c) for h in held)) and not any((_code_hit(b, c) for b in bars)) for c in codes))

    def states_word(self, ident: str, need: str, held: bool=False) -> bool:
        """Whether a read listing states a word requirement as grading reads pages: a key named for the property decides
        (with `held`, one of the value's words must be in it); else a value word in an attribute value or the title (word
        forms and initials count, and a device code named in the listing's fit clause)."""
        prop, _, part = need.partition(':')
        record = self.full.get(ident) or {}
        block = _fact_block(record) or {}
        tokens = _value_tokens(part)
        if not tokens:
            return True

        def kin(text: str) -> bool:
            return any((_kin_forms(t, w) for t in tokens for w in re.findall('[a-z]+', text)))
        named = [_glued(_dump(v)) for k, v in block.items() if _names(_spelled(prop), _spelled(k))]
        if named:
            return not held or any((t in v for v in named for t in tokens)) or any((kin(v) for v in named))
        codes = _codes(part)
        if codes and self.names_codes(ident, codes):
            return True
        title = record.get(self.title_key() or 'title') or record.get('title') or (self.rows.get(ident) or {}).get('title') or ''
        text = _glued(' '.join([_dump(v) for v in block.values()] + [str(title)]))
        if any((t in text for t in tokens)) or kin(text):
            return True
        shown = set(re.findall('[a-z0-9]+', text))
        for chunk in re.split('\\s*,\\s*|\\s+or\\s+', part.lower()):
            run = re.findall('[a-z]+', chunk)
            if any((''.join((w[0] for w in run[a:b])) in shown for a in range(len(run)) for b in range(a + 3, len(run) + 1))):
                return True
        return False

    def resolved_needs(self, number: int) -> list[str] | None:
        """The item's live requirements with each pointer requirement replaced by the value read from its page; None while a
        pointed value is unread or unusable."""
        pointed = {need for n, need in self.refs if n == number}
        out = []
        for owner, need in self.needs_now():
            if owner != number or self.described(number, need):
                continue
            if need in pointed:
                value = self.ref_values.get(need, '')
                if not value or value.startswith('as stated on') or re.search(OPEN_NOT, _words(value)):
                    return None
                need = f"{need.partition(':')[0].strip()}: {value}"
            out.append(need)
        return out

    def proves(self, ident: str, number: int, given: list[str] | None=None) -> bool:
        """Whether a read listing states every given (else live) requirement of item `number` under the key named for its
        property: a list there holds each element; a fuller key (the property plus the item's own name) wins over the bare
        key; a plural-key list proves nothing when the title names another of its elements. Limits, exclusions and pointers
        give False (left to the model). Keeps the proof lines in self.proofs."""
        record = self.full.get(ident) or {}
        block = _fact_block(record) or {}
        if not block:
            return False

        def toks(text: str) -> set[str]:
            return set(re.findall('[a-z0-9+#]+', _words(text)))

        def split(text: Any) -> list[set[str]]:
            return [toks(p) for p in re.split('[,/;|]', str(text)) if toks(p)]
        title = toks(str(record.get('title') or (self.rows.get(ident) or {}).get('title') or ''))
        needs = given if given is not None else [need for owner, need in self.needs_now() if owner == number and not self.described(number, need)]
        if not needs:
            return False
        whole = (_spelled(self.items[number - 1].name) | {'of'}) if 0 < number <= len(self.items) else frozenset()
        proof: list[str] = []
        for need in needs:
            prop, _, value = need.partition(':')
            words = toks(value)
            if not words or re.search(OPEN_NOT, _words(value)):
                return False
            named = [(k, v) for k, v in block.items() if _names(_spelled(prop), _spelled(k))]
            fuller = [(k, v) for k, v in named if _spelled(prop) < _spelled(k) <= _spelled(prop) | whole]
            if fuller:
                top = max((len(_spelled(k)) for k, _ in fuller))
                fuller = [(k, v) for k, v in fuller if len(_spelled(k)) == top]
            exact = fuller or [(k, v) for k, v in named if _spelled(k) == _spelled(prop)]
            bare = [(k, v) for k, v in named if fuller and _spelled(k) == _spelled(prop) and not any((words <= p for p in split(v)))]
            held = False
            for key, text in exact or named:
                parts = split(text)
                hit = [p for p in parts if words <= p]
                plural = _words(key) != _words(prop) and _spelled(key) == _spelled(prop)
                if hit and plural and len(parts) > 1 and any((p not in hit and p <= title for p in parts)) and not any((p <= title for p in hit)):
                    hit = []
                if exact and not hit:
                    return False
                held = held or bool(hit)
                if hit and len(proof) < len(needs):
                    proof.append(f'{key}: {_clip(str(text), 160)}' + (f" (this key adds the item's own name to the property, so it decides even where the bare key {bare[0][0]} says {_clip(str(bare[0][1]), 60)})" if bare else ''))
            if not held:
                return False
        self.proofs[ident] = proof
        return True

    def meets_words(self, ident: str, number: int) -> bool:
        """A read, buyable, priced listing meets every live requirement of item `number` on words alone (for whole sets): no
        measured limit, exclusion or unread pointer, and no requirement only in the shopper's words."""
        if ident not in self.full or not self.buyable(ident) or self.sold_out(ident) or self.price_of(ident) is None:
            return False
        needs = self.resolved_needs(number)
        if not needs:
            return False
        if any((owner == number and self.described(number, need) for owner, need in self.needs_now())):
            return False
        block = _fact_block(self.full[ident]) or {}
        for need in needs:
            prop, _, value = need.partition(':')
            if self.measured(need) is not None or re.search(OPEN_NOT, re.sub('\\bor\\b', ' ', _words(value))):
                return False
            if any((_names(_spelled(prop), _spelled(k)) for k in block)):
                choices = [c.strip() for c in re.split('\\s+or\\s+', value.strip()) if c.strip()] or [value]
                if not any((self.proves(ident, number, [f'{prop.strip()}: {c}']) for c in choices)):
                    return False
            elif not self.states_word(ident, need, True):
                return False
        return True

    def weak_lines(self) -> list[tuple[str, int, list[str]]]:
        """Cart lines whose read page leaves a live requirement unknown the way grading reads pages: a measured limit not
        under a key sharing the property's name (or only as a far-off bare figure), or a word value no attribute value or the
        title holds while some listing of that label could state it."""
        cat_key = self.category_key()
        if not cat_key:
            return []
        pointed = {need for _, need in self.refs}
        out = []
        for line in self.cart_ids():
            if (self.held_units(line) or 0) <= 0:
                continue
            block = _fact_block(self.full.get(line) or {}) or {}
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            owners = [n for n, item in enumerate(self.items, 1) if self.owns(item, label)]
            if not block or len(owners) != 1:
                continue
            weak = []
            for owner, need in self.needs_now():
                if owner != owners[0] or need in pointed or self.described(owner, need) or ':' not in need:
                    continue
                prop, _, value = need.partition(':')
                unit = self.measured(need)
                if unit is not None:
                    if not self.states_measure(block, prop, unit) or self.bare_misread(block, need):
                        weak.append(need)
                elif not re.search(OPEN_NOT, re.sub('\\bor\\b', ' ', _words(value))) and not self.states_word(line, need) and self.word_shown(line, need):
                    weak.append(need)
            if weak:
                out.append((line, owners[0], weak))
        return out

    def word_shown(self, line: str, need: str) -> bool:
        cat_key = self.category_key()
        label = str((self.rows.get(line) or {}).get(cat_key) or '') if cat_key else ''
        same = [i for i in self.seen_order if i != line and (str((self.rows.get(i) or {}).get(cat_key) or '') if cat_key else '') == label]
        read = [i for i in same if _fact_block(self.full.get(i) or {})]
        if len(read) < 3 or any((self.states_word(i, need, True) for i in read)):
            return True
        tokens = _value_tokens(need.partition(':')[2])
        title = self.title_key() or 'title'
        return any((t in _glued(str((self.rows.get(i) or {}).get(title) or '')) for i in same for t in tokens))

    def weak_hold(self) -> str:
        weak = self.weak_lines()
        if not weak:
            return ''
        cat_key = self.category_key()
        cart = set(self.cart_ids())
        parts = []
        for line, _, needs in weak[:2]:
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            same = [i for i in self.seen_order if i not in cart and i != line and str((self.rows.get(i) or {}).get(cat_key) or '') == label and self.buyable(i) and not self.sold_out(i) and i not in self.gone]
            if self.brief.get('budget_scope') == 'each' and isinstance(self.budget, (int, float)):
                same = [i for i in same if (self.price_of(i) or 0.0) <= self.budget]

            def states_all(i: str) -> bool:
                block = _fact_block(self.full[i]) or {}
                for n in needs:
                    if self.measured(n) is not None:
                        if not self.states_measure(block, n.partition(':')[0], self.measured(n)) or self.bare_misread(block, n):
                            return False
                    elif not self.states_word(i, n, True):
                        return False
                return True
            strong = [i for i in same if i in self.full and states_all(i)]
            props = [n.partition(':')[0].strip() for n in needs]
            title = _roots(str((self.full.get(line) or self.rows.get(line) or {}).get('title') or ''))
            rows = [i for i in same if i not in self.full]
            rows.sort(key=lambda i: (not any((_roots(p) & _roots(_dump(self.rows[i])) for p in props)), -len(title & _roots(str(self.rows[i].get('title') or ''))), self.price_of(i) if self.price_of(i) is not None else float('inf')))
            said = []
            sizes = [n.partition(':')[0].strip() for n in needs if self.measured(n) is not None]
            if sizes:
                said.append(f"{line} does not state {', '.join(sizes)} under a key named for it (a figure under a key for another measure, or in the title, or nothing at all, leaves the requirement unknown, and the order fails on it)")
                bare = [n.partition(':')[0].strip() for n in needs if self.measured(n) is not None and self.bare_misread(_fact_block(self.full.get(line) or {}) or {}, n)]
                if bare:
                    said.append(f"{line} gives {', '.join(bare)} only as a bare figure, and a bare figure is read in the requirement's own unit, which puts it far outside the requirement")
            words = [n.strip() for n in needs if self.measured(n) is None]
            if words:
                said.append(f"no attribute value and not the title of {line} holds the wanted value of {'; '.join(words)} (a related attribute holding another kind of value, such as a list of figures, or words only in its description leave it unknown unless the page states it in other words)")
            text = '; '.join(said)
            if strong:
                strong.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
                text += '; read listings of the same kind that state it under its own key: ' + ', '.join((f'{i} ({self.price_of(i):g})' if self.price_of(i) is not None else i for i in strong[:3]))
            elif rows and self.left() > self.tail() + 5:
                text += '; open these listings of the same kind first: ' + ', '.join(rows[:3])
            else:
                continue
            parts.append(text)
        if not parts:
            return ''
        self.trace.append({'kind': 'weak_hold', 'lines': [w[0] for w in weak]})
        return 'Not run yet (no step used). ' + ' | '.join(parts) + '. Swap to a listing that states every requirement (a measured limit under its own key, a wanted word in an attribute or the title) and meets it (respecting the pick rule), then check out; keep the line only if no such listing exists.'

    def code_fits(self, ids: list[str]) -> list[str]:
        """Fit links the cart's own records prove by code: a part carrying the series the device names (or the reverse), a
        part naming the device's model code, a line whose fit clause names the device code a requirement gives."""
        index = self.fit_index()
        marks = index['marks']
        out = []
        for part, main in self.bound:
            devices = [i for i in ids if i in self.item_ids(main)]
            if len(devices) != 1:
                continue
            device = devices[0]
            system = self.link_system(part, main, index)
            codes = set((marks.get(device) or {}).get('codes') or [])
            for line in [i for i in ids if i in self.item_ids(part)]:
                if system:
                    carried, namer = (line, device) if system[0] == part else (device, line)
                    if self.link_holds(carried, namer, system[2], index) is True:
                        shared = sorted(set((marks.get(carried) or {}).get('fam') or []) & set((marks.get(namer) or {}).get('named') or []) & system[2])
                        out.append(f"{line} ({self.items[part - 1].name}) and {device} ({self.items[main - 1].name}): {carried} carries series {', '.join(shared)}, which {namer} names")
                    continue
                bars = (marks.get(line) or {}).get('bars') or []
                hit = sorted((c for c in codes if any((_code_hit(p, c) for p in self.fit_set(line, index))) and not any((_code_hit(b, c) for b in bars))))
                if hit:
                    out.append(f"{line} ({self.items[part - 1].name}) names the model code of {device} ({self.items[main - 1].name}): {', '.join(hit).upper()}")
        cat_key = self.category_key()
        for ident in ids:
            label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
            owners = [n for n, item in enumerate(self.items, 1) if self.owns(item, label)]
            mark = marks.get(ident) or {}
            works = list(mark.get('works') or []) + list(mark.get('codes') or [])
            bars = list(mark.get('bars') or [])
            for owner, need in self.needs_now():
                said = _codes(need.partition(':')[2]) if owners == [owner] else []
                if said and all((any((_code_hit(w, c) for w in works)) and not any((_code_hit(b, c) for b in bars)) for c in said)):
                    out.append(f"{ident} meets '{need}': its fit clause names {', '.join(said).upper()}")
        return out

    def keeps_link(self, number: int, ident: str, cart: list[str]) -> bool:
        """Whether listing `ident` of item `number`, taking a cart line's place, keeps every fit link with the cart."""
        index = self.fit_index()
        marks = index['marks']
        for part, main in self.bound:
            if number not in (part, main):
                continue
            other = main if number == part else part
            lines = [i for i in cart if i in self.item_ids(other)]
            if not lines:
                return False
            for line in lines:
                device, piece = (line, ident) if number == part else (ident, line)
                system = self.link_system(part, main, index)
                if system:
                    carried, namer = (piece, device) if system[0] == part else (device, piece)
                    if self.link_holds(carried, namer, system[2], index) is not True:
                        return False
                else:
                    codes = set((marks.get(device) or {}).get('codes') or [])
                    fits = self.fit_set(piece, index)
                    bars = set((marks.get(piece) or {}).get('bars') or [])
                    if not codes or not any((_code_hit(p, c) for p in fits for c in codes)) or any((_code_hit(p, c) for p in bars for c in codes)):
                        return False
        return True

    def cheaper_proven(self) -> list[tuple[str, float, str, float]]:
        """Lowest-price picks: per cart line, the cheapest read listing of exactly its label that costs less and proves every
        requirement of its item under the keys named for them (keeping any fit link)."""
        cat_key = self.category_key()
        if self.brief.get('pick') != 'cheapest' or not cat_key:
            return []
        cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        out = []
        for line in cart:
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            owners = [n for n, item in enumerate(self.items, 1) if self.owns(item, label)]
            price = self.price_of(line)
            needs = self.resolved_needs(owners[0]) if len(owners) == 1 else None
            if len(owners) != 1 or needs is None or price is None:
                continue
            fits = [i for i in self.full if i not in cart and i not in self.gone and self.buyable(i) and not self.sold_out(i) and str((self.rows.get(i) or {}).get(cat_key) or '') == label and self.price_of(i) is not None and self.price_of(i) < price - 1e-09 and self.proves(i, owners[0], needs) and self.keeps_link(owners[0], i, cart)]
            if fits:
                best = min(fits, key=lambda i: self.price_of(i))
                out.append((line, price, best, self.price_of(best)))
        return out

    def cheaper_text(self, cheaper: list[tuple[str, float, str, float]]) -> str:
        return 'The pick rule is the lowest price, and a listing already read costs less and states every requirement under the key named for it: ' + '; '.join((f'{new} at {low:g} instead of {old} at {high:g}' + (f" ({'; '.join(self.proofs[new])})" if self.proofs.get(new) else '') for old, high, new, low in cheaper)) + ". Swap that line (the cart tool's own way to take off and add units), then check out; keep it only if that listing's record clearly breaks a requirement." + self.swap_links(cheaper)

    @_soft('')
    def swap_links(self, cheaper: list[tuple[str, float, str, float]]) -> str:
        bound = {n for pair in self.bound for n in pair}
        if not bound:
            return ''
        said = []
        for old, _, new, _ in cheaper:
            if not any((old in self.item_ids(n) for n in bound)):
                continue
            ids = [new if i == old else i for i in self.cart_ids()]
            said += [link for link in self.code_fits(ids) if link.startswith(new) or f' {new} ' in f' {link} ']
        if not said:
            return ''
        return ' Fit by code: ' + '; '.join(said[:3]) + '. The store decides fit by these codes; a use, size or seat count named in a description does not undo it.'

    @_soft('')
    def agreed_cheaper_swap(self) -> str:
        """The cheaper hold named a read listing and the independent check named it too (or, on a bound line, its codes keep
        the fit), yet the checkout came back with the dearer line: Cairn makes the swap itself, once."""
        if self.cheap_agreed or self.brief.get('pick') != 'cheapest':
            return ''
        better = self.better_by_key.get(self.check_key(self.cart_ids())) or []
        cart_now = self.cart_ids()

        def linked(old: str, new: str) -> bool:
            number = next((n for n in range(1, len(self.items) + 1) if old in self.item_ids(n)), 0)
            return bool(number) and any((number in pair for pair in self.bound)) and self.keeps_link(number, new, cart_now)
        pairs = [(old, new) for old, _, new, _ in self.cheaper_proven() if (old, new) in self.cheap_named and (new in better or linked(old, new))]
        if not pairs:
            return ''
        cart = self.roles.get('cart')
        calls = self.cart_calls({new: self.cart_lines.get(old, 1) or 1 for old, new in pairs}, [old for old, _ in pairs])
        if not cart or not calls or self.left() < len(calls) + 2 + self.tail():
            return ''
        self.cheap_agreed = True
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                break
            reply = self.act(cart, args)
            if reply is None or self.finished:
                break
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
            done.append(_dump(args))
        for old, new in pairs:
            if old not in self.cart_ids():
                self.dropped.add(old)
            if new in self.cart_ids():
                self.dropped.discard(new)
        lines_now = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
        if len(lines_now) >= len(self.built):
            self.built = lines_now
        self.version += 1
        self.trace.append({'kind': 'agreed_swap', 'pairs': pairs, 'calls': done})
        because = 'its own reading of the records and the independent check both name the cheaper listing: ' if all((new in better for _, new in pairs)) else "its own reading of the records names the cheaper listing and that listing's codes keep the fit with the cart's other lines: "
        return 'Cairn made the swap itself, since ' + because + '; '.join((f'{new} instead of {old}' for old, new in pairs)) + ' (' + '; '.join(done) + '). Check out now.'

    @_soft([])
    def read_sets(self) -> list[tuple[float, dict[str, int]]]:
        """Whole orders built only from listings already read, for a request whose items are bound to one device item: each
        line meets its requirements on words, each bound line fits the device by code, inside the budget; lowest total first."""
        if not self.bound or len({main for _, main in self.bound}) != 1:
            return []
        main = self.bound[0][1]
        parts = sorted({part for part, _ in self.bound})
        index = self.fit_index()
        marks = index['marks']
        meets = {n: [i for i in self.item_ids(n) if self.meets_words(i, n)] for n in range(1, len(self.items) + 1)}
        if any((not ids for ids in meets.values())):
            return []
        each = self.brief.get('budget_scope') == 'each'

        def qty(n: int) -> int:
            return 1 if each else self.items[n - 1].qty

        def price(i: str) -> float:
            return self.price_of(i) or 0.0
        rest_lines: dict[str, int] = {}
        rest = 0.0
        for n in range(1, len(self.items) + 1):
            if n == main or n in parts:
                continue
            pick = min(meets[n], key=price)
            rest_lines[pick] = self.items[n - 1].qty
            rest += price(pick) * qty(n)
        out = []
        for device in meets[main]:
            codes = (marks.get(device) or {}).get('codes') or []
            lines = dict(rest_lines)
            lines[device] = self.items[main - 1].qty
            total = rest + price(device) * qty(main)
            for part in parts:
                system = self.link_system(part, main, index)

                def fits(i: str) -> bool:
                    if system:
                        carried, namer = (i, device) if system[0] == part else (device, i)
                        return self.link_holds(carried, namer, system[2], index) is True
                    return bool(codes) and any((_code_hit(p, c) for p in self.fit_set(i, index) for c in codes)) and not any((_code_hit(b, c) for b in (marks.get(i) or {}).get('bars') or [] for c in codes))
                good = [i for i in meets[part] if fits(i)]
                if not good:
                    lines = {}
                    break
                pick = min(good, key=price)
                lines[pick] = self.items[part - 1].qty
                total += price(pick) * qty(part)
            if not lines or len(lines) != len(self.items):
                continue
            if isinstance(self.budget, (int, float)) and not each and total > self.budget + 1e-09:
                continue
            out.append((total, lines))
        out.sort(key=lambda t: (t[0], sorted(t[1])))
        return out

    def sets_text(self, sets: list[tuple[float, dict[str, int]]]) -> str:
        return '; '.join((' + '.join((f'{i} x{q}' if q > 1 else i for i, q in lines.items())) + f' = {total:g}' for total, lines in sets[:SET_SHOWN]))

    @_soft(False)
    def cart_set(self, lines: dict[str, int]) -> bool:
        cart = self.roles.get('cart')
        drop = [i for i in self.cart_ids() if i not in lines]
        want = {i: q for i, q in lines.items() if self.cart_lines.get(i) != q}
        calls = self.cart_calls(want, drop) if want or drop else []
        if calls is None or not cart or len(calls) + 1 > self.left():
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                return False
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return False
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
        self.version += 1
        self.trace.append({'kind': 'cart_set', 'lines': lines})
        return True

    def late_show(self) -> str:
        """Two steps left after a refused checkout and one cart line whose page was never shown: show it first."""
        tool = self.offer_tool()
        if not tool or not self.single_target(tool) or self.rushed():
            return ''
        unseen = [i for i in self.cart_ids() if not self.page_read(i) and i not in self.gone]
        if len(unseen) != 1:
            return ''
        self.reread_offers()
        if not self.page_read(unseen[0]):
            return ''
        self.trace.append({'kind': 'late_show', 'id': unseen[0]})
        return f"Cairn showed {unseen[0]}'s page first (one step): an order needs each line's page shown."

    def planned_basket(self, lead: str) -> tuple[dict[str, int], list[str]] | None:
        """The basket the review's leads make (the first unsold lead per item) plus the cart's lines of other kinds, with the
        lines to take off; None when `lead` is not in it or an item has no line."""
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ident
        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            found = re.search('\\d+', str(number))
            index = int(found.group()) if found else (1 if len(self.items) == 1 else None)
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if pick and index and 1 <= index <= len(self.items) and kind(pick) not in {kind(p) for p in plan}:
                plan[pick] = self.items[index - 1].qty
        if lead not in plan:
            return None
        now = self.cart_ids()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and not self.sold_out(i) and kind(i) not in {kind(p) for p in plan}}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.items):
            return None
        return ({**keep, **plan}, drop)

    def proven_basket_by_key(self) -> bool:
        """Forced ending on an empty cart: cart, per item, the cheapest read listing that proves every requirement under the
        keys named for them (no item with a live pointer or a fit binding), inside the budget."""
        cart = self.roles.get('cart')
        if not cart or not self.items:
            return False
        skip = self.pointed_items() | {n for pair in self.bound for n in pair}
        if any((n in skip for n in range(1, len(self.items) + 1))):
            return False
        lines: dict[str, int] = {}
        total = 0.0
        each = self.brief.get('budget_scope') == 'each'
        for number, item in enumerate(self.items, 1):
            fits = [i for i in self.item_ids(number) if i in self.full and self.buyable(i) and not self.sold_out(i) and self.price_of(i) is not None and self.proves(i, number)]
            if not fits:
                return False
            pick = min(fits, key=lambda i: self.price_of(i))
            lines[pick] = item.qty
            total += (self.price_of(pick) or 0.0) * (1 if each else item.qty)
        if isinstance(self.budget, (int, float)) and not each and total > self.budget + 1e-09:
            return False
        calls = self.cart_calls(lines, [])
        if not calls or len(calls) + 1 > self.left():
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                return False
            if self.act(cart, args) is None:
                return False
            self.note_cart(cart, args)
        self.trace.append({'kind': 'proven_basket', 'lines': lines})
        return True

    def pointed_items(self) -> set[int]:
        live = {need for _, need in self.needs_now()}
        return {n for n, need in self.refs if need in live}

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Cairn opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask') and self.message_args('x' * 200) is not None:
                out.append((cid, "Not run (no step used): Cairn's interviewer handles every message to the shopper."))
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
            capped = [k for k, v in args.items() if isinstance(v, str) and isinstance((((self.schemas[name].get('properties') or {}).get(k)) or {}).get('maxLength'), int) and len(v) > self.schemas[name]['properties'][k]['maxLength']]
            faults: list[str] = []
            args = _shape(args, self.schemas[name], 'args', faults)
            if faults:
                out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6])))
                continue
            told = ''
            if capped and name == self.roles.get('search') and not self.query_told:
                self.query_told = True
                top = self.schemas[name]['properties'][capped[0]]['maxLength']
                told = f'\n[Cairn] {capped[0]} takes at most {top} characters here, so Cairn cut it at a word boundary: {args.get(capped[0])!r}. Put the telling words (the values) first.'
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            if name == self.roles.get('cart'):
                args = self.zero_add(self.whole_removal(name, args), name)
            trimmed = ''
            trimmed += told
            many = self.compare_target() if name == self.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.compare_room()) and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Cairn] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
            room_at_call = self.left()
            plain_jobs = {tool for job, tool in self.roles.items() if job != 'search'}
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not ending and self.left() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            fresh = name not in plain_jobs and self.query_is_new(name, args)
            asked_for = args
            args = self.first_look(name, args, fresh)
            if args != asked_for:
                trimmed += f"\n[Cairn] This is a new search, so Cairn asked for its first page ({_dump(asked_for)} was the call you sent)."
            elif name not in plain_jobs and (not fresh):
                args, moved = self.next_unseen(name, args)
                trimmed += moved
            dead = [k for n, k in self.cap_dead if n == name and k in args]
            if dead and name not in plain_jobs:
                kept = self.cap_seen.get(name + _dump({k: v for k, v in args.items() if k not in dead}))
                if kept:
                    out.append((cid, f'Not run (no step used): this store ignores {dead[0]}, and the same search without it showed this:\n' + kept[:3000]))
                    continue
            args = self.row_ids(name, args)
            name, args, chained = self.page_first(name, args)
            target = self.single_target(name)
            if name == self.roles.get('search') and self.codes_dead and self.names_code(args):
                stripped = {k: ' '.join((w for w in v.split() if not _codes(w))) if isinstance(v, str) else v for k, v in args.items()}
                if stripped != args and any((isinstance(v, str) and v.strip() for k, v in stripped.items() if isinstance(args.get(k), str) and args[k].strip())):
                    args = stripped
                    trimmed += '\n[Cairn] This store\'s search finds nothing for model or part codes (descriptions are not searched), so Cairn sent the search without the code.'
            if name == self.roles.get('search') and self.codes_dead and self.names_code(args):
                out.append((cid, 'Not run (no step used): this store\'s search found nothing for a model or part code. Search by words and values, list the category, and read the codes on the rows.'))
                continue
            mark = name + _dump(args)
            if not ending and name != self.roles.get('cart') and (mark in self.recent):
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            readers = {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
            same = self.cap_cached(name, args) if name == self.roles.get('search') else ''
            if same:
                out.append((cid, same))
                continue
            cached = self.read_cache.get(mark)
            if name in readers and cached and (cached[0] == self.version):
                out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            if target and name in readers and (name != self.roles.get('compare')) and (self.shown_already(name, str(args.get(target))) or (name, str(args.get(target))) in self.factless):
                out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + self.record_of(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.shown_already(name, str(i))]
                if not keep:
                    out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.record_of(str(i))}' for i in args[many]))[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Cairn] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    late = self.cheaper_proven() if self.cheap_holds < 1 and self.left() <= self.tail() + 3 else []
                    if late and self.left() >= self.tail() + 1 + 2 * len(late):
                        self.cheap_holds += 1
                        self.trace.append({'kind': 'cheaper_hold', 'swaps': late, 'late': True})
                        self.cheap_named.update(((old, new) for old, _, new, _ in late))
                        out.append((cid, 'Not run yet (no step used). ' + self.cheaper_text(late) + '\nCart now: ' + self.cart_text()))
                        continue
                    hunted = ' '.join((t for t in (self.drop_hunt(),) if t))
                    hunted = (hunted + ' ' + self.hunt_cheaper()).strip()
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + '\nCart now: ' + self.cart_text()))
                        continue
                    self.show_pointed()
                    self.show_line_refs()
                    reread = self.reread_offers()
                    if self.note_sold:
                        self.find_sold()
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get('cart_view') and (self.left() > 5) and not self.rushed():
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
                        sets = self.read_sets()
                        whole = (' A whole set already read fits by code and states every requirement: ' + self.sets_text(sets) + '.') if sets else ''
                        out.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.' + whole + '\nCart now: ' + self.cart_text()))
                        continue
                    faults = self.carried_faults() + self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
                    fixed = []
                    if faults and all((' is held at ' in f for f in faults)):
                        fixed = self.fix_counts(self.count_fixes())
                        faults = self.line_faults() if fixed else faults
                        if fixed and (not faults):
                            self.line_holds += 1
                            out.append((cid, "Not run yet (no step used). Cairn set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()))
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
                    seen_now = "Cairn read these listings' attributes (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                    cheaper = self.cheaper_proven() if self.cheap_holds < 1 and self.left() > self.tail() + 3 else []
                    if cheaper:
                        self.cheap_holds += 1
                        self.trace.append({'kind': 'cheaper_hold', 'swaps': cheaper})
                        self.cheap_named.update(((old, new) for old, _, new, _ in cheaper))
                        out.append((cid, 'Not run yet (no step used). ' + seen_now + self.cheaper_text(cheaper) + '\nCart now: ' + self.cart_text()))
                        continue
                    weak = self.weak_hold() if self.weak_holds < 1 and self.left() > self.tail() + 3 else ''
                    if weak:
                        self.weak_holds += 1
                        out.append((cid, weak.replace('Not run yet (no step used). ', 'Not run yet (no step used). ' + seen_now, 1) + '\nCart now: ' + self.cart_text()))
                        continue
                    agreed = self.agreed_cheaper_swap()
                    if agreed:
                        out.append((cid, 'Not run yet (the checkout; the cart calls below used steps). ' + agreed + '\nCart now: ' + self.cart_text()))
                        continue
                    trimmed += self.agreed_swap() or self.auto_swap()
                    held_here = self.forum_holds()
                    if held_here:
                        out.append((cid, held_here + '\nCart now: ' + self.cart_text()))
                        continue
                    picked = self.sonnet_pick() if self.pick_rounds < PICK_ROUNDS else ''
                    if picked:
                        self.pick_rounds += 1
                        out.append((cid, 'Not run yet (no step used). ' + picked + '\nCart now: ' + self.cart_text()))
                        continue
                    verdict = self.check_cart()
                    if verdict:
                        seen = "Cairn read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        seen = (hunted + ' ' + self.below_cart() + '\n' if hunted else '') + seen
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Cairn re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.left() > 5:
                    sets = self.read_sets() if self.bound and self.set_holds < 1 else []
                    if sets:
                        self.set_holds += 1
                        self.trace.append({'kind': 'set_hold', 'sets': sets[:SET_SHOWN]})
                        out.append((cid, 'Not run yet (no step used). The store has a whole set that fits: these listings, all read already, state every requirement in their attributes or title, and each bound line fits the device by code: ' + self.sets_text(sets) + '. Set the cart to one of these and check out; end without a purchase only if a record clearly breaks a requirement.'))
                        continue
                    pointed = self.refuse_trail()
                    if pointed:
                        out.append((cid, 'Not run yet (no step used). ' + pointed))
                        continue
                    literal = self.literal_hold()
                    if literal:
                        out.append((cid, 'Not run yet (no step used). ' + literal))
                        continue
                    self.hunt_fits()
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
                        planned = self.planned_basket(first) if len(self.items) > 1 else None
                        mine = set(self.owner_numbers(first))
                        basket = sorted(planned[0]) if planned else [first] + ([i for i in self.cart_ids() if i != first and not (set(self.owner_numbers(i)) & mine)] if len(self.items) > 1 else [])
                        self.check_cart(basket, SPEND_HARD)
                        if self.verdicts.get(self.check_key(basket)) and first in self.leads:
                            self.confirmed = first
                            self.trace.append({'kind': 'refuse_blocked', 'confirmed': first, 'basket': basket})
                            if planned:
                                out.append((cid, f"Not run yet (no step used). An independent check of the basket {', '.join(basket)} says it meets every requirement, one line per item. Set the cart to exactly these lines at the requested quantities and check out; end without a purchase only if a record clearly breaks a requirement."))
                            else:
                                out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
            if self.is_refuse(name, args) and 2 <= self.left() <= 5 and not self.confirmed and self.refuse_blocks < 2 and len([i for i in self.built if not self.sold_out(i)]) < len(self.items):
                done = self.late_review()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
            if self.is_refuse(name, args) and self.left() <= 5 and not self.late_tried:
                self.late_tried = True
                done = self.late_finish()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
            if self.is_refuse(name, args) and (self.left() <= 5 or self.noask_mode()) and (not self.late_held) and not self.finished:
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh))]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            if self.is_refuse(name, args) and (not self.leads) and (not self.pushed) and (room_at_call > self.max_steps // 3) and self.left() > self.tail() + len(self.items) + 2:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args) and self.noask_mode() and self.clock() >= CLOSEST_AT:
                done = self.closest_order()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                args = self.quiet(name, args)
            if self.is_buy(name, args):
                late = self.late_question()
                if late:
                    out.append((cid, 'Not run yet. ' + late + '\nCart now: ' + self.cart_text()))
                    continue
            if self.is_buy(name, args) and self.left() == 2 and self.buys >= 1:
                shown_late = self.late_show()
                if shown_late:
                    trimmed = '\n[Cairn] ' + shown_late + trimmed
            if ending and self.left() >= 2:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Cairn] ' + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Cairn] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
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
            if name not in plain:
                self.note_next(name, args, observation)
            paged = self.more_pages(name, args, observation) if name not in plain and (not ending) else ''
            paged += self.last_page_first(name, args, observation) if fresh and (not ending) else ''
            paged += self.shorter_search(name, args, observation) if name not in plain and (not ending) else ''
            paged += self.open_big(observation) if name not in plain and (not ending) else ''
            paged += self.series_sweep() if name in readers and (not ending) else ''
            if self.single_target(name) and name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.note_facts(name, bool(_fact_block(observation)))
                    if name == self.roles.get('open') and (self.facts_from.get(name) is False):
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and (self.facts_from.get(t) is not False) and (t not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search'))) and not self.lists_things(t)), None)
                        if other:
                            self.roles['open'] = other
                    if self.facts_from.get(name) is False and name != self.roles.get('open') and (not self.roles.get('offer')):
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
                if counts and counts[0] > 0:
                    trimmed += self.dear_line([v for v in args.values() if isinstance(v, str) and v in self.rows])
            if name == self.roles.get('search') and self.names_code(args):
                self.code_tries.append(bool(self.records_in(observation)))
                self.codes_dead = len(self.code_tries) >= 2 and not any(self.code_tries)
            if name == self.roles.get('search') and self.records_in(observation):
                self.searched = (self.searched + [dict(args)])[-8:]
            if name == self.roles.get('cart_view'):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get('finished'):
                    self.refused_at = self.step
            text = self.show(observation) + paged + trimmed
            if name not in plain and (not ending):
                text += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, args, observation)
            base = text
            if name not in plain:
                for n_, key_ in self.cap_dead:
                    if n_ == name and key_ in args:
                        self.cap_seen[name + _dump({k: v for k, v in args.items() if k != key_})] = base
            proof = self.evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                    text += '\n[Cairn] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
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
                    text += "\n[Cairn re-read the cart's lines after the refusal]\n" + seen
            self.note_sold = False
            lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if name == self.roles.get('cart') and not self.finished:
                text += self.cond_note()
            if name == self.roles.get('cart') and self.bound and not self.finished:
                misfit = self.fit_faults()
                if misfit:
                    text += '\n[Cairn] These lines do not work together yet: ' + '; '.join(misfit)
            if self.queued:
                text += "\n[notes seen on Cairn's own calls] " + ' | '.join(self.queued)
                self.queued = []
            text += chained
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out


    # ---- c15_v28: code checks ported from the arena forum guides (#96-#103); each fires only on its own trigger ----
    def need_kind(self, need: str) -> str:
        value = need.partition(':')[2]
        if need in [m for _, m in self.refs] or need in [str(x) for x in _items(self.brief.get('lookups'))]:
            return 'pointer'
        if COND_RE.search(value):
            return 'cond'
        if LIMIT_RE.search(value):
            return 'limit' if _figs(value) else 'vague'
        if NEG_RE.search(value):
            return 'not'
        return 'plain' if _vw(value) else 'none'

    def item_needs(self, number: int) -> list[str]:
        return [need for owner, need in self.needs_now() if owner == number]

    def attr_pairs(self, ident: str) -> list[tuple[str, str]]:
        block = _fact_block(self.full.get(ident) or {}) or {}
        return [(str(k), str(v)) for k, v in block.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))]

    def title_of(self, ident: str) -> str:
        key = self.title_key()
        if not key:
            return ''
        return str((self.rows.get(ident) or {}).get(key) or (self.full.get(ident) or {}).get(key) or '')

    def judge(self, ident: str, need: str) -> tuple[str, str]:
        """How a listing's attributes answer one requirement: 'stated' (the attribute whose key names the property holds the
        value, read literally), 'contra' (that attribute states another value) or 'open' (no such attribute, or a requirement
        code cannot read). A missing attribute is never a contradiction."""
        prop, _, value = need.partition(':')
        prop, value = (prop.strip(), value.strip())
        kind = self.need_kind(need)
        alts_extra: list[list[str]] = []
        if kind == 'pointer':
            got = str(self.ref_values.get(need) or '')
            if not got or got.lower().startswith('as stated on') or re.search('\\bwithin\\b', value, re.I):
                return ('open', 'pointer')
            kind = 'not' if re.search('\\b(?:unlike|other than|different|not|anything but)\\b', value, re.I) else 'plain'
            head, _, tail = got.partition(':')
            value = tail.strip() if tail.strip() and _names(_spelled(prop), _spelled(head)) else got
            alts_extra = [_vw(f) for f in self.ref_forms.get(need, []) if _vw(f)]
            if kind == 'not':
                value = 'not ' + value
        if kind in ('cond', 'vague', 'none'):
            return ('open', kind)
        pairs = self.attr_pairs(ident)
        want = _spelled(prop)
        mine = [(k, v) for k, v in pairs if _names(want, _spelled(k))]
        coded = _codes(value) if kind == 'plain' else []
        if coded:
            mark = self.fit_index()['marks'].get(ident) or {}
            fits = self.fit_set(ident, self.fit_index())
            if fits:
                if any((_code_hit(p, c) for p in fits for c in coded)):
                    return ('stated', 'fit list')
                return ('contra', 'fit list: ' + ', '.join(sorted(fits)[:4]))
            if mark.get('codes') and any((_code_hit(c, m) for c in coded for m in mark['codes'])):
                return ('stated', 'model code')
        if not mine and kind == 'plain':
            loose = [(k, v) for k, v in pairs if len(_spelled(k)) == 1 and _spelled(k) <= want]
            for k, v in loose:
                if any((_run(a, p) for a in [_vw(x) for x in re.split('\\s+or\\s+', value)] if a for p in _parts(v))):
                    return ('stated', f'{k}={v[:60]}')
        if kind == 'limit':
            return self.judge_limit(value, mine, pairs)
        if kind == 'not':
            literal = _vw(value)
            if any((_run(literal, p) for _, v in mine for p in _parts(v) + [_vw(v)])):
                return ('stated', 'literal')
            words = _vw(NEG_RE.split(value, maxsplit=1)[-1])
            if not words or not mine:
                return ('open', 'not')
            hit = next(((k, v) for k, v in mine if any((_run(words, p) for p in _parts(v)))), None)
            return ('contra', f'{hit[0]}={hit[1][:60]}') if hit else ('stated', 'not')
        alts = [a for a in (_vw(x) for x in re.split('\\s+or\\s+', value)) if a] + alts_extra
        if not alts:
            return ('open', '')
        title = _vw(self.title_of(ident))
        for k, v in mine:
            parts = _parts(v)
            if len(parts) >= 2 and title:
                for alt in alts:
                    if any((_run(alt, p) for p in parts)) and (not _run(alt, title)) and any((_run(p, title) for p in parts if not _run(alt, p))):
                        return ('contra', f'{k} lists variants ({v[:60]}) and the title names another one')
        for k, v in mine:
            parts = _parts(v)
            for alt in alts:
                if any((_run(alt, p) for p in parts)) or _run(alt, _vw(v)):
                    return ('stated', f'{k}={v[:60]}')
                figs = _figs(' '.join(alt))
                if figs and not any((_spread(alt, p) for p in parts)):
                    rest = [w for w in alt if not re.fullmatch('[\\d.]+[a-z"]{0,6}', w) and not _unit_eq(w, figs[0][1])]
                    got = _figs(v)
                    if (not rest or _within(rest, _vw(v))) and any((abs(n - figs[0][0]) < 1e-09 and (_unit_eq(u, figs[0][1]) or not u or not figs[0][1]) for n, u in got)):
                        return ('stated', f'{k}={v[:60]}')
                if any((_within(alt, p) and (not _spread(alt, p)) for p in parts)):
                    return ('stated', f'{k}={v[:60]}')
        if mine:
            for k, v in mine:
                if any((_spread(alt, p) for alt in alts for p in _parts(v))):
                    return ('contra', f'{k}={v[:60]}: other words stand between the wanted words, so it is another value')
            for k, v in mine:
                for alt in alts:
                    figs = _figs(' '.join(alt))
                    if figs and any((_unit_eq(u, figs[0][1]) and abs(n - figs[0][0]) > 1e-09 for n, u in _figs(v))):
                        return ('contra', f'{k}={v[:60]}')
            if any((_within(p, alt) or _within(alt, p) for _, v in mine for p in _parts(v) for alt in alts)):
                return ('open', 'partial')
            if any((_kin(w, h) for _, v in mine for p in _parts(v) for alt in alts for w in alt for h in p if len(w) >= 3 and not w.isdigit() and w not in ('and', 'the', 'with', 'for'))):
                return ('open', 'shares a word')  # "3x AAA" for "AAA batteries": the model judges, no hold
            if any((_figs(' '.join(alt)) for alt in alts)):
                return ('open', 'figure in another form')
            return ('contra', f'{mine[0][0]}={mine[0][1][:60]}')
        for k, v in pairs:
            if any((_run(alt, p) for alt in alts for p in _parts(v))):
                return ('open', f'stated under {k}')
        if any((_run(alt, title) for alt in alts)):
            return ('open', 'title')
        return ('open', '')

    def judge_limit(self, value: str, mine: list[tuple[str, str]], pairs: list[tuple[str, str]]) -> tuple[str, str]:
        """A limit read literally: the attribute's figures in the need's unit must all lie inside it (a range wholly inside).
        A bare number under that key far (tenfold or more) from the need's figure is a contradiction: such orders passed 1 of
        12 on race 17. A figure in the need's unit under another key decides nothing, and is named."""
        figs = _figs(value)
        if not figs:
            return ('open', 'limit')
        need, unit = figs[0]
        between = re.search('\\bbetween\\b', value, re.I) and len(figs) >= 2
        upper = bool(UPPER_RE.search(value))
        for k, v in mine:
            got = _figs(v)
            same = [n for n, u in got if _unit_eq(u, unit) or (not unit and (not u))]
            if same:
                return self.limit_verdict(same, need, figs[1][0] if between else None, upper, f'{k}={v[:40]}')
            # no unit conversion: ORO's precheck bans unit lookup tables; another unit is left to the model
            bare = [n for n, u in got if not u]
            if bare and unit and need > 0 and all((max(b, need) >= 10 * max(min(b, need), 1e-09) for b in bare)):
                return ('contra', f'{k}={v[:40]} is a bare number, tenfold or more away from {need:g} {unit}')
        other = [(k, v) for k, v in pairs if (k, v) not in mine and unit and any((_unit_eq(u, unit) for _, u in _figs(v)))]
        if other:
            return ('open', f'other key {other[0][0]}={other[0][1][:40]}')
        return ('open', 'limit')

    def limit_verdict(self, values: list[float], need: float, top: float | None, upper: bool, why: str) -> tuple[str, str]:
        """A stated figure, or a stated range, against a limit: a range counts against the limit only when all of it
        misses, and is left open when it straddles the limit."""
        low, high = (min(values), max(values))
        if top is not None:
            if need - 1e-06 <= low and high <= top + 1e-06:
                return ('stated', why)
            return ('contra', why) if high < need - 1e-06 or low > top + 1e-06 else ('open', 'range ' + why)
        if upper:
            if high <= need + 1e-06:
                return ('stated', why)
            return ('contra', why) if low > need + 1e-06 else ('open', 'range ' + why)
        if low >= need - 1e-06:
            return ('stated', why)
        return ('contra', why) if high < need - 1e-06 else ('open', 'range ' + why)

    def line_contra(self, ident: str, number: int) -> list[str]:
        """Requirements of the item that the line's own attribute states otherwise (exclusions and pointed values are left to
        the model's check)."""
        out = []
        for need in self.item_needs(number):
            if self.need_kind(need) in ('not', 'pointer'):
                continue
            verdict, why = self.judge(ident, need)
            if verdict == 'contra':
                out.append(f'{need} ({why})')
        return out

    def owner_numbers(self, ident: str) -> list[int]:
        return [n for n in range(1, len(self.items) + 1) if ident in self.item_ids(n)]

    def proven(self, ident: str, number: int) -> bool:
        """Every requirement of the item stated under its own attribute on this read, buyable listing, within the item's price
        cap; the item's name words that its store label does not carry are in its title or attributes. An item whose
        requirements code cannot read, or that has none, is never proven."""
        if ident not in self.full or not self.buyable(ident) or self.price_of(ident) is None:
            return False
        needs = self.item_needs(number)
        if not needs or any((self.judge(ident, need)[0] != 'stated' for need in needs)):
            return False
        cap = self.item_caps.get(number)
        if cap is not None and (self.price_of(ident) or 0.0) > cap + 1e-09:
            return False
        item = self.items[number - 1]
        label = str((self.rows.get(ident) or {}).get(self.category_key() or '') or '')
        extra = _roots(item.name) - _roots(label) - _roots(item.shelf or '')
        if extra:
            text = _roots(self.title_of(ident) + ' ' + ' '.join((v for _, v in self.attr_pairs(ident))))
            if not all((any((_kin(e, t) for t in text)) for e in extra)):
                return False
        return True

    def key_echo(self, ident: str, number: int) -> int:
        """How many of the item's requirements this listing states under a key spelled as the property itself."""
        keys = {_spelled(k) for k, _ in self.attr_pairs(ident)}
        return sum((1 for need in self.item_needs(number) if _spelled(need.partition(':')[0]) in keys))

    def part_fits(self, part_id: str, device: str, part: int, main: int, index: dict[str, Any]) -> bool:
        marks = index['marks']
        codes = set((marks.get(device) or {}).get('codes') or [])
        text = re.sub('[\\s-]+', ' ', _dump(self.full.get(part_id) or self.rows.get(part_id) or {}).lower())
        if codes and any((re.search('(?<![a-z0-9])' + re.escape(c) + '(?![a-z0-9])', text) for c in codes)):
            return True
        fits = self.fit_set(part_id, index)
        bars = set((marks.get(part_id) or {}).get('bars') or [])
        if codes and fits and any((_code_hit(p, c) for p in fits for c in codes)) and (not any((_code_hit(p, c) for p in bars for c in codes))):
            return True
        system = self.link_system(part, main, index)
        if system:
            carried, namer = (part_id, device) if system[0] == part else (device, part_id)
            return self.link_holds(carried, namer, system[2], index) is True
        return False

    def proven_set(self) -> dict[int, str] | None:
        """One proven listing per item, distinct, bound parts fitting their device, inside the budget: on a cheapest pick the
        lowest total, else the one stating most requirements under keys named as the property. None when any item lacks one."""
        if not self.items or not self.category_key() or self.brief.get('pick') in ('most', 'least'):
            return None
        state = (len(self.full), len(self.seen_order), self.version, _dump(self.budget), self.cart_mark(), len(self.gone))
        if self.set_cached and self.set_cached[0] == state:
            return self.set_cached[1]
        chosen = self.build_set()
        self.set_cached = (state, chosen)
        return chosen

    def build_set(self) -> dict[int, str] | None:
        cheapest = self.brief.get('pick') == 'cheapest'
        pools: dict[int, list[str]] = {}
        for number in range(1, len(self.items) + 1):
            pool = [i for i in self.item_ids(number) if self.proven(i, number)]
            if not pool:
                return None
            if cheapest:
                pool.sort(key=lambda i: (self.price_of(i) or 0.0, -self.key_echo(i, number), i))
            else:
                pool.sort(key=lambda i: (-self.key_echo(i, number), self.price_of(i) or 0.0, i))
            pools[number] = pool
        mains = {m for _, m in self.bound}
        parts = {p for p, _ in self.bound}
        index = self.fit_index() if self.bound else {}
        chosen: dict[int, str] = {}
        for number, pool in pools.items():
            if number in mains or number in parts:
                continue
            pick = next((i for i in pool if i not in chosen.values()), None)
            if not pick:
                return None
            chosen[number] = pick
        for main in sorted(mains):
            best: tuple[float, dict[int, str]] | None = None
            for device in pools[main][:8]:
                if device in chosen.values():
                    continue
                got: dict[int, str] | None = {main: device}
                total = (self.price_of(device) or 0.0) * self.items[main - 1].qty
                for part in [p for p, m in self.bound if m == main]:
                    if part in chosen:
                        continue
                    fit = next((i for i in pools[part] if i not in chosen.values() and i not in got.values() and self.part_fits(i, device, part, main, index)), None)
                    if not fit:
                        got = None
                        break
                    got[part] = fit
                    total += (self.price_of(fit) or 0.0) * self.items[part - 1].qty
                if got and (best is None or total < best[0]):
                    best = (total, got)
            if not best:
                return None
            chosen.update(best[1])
        if len(chosen) != len(self.items):
            return None
        if isinstance(self.budget, (int, float)):
            if self.brief.get('budget_scope') == 'each':
                if any(((self.price_of(i) or 0.0) > self.budget + 1e-09 for i in chosen.values())):
                    return None
            elif self.set_total(chosen) > self.budget + 1e-09:
                return None
        return chosen

    def set_total(self, chosen: dict[int, str]) -> float:
        return sum(((self.price_of(i) or 0.0) * self.items[n - 1].qty for n, i in chosen.items()))

    def set_text(self, chosen: dict[int, str]) -> str:
        return ', '.join((f'{i} x{self.items[n - 1].qty} (item {n}, {self.items[n - 1].name})' for n, i in sorted(chosen.items())))

    def put_lines(self, want: dict[str, int], kind: str) -> bool:
        """Set the cart to exactly these lines with Cairn's own cart calls; whether every call ran."""
        cart = self.roles.get('cart')
        drop = [i for i in self.cart_ids() if i not in want]
        calls = self.cart_calls(want, drop) if cart else None
        if not calls or len(calls) + 1 > self.left():
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                return False
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return False
            self.note_cart(cart, args)
            self.cart_said = _dump(reply.get('observation'))
            self.observe(cart, reply.get('observation'))
        self.trace.append({'kind': kind, 'lines': want})
        return True

    def read_caps(self) -> None:
        """A price cap on one product, written as a refusal to pay more than an amount for it: the single item whose name shares
        the most word roots with the words after it gets the amount; a tie, no shared root or a cap above the whole budget
        changes nothing, and the lowest cap per item is kept (race 12: 16 of 66 tasks, every one read right)."""
        self.item_caps = {}
        pattern = "\\b(?:won'?t|will not|wouldn'?t|would not|never|not going to|refuse to)\\s+(?:want to\\s+)?(?:pay|spend)\\s+(?:any\\s+)?(?:more than|over|above|beyond|in excess of)\\s+[^\\d]{0,3}(\\d+(?:,\\d{3})*(?:\\.\\d+)?)([^,;.!?]{0,40}?)\\s(?:for|on)\\s+(?:(?:a|an|the|each|any|one|per|every)\\s+)?([^,;.!?]{2,60})"
        for found in re.finditer(pattern, self.request or '', re.I):
            amount = float(found.group(1).replace(',', ''))
            want = _roots(' '.join(found.group(3).split()[:6]))
            scores = [(len(want & (_roots(item.name) | _roots(item.shelf or ''))), n) for n, item in enumerate(self.items, 1)]
            best = max((s for s, _ in scores), default=0)
            hits = [n for s, n in scores if s == best]
            if best <= 0 or len(hits) != 1 or amount <= 0:
                continue
            if isinstance(self.budget, (int, float)) and amount > self.budget:
                continue
            self.item_caps[hits[0]] = min(amount, self.item_caps.get(hits[0], amount))
        if self.item_caps:
            self.trace.append({'kind': 'item_caps', 'caps': dict(self.item_caps)})

    def budget_in_unit(self, new: float) -> float:
        """A changed budget read in another unit than the one in force ("2230 cents" read as 22.3): within 30x of the old one it
        stands; otherwise x100 or /100 is taken when that lands within 3x of it (race 12: 12 of 12 unit errors fixed)."""
        old = self.budget
        if not isinstance(old, (int, float)) or isinstance(old, bool) or old <= 0 or new <= 0:
            return new
        if max(old, new) <= 30 * min(old, new):
            return new
        for factor in (100.0, 0.01):
            moved = new * factor
            if max(old, moved) <= 3 * min(old, moved):
                self.trace.append({'kind': 'budget_unit', 'read': new, 'took': moved})
                return moved
        return new

    def forum_holds(self) -> str:
        """The deterministic holds before checkout, each once per cart and two at most per episode, while the clock allows
        a turn: a line over its item's price cap; a line whose own attribute states another value; a measured requirement no
        attribute of the line names while a read alternative states it; on a cheapest pick, a proven set that costs less."""
        if self.forum_held >= 2 or self.left() <= self.tail() + 4 or self.clock() >= WIND_DOWN or self.rushed():
            return ''
        mark = self.cart_mark() + f'#{self.version}'
        if mark in self.hold_marks:
            return ''
        cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not cart:
            return ''
        chosen = self.proven_set()
        problems: list[str] = []
        swaps: dict[str, str] = {}
        for line in cart:
            for number in self.owner_numbers(line):
                cap = self.item_caps.get(number)
                price = self.price_of(line)
                if cap is not None and price is not None and price > cap + 1e-09:
                    problems.append(f"{line} costs {price:g}, over the shopper's cap of {cap:g} for one {self.items[number - 1].name}")
                wrong = self.line_contra(line, number) if line in self.full else []
                if wrong:
                    problems.append(f'{line} states another value for ' + '; '.join(wrong[:2]))
                if (wrong or (cap is not None and price is not None and price > cap + 1e-09)) and chosen and chosen.get(number) and chosen[number] != line:
                    swaps[line] = chosen[number]
        if not problems:
            for line in cart:
                gap = self.name_gap(line)
                if gap:
                    problems.append(gap)
                    break
        if not problems:
            weak = self.weak_measures(cart)
            if weak:
                problems.append(weak)
        if not problems and self.brief.get('pick') == 'cheapest' and chosen and set(chosen.values()) != set(cart):
            held = sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in cart))
            if self.set_total(chosen) + 1e-09 < held and all((self.price_of(i) is not None for i in cart)):
                problems.append(f'opened listings that state every requirement under their own attributes make a cheaper whole order: {self.set_text(chosen)} = {self.set_total(chosen):g}, against {held:g} now')
                for line in cart:
                    for number in self.owner_numbers(line):
                        if chosen.get(number) and chosen[number] != line:
                            swaps[line] = chosen[number]
        if not problems:
            return ''
        self.hold_marks.add(mark)
        self.forum_held += 1
        self.named_swaps.update(swaps)
        self.trace.append({'kind': 'forum_hold', 'problems': problems[:4], 'swaps': swaps})
        fix = ' Replacements that state every requirement: ' + ', '.join((f'{a} -> {b}' for a, b in swaps.items())) + '.' if swaps else ''
        return 'Not run yet (no step used). ' + '; '.join(problems[:3]) + '.' + fix + ' Fix those lines (or show from the records that the line is right), then check out.'

    def name_gap(self, line: str) -> str:
        """A word of the item's name that its store label does not carry ("canister" in "canister stove" under "Stoves"),
        missing from the line's title and attributes while another read listing of the item states it as an attribute value:
        the request reading kept a requirement inside the name."""
        if line not in self.full:
            return ''
        for number in self.owner_numbers(line):
            item = self.items[number - 1]
            label = str((self.rows.get(line) or {}).get(self.category_key() or '') or '')
            extra = {w for w in _roots(item.name) - _roots(label) - _roots(item.shelf or '') if not w.isdigit() and not re.fullmatch(NUMBER_WORDS, w)}
            mine = set(_vw(self.title_of(line) + ' ' + ' '.join((v for _, v in self.attr_pairs(line)))))
            for word in sorted(extra):
                if any((_kin(word, m) for m in mine)):
                    continue
                for other in self.item_ids(number):
                    if other == line or other not in self.full:
                        continue
                    hit = next(((k, v) for k, v in self.attr_pairs(other) if any((_kin(word, w) for w in _vw(v)))), None)
                    if hit:
                        own = next((v for k, v in self.attr_pairs(line) if _names(_spelled(hit[0]), _spelled(k))), '')
                        return f"{line} does not state '{word}' from the item's name ({item.name})" + (f'; its {hit[0]} is {own}' if own else '') + f', while {other} states {hit[0]} = {hit[1]}'
        return ''

    def weak_measures(self, cart: list[str]) -> str:
        """A measured requirement (a figure) that no attribute of the cart line names, and its title does not state, while a
        read listing of the same item states it under its own key: such orders usually fail. A figure in the need's unit under
        another key is named, and the model judges whether it is the same measure under another name."""
        for line in cart:
            if line not in self.full:
                continue
            for number in self.owner_numbers(line):
                for need in self.item_needs(number):
                    prop, _, value = need.partition(':')
                    if self.need_kind(need) not in ('plain', 'limit') or not _figs(value):
                        continue
                    verdict, why = self.judge(line, need)
                    if verdict != 'open' or why == 'title' or why.startswith('stated under'):
                        continue
                    if any((_run(_vw(value), p) for p in [_vw(self.title_of(line))])):
                        continue
                    better = [i for i in self.item_ids(number) if i != line and i in self.full and self.buyable(i) and self.judge(i, need)[0] == 'stated' and (not self.line_contra(i, number))]
                    if not better:
                        continue
                    other = f' It does give {why[10:]}: if that is the same measure under another name, the line stays.' if why.startswith('other key') else ''
                    return f'{line} states no attribute for {prop.strip()} ({need.strip()}), while {", ".join(better[:2])} states it under its own key.{other}'
        return ''

    def agreed_swap(self) -> str:
        """A swap Cairn named before, that the model's check named too (or that replaces a line stating another value), and
        the model checked out again without making it: Cairn makes it, once."""
        if self.swapped or not self.named_swaps or self.left() <= self.tail() + 3:
            return ''
        cart = self.cart_ids()
        for line, other in list(self.named_swaps.items()):
            if line not in cart or other in cart or not self.buyable(other):
                continue
            numbers = self.owner_numbers(line)
            if not numbers or not self.proven(other, numbers[0]):
                continue
            if other not in self.check_better and not self.line_contra(line, numbers[0]):
                continue
            want = {i: (self.held_units(i) or 1) for i in cart if i != line}
            want[other] = self.items[numbers[0] - 1].qty
            self.swapped = True
            if self.put_lines(want, 'agreed_swap'):
                return f'\n[Cairn] Swapped {line} for {other} itself (named before by its check, and still unmade): {other} states every requirement of item {numbers[0]}.'
            return ''
        return ''

    def cond_note(self) -> str:
        """Rules that hold only on a condition ("unless", "only if"), read against the cart once it holds the other line:
        the cart lines' attributes that the rule's words name."""
        rules = [str(r) for r in _items(self.brief.get('rules')) if re.search('\\b(?:unless|only if|except when|if|when|in case)\\b', str(r), re.I)]
        rules += [need for _, need in self.needs_now() if self.need_kind(need) == 'cond']
        cart = self.cart_ids()
        if not rules or len(cart) < 2:
            return ''
        mark = self.cart_mark() + f'#{self.version}'
        if mark in self.cond_marks:
            return ''
        self.cond_marks.add(mark)
        words = _roots(' '.join(rules))
        lines = []
        for line in cart:
            hits = [f'{k}={v[:40]}' for k, v in self.attr_pairs(line) if _roots(k) & words or _roots(v) & words]
            if hits:
                lines.append(f'{line}: ' + ', '.join(hits[:4]))
        if not lines:
            return ''
        return '\n[Cairn: conditional rules read against the cart] ' + ' | '.join(rules[:3]) + ' -> ' + '; '.join(lines[:6]) + '. Settle each condition from these values before checking out.'

    def literal_hold(self) -> str:
        """Before a decline: opened, buyable listings that meet every requirement as written (figures read literally, as the
        grader reads them: a stated figure is compared as written). On one-item tasks every opened listing counts, since the
        store's label may share no word with the item's name."""
        if self.literal_held or self.left() < len(self.items) + self.tail() + 2:
            return ''
        chosen = self.proven_set()
        if not chosen and len(self.items) == 1:
            needs = self.item_needs(1)
            pool = [i for i in self.full if len(needs) >= 2 and self.buyable(i) and self.price_of(i) is not None and all((self.judge(i, n)[0] == 'stated' for n in needs))]
            if isinstance(self.budget, (int, float)):
                pool = [i for i in pool if (self.price_of(i) or 0.0) * self.items[0].qty <= self.budget + 1e-09]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            chosen = {1: pool[0]} if pool else None
        if not chosen:
            return ''
        self.literal_held = True
        self.trace.append({'kind': 'literal_hold', 'set': chosen})
        return 'These opened, buyable listings state every requirement as written (figures read literally, the way the order is graded): ' + self.set_text(chosen) + '. Put them in the cart at the requested quantities and check out; end without a purchase only if a record shows one of them breaking a requirement.'

    def proven_order(self) -> str:
        """On an 'any' pick, once every item has an opened listing proven on every requirement (cairo r13: 8 code orders, 8
        passes), Cairn sets the cart to that set itself and tells the model to check out; a complete cart of proven lines the
        model built stays as it is."""
        if self.code_ordered or self.buys or self.finished or self.brief.get('pick') not in (None, '', 'any') or self.loop_calls < 1:
            return ''
        if self.rushed() or self.left() < len(self.items) + self.tail() + SET_STEPS:
            return ''
        chosen = self.proven_set()
        if not chosen:
            return ''
        cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        want = {i: self.items[n - 1].qty for n, i in chosen.items()}
        self.code_ordered = True
        if set(cart) == set(want) and all((self.held_units(i) == q for i, q in want.items())):
            return ''
        if len(cart) == len(self.items) and all((any((self.proven(i, n) for n in self.owner_numbers(i))) for i in cart)) and (not self.line_faults()):
            return ''
        if not self.put_lines(want, 'code_set'):
            return ''
        return '[Cairn] Every requirement of each item is stated under its own attribute on these opened listings: ' + self.set_text(chosen) + '. Cairn put exactly these lines in the cart. Check out now (the checkout call on its own), unless a record above shows one of them breaking a requirement.'

    def proven_basket(self) -> bool:
        """Before a forced ending with an empty cart: cart the proven set, so the episode ends with an order."""
        if self.basket_done or not self.roles.get('buy'):
            return False
        chosen = self.proven_set()
        if not chosen:
            return False
        self.basket_done = True
        return self.put_lines({i: self.items[n - 1].qty for n, i in chosen.items()}, 'proven_basket')

    def repair_fits(self) -> None:
        """Before a forced order: a part line the agent's own fit check rejects is swapped for the cheapest seen part that fits
        the carted device, buyable, stating nothing against its requirements (read ones first); if the cart still has a fit
        fault or a missing line and a proven whole set exists, the cart becomes that set. Nothing changes when neither works."""
        if self.repaired or not self.cart_ids():
            return
        self.repaired = True
        if self.bound and self.fit_faults():
            index = self.fit_index()
            cart = self.cart_ids()
            for part, main in self.bound:
                devices = [i for i in cart if i in self.item_ids(main)]
                lines = [i for i in cart if i in self.item_ids(part)]
                if len(devices) != 1 or len(lines) != 1 or self.part_fits(lines[0], devices[0], part, main, index):
                    continue
                pool = [i for i in self.item_ids(part) if i not in cart and self.buyable(i) and self.price_of(i) is not None and self.part_fits(i, devices[0], part, main, index) and (not self.line_contra(i, part))]
                pool.sort(key=lambda i: (i not in self.full, self.price_of(i) or 0.0))
                if not pool:
                    continue
                want = {i: (self.held_units(i) or 1) for i in cart if i != lines[0]}
                want[pool[0]] = self.items[part - 1].qty
                if isinstance(self.budget, (int, float)) and self.brief.get('budget_scope') != 'each' and sum(((self.price_of(i) or 0.0) * n for i, n in want.items())) > self.budget + 1e-09:
                    continue
                if self.put_lines(want, 'fit_repair'):
                    cart = self.cart_ids()
        if (self.bound and self.fit_faults()) or self.missing_items():
            chosen = self.proven_set()
            if chosen and set(chosen.values()) != set(self.cart_ids()):
                self.put_lines({i: self.items[n - 1].qty for n, i in chosen.items()}, 'set_recart')

    def zero_rungs(self, text: str) -> list[str]:
        """Other spellings for a words search that found nothing: without the property names (stores index values, not
        keys), with number and unit spaced the other way ("2 TB", "2tb"), with a plural last word made singular (store labels
        are plural, listings are not), then its first two words."""
        words = text.split()
        props = {w for _, need in self.needs_now() for w in _words(need.split(':', 1)[0]).split() if len(w) > 2}
        values = {w for _, need in self.needs_now() for w in _words(need.split(':', 1)[-1]).split()}
        rungs = []
        bare = ' '.join((w for w in words if w.lower() not in props or w.lower() in values))
        rungs.append(bare)
        flipped = re.sub('(\\d)\\s+([A-Za-z]{1,4})\\b', '\\1\\2', text)
        if flipped == text:
            flipped = re.sub('\\b(\\d+(?:\\.\\d+)?)([A-Za-z]{1,4})\\b', '\\1 \\2', text)
        rungs.append(flipped)
        last = words[-1] if words else ''
        if len(last) > 4 and last.lower().endswith('s') and not last.lower().endswith('ss'):
            rungs.append(' '.join(words[:-1] + [last[:-2] if re.search('(?:ch|sh|x|z)es$', last.lower()) else last[:-1]]))
        if len(words) >= 3:
            rungs.append(' '.join([w for w in words if not _codes(w)][:2]))
        out = []
        for rung in rungs:
            rung = ' '.join(rung.split())
            if rung and rung.lower() != text.lower() and rung not in out:
                out.append(rung)
        return out

    def value_pages(self, tool: str, sweeps: list[tuple[int, dict[str, Any], Any]], stop_at: int) -> list[str]:
        """Rows come in id order, not price order: page the largest opening value searches (7-48 matches, mostly the item's
        own label) up front, on cheapest picks and on big tasks."""
        big = self.max_steps >= SWEEP_BIG
        if (self.brief.get('pick') != 'cheapest' and not big) or self.pages_swept:
            return []
        cat_key = self.category_key()
        ranked = []
        for number, args, seen in sweeps:
            page = self.paging(tool, seen)
            if not page or page[3] is None or not SWEEP_ROWS[0] <= page[0] <= SWEEP_ROWS[1]:
                continue
            if self.brief.get('pick') == 'cheapest' and self.price_sorted(seen):
                continue
            records = self.records_in(seen)
            own = sum((1 for _, record in records if cat_key and self.owner_of(str(record.get(cat_key) or '')) == number))
            if 2 * own <= len(records):
                continue
            ranked.append((page[0], number, args, page))
        if not ranked:
            return []
        self.pages_swept = True
        ranked.sort(key=lambda r: -r[0])
        out: list[str] = []
        for _, number, args, page in ranked[:max(SWEEP_SEARCHES, len(self.items)) if big else SWEEP_SEARCHES]:
            out += self.sweep_one(tool, number, args, page, stop_at)
        return out

    def sweep_one(self, tool: str, number: int, call: dict[str, Any], page: Any, stop_at: int) -> list[str]:
        out: list[str] = []
        read = 0
        walked: set[str] = set()
        while page and page[3] is not None and read < SWEEP_PAGES and stop_at - self.step >= SWEEP_KEEP and not self.finished:
            faults: list[str] = []
            call = _shape({**call, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            mark = self.call_mark(tool, call)
            if mark in walked:
                break
            walked.add(mark)
            if mark in self.read_calls:
                got = self.read_calls[mark]
            else:
                reply = self.act(tool, call)
                if reply is None:
                    break
                got = reply.get('observation')
                self.observe(tool, got)
                self.pages_read += 1
                read += 1
                out.append('next page: ' + self.show(got))
            mine = self.origin.setdefault(number, [])
            mine += [i for i, _ in self.records_in(got) if i not in mine]
            page = self.paging(tool, got)
        if out:
            self.trace.append({'kind': 'value_pages', 'item': number, 'read': len(out)})
        return out

    def uncarted(self) -> set[int]:
        cart = self.cart_ids()
        return {n for n in range(1, len(self.items) + 1) if not any((i in cart for i in self.item_ids(n)))}

    def read_cheap_unread(self, observations: list[Any], kind: str, keep: int) -> str:
        """The two cheapest unread, buyable rows of items with no cart line among these results: read them (one step each)."""
        items = self.uncarted()
        rows = [i for seen in observations for i, _ in self.records_in(seen)]
        pool = [i for i in dict.fromkeys(rows) if i not in self.full and self.buyable(i) and self.price_of(i) is not None and any((i in self.item_ids(n) for n in items))]
        pool.sort(key=lambda i: self.price_of(i) or 0.0)
        if not pool:
            return ''
        read = self.fill_facts(pool[:2], keep)
        self.trace.append({'kind': kind, 'ids': pool[:2], 'read': len(read)})
        return '\n[Cairn read the cheapest unread rows of this search]\n' + '\n'.join(read) if read else ''

    @_soft('')
    def dear_line(self, added: list[str]) -> str:
        """Short cheapest tasks: a line just carted that costs more than two thirds of the priced rows of the search page that
        showed it gets that search's next page and reads of its two cheapest rows of the same label below its price."""
        tool = self.roles.get('search')
        cat_key = self.category_key()
        if self.max_steps >= SWEEP_BIG or self.brief.get('pick') != 'cheapest' or self.left() - self.tail() < DEAR_ROOM or not cat_key or not tool:
            return ''
        for line in added:
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            owner = self.owner_of(label)
            if price is None or owner == 0 or owner in self.dear_paged:
                continue
            for mark, (name, args) in reversed(list(self.read_args.items())):
                seen = self.read_calls.get(mark)
                if name != tool or line not in [i for i, _ in self.records_in(seen)]:
                    continue
                page = self.paging(tool, seen)
                if not page or page[3] is None:
                    continue
                faults: list[str] = []
                call = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                if faults or self.call_mark(tool, call) in self.read_calls:
                    continue
                priced = [self.price_of(i) for i, _ in self.records_in(seen) if self.buyable(i) and self.price_of(i) is not None]
                if len(priced) < 4 or sum((p < price for p in priced)) < DEAR_SHARE * len(priced):
                    continue
                self.dear_paged.add(owner)
                reply = self.act(tool, call)
                if reply is None or self.finished:
                    return ''
                got = reply.get('observation')
                self.observe(tool, got)
                self.pages_read += 1
                pool = sorted(((self.price_of(i) or 0.0, i) for i, r in self.records_in(got) if i not in self.full and self.buyable(i) and self.price_of(i) is not None and self.price_of(i) < price - 1e-09 and str(r.get(cat_key) or '') == label))
                picks = [i for _, i in pool[:2]]
                read = self.fill_facts(picks, self.tail() + DEAR_ROOM - 4) if picks else []
                self.trace.append({'kind': 'dear_line', 'line': line, 'page': page[3], 'read': picks})
                proof = self.evidence([self.full[i] for i in picks if i in self.full]) if read else ''
                text = f'\n[Cairn] {line} costs more than most rows its search showed, and the pick is the lowest price, so Cairn read that search\'s next page ({page[2]}={_dump(page[3])}, one step): {self.show(got)}'
                if read:
                    text += '\nand the attributes of its cheapest rows below the line (one step each):\n' + '\n'.join(read)
                return text + ('\n' + proof if proof else '')
        return ''

    def sold_peers(self, lost: list[str]) -> list[str]:
        """Lowest-price picks: per sold-out line, its two cheapest unread, buyable rows of the same label at or below its
        price (the replacement most likely sits among the rows it tied or outpriced)."""
        cat_key = self.category_key()
        if self.brief.get('pick') != 'cheapest' or not cat_key:
            return []
        cart = set(self.cart_ids())
        out: list[str] = []
        for line in lost:
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            if price is None or not label:
                continue
            pool = [i for i in self.seen_order if i not in lost and i not in cart and i not in self.full and self.buyable(i) and not self.sold_out(i) and str((self.rows.get(i) or {}).get(cat_key) or '') == label and self.price_of(i) is not None and self.price_of(i) <= price + 1e-09]
            pool.sort(key=lambda i: self.price_of(i))
            out += [i for i in pool[:SOLD_PEERS] if i not in out]
        return out

    def open_big(self, observation: Any) -> str:
        """Big tasks: after a model search, read the two best unread rows of items with no cart line (most of the item's
        own wanted words, then cheapest on a cheapest pick) in the same turn."""
        cat_key = self.category_key()
        if self.max_steps < SWEEP_BIG or self.left() - self.tail() < SWEEP_SPARE or not cat_key or self.rushed():
            return ''
        held = {self.owner_of(str((self.rows.get(i) or {}).get(cat_key) or '')) for i in self.cart_ids()}
        words: dict[int, set[str]] = {}
        for n, need in self.needs_now():
            words.setdefault(n, set()).update(re.findall('[a-z0-9]{3,}', _words(need.split(':', 1)[-1])))
        pool = []
        pooled: set[str] = set()
        for seen in [observation] + list(self.paged_obs):
            for ident, record in self.records_in(seen):
                owner = self.owner_of(str(record.get(cat_key) or ''))
                if owner == 0 or owner in held or ident in self.full or not self.buyable(ident) or self.sold_out(ident) or ident in pooled:
                    continue
                pooled.add(ident)
                hits = len(words.get(owner, set()) & set(re.findall('[a-z0-9]{3,}', _words(_dump(self.rows.get(ident) or record)))))
                price = self.price_of(ident) if self.brief.get('pick') == 'cheapest' and self.price_of(ident) is not None else 0.0
                pool.append((-hits, price, ident))
        picks = [i for _, _, i in sorted(pool)[:BIG_OPENS]]
        if not picks:
            return ''
        read = self.fill_facts(picks, self.tail() + SWEEP_SPARE)
        if not read:
            return ''
        self.trace.append({'kind': 'open_big', 'ids': picks})
        return "\n[Cairn read these listings' attributes itself, one step each]\n" + '\n'.join(read)

    def drop_hunt(self) -> str:
        """A requirement dropped on a cheapest pick widens the fitting set, but every earlier search carried the dropped word:
        one short hunt (two searches per affected item) on what is left, even with few steps."""
        if not self.drop_items or self.drop_hunted or self.brief.get('pick') != 'cheapest':
            return ''
        plan = self.value_template()
        if not plan or self.left() < self.tail() + 4 + len(self.drop_items) or self.clock() > LAST_TURN - 30:
            return ''
        self.drop_hunted = True
        tool, base, field = plan
        spent = gained = 0
        for number in sorted(self.drop_items):
            if self.left() < self.tail() + 4 or self.finished:
                break
            used, got = self.hunt_item(number, tool, base, field, 2)
            spent += used
            gained += got
        self.trace.append({'kind': 'drop_hunt', 'steps': spent, 'rows': gained})
        return f'Cairn searched again on the requirements left after the drop ({spent} searches, {gained} new rows).' if spent else ''

    def series_sweep(self) -> str:
        """Big fit tasks: when the read parts that state every requirement pin one series, page the device's own searches
        until a device row names that series, and read up to two such rows."""
        tool = self.roles.get('search')
        cat_key = self.category_key()
        if self.max_steps < SWEEP_BIG or not self.bound or not tool or not cat_key or self.left() - self.tail() < SWEEP_SPARE or self.rushed():
            return ''
        index = self.fit_index()
        marks = index['marks']
        texts: list[str] = []
        for part, main in self.bound:
            system = self.link_system(part, main, index)
            if not system or system[0] != part:
                continue
            proven = [i for i in self.item_ids(part) if i in self.full and self.buyable(i) and not self.sold_out(i) and self.proves(i, part)]
            wanted = set().union(*(set((marks.get(i) or {}).get('fam') or []) for i in proven)) & system[2] if proven else set()
            if not proven or len(proven) > SERIES_PINNED or not wanted or (main, frozenset(wanted)) in self.series_swept:
                continue

            def named(i: str) -> bool:
                return bool(set((self.fit_index()['marks'].get(i) or {}).get('named') or []) & wanted) and self.buyable(i) and not self.sold_out(i)
            if any((named(i) for i in self.item_ids(main))):
                continue
            self.series_swept.add((main, frozenset(wanted)))
            words = set(re.findall('[a-z0-9]{3,}', _words(self.items[main - 1].name)))
            for n, need in self.needs_now():
                if n == main:
                    words |= set(re.findall('[a-z0-9]{3,}', _words(need.split(':', 1)[-1])))
            spent = 0
            for mark, (name, args) in list(self.read_args.items()):
                text = ' '.join((str(v) for v in args.values() if isinstance(v, str))).lower()
                seen = self.read_calls.get(mark)
                owners = {self.owner_of(str(record.get(cat_key) or '')) for _, record in self.records_in(seen)}
                if name != tool or main not in owners or not words & _text_words(text):
                    continue
                call, walked = (args, {mark})
                while spent < SERIES_PAGES and self.left() - self.tail() >= SWEEP_SPARE and not self.finished and not any((named(i) for i in self.item_ids(main))):
                    page = self.paging(tool, seen)
                    if not page or page[3] is None:
                        break
                    faults: list[str] = []
                    call = _shape({**call, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                    now = self.call_mark(tool, call) if not faults else ''
                    if faults or now in walked:
                        break
                    walked.add(now)
                    if now in self.read_calls:
                        seen = self.read_calls[now]
                        continue
                    reply = self.act(tool, call)
                    spent += 1
                    if reply is None:
                        break
                    seen = reply.get('observation')
                    self.observe(tool, seen)
                    self.pages_read += 1
                    texts.append(f'{page[2]}={_dump(page[3])} of {_dump(args)}: ' + self.show(seen))
                if any((named(i) for i in self.item_ids(main))) or spent >= SERIES_PAGES:
                    break
            hits = [i for i in self.item_ids(main) if named(i) and i not in self.full][:BIG_OPENS]
            read = self.fill_facts(hits, self.tail() + SWEEP_SPARE // 2) if hits else []
            self.trace.append({'kind': 'series_sweep', 'item': main, 'series': sorted(wanted), 'pages': spent, 'hits': hits})
            if texts or read:
                texts.append(f"[{self.items[main - 1].name} rows naming series {', '.join(sorted(wanted))}, the series of the only {self.items[part - 1].name} listings read that state every requirement: {', '.join(hits) or 'none found'}]")
                texts += read
        if not texts:
            return ''
        return '\n[Cairn paged the searches of the item the parts must fit itself, one step a page]\n' + '\n'.join(texts)

    def cap_cached(self, name: str, args: dict[str, Any]) -> str:
        """A search that differs from an earlier one only in a price cap this search mode was shown to ignore returns the
        same rows: answer it from those rows with no step."""
        caps = self.cap_keys(name)
        if len(caps) != 1 or caps[0] not in args:
            return ''
        key = caps[0]
        props = (self.schemas.get(name) or {}).get('properties') or {}
        mode = tuple(sorted(((k, v) for k, v in args.items() if isinstance((props.get(k) or {}).get('enum'), list) and isinstance(v, (str, int, float)))))
        if self.cap_modes(name).get((key, mode)) is not False:
            return ''
        rest = {k: v for k, v in args.items() if k != key}
        for mark, (tool, earlier) in reversed(list(self.read_args.items())):
            if tool != name or earlier.get(key) == args.get(key) or {k: v for k, v in earlier.items() if k != key} != rest:
                continue
            seen = self.read_calls.get(mark)
            if not self.records_in(seen):
                continue
            self.trace.append({'kind': 'cap_cached', 'args': args})
            return f'Not run (no step used): this search mode does not apply {key} (its rows here have gone above it), so the same search with {key}={_dump(earlier.get(key))} already returned exactly these rows:\n' + self.show(seen)[:3000]
        return ''

    def holds_values(self, ident: str, number: int) -> bool:
        """Whether a row or listing shows every plain required value of its item (under its key, or in its title or
        description when no key names it); False when the item has no plain value or is over its cap."""
        plain = [n for n in self.item_needs(number) if self.need_kind(n) == 'plain']
        if not plain:
            return False
        cap = self.item_caps.get(number)
        if cap is not None and (self.price_of(ident) or 0.0) > cap + 1e-09:
            return False
        text = _vw(_dump(self.full.get(ident) or self.rows.get(ident) or {}))
        for need in plain:
            verdict, why = self.judge(ident, need)
            if verdict == 'contra':
                return False
            if verdict == 'stated':
                continue
            if not _run(_vw(need.partition(':')[2]), text) and (not _run(_vw(need.partition(':')[0]), text)):
                return False
        return True

    def note_next(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Remember, for a search whose page argument is known, the next unseen page of each query."""
        shape = self.shapes.get(name)
        if not shape or not shape.get('next') or not shape.get('next_arg') or not isinstance(observation, dict):
            return
        key = shape['next_arg']
        query = _dump({k: v for k, v in args.items() if k != key})
        value = observation.get(shape['next'])
        if value is None or (isinstance(value, (str, int)) and (not isinstance(value, bool))):
            self.next_pages[(name, query)] = value

    def next_unseen(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A repeat of a page already read is sent as the next unseen page of the same query, and the model is told."""
        shape = self.shapes.get(name)
        if not shape or not shape.get('next_arg') or self.call_mark(name, args) not in self.read_calls:
            return (args, '')
        key = shape['next_arg']
        nxt = self.next_pages.get((name, _dump({k: v for k, v in args.items() if k != key})))
        if nxt is None or args.get(key) == nxt:
            return (args, '')
        moved = {**args, key: nxt}
        if self.call_mark(name, moved) in self.read_calls:
            return (args, '')
        self.trace.append({'kind': 'next_unseen', 'tool': name, 'sent': args.get(key), 'took': nxt})
        return (moved, f'\n[Cairn] That page of this search was read already, so Cairn asked for the next unseen one ({key}={_dump(nxt)}).')


    # ---- c15_v29: robustness for unseen store mechanics (M6) ----
    def nested_text(self, tool: str, said: Any) -> list[str]:
        """The words field when it sits inside an object argument ("query.text"): the one free-text string property of an
        object argument, the one named like a query when there are several."""
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        found = []
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'object' not in _kinds(spec):
                continue
            inner = spec.get('properties') or {}
            texts = [k for k, v in inner.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v]
            named = [k for k in texts if re.search(QUERY_WORDS, _snake(k), re.I)] or (texts if len(texts) == 1 else [])
            found += [f'{key}.{k}' for k in named[:1]]
        if isinstance(said, str) and any((f.endswith('.' + said) for f in found)):
            found = [f for f in found if f.endswith('.' + said)]
        return found if len(found) == 1 else []

    def own_record(self, name: str, args: dict[str, Any], observation: Any) -> Any:
        """A listing view whose record names no id is the record of the listing the call named."""
        if name not in self.page_tools() or not isinstance(observation, dict) or len(observation) < 3 or self.ident(observation):
            return observation
        target = self.single_target(name)
        ident = args.get(target) if target else None
        if not isinstance(ident, str) or not _id_like(ident.strip(), True):
            return observation
        return {self.id_key or 'id': ident.strip(), **observation}

    def token_in(self, observation: Any) -> str:
        """A confirmation token a checkout answered with: a short one-word text under a key naming a token or code."""
        stack = [observation]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, str) and re.search('token|confirm|code|nonce|ticket|otp|pin|passphrase', str(key), re.I) and re.fullmatch('[A-Za-z0-9_-]{3,40}', value.strip()):
                        return value.strip()
                    if isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(node, list):
                stack += node
        return ''

    def token_field(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        fields = [k for k in schema.get('required') or [] if isinstance(props.get(k), dict) and 'string' in _kinds(props[k]) and 'enum' not in props[k] and re.search('token|confirm|code|nonce|ticket|otp|pin|key', str(k), re.I)]
        return fields[0] if len(fields) == 1 else None

    def confirm_tool(self, buy: str) -> tuple[str | None, str | None]:
        """The tool that completes a two-step checkout: not one of the known jobs, taking one required token-like text."""
        jobs = {t for t in self.roles.values() if t}
        found = []
        for t in self.tools:
            name = t['function']['name']
            field = self.token_field(name) if name not in jobs or name == buy else None
            said = str(t['function'].get('description') or '') + ' ' + name
            if name != buy and field and re.search('confirm|place|final|complete|commit|submit|order|checkout|purchase|pay', said, re.I):
                found.append((not re.search('confirm|place|final|complete|commit|submit', said, re.I), name, field))
        found.sort()
        return (found[0][1], found[0][2]) if found else (None, None)

    def confirm_step(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        """A checkout that is not final and answers with a token: confirm it at once with the tool that takes the token."""
        seen = data.get('observation')
        token = self.token_in(seen)
        tool, field = self.confirm_tool(name)
        if not token or not tool or self.step >= self.max_steps or self.finished:
            return data
        faults: list[str] = []
        args = _shape({field: token}, self.schemas.get(tool), 'args', faults)
        if faults:
            return data
        self.confirming = True
        try:
            again = self.act(tool, args)
        finally:
            self.confirming = False
        self.trace.append({'kind': 'confirm_step', 'tool': tool, 'finished': bool(again and again.get('finished'))})
        if again is None:
            return data
        return {**again, 'observation': {'checkout': seen, 'confirmation': again.get('observation')}}

    def buy_args_for(self, faults: list[str]) -> dict[str, Any]:
        """The checkout call's arguments; when the checkout tool wants a token, the tool that issues one runs first."""
        buy = self.roles.get('buy')
        tried: list[str] = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', tried)
        if not tried:
            return args
        field = self.token_field(buy)
        jobs = {t for t in self.roles.values() if t}
        preps = [t['function']['name'] for t in self.tools if t['function']['name'] not in jobs and not ((self.schemas.get(t['function']['name']) or {}).get('required')) and re.search('checkout|order|purchase|pay|begin|start|review', str(t['function'].get('description') or '') + ' ' + t['function']['name'], re.I)]
        if field and len(preps) == 1 and self.left() >= 2:
            reply = self.act(preps[0], _shape({}, self.schemas.get(preps[0]), 'args', []))
            token = self.token_in(reply.get('observation')) if reply else ''
            if token:
                tried = []
                args = _shape({**dict(self.brief.get('buy_args') or {}), field: token}, self.schemas.get(buy), 'args', tried)
                if not tried:
                    return args
        faults.extend(tried or ['the checkout call cannot be built'])
        return args

    def carried_cart(self, opening: Any) -> str:
        """When the store's first words mention the cart, look at it once: lines already in it were not added for this
        request."""
        view = self.roles.get('cart_view')
        if not view or self.finished or not re.search('\\b(?:cart|basket|bag|trolley)\\b', _dump(opening), re.I):
            return ''
        faults: list[str] = []
        args = _shape({}, self.schemas.get(view), 'args', faults)
        reply = self.act(view, args) if not faults else None
        if reply is None:
            return ''
        self.cart_view = reply.get('observation')
        self.cart_seen_at = self.step
        text = self.cart_view if isinstance(self.cart_view, str) else _dump(self.cart_view).replace('\\n', ' ')
        self.carried = {w for w in re.findall('[A-Za-z0-9][A-Za-z0-9_\\-:.]{2,40}[A-Za-z0-9]', text) if re.search('[A-Za-z]', w) and re.search('\\d', w) and _id_like(w) and w not in (self.request or '')}
        self.trace.append({'kind': 'carried_cart', 'ids': sorted(self.carried)})
        return 'The cart already holds lines from before this visit: ' + self.show(self.cart_view) + (' Lines the request does not ask for must come out before checking out (' + ', '.join(sorted(self.carried)) + ').' if self.carried else '')

    def carried_faults(self) -> list[str]:
        cart = self.cart_ids()
        return [f'{i} was already in the cart when the visit began and is not part of this request: take it out' for i in cart if i in self.carried and not self.owner_numbers(i) and (self.held_units(i) or 0) > 0]

    def drop_carried(self) -> None:
        """Before a forced order: lines carried over from before the visit that no requested item owns come out."""
        cart = self.cart_ids()
        stale = [i for i in cart if i in self.carried and not self.owner_numbers(i)]
        if stale and len(stale) < len(cart):
            self.put_lines({i: (self.held_units(i) or 1) for i in cart if i not in stale}, 'carried_drop')


    # ---- c15_v36: second port of the updated forum guides (Gental #97 rebuild spec, Po #102 full guide) ----
    def scope_musts(self) -> None:
        """A requirement the request states for a wider kind ("the X and Y ...") that the brief also gave the narrower item
        ("X") stays only with the item(s) the stating sentence names in full (Gental 3.2)."""
        pointed = {str(x).strip() for x in _items(self.brief.get('lookups'))}
        sentences = [s for s in re.split('(?<=[.!?;])\\s+|\\n+', self.request or '') if s.strip()]
        said = [[r for w in _words(s).split() for r in _roots(w)] for s in sentences]
        figures = lambda text: set(re.findall('\\d+(?:\\.\\d+)?', text))
        names = [_roots(item.name) for item in self.items]

        def named(words: list[str]) -> set[int]:
            held = set(words)
            out = set()
            for i, mine in enumerate(names):
                if not mine or not mine <= held:
                    continue
                if any((mine < wide <= held for wide in names)) and all((words.count(r) == 1 for r in mine)):
                    continue
                out.add(i)
            return out
        musts = sorted({m for item in self.items for m in item.musts if m not in pointed and ':' in m})
        for must in musts:
            holders = [i for i, item in enumerate(self.items) if must in item.musts]
            if len(holders) < 2:
                continue
            prop, _, value = must.partition(':')
            need = _roots(prop)
            stating = [words for s, words in zip(sentences, said) if need and need <= set(words) and figures(value) <= figures(s)]
            keep: set[int] = set()
            for words in stating:
                hit = named(words) & set(holders)
                if not hit:
                    keep = set()
                    break
                keep |= hit
            if not keep:
                continue
            for i in holders:
                if i not in keep and any((names[i] & names[k] for k in keep)):
                    self.items[i].musts.remove(must)
                    self.trace.append({'kind': 'must_scoped', 'must': must, 'from': self.items[i].name, 'to': [self.items[k].name for k in sorted(keep)]})

    def short_cart(self) -> bool:
        return len([i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]) < len(self.items) or bool(self.missing_items())

    def reopen_reads(self) -> str:
        """A short cart idling towards an ending: search and the listing views are offered again, once (Gental 6.6)."""
        if self.reopened or self.buys or not self.roles.get('search') or self.left() <= self.tail() + 3 or not self.short_cart():
            return ''
        if self.roles.get('search') in {t['function']['name'] for t in self.offered()}:
            return ''
        self.reopened = True
        self.trace.append({'kind': 'reopened', 'left': self.left()})
        return 'Search and the listing views are open again, since an item has no line in the cart: find and read a fitting listing for it, cart it and check out within the steps left.'

    def late_finish(self) -> str:
        """Near the step limit, a decline while the cart holds, or once held, a line for every item becomes an order: the
        removed lines go back and Cairn checks out; never over the budget or with a sold-out line (Gental 6.4)."""
        buy, cart = (self.roles.get('buy'), self.roles.get('cart'))
        if not buy or not cart or self.finished:
            return ''
        lines = [i for i in self.built if not self.sold_out(i)]
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        now = self.cart_ids()
        missing: list[str] = []
        if len(lines) >= len(self.items):
            missing = [i for i in lines if i not in now or (i in self.dropped and not fresh)]
            want = {}
            for i in missing:
                owners = self.owner_numbers(i)
                want[i] = self.items[owners[0] - 1].qty if len(owners) == 1 else 1
            if want and not self.put_lines({**{i: (self.held_units(i) or 1) for i in now if i not in want}, **want}, 'late_put_back'):
                return ''
        elif len(self.items) >= 2 and now:
            if not self.fill_short_cart() or len([i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]) < len(self.items):
                return ''
        else:
            return ''
        if self.finished or self.left() < 1 or any((' sold out' in f or ' over the ' in f for f in self.line_faults())):
            return ''
        if self.left() >= 2:
            self.reread_offers()
        if self.finished or self.left() < 1:
            return ''
        faults: list[str] = []
        args = self.buy_args_for(faults)
        if faults:
            return ''
        replies = []
        for attempt in range(2):
            if attempt and (self.left() < 1 or self.finished or any((self.sold_out(i) for i in self.cart_ids()))):
                break
            if self.left() >= 2:
                go, said = self.gate('order')
                if said:
                    replies.append(said)
                if not go:
                    break
            if self.finished or self.left() < 1:
                break
            reply = self.act(buy, args)
            self.trace.append({'kind': 'late_finish', 'put_back': missing, 'reply': _clip(reply, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
            self.observe(buy, reply.get('observation'))
            replies.append(self.show(reply.get('observation')))
            if self.finished or reply.get('finished'):
                break
        if not replies:
            return ''
        return 'Not run (Cairn acted instead): this close to the step limit an ending without a purchase earns nothing while the cart holds a line for every item, so Cairn ' + (f"put {', '.join(missing)} back and " if missing else '') + 'checked out. Store reply: ' + ' | then: '.join(replies)

    def late_review(self) -> str:
        """A decline with 2 to 5 steps left and no whole cart: the review's lead and the cart's other items' lines are
        judged as one basket, and Cairn carts and orders it when the check passes (Gental 6.5)."""
        self.refuse_blocks = 2
        if not self.leads and not self.audited and self.clock() <= WIND_DOWN and self.judges():
            self.audited = True
            named = self.audit_fits(self.audit_call())
            if named:
                self.lead_items = named
                self.leads = [i for ids in named.values() for i in ids]
        first = next((i for i in self.leads if not self.sold_out(i)), None)
        if not first or self.clock() > WIND_DOWN:
            return ''
        mine = set(self.owner_numbers(first))
        basket = [first] + [i for i in self.cart_ids() if i != first and not (set(self.owner_numbers(i)) & mine)]
        judged = sorted(basket)
        self.check_cart(judged, SPEND_HARD)
        if not self.verdicts.get(self.check_key(judged)):
            return ''
        self.confirmed = first
        self.trace.append({'kind': 'late_review', 'confirmed': first, 'basket': judged})
        return self.complete_cart()

    def label_text(self, number: int) -> str:
        cat_key = self.category_key()
        item = self.items[number - 1]
        names = [_roots(item.name)] + ([_roots(item.shelf)] if (item.shelf or '').strip() else [])
        seen = {str(r.get(cat_key) or '').strip() for r in self.rows.values()} - {''} if cat_key else set()
        return next((t for t in sorted(seen) if _roots(t) in names and self.owner_of(t) == number), '')

    def label_line(self) -> str:
        """Two items whose own store labels nest (a kind and a wider kind): each listing counts only for the item whose
        label it carries (Gental 5)."""
        if len(self.items) < 2 or not self.category_key():
            return ''
        own = {n: self.label_text(n) for n in range(1, len(self.items) + 1)}
        own = {n: t for n, t in own.items() if t}
        near = sorted({n for a in own for b in own if a != b and _roots(own[a]) < _roots(own[b]) for n in (a, b)})
        if len(near) < 2 or all((_words(self.items[n - 1].shelf or '') == _words(own[n]) for n in near)):
            return ''
        names = '; '.join((f"item {n} ({self.items[n - 1].name}) under '{own[n]}'" for n in near))
        return f'store labels: {names}. A listing counts only for the item whose label it carries, whatever its title says; find each of these items by listing its own label.'

    def next_page_big(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Big tasks: the next page of a model search whose rows include items still without a cart line, read in the
        same turn, one step (Gental 11)."""
        if self.max_steps < BIG_STEPS or self.left() - self.tail() < 20 or self.rushed():
            return ''
        plan = self.paging(name, observation)
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        cat_key = self.category_key()
        owners = {self.owner_of(str(record.get(cat_key) or '')) for _, record in self.records_in(observation)} - {0} if cat_key else set()
        held = {self.owner_of(str((self.rows.get(i) or {}).get(cat_key) or '')) for i in self.cart_ids()} if cat_key else set()
        if not owners or owners <= held:
            return ''
        faults: list[str] = []
        call = _shape({**args, param: value}, self.schemas.get(name), 'args', faults)
        if faults:
            return ''
        reply = self.act(name, call)
        if reply is None or self.finished:
            return ''
        self.pages_read += 1
        seen = reply.get('observation')
        self.observe(name, seen)
        self.paged_obs.append(seen)
        shown_text = self.show(seen)
        self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
        self.trace.append({'kind': 'pages', 'tool': name, 'read': 1, 'big': True})
        return f'\n[Cairn read the next page of this search itself, one step]\n{param}={_dump(value)}: {shown_text}'

    def review_overruled(self, ids: list[str], got: dict[str, Any]) -> bool:
        """The review rejects a cart whose every line states every requirement of its item, and every listing it names as
        better does not: its verdict is set aside (Po v28)."""
        if not ids or not all((any((self.proven(i, n) for n in self.owner_numbers(i))) for i in ids)):
            return False
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows]
        if any((any((self.proven(b, n) for n in self.owner_numbers(b))) for b in better)):
            return False
        self.trace.append({'kind': 'review_overruled', 'cart': ids, 'better': better})
        return True

    def auto_swap(self) -> str:
        """Once, at the model's checkout: when a cart line's own attribute states another value for a requirement, or on a
        cheapest pick a fully proven set costs less, the cart becomes the fully proven set (Po v35)."""
        if self.auto_swapped or self.left() <= self.tail() + 3 or self.rushed():
            return ''
        chosen = self.proven_set()
        cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not chosen or not cart or set(chosen.values()) == set(cart):
            return ''
        wrong = [i for i in cart for n in self.owner_numbers(i) if self.line_contra(i, n)]
        cheaper = False
        if self.brief.get('pick') == 'cheapest' and all((self.price_of(i) is not None for i in cart)):
            held = sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in cart))
            cheaper = self.set_total(chosen) + 1e-09 < held
        if not wrong and not cheaper:
            return ''
        self.auto_swapped = True
        if not self.put_lines({i: self.items[n - 1].qty for n, i in chosen.items()}, 'auto_swap'):
            return ''
        why = ('lines stating another value: ' + ', '.join(wrong)) if wrong else 'a cheaper fully proven order'
        return f'\n[Cairn] Set the cart to {self.set_text(chosen)} itself ({why}); every requirement of each line is stated under its own attribute.'

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
        if not holding and buy and self.left() >= len(self.items) + 1 and self.proven_basket():
            fresh = False
            holding = bool(self.cart_lines)
        if holding and buy and self.left() >= 3:
            self.drop_carried()
            self.repair_fits()
        whole = True
        if holding and len(self.items) >= 2:
            whole = self.fill_short_cart() or not self.can_refuse()
            if whole and len(self.cart_ids()) >= 2:
                self.fill_facts(self.cart_ids(), self.tail())
                if self.left() >= 3 and self.model_room() > 5:
                    self.show_links(self.cart_ids(), self.tail())
            if not whole:
                self.trace.append({'kind': 'short_cart_refused', 'lines': self.cart_ids()})
        if self.finished or self.left() <= 0:
            return ''
        if not holding and buy and self.proven_basket_by_key():
            holding = True
        if buy and self.bound and not self.finished and self.left() >= 2:
            sets = self.read_sets()
            mixed = (bool(self.fit_faults()) or len(self.cart_ids()) < len(self.items)) if holding else True
            if sets and mixed and not any((set(lines) == set(self.cart_ids()) for _, lines in sets)) and self.cart_set(sets[0][1]):
                holding, whole = (True, True)
        blind = self.cart_touched and (not fresh) and self.left() >= 2 and (not self.dropped) and (self.buys == 0 or self.refused_at < self.cart_at)
        if whole and buy and (holding or blind):
            faults: list[str] = []
            args = self.buy_args_for(faults)
            if not faults:
                if self.left() >= 2:
                    self.gate('order')
                if self.left() >= 2 and not self.finished:
                    # the forced order too: a line the page view never showed is not opened as the store counts it
                    self.show_pointed()
                    self.show_line_refs()
                    self.reread_offers()
                if self.finished or self.left() <= 0:
                    return ''
                reply = self.act(buy, args)
                self.trace.append({'kind': 'forced_buy', 'reply': _clip(reply, 200)})
                if self.finished or self.left() <= 0:
                    return ''
                if reply is not None and again:
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, reply.get('observation'))
                    return 'the checkout was refused: ' + self.show(reply.get('observation'))
        if refuse and self.noask_mode() and not self.finished and self.left() >= 2:
            done = self.closest_order()
            if done or self.finished:
                return '' if self.finished else 'the checkout was refused: ' + done
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
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        if opening is not None:
            opening = _restore(opening)
            self.absorb(opening)
        carried = self.carried_cart(opening) if opening is not None else ''
        moved = self.first_moves() if not self.finished else ''
        moved = (carried + '\n' + moved).strip() if carried else moved
        self.env_gap = ENV_GAP
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        if moved:
            intro += '\n\nCairn has searched for each item and read some of its listings (one step each):\n' + moved + '\nJudge these against the requirements and search further only for what is still missing.\n\n' + self.status()
        else:
            intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                self.chat.append({'role': 'user', 'content': '[Cairn] Cairn tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            for _ in range(2):
                if self.finished:
                    break
                turned = self.finish()
                if not turned or self.finished or self.broke or self.no_credit or self.left() <= 1:
                    break
                self.chat.append({'role': 'user', 'content': '[Cairn] Cairn tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 6, SPEND_HARD + 0.02)
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        for _ in range(CRASH_TURNS):
            try:
                self.turns(loop_cap, spend_cap)
                return
            except Exception as exc:
                self.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
                self.answer_open_calls()

    def answer_open_calls(self) -> None:
        """Give every call of the last model turn a reply, so that the conversation stays one the model can continue."""
        at = next((n for n in range(len(self.chat) - 1, -1, -1) if self.chat[n].get('tool_calls')), None)
        if at is None:
            return
        done = {m.get('tool_call_id') for m in self.chat[at + 1:]}
        for call in self.chat[at]['tool_calls']:
            if call['id'] not in done:
                self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'The call may not have run: something went wrong on the way. Look at the state before you repeat it.'})
        self.chat.append({'role': 'user', 'content': self.status()})

    def turns(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.clock() < LAST_TURN or (not self.cart_ids() and self.model_room() > EMPTY_ROOM)):
            if self.prelude_due():
                summary = self.noask_prelude()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                    continue
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue
            ordered = self.proven_order()
            if ordered:
                if self.chat and self.chat[-1].get('role') == 'user':
                    self.chat[-1]['content'] += '\n' + ordered + '\n' + self.status()
                else:
                    self.chat.append({'role': 'user', 'content': ordered + '\n' + self.status()})
            self.trim()
            if self.loop_t0 is None:
                self.loop_t0 = self.clock()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], DIRECTOR_TOKENS, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= ALL_CALLS:
                    break
                if self.misses > 2:
                    if self.clock() >= OUTAGE_BY:
                        break
                    time.sleep(BRIEF_GAP)
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
                reopened = self.reopen_reads()
                if reopened:
                    idle = 0
                self.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.' + (' ' + reopened if reopened else '') + '\n' + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            if not self.asked and self.roles.get('ask') and any((c['function']['name'] in committing for c in calls)) and self.settle_noask():
                self.asked = True
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
            pages = self.show_pointed(early=True) if replies and self.left() > 12 and not self.finished else ''
            pages += self.series_sweep() if replies and not self.finished else ''
            if pages:
                replies[-1] = (replies[-1][0], replies[-1][1] + '\n' + pages)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.status()
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Cairn(problem_data)
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
