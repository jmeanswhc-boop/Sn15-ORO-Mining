"""flint: a shopping agent for ORO SN15 store worlds (sn15-d6 builder line), built on the public code of the race-17
top agent Cairn v2 and kept as our own file: every docstring and comment here is ours, and our fixes are listed below.

How an episode runs. A brief turns the shopper's request and the store's tool list into items, firm requirements,
roles for each tool and the process steps the request asks for. The interview then asks the shopper about the
properties the store's listings vary on: its first message is short and later ones grow into lists while the shopper
answers them, and a reply that asks for one question at a time turns lists off for the rest of the episode. A light
model walks the store with the world's own tools while code keeps the ledger of rows, pages, cart lines and
requirements, reads results in whatever shape the world returns them, and spends steps of its own where the model is
weak: more pages and searches before giving up on an item, a walk down the price range when the pick must be the
cheapest, every bought line's page opened before a checkout. A heavier model reviews the cart before each checkout and
the evidence before any ending without a purchase. A clock keeps the episode inside the validator's time limit.

Our fixes over the base: a one-item question first probes the shopper with a short list whenever another message would
remain and grows to the full list only after the shopper answers a list (the base ramped from one property, which
left list-taking shoppers mostly unasked when only two messages were allowed); an unproven list never takes the last
message or the late question; every message names the properties of a single item (combined questions about several
items mostly drew "say that again" replies in race 17); promotional flags never move a listing in or out of stock and
negative stock flags count the right way round; the count of failed model turns starts over after a turn that
worked; a listing id sent as a whole number counts as that listing in the page and cart bookkeeping.
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
NO_NAME = ('none', 'null', 'na', 'unknown')
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
STOCK_NEGATIVE = '(?:sold ?out|soldout|out of stock|outofstock|unavailable|not available|not for sale|backordered|on back ?order|discontinued)'
STOCK_POSITIVE = '(?:in ?stock|for ?sale|buy|purchas|orderable|available for)'
PROMO_FLAG = '\\b(?:sale|deal|promo\\w*|discount\\w*|clearance|featured|sponsor\\w*|recommend\\w*|best|top|new|prime|free|ship\\w*|deliver\\w*|pickup|gift|bundle|limited|exclusive|popular|trending|verified|rated|choice)\\b'
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
LIST_STEPS = (1, 3)
LIST_STEPS_MIXED = (1, 3)
CAP_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
CAP_NOT = 'result|row|page|count|size|num|item|step'
CAP_PRICED = 'price|cost|budget|spend|money|amount'
CAP_LOW = '(?:^|_)(?:min|minimum|low|lowest|least|from|start|offset|floor)(?:_|$)'
EARLY_TRAILS = 3
CAP_CHANCE = 0.05
WALK_ROOM = 10
WALK_CALLS = 4
SHELF_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MODE = 'field|filter|facet|brows|categor|list|structured'
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_short: dict[int, ProxyClient] = {}
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Cairn, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun); a store\'s search can read attribute values and shop names as well as titles and may want every word, so a few required values plus the noun can list nearly the whole fitting set, and a model or part code may find nothing (one try tells). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Cairn knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Cairn\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Cairn carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Cairn\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Cairn adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category. A listing under the requested label is that kind of product even when its title or description names something else, so judge it on the requirements, price and stock.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value; when the requirement asks for what goes with that listing\'s value (its partner, its counterpart) and the page names what its value pairs with, the requirement is that other value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Cairn reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Price and stock can change silently: Cairn re-reads the cart\'s lines and looks at the cart when you call the checkout, so do not spend a turn on either.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"search_words": {"tool": the tool that searches listings by free words (when one tool lists a category and another takes words, the one that takes words), "field": its parameter that takes the words, "args": the fixed arguments that make it search by words (for a tool with a mode argument, that mode), else {}}.\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write \"not X\" only when the answer rejects X itself.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails, though a value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: its page not read, seen only as a search result row or by its attributes).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page; when the requirement asks for what goes with that value and the page names what its value pairs with, that other value; or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
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
VALUE_ROOM = 12
VALUE_KEEP = 8
LOOK_ALIKES = 2
LINK_SEARCHES = 3
LINK_ROWS = 3
LINK_FAMILIES = 12

def _codes(text: Any) -> list[str]:
    """Pulls device or model codes out of free text: a capital letter, up to five more letters, an optional space or
    hyphen, then either three to six digits or a short number followed by one to three x's acting as a series
    wildcard. Each hit is lowercased and its spaces or hyphens collapsed to one space, so 'KX-2040' and 'KX 2040'
    compare equal. A capitalised word followed by a year from the 1900s or 2000s (such as 'Autumn 2025') is skipped,
    since it names an edition or season rather than a model. Codes come back in reading order; None reads as empty
    text."""
    found = re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))
    return [re.sub('[\\s-]+', ' ', c).strip().lower() for c in found if not re.fullmatch('[A-Z][a-z]+ (?:19|20)\\d\\d', c)]

def _code_hit(pattern: str, code: str) -> bool:
    """Tests whether a code taken from an accessory's compatibility text matches a device code, after deleting spaces
    and hyphens from both. When the accessory's code ends in lowercase x's directly after a digit, the x's are
    wildcards: the device code must start with the part before them and have exactly the same total length ('rt40xx'
    accepts 'rt4071' but not 'rt40712'). Any other pair must be identical. Both inputs are expected in the lowercased
    form that _codes produces."""
    wide, plain = (re.sub('[\\s-]+', '', pattern), re.sub('[\\s-]+', '', code))
    stem = wide.rstrip('x')
    if stem != wide and stem[-1:].isdigit():
        return plain.startswith(stem) and len(plain) == len(wide)
    return wide == plain

def _fit_marks(text: Any) -> dict[str, Any]:
    """Collects the compatibility clues in a listing's text into a dict that the accessory/device pairing logic reads.
    'own' is a part number written inside brackets or parentheses (optionally after the word part), upper-cased;
    'codes' are the listing's own model numbers, taken after 'model', 'model no.', 'model number' or 'model #' and
    after alias phrases such as 'also known as'. 'works' and 'bars' come from 'works with', 'compatible with' or
    'fits' clauses not preceded by 'not' or 'never', split at 'except', 'excluding' or 'but not' into accepted and
    excluded codes. 'also' holds (part, codes) pairs from sentences stating that some other part also fits certain
    models (those sentences are kept out of 'works'), 'more' is a part number the text refers the reader to for
    details, and 'fam' and 'named' are the series sets from _series_marks. Each clause runs to the next full stop or
    semicolon."""
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
    """Finds series names in a record's text and returns two sorted, upper-cased lists. The first holds the
    letter-and-digit prefix of every hyphenated code (the 'KX3' in 'KX3-880'). The second holds standalone words of
    one to five letters plus one to three digits that remain once those full codes and any negative clause (except,
    excluding, but not, not for, not compatible with, does not fit or take, up to the next full stop or semicolon)
    have been cut out, so a series the text rules out is not reported. The second list is only a set of candidates;
    the pairing logic later keeps those that line up with codes found on other listings."""
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
    """Detects a store reply that carries nothing except a failure. Only dicts with one to three keys can qualify: the
    reply counts as failed when a key whose name mentions an error, failure, fault (not default), trouble (not
    troubleshooting), denial, refusal or invalidity has a truthy value, or when the dict's single value is a string
    of at most 40 characters without spaces, the look of a bare error code. Lists, strings and larger dicts are never
    treated as failures. The second rule also catches any one-key reply whose value is a single word."""
    if not isinstance(observation, dict) or not 0 < len(observation) <= 3:
        return False
    if any((value and re.search(FAIL_KEY, str(key), re.I) for key, value in observation.items())):
        return True
    only = next(iter(observation.values()))
    return len(observation) == 1 and isinstance(only, str) and 0 < len(only) <= 40 and ' ' not in only

def _plain(text: str) -> Any:
    """Converts a piece of text into the scalar a JSON reply would have held. 'true' and 'false' in any case become
    booleans, and 'none', 'null' or an empty string become None. A number, optionally led by a dollar, euro or pound
    sign and optionally grouped with thousands commas, becomes an int when written without a decimal point and a
    float otherwise. Numbers starting with a zero followed by another digit (postcodes, part numbers), negative
    numbers and everything else are returned as the stripped text."""
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
    """Turns the right-hand side of a name=value entry into the value stored in an attribute block. Numbers are
    converted as _plain does, but words _plain would turn into a boolean or None are kept as their literal text,
    because inside an attribute list 'false' or 'none' is a catalogue value, not a flag. Only an empty side yields
    None."""
    found = _plain(text)
    return text.strip() if isinstance(found, bool) or (found is None and text.strip()) else found

def _pair(text: str, mark: str) -> tuple[str, str] | None:
    """Splits text at the first occurrence of a separator such as ': ' or '=' into a (name, value) tuple, or returns
    None when the text does not begin with a usable field name. Text that ends with the separator's bare character (a
    trailing colon) gives the name with an empty value. A leading bullet or list number is removed from the name,
    which must then start with a letter, use only letters, digits and simple punctuation, stay under 40 characters
    and have at most five words; the value is stripped."""
    head, found, rest = text.partition(mark)
    if not found and text.endswith(mark.strip()):
        head, found, rest = (text[:-len(mark.strip())], mark, '')
    name = re.sub(LIST_BULLET, '', head).strip()
    if not found or not re.fullmatch(FIELD_NAME, name) or len(name.split()) > 5:
        return None
    return (name, rest.strip())

def _split_fields(text: str, between: str, mark: str) -> tuple[list[tuple[str, str]], int]:
    """Breaks text at the outer separator `between` into (name, value) fields, using `mark` between name and value, and
    also reports how many pieces carried their own name. A piece without a name is appended to the preceding field's
    value together with the separator, so values that contain the separator survive; if the very first piece has no
    name the whole text is rejected with an empty result. When splitting on newlines, a line made of a name and a
    trailing colon starts a heading, and the indented or bulleted lines after it are stored under that name, one per
    line, until an ordinary line appears."""
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
    """Tries to read a single text field as a set of listing attributes, returning a dict or None. Accepted layouts,
    tried in this order: a JSON object (with a lenient regex fallback that needs two or more key/value pairs when the
    JSON is malformed); name=value entries split at semicolons, pipes or newlines, where a piece without '=' extends
    the previous value and two names are required unless the whole text is one short entry; a run of two or more
    bullet lines each of the form 'name: value'; and 'name: value' parts separated by commas or semicolons. The last
    layout is refused when a name contains a full stop or a value runs past 40 characters, which stops sentences in
    descriptions from being taken as attributes. Texts under five characters are ignored."""
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
    """Parses a text that spells out one record's fields with an unknown pair of separators, returning a dict or None
    when the text does not look like fields. Outer separators (pipes, newlines, semicolons, tabs, slashes, bullets,
    commas) are tried in a fixed order and, for each, every name/value marker (colon, equals sign, arrow, dash); a
    combination counts only when at least three pieces are named, the named pieces are a large enough share of all
    pieces (70 percent with commas, 15 percent otherwise) and names are not mostly repeats. The first outer separator
    that yields any candidate wins, choosing the marker that names the largest number of pieces. In the record each
    name keeps its first value; a value that itself reads as an attribute block becomes a nested dict, identifier
    fields stay text and other values go through _plain."""
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
    """Reads a plain-text table into a list of row dicts. Blank lines and rule lines made only of pipes, dashes, plus
    signs, colons, equals signs or spaces are dropped and outer pipes trimmed; the first remaining line must split,
    at a pipe, tab, semicolon or comma, into three or more cells that are all valid field names, and every other line
    into exactly as many cells. Cells pass through _plain and are keyed by the header names. Returns None when fewer
    than two lines remain or no delimiter fits."""
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
    """Recovers structured data from a text reply, returning a dict, a list of dicts, or None when the text holds no
    records (anything under 12 characters is skipped). Text that starts like JSON is decoded and normalised with
    _restore; if decoding fails the other readings are tried. The text is then cut at blank lines and, failing that,
    at single lines, and the pieces become a list of records when two or more parse, at least four in five parse and
    every parsed record starts with the same field name. Otherwise a multi-line text that is not one record is tried
    as a table, and anything else as a single record. As fallbacks, short HTML-like markup is flattened (inline tags
    dropped, adjacent table cells joined as 'name: value', other tags turned into line breaks) and parsed again, and
    in multi-line text the first run of two or more spaces or the first tab on each line is read as the gap between
    name and value."""
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
    """Handles a text field inside a record that already has an id and fields of its own. The text becomes an attribute
    dict only when it is made wholly of short name=value pieces (at least two, names up to 40 characters without
    colons) separated by semicolons, pipes or newlines; anything else, such as prose that happens to contain an
    equals sign, is returned unchanged. If the pieces qualify but _attr_block still cannot read them, the original
    text is kept."""
    if len(text) < 5 or '=' not in text or not re.search('; | \\| |\\n', text):
        return text
    parts = [p for p in re.split(';\\s+|;$|\\s\\|\\s|\\n', text.strip()) if p.strip()]
    if len(parts) < 2 or not all((re.fullmatch('[^=:]{1,40}=[^=]{0,200}', p.strip()) for p in parts)):
        return text
    return _attr_block(text) or text

def _pairs_block(rows: list[Any], in_record: bool) -> dict[str, Any] | None:
    """Collapses a list that gives attributes one entry at a time into one name-to-value dict, or returns None when the
    list is something else. Three layouts qualify, each needing at least two entries: two-key objects that all share
    the same keys and whose first key looks like a name or label field (name, key, label, attr, prop, field, spec,
    title or k), with the first key read as the name and the second as the value; two-element [name, value] lists;
    and, only when `in_record` is set, short 'name: value' or 'name=value' strings with names of at most three words
    and values of at most 40 characters. A name over 60 characters or a value that is itself a list or dict
    disqualifies the list."""
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
    """Rewrites a columnar result into ordinary row objects. When one key holds a list of two or more strings that are
    valid field names and another key holds a non-empty list of lists each as long as that header, the header key is
    removed and every inner list is zipped with it into a dict. Only the first such header/rows pair is converted;
    any other node comes back untouched."""
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
    """Converts integer identifiers in a record to strings so listings are always keyed by text. Only records holding at
    least one string with a space in it (so they look like real listings with a title) are changed, and only
    non-boolean integers of magnitude 100 or more under keys that look like ids. When a tool parameter is typed as an
    integer, _shape turns the text back into a number at call time."""
    if not any((isinstance(v, str) and ' ' in v for v in record.values())):
        return record
    return {k: str(v) if isinstance(v, int) and not isinstance(v, bool) and re.search(ID_FIELD, k, re.I) and abs(v) >= 100 else v for k, v in record.items()}

def _keyed_rows(node: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Turns a dict whose keys are listing codes into a row list. It applies only when there are two or more entries,
    every key contains a digit and no space, every value is a dict of three or more fields, and the values share at
    least three field names. Each row gains an 'id' field set to its key unless it already has a field named like an
    identifier; when the conditions fail, None is returned."""
    rows = list(node.values())
    if len(rows) < 2 or not all((isinstance(r, dict) and len(r) >= RECORD_FIELDS for r in rows)) or not all((re.search('\\d', str(k)) and ' ' not in str(k) for k in node)):
        return None
    if len(set(rows[0]).intersection(*rows[1:])) < RECORD_FIELDS:
        return None
    return [r if any((re.search(ID_FIELD, str(x), re.I) for x in r)) else {'id': str(k), **r} for k, r in node.items()]

def _holds_id(node: dict[str, Any]) -> bool:
    """True when any key of the dict is named like an identifier field (id, sku, ref, asin, productId and similar, with
    ordinary words that merely end in 'id', such as 'paid' or 'valid', excluded). Used as a quick sign that a dict
    describes a listing."""
    return any((re.search(ID_FIELD, str(k), re.I) for k in node))

def _flat_block(node: dict[str, Any]) -> dict[str, Any]:
    """Flattens one level of grouping inside an attribute dict, so a group such as {'dimensions': {'depth': 40}} becomes
    {'dimensions depth': 40}. Only scalar members of a group are carried up; lists or dicts nested inside a group are
    dropped. A dict with no nested dicts is returned as is."""
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
    """Normalises one member of a list or dict by running _restore one level deeper, passing on whether the parent looks
    like a record. The exception is a string that parsed into a small dict (fewer than five fields) with no
    identifier key and no id-like value: the original string is kept, so a short store message full of colons or
    semicolons is not turned into a fake record."""
    found = _restore(value, depth + 1, carded, page)
    if isinstance(value, str) and isinstance(found, dict) and len(found) < NOTE_FIELDS and not _holds_id(found) and not any((_id_like(v) for v in found.values())):
        return value
    return found

def _restore(value: Any, depth: int=0, in_record: bool=False, page: bool=True) -> Any:
    """Normalises any store result into one canonical layout used everywhere downstream: each listing a dict inside a
    list, its attributes grouped in a sub-dict, ids as text. Strings are parsed with _records_text. Lists are first
    tried as a name/value attribute list (the plain-string form of that only inside a record on a page view) and are
    otherwise restored element by element. Dicts go through column-to-row, id-to-text and code-keyed-row conversion;
    then each member is restored, except that string members of a dict with three or more keys are only read as
    name=value blocks, and only when the dict looks like a listing. Down to depth 3, grouped attribute dicts are
    flattened, and a dict whose only member was a text that parsed into a record of three or more fields is replaced
    by that record. Recursion stops past depth 7 and other value types pass through. Every later check that reads
    rows or attributes relies on this conversion, whatever format the world uses."""
    if depth > 7:
        return value
    if isinstance(value, str):
        found = _records_text(value)
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
    """Merges the values of two spellings of one attribute into a single comma-separated list without duplicates. Both
    values are split at commas that do not sit inside numbers (so '1,200' stays whole), and elements of the second
    that are not already present, ignoring case, are appended to the first. When nothing is new the first value is
    returned unchanged; otherwise the merged text is entered in the module table _FIRST_SPELLING, mapped to the
    original first value, so later code can still recover how the key was first written."""
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
    """Lowercases text and removes the space between a number and a following unit of one to three letters, so '12 V'
    and '12V' both read '12v'. A letter run that continues into further letters or digits is not treated as a unit."""
    return re.sub('(\\d)\\s+([a-z]{1,3})(?![a-z0-9])', '\\1\\2', text.lower())

def _text_words(text: str) -> set[str]:
    """Returns the set of lowercase alphanumeric tokens in a text plus the tokens of its glued form, so '500 ml' gives
    '500', 'ml' and '500ml'. Used to score word overlap between search wording and result rows."""
    low = text.lower()
    return set(re.findall('[a-z0-9]+', low)) | set(re.findall('[a-z0-9]+', _glued(low)))

def _value_tokens(value: str) -> list[str]:
    """Picks the tokens of a requirement value worth looking for in search rows. Tokens of three or more characters are
    preferred; when there are none, two-character tokens containing a letter (such as 's2', '5l' or 'xs') are used,
    and failing those, bare numbers. The value is glued first, so a number and its unit form one token. A single
    letter with no digits yields an empty list."""
    low = _glued(value)
    return re.findall('[a-z0-9]{3,}', low) or re.findall('[0-9]+[a-z]|[0-9]*[a-z][a-z0-9]+', low) or re.findall('[0-9]+', low)

def _fold_twins(node: Any, depth: int=0) -> Any:
    """Removes duplicate spellings of a key from dicts at every level, down to depth 5. Keys are compared by their
    normalised words (camelCase, underscores and case ignored): the first spelling stays, a later one with an equal
    value is discarded, two differing text values are merged into one list with _join_twins (identifier keys
    excepted), and any other differing value is kept under its own key. If merging would shrink a dict of two or more
    keys to a single key, the dict keeps all its keys and only its members are folded. A merged value lets a
    requirement match whichever spelling lists the wanted value, and the record shown in later prompts gets shorter."""
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
    """Serialises a value as compact JSON (no spaces after separators, non-ASCII characters kept), falling back to str()
    for objects JSON cannot encode. Used for prompts, trace text and fingerprints such as the cart mark."""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def _clip(value: Any, limit: int, depth: int=0) -> Any:
    """Returns a copy of a nested value in which every string longer than `limit` is cut to that length with '...'
    appended. Dicts and lists are walked recursively, anything nested deeper than seven levels becomes '...', and
    numbers, booleans and None pass through. Keeps bulky store data short inside prompts."""
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
    """Extracts a JSON object from a model's reply text. A surrounding Markdown code fence (with or without a language
    tag) is removed, then the whole text is parsed; if that raises, the span from the first '{' to the last '}' is
    parsed instead. Returns the dict, or None for non-string input, unparsable text, or valid JSON that is not an
    object (in that case the brace span is not tried)."""
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
    """Normalises a key or name into lowercase words separated by single spaces: camelCase boundaries become spaces, and
    runs of whitespace, underscores, dots or hyphens collapse to one space, so 'heelHeight', 'heel_height' and
    'Heel-Height' all read 'heel height'. This is the basic comparison form for attribute keys and labels across the
    agent."""
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def _snake(key: str) -> str:
    """The _words form of a key joined with underscores ('heelHeight' becomes 'heel_height'), used when testing
    parameter names against patterns written for snake_case."""
    return _words(key).replace(' ', '_')

def _stem(text: str) -> frozenset[str]:
    """Returns the frozenset of a text's normalised words with a final 's' dropped from every word longer than three
    letters, a cheap singular form for comparing names and labels. Unlike _roots and _spelled it also trims words
    ending in 'ss' ('glass' becomes 'glas'), which is harmless only while both sides of a comparison use it."""
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _words(text).split()))

def _roots(text: str) -> set[str]:
    """Returns a set of rough word roots for loose matching: 'es' is removed after ch, sh, ss, x or z, otherwise a final
    's' is removed from words longer than three letters that do not end in 'ss'. Roots shorter than three characters
    and the filler words and, for, the and with are left out."""
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def _spelled(text: str) -> frozenset[str]:
    """Returns the words of a name as a frozenset, with a plural 's' removed (words over three letters not ending in
    'ss') and a final 'our' rewritten as 'or', so plurals and British or American spellings coincide ('flavours' and
    'flavor' both give 'flavor'). Used when comparing question labels with requirement and key names."""
    return frozenset((re.sub('our$', 'or', w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w) for w in _words(text).split()))

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    """Decides whether a set of requirement words and a set of key words refer to the same property. They match when
    equal, when every requirement word appears in the key (so 'colour' matches 'strap colour'), or when the key has
    two or more words and all of them appear in the requirement. An empty set on either side never matches."""
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

def _kinds(schema: dict[str, Any]) -> list[str]:
    """Lists the JSON-schema type names a parameter schema declares, whether 'type' is one string or a list of them; a
    missing or malformed type gives an empty list."""
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]

def _fits_kind(value: Any, kind: str) -> bool:
    """Checks a Python value against one JSON-schema type name. Booleans are not accepted as integers or numbers, 'null'
    needs None, and an unknown type name accepts anything."""
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
    """Fits a value to a JSON schema before it goes to the store, repairing what it can and appending a readable
    complaint to `faults` for what it cannot; `where` is the path used in those messages (for example
    'args.items[0]'). On a type mismatch it tries, in the order the schema lists its types, to turn an integral float
    into an int, a numeric string into a number, a number into a string, 'true'/'false' into a boolean or a single
    value into a one-element list; if nothing fits it records the fault and stops checking that value. It then maps
    enum values case-insensitively onto their canonical spelling, cuts strings to maxLength, and reports strings
    under minLength, numbers outside minimum/maximum and lists outside minItems/maxItems. Lists are shaped entry by
    entry and objects property by property, with unknown keys reported when additionalProperties is false and missing
    required keys reported. Returns the possibly converted value; a numeric string with a fraction sent to an integer
    field is truncated rather than reported."""
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
    """Coerces a value into a list: a list is returned as is, a non-empty string or dict is wrapped in a one-element
    list, and anything else (None, empty values, numbers) gives an empty list. Used for model JSON fields that may
    arrive as one entry or as a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []

def _count(value: Any) -> int | None:
    """Reads a whole number from a model or store field: ints as they are, floats without a fraction, and strings of
    digits optionally followed by '.0'. Booleans, strings with a sign or a real fraction, and any other type give
    None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        return int(float(value))
    return None

def _as_id(value: Any) -> Any:
    """A call argument in the form listing ids are kept in: a whole number (not a true/false value) becomes its text,
    anything else passes through unchanged. Stores whose tools take numeric ids then count in the page and cart
    bookkeeping like stores with text ids."""
    if isinstance(value, int) and (not isinstance(value, bool)):
        return str(value)
    return value

def _id_like(value: Any, named: bool=False) -> bool:
    """Tells whether a value could be a listing identifier: a string of 3 to 64 characters after trimming, with no
    internal spaces. Unless `named` says the field's name already marks it as an id, the string must also contain a
    digit."""
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip():
        return False
    return named or bool(re.search('\\d', value))

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    """Chooses which key of a set of row dicts holds the listing identifier. Candidate keys come from the first row; a
    key qualifies when every row has an id-like value under it (fields named like ids may hold codes without digits)
    and, with more than one row, those values are all distinct. Among qualifying keys, id-named ones win, then the
    one whose values are longest on average. Returns None when no key qualifies."""
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
    """Searches a result for a row dict that lists `ident` among its text values, compared after trimming. Rows are
    recognised only as dicts sitting directly in a list, but the search descends through nested lists and dicts up to
    five levels. Used to tell whether a listing named in a call actually came back in the reply."""
    if depth > 5:
        return False
    if isinstance(node, list):
        return any((isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node)) or any((_in_rows(v, ident, depth + 1) for v in node))
    if isinstance(node, dict):
        return any((_in_rows(v, ident, depth + 1) for v in node.values()))
    return False

def _llm_within(seconds: float) -> ProxyClient:
    """Returns a model client whose request timeout fits the seconds left in the episode. With 50 seconds or more the
    shared default client is used; below that, a client with the time rounded down to a multiple of five (never under
    five) is created once, cached in _llm_short and reused, each allowing one retry."""
    if seconds >= 50:
        return _llm
    limit = max(5, int(seconds) // 5 * 5)
    if limit not in _llm_short:
        _llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return _llm_short[limit]

def _fact_block(node: Any) -> dict[str, Any] | None:
    """Finds a record's attribute block: among the dict-valued members of a dict, the one with the most plain text or
    number entries (booleans excluded), returned as a dict holding only those entries. At least two such entries are
    needed; otherwise, and for non-dict input, the result is None."""
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

    """Decorator factory for optional helper methods whose failure must not end the episode. When a decorated method
    raises any exception, a 'soft_crash' entry with the method name and up to 300 characters of the error goes into
    the agent's trace, and the method returns `fallback` instead, calling it first when it is callable (so list gives
    a fresh empty list)."""
    def wrap(method: Any) -> Any:

        """Applies the guard to one method and returns the guarded replacement, which takes the same arguments as the
        original."""
        def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Runs the wrapped method and passes its result through; on any exception it records the crash in
            self.trace and returns the fallback value, or the result of calling it."""
            try:
                return method(self, *args, **kwargs)
            except Exception as exc:
                self.trace.append({'kind': 'soft_crash', 'where': method.__name__, 'error': f'{type(exc).__name__}: {exc}'[:300]})
                return fallback() if callable(fallback) else fallback
        return guarded
    return wrap

class _Stamped(list):

    """List subclass used as the episode trace. Every appended entry is stamped with the elapsed episode time, so the
    log shows when each event happened."""
    def __init__(self, clock: Any):
        """Creates an empty trace and keeps the zero-argument callable that reports elapsed seconds."""
        super().__init__()
        self.clock = clock

    def append(self, entry: dict[str, Any]) -> None:
        """Adds a dict entry after setting its 'at' field to the current clock reading, rounded to a tenth of a second,
        unless the caller already supplied one."""
        entry.setdefault('at', round(self.clock(), 1))
        super().append(entry)

def _filler(spec: dict[str, Any]) -> Any:
    """Produces a safe placeholder for a mandatory parameter the agent has no real content for, so a call such as a
    message to the shopper is not rejected for a missing field. Preference order: the first enum option, a const or
    default value, the minimum (or 1) for numbers, False for booleans, and the word 'question' cut to maxLength for
    strings. Returns None when none applies, which tells the caller to drop the call."""
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
    """Collects every string inside a nested structure of dicts and lists into one flat list, in order. Used to find the
    longest text an argument set carries, such as a message body."""
    if isinstance(node, dict):
        return [s for v in node.values() for s in _strings(v)]
    if isinstance(node, list):
        return [s for v in node for s in _strings(v)]
    return [node] if isinstance(node, str) else []

def _says(node: Any, text: str) -> bool:
    """True when some leaf anywhere inside a nested argument structure equals `text` exactly. Used to check whether a
    planned message text was actually sent in the arguments of a call."""
    if isinstance(node, dict):
        return any((_says(v, text) for v in node.values()))
    if isinstance(node, list):
        return any((_says(v, text) for v in node))
    return node == text

HEAVY_KIN = (HEAVY, 'anthropic/claude-sonnet-5', 'anthropic/claude-sonnet-4.6')
KIN_REST = 30.0
_DUE_KEY = re.compile('(?:^| )(?:arriv[a-z]*|deliver[a-z]*|ship|ships|shipping|shipped|dispatch[a-z]*|eta)(?: |$)')
_NOT_DUE_KEY = re.compile('(?:^| )(?:power|weight|member[a-z]*|relation[a-z]*|scale|method|cost|costs|fee|fees|charge|charges|price|free|address|country|region|zone|area|type|mode|option|options|size)(?: |$)')
_DUE_VALUE = re.compile('(?:^|[^a-z])(?:before|by|within|until|no later|in time|tomorrow|today|tonight|next|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|january|february|march|april|june|july|august|september|october|november|december|[0-9]+ ?(?:st|nd|rd|th)|[0-9]+ ?(?:business |working )?(?:day|days|hour|hours|week|weeks)|[0-9]+[/.-][0-9]+)(?:$|[^a-z])', re.I)
_SHOWN_KEY = re.compile('(?:^| )(?:arriv[a-z]*|deliver[a-z]*|ship[a-z]*|dispatch[a-z]*|eta|lead time|handling|transit|courier|carrier)(?: |$)')
_SHOWN_TEXT = re.compile('(?:^|[^a-z])(?:ship[a-z]*|deliver[a-z]*|arriv[a-z]*|dispatch[a-z]*|get it by|next.day|same.day|[0-9]+.day shipping)(?:$|[^a-z])', re.I)


QTY_KEY = '\\b(?:qty|quantity|count|units?)\\b'


def _unit_count(fields: dict[str, Any], rows: dict[str, Any], default: int | None) -> int | None:
    """The unit count a cart call or cart line gives: the first whole number under a field named for a quantity (qty,
    quantity, count, unit or units), else the first whole number that is not a known listing id, else the default. A
    seller or offer number beside the listing id is never read as the count when a quantity field is there."""
    for key, value in fields.items():
        if not isinstance(value, str) and _count(value) is not None and re.search(QTY_KEY, _words(str(key))):
            return _count(value)
    return next((_count(v) for v in fields.values() if _count(v) is not None and (not isinstance(v, str)) and _as_id(v) not in rows), default)


def _truth(value: Any) -> bool | None:
    """Read a flag as true or false: a boolean as it is, a number as nonzero, and the words true, yes, y or 1 against
    false, no, n or 0 in any case; anything else gives None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ('true', 'yes', 'y', '1'):
            return True
        if low in ('false', 'no', 'n', '0'):
            return False
    return None


def _faulted(seen: Any) -> bool:
    """Say whether a store reply reports a fault: ok is False, the reply is only a failure report (_failed: a small
    object with a failure key, or one bare code), or a top-level field whose name means an error, failure, fault (not
    default), denial or refusal holds something."""
    return isinstance(seen, dict) and (seen.get('ok') is False or _failed(seen) or any((v and re.search(FAIL_KEY, str(k), re.I) for k, v in seen.items())))


def _deadline(must: str) -> bool:
    """Say whether a requirement is a delivery deadline: its property is about arriving, delivery, shipping or dispatch
    (not a product property that only shares the word, such as power delivery, shipping weight, a delivery method or
    its cost), and its value names a time: before, by, within, until, a weekday, a month, an ordinal day, a number of
    days, hours or weeks, or a date."""
    key, _, value = must.partition(':')
    words = _words(key)
    return bool(_DUE_KEY.search(words)) and (not _NOT_DUE_KEY.search(words)) and bool(_DUE_VALUE.search(value))


def _shows_delivery(value: Any, depth: int=0) -> bool:
    """Say whether a store result or listing record shows anything about delivery: a field, in the first levels of
    nesting, named for arrival, delivery, shipping, dispatch, ETA, lead time, handling, transit or the carrier, or a
    short field value such as 'Ships in 2 days', 'Arrives tomorrow', 'Get it by Friday' or 'Free 2-day shipping'.
    Long free text such as a description is not read."""
    if depth > 4:
        return False
    if isinstance(value, dict):
        return any((_SHOWN_KEY.search(_words(str(k))) or _shows_delivery(v, depth + 1) for k, v in value.items()))
    if isinstance(value, list):
        return any((_shows_delivery(v, depth + 1) for v in value[:50]))
    return isinstance(value, str) and len(value) <= 60 and bool(_SHOWN_TEXT.search(value))


class Item:

    """One requested product as the agent tracks it: its name in the request's words, the store category label it should
    come from, the quantity to buy and the firm requirements as 'property: constraint' strings. Requirements learned
    from the shopper, requirements later withdrawn and properties the shopper is indifferent to are kept in separate
    lists."""
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        """Stores the product name, shelf label, quantity and initial requirement list taken from the brief, and starts
        empty lists for requirements added from the shopper's answers (told), requirements given up (dropped) and
        properties the shopper does not care about (free)."""
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Cairn:

    """One episode of a shopper's errand in an ORO store world, from reading the request to the final order or the
    decision to buy nothing. The object keeps everything the episode learns (the tools and their argument shapes,
    every row and page the store has shown, the cart, the requirements told by the request and by the shopper, the
    spend and the clock) and drives a light model through the store's own tools. Code around that loop does the parts
    a model does unreliably: it plans and phrases the questions to the shopper, searches and pages on its own before
    a decline or while a cheaper fit may exist, keeps the requested process steps, and has a stronger model review
    the cart before a checkout and the evidence before a decline."""
    def __init__(self, problem: dict[str, Any]):
        """Sets up one episode from the problem dict. From problem['environment'] it copies the binding, which is sent
        with every store call, and reads the policy view: the shopper's request text, the tool definitions (keeping
        only entries with a named function), each tool's parameter schema (an empty object schema when missing) and
        the step limit (30 when absent or not positive). It starts the episode clock and a time-stamped trace, then
        initialises the whole working state empty: step, call, cost and wait counters; the brief, tool roles, items,
        budget and message limits; process-step gate records; seen rows, opened pages and the tools each was viewed
        with; cart contents and sold-out tracking; reference chains and their resolved values; question planning and
        answer records; search, hunt and paging bookkeeping; and the pacing figures for store calls. Nothing is sent
        to a model or the store here."""
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
        self.pushed = False
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.late_held = False
        self.early_held = False
        self.unshown: list[str] = []
        self.unshown_from: dict[str, tuple[int, bool]] = {}
        self.delivery_seen = False
        self.kin_rest: dict[str, float] = {}
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

    def left(self) -> int:
        """Number of store steps still available: the step limit minus the steps used so far."""
        return self.max_steps - self.step

    def clock(self) -> float:
        """Seconds elapsed since the agent was created, read from the monotonic clock."""
        return time.monotonic() - self.started

    def model_room(self) -> float:
        """Seconds still usable for work: the 300-second task limit minus a 20-second safety margin minus the time
        already spent. Model timeouts and the checks that stop late model calls are measured against it."""
        return TASK_SECONDS - TIME_MARGIN - self.clock()

    def rushed(self) -> bool:
        """True when the usable time left is no more than twice the average duration of a store call so far, that
        average counted as at least one second (and as one second before any call). Optional reads consult it so the
        remaining time stays available for placing the order."""
        pace = self.env_spent / self.env_count if self.env_count else 1.0
        return self.model_room() <= 2 * max(pace, 1.0)

    def msgs_free(self) -> int:
        """Messages to the shopper still available for questions: the message limit minus messages already sent minus
        the part of the reserve (messages kept for order-time or ending steps) that the process gates have not used
        yet."""
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        """A stable text fingerprint of the cart, the sorted listing-to-quantity pairs as compact JSON. Gate approvals
        and cheaper-listing checks are keyed on it so they run again after the cart changes."""
        return _dump(sorted(self.cart_lines.items()))

    def row_ids(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Repairs listing arguments that the model filled with a code taken from a title instead of the listing id. It
        only acts on the open, offer, compare and cart tools, once rows have been seen and a title field is known:
        each string argument or list element of 3 to 30 characters that contains a digit, is not already a known id
        and is not an enum parameter is looked up as a separate token in the titles of seen rows, and when exactly
        one row matches it is swapped for the matching row's id (logged as 'row_id'). Everything else passes through
        unchanged."""
        if name not in {self.roles.get(job) for job in ('open', 'offer', 'compare', 'cart')} or not self.rows:
            return args
        title = self.title_key()
        if not title:
            return args

        props = (self.schemas.get(name) or {}).get('properties') or {}

        def known(value: Any, key: str) -> Any:
            """Returns the replacement for one argument value: the id of the single seen row whose title contains the
            value as a separate token, or the value itself when it is not a candidate or matches no row or several
            rows."""
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
        """The set of tools that show a listing's page: the offer tool, or the open tool when the store has no separate
        offer tool, plus the side-by-side compare tool when there is one. Calls to these tools are what mark a
        listing as opened."""
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def facts_tool(self) -> str | None:
        """Names the single-listing tool that shows attributes. A tool already observed to return attribute blocks and
        taking exactly one text target wins; otherwise the 'open' role tool is used when it takes a single target and
        has not been seen to lack attributes. Returns None when neither applies."""
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get('open')
        return tool if self.facts_from.get(tool or '') is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        """Whether the compare tool has been seen to return attribute blocks: True or False once observed, None while
        untested or when the store has no compare tool."""
        return self.facts_from.get(self.roles.get('compare') or '')

    def compare_bare(self) -> bool:
        """True when the compare tool should be treated as showing no attributes: it was seen without them, or it is
        still untested while the single-listing attribute tool is known to show them or a separate offer tool exists.
        Callers then read listings one at a time instead of side by side."""
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))))

    def gate_due(self, stage: str) -> bool:
        """True when the request has a process step for this stage ('choose', 'order' or 'end') that has not been
        confirmed for the current state. Confirmations are stored against the cart fingerprint plus a version counter
        that moves whenever requirements or the task change, so either kind of change makes the step due again."""
        return any((p['when'] == stage for p in self.process)) and (stage, self.cart_mark() + f'#{self.version}') not in self.gate_ok

    def tail(self) -> int:
        """Steps to hold back for finishing: two when an order-time process step needs a message to the shopper,
        messages remain and that step is still due (one for the message, one for the checkout), otherwise one."""
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        """Picks the models for a check from the money spent so far: the heavy model followed by the light one while
        spend is within `limit` and a heavy call still fits under the hard cap, the light model alone until the hard
        cap is reached, and an empty list after that, which callers take as a signal to skip the check."""
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        """Gets one chat reply, trying the given models in order. Each model counts once against the episode call cap,
        and no attempt starts after the cap is reached or when fewer than CALL_FLOOR seconds of usable time remain
        (logged as 'late_call_skipped'). Transient failures (network errors, 402, 408, 425, 429 and 5xx) are retried
        with jittered back-off of 4, 10, 20 and 35 seconds, network errors and 408 only once, while the episode's
        total waiting stays within 45 seconds and the pause leaves time for the call; an empty reply or any other
        error moves straight to the next model. A 402 that outlasts the retries marks the run as out of credit and
        ends the attempt without trying further models. Returns the message dict, or None when every model failed. A
        heavy-model call climbs a ladder of kin models (HEAVY_KIN): a 429 hands it to the next kin at once, gives the
        call back to the cap and rests the limited model for KIN_REST seconds; a kin whose retries end on a 5xx hands
        it on as well. A timeout or an empty reply moves to the next requested model as before, and light and spare
        calls are unchanged."""
        for model in models:
            ladder = HEAVY_KIN if model == HEAVY else (model,)
            for choice in ladder:
                if self.calls >= ALL_CALLS:
                    return None
                if self.model_room() < CALL_FLOOR:
                    self.trace.append({'kind': 'late_call_skipped', 'at': round(self.clock(), 1)})
                    return None
                if choice != ladder[-1] and time.monotonic() < self.kin_rest.get(choice, 0.0):
                    continue
                self.calls += 1
                tries = 0
                while True:
                    message, status = self.call_model(choice, messages, max_tokens, tools)
                    if message is not None:
                        return message
                    if status == 429 and choice != ladder[-1]:
                        self.kin_rest[choice] = time.monotonic() + KIN_REST
                        self.calls -= 1
                        self.trace.append({'kind': 'kin_next', 'from': choice})
                        break
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
                if status != 429 and not (isinstance(status, int) and status >= 500):
                    break
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, effort: str | None=None) -> tuple[dict[str, Any] | None, Any]:
        """Sends one chat-completion request and returns (message, status). For Anthropic models with more than one
        message, the first and last messages are copied with ephemeral cache-control markers so the long shared
        prefix can be cached. Requests use temperature 0, offer the tools with automatic choice when given, and go
        through a client whose timeout fits the remaining time. A reply's reported cost is added to the running
        spend; a reply with neither text nor tool calls counts as a failure with status 0, and errors are logged with
        the HTTP status or the provider's error code (None when that code is not numeric). An optional effort asks
        the provider for that reasoning effort."""
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
        if effort:
            body['reasoning'] = {'effort': effort}
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

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        """Sends a single-message prompt and returns the first JSON object found in a reply, trying the models one after
        another until one answers with parseable JSON. It gives up early with None when a store call has failed for
        lack of payment, model credit has run out or the call cap is reached."""
        for model in models:
            message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get('content'))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Performs one store tool call and returns the environment's reply dict, or None when it cannot. The payload is
        the episode binding plus the next step number and the action, and calls are spaced at least env_gap seconds
        apart; the time each call takes feeds the pace used by rushed(). On success the step counter follows the
        reply, the finished flag is picked up, the observation is normalised with unspelled (shopper replies from the
        message tool are left raw), note_read records what was learned, a single-listing reply from the attribute
        tool that has an id but no attribute block is noted in factless, and for page tools whose reply is not an
        error each string argument value is marked as opened, with the tool added to that listing's views (list
        arguments only for values the reply mentions). Failures get up to four attempts: a 409 saying the episode is
        finished, finalised or quarantined sets finished; a 409 naming the expected step resynchronises the counter
        and retries; a 409 about reusing a step advances the counter and retries; 400, 401, 403, 404 and 422 give up
        at once (402 also marks the run as broke); other errors are retried immediately."""
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
                seen = data.get('observation')
                self.note_read(name, args, seen)
                if name == self.facts_tool() and isinstance(seen, dict) and not _failed(seen) and self.ident(seen) and not _fact_block(seen):
                    self.factless.add((name, self.ident(seen)))
                if name in self.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))) and not _failed(seen):
                    shown = _dump(seen)
                    for value in args.values():
                        named = [v.strip() for v in map(_as_id, value if isinstance(value, list) else [value]) if isinstance(v, str) and (not isinstance(value, list) or v.strip() in shown)]
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
        """Normalises a tool reply with _restore and _fold_twins; lists of 'name: value' strings are read as attributes
        for every tool except search. For listing view tools (open, offer, compare) it goes further: it gathers the
        field names seen on search rows of listings not yet opened, and for any listing record without an attribute
        block it moves the plain text and number fields those rows do not carry into an 'attributes' sub-dict,
        provided there are at least three. Stores whose pages put attributes right beside the id and price thus give
        the fit checks the same shape as stores with a nested block. Records are handled at the top level or inside
        lists under top-level keys."""
        views = {self.roles.get('open'), self.roles.get('offer'), self.roles.get('compare')}
        out = _fold_twins(_restore(observation, page=name in views or name != self.roles.get('search')))
        if name not in views:
            return out
        self.row_keys.update((k for ident, row in self.rows.items() if ident not in self.opened for k in row))
        known = self.row_keys
        if not known:
            return out

        def gathered(record: Any) -> Any:
            """Returns the record with its extra scalar fields (absent from search rows, booleans excluded) nested under
            'attributes', or the record unchanged when it is not an id-carrying dict, already has an attribute block,
            or has fewer than three such fields."""
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

    def read_request(self) -> None:
        """Turns the shopper's request into the agent's working brief before any store call. It sends the BRIEF prompt
        with every tool's name, description and parameter schema, the request and any opening store note (strings
        clipped to 1500 characters) to the light model; a brief missing items, roles or a checkout tool is asked
        again of the heavy model, and a brief in which two items share a store label gets a second opinion from the
        heavy model that replaces it only when complete. If no brief arrives at all, it retries every 10 seconds
        until 120 seconds into the episode. From the brief it then keeps only roles that name real tools; when one
        tool both checks out and declines and the mode arguments are missing or identical, asks a model for the
        arguments that tell the two apart; builds the items (quantity at least 1) and merges items that share a label
        and quantity and have no requirement on the same property with a different value; remaps 'fits' pairs to the
        merged numbering; records lookup requirements as references to their item, matching by property name when the
        text differs; and runs unmix. Finally it sets the budget (positive numbers only, numeric strings accepted),
        the message limit (3 by default when a message tool exists), a tighter step limit when the request states one
        from 10 up to below the store's limit, the process steps with their timing normalised to choose, order or
        end, and the message reserve: the largest message need among order and end steps, capped by the limit."""
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
        """True when two or more items in a brief carry the same non-empty store label after normalisation, a hint that
        the brief split one product in two or mixed up the labels; read_request then asks the heavy model again."""
        shelves = [_words(str(e.get('shelf') or '')) for e in _items(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def unmix(self) -> None:
        """Repairs briefs that turned the description of a referenced listing into requirements on the product itself.
        For every item that has both a requirement pointing at another listing and other requirements, it asks the
        heavy model (light as fallback) which of the other requirements only describe that referenced listing. Those
        are removed from the item and appended in parentheses to the first pointing requirement ('... (that listing:
        a; b)'), the reference list is updated to the new text, and the change is logged as 'unmix'."""
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
        """Decides whether a requirement is worded as a description that no attribute key of the item's listings
        addresses, in which case the ledger shows it as deciding nothing. It can hold only for a requirement that
        does not point at another listing, has a property name and a value of four or more words, avoids words such
        as listing, page, unlike or same as, and does not mention another requested item; and only once at least
        three opened listings in the item's category have attribute blocks, none of whose keys shares a word stem
        with the requirement's property."""
        if need in [m for _, m in self.refs] or need in [str(x) for x in _items(self.brief.get('lookups'))]:
            return False
        prop, _, value = need.partition(':')
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
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
        """Builds the bulleted task summary shown to the buying model every turn. Per item it gives the name, store
        label, quantity and current requirements (from the request and the shopper, minus withdrawn ones), adding the
        value a resolved reference page supplied and listing description-like requirements separately as
        non-deciding. It then adds the budget with unit and scope, the pick rule, the in-stock rule, wishes as
        tie-breakers, the request's working rules, the quiet-decline instruction, each process step with when it
        falls, and, once known, which tool shows attributes, which tools show none and which tools show a listing's
        page. Delivery deadlines that no listing can show (unshown) are listed after the items as no requirement."""
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
            parts.append(text)
        if self.unshown:
            parts.append('not requirements (no listing in this store states a delivery or arrival time, so none can be checked): ' + '; '.join(self.unshown))
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
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        """Composes the per-turn status block: steps used and left, then the most urgent applicable warning, from a
        final-steps notice when little beyond the checkout reserve is left, through 'few steps left' (six or fewer,
        or a few more while nothing has been checked out), to 'running low' and 'half the steps used with an empty
        cart' prompts shown only before the first checkout. After 200 seconds it adds a time warning unless the
        final-steps notice already applies. It then lists up to three unresolved references to look up (before any
        checkout), the candidate listings to review, the accessory/device fit plan when items must fit each other,
        the last four task changes and the ledger."""
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
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        plan = self.fit_plan() if self.bound and self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        text += '\n[ledger]\n' + self.ledger()
        return text

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        """Returns the listing id of a record, or None. The record must be a dict of at least three fields; the learned
        id key is used when it holds an id-like value, and otherwise, for list members, top-level records or records
        with nested dicts, this record's best id candidate is accepted when its key mentions 'id' or its value is a
        listing already seen."""
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
        """Decides whether a list is a list of listing rows and, where needed, learns which key holds their ids. At
        least half the entries must be dicts with three or more fields; if the current id key fits all of them the
        answer is yes, otherwise a key chosen with _learn_id becomes the new id key. Returns False when no key works."""
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

        """Folds a store result into the episode's memory of product rows. Every dict the id logic recognises as a
        listing (searched up to six levels deep, with each list first checked for a learnable id column) is merged
        into rows under its id, a first sighting appends the id to seen_order, and a recognised record is not
        descended into further. A boolean field flipping from True to False marks the id unavailable and a
        False-to-True flip clears that mark, whatever the field is called. When the record carries a nested attribute
        block a full copy goes into self.full, and later partial sightings of an id already read in full are merged
        into that copy too. Afterwards the whole result is scanned once for delivery details (delivery_seen), and
        drop_unshown_deadlines sets aside or puts back a delivery deadline accordingly."""
        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            """Recursive visitor behind absorb: gives up past depth 6, merges a dict that yields an id as a listing,
            otherwise recurses into a dict's values, and for a list first asks learn_rows whether its elements look
            like rows so the children are identified with the in-list flag. Returns nothing; all effects land on
            rows, seen_order, unavailable and full."""
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if ident:
                    if ident not in self.rows:
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
                    old = self.rows[ident]
                    flips = [self.stock_flag(k) * (1 if v is True else -1) for k, v in node.items() if isinstance(v, bool) and isinstance(old.get(k), bool) and old.get(k) is not v]
                    if any((f < 0 for f in flips)):
                        self.unavailable.add(ident)
                    elif any((f > 0 for f in flips)):
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
        if not self.delivery_seen and _shows_delivery(observation):
            self.delivery_seen = True
        self.drop_unshown_deadlines()

    def drop_unshown_deadlines(self) -> None:
        """Set aside, or put back, the delivery deadlines (_deadline) among the requirements in force. A deadline is set
        aside while nothing the store has returned shows a delivery detail (delivery_seen, kept by absorb from whole
        results, plus every listing row and full record through _shows_delivery), once five or more listings were
        seen and a listing page was opened: one from the request leaves the musts, one from a reply joins dropped,
        each is listed in unshown with where it came from, the version goes up and a note tells the model why. As
        soon as a result shows delivery details, every unshown deadline goes back where it was and the version goes
        up again. Story deadlines ('it would have to arrive before the 15th') otherwise make the cart check reject
        every fitting listing."""
        if self.delivery_seen:
            if self.unshown:
                for must in self.unshown:
                    number, was_must = self.unshown_from.get(must, (0, True))
                    if 1 <= number <= len(self.items):
                        item = self.items[number - 1]
                        if was_must and must not in item.musts:
                            item.musts.append(must)
                        elif not was_must and must in item.dropped:
                            item.dropped.remove(must)
                self.trace.append({'kind': 'deadline_restored', 'musts': list(self.unshown)})
                self.unshown = []
                self.version += 1
            return
        dated = [(number, item, must) for number, item in enumerate(self.items, 1) for must in item.musts + item.told if must not in item.dropped and _deadline(must)]
        if not dated or len(self.rows) < 5 or not self.opened:
            return
        if any((_shows_delivery(record) for record in list(self.rows.values()) + list(self.full.values()))):
            self.delivery_seen = True
            return
        for number, item, must in dated:
            was_must = must in item.musts
            if was_must:
                item.musts.remove(must)
            else:
                item.dropped.append(must)
            if must not in self.unshown:
                self.unshown.append(must)
                self.unshown_from[must] = (number, was_must)
            self.trace.append({'kind': 'deadline_dropped', 'must': must})
            self.queued.append(f"(Cairn) no listing in this store states a delivery or arrival time, so '{must}' cannot be checked against any listing and is not a requirement")
        self.version += 1

    def labels(self) -> set[str]:
        """Gathers every enum option string found anywhere in the tool schemas, trimmed and lowercased, by recursing
        through nested schema dicts up to six levels. The set lets category_key recognise a row field whose values
        are the store's own category choices. Options inside list-valued schema parts such as anyOf are not visited."""
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            """Adds the string options of one schema node's enum to the shared set, then descends into each dict-valued
            member; stops beyond depth 6 or on non-dict input."""
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
        """Guesses which row field names a listing's category. Candidate fields hold short strings (1-40 characters, no
        run of three digits) on at least 80% of known rows and take at most 40 distinct values. The winner has the
        largest share of values that also appear as enum options in the tool schemas, then the fewest distinct
        values, then the widest coverage, then the longest average text. Returns the field name or None; it is
        recomputed from all rows on every call."""
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
        """Summarises which attribute keys the fully read listings carry, per category label, as raw material for
        planning questions. Each record in self.full contributes its nested attribute block under its category value
        (or 'listings'), noting per key how many records carry it and each record's value, plus the value's first
        spelling before twin merging. Labels are narrowed to those matching an ordered item by shelf or name words, a
        shared stem of 4+ letters, or owner_of, unless none match. Within a label, keys with long values (mean over
        40 characters), mostly slash-combined values, or values that look like unique free text are dropped, and keys
        that twins judges to be the same property are folded into the earlier one. Returns tuples (label, key, record
        count, up to four distinct values clipped to 40 characters), at most TABLE_KEYS per label, biggest labels and
        most common keys first."""
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
        """Flags a table key that merely resembles a requirement's property. For the item owning the label, each plain
        'property: value' requirement is checked (values containing comparison or negation words are skipped): when
        its property shares a stem word with this key without being the same name, and the requirement's words are
        not all found among the key's sampled values, a bracketed warning naming the requirement's property is
        returned. This keeps the question planner from treating a lookalike key as the stated one; '' when nothing
        applies or no item owns the label."""
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
        """Decides whether two attribute keys from the key table describe one property under different spellings. The
        names must share a word; values are then compared listing by listing wherever both keys are filled. If
        neither name's words contain the other's, at least two shared listings are needed and all must agree; if one
        name is a word subset of the other, most shared listings must agree, and with no shared listing a common
        non-empty value is enough."""
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            """Value match used by twins: identical strings, or the shorter value's alphanumeric words form a subset of
            the longer value's while the longer has at most twice as many words plus two."""
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
        """Buckets every seen listing id by its category value, in first-seen order, and keeps the buckets whose label
        corresponds to an ordered item (the same word test as key_table, or owner_of); all buckets survive when none
        corresponds. Returns the biggest buckets first, no more of them than there are items (at least one); widen
        uses them to spread listing reads across items. Without a category field all ids share one bucket."""
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
        """Finds the lone free-text parameter a tool is aimed at, such as a listing id or a search phrase: the first
        required string parameter without an enum, or else the only such string parameter. Returns None when the tool
        is missing, no such parameter exists, or the schema requires any other parameter, so callers can safely fill
        exactly one field."""
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        """Reads a listing's price as a float from the field named by price_key. A dict value yields its first
        non-boolean number, and a string such as '$1,299.50 USD' is parsed allowing a short currency prefix and a
        short unit suffix; any other value, or an unknown price field, gives None."""
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str):
            figure = re.fullmatch('\\s*[^\\d\\s]{0,3}\\s?(\\d[\\d,]*(?:\\.\\d+)?)\\s*[A-Za-z]{0,6}\\.?\\s*', value)
            value = float(figure.group(1).replace(',', '')) if figure else None
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def stock_flag(self, key: str) -> int:
        """How a true/false record key bears on availability: -1 for a key that means not available when true (sold out,
        out of stock, unavailable, not for sale, backordered, discontinued), 1 for a key that names stock or sale
        explicitly (in stock, for sale, can buy, purchasable, orderable), 0 for a promotional, delivery or rating
        flag that says nothing about stock, and 1 for any other key, which keeps the base's reading that true means
        buyable. The stock key the result reader learned is never treated as promotional."""
        words = _words(key)
        if re.search(STOCK_NEGATIVE, words, re.I):
            return -1
        if re.search(STOCK_POSITIVE, words, re.I):
            return 1
        if key != self.stock_key and re.search(PROMO_FLAG, words, re.I):
            return 0
        return 1

    def buyable(self, ident: str) -> bool:
        """Whether a listing can still be purchased. It cannot when it was marked unavailable or gone, when its stock
        field reads like an out-of-stock word ('no', '0', 'sold out', 'unavailable' and similar), or when that field
        is False or a number at or below zero; with no stock field known the listing counts as buyable. A stock field
        whose name means not available (sold out, out of stock) is read by its truth in any form, a boolean, 0 or 1,
        or a yes/no or true/false word, before the field's text is matched against sold-out words."""
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        if self.stock_key and self.stock_flag(str(self.stock_key)) < 0 and _truth(value) is not None:
            return not _truth(value)
        if isinstance(value, str) and re.search(STOCK_OUT, value.strip(), re.I):
            return False
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """Pulls (id, record) pairs out of a result without deep recursion. A dict that is itself a listing yields just
        itself; otherwise its list values contribute their identifiable dict elements, and each dict value
        contributes either itself (when it is an identifiable listing) or the identifiable rows of its own lists. A
        top-level list yields its identifiable elements; anything else gives []."""
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
        """Remembers, for a viewing tool, which fields its results have ever carried, and for each listing in the result
        which fields were seen under the current requirement version; a version change starts that listing's set
        afresh. shown_already relies on these sets to recognise views that would return nothing new."""
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        """True when opening the listing again with this tool cannot add anything: the tool's known field set is
        non-empty, the listing was seen under the current version with every one of those fields, and it has not been
        written off as gone. The call loop uses it to answer a repeated view without spending a step."""
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and (keys <= have) and (ident not in self.gone)

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """Learns, once per tool, how its list results are laid out by asking a light model (heavy as fallback) to name
        the key of the row list, the total-count key, the next-page key, the parameter that accepts that value, and
        the price and stock keys inside a row. It only tries for a dict holding a list of at least two dicts, and the
        per-tool cache is set to None before the request, so a failed or unusable reply is never retried. Each
        proposed key is kept only if it really exists in the result, the schema or the first row, and a count that is
        not an integer is discarded. The first learned price and stock keys become the agent's price_key and
        stock_key, and the shape is written to the trace."""
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
            """Returns the proposed key when it is a string present in the given mapping, else None, so key names the
            model invented are ignored."""
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), props), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
        if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
            shape['count'] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        """Turns a list result into paging facts using the tool's learned shape: (total matches, rows on this page, the
        parameter that takes the next-page value, that value). The value is None when the store offers no next page.
        Returns None when the shape lacks a count, next key or next parameter, or when the result's rows, count or
        next value have unexpected types."""
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
        """True when a result holds at least three listings, all with known prices, in non-decreasing price order;
        callers then trust the first page to hold the cheapest rows and skip extra paging."""
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """After a first-page search for a cheapest, most or least pick, fetches the following pages itself when the
        store reports more matches than it showed. Nothing happens when paging facts are missing, the call already
        asked for a later page (an integer above 1 or any non-empty cursor), no next value exists, or, for cheapest,
        the rows already come price-sorted. The needed page count is cut to two when the auto-page allowance
        (AUTO_PAGES) and at least 14 steps beyond the tail permit; if it still exceeds two, the allowance or the step
        budget, only a hint is returned suggesting narrower words or the page argument. Otherwise each page costs one
        step: it is observed, kept in paged_obs for cheapest_here, stored in read_cache under the call's key so an
        identical later request is answered free, and its text is collected until no next page remains or a call
        fails. Returns the gathered pages, with a note when matches remain unread, or ''; paged_obs is reset on every
        call."""
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
        rest = f"\n[Cairn] {count} listings matched and Cairn has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Cairn read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Warns when a search ignored its own price ceiling. Ceiling arguments are numeric arguments whose name
        suggests a maximum and whose name or description mentions money, or the single cap key reported by cap_keys;
        the lowest such value is compared with each returned row's price. When rows exceed it, the note names the
        argument and how many rows cost more, and if the tool has enum mode arguments it suggests switching each to
        its first other option with the same ceiling. Returns '' when no ceiling was passed, the price field is
        unknown, or every row respects it."""
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            """Whether a parameter's name or schema description mentions price, cost, amount, budget, spend or money."""
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
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in modes.items()}
        hint = '; list the category with ' + ', '.join((f'{k}={_dump(o[0])}' for k, o in others.items() if o)) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[Cairn] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        """For cheapest picks, lists up to six buyable listings from the given results (the search plus any pages read
        automatically) with their prices, lowest first, so the model sees the low end at once. Returns '' for other
        picks, without a price field, or when fewer than two priced buyable rows appear."""
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Cairn] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def wider_hold(self) -> str:
        """Holds back a checkout once for cheapest picks while steps are plentiful (at least 14 plus two per item). It
        sets wider_held and returns a message saying the checkout did not run and asking the model to search each
        item under other product names, inspect buyable rows priced under the current cart lines (whose prices it
        lists), read the cheapest of them and swap in a fit before checking out. Returns '' when already used since
        the last reset, for other picks, or when steps are short."""
        if self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def note_facts(self, tool: str, shown: bool) -> None:
        """Tracks whether a single-listing tool returns attribute blocks. One view with attributes marks the tool as a
        reliable source for good; while it has never shown attributes, each bare view counts against it, and the
        second such view marks it attribute-less so later reads go elsewhere."""
        if shown:
            self.facts_from[tool] = True
        elif self.facts_from.get(tool) is not True:
            self.bare_pages[tool] = self.bare_pages.get(tool, 0) + 1
            if self.bare_pages[tool] >= 2:
                self.facts_from[tool] = False

    def lists_things(self, tool: str) -> bool:
        """True when a tool behaves like a browsing or search tool: its sole target parameter is named like a
        shelf/category or a free-text query, or a row-list shape has been learned for it. Used to keep such tools
        from being adopted as the per-listing view tool."""
        target = self.single_target(tool) or ''
        return bool(re.search(SHELF_ARG, _snake(target), re.I) or re.search(QUERY_WORDS, _snake(target), re.I)) or tool in self.shapes

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Opens listings to obtain their full attribute records while keeping at least `keep` steps in reserve.
        Candidates are ids already seen as rows, not yet read in full and not sold out. If the compare tool takes an
        id array and is known to return attributes, ids go out in batches of compare_room: a batch that returns rows
        raises compare_ok and leaves any ids still lacking attributes for single opens, while an empty reply lowers
        compare_bad and either retries smaller or stops. Remaining ids are then opened one at a time with the
        per-listing view tool, skipping id/tool pairs known to be attribute-less and stopping at the reserve, a
        finished episode, time pressure or a tool found bare; each view feeds note_view and note_facts. Returns 'id:
        <shown text>' snippets of at most 1500 characters each, or [] when neither a single-view tool nor an
        attribute-giving compare tool exists."""
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
        """For orders with several items or several cart lines, chases part codes that a cart line's text refers to
        ('see part X'). Each such code is searched once with the value-search template, unless some known listing
        already carries it as its own code, and the listings whose own code matches are then opened through
        fill_facts with a larger reserve. It stops on short steps, time pressure, a schema fault or a failed call;
        nothing is returned, the gathered rows only enrich the agent's records."""
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
        """Finds a browsing tool that takes exactly one free-text shelf/category parameter and exactly one price-ceiling
        parameter, and fixes a value for each of its multi-option enum parameters. For each such enum it prefers a
        mode in which the ceiling was observed holding, then the mode used by the latest shelf listing done without
        query words, then the one option whose name suggests browsing or filtering, then (for a single required mode)
        the one option never used with words. When the mode is the tool's single required enum, modes in which the
        ceiling failed are ruled out. Returns (tool, fixed mode arguments, shelf parameter, ceiling parameter), or
        None as soon as the first fitting tool leaves a mode undecided."""
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
        """Picks the numeric parameters of a tool that could act as a price ceiling, ignoring ones named like counts,
        pages or sizes and ones named like lower bounds. The first non-empty group wins: parameters cap_modes saw
        holding as a ceiling, then those named both as a maximum and as money, then those named as money, then those
        named as a maximum. Returns [] when none qualify."""
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
        """Judges from this tool's recorded calls whether each numeric argument behaved as a price ceiling, keyed by
        argument and by the call's enum-mode settings. For a call whose rows all have prices, an argument turns False
        for good once any row costs more than its value. It turns True when every row sat at or under the value and,
        judged against the known prices of rows with the same labels, that outcome would be unlikely by chance
        ((share of those prices at or under the value) raised to the row count falls below CAP_CHANCE). Rebuilt in
        full on every call."""
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
        """Walks a capped shelf listing downward to reach cheaper rows for each cart line with a price of at least 2 and
        a category label, once per line and price. The first request sets the ceiling to the integer just under the
        line's price, and up to WALK_CALLS requests follow, each lowering the ceiling under the lowest price
        returned. The walk stops when no rows come back, a row breaks the ceiling, paging facts are missing,
        everything fits on one page or the page is price-sorted; when everything fits in two pages it reads the
        second page and stops. Step reserves (tail plus 7 for the first call, tail plus WALK_ROOM afterwards) and
        time pressure end it early; results only reach the row store."""
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
        """For cheapest picks, once per cart state and requirement version, looks for unread listings that could
        undercut the cart. It first runs list_below, then for each priced cart line collects seen, unread, buyable
        rows of the same category label that cost less, cheapest first, with the cheapest row found by an
        all-requirements search moved to the front. Up to the compare batch size (or two) per line are opened through
        fill_facts with a reserve of the tail plus 6, the total capped by the compare size per line or by six.
        Returns the opened snippets, or [] when the pick is not cheapest, this state was already handled or no
        category field exists."""
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
        """Lists, for each priced cart line with a category label, up to CHEAPER_SHOWN unread buyable rows of the same
        label that cost less, lowest first, as 'id at price'. Returns one sentence naming them, or '' when there are
        none; errors are swallowed to ''."""
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
        """Guesses the row field holding product names. Candidates are string fields other than the category field, with
        values up to TITLE_CHARS long, present on at least 80% of rows, averaging two or more words, and with
        distinct values for at least half of their occurrences. Among those whose distinct-value count reaches 90% of
        the highest such count, the field with the shortest average length wins; None when nothing qualifies."""
        cat_key = self.category_key()
        totals: dict[str, list[Any]] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if key != cat_key and isinstance(value, str) and len(value) <= TITLE_CHARS:
                    slot = totals.setdefault(key, [0, 0, 0, set()])
                    slot[0] += len(value.split())
                    slot[1] += 1
                    slot[2] += len(value)
                    slot[3].add(value)
        kept = [(slot[2] / slot[1], key, len(slot[3])) for key, slot in totals.items() if slot[1] * 5 >= len(self.rows) * 4 and slot[0] >= slot[1] * 2 and len(slot[3]) * 2 >= slot[1]]
        most = max((distinct for _, _, distinct in kept), default=0)
        ranked = [(mean, key) for mean, key, distinct in kept if distinct * 10 >= most * 9]
        return min(ranked)[1] if ranked else None

    def hunt_words(self, number: int) -> tuple[str, list[str]]:
        """Supplies search vocabulary for one item: a noun plus a ranked list of extra words. The noun is the longest
        word of 4+ letters in the item's most frequent store label (or its shelf/name when no label is known) that is
        absent from every other item's shelf and name, left blank once a noun-only search came back empty. Extra
        words come from titles of the item's rows and short attribute values of its fully read records: a word must
        occur in at least two names and in no more than max(2, HUNT_COMMON of them), must differ from the noun, and
        must not share a root with the item's own name or shelf. Words are ordered by frequency, then alphabetically."""
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
        """Counts known rows whose category label belongs to the given item; without a category field every row counts.
        Counts known rows whose category label belongs to the given item; without a category field every row counts."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for row in self.rows.values() if not cat_key or self.owns(item, str(row.get(cat_key) or ''))))

    @_soft('')
    def hunt_cheaper(self) -> str:
        """Runs extra scripted searches, with no LLM turn, to uncover lower prices for cheapest picks. It runs at most
        HUNT_ROUNDS times, only when the requirements, budget or set of gone listings changed since the previous run,
        only with a value-search template and a price field, and only before HUNT_BY seconds. Its step budget is the
        smaller of HUNT_STEPS and what remains after the tail, HUNT_KEEP and two steps per item; it needs at least
        three and shares them among items (three each at least) through hunt_item. Sets wider_held when any search
        ran, traces the counts, and returns a sentence with the searches made and new rows seen, or ''."""
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

    def hunt_item(self, number: int, tool: str, base: dict[str, Any], field: str, room: int) -> tuple[int, int]:
        """Uses at most `room` store calls for a single item. A base query that returns rows is found first and cached
        per item and value set, trying all required values with the noun, the values alone, all but the last value
        with the noun, and the noun alone; a noun-only miss disables the noun for later hunts, and a query that could
        not be issued ends the item. The base is then refined by prefixing the next untried word from hunt_words
        until the room runs out, HUNT_IDLE searches in a row add no rows, the schema refuses the call, the episode
        ends or the hunt deadline passes. Returns (steps spent, new rows of the item)."""
        tried = self.hunt_tried.setdefault(number, set())
        noun, _ = self.hunt_words(number)
        values = [value for value, _ in self.value_words(number)]
        spent = gained = idle = 0
        issued = refused = False

        def run(query: str) -> bool:
            """Executes one hunt query and follows its pages, recording the query as tried. Pages whose call key is
            already in read_calls are reused free and treated as productive; new pages cost a step each, update the
            spent, gained and idle counters, and stop once the room is used. Rows from a query that contains every
            required value are added to value_rows. Paging continues while a next page exists and either the page
            added at least HUNT_PAGE_NEW item rows or the total is at most HUNT_WHOLE. Returns whether any rows
            appeared; a schema fault sets the refused flag."""
            nonlocal spent, gained, idle, issued, refused
            tried.add(query.lower())
            faults: list[str] = []
            args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
            refused = refused or bool(faults)
            if faults or spent >= room or self.finished:
                return False
            issued = True
            holds_values = bool(values) and all((v.lower() in query.lower() for v in values))
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
        mark = (number, tuple(values))
        stem = self.hunt_stem.get(mark)
        if stem is None:
            tries = ([' '.join((values + [noun])).strip(), ' '.join(values)] if values else []) + ([' '.join((values[:-1] + [noun])).strip()] if len(values) >= 2 else []) + ([noun] if noun else [])
            stem = ''
            for query in dict.fromkeys(tries):
                issued = False
                if run(query):
                    stem = query
                    break
                if not issued:
                    return (spent, gained)
                if query == noun:
                    self.hunt_no_noun.add(number)
                    noun = ''
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
        """A last scripted search sweep, run once and before HUNT_BY, when the agent is about to end without a purchase.
        It needs a value-search template and at least three steps beyond a reserve of the tail plus one per item plus
        one. Items without a held cart line are targeted (all items when each has one): unread_pages first follows
        further pages of earlier searches containing their requirement words, then each item gets a share of new
        queries built from series codes, requirement values, hunt words and the noun. Afterwards up to HUNT_FIT_OPENS
        unread, buyable, within-budget rows per item, ranked by how many requirement tokens their text holds and then
        by price, are opened through fill_facts so the audit can judge them. Returns the opened snippets; a failed
        store call returns [] at once."""
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
        """Spends up to `room` steps reading further pages of searches already made with this tool whose string
        arguments contain one of an item's requirement tokens. A search is skipped when all its rows belong to other
        items. Two passes run: first each item is held to an even share, then any item may use what is left; pages
        already in read_calls are stepped through free, and call keys guard against loops. Returns the number of
        steps used."""
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
        """Builds a canonical key for a tool call so equivalent requests compare equal: arguments sorted by name,
        strings lowercased with whitespace collapsed, and integer arguments dropped when they equal the schema
        minimum or default (or equal 1 for the tool's known page parameter when the schema gives neither). Returns
        the tool name followed by the JSON of what remains."""
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
        """Stores the result of a search or browse call under its canonical key in read_calls, with the arguments in
        read_args, for calls from the model and from scripted helpers alike. Page views, compare, ask, cart, cart
        view, buy and refuse tools are ignored, as are failure results and results that are not containers; the query
        is also registered with query_is_new."""
        if name in self.page_tools() or name in {self.roles.get(r) for r in ('ask', 'cart', 'cart_view', 'buy', 'refuse')}:
            return
        self.query_is_new(name, args)
        if isinstance(seen, (dict, list)) and not _failed(seen):
            self.read_calls[self.call_mark(name, args)] = seen
            self.read_args[self.call_mark(name, args)] = (name, dict(args))

    def query_is_new(self, name: str, args: dict[str, Any]) -> bool:
        """Registers a call's non-integer arguments (booleans included) for the tool and reports True only the first
        time that combination appears. Integer arguments such as page numbers are left out, so the same words on
        another page do not count as new."""
        mark = _dump({k: v for k, v in args.items() if isinstance(v, bool) or not isinstance(v, int)})
        seen = self.page_queries.setdefault(name, set())
        fresh = mark not in seen
        seen.add(mark)
        return fresh

    @_soft('')
    def last_page_first(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Detects a search tool's hidden page parameter when a new search opens beyond page one. It acts when the page
        parameter is still unknown, exactly one integer argument sits above its minimum (or above 1), more than the
        tail plus 4 steps remain, the learned shape shows a total and a next key, the next value is empty and the
        total exceeds the rows shown. The same call is then repeated with that argument at its minimum; if the reply
        holds no fewer rows and none of the earlier ids, the argument is recorded in page_args and the first page is
        returned with an explanatory note. The probe costs a step even when the guess fails; returns '' otherwise or
        on error."""
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
        """Reports whether any string argument contains a product identifier (model number or part number) matching one
        harvested from seen listings (model numbers, alternate names, compatibility lists). Errors yield False."""
        said = {c for v in args.values() if isinstance(v, str) for c in _codes(v)}
        if not said:
            return False
        known: set[str] = set()
        for mark in self.fit_index()['marks'].values():
            known.update(mark.get('codes') or [])
            known.update(mark.get('works') or [])
        return bool(said & known)

    def first_look(self, name: str, args: dict[str, Any], fresh: bool) -> dict[str, Any]:
        """Resets the page argument of a fresh query to its schema minimum (or 1) when the page parameter of this tool
        has been learned, so a new search never starts part-way through its results; repeated queries and non-integer
        page values pass unchanged."""
        key = self.page_args.get(name)
        if not key or not fresh or not isinstance(args.get(key), int):
            return args
        props = (self.schemas.get(name) or {}).get('properties') or {}
        low = props.get(key, {}).get('minimum') if isinstance(props.get(key, {}).get('minimum'), int) else 1
        return {**args, key: low}

    @_soft('')
    def shorter_search(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """When a value-search query of three or more words returns an entirely empty result, retries automatically with
        the first two non-code words and without integer arguments. It runs only for the value-search tool, with more
        than the tail plus SHORT_KEEP steps left, and only when the result has no non-empty list or string and no
        positive number at its top level. Returns a note with the retry's result, which costs one step, or ''."""
        plan = self.value_template()
        if not plan or plan[0] != name or self.left() <= self.tail() + SHORT_KEEP or self.finished or not isinstance(observation, dict):
            return ''
        field = plan[2]
        text = args.get(field)
        if not isinstance(text, str) or self.records_in(observation) or len(text.split()) < 3:
            return ''
        for value in observation.values():
            if (isinstance(value, (list, str)) and value) or (isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0):
                return ''
        short = ' '.join([w for w in text.split() if not _codes(w)][:2])
        if not short:
            return ''
        faults: list[str] = []
        call = _shape({k: v for k, v in {**args, field: short}.items() if isinstance(v, bool) or not isinstance(v, int)}, self.schemas.get(name), 'args', faults)
        if faults:
            return ''
        reply = self.act(name, call)
        if reply is None:
            return ''
        seen = reply.get('observation')
        self.observe(name, seen)
        self.trace.append({'kind': 'shorter', 'from': text, 'to': short, 'rows': len(self.records_in(seen))})
        return f'\n[Cairn] Nothing has all those words; the same search with {short!r} shows:\n' + self.show(seen)

    @_soft('')
    def first_moves(self) -> str:
        """Opening routine for tasks with at least OPEN_ITEMS items and OPEN_MIN_STEPS steps: it issues the first
        searches and listing reads itself so the opening LLM turns are not spent on them. Store calls are spaced at
        OPEN_GAP during the routine, and a step ceiling keeps the tail, free messages, OPEN_LATER steps per item and
        half of HUNT_KEEP in reserve. Each item is searched with up to two short plain requirement values plus its
        noun and then with the noun alone, stopping early once it has OPEN_EACH rows; a query with rows is remembered
        in searched and may get one extra page through more_rows_for_keys, while refusals and empty answers are
        noted. When rows exist and questions can be sent, widen is repeated until each found item has OPEN_EACH fully
        read records, a share of max_steps is used, the ceiling is hit or no progress is made. Returns the search
        texts, every fully read listing via page_text and an evidence summary; '' when no value template exists or
        the task is too small."""
        plan = self.value_template()
        if not plan or self.max_steps < OPEN_MIN_STEPS or len(self.items) < OPEN_ITEMS:
            return ''
        tool, base, field = plan
        self.env_gap = OPEN_GAP
        shown: list[str] = []
        stop_at = self.step + self.left() - self.tail() - self.msgs_free() - OPEN_LATER * len(self.items) - HUNT_KEEP // 2
        for number, item in enumerate(self.items, 1):
            wanted = [m.split(':', 1)[1].strip() for m in item.musts if ':' in m and m not in [need for _, need in self.refs]]
            wanted = [w for w in wanted if w and len(w.split()) <= 3 and not re.search(OPEN_NOT, w, re.I)]
            noun = item.name if len(item.name.split()) <= 3 else ' '.join(item.name.split()[-2:])
            tries = list(dict.fromkeys(([' '.join(wanted[:2] + [noun])] if wanted else []) + [noun]))
            for query in tries:
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
                    shown.append(f'{query}: ' + self.show(seen))
                    shown += self.more_rows_for_keys(tool, args, seen, stop_at, number)
                    if self.rows_of_item(number) >= OPEN_EACH:
                        break
                else:
                    shown.append(f"{query}: {'the store refused the call: ' + self.show(seen) if _failed(seen) else 'no rows'}")
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
        shown += ['read in full: ' + self.page_text(ident, record) for ident, record in self.full.items()]
        proof = self.evidence(list(self.full.values()))
        return '\n'.join(shown + ([proof] if proof else []))

    def page_text(self, ident: str, record: dict[str, Any]) -> str:
        """Formats an opened listing as prompt text: its id, its attribute block clipped to PAGE_ATTR_CHARS, then its
        other top-level scalar fields clipped to PAGE_REST_CHARS. Nested dicts and lists other than the attributes
        are omitted."""
        block = _fact_block(record) or {}
        rest = {k: v for k, v in record.items() if v is not block and not isinstance(v, (dict, list))}
        return f'{ident}: attributes ' + _dump(_clip(block, 120))[:PAGE_ATTR_CHARS] + ' | ' + _dump(_clip(rest, 200))[:PAGE_REST_CHARS]

    def more_rows_for_keys(self, tool: str, args: dict[str, Any], seen: Any, stop_at: int, number: int) -> list[str]:
        """During the opening routine, fetches one extra page of a search when a next page exists, the item still has
        fewer than OPEN_EACH rows, and the step ceiling and episode allow it. Returns a one-element list with the
        page text, or []."""
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
        """Processes a fresh store result: merges its listings through absorb, then extracts free-text sentences outside
        listing records that have not been seen before. When there are some, a light model interprets them through
        take_note, which may also update requirements, budget and availability, and a non-empty interpretation is
        queued for the model tagged with the tool name. Returns that interpretation or ''."""
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ''
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def widen(self) -> None:
        """Reads more listings in full, spread across items, ahead of the interview. With a compare tool whose only
        required parameter is an id array and which is not known to return bare rows, it picks unread ids round-robin
        from item_groups (groups with the fewest reads first) up to the array limit or compare_room, provided more
        than 12 steps remain; a single pick goes through fill_facts instead. After the compare call it records
        whether attributes came back and how many ids the tool accepts, falls back to widen_singly when the rows lack
        attributes, and retries with two ids if nothing new got read. Without such a compare tool it goes straight to
        widen_singly."""
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
        """Opens listings one at a time with the per-listing view tool to fill each item's attribute picture. For each
        item group, unread buyable rows are ranked by how many requirement-value words their text holds (price breaks
        ties for cheapest picks) and limited to WIDEN_EACH (4 when there are three or more items) minus those already
        read. Each group's leading picks, up to four less its existing reads, are interleaved and opened first (at
        most eight, keeping 22 steps), and the rest follow up to ten reads in total while 30 steps remain."""
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

    def ready_to_ask(self) -> bool:
        """Decides whether the interview should start now. It needs an unused interview, an ask tool, free messages,
        more than 8 steps left and a clock before ASK_BY; within the first ASK_TURNS loop turns every item must also
        have a fully read listing with attributes. It then says yes once 3 x items + 1 listings are read in full, or
        once the step count reaches the larger of 8 and a quarter of max_steps."""
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
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
        """Builds arguments for the ask tool that carry `text`. The text goes into a free-text string parameter
        (required ones first), preferring names not shaped like ids, types or formats, then names like message,
        question or body, then the largest maxLength; failing that, into the first array parameter as a one-element
        list or into the first object parameter's first string field. Other required parameters get placeholder
        values from _filler, and None is returned when a placeholder is impossible, nothing could hold the text, or
        schema shaping reports a fault. Callers also probe it with a long dummy text to learn whether the ask tool is
        usable at all."""
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
        """Runs the question phase and returns its summary for the model. It marks the interview as used and returns ''
        without free messages; otherwise it first reads more listings with widen, plans per-item question lanes with
        plan_lanes (sized by the remaining messages, or the list cap in listing mode) and resets the per-lane
        counters. In listing mode the items are split into groups across the free messages and each group gets a
        combined question first; single questions follow through ask_next while messages, more than 6 steps and time
        before QUESTION_BY remain. Afterwards it follows requirements that point at other listings (early_trail),
        runs value searches for items whose questions found something, and bumps the requirement version when answers
        added requirements. The summary lists each Q/A pair, the new firm requirements with an instruction to apply
        them, guidance for requirements whose value comes from another listing, the trail and search results, and any
        queued notes, which it then clears. Every message is about one item: the combined several-item messages of
        the base are not sent."""
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
        groups: list[list[int]] = []
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
        """Spends one extra shopper message just before checkout to uncover a hidden requirement the interview never
        reached. It runs at most once per episode, and only when an ask tool exists, the message limit is not yet
        used up, no order-stage process step needs messages, at least LATE_STEPS steps remain, the clock is below
        LATE_BY seconds and the cost budget still allows a judge model. If no question lanes were planned earlier,
        they are planned here (a full list when lists are on, otherwise one label per question) with fresh found,
        spent and used counters. The question itself goes through ask_next with LATE_STEPS - 2 as the step floor, and
        a budget change in the reply is reported too. If the reply adds nothing the result is ''; otherwise the
        version is bumped and the loop model gets the question and answer, the new firm requirements and an
        instruction to recheck and repair the cart before buying, plus guidance for needs whose value must be read
        from another listing and the output of a value search for that item."""
        if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit:
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
        """Picks the free-text field of the search tool to reuse for a new word search, modelled on the latest earlier
        search that put letters in that field. Candidates are non-enum string properties, preferring those whose
        names look like a query; when no name looks like one, a field qualifies only if past searches gave it at
        least two different values, at least one of them several words long, and never a store category label. The
        base arguments keep just the enum and required arguments of that search, lose optional strings that echo an
        item, shelf or category name, and set every enum to the value most worded searches used. Gives (tool, base
        args, text field), or None when no earlier search fits."""
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
        """Lists the search phrases that one item's live requirements can supply, each paired with a flag telling
        whether a shopper answer (rather than the request) provided it. Needs that are not in 'property: value' form,
        needs whose value is read from another listing, and values phrased as negations, comparisons, ranges,
        alternatives or references to a page are skipped. A minimum such as 'at least 40 cm' contributes only its
        amount and unit, and only when it came from a shopper answer, because searching the bare figure tends to
        surface the listing that meets the hidden bound."""
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
        """Works out which search call a value search should imitate. search_template() is used whenever it finds
        something or there is no search tool. Next comes a search_words hint in the brief: its tool, text field and
        fixed arguments are taken when the field is a free string and the arguments pass the schema. Failing that,
        the search tool must expose exactly one query-like text field and at most one usable mode enum, where the
        mode is an option whose name mentions query words or, for a required enum with exactly two options, whichever
        option the last search did not use; required values of the last search fill the remaining base arguments. The
        result is (tool, base args, field); otherwise None, if the tool's parameters are ambiguous: several query
        fields, several modes, or enums without a mode to choose."""
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
        if len(named) != 1 or len(modes) > 1 or (enums and not modes):
            return None
        base = {k: last[k] for k in required if k in last and k != named[0]}
        base.update(modes)
        return (tool, base, named[0])

    def value_search(self, numbers: list[int]) -> str:
        """Runs short word searches made of the requirement values that shopper answers revealed, for each item number
        given, and opens the first suitable rows. It needs a search template and at least VALUE_ROOM spare steps
        beyond the closing steps, spends no more than eight steps and keeps VALUE_KEEP in reserve. An item is skipped
        unless an answer supplied at least one of its values; it then gets at most two queries: all its values
        together (one-word values and amounts first), then the same set minus one value (the last one taken from the
        request, or the first one when all came from answers); a single short value is first combined with the item's
        name. Hits count only when buyable, filed under a category the item owns and priced within the budget, a cap
        that is dropped when every hit costs over ten times it. Rows from the item's home label are ranked first, up
        to three not yet opened are read through fill_facts, and the item is done as soon as one query produced hits.
        Returns a bracketed report of queries, row ids and opened pages for the loop model, or '' when no search ran."""
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
        """Gives the normalised category label under which an item's own listings sit, or '' when there is no category
        key or no owned label. Opened records are tried first and search rows only if no opened record carries an
        owned label; within the pool, the shelf the request named, or the item's name, is chosen when it occurs,
        otherwise the owned label seen most often. value_search uses this to put rows from the item's main category
        ahead of neighbouring categories."""
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
        """Splits the items that still have question lanes into as many consecutive groups as there are free messages,
        so that each item is named in some message when there are more items than messages. Items keep the plan order
        (hinted items first) followed by the others; group sizes differ by at most one, with the bigger groups first.
        An empty list means no grouping is needed: either no message is free or every item can get a message of its
        own."""
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
        """Sends one combined message covering every item in the group with a few labels not yet asked or told, then
        reads the reply. Each item's share of labels is the current list size split across the group, but at least
        MIX_LABELS once lists have reached full size. When only one item still has labels, it falls back to an
        ordinary single-item question. Otherwise the text reads like 'What a, b or c would you like for the X, and
        what d or e for the Y?', and the per-item label count shrinks, then trailing items are dropped, until the
        messaging schema carries the text whole. After sending, each named item's spent counter and the list counter
        go up, the reply is read with the 'don't mind' part discarded, and every item that gained needs is settled
        with its labels. A complaint about too many questions with no need disclosed turns lists off for the episode,
        while a reply that discloses nothing only spends the leading label of the first item. The delivered batch is
        returned, or an empty list when nothing went out."""
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
        """Processes the reply to a single-item question that was just sent: the message is counted against the item and
        the list counter, the light model reads the answer first, and the needs found are stored through take_heard.
        For a question naming several labels the 'don't mind' entries are thrown away, because such a reply cannot be
        pinned to one label. Every item that gained a need has its found counter raised, and settle() then updates
        the asked item's lane, including whether the shopper objected that the question asked too much. A list reply
        that states nothing and asks, in so many words, for one question at a time counts as a refusal even when the
        reader returned nothing; a list reply that states a preference, a dropped requirement or a property the
        shopper does not mind, without a refusal, proves that this shopper takes lists for the rest of the episode."""
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
        """Upper bound on how many labels one question may list: LIST_SIZE when the request has a single item, the
        smaller LIST_MIXED when several items share the messages."""
        return LIST_SIZE if len(self.items) == 1 else LIST_MIXED

    def list_now(self) -> int:
        """Number of labels the next listed question should carry. For a one-item task, until the shopper has answered a
        list (a preference or a don't-mind, without a refusal), a question is a short probe of LIST_PROBE labels;
        after such an answer the full one-item size applies. A task with several items keeps the short ramp (three,
        three, then the several-item size). Until lists are proven, the last free message, the late question and any
        question asked within 20 seconds of the questioning cut-off carry one label, so a strict shopper can still
        answer them."""
        steps = LIST_STEPS if len(self.items) == 1 else LIST_STEPS_MIXED
        return min(steps[self.list_asks], self.list_cap()) if self.list_asks < len(steps) else self.list_cap()

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        """Chooses the item and the labels for the next planned question from the lane state passed in. Items that have
        not yielded any need yet come before items that have; then the item asked least goes first, unless messages
        are scarce (SCARCE_MESSAGES or fewer with lists off), then plan order, then item number. The first item that
        still has a label neither asked nor already told gets up to list_now() labels when lists are on, otherwise
        just one. (0, []) means no label is left for any item."""
        first = {n: i for i, n in enumerate(self.plan_order)}
        scarce = self.msg_limit <= SCARCE_MESSAGES and (not self.listing)
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number] and (not self.told_about(number, label))]
            if left:
                return (number, left[:self.list_now() if self.listing else 1])
        return (0, [])

    def told_about(self, number: int, label: str) -> bool:
        """Checks whether the shopper already stated a need for this label on the given item, so the same thing is not
        asked twice. Words are compared with plural endings and British -our spellings folded; a told property covers
        the label when the label's words all appear in the told name, or when a multi-word told name appears entirely
        within the label. A told name made of one word is not enough to cover a label of several words."""
        mine = _spelled(label)
        return any((_names(mine, _spelled(t.split(':', 1)[0])) for t in self.items[number - 1].told))

    def look_alikes(self, number: int, label: str) -> list[str]:
        """Finds up to LOOK_ALIKES other attribute keys on the item's listings that a shopper might mistake for `label`,
        so a retry can ask about the label apart from them. Key names and their value words are gathered from the
        item's opened listings; a key scores three points for every name word it shares with the label (ignoring
        words that more than two keys contain) plus up to three for value words in common with the label's own key.
        Keys with no overlap are ignored, and ties keep the order in which keys were met."""
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
        """Phrases the question for one item and a list of labels. Two or more labels give 'What a, b or c would you
        like for the X?'; a single label with recorded look-alike keys for that item becomes 'Apart from the ...,
        which <label> would you like for the X?'; anything else is worded by question(), with example values only
        when `shown` is set."""
        if len(labels) >= 2:
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.item_said(number)}?"
        alike = self.apart.get(labels[0], {}).get(number)
        if alike:
            return f"Apart from the {' and the '.join(alike)}, which {labels[0]} would you like for the {self.item_said(number)}?"
        return self.question(number, labels[0], self.samples(number, labels[0]) if shown else None)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """Finds how many labels, taken from the front of the list, fit into one question that the messaging tool's
        schema accepts unchanged. Since an over-long message would be clipped by argument shaping, losing trailing
        labels is preferred to sending a cut-off question. The result is the accepted labels with their question
        text, or ([], '') when even the one-label question is rejected."""
        for size in range(len(labels), 0, -1):
            text = self.ask_text(number, labels[:size], shown)
            args = self.message_args(text)
            if args and _says(args, text):
                return (labels[:size], text)
        return ([], '')

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Asks the shopper the next question and processes the reply. Queued follow-ups, minus any an answer already
        covered, take priority over the planned lanes chosen by next_ask(). If the question does not fit the message
        schema, a version without examples is tried; if that also fails, the label is marked as asked and the next
        candidate is tried, up to LIST_SIZE times. The message is sent only while more than `floor` steps remain,
        after which finish_ask stores the new needs (appending each to `told`) and settles the labels. The result is
        the list of delivered (item, question, reply) triples, empty if no message went out."""
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
        """Updates one item's question plan after a reply, using what it disclosed about the properties just asked. With
        a single label, the label is marked asked and a complaint about the question's breadth turns lists off; if
        that complaint came with no need, a narrower retry is queued, first by naming only one of the products a
        plural compound name joins, otherwise (only once per episode) by asking about the label apart from its
        look-alike keys. With a list, a complaint and no need turns lists off and re-asks the first label alone.
        Otherwise the earliest label the new needs map to is found: that label and all earlier ones are marked asked
        and the later labels move to the front of the lane. When nothing maps, just the leading label is spent and
        the other listed labels return in reverse order. Every second miss in a row instead orders those labels from
        the middle outward and puts them behind the lane's labels that have not been named yet, so fresh properties
        get tried. A refusal of a list turns lists off even when the same reply states a preference; the single
        question that follows a refused list takes a label no earlier answered list named, since those are known not
        to be what the strict shopper tracks."""
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
        """Locates the first label of a listed question that the reply's new needs correspond to. For every new need,
        labels are scored by the overlap between their words and the need's property name, spelling variants folded;
        when all score zero, scoring falls back to the overlap between the need's value and the values recorded under
        each label's key on opened listings. The lowest index any need lands on is returned, and -1 means no need
        could be placed, so the reply is handled like an empty one."""
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
        """Orders an item's attribute keys from the key table into fallback question labels. Keys come from categories
        the item owns (or, for a single-item request with none, from all categories) and are ranked by how many
        listings carry them, keys whose values are mostly numbers with short units after descriptive ones, and keys
        with more distinct values earlier. Keys the request already settles are removed; plain property keys come
        first, then catch-all keys such as features or specifications, and last the keys that are not real properties
        (ids, SKUs, codes, warranty, images, condition, variants), although a key like 'phone model' still counts as
        a property."""
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
        """Hands unclaimed store category labels to an item when only one item could own them. With two or more items
        and a known category key, part_shelves() first corrects shared shelves, then the labels of opened listings
        are collected; if some labels belong to no item and exactly one item has no label of its own, that item
        adopts all the strays. This keeps an item whose store category is worded unlike its name from looking absent
        from the store."""
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
        """Maps a store category label to the number of the item it belongs to, or 0 for an empty label or one sharing
        no root word with any item. Adopted labels are returned as recorded; otherwise an exact match with an item's
        shelf scores 3, an exact match with its name 2, and anything else the ratio of shared to combined root words
        between the label and the item's shelf plus name. The top score wins, the earlier item on ties, and the
        answer is cached per label and item count."""
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
        """True when a non-empty category label belongs to the given item, either because owner_of picks that item or
        because the label is exactly the item's requested shelf. This decides throughout the agent whether a listing
        is a candidate for an item."""
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1 or _words(label) == _words(item.shelf or ''))

    def records_of_item(self, number: int) -> int:
        """Counts opened listings that have an attribute block and sit under a category the item owns, 0 when there is
        no category key. plan_lanes divides key counts by this number to get each key's share of the item's listings."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        """Builds each item's ordered list of question labels, its lane. A model (heavy first) receives the PLAN prompt
        with the question count, the request, what the request settles for each item, any properties the brief says
        to ask first, and a numbered table of attribute keys on opened listings with counts, sample values,
        not-stated notes and a flag on catch-all keys. Every pick is cut down to its first property (at 'and', 'or',
        slashes, commas, semicolons or brackets), discarded if empty or longer than five words, de-duplicated, and
        marked as hinted when the model gave a hint, which also enters the item into the plan order. Per item the
        picks are respelled to the item's own key names; then up to three of its most common non-catch-all keys are
        guarded into the first three places: a key takes the slot of the first unhinted pick whose key share is at
        least GUARD_GAP lower (on multi-item requests that pick moves to fourth place, on single-item requests the
        key is inserted ahead of it) or fills a short lane, but never at the cost of pushing a hinted pick out of the
        top three. The remaining keys from lane_keys() are appended and catch-all labels moved to the end. Returns a
        mapping from item number to labels."""
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
                """Gives the item's own key name that equals a planned label once a trailing plural s and the -our
                spelling are ignored, so the lane uses the store's wording; the label comes back unchanged when no
                key matches."""
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                """Fraction of the item's opened listings that carry the key behind a label: the exact key's share when
                it exists, else the largest share among keys whose stemmed names contain or are contained in the
                label, else 0."""
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
        """Splits an item name into separate products when the name is a list of plural nouns separated by commas or the
        word 'and' (for example 'drawer pulls and knobs'). Every part must end in a plain plural s after at least
        three letters, not in -ss or -us; when any part fails that test, or the name has no 'and', the result is
        empty, which keeps single products described with 'and' intact."""
        name = self.items[number - 1].name
        if not re.search('\\band\\b', name, re.I):
            return []
        parts = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I) if p.strip()]
        return parts if len(parts) >= 2 and all((re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts)) else []

    def item_said(self, number: int) -> str:
        """Chooses the wording used for the item inside questions. An ordinary name is used unchanged; a name made of
        plural products joined by 'and' is rewritten with 'or', so the shopper does not take it as several questions
        at once. Once the shopper has asked for one thing at a time about this item, just one of the joined products
        is named: the one whose last word appears in the largest number of the item's search rows outside the
        category field, earlier parts winning ties."""
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
        """Forms a one-label question, 'What <label> would you like for the <item>?', adding ', for example A or B' when
        sample values are passed. The example branch reads two samples, so a list holding a single sample would raise
        an IndexError."""
        text = f'What {label} would you like for the {self.item_said(number)}'
        return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def samples(self, number: int, label: str) -> list[str]:
        """Picks up to two example values for a label from the key table: values of the key with exactly that name that
        are non-empty, at most 30 characters long and absent from the text of the item's request-settled needs. Empty
        when no key has that name."""
        want = _words(label)
        settled = ' '.join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            if name == want:
                return [v for v in values if v and v not in settled and (len(v) <= 30)][:2]
        return []

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        """Sends the given questions one at a time through the messaging tool and returns (item number, question, reply
        text) for each one delivered. Sending stops when the episode is finished, no more than `floor` steps remain,
        the message limit is reached, the text cannot be turned into valid arguments, the call fails, or a reply
        reports an error (that reply is still included). Each delivered message is counted against the limit and
        logged in the done record together with the cart state at that moment, in the step list and in the trace, so
        later gate checks can cite it."""
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
        """Applies a parsed reading of shopper replies to the items. New needs are appended to the item's told list and
        to `told` (prefixed with the item number when there are several items), and needs flagged as pointing to
        another listing are added to refs. A dropped property marks every matching request or told need of that item
        as dropped and reports it in `told`; 'don't mind' properties go to the item's free list; and a positive
        numeric budget replaces the current one. Item numbers outside the valid range are clamped."""
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
        """Has a model (heavy, then light, by default) read shopper replies to the listed questions and return JSON with
        needs, drops, 'don't mind' properties, complaints about crowded questions and any new budget. When
        quiet_reply() finds nothing worth reading the model call is skipped and {} is returned; a question tagged
        with item 0 is shown to the model as covering several items. Returns the parsed object, or {} on failure. A
        reply to a list sent before the shopper proved it takes lists is always read, so a short don't-mind answer
        can still prove it."""
        if self.quiet_reply(sent):
            self.trace.append({'kind': 'quiet_reply', 'a': sent[0][2][:80]})
            return {}
        self.hear_reads += 1
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f"{i}. ({f'about item {n}' if n else 'about several items'}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    @_soft(False)
    def quiet_reply(self, sent: list[tuple[int, str, str]]) -> bool:
        """Decides whether shopper replies can be skipped without a model reading because they cannot carry a
        requirement. A reply is quiet only when it holds no digit, no word from a vocabulary built from item and
        shelf names, the property names of known needs, key-table labels and keys, attribute value words of four or
        more letters on opened listings and the question's own content words, and nothing hinting at how questions
        should be put (one at a time, too many and so on). Before any reply has been read by the model nothing is
        skipped, because the first reading reveals the shopper's stance on listed questions, and an error inside the
        check counts as not quiet. Each skip saves a model call worth several seconds of a short episode."""
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
        """Makes sure the process steps the request demands for a stage ('order' or 'end') are done before the closing
        call. It passes at once with (True, '') when the stage has no steps, the episode is over, or the current cart
        and version already passed. Otherwise, for up to GATE_ROUNDS rounds, a model (light first) sees the due
        steps, ledger, cart, numbered record of messages and step calls, recent store notes, messages and steps left
        and the non-closing tools, and reports which steps are done and one call that would do a missing step. Once
        the gate has done something, needs in the model's reading are stored, and either a new need or a hold request
        bumps the version and returns (False, note) so the cart gets fixed before retrying. If the model offers no
        call while a step lacks proof (for order steps that involve messages, proof must come after the last cart
        change), one reminder line is added to the record and the model is asked again with the heavy model first. A
        proposed message is sent as the longest string in its arguments and any other tool is executed and absorbed,
        both added to the report; a round with no call marks the cart state as passed. Every other exit, including
        model failure, an invalid call or used-up rounds, returns (True, report or ''), so the gate never blocks the
        closing call outright."""
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
        """Pulls the readable reply text out of a messaging tool's result. A plain string comes back unchanged; for
        structured results every string of two or more words within four levels of nesting is joined and cut at 800
        characters, and when none exists a 400-character dump of the result stands in."""
        if isinstance(observation, str):
            return observation
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            """Recursive helper that descends through dicts and lists to depth four and appends every string of at least
            two words to the enclosing list."""
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
        """Collects free-text sentences in a tool result that may be store or shopper notes (budget changes, sold-out
        notices) rather than listing data. Only dict results that are not themselves records are scanned: top-level
        strings of five or more words, such strings inside top-level lists of strings, and strings of six or more
        words inside nested dicts that are not records. Lines already heard are left out, so each note is interpreted
        only once."""
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
        """Has a model (light, then spare) interpret store or shopper note lines against the current ledger and applies
        what it reports. The lines are marked heard first; ids reported unavailable join the unavailable set when
        they are known rows, sold-out flags are recorded, and a single sold-out line is remembered in sold_lines, all
        before the note's kind is checked. Only notes classed as a change (or left unclassed) go further: a new
        positive budget, dropped needs (matched exactly, else by property, else recorded as given), added needs
        (un-dropped if needed), unavailable ids and price moves are recorded. With a cheapest-pick brief a drop,
        unavailability or budget change also resets the cheaper-search state. Any change bumps the version, clears
        the recent list and stores the note; the return value is the model's summary followed by the changes in
        parentheses, or ''."""
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
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or '; '.join(changed))
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

    def needs_now(self) -> list[tuple[int, str]]:
        """Lists every requirement still in force as (item number, need) pairs: needs from the request plus needs
        learned from answers or notes, without the ones that were dropped."""
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence(self, observation: Any) -> str:
        """Builds a hint that links listings in a tool result to the current requirements. It finds records with an id
        and an attribute block in the observation, up to six levels deep, and for at most twelve of them lists, per
        live need (at most eight), the stored attribute keys whose names match the need's property, with values cut
        at 80 characters and an item tag when there are several items. The prefix warns that a key about another part
        or aspect settles nothing and that a key with a different name may still hold the property. Returns '' when
        no record or need is present or nothing matches."""
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            """Recursive helper that searches dicts and lists down to depth six for records with an id and an attribute
            block, collecting each new id once and not descending further into a record it has taken."""
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
                hits = [f'{k}={str(v)[:80]}' for k, v in block.items() if _names(_stem(prop), _stem(k))]
                tag = f'(item {number}) ' if many else ''
                if hits:
                    parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]))
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        if not lines:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)

    def cart_text(self) -> str:
        """Describes the cart for prompts: the store's own cart view, clipped to at most 3000 characters, when it was
        fetched after the last cart change, otherwise Cairn's own tally of listing ids and units from the cart calls
        it made."""
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by Cairn's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def held_units(self, ident: str) -> int:
        """Returns how many units of a listing the cart holds. With a current store cart view it looks for an object
        that has the id as a string value and exactly one quantity-like numeric field (qty, quantity, count, units);
        failing that it uses Cairn's own cart record, which defaults to 1 even for an id that is not there."""
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
        return self.cart_lines.get(ident, 1)

    def cart_ids(self) -> list[str]:
        """Returns the listing ids in the cart: with a current cart view, every seen row id whose text appears anywhere
        in the dumped view (a plain substring test); otherwise the ids in Cairn's own cart record."""
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        """Returns the most complete stored record of a listing, an opened page before a search row, as clipped JSON of
        at most 3000 characters, or '(no record seen)'."""
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else '(no record seen)'

    def others(self, skip: list[str], limit: int=30) -> str:
        """Lists other seen listings for a prompt, leaving out the given ids and anything sold out, with at most `limit`
        records of up to 900 characters each. The order is newest first, except under a cheapest-pick brief, where
        opened listings lead and everything is sorted by price with unknown prices last."""
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
        """Builds compatibility marks for every seen listing and caches them until new rows, opened pages or views
        arrive. Each record's text, including 'key value' strings from nested attribute blocks, is parsed by
        _fit_marks for its own part code, model codes, compatible and excluded codes, series and 'part X also fits'
        statements; the latter are pooled into notes keyed by part code. Returns {'marks': per-id marks, 'notes':
        part code to codes}."""
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
        """Returns the device codes a listing is said to work with: those named in its own compatibility text, plus any
        that other records attribute to its part code."""
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or [])
        if mark.get('own'):
            found |= index['notes'].get(mark['own'], set())
        return found

    def item_ids(self, number: int) -> list[str]:
        """Returns the seen listing ids, in the order they were first seen, whose search row sits under a category the
        item owns; empty without a category key."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def link_system(self, part: int, main: int, index: dict[str, Any]) -> tuple[int, int, set[str]] | None:
        """Detects a series coding scheme that ties two bound items together. Each item is tried as the series carrier:
        at least LINK_ROWS of its rows, and no fewer than half, must have series-prefixed codes drawn from at most
        LINK_FAMILIES series, and some row of the other item must mention a series from that set. The result is
        (carrier item, mentioning item, series set), or None if the listings gathered up to now reveal no scheme."""
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
        """Tests, within a detected series scheme, whether the carrier row's series matches a series the other row
        mentions. True or False when both rows refer to the scheme, None if one of the two mentions no series of it."""
        own = set((index['marks'].get(carried) or {}).get('fam') or []) & families
        named = set((index['marks'].get(namer) or {}).get('named') or []) & families
        return bool(own & named) if own and named else None

    def link_faults(self, part: int, main: int, device: str, lines: list[str], index: dict[str, Any]) -> list[str]:
        """Lists series mismatches between the device line and the part lines of one bound pair when a series scheme
        links them. For every line whose series clearly disagrees with the other side, it states both series and
        offers either up to four seen, unsold carrier listings of the wanted series or advice to search for that
        series or switch to another main item. Empty when there is no scheme or no clear mismatch."""
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
            wanted = set(marks[namer].get('named') or []) & families
            good = [i for i in self.item_ids(carrier) if set(marks[i].get('fam') or []) & wanted and not self.sold_out(i) and i != carried][:4]
            faults.append(f"{carried} is of series {', '.join(sorted(set(marks[carried].get('fam') or []) & families))}, while {namer} takes {', '.join(sorted(wanted))}" + (f"; seen listings of that series: {', '.join(good)}" if good else f"; search that series with the item's words, or take another {self.items[main - 1].name} whose series the wanted lines carry"))
        return faults

    def series_for(self, number: int) -> list[str]:
        """Returns up to LINK_SEARCHES series to search by for an item that a series scheme links to another: first
        those found on the cart's lines for either linked item, then the others by how often rows of the two items
        show or mention them. Empty when no bound pair with a series scheme includes the item."""
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
        """Checks that the cart's parts fit its device for every bound (part, main) pair having exactly one device line
        and at least one part line. A series mismatch from link_faults takes precedence; otherwise, when the device
        states model codes, every part line whose compatibility list does not cover them (or explicitly excludes
        them) yields a fault naming what it fits, plus up to four seen alternatives that would fit or, with none, a
        short fit plan. Part lines without any compatibility statement are not flagged. Returns the fault texts for
        the loop model."""
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
        """Summarises which device codes the seen parts support, to steer the choice of device. For each main item with
        bound parts, codes from part compatibility lists (and series codes when the part carries the series) are
        counted per part item over unsold rows, ranked by how many part items and rows back them, and the top six are
        listed with up to two seen devices matching each. Under a cheapest-pick brief the cheapest complete sets are
        appended. The text gets an explanatory prefix unless `short` is set; '' when there are no bound items or
        nothing to report."""
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
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:6]
            devices = {code: [i for i in self.item_ids(main) if code in ((index['marks'].get(i) or {}).get('named') or []) or any((_code_hit(code, c) for c in index['marks'].get(i, {}).get('codes') or []))][:2] for code, _ in ranked}
            text = '; '.join((f'{code}: ' + ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') for code, per in ranked))
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
        """Finds the cheapest complete sets pairing a single main-item device with one fitting part for every bound part
        item, among buyable, priced rows seen so far, without checking other requirements. A part fits when its
        compatibility codes cover the device's model codes and do not exclude them, or when the series scheme
        confirms a match; set totals multiply prices by each item's quantity. Returns the SET_SHOWN lowest totals as
        text, or '' when no set is complete or the computation fails (the soft wrapper catches errors)."""
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
        """Returns the numbers of items that have no line in the cart; it only runs when the count of held lines (units
        above zero) is below the number of items and a category key exists. Items with no seen listing in their
        category are skipped, since the store may not offer them at all."""
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not cat_key or len(ids) >= len(self.items):
            return []
        labels = [str(r.get(cat_key) or '') for r in self.rows.values()]
        held = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids]
        return [n for n, item in enumerate(self.items, 1) if any((self.owns(item, label) for label in labels)) and (not any((self.owns(item, label) for label in held)))]

    def states_needs(self, ident: str, number: int) -> bool:
        """Tests whether an opened listing literally meets every live need of the item: some attribute whose name
        matches the need's property must contain all words of the need's value. A need with an empty value or with
        limit or negation words (not, at least, under, max and the like) makes the answer False, because word
        matching cannot verify it. True when the item has no live needs."""
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

    def fill_short_cart(self) -> bool:
        """Repairs a cart that lacks items just before an order is forced. For each missing item it picks, among
        listings already opened that are buyable, priced, not yet in the cart and literally meet all its needs, the
        lowest-priced one, provided it stays within the budget (per item or for the whole order), and adds it with
        the cart tool. Items in bound pairs are never filled this way. It returns False as soon as an item cannot be
        covered, the cart calls cannot be built, too few steps remain or a call fails, leaving earlier additions in
        place, which tells the caller to decline instead; True once every gap is filled."""
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
        """Gives an item its own store label when two requested items were placed on the same shelf. It applies only
        when no other item has the same name words, exactly one seen category label has the item's name roots and
        differs from its current shelf, and no item already uses that label. After any change the label-owner and fit
        caches are cleared."""
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
        """Returns a seen category label that contains every root word of the item's multi-word name, when the cart
        line's current label does not and is not the item's requested shelf; '' otherwise. line_faults uses it to
        send a line to the category that names the item completely."""
        cat_key = self.category_key()
        mine = _roots(self.items[number - 1].name)
        if len(mine) < 2 or mine <= _roots(label) or not cat_key or (self.items[number - 1].shelf and _roots(label) == _roots(self.items[number - 1].shelf)):
            return ''
        return next((seen for seen in sorted({str(r.get(cat_key) or '') for r in self.rows.values()}) if seen and mine <= _roots(seen)), '')

    def named_label(self, number: int, label: str) -> str:
        """Returns the category label the request gave for the item, as spelled on a seen row, when the line's label is
        a different one; '' when the request named no shelf, no row carries it or the labels already agree."""
        item = self.items[number - 1]
        cat_key = self.category_key()
        wanted = _words(item.shelf or '')
        if not wanted or not cat_key or _words(label) == wanted:
            return ''
        here = next((str(r.get(cat_key)) for r in self.rows.values() if _words(str(r.get(cat_key) or '')) == wanted), '')
        return here

    def line_faults(self) -> list[str]:
        """Lists cart problems for the loop model to fix before an order. For every item that some seen row belongs to,
        it reports a missing line, more than one line belonging to that item alone, a unit count other than the
        requested quantity, and a line taken from a weaker label when the store has one naming the item fully or the
        one the request named. When every item is known, lines matching no requested item are flagged for removal,
        and sold-out lines are always flagged. With all prices known, a per-item budget flags lines above it and a
        total budget flags an excessive total; overshoots beyond ten times the budget are ignored as likely unit
        mismatches."""
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
                faults.append(f'item {n} ({item.name}) has no line in the cart')
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
        """Finds cart lines whose unit count is the only thing wrong. A line qualifies when it holds a positive number
        of units, its category label belongs to exactly one requested item, it is the only cart line for that item,
        it is not sold out, and the units held differ from that item's requested quantity. Returns (listing id, units
        held, units wanted) triples, or an empty list when the store's category field is not known yet. fix_counts
        takes this list and corrects the quantities without involving the model."""
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
        """Sends the cart calls that bring each (id, held, wanted) line from count_fixes to its requested quantity. It
        reads the cart tool's schema for the listing-id text field, the integer count field and an optional mode
        enum: a set-style tool (see cart_sets) gets one call carrying the wanted count, while a tool with a mode gets
        a removal of the held units (using the option whose name mentions remove, delete, drop, clear or take, else
        the second option) followed by an add of the wanted units. Nothing is sent when the schema has no id or count
        field, when any line can be fixed neither way, or when the calls plus three spare steps do not fit in the
        steps left. Each call is shaped against the schema, sent, applied to Cairn's cart record and observed; the
        loop stops at a shaping fault, a failed call or the end of the episode. Returns one 'args -> reply' line per
        call made and logs a count_fix trace entry."""
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
        """Builds the memo key for an independent cart check: the listing ids sorted and joined with '|', then '#' and
        the current state version. The same set of lines under an unchanged version gives the same key, so check_cart
        judges it only once, while anything that bumps the version (a new requirement, a sold-out line, a moved
        price) makes the set eligible for a fresh check."""
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        """Asks a judge model whether a cart is exactly right before a checkout goes through, and returns a hold message
        when the judge says it is not. Given ids, those listings are judged as a planned cart at the requested
        quantities; without ids the live cart is judged. It returns '' at once when this set of ids was already
        judged under the current version, no judge fits the spending limit, or there are no ids, so a second checkout
        of an unchanged cart is not held again. The prompt carries the request, the requirements ledger, the cart,
        every line's record (flagged SOLD OUT where known) and, under a cheapest, most or least pick rule, other seen
        listings to compare against. To save cost, when both models are affordable, a plain check of the live cart
        before any audit and without such a pick rule goes to the light model first and reaches the heavy one only
        when the light model objects or gives no answer; otherwise the heavy model is asked first with the light one
        as fallback. The verdict is stored in verdicts: True only for an answer that does not say ok=false, so a
        missing answer counts as unconfirmed even though it does not hold the checkout. On a failing verdict the
        judged ids leave leads, and the message lists up to five problems plus up to four suggested replacements that
        are known and still in stock."""
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
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
        judges = self.judges(limit)
        got = None
        if len(judges) > 1 and not lead and not self.audited and self.brief.get('pick') not in ('cheapest', 'most', 'least'):
            got = self.ask_json(prompt, [LIGHT], 3000)
            if got:
                judges = [HEAVY] if got.get('ok') is False and HEAVY in self.judges(limit) else []
        if judges and not self.no_credit:
            got = self.ask_json(prompt, judges, 3000) or got
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get('problems'))][:5]
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def refuse_trail(self) -> str:
        """Gives a decline attempt a single chance to resolve requirements whose value lives on another listing. It acts
        only on its first call (the refuse_trailed flag is set even when nothing turns up), with at least TRAIL_ROOM
        steps left, before HUNT_BY seconds, and while an in-force requirement of that kind still has no value. It
        runs trail() up to the HUNT_BY deadline and, when that produced a report, returns it together with an
        instruction to search for the values found and buy a fitting listing, keeping the no-purchase ending for the
        case where nothing fits; otherwise ''."""
        live = {need for _, need in self.needs_now()}
        if self.refuse_trailed or self.left() < TRAIL_ROOM or self.clock() > HUNT_BY or not any((need in live and need not in self.ref_values for _, need in self.refs)):
            return ''
        self.refuse_trailed = True
        text, _ = self.trail(HUNT_BY)
        return text + 'Search for those values now and buy what fits; end without a purchase only if nothing does.' if text else ''

    def early_trail(self) -> str:
        """Runs the chain resolver ahead of checkout; the interview calls it after each question round. It does nothing
        unless an in-force requirement takes its value from another listing and that value is still unknown, at least
        TRAIL_ROOM steps remain and the clock is within HUNT_BY. The resolver gets a fresh round counter capped at
        EARLY_TRAILS, and the counters the checkout relies on (rounds used and holds given) are saved and restored
        afterwards, so this pass does not eat into what the buy-time check may spend. Values found stay in
        ref_values. Returns the resolver's report text (its hold flag is ignored) and logs the steps left with up to
        three values found."""
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

    def page_read(self, ident: str) -> bool:
        """True when the listing id is in the opened set and, if the store has a separate offer tool that takes a single
        listing id, that tool has also shown this listing (its name is recorded in views for the id). Without such a
        tool, being opened is enough. The chain resolver counts a link as seen only under this test, because a search
        row or a view that shows properties but not the offer text can miss the description that leads to the next
        listing."""
        page = self.roles.get('offer')
        return ident in self.opened and (not page or not self.single_target(page) or page in self.views.get(ident, ()))

    def trail(self, stop: float | None=None) -> tuple[str, bool]:
        """Resolves requirements whose value has to be read from another listing, possibly through a chain of several
        pages, and reports what is still missing. Each round sends the light model (with the heavy one as fallback
        when affordable) the unresolved in-force requirements and up to 60 records, newest first: listings whose page
        was read marked OPENED, the rest marked ROW. For every chain in the answer it stores a value, and bumps the
        version, when all links were read and the chain is done or ends without a further link to find (falling back
        to 'as stated on <last id>'); it queues known links whose page is unread for opening; and it records a link
        that matches no seen row as lost, with the description and search words the model gave. When nothing is
        queued but a link is lost it may call find_link (more than 12 steps left, fewer than 3 link searches so far);
        if that shows the shop's page is already open, the report points the model at that page instead. Up to three
        queued links per round are opened with the offer or open tool, one step each, and no more once six pages are
        in the report. Rounds stop when nothing is pending, the round cap is reached (6, or 4 with 15 or fewer steps
        left, or trail_room), spending reaches SPEND_HARD, the stop time passes, nothing can be opened, steps run
        short or the episode ends. Returns (text, hold): text lists the pages opened and the values found; hold is
        True, with instructions to find and open the remaining links, while links are still lost, more than 5 steps
        are left and the hold budget allows (3 holds, or 2 with 15 or fewer steps left)."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < min(6 if self.left() > 15 else 4, self.trail_room or 6) and (self.cost < SPEND_HARD) and (stop is None or self.clock() <= stop):
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                record = self.full.get(ident) or self.rows.get(ident) if self.page_read(ident) else None
                if record and len(records) < 45:
                    records.append('OPENED ' + _dump(_clip(record, 300))[:1400])
            for ident in reversed(self.seen_order):
                if len(records) >= 60:
                    break
                if not self.page_read(ident) or not (self.full.get(ident) or self.rows.get(ident)):
                    records.append('ROW ' + (_dump(_clip(self.full[ident], 120))[:700] if ident in self.full else _dump(_clip(self.rows.get(ident), 80))[:300]))
            prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.ask_json(prompt, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
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
                if chain.get('done') is True and links and all((self.page_read(i) for i in ids)) and str(chain.get('value') or '').strip():
                    self.ref_values[need] = str(chain['value']).strip()
                    self.version += 1
                    continue
                if links and all((self.page_read(i) for i in ids)) and (not str(links[-1].get('find') or '').strip()):
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
                if faults or self.rushed():
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.observe(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.show(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[Cairn opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (text, hold)

    def not_maker_keys(self) -> tuple[Any, ...]:
        """Returns a 3-tuple of the id key, the category key and the title key; any of them may be None when not yet
        known. The shop-matching helpers ignore these fields when they gather candidate seller or brand strings from
        a row, so an identifier, a category label or a product title is never taken for the shop that a requirement
        describes."""
        return (self.id_key, self.category_key(), self.title_key())

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        """Picks, from the given ids, the listings not yet opened whose row names the shop or brand mentioned in a
        description. A row matches when one of its short text values (3 to 40 characters, outside the id, category
        and title fields, and not a placeholder such as 'none' or 'unknown') occurs in the description as a whole
        word or phrase, ignoring case. find_link uses it to list unopened candidates for a link such as 'the listing
        sold by <shop> with ...'."""
        not_names = self.not_maker_keys()
        low = about.lower()
        out = []
        for ident in ids:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and k not in not_names and v.strip().lower().replace('/', '') not in NO_NAME]
            if ident not in self.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def opened_shop(self, about: str) -> list[str]:
        """Applies the same shop-name test as shop_rows to the listings already opened: returns each opened id whose row
        carries a short text value (3 to 40 characters, not an id, category or title field, not a placeholder) that
        occurs as a whole word or phrase in the description. find_link uses it to tell whether the shop a chain names
        already has a page in hand."""
        not_names = self.not_maker_keys()
        low = about.lower()
        out = []
        for ident in self.opened:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and k not in not_names and v.strip().lower().replace('/', '') not in NO_NAME]
            if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def find_link(self, about: str, words: str) -> list[str]:
        """Searches for the listing a chain link describes when no seen row was matched to it. It first derives the
        value words (from the search words if given, else from the description: words of three or more characters or
        plain numbers, minus filler words and minus every word in the short text fields of rows already matched to
        that shop); when an opened page of that shop contains all of them, it logs a link_known trace entry and
        returns an empty list so the caller can send the model to that page. Otherwise it reuses the rows remembered
        for the same search words, or runs up to two store searches built from value_template() (the full words, then
        their first two words) until one returns rows, counting each in link_searches and remembering the ids it saw.
        Candidates are the unopened rows of the named shop from that search, or failing that from all seen rows
        newest first, kept only when the row shares a description word. Returns at most two ids to open."""
        not_names = self.not_maker_keys()
        known: list[str] = []
        named = {w for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and k not in not_names and v.strip().lower().replace('/', '') not in NO_NAME for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
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
        plan = self.value_template()
        memo = ' '.join(words.lower().split())
        if memo and memo in self.link_seen:
            known = self.shop_rows(about, self.link_seen[memo])
        elif plan and words:
            tool, base, field = plan
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
                self.link_seen[memo] = [i for i, _ in self.records_in(observation)]
                known = self.shop_rows(about, self.link_seen[memo])
                self.trace.append({'kind': 'link_search', 'words': query[:80], 'rows': known[:4]})
                if self.link_seen[memo]:
                    break
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            """Filters ids to the rows whose serialised record contains at least one of the description's words
            (shop-name and filler words removed); when that word set is empty every id passes. It keeps find_link
            from offering a shop's listing of an unrelated kind."""
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

    def offer_tool(self) -> str | None:
        """Name of the tool used to read a listing's offer: the offer role when the store has one, else the open role,
        else None. reread_offers and the offer bookkeeping in run_calls depend on it."""
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        """Re-opens cart lines whose offer was last read before the current window, to catch price and stock changes the
        store makes without announcing them. The window starts two steps back or at the last refused checkout,
        whichever is later, and lines already in gone are skipped. It does nothing without an offer or open tool that
        takes one listing id, when no line is stale, or when fewer than (stale lines + 5) steps remain. Each re-read
        costs a step, updates offer_seen and is observed, which can mark a line unavailable; it stops early when time
        is short, shaping fails, a call fails or the episode ends. Returns one line per listing whose scalar row
        fields changed, naming those fields and showing the clipped reply, and bumps the version when anything
        changed; '' when nothing moved."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if not stale or self.left() < len(stale) + 5:
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
        """True when a listing can no longer be bought as far as Cairn knows: it is in unavailable (a stock flag that
        turned false, a store note, or find_sold's guess) or in gone (a cart line already handled as sold out). A
        plain lookup with no side effects."""
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        """Deals with cart lines that have newly become unbuyable and writes the instruction the model receives about
        them. Each such line (sold out but not yet in gone) is moved into gone at the current step and removed from
        leads, and the version is bumped, so the warning is given once per listing; replacements() then proposes
        stand-ins among listings already read. The text says the line will not return and must not be reopened, then:
        with six or fewer steps left, add the first stand-in and remove the dead line in this turn; under a cheapest
        pick rule, run one search below the first stand-in's price before swapping unless the whole fitting set was
        seen; otherwise swap in the first stand-in; with no stand-in, remove the line and find another fit. Returns
        '' when no new sold-out line is in the cart."""
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ''
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        text = f"[Cairn] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get('pick') == 'cheapest':
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def find_sold(self) -> str:
        """Works out which cart line a refused checkout or an unnamed sold-out note was about. If a live cart line is
        already known to be sold out it returns '' and leaves the rest to swap_hint. With a single live line it marks
        that line unavailable outright, assuming it must be the one; with several it calls reread_offers and returns
        its report of what changed. Returns '' for an empty cart."""
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any((self.sold_out(i) for i in ids)):
            return ''
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ''
        return self.reread_offers() if ids else ''

    def replacements(self, lost: list[str]) -> list[str]:
        """Asks a judge model which already-read listings could stand in for sold-out cart lines. The prompt holds the
        request, the requirements ledger, the sold-out lines' records, the current cart and up to 25 other seen
        records, newest first, leaving out the lost, gone and sold-out ones; the light model is tried first and the
        heavy one only if that gives no JSON. Accepted ids must be known rows that are not lost, not gone, not
        already in the cart and not repeated, and each keeps the price the model reported (infinity when missing).
        Under a cheapest pick rule the ids are sorted by that price, otherwise the model's order stands. Returns up
        to four ids, or [] when no judge fits the budget or there are no records to offer."""
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
        """Gets a second opinion when the model wants to end without buying. It runs at most once per episode and only
        while a judge fits the budget. It asks audit_call() which seen listings appear to fit each item; if none, it
        opens the unread rows the reviewer flagged as promising (through open_rows) and, when that gained a page,
        asks once more. When candidates come back they are stored per item in lead_items and as one flat list in
        leads, and the return value is a hold telling the model to open them and buy what truly fits, with the
        reviewer's reason attached; otherwise ''."""
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

    def covers(self, lines: list[str]) -> bool:
        """Say whether a set of listings has a line for every requested item, judged by the store's category labels
        (owns). With one item, or before the category key is known, any set counts, so two lines for one item never
        pass for a cart that covered every item."""
        cat_key = self.category_key()
        if len(self.items) < 2 or not cat_key:
            return True
        labels = [str((self.full.get(i) or self.rows.get(i) or {}).get(cat_key) or '') for i in lines]
        return all((any((self.owns(item, label) for label in labels)) for item in self.items))

    def audit_call(self) -> dict[str, Any] | None:
        """Builds and sends the reviewer prompt for audit(). Up to 30 seen listings that are not sold out are included,
        those read in full first as OPENED records and the rest as ROW records, each clipped, below the request and
        the requirements ledger. The light model answers first; when the heavy model is affordable, credit remains
        and the light answer names no fitting listing, the heavy model is asked and its answer replaces the light one
        if it parses. Returns the parsed JSON (expected keys fits, open and reason), or None when there were no
        records or no answer, and logs it in the trace. When both answers came back empty (a reasoning model may
        spend all its tokens before writing), the light model is asked once more at low reasoning effort while the
        call cap and time allow, and that answer is used; a 402 there marks the run out of credit."""
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
        if got is None and judges and (not self.no_credit) and (not self.broke) and self.calls < ALL_CALLS and self.model_room() > CALL_FLOOR + 20:
            self.calls += 1
            message, status = self.call_model(LIGHT, [{'role': 'user', 'content': prompt}], 4000, None, 'low')
            if status == 402:
                self.no_credit = True
            got = _json_obj((message or {}).get('content'))
            self.trace.append({'kind': 'audit_retry', 'answered': got is not None})
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        """Cleans the 'fits' map of a reviewer answer into {item number: listing ids}. Each item keeps at most four ids
        that are known rows and not sold out, and items left empty are dropped; a missing answer or a 'fits' value
        that is not a dict gives {}."""
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        """Reads the pages of rows the reviewer flagged, so a second review can judge them on their attributes. It
        prefers one side-by-side call holding as many ids as the compare view allows, when that view takes a single
        id list and is not known to lack attributes; else it opens the first two rows one at a time with the tool
        that shows attributes; with neither it gives up. It also declines when the steps left would not cover these
        reads plus a reserve of at least five steps, or of the closing steps plus the cart calls needed to put one
        line per item in the cart. Returns True when at least one new full record was gained, and logs an audit_open
        trace entry."""
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
        """Translates a wanted cart (listing id to units) plus ids to remove into argument dicts for the store's cart
        tool, or None when the schema gives no safe way to do it. When a cart property is an array of objects, one
        call replaces the whole cart with every wanted line, using the object's text id field and, if present, its
        integer count field; removals happen by leaving lines out. Otherwise it uses the scalar id field, the integer
        count field and an optional mode enum whose first option is taken as add and second as remove. A removal uses
        the remove option with the held units, a count of 0 on a set-style tool, or a negative count when the
        schema's minimum allows it; an addition skips lines already held at the wanted count and sends the add option
        with the wanted count. Returns None when no id field is found, a removal has no supported form, or a line
        already held needs a different count on an additive tool."""
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
        """Builds the cart and checks out on its own after an independent check confirmed a reviewer lead while the
        model was trying to end without buying. It runs once and needs the confirmed listing to be still buyable plus
        both a cart and a checkout tool. The plan takes the first buyable lead of each reviewer item at that item's
        quantity, one per category label, and must contain the confirmed listing; current lines of other,
        still-needed kinds are kept and the rest removed, and it gives up if a requested item would be left without a
        line. Before touching the cart it reads the planned lines' attributes and linked pages and requires a passing
        check_cart verdict on them. It then sends the cart calls, shapes the fixed checkout arguments and tries the
        checkout at most twice: the second try first re-reads the offers and re-checks the cart, and each try must
        pass the order gate (approvals or messages the request requires). Returns a report of the cart calls and the
        store's replies, which stands in for the model's decline, or '' when it did not check out; note that '' can
        follow cart calls that were already sent."""
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or (not cart) or (not buy):
            return ''
        self.completed = True
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            """Grouping key of a listing: the lowercased, stripped category label from its row, or the id itself when
            the store's category field is unknown. complete_cart uses it so each kind of product contributes only one
            cart line."""
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
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
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
        """True when the call goes to the checkout tool and every fixed argument the brief lists for buying (buy_args)
        has exactly the value sent. On a store whose single finish tool carries a buy-or-abstain choice, this tells a
        purchase apart from a decline made with the same tool."""
        if name != self.roles.get('buy'):
            return False
        fixed = self.brief.get('buy_args') if isinstance(self.brief.get('buy_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        """True when the call goes to the no-purchase tool and matches every fixed argument the brief recorded for
        declining (refuse_args). It mirrors is_buy, so a shared finish tool with an enum switch is classified by the
        switch value rather than by its name."""
        if name != self.roles.get('refuse'):
            return False
        fixed = self.brief.get('refuse_args') if isinstance(self.brief.get('refuse_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def offered(self) -> list[dict[str, Any]]:
        """Chooses the tool list the model sees on its next turn. The shopper-messaging tool is hidden whenever Cairn
        can build a message call for it itself, since the interviewer owns every question. As steps run out the list
        narrows: at the closing reserve only the checkout and no-purchase tools remain; within three steps of it the
        cart and cart-view tools come back; and while no checkout has been tried and the steps left are within the
        item count plus five, the offer tool too. Narrowing applies only when an ending tool exists, and the full
        list is kept if the narrowed one would be empty."""
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
        tools = [t for t in self.tools if t['function']['name'] != ask or self.message_args('x' * 200) is None]
        if keep is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        return tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Rewrites the free-text arguments of a no-purchase call into one neutral sentence saying nothing in the store
        meets every requirement, cut to the field's maxLength. A non-enum string field that is sent or required is
        replaced when it is empty, when it contains a seen listing id, or whenever the brief does not set
        quiet_decline to False (the usual case). This keeps the closing call free of listing ids, questions or other
        text the request did not ask for."""
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
        """Repairs listing references in calls to the cart, open, offer and compare tools when the model passes text
        instead of an id. It acts only when the store's id field is known and the tool has exactly one free text or
        array field that is known to carry ids (always assumed for the cart tool). Each value, or each element of a
        list, goes through fix(): a unique match becomes that row's id, while an ambiguous or unknown text yields a
        message (listing up to six candidates, or naming the id field) and the call is then refused without using a
        step. Returns (repaired args, '') on success or (original args, problem) on the first failure."""
        if name not in {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.id_key:
            return (args, '')
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _kinds(v) or 'array' in _kinds(v)) and ('enum' not in v)]
        if len(free) != 1 or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args):
            return (args, '')

        def fix(value: Any) -> tuple[Any, str]:
            """Maps one value to a listing id. A value equal to a seen id after stripping comes back stripped;
            non-strings, single tokens of 3 to 64 characters without spaces, and texts shorter than 8 characters come
            back unchanged. Other text is compared, lowercased, with the start of every non-id string field of every
            seen row: one hit returns that row's id, while several hits or none return the value with a message
            explaining the problem."""
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
        """Tells whether the cart tool sets a line's units outright instead of adding to them. It is False whenever the
        schema has an enum with two or more options (a mode switch), and otherwise True when some integer field
        declares a minimum of 0, read as 'a count of 0 empties the line'. cart_calls, fix_counts and note_cart pick
        their arithmetic from this."""
        props = self.schemas.get(name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in _kinds(s) and (s.get('minimum') == 0) for s in props.values()))

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Fills in the quantity of a cart call that picks a removal option (any enum value after the first) for a
        listing Cairn records as held but leaves out the integer count field. The count becomes the recorded units,
        capped at the schema's maximum, so the entire line leaves the cart rather than the single unit a store tends
        to take by default. Any other call, or a schema without an integer field, is returned unchanged."""
        spec = self.schemas.get(name, {}).get('properties') or {}
        count = next((k for k, v in spec.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        removal = any((isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items()))
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.cart_lines), None)
        return {**args, count: min(self.cart_lines[ident], spec[count].get('maximum', self.cart_lines[ident]))} if count and removal and ident and count not in args else args

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        """Keeps Cairn's own record of the cart (cart_lines, id to units) in step with a cart call just sent, and stamps
        cart_touched and cart_at so later code knows any earlier cart view is stale. A call counts as a removal when
        an enum argument holds any option after the first. Array entries that are objects, or an empty array of
        objects, mean the whole cart was replaced, so the record is rebuilt from those lines with their counts
        (default 1); id strings, alone or in a list, take the call's loose integer count. Otherwise each line is
        updated in place: a removal subtracts the count (one unit when the tool has a count field but none was sent,
        the whole line when it has none), a set-style tool takes the count as the new total, and an additive tool
        adds it (default 1); lines at zero or below are deleted. Counts come from _unit_count, so a field named for a
        quantity wins over any other whole number on the line."""
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get('properties') or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2 and (value in options[1:]):
                removal = True
        loose_count = _unit_count(args, self.rows, None)
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
                        ident = next((_as_id(v) for v in line.values() if isinstance(_as_id(v), str) and _as_id(v) in self.rows), None)
                        count = _unit_count(line, self.rows, 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(_as_id(line), str) and _as_id(line) in self.rows:
                        lines.append((_as_id(line), loose_count))
            elif isinstance(_as_id(value), str) and _as_id(value) in self.rows:
                lines.append((_as_id(value), loose_count))
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
        """Renders a store reply as text for the model's conversation. Strings inside are clipped to SHOW_CHARS when the
        reply holds listing records and to TEXT_CHARS otherwise, the result is dumped as compact JSON, and anything
        past OBS_LIMIT characters is cut off with a '...(cut)' marker."""
        text = _dump(_clip(observation, SHOW_CHARS if self.records_in(observation) else TEXT_CHARS))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT] + '...(cut)'
        return text

    def compare_target(self) -> str | None:
        """Name of the compare tool's array argument that takes listing ids, or None. It is returned only when the
        compare tool has an array-typed field and requires no other argument, so a call carrying just that list is
        valid."""
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def compare_room(self) -> int:
        """How many listings to send in one side-by-side call. A maxItems declared in the schema wins. Otherwise, once a
        batch has come back empty (compare_bad holds the smallest such size), it is one less than that size, capped
        at the largest batch that worked (counted as at least 2) and never below 2; with no failure seen it is the
        largest working batch, but at least 4."""
        spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.compare_target() or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Folds several single-listing open calls from one model turn into a single side-by-side compare call, saving
        steps. It applies only when the store has distinct open and compare tools, open takes one id, compare takes a
        single id list, and the compare view is not known to lack attributes. Open calls whose only argument is a
        seen, not-gone listing id are collected up to compare_room(); with at least two, the first becomes the
        compare call (keeping its call id) with all those ids, the others become 'joined' placeholders that run_calls
        answers without a step, and every other call stays as sent."""
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
        """Executes one model turn's tool calls and returns (call id, reply text) for each, acting as the gatekeeper
        between the model and the store. Open calls are first merged by join_opens. A call is then refused without
        using a step when the episode is over, it messages the shopper while the interviewer owns messaging, the tool
        is unknown or currently withheld by offered(), its arguments are not one JSON object or fail the schema, an
        id reference cannot be resolved, it opens a listing already gone, it takes units off a listing not in the
        cart, it is a finishing call sent together with others, it is any other call inside the closing reserve, it
        is a code search after code searches proved useless, it repeats one of the last three calls, it is a read
        whose cached result is still current, or it re-reads a view with nothing new to show. Calls are also adjusted
        on the way: removals get the held count, compare lists are trimmed to the known room and to listings with
        something new, new searches start on their first page, and part codes become row ids. A checkout with more
        than two steps left first lets Cairn hunt for cheaper listings and re-read the cart's offers, and is held
        while a check fails: a one-time wider-search hold under a cheapest pick, a cart view that reveals a change, a
        sold-out line, an unresolved pointing requirement, a part that does not fit its device, wrong lines or
        quantities (quantities alone are corrected by Cairn itself), or a failing judge verdict. A decline is held
        for unresolved pointing requirements, the audit's leads (twice, the second time after an independent check of
        the first lead), an earlier full cart worth restoring when few steps are left, or a give-up with more than a
        third of the steps left; a decline that gets through may be replaced by complete_cart's own order, and
        otherwise has its text neutralised by quiet(). Finishing calls also pass the late-question and process gates.
        After a call runs, its reply feeds the row store, learned tool roles and views, the cart record, code-search
        and compare-size statistics and the read cache, and the returned text gains paging, cheapest-row and evidence
        notes, store notes, sold-out handling, fit warnings and an episode-finished marker. A decline with more than
        five steps left, after the cart once held an in-stock line for every requested item (covers), is held once on
        its own flag, naming those lines and asking for their records to be read again; it is not held while a
        confirmed lead waits for complete_cart or when the cart check already failed exactly those lines, and a cart
        reply that reports a fault does not count toward that cart."""
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
            faults: list[str] = []
            args = _shape(args, self.schemas[name], 'args', faults)
            if faults:
                out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6])))
                continue
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            if name == self.roles.get('cart'):
                args = self.whole_removal(name, args)
            trimmed = ''
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
            args = self.row_ids(name, args)
            if name == self.roles.get('search') and self.codes_dead and self.names_code(args):
                out.append((cid, 'Not run (no step used): this store\'s search found nothing for a model or part code. Search by words and values, list the category, and read the codes on the rows.'))
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
                    hunted = self.hunt_cheaper()
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + '\nCart now: ' + self.cart_text()))
                        continue
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
                        out.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.cart_text()))
                        continue
                    faults = self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
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
                    pointed = self.refuse_trail()
                    if pointed:
                        out.append((cid, 'Not run yet (no step used). ' + pointed))
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
                        self.check_cart([first], SPEND_HARD)
                        if self.verdicts.get(self.check_key([first])) and first in self.leads:
                            self.confirmed = first
                            self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                            out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
            if self.is_refuse(name, args) and self.left() > 5 and (not self.early_held) and (not (self.confirmed and (not self.completed))):
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh))]
                if len(lines) >= len(self.items) and self.covers(lines) and len(missing) + 1 <= self.left() and self.verdicts.get(self.check_key(lines)) is not False:
                    self.early_held = True
                    self.trace.append({'kind': 'early_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Earlier in this episode the cart held a line for every requested item: ' + ', '.join(lines) + '. Ending without a purchase earns nothing when the store has a fitting listing, and almost no request has none. Re-read the records of those lines against the requirements in force: a line fails when its record states a different value, or when it leaves out something the shopper asked for. ' + redo + ' End without a purchase only if a record breaks a requirement and your search found no other listing for that item; then send the ending call again.'))
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
            paged = self.more_pages(name, args, observation) if name not in plain and (not ending) else ''
            paged += self.last_page_first(name, args, observation) if fresh and (not ending) else ''
            paged += self.shorter_search(name, args, observation) if name not in plain and (not ending) else ''
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
                if len(lines) >= len(self.built) and (not _faulted(observation)):
                    self.built = lines
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
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if name == self.roles.get('cart') and self.bound and not self.finished:
                misfit = self.fit_faults()
                if misfit:
                    text += '\n[Cairn] These lines do not work together yet: ' + '; '.join(misfit)
            if self.queued:
                text += "\n[notes seen on Cairn's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def can_refuse(self) -> bool:
        """Checks whether ending without a purchase is possible on this store: a no-purchase tool exists and its fixed
        arguments from the brief, after quiet() fills the free-text fields, pass the schema without faults. finish()
        uses it to decide whether an incomplete multi-item cart should still be bought because declining is not an
        option."""
        refuse = self.roles.get('refuse')
        if not refuse:
            return False
        faults: list[str] = []
        _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
        return not faults

    def finish(self, again: bool=False) -> str:
        """Ends the episode on Cairn's own authority when the model loop stops without a finishing call. It does nothing
        once the episode is over, the service is broken or no step is left. With a held cart on a multi-item request
        it first tries fill_short_cart to add missing lines; if that fails it still buys only when a decline cannot
        be sent, and with a complete cart of two or more lines it reads their attributes and, time allowing, their
        linked pages. It then checks out, after the order gate when two steps remain, if the cart holds lines or was
        changed since the last view with two steps left. With again=True a refused checkout is observed, counted and
        returned as text so run() can give the model another round; otherwise, or when there was nothing to buy, it
        sends the quieted no-purchase call after the end gate and returns the store's reply if that ending was
        refused as well. Returns '' when the episode finished or nothing was refused."""
        if self.finished or self.broke or self.left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
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
        if whole and buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            faults: list[str] = []
            args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
            if not faults:
                if self.left() >= 2:
                    self.gate('order')
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
        """Keeps the chat under HISTORY_LIMIT characters of content. When over the limit it shortens the older half of
        the tool replies (at least one): each reply longer than 600 characters is cut to its first 400 and tagged
        with up to 20 quoted listing ids found in it, so the model can still refer to those listings. One pass may
        not bring the history under the limit, and later passes pick the same oldest replies again."""
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
        """Plays one whole episode and returns the trace. It reads the shopper's request into the brief, absorbs the
        store's opening view when one is given, and lets first_moves() search each item and read some listings before
        the model starts. The model then gets the GUIDE system prompt and an opening message with the request, the
        opening view and either those first results or advice to begin by searching, plus the status ledger, and the
        turn loop runs within LOOP_CALLS model turns and SPEND_HARD spending. If the loop ends unfinished,
        finish(again=True) forces an ending; when the store refuses it and the service works, credit remains and more
        than one step is left, the refusal is shown to the model for up to ten more turns under a slightly higher
        spending cap, and a final finish() follows if still needed. An end entry with steps, cost and model calls
        closes the trace; an episode already finished on entry returns its trace at once."""
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        if opening is not None:
            opening = _restore(opening)
            self.absorb(opening)
        moved = self.first_moves() if not self.finished else ''
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
            if not self.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        """Runs the turn loop with crash recovery: up to CRASH_TURNS attempts of turns(loop_cap, spend_cap). An
        exception is logged in the trace (type and message, clipped) and answer_open_calls() mends the chat before
        the next attempt; a clean return ends it. Errors therefore never escape the model loop, and the caller always
        reaches its forced finish."""
        for _ in range(CRASH_TURNS):
            try:
                self.turns(loop_cap, spend_cap)
                return
            except Exception as exc:
                self.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
                self.answer_open_calls()

    def answer_open_calls(self) -> None:
        """Repairs the chat after a crash. It finds the newest assistant message that carries tool calls and adds a tool
        message for each of its calls that has no result yet, warning that the call may not have gone through and
        that the state should be checked before retrying; a user message with the current status follows. Nothing
        happens when no assistant message has tool calls. Chat APIs reject a history in which a tool call lacks its
        result, so without this the next model call would fail."""
        at = next((n for n in range(len(self.chat) - 1, -1, -1) if self.chat[n].get('tool_calls')), None)
        if at is None:
            return
        done = {m.get('tool_call_id') for m in self.chat[at + 1:]}
        for call in self.chat[at]['tool_calls']:
            if call['id'] not in done:
                self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'The call may not have run: something went wrong on the way. Look at the state before you repeat it.'})
        self.chat.append({'role': 'user', 'content': self.status()})

    def turns(self, loop_cap: int, spend_cap: float) -> None:
        """The main model loop. It keeps going while the episode is open, the service works, steps remain, model turns
        are under loop_cap and spending under spend_cap, and either the clock is before LAST_TURN or the cart is
        still empty with more than EMPTY_ROOM seconds of model time left. When the interviewer is ready, a question
        round runs first and its summary is added as a user message. Otherwise the history is trimmed and the light
        model, with the spare one as fallback, is asked for its next move with the tools from offered(); a failed
        call counts as a miss, which ends the loop on lost credit or the call cap and, from the third miss on (misses
        are never reset), either pauses BRIEF_GAP seconds or stops once OUTAGE_BY has passed. Tool calls are
        normalised (missing ids generated, arguments re-encoded as a JSON object). A reply without calls gets a
        nudge, and three such replies in a row end the loop. The first time the model tries to change the cart, buy
        or decline before the shopper has been asked, while messages are free, more than eight steps remain and the
        clock is before ASK_BY, those calls are answered as not run and the interview runs instead. Otherwise the
        calls go through run_calls and each reply is appended as a tool message, the last one carrying the status
        ledger."""
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
                if self.no_credit or self.calls >= ALL_CALLS:
                    break
                if self.misses > 2:
                    if self.clock() >= OUTAGE_BY:
                        break
                    time.sleep(BRIEF_GAP)
                continue
            self.misses = 0
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

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Module entry point the runner calls with the task payload; it returns the episode trace as a list of dicts and
    never raises. If building the Cairn episode fails, the trace is a single crash entry. If run() raises, the crash
    is logged and, when the episode is still open, finish() is tried so the store still gets an ending; a failure
    there is logged as a second crash entry."""
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
