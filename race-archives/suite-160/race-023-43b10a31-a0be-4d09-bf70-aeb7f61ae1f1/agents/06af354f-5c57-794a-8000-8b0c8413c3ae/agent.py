"""Cairn: buys for one shopper inside an ORO store world. Built on Halo v3, the public race-15 top agent.

A light model walks the store with the world's own tools; code keeps the ledger, plans the questions to the shopper
and checks the cart before an order and the evidence before a decline. Beyond the base:
- a message the request kept for the ending is spent on one more planned question when Cairn is about to buy
  (late_question), so a tight message limit no longer leaves a hidden need unasked;
- a clock keeps every episode inside the validator's 300 s limit: no interview, question, loop turn or model request
  starts when its result could not be used in time, and the model timeout shrinks with the time left;
- each message to the shopper names up to LIST_SIZE planned properties of one item while the shopper answers such
  lists, rotating them so that a property the shopper does not mind cannot hide the ones behind it (settle), and
  falls back to one property per message after a reply that asks for one question at a time.
- USE_CHIDORI_ASK swaps that interviewer for ChidoriAsk, Cairn's own code for the design of Chidori's question engine
  (planner over property menus, per-item queues, widths read from the replies, mixed messages, one-call reply reading).
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
USE_ASK_WIDTHS = False  # off (sweep r18 asking 20244: wider lists put the hidden need 3rd and it went unanswered; king's widths kept) # ask: labels per list from how the shopper takes lists (2 first, 3 when it is the only message or on several items; x2 after a list answered without complaint; full after a need told from past a list's first label)
USE_ASK_FRESH_FIRST = False  # off (sweep 20244 t-6b06: the king's rule leads with the list's later labels, which revealed the need; fresh-first demoted it) # ask: after a reply that disclosed nothing, two labels never named lead the next list
USE_ASK_LOST = False  # ask (prompt text): HEAR also reports a reply that lost track; a list of 6+ it lost halves the widths after it
USE_ASK_PLAN_DECOY = False  # ask (prompt text): PLAN ranks the property particular to the kind of product before generic colour/size/pattern/material
USE_CHIDORI_ASK = True  # ask engine rebuilt from the design of Chidori (top of the arena's ask-engine test: r15 0.724, r18 0.641): one planner call over property menus, per-item queues, list widths read from the shopper's replies, mixed messages, one-call reply reading with an exclusion double-check (ChidoriAsk); False: the king's interviewer below
CH_FIRST = 2  # labels in the episode's first list on one item (3 when it is the only free message)
CH_FIRST_MULTI = 3  # the same on several items
CH_CALM = 3  # widest list on one item while no reply has shown the shopper reads past a list's first label (Chidori doubles each time: race 18 run 20244 lost t-a598 on a list grown to 6)
CH_CALM_MULTI = 3
CH_FULL = 6  # widest list once a need was told from past a list's first label (Chidori: 8)
CH_FULL_MULTI = 4  # (Chidori: 6; run 20244 t-a598: a 6-label list on a second item had the need 3rd and drew nothing)
CH_LOST_LONG = 6  # a list this long the shopper did not follow halves the widths after it
CH_TOPS = 2  # most-carried properties of an item kept among its first questions
CH_TOP_MIN = 3  # listings that must carry a property for it to count as most-carried
CH_MENU = 40  # menu entries per item shown to the planner
CH_NARROW_MIN = 4  # messages a task must allow before a single question the shopper turned away is asked again, narrowed
CH_TRIES = 6  # labels set aside in a row (question too long for the message tool) before a message is given up
CH_STEPS_MAX = 2  # process steps a shopper's replies may add in one episode
CH_PLAN_TOKENS = 3000
CH_READ_TOKENS = 1500
CH_KINDS = ('need', 'repeat', 'free', 'split', 'unclear', 'other')
USE_T_FIT_PLAN_REFRESH = True  # time fix (sanity runs 19897-19905): the fit plan goes into the status only when it changes or every STATUS_REFRESH statuses (it went out 30 times, 57k chars, on a 4-item fit task that ran out of clock)
USE_T_PROVEN_RESPECTS_CHECK = True  # #110 items 7/8: never order or swap to a set the final check rejected (sweep 20205 t-175e)
USE_T_E5_NAME_GUARD = True  # E5 fix (sweep 20124 t-8f9f): a written 1 counts, and a longer item's name is not this item's
USE_T_PAGE_TOOL_ANY = True  # #97 item 42 generalized: the page view may be a side-by-side tool (race 18: WeighListings)
STATUS_REFRESH = 4  # statuses after which an unchanged repeated status line is shown again
ASK_FIRST = 2
ASK_FIRST_MIXED = 3
ASK_FULL = 8
ASK_FULL_MIXED = 6
LIST_STEPS_MIXED = (3, 3)  # the same on several items, where each question goes to another item (simulated: 3,3,8 level or above 1,3,8 on 2-3 items)
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
VALUE_ROOM = 12  # steps beyond the finish before a value search runs at all (race 16: limit 35, answers at about step 15)
VALUE_KEEP = 8  # steps a value search leaves beyond the finish, for opening and buying what it found
LOOK_ALIKES = 2  # keys set aside in one question (race 16: one or two)
LINK_SEARCHES = 3  # series searched per item in a decline hunt (race 16: up to 9 series)
LINK_ROWS = 3  # rows of one item carrying a series code before the codes count as a system (race 16 covers: nearly every row)
LINK_FAMILIES = 12  # more distinct series than this are model numbers, not a system (race 16: 9)
USE_E2_FIT_TIE_BY_CODE = True  # #111 E2: fit plan breaks ties between codes by the code's name (no hash-seed order)
USE_E3_KEEP_RELATION_WORDING = True  # #111 E3: a "same as" lookup takes the request sentence's own relation wording when that sentence states no sameness
USE_E4_CONDITIONS_NOT_LOOKUPS = True  # #111 E4: a requirement with a waiver/restriction clause and no page/shop talk is never a pointer lookup
USE_E5_QTY_BEFORE_NAME = True  # #111 E5: quantity 1 becomes the one number written right before the item's full name
USE_E6_QTY_TIMES_FORM = True  # #111 E6: quantity from a count written in multiplication form ("3 x <item words>")
USE_E7_ITEM_PRICE_CAPS = True  # #111 E7: per-item price caps written in the request (ledger, line check, hunts, value search, sets, short-cart filler)
USE_E8_FIGURE_STAYS_FIRM = True  # #111 E8: a requirement whose value holds a digit is never filed as "shopper's own words"
USE_E9_HEAD_OF_LONG_NAME = True  # #111 E9: Cairn's opening search for a long item name uses its head (the part before and/or/&)
USE_E10_POINTER_GIVES_ONLY = True  # #111 E10: a resolved pointer says which property the page gives and that the rest only picks out that page
USE_E11_CONDITION_NOTES = True  # #111 E11: status notes deciding waiver/restriction clauses from the other item's cart line
USE_E12_SHORT_ITEM_KEEPS_TOOLS = True  # #111 E12: no tool narrowing (items + 5 window) while an item has no cart line; status names it
USE_E13_LAST_STEPS_OPEN_SEARCH = True  # #111 E13: last tier keeps the facts tool (>= 3 left) and search (>= 4 left) while an item has no line
USE_E14_ACTOR_TOKENS_8000 = True  # #111 E14: actor reply cap 3000 -> 8000 tokens (request parameter: redraws actor replies)
USE_E15_LATE_LOW_EFFORT = False  # #111 E15: low reasoning effort on tool-offering Anthropic actor turns after EFFORT_AFTER s (request parameter)
USE_E16_WORD_BOUNDARY_CUT = True  # #111 E16: an over-long string argument is cut at a word boundary; the actor is told once per tool and field
USE_E17_PAGE_ARG_FROM_SHAPE = True  # #111 E17: a search's page argument is also taken from the reply-shape reading (integer next_arg)
USE_E18_TOP_ROWS_OF_MIDLIST = True  # #111 E18: a fresh search sent past its first rows also gets its first rows read by Cairn (one step)
USE_E19_SKIP_SEEN_OFFSETS = True  # #111 E19: a repeated search at a row offset already shown is sent at the next unseen offset
USE_E20_NEXT_ARG_FROM_SCHEMA = True  # #111 E20: next-page parameter guessed from the schema when the shape reading names none
USE_E21_DROP_DEAD_FIT_CODES = True  # #111 E21: after a 0-row actor search for a required fit code, later actor searches go without the code
USE_E22_ZERO_COUNT_ADD = True  # #111 E22: a cart add with count 0 for a listing not in the cart is sent with the item's quantity
USE_E23_LATE_KEEP_LINES = True  # #111 E23: with few steps left a turn's cart calls that leave fewer (unsold) lines are not run
BASE_ACTOR_TOKENS = 3000  # the king's actor reply cap
ACTOR_TOKENS = 8000  # #111 E14 actor reply cap (guide's value; the actor's hidden reasoning used all of 3000)
EFFORT_AFTER = 90.0  # #111 E15: episode seconds after which late actor turns ask for low effort (guide's value; 45 s was worse)
QTY_LOW, QTY_HIGH = (2, 20)  # #111 E5: a number in this range right before the item's name is a count (guide's range; larger is a size or code)
TIMES_WORDS = 6  # #111 E6: words after a multiplication-form count that are matched against the item names
CAP_SANITY = 10  # #111 E7: a price more than this many times a cap is read as a unit mix-up, as the base's budget check does
CONDITION_NOTES = 4  # #111 E11: at most this many condition notes in one status
LATE_GUARD_STEPS = 4  # #111 E23: steps left at or below which a turn may not take unsold lines out of the cart
COND_CLAUSE = '\\b(?:unless|except\\s+(?:if|when|where)|only\\s+(?:if|when|where)|provided(?:\\s+that)?|as\\s+long\\s+as|so\\s+long\\s+as|in\\s+case|if)\\b'  # E4/E11: waiver or restriction clause
COND_WAIVER = '^(?:unless|except)'  # E11: the clause words that waive the rule when the condition holds
COND_VERB = '\\b(?:is|are|was|were|be|has|have|uses?|comes?\\s+with|carries|takes)\\b'  # E11: the verb between a condition's subject and its values
POINTER_TALK = '\\b(?:page|listing|listed|points?|pointer|sells?|sold|shop|store|seller|vendor)\\b'  # E4/E11: a clause that speaks about another listing or what a shop sells
SAMENESS = '\\b(?:same|identical|matching|match(?:es|ed)?|equal(?:s|ling)?|alike)\\b'  # E3: wording that claims sameness
AUX_WORDS = '\\b(?:should|must|shall|will|would|could|can|may|might|needs?\\s+to|has\\s+to|have\\s+to|ought\\s+to|is\\s+to|are\\s+to)\\s+(?:be\\s+)?'  # E3: auxiliary before a sentence's main verb
LEAD_FILLER = '^(?:(?:should|must|shall|will|would|could|can|may|might|needs?|has|have|ought|is|are|was|were|be|been|to|that|which|it|its|the|a|an)\\b\\s*)+'  # E3: function words before the relation's verb
JOIN_WORDS = '\\s+(?:and|or)\\s+|\\s*&\\s*'  # E9: words joining two names in one item name
FIT_INTRO = '\\b(?:for|fits?|fitting|fitment|compatible(?:\\s+with)?|works\\s+with|model|code)\\s*:?\\s*$'  # E21: the word introducing a fit code
FLOOR_BEFORE = '(?:\\bno\\s+(?:less|fewer)\\s+than|(?<!\\bno )(?<!\\bnot )\\b(?:at\\s+least|min(?:imum)?|more\\s+than|over|above|upwards\\s+of))\\s*$'  # E7: a figure after these is a floor, not a cap
REMOVE_MODE = 'remov|delet|drop|clear|take|subtract|minus|decreas|reduc'  # E22: a cart mode value that takes units off
MONEY_SIGN = '[^\\w\\s.,;:!?()\\[\\]{}"\'/%&+=<>#@~^|*\\\\-]'  # E7: a sign written before an amount (a symbol that is no letter, digit or punctuation)
OFFSET_ARG = '(?:^|_)(?:offset|start|skip|from|begin)(?:_|$)'  # E17: a page argument named as a row offset starts at 0
USE_E24_FIT_SCAN = True  # #111 E24: after an actor search, rows naming an item's required fit code are marked and a few more pages read
USE_E25_FIT_SCAN_OPEN = True  # #111 E25: the fit scan opens the cheapest unopened fitting rows and gives each fitting row a one-line verdict
USE_E26_FIT_SHOP_SWEEP = True  # #111 E26: the fit scan searches "shop + kind" for an item with no live fitting row
USE_E27_CHAIN_PAGING = True  # #111 E27: on tasks with no cheapest/most/least pick, a search naming a chain's shop is paged while a pointer is open
USE_E28_CODE_EITHER_WAY = True  # #111 E28: a fit code matches when a series wildcard on either side covers the other
USE_E29_WIDER_LINK_SEARCH = True  # #111 E29: the link search also tries "shop + strongest described value" and the shop alone
USE_E30_CHAIN_OPEN_RESERVE = True  # #111 E30: chain pages are opened when the steps left cover them plus the tail plus one
USE_E31_CUT_KEEPS_PROSE = True  # #111 E31: a record cut for a judge prompt writes its plain fields first, its nested blocks after
USE_E32_DROP_BY_PROPERTY_WORDS = True  # #111 E32: a store note's dropped requirement is matched by its property's words
USE_E33_BUDGET_TIMES_100 = True  # #111 E33: a store note's new budget off by a factor of 100 is rescaled
USE_E34_OFFER_VIEW_UNSEEN_LINES = True  # #111 E34: cart lines never offer-viewed get an offer view even when the re-read lacks its spare steps
USE_E35_CHAIN_OFFER_VIEWS = True  # #111 E35: resolved chain pages get an offer view at checkout; offer views before the forced buy
USE_E36_FLIPPED_SPACING_HUNT = True  # #111 E36: the cheapest hunt also searches the values with the number-unit spacing flipped
USE_E37_LATE_DROP_HUNT = True  # #111 E37: after a store note drops a requirement on a cheapest pick, one short extra cheapest hunt
USE_E38_BARE_NUMBER_HOLD = True  # #111 E38: checkout hold for a line whose attribute is a bare number ten-fold off a requirement's figure
USE_E39_VARIANT_LIST_HOLD = True  # #111 E39: checkout hold for a line listing several sizes while its title names another variant
USE_E40_WEAK_EVIDENCE_HOLD = True  # #111 E40: checkout hold for a bounded figure requirement no key of the line states
USE_E41_WEAK_SAME_MEASURE_KEY = True  # #111 E41: the weak hold names keys holding a figure in the requirement's unit and lets the line stay
USE_E42_FIT_CODE_HOLD = True  # #111 E42: checkout hold for a line whose record does not name the item's required fit code
USE_E43_LOOK_ALIKE_KEY_HOLD = True  # #111 E43: checkout hold for a line stating the measure only under a neighbouring key in the same unit
USE_E44_CHEAPER_MORE_ROUNDS = True  # #111 E44: after a passing review, up to 2 more rounds of cheaper unread rows (cheapest picks)
USE_E45_LIMIT_BREAKERS_LAST = True  # #111 E45: cheaper rows whose own figure breaks an at-least/at-most limit go last
USE_E46_BETTER_WORD_CONTRADICTION = True  # #111 E46: a review's "better" listing whose own key contradicts a word requirement is left out
USE_E47_LINKS_ONE_LINE_CARTS = True  # #111 E47: "see part X" references followed for one-line carts too, and before a single-item forced buy
USE_E48_DECLINE_CHAIN_ROUND = True  # #111 E48: one more chain round at a decline when listings were opened since the last round
USE_E49_LITERAL_FIGURE_HOLD = True  # #111 E49: decline hold naming opened listings whose attributes meet every requirement as written
USE_E50_LITERAL_LIST_SEPARATORS = True  # #111 E50: the literal word check removes list punctuation from both sides
USE_E51_SINGLE_ITEM_ALL_OPENED = True  # #111 E51: on single-item tasks the literal check looks at every opened listing
USE_E52_NEAREST_RULE_HOLD = True  # #111 E52: decline hold when a light call finds the request's nearest-listing rule
USE_E53_LATE_PROVEN_ORDER = True  # #111 E53: a late decline with literally proven listings for every item becomes a code order
FIT_SCAN_PAGES = 4  # #111 E24: further pages one fit scan reads (guide's value)
FIT_SCAN_TOTAL = 36  # #111 E24: pages all fit scans of an episode read together (guide's value)
FIT_SCAN_ROOM = 12  # #111 E24/E26: steps beyond the tail a fit-scan page or sweep search leaves
FIT_SCAN_ENOUGH = 3  # #111 E24: live fitting rows per item that end the paging early
FIT_SCAN_SHOWN = 8  # #111 E24: fitting rows shown per item
FIT_OPEN_ROOM = 15  # #111 E25: steps beyond the tail before the fit scan opens rows
FIT_OPEN_EACH = 3  # #111 E25/E26: fitting rows opened per item in one scan
FIT_OPEN_SCAN = 6  # #111 E25: fitting rows opened in one scan
FIT_SCAN_BY = 160.0  # #111 E25/E26: episode seconds after which the fit scan opens and sweeps nothing (guide's value)
SWEEP_SHOPS = 3  # #111 E26: shops swept per item
SWEEP_WORDS = 3  # #111 E26: kind words after the shop name
CHAIN_SHOPS = 6  # #111 E27: shop names a chain search is recognised by
CHAIN_PAGES = 2  # #111 E27: further pages a chain search may need to be paged at all
CHAIN_PROSE_WORDS = 8  # #111 E27: words a text field of an opened listing needs to count as its long text
LINK_CAP = 3  # the king's link searches per episode
LINK_CAP_WIDE = 5  # #111 E29: link searches per episode with the wider link search
LINK_VALUES = 2  # #111 E29: strongest described values searched with the shop's name
BUDGET_JUMP = 30  # #111 E33: a new budget this many times off the one in force is suspect
BUDGET_NEAR = 3  # #111 E33: ... and rescaled when a factor of 100 brings it within this factor
LATE_HUNT_ROOM = 2  # #111 E37: searches of the extra hunt when the normal room is gone
HOLD_ROOM = 3  # #111 E38-E43: steps beyond the tail a checkout hold needs
WEAK_LINES = 2  # #111 E40: cart lines one weak hold names
WEAK_SHOWN = 3  # #111 E40/E43: listings or rows a hold names per line
SAME_MEASURE_KEYS = 2  # #111 E41: same-unit keys a weak hold names
FIT_HOLD_SHOWN = 4  # #111 E42: fitting rows a fit-code hold names
CHEAPER_ROUNDS = 2  # #111 E44: extra rounds of cheaper rows after a passing review
CHEAPER_ROOM = 5  # #111 E44: steps beyond the tail an extra cheaper round needs
DECLINE_TRAIL_SECONDS = 21.0  # #111 E48: model seconds a decline chain round needs
LITERAL_SHOWN = 2  # #111 E49: proven listings named per item
NEAREST_SECONDS = 26.0  # #111 E52: model seconds the nearest-rule check needs
NEAREST_SENTENCE = 12  # #111 E52: characters the quoted sentence needs
LATE_ORDER_STEPS = 5  # #111 E53: steps left at or below which a decline may become a code order
NEAREST_RULE = 'Read the shopper\'s request below. Decide whether it says what to do when no listing meets every requirement, namely that the nearest listing (the one missing the fewest requirements) is to be bought anyway. Return ONE JSON object and nothing else: {"nearest": true or false, "sentence": the request\'s sentence that says so, copied exactly, or ""}'  # #111 E52: new prompt of the nearest-rule check
_NUM = '(\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(\\d+))?'
_UNIT = '([A-Za-z%°µ"\\\'][A-Za-z%°µ]{0,11})(?![A-Za-z])'
FIG_UNIT = '(?<![\\w.])' + _NUM + '\\s*-?\\s*' + _UNIT  # #111 E38-E49: a figure written with its unit
FIG_RANGE = '(?<![\\w.])(\\d+(?:\\.\\d+)?)\\s*(?:-|–|to)\\s*(\\d+(?:\\.\\d+)?)\\s*-?\\s*' + _UNIT  # #111: a range with one unit
LIMIT_MIN = '\\b(?:at\\s+least|no\\s+(?:less|fewer|lower|smaller|shorter)\\s+than|not\\s+(?:less|fewer|below|under)(?:\\s+than)?|min(?:imum)?|from|upwards\\s+of)\\s*:?\\s*$'  # an inclusive floor before a figure
LIMIT_MIN_STRICT = '(?<!no )(?<!not )\\b(?:(?:more|greater|higher|larger|bigger|longer)\\s+than|over|above|exceeding)\\s*:?\\s*$'  # a strict floor before a figure
LIMIT_MAX = '\\b(?:at\\s+most|no\\s+(?:more|greater|higher|larger|bigger|longer)\\s+than|not\\s+(?:more|over|above)(?:\\s+than)?|max(?:imum)?|up\\s+to|within)\\s*:?\\s*$'  # an inclusive cap before a figure
LIMIT_MAX_STRICT = '(?<!no )(?<!not )\\b(?:(?:less|fewer|lower|smaller|shorter)\\s+than|under|below)\\s*:?\\s*$'  # a strict cap before a figure
LIMIT_MIN_AFTER = '^\\s*(?:\\+|(?:or\\s+(?:more|higher|greater|larger|above|over|better|longer)|and\\s+(?:up|above|over|more)|plus)\\b)'  # a floor after a figure
LIMIT_MAX_AFTER = '^\\s*(?:or\\s+(?:less|lower|fewer|smaller|below|under|shorter)|and\\s+(?:under|below|less))\\b'  # a cap after a figure
NEGATION = '\\b(?:not|no|never|without|excluding|except|unlike|anything\\s+but|other\\s+than|different\\s+from)\\b'  # #111 E49: a requirement that excludes
DIFFERENCE = '\\b(?:unlike|other\\s+than|differ\\w*|not\\s+the\\s+same)\\b'  # #111 E49: a pointer worded as a difference
LIST_MARKS = '[,;/|()\\[\\]{}]+'  # #111 E50: list punctuation removed before the literal word check
FILLER_WORDS = ('a', 'an', 'the', 'and', 'or', 'of')  # #111 E49: words the literal word check does not require
# ---- #97 (Gental's Gen_cairn race-16 guide), items 1-33. Code items default True; prompt-wording items default False
# (their alternative texts are built from the king's constants below, which stay byte-identical).
USE_G1_CUT_TURN_TO_SPARE = True  # #97 item 1: a tool turn cut at the token cap with no call goes straight to the next model (cap itself: E14)
USE_G2_COMPLETE_REREAD = True  # #97 item 2: cart completion re-reads the lines' offer pages once before its checkout (tight-step rule: E34)
USE_G3_LATE_PAGE_SHOW = True  # #97 item 3: one step beyond the reserve after a refused checkout, the one never-shown line's page is shown first
USE_G4_BRIEF_RELATION_WORDING = False  # #97 item 4: (prompt BRIEF) a pointer must keeps the request's own relation wording
USE_G5_LEDGER_DROPPED = True  # #97 item 5: (code) the ledger lists requirements the shopper withdrew, flagged as no longer counting
USE_G5_CHECK_DROPPED_RULE = False  # #97 item 5: (prompts CHECK, READING -> AUDIT, SWAP) withdrawn requirements no longer count
USE_G6_CHECK_MEASURED_RULES = False  # #97 item 6: (prompts CHECK, READING -> AUDIT, SWAP) measured limits under their own key; plural variant lists
USE_G7_TRAIL_RULES = False  # #97 item 7: (prompt TRAIL) record ids, link values, value-only search words, deferring pages, partner wording
USE_G8_TRAIL_REQUEST = False  # #97 item 8: (prompt TRAIL) the chain judge also reads the request and Cairn's chain findings
USE_G9_NEED_ON_NAMED_ITEMS = True  # #97 item 9: a requirement the brief gave several items stays only on the items its sentences name
USE_G10_MEASURED_LIMITS = True  # #97 item 10: a bound/condition word after a figure is no unit; which requirements are measured limits
USE_G12_BARE_FIGURE_NOTE = True  # #97 item 12: evidence notes a bare figure ten-fold off a measured limit with its reading in that unit
USE_G13_WORD_IN_TITLE = True  # #97 item 13: literal check: with no key for the property, a word value stated in the title/attributes (or initials) counts
USE_G14_WORD_FORMS = True  # #97 item 14: literal check: word forms of one stem count as the same word
USE_G15_BARRED_CODES = True  # #97 item 15: literal check: a code the listing's fit clause bars is not stated
USE_G16_POINTER_BOUND_VALUE = True  # #97 item 16: literal check: a chain value that holds a bound/condition word stays unjudged
USE_G17_KEY_PROOF = True  # #97 item 17: literal check: one list part holds the words; the fuller key decides; plural variant lists
USE_G19_SINGLE_VALUE_OPENERS = True  # #97 item 19: opening searches of each value alone with the noun when the combined one is empty; own rows kept
USE_G20_FRONT_PAGING = True  # #97 item 20: cheapest picks: page the largest qualifying own value searches up front
USE_G21_DEAR_LINE_PAGE = True  # #97 item 21: a carted line dearer than most of its search page: Cairn reads the next page and its cheaper rows
USE_G22_DEVICE_SERIES_SWEEP = True  # #97 item 22: clock-bound fit tasks: page the device's searches until a row names the proven parts' series
USE_G23_BIG_FRONT_PAGING = True  # #97 item 23: clock-bound tasks: up-front paging for every pick rule and one search per item
USE_G24_OWN_LANE_RANK = True  # #97 item 24: widen lanes rank own-search rows first, then the owning items' requirement words
USE_G25_LOOSE_LABEL_LANES = True  # #97 item 25: a store label naming its item only loosely takes no widen lane
USE_G26_NESTED_LABEL_LEDGER = True  # #97 item 26: ledger line for nested store labels; the no-line fault names the item's own label
USE_G27_HUNT_CHOICE_SETS = True  # #97 item 27: cheapest hunt: a value set per choice, no codes, a number-only set, every set searched
USE_G28_ONE_PAGE_ROOM = True  # #97 item 28: one remaining page of a model search is auto-read with less spare room
USE_G29_BIG_NEXT_PAGE = True  # #97 item 29: clock-bound tasks, other pick rules: the next page of a model search read in the same turn
USE_G30_BIG_ROW_READS = True  # #97 item 30: clock-bound tasks: Cairn reads the two best unread rows of each model search
USE_G31_SHORT_CHEAP_READS = True  # #97 item 31: cheapest picks, clock not binding: read the two cheapest rows of a search paged to its end
USE_G32_IGNORED_CAP_REPEAT = True  # #97 item 32: a repeat differing only in a cap this search mode ignores is answered from the earlier rows
USE_G33_SOLD_PEERS = True  # #97 item 33: a sold-out line's cheaper same-label peers are read; search-under-price text only when needed
STEP_SECONDS = 4.0  # #97 items 21-23, 29-31: model + store seconds one step costs in the loop (race 16: ~264 s for ~68 steps on big
# tasks); the clock binds (a "big" task) while the steps left beyond the tail would take longer than the model time left
FRONT_PAGES = 2  # #97 item 20: further pages read per up-front search
FRONT_SPAN = 8  # #97 item 20: a value search matching more than this many pages of rows is too broad to page up front (race 16: 48 rows, 6 a page)
FRONT_ROOM = 6  # #97 item 20: steps kept before the first-moves stop line while paging up front (room for the widen reads after it)
FRONT_SEARCHES = 2  # #97 items 20/23: searches paged up front (on clock-bound tasks: one per item when more)
DEAR_ROOM = 12  # #97 item 21: steps beyond the tail before the dear-line check reads a page
DEAR_ROWS = 4  # #97 item 21: buyable priced rows a search page needs before its prices say anything
DEAR_SHARE = 2 / 3  # #97 item 21: share of those rows cheaper than the carted line that makes it "dear"
DEAR_READS = 2  # #97 items 21/30/31/33: rows read after a page (or per sold line)
DEAR_KEEP = 8  # #97 item 21: steps beyond the tail those reads leave
BIG_ROOM = 20  # #97 items 22/29/30: steps beyond the tail clock-bound reads leave (a big task's buying room)
SERIES_PAGES = 6  # #97 item 22: fresh pages one device-series sweep reads
SERIES_KEEP = 10  # #97 item 22: steps beyond the tail the device-row reads after a sweep leave
LOOSE_SHARE = 0.4  # #97 item 25: a label scoring below this share of its item's best label takes no widen lane ...
LOOSE_FLOOR = 0.5  # #97 item 25: ... when that best label scores at least this (a clear own label exists)
CHOICE_SETS = 3  # #97 item 27: choices of a closed list that get their own hunt value set
ONE_PAGE_ROOM = 6  # #97 item 28: spare steps beyond the tail that auto-read one remaining page (two pages keep the king's 12)
SHORT_READ_ROOM = 14  # #97 item 31: steps beyond the tail before the cheapest rows of a paged-out search are read
SHORT_READ_KEEP = 11  # #97 item 31: steps beyond the tail those reads leave
PEER_KEEP = 5  # #97 item 33: steps beyond the tail the sold-out peer reads leave
BARE_OFF = 10  # #97 item 12: a bare figure this many times off a requirement's figure reads as another unit (E38's ten-fold)
TRAIL_REQUEST_CHARS = 3000  # #97 item 8: characters of the request the chain judge reads
TRAIL_FINDINGS = 6  # #97 item 8: latest chain findings the chain judge reads
LIST_SPLIT = '\\s*[,;/|]\\s*'  # #97 item 17: list separators inside one attribute value
G4_RELATION = '; in such a must keep the request\'s own words for how the wanted value relates to that listing\'s value (the same as, unlike, matching, going with, a partner of, within a margin of), unchanged: never reduce them to plain sameness'
G5_RULE = ' The requirements in force are the shopper\'s latest word and override the request text: a requirement the ledger marks as withdrawn no longer counts, whatever the request says.'
G6_RULE = ' A figure with a unit (a measured limit) counts only under a key that names that property: the same figure under a key for some other measure, or only in the title, leaves the requirement unproven, and the listing fails it. A list under the plural of the property\'s key (such as "sizes") is the set of variants the store offers, so it does not state this listing\'s own value when the title names a different variant.'
G7_RULES = '\nMore rules: a link\'s "id" is always the id of a record below (its id field), never a code printed in a title. Give each link also "values": the values its description gives, as a few bare words without property names or shop names. A link\'s "find" is the shop\'s name followed by those values only, without property names and without the kind of product, because the store\'s search needs every word to appear in the listing (for "the lamp Elmora sells with shade colour sage and base material oak" give "Elmora sage oak"). A page whose text sends the reader on to another listing for the property does not end the chain, even when its attributes show a value: that further listing is the next link. Take the partner or counterpart value only when the request asks for a value that differs from and pairs with the pointed one; a request to match the pointed value wants that value itself, whatever verb it uses.'
G8_REQUEST_HEAD = "\n\nTHE SHOPPER'S REQUEST (the source the requirements above were condensed from; its own words for how the wanted value relates to the value along the chain decide which value is wanted):\n"
G8_FINDINGS_HEAD = "\n\nCAIRN'S LATEST FINDINGS ON PAGES ALONG THE CHAINS (a page rejected as a link, or one that defers the value to another listing):\n"
G8_RECORDS_HEAD = '\n\nRECORDS (newest first; a page counts as the pointed listing only when it shows all the values its description names):\n'
assert BRIEF.count('are never musts of this product}') == 1 and CHECK.count('\nWhen the pick rule is cheapest') == 1 and READING.endswith('.')
BRIEF_G4 = BRIEF.replace('are never musts of this product}', 'are never musts of this product' + G4_RELATION + '}', 1)  # #97 item 4
# ---- #97 (Gental's Gen_cairn race-16 guide), items 34-76: pointer chains, holds before the checkout and the cart check,
# endings, build-13 ports. Code items default True; prompt-wording items and the items the guide reports as failed, never
# fired or without net gain default False. Items already ported from Lion's #111 keep their E switch (60 = E13, 65 = E16,
# 67 = E22, 68 = E33, 75 = E31, 76 = E15); items 43 (inside 36), 11 (inside 44) and 18 (inside 54) have no switch of their own.
USE_G34_CHAIN_ROUNDS = True  # #97 item 34: early chain follow: more rounds and pages, a buying reserve; one more round after a rejected/deferring page
USE_G35_LINK_MEMORY = True  # #97 item 35: link searches remembered with their rows and next page, repeats answered or paged; two gaps a round
USE_G36_LINK_WORDS = True  # #97 items 36/43: link search words without attribute names; ranked candidates (replaces of_kind and E29's walk)
USE_G37_ATTR_LINK = True  # #97 item 37: an opened page is the link only when its attribute block holds every described value
USE_G38_TITLE_CODE_LINK = True  # #97 item 38: a link the judge names by a title code is mapped to the one row carrying it
USE_G39_CHAIN_NEED_OVERLAP = True  # #97 item 39: a chain whose requirement name is reworded is matched by clear word overlap
USE_G40_LINK_REJECT = True  # #97 item 40: a read link whose page lacks one of its own described values is rejected and searched again
USE_G41_DEFER_PAGE = True  # #97 item 41: a last page whose prose names the property but not the value defers it (once per page)
USE_G42_CHAIN_PAGE_VIEW = True  # #97 item 42: an attribute read of a chain shop's listing is sent as its page view
USE_G44_WEAK_LINES = True  # #97 items 44/11: cart lines that do not state a requirement the way the grader reads it (helper of 45/46)
USE_G45_WEAK_LINE_HOLD = True  # #97 item 45: one checkout hold naming listings that state what a weak cart line does not
USE_G46_WEAK_CART_HEAVY = True  # #97 item 46 (code): a cart with weak lines skips the light pre-screen and goes to the heavy checker
USE_G46_CHECK_WEAK_SECTION = False  # #97 item 46 (prompt CHECK at run time): CHECK is told the weak lines and the reading rule
USE_G47_CHECK_CODE_LINKS = False  # #97 item 47 (prompt CHECK at run time): CHECK is told the fit links the cart establishes by code
USE_G48_SWAP_KEEPS_LINKS = True  # #97 item 48: a swapped-in listing of a fit-bound item must keep every fit link (helper of 49/52/53/54)
USE_G49_CHEAPER_HOLD = True  # #97 item 49: one checkout hold naming a cheaper read listing that establishes every requirement
USE_G50_CHEAPER_HOLD_LATE = True  # #97 item 50: the same cheaper hold near the step limit, first in the checkout branch
USE_G51_CHEAPER_CODE_LINKS = False  # #97 item 51: the cheaper hold names the code links a swapped-in bound line keeps (guide: 0 of 1, failed)
USE_G52_AGREED_SWAP = True  # #97 item 52: Cairn makes the cheaper swap itself when its hold and CHECK name the same listing
USE_G53_BOUND_SWAP = True  # #97 item 53: ... also when the cheaper line of a fit-bound item keeps its code link
USE_G54_WHOLE_SETS = True  # #97 items 54/18: whole device-and-parts sets of read listings: status, misfit hold, decline hold, forced finish
USE_G55_BASKET_PLAN = True  # #97 item 55: cart completion plans one line per item from any review lead
USE_G56_BASKET_REVIEW = True  # #97 item 56: the second decline review checks the whole planned basket on multi-item tasks
USE_G57_LATE_FINISH = True  # #97 item 57: a late decline with a whole cart (built lines): Cairn puts lines back and checks out
USE_G58_LATE_SHORT_REVIEW = True  # #97 item 58: a late decline with a short cart: review, whole-basket check, then order
USE_G59_REOPEN_SEARCH = True  # #97 item 59: search reopened once when the model stalls with an item uncarted
USE_G61_PUSH_STEPS_AT_CALL = True  # #97 item 61: the early-decline push is judged on the steps left when the call arrived
USE_G62_FORCED_BASKET = False  # #97 item 62: forced ending on an empty cart orders the cheapest established basket (guide: never fired)
USE_G63_NO_BLIND_CHECKOUT = True  # #97 item 63: forced finish: no blind checkout of an emptied cart or one unchanged since a refusal
USE_G64_MORE_ROUNDS = True  # #97 item 64: up to two more working rounds after a refused forced ending
USE_G66_NEXT_ARG_STRICT = True  # #97 item 66: the schema guess of the paging parameter skips words fields; integers prefer integer params
USE_G69_CODE_EXEMPT_SERIES = True  # #97 item 69: a part naming the device's model code is exempt from the inferred-series veto
USE_G70_SPLIT_WORDS = True  # #97 item 70: a multi-word value whose words stand apart (other words between) is not stated
USE_G71_SET_FIGURES = True  # #97 item 71: whole read sets judge at-least and exact figure requirements as written
USE_G72_LITERAL_HOLD_TOTAL = True  # #97 item 72 (extends E49): more than five steps, no fit-bound items, total budget, one use shared with sets
USE_G73_CLOCK_BEST_ORDER = False  # #97 item 73: late on the clock a multi-item decline becomes the best order read (guide: no net gain)
USE_G74_FORCED_FIT_REPAIR = True  # #97 item 74: a forced order first swaps a cart line the fit check rejects for a read one that fits
EARLY_ROUNDS = 6  # #97 item 34: chain rounds of the early follow (the king's 3 = EARLY_TRAILS)
EARLY_PAGES = 12  # #97 item 34: pages the early follow opens in all (other follows keep the king's 6)
ROUND_PAGES = 3  # #97 item 34: pages one chain round opens (the king's value)
LINK_CAP_MEMO = 6  # #97 item 35: link searches per episode (king 3, E29 5)
LINK_GAPS = 2  # #97 item 35: missing links searched in one round
LINK_STEPS = 12  # #97 items 35/36: steps left a link search needs (the king's value)
LINK_CANDIDATES = 2  # #97 item 36: link candidates returned (the king's value)
LINK_WORD_CHARS = 3  # #97 items 36/40: a value word needs this many characters unless it holds (36) / starts with (40) a digit
LINK_ID_CHARS = 6  # #97 item 38: a known id of this many characters found inside a judge's link name is that link
CODE_LINK_CHARS = (3, 40)  # #97 item 38: length of a link name read as a title code
NEED_SHARE = 0.6  # #97 item 39: share of shared words a reworded chain requirement needs ...
NEED_LEAD = 0.15  # #97 item 39: ... and its lead over the next pending requirement
FINDINGS_KEPT = 12  # #97 items 40/41: chain findings kept for the judge
FINDING_WORDS = 3  # #97 item 40: missing words a rejection finding names
LONG_TEXT = 40  # #97 item 41: characters of a text field that counts as a page's long text
CHAIN_WIDEN = 5  # #97 item 42: widening passes over viewed chain pages for shop names
WEAK_GUARD_READ = 3  # #97 item 44: read listings of a label before "no listing of it states the word" may silence a weak line
WEAK_UNREAD_ROOM = 5  # #97 item 45: steps beyond the tail before the weak hold names unread rows
SET_ROOM = 5  # #97 items 54/72: steps left before a decline hold may name sets or proven listings
LATE_REVIEW_BY = WIND_DOWN  # #97 item 58: clock limit for the late review and basket check (guide: 200 s = the king's WIND_DOWN)
MORE_ROUND_TURNS = 6  # #97 item 64: model turns of each extra working round (the king's retry round allows 10)
MORE_ROUND_SPEND = 0.02  # #97 item 64: spend allowance above SPEND_HARD for the extra rounds
LATE_ORDER_ROOM = 65.0  # #97 item 73: model seconds left from which a decline becomes the best order (guide: from 215 s of 300 s, 20 s margin)
CODE_LINKS_SHOWN = 3  # #97 item 51: code links a cheaper hold names
G46_SECTION_HEAD = "\n\nLINES THAT DO NOT STATE A REQUIREMENT THE WAY THE ORDER IS JUDGED (a measured limit counts only under a key named for that property and in its unit, a bare figure is read in the requirement's unit, and a wanted word counts only in an attribute value or the title, never the description alone; treat each requirement below as failed unless the record shows it some other way):\n"  # #97 item 46
G47_SECTION_HEAD = "\n\nFIT LINKS THE CART'S OWN RECORDS ESTABLISH BY CODE (the store decides fit by these codes and series, so every link below holds; another use or size a description mentions does not cancel it):\n"  # #97 item 47
# ---- stage 3: #110 (our cairo race-13 guide, P110), #101 (Sam's Cairn-easy, S101), #103 (Hans's 29-Cairn race 12, H103),
# #104 (Richard's c15_v8b race 15, R104). Code items default True; prompt-wording items and items a guide reports as never
# fired default False (the king's prompt texts stay byte-identical when they are off). Items already ported keep their
# E/G switch (see the stage report); #110 items 4-5 (code chain walk), #104 items 1 and 8 are not ported.
USE_P110_1_NEAREST_ORDER = True  # #110 item 1: on a request a light reading finds a nearest-listing rule in, the final refusal and the forced ending order the nearest listing per item (one heavy pick, else a code pick; inside the budget)
USE_P110_1_CHECK_NEAREST = False  # #110 item 1 (prompt CHECK at run time): CHECK reads the nearest rule and sees the other listings read
USE_P110_2_ROW_PRICE = True  # #110 item 2: with no price key learned, a listing's price is its own field named for price or cost
USE_P110_3_CAP_DESCENT = True  # #110 item 3: cheaper-fit descent through the price-capped category listing (plausible rows set the caps)
USE_P110_6_SET_CHECK = True  # #110 item 6: lowest-total opened set contradicting no requirement: status line on cheapest picks, checkout hold
USE_P110_7_PROVEN_ORDER_ANY = True  # #110 items 7/9: on "any" picks Cairn orders a strictly proven set itself at the top of a turn (twice at most)
USE_P110_8_PROVEN_SWAP = True  # #110 items 8/9: at the model's checkout Cairn swaps the cart to a strictly proven set when a cart line has a gap (a dearer cart on a cheapest pick: G49/G52/G53, or this swap when they are off)
USE_P110_10_REVIEW_SET_ASIDE = True  # #110 item 10 (extends E46): CHECK's verdict is set aside when the cart shows no gap and every listing it names as better shows one
USE_P110_11_NEAR_SETTLE_HOLD = True  # #110 item 11: on a nearest-rule request, a checkout holding a line its own attributes contradict is held once while many steps remain
USE_P110_12_SEEN_OWNER_LINK = True  # #110 item 12 (extends E47): a referred part seen only as a row is opened; references followed right before every checkout
USE_P110_13_SHORT_VALUE_FIRM = True  # #110 item 13 (extends E8): a value with fewer than four words of three or more letters is never "described"
USE_P110_14_KEYED_PAIRS = True  # #110 item 14 (extends G37): an opened shop page is the link only when no "property value" pair of the description is contradicted under that key
USE_P110_15_IDLE_OTHER_MODEL = True  # #110 item 15: after a model turn with no tool call, the next turn goes to the other loop model first
USE_S101_5_WIDEST_RUNGS = True  # #101 item 5: the cheapest hunt also runs the wider rungs of its query ladder after the first rung with rows
USE_S101_6_SINGULAR_NOUN = True  # #101 item 6: the hunt also tries the store's own singular of its category noun (a title word of the label's rows)
USE_S101_7_REPLY_NOTES = True  # #101 item 7: a shopper reply with more than one passage has its further passages read by the world-note reader
USE_S101_8_WORD_CONTRADICTION = True  # #101 item 8: a cart line whose own key for a word requirement's property states another value is a line fault
USE_S101_9_DONT_CARE_DROP = False  # #101 item 9: a volunteered don't-mind on a live requirement nobody asked about drops it (guide: never fired)
USE_S101_10_DONT_CARE_POINTER = False  # #101 item 10: ... also a don't-mind on an unresolved pointer requirement (guide: an alternative inside 9's gate)
USE_H103_1_BRIEF_VALUE_CLAUSES = False  # #103 item 1 (prompt BRIEF): a "property is value" clause is a must with only the value kept
USE_H103_2_BUDGET_UNIT = True  # #103 item 2 (extends E33): a reply's budget is rescaled too; a fractional budget against whole-number prices reads x100
USE_H103_3_STORE_WORDS = True  # #103 item 3: the stated-value check reads the store's own values instead of a fixed word set
USE_H103_4_SPLIT_NOTE = True  # #103 item 4 (extends G70): the evidence block notes a hit holding the wanted words only apart
USE_H103_6_FIT_PLAN_VALUES = True  # #103 item 6: the fit plan ranks series by parts holding every plain required value; switch-device note
USE_H103_7_FIT_PLAN_CAPS = True  # #103 item 7 (extends E7): a part over its item's price ceiling counts for no series in the fit plan; the plan names the ceiling
USE_R104_2_HEAVY_PICK = True  # #104 item 2: once, at a checkout, one strong-model pick per cart line among read listings of its label; a differing pick holds once
USE_R104_3_COMPACT_ROWS = True  # #104 item 3: rows of a many-row reply are shown without their empty fields (titles and descriptions not cut)
USE_R104_4_NEEDLESS_DECLINE_HOLD = True  # #104 item 4: a decline while the cart holds a built line for every item is held once
USE_R104_5_RELATION_NOT_POINTER = True  # #104 item 5: a lookup that only relates two requested items (no page or shop talk) is no pointer
USE_T_R104_2_EVERY_PICK = True  # #104 item 2 as written: the one strong-model pick reviews every cart line on every pick rule (not only lines with a gap on "any" picks)
USE_T_R104_3_CUTS = True  # #104 item 3 as written: in a reply listing 2+ listings, titles shown cut to ROW_TITLE_CHARS and other long text to ROW_TEXT_CHARS (the code keeps the full rows)
USE_T_R104_5_NAMED_REFERRAL = True  # #104 item 5 as written: a cart line whose page refers by name to another listing for a value gets that listing searched and opened before checkout
ROW_TITLE_CHARS = 110
ROW_TEXT_CHARS = 90
REFERRAL = '\\b(?:refer(?:s|red|ring)?\\s+to|see|check|consult|look\\s+at|taken\\s+from|given\\s+(?:on|in|by))\\s+(?:the\\s+|a\\s+)?([A-Za-z0-9][^.;:!?()\\n]{2,70}?)\\s+(?:listing|page|product)\\b'  # a page sending the reader to another listing by name
USE_R104_7_GUIDE_RULES = False  # #104 item 7 (prompt GUIDE): batch independent calls; a measured limit needs a figure of that measure
# ---- stage 12 (race-16 needs-in-the-request study of a teammate's v13 against ours): each switch off = cairn_guides_v5 ----
USE_R16_SCOPE_ROOTS = True  # a requirement the brief gave several items is scoped by word roots: sentence words without punctuation ("pattern;"), an item named when every root of its name or store label stands in the sentence (so "slipcovers and furniture covers" names "slipcover or furniture cover", not the narrower "furniture cover"), a value read in the clause that states it, a property-less sentence ("1 cctv camera in cream") stating a word value, and a sentence naming no holder but sharing a name word with two (an intro "a cream camera") set aside when another sentence names one
USE_R16_CAP_FULLEST_NAME = True  # a per-item price cap whose sentence names several items goes to the one item whose whole name or store label has the most words there (the store label "slipcovers and furniture covers" beats "furniture cover"); a request rule restating that cap is marked in the ledger as that item's own limit
USE_R16_REFUSED_LINES = True  # lines the cart check refused (its problem names the line's item first, or the line alone) rank last in the best read order, are not chosen by the short-cart fill and are not put back by the late finish (which also skips a line whose item already holds one)
USE_R16_LINK_PAGE_UNSHOWN = True  # on a store with a page view apart from the attribute view, a chain-link candidate is any row of the named shop whose page was never shown (attribute reads included), described values match by word roots ("1 piece" for "1 pieces"), a shop row whose attributes hold every value but whose page was never shown is opened rather than taken as known, and the chain judge sees attribute-read rows uncut like opened ones
USE_R16_LINK_KIND_FACTS = True  # a chain page of a requested item's own store label whose attributes were never read gets them read (one step, with room beyond the tail), so the listing the chain points at can be judged as a purchase itself
USE_R16_SOLD_TIES = True  # lowest-price pick: once per sold-out cart line, its store label is listed capped at its own price (equal prices kept) so the sold-out peer reads can open the same-price twins the store keeps
R16_TIE_ROOM = 6  # USE_R16_SOLD_TIES: steps beyond the tail the capped listing needs (list, reads, swap, checkout)
R16_KIND_ROOM = 6  # USE_R16_LINK_KIND_FACTS: steps beyond the tail an attribute read of a chain page of the item's own label leaves
NEAREST_PICK_SECONDS = 12.0  # #110 item 1: model seconds the heavy nearest pick needs (else the code pick, #97 item 73's ranking)
NEAREST_RECORDS = 36  # #110 item 1: listings shown to the nearest pick (guide's value), opened first, cheapest first
NEAREST_EACH = 4  # #110 item 1: listings per item at least in that share
NEAREST_RESCUES = 2  # #110 item 1: nearest carts per episode (a store refusal can need a second)
NEAREST_ROW_CHARS = 700  # #110 item 1: characters of one listing record the nearest pick reads
ROW_PRICE_KEY = 'price|cost'  # #110 item 2: a record field holding the listing's own price
TEXT_FIELD = MESSAGE_FIELD[:-len('(?:_|$)')].rstrip(')') + '|comments?|reasons?|justif\\w*|explan\\w*|why|remarks?)(?:_|$)'  # #110 items 1/7: an optional checkout field that takes free text (the king's message-field words and a few more)
DESCENT_WHOLE = 30  # #110 item 3: a capped shelf listing of at most this many rows is read to its last page (guide's value)
DESCENT_PAGES = 2  # #110 item 3: pages read a round of a longer capped listing (guide's value)
DESCENT_ROUNDS = 5  # #110 item 3: caps tried per cart line (guide's value)
SET_ITEMS = 6  # #110 items 6-8: opened listings per item the set check combines, best first (guide's value)
SET_COMBOS = 300  # #110 item 6: device combinations the set check weighs at most (several device items)
PROVEN_TRIES = 2  # #110 item 7: code orders at most this many times an episode (guide's value)
PROVEN_SWAPS = 2  # #110 item 8: code swaps at the model's checkout at most this many times an episode (guide's value)
PROVEN_SECONDS = 40.0  # #110 item 7: model seconds left before Cairn orders a proven set itself (guide's value)
SETTLE_STEPS = 12  # #110 item 11: steps left (or a quarter of the step limit, whichever is more) for the settle hold (guide's value)
SETTLE_SECONDS = 60.0  # #110 item 11: model seconds left for the settle hold (guide's value)
SINGULAR_ROWS = 2  # #101 item 6: title rows of the label a singular noun must appear in
BUDGET_PRICES = 3  # #103 item 2: priced rows that must all be whole numbers before a fractional budget reads x100
PICK_SECONDS = 30.0  # #104 item 2: model seconds left for the strong-model pick at a checkout
PICK_ROWS = 10  # #104 item 2: read listings of a cart line's label shown to the pick (guide's value)
NEAREST_PICK = 'A shopper asked that, when no listing meets every requirement, the listing missing the fewest of them be ordered instead (being out of stock or over the budget counts as a miss; among equally near ones the cheapest, or one within any price margin the request allows). Below are the requested items, the requirements in force and the listings read for each item. Choose that nearest listing for each item: of the item\'s own kind, in stock, the whole order inside the budget, and a part that must work with another requested item fitting the listing chosen for that item. Return ONE JSON object and nothing else: {"lines": [{"item": item number, "id": the listing id exactly as shown, "misses": [the requirements it does not meet]}]}; leave an item out only when no listing of its kind is shown.'  # #110 item 1: the heavy nearest pick (a new reading of our own)
CHECK_NEAREST = ' When the request says to order the nearest listing when nothing meets everything, a line that misses a requirement is right when no listing shown meets every requirement of its item, and wrong only when a listing shown misses fewer requirements, or as many and costs less beyond the margin the request allows.'  # #110 item 1 (prompt, off by default)
SETTLE_HOLD = 'Not run yet (no step used). The request wants the nearest listing only when nothing meets every requirement, and {lines} in the cart states another value than a requirement while {left} steps are left. Order a nearest listing only after a real search finds nothing that meets everything: list the item\'s category (with its section or label filter, paging on) and search the missing value with the product noun; open what may fit. Then check out (the nearest listing if nothing better turned up).'  # #110 item 11
LINE_PICK = 'A shopping agent is about to check out. For each cart line below, choose the listing the shopper should get from that line\'s candidates (the line\'s own listing is one of them, records shown): one that meets every requirement in force and, when the pick rule asks for the lowest price or the most or least of a measure, the best by that rule. Keep the line\'s own listing when it meets every requirement and the pick rule does not favour another. Return ONE JSON object and nothing else: {"lines": [{"line": the cart line\'s id, "pick": the chosen listing id exactly as shown, "why": "one short reason"}]}'  # #104 item 2: the strong-model pick (a separate call)
H103_CLAUSES = ' When the request ties a property of the product to a value inside a clause ("its finish should be matte", "the colour it comes in is navy"), that pairing is a must on that property, never the product\'s name or store label; record just the value, leaving out the words of the sentence around it (a verb or other words after the value); and a value worded as something missing or refused ("handle-free", "comes without a lid") is recorded as that exact value for the listing to show, not as an exclusion.'  # #103 item 1 (prompt, off by default)
R104_BATCH = '\n- Each reply costs several seconds of a short clock: send independent calls (searches, listings to open) together in one reply.'  # #104 item 7 (prompt, off by default)
R104_MEASURE = ' A requirement with a measured limit is met only when the listing gives a figure for that same measure; descriptive words that merely hint at it do not count.'  # #104 item 7 (prompt, off by default)
assert GUIDE.count('\n- Search tools can be narrow') == 1 and GUIDE.count('limits include the edge value, and a range must lie wholly inside a limit.') == 1 and BRIEF.count('do not swap it for a neighbouring property.') == 1 and CHECK.count('\nReturn ONE JSON') == 1
GUIDE_R104 = GUIDE.replace('\n- Search tools can be narrow', R104_BATCH + '\n- Search tools can be narrow', 1).replace('limits include the edge value, and a range must lie wholly inside a limit.', 'limits include the edge value, and a range must lie wholly inside a limit.' + R104_MEASURE, 1)  # #104 item 7


def _reading() -> str:
    """#97 items 5/6: the reading rules AUDIT and SWAP carry: the king's READING, plus the prompt rules switched on."""
    return READING + (G5_RULE if USE_G5_CHECK_DROPPED_RULE else '') + (G6_RULE if USE_G6_CHECK_MEASURED_RULES else '')


def _check_text() -> str:
    """#97 items 5/6: CHECK, with the prompt rules switched on added after its reading rules (the king's text when off)."""
    extra = (G5_RULE if USE_G5_CHECK_DROPPED_RULE else '') + (G6_RULE if USE_G6_CHECK_MEASURED_RULES else '')
    return CHECK.replace('\nWhen the pick rule is cheapest', extra + '\nWhen the pick rule is cheapest', 1) if extra else CHECK


def _audit_text() -> str:
    reading = _reading()
    return AUDIT if reading == READING else AUDIT.replace(READING, reading, 1)


def _swap_text() -> str:
    reading = _reading()
    return SWAP if reading == READING else SWAP.replace(READING, reading, 1)


def _trail_text() -> str:
    """#97 item 7: TRAIL, with the extra chain rules when switched on."""
    return TRAIL + G7_RULES if USE_G7_TRAIL_RULES else TRAIL


def _empty(value: Any) -> bool:
    """#104 item 3: a field that shows nothing (None, a blank text, an empty list or object)."""
    return value is None or (isinstance(value, str) and not value.strip()) or (isinstance(value, (list, dict)) and not value)


def _drop_empty(node: Any, depth: int=0) -> Any:
    """#104 item 3: a reply with the empty fields of its listing rows left out (a list of two or more objects); the rest of
    the reply as it was."""
    if depth > 4:
        return node
    if isinstance(node, list):
        rows = len(node) >= 2 and all((isinstance(r, dict) for r in node))
        return [{k: v for k, v in r.items() if not _empty(v)} if rows else _drop_empty(r, depth + 1) for r in node]
    if isinstance(node, dict):
        return {k: _drop_empty(v, depth + 1) for k, v in node.items()}
    return node


def _price_value(value: Any) -> float | None:
    """#110 item 2: a price field's amount (a number, a number in text with an optional sign or unit, or the first number of
    an object), as the king's price_of reads its learned key."""
    if isinstance(value, dict):
        value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
    if isinstance(value, str):
        figure = re.fullmatch('\\s*[^\\d\\s]{0,3}\\s?(\\d[\\d,]*(?:\\.\\d+)?)\\s*[A-Za-z]{0,6}\\.?\\s*', value)
        value = float(figure.group(1).replace(',', '')) if figure else None
    return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None


def _bound_word(word: str) -> bool:
    """#97 item 10: whether a word is one of the king's bound or condition words (OPEN_NOT), so it is no unit."""
    return bool(word) and bool(re.fullmatch(OPEN_NOT, word.strip(), re.I))


def _unit_hit(pattern: str, text: str, group: int) -> Any:
    """#97 item 10: the first match of a figure pattern; with G10 the first whose unit (`group`) is no bound or condition word."""
    for hit in re.finditer(pattern, text):
        if not (USE_G10_MEASURED_LIMITS and _bound_word(hit.group(group) or '')):
            return hit
    return None


def _same_stem(a: str, b: str) -> bool:
    """#97 item 14: two all-letter words as forms of one stem: a shared first part of four or more letters and each leftover
    ending empty or two to four letters holding a vowel (y counts); words one letter apart never match this way."""
    a, b = (a.lower(), b.lower())
    if a == b:
        return True
    if not (a.isalpha() and b.isalpha()):
        return False
    size = 0
    while size < min(len(a), len(b)) and a[size] == b[size]:
        size += 1
    if size < 4:
        return False
    for rest in (a[size:], b[size:]):
        if rest and not (2 <= len(rest) <= 4 and re.search('[aeiouy]', rest)):
            return False
    return True


def _words_hold(want: set[str], have: set[str]) -> bool:
    """#97 items 13/14: every wanted word among the held words (with G14 a form of one stem counts)."""
    return bool(want) and all((w in have or (USE_G14_WORD_FORMS and any((_same_stem(w, h) for h in have))) for w in want))


def _initials(words: list[str]) -> set[str]:
    """#97 item 13: the initials of three or more words in a row ("small form factor" -> "sff")."""
    return {''.join((w[0] for w in words[i:j])) for i in range(len(words)) for j in range(i + 3, len(words) + 1)}


def _bare_note(value: Any, limit: dict[str, Any] | None) -> str:
    """#97 item 12: for an attribute value that is only a number, BARE_OFF times or more above or below a measured limit's
    first figure: a note giving how it reads in the requirement's own unit; '' otherwise."""
    if not limit or not limit.get('unit') or not limit.get('num') or limit['num'] <= 0:
        return ''
    bare = _bare_number(value)
    if bare is None or bare <= 0 or max(bare / limit['num'], limit['num'] / bare) < BARE_OFF:
        return ''
    return f" [a bare number: read in the requirement's unit it is {bare:g} {limit['unit']}]"


def _measured_limit(value: str) -> dict[str, Any] | None:
    """#97 item 10: a requirement value that is a measured limit: a figure with a unit (no bound word read as the unit) and either
    a decimal point or a bound/condition word; a plain whole amount is none. Without G10, Lion's reading (bounded or decimal)."""
    limit = _need_limit(value)
    if not limit:
        return None
    if limit['decimal'] or limit['kind'] != 'exact' or (USE_G10_MEASURED_LIMITS and re.search(OPEN_NOT, str(value), re.I)):
        return limit
    return None


def _compact(text: Any) -> str:
    """#97 items 40/41: a text as letters and digits only (spaces and signs ignored), lower case."""
    return re.sub('[^a-z0-9]+', '', str(text or '').lower())


def _run_words(text: Any) -> list[str]:
    """#97 item 70: a text's words in order, a trailing s folded on words longer than three letters (not a double s)."""
    out = []
    for word in re.findall('[a-z0-9%]+', str(text or '').lower()):
        out.append(word[:-1] if len(word) > 3 and word.endswith('s') and not word.endswith('ss') else word)
    return out


def _split_run(option: str, text: Any) -> bool | None:
    """#97 item 70: for a plain wanted value of two or more words: True when a list part of `text` holds its words as one
    run, False when no part does while some part holds them in the same order with other words between, None otherwise
    (one word, or the words not all in one part in order)."""
    want = [w for w in _run_words(option) if w not in FILLER_WORDS]
    if len(want) < 2:
        return None
    gapped = False
    for part in re.split('[,/;|]', str(text or '')):
        have = [w for w in _run_words(part)]
        size = len(want)
        if any(have[i:i + size] == want for i in range(len(have) - size + 1)):
            return True
        at = 0
        for word in have:
            if at < size and word == want[at]:
                at += 1
        gapped = gapped or at == size
    return False if gapped else None


def _plain_words_value(value: str) -> bool:
    """#97 item 70: a word requirement the split-words rule applies to: two or more words, no limit, exclusion, measure,
    bound/condition word or list separator."""
    text = str(value or '')
    if re.search('[,/;|]', text) or re.search('\\d', text) or re.search(OPEN_NOT, text, re.I) or re.search(NEGATION, text, re.I):
        return False
    return len([w for w in _run_words(text) if w not in FILLER_WORDS]) >= 2

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

def _restore(value: Any, depth: int=0, in_record: bool=False, page: bool=True) -> Any:
    """A store result in the one form the rest of the agent reads: lists of row objects, a listing's attributes as a nested
    object. Worlds spell results differently (race 15 gave each record as one text, its attributes "a=b; c=d" inside it, and
    a result the agent cannot read leaves every record-based check blind), so the spelling is undone here, once, for all."""
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
            return None
    return None

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

def _cut_words(value: str, limit: int) -> str:
    """#111 E16: `value` cut to `limit` characters at the last space before the limit when the cut would fall inside a word."""
    cut = value[:limit]
    if limit <= 0 or value[limit:limit + 1].isspace() or cut[-1:].isspace() or ' ' not in cut.strip():
        return cut.rstrip() or cut
    return cut.rsplit(' ', 1)[0].rstrip() or cut

def _shape(value: Any, schema: Any, where: str, faults: list[str], cuts: list[tuple[str, int, str]] | None=None) -> Any:
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
        if USE_E16_WORD_BOUNDARY_CUT:
            value = _cut_words(value, schema['maxLength'])
            if cuts is not None:
                cuts.append((where, schema['maxLength'], value))
        else:
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
            value = [_shape(v, schema['items'], f'{where}[{i}]', faults, cuts) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for name in [n for n in value if n not in props]:
                faults.append(f'{where}.{name} is not an allowed argument')
        for name in schema.get('required') or []:
            if name not in value:
                faults.append(f'{where}.{name} is required')
        value = {k: _shape(v, props.get(k), f'{where}.{k}', faults, cuts) if k in props else v for k, v in value.items()}
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

HEAR_ASK = HEAR.replace('want it in simpler words], "budget"', 'want it in simpler words], "lost": [numbers of the questions whose answer says the shopper did not follow it, lost track of it or wants it put more simply], "budget"')
PLAN_ASK = PLAN.replace("2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item's listings;", "2. Then the listed keys, ranked: first the property most particular to that kind of product (how it is built or fastens, its type or style, the user it is made for, its main technical spec, what sets its listings apart), then the ones a buyer of that kind of product most often has a view on, favouring keys carried by most of the item's listings with values that differ between listings; generic properties almost every product has (colour, size, pattern, material) come after those unless a hint points at them, since a shopper more often does not mind them;")
CH_PLAN = 'Rank what an assistant buying for a shopper should ask about. Besides what the request says, the shopper keeps a few firm wishes to themselves, each tied to one property of one product on the list, and reveals a wish only when a message mentions that exact property (its name or an everyday synonym). One message may list several properties, yet the shopper tends to reply about the property listed first and gets less reliable further down the list; a property they do not care about gets only a shrug. So the ranking is what counts.\n\nInput: the request; the subjects it asks to be checked with the shopper; the wishes it mentions; and per product its number, name, shelf label, requirements already known, properties already asked about, and a MENU of the properties seen on its opened listings, each with how many listings show it ("listings" out of "of") and sample values.\n\nOutput ONE JSON object only: {"questions": [{"item": product number, "property": "the property as the menu writes it, in plain words", "hint": "the request\'s own words that suggest it, or an empty string"}, ...], "junk": ["menu properties nobody would hold a wish about"]}, COUNT questions in all, strongest first.\n\nHow to rank:\n1. Clues in the story lead. Requests often suggest a hidden wish without spelling it out: the person who will wear or use the thing, the way it must close or stay put, the parts inside it, how it has to look or blend in, the wear it has to survive. Look at what the shopper keeps returning to or fusses over, not at facts they state outright (a stated age, room or size is background). Tie each clue to the nearest menu property (wearer -> audience, gender or age group; staying put -> closure or fastening; the inside -> processor or similar part; looks -> style, colour or pattern; hard wear -> material) and copy the request\'s words into "hint". A subject the request wants checked counts as a clue.\n2. With no clue, open with what distinguishes this kind of product from similar ones (its type or style, its build or fastening, the user it is meant for, its key technical figure). Leave the near-universal ones (colour, size, pattern, material) for later, since shoppers are often indifferent to them.\n3. Next, the properties people buying this kind of product usually decide on, preferring ones that many of its listings show and whose values differ from one listing to the next. One value on every listing, a variant or option chooser, a code or model number, or a warranty, delivery or packaging remark is a weak question: list such entries in "junk".\n4. Skip anything a known requirement already pins down, anything about price, stock or quantity, and repeats of one property under another spelling. A key whose values only echo a stated requirement is still open; how many units come in a pack is a normal property as well. Keep a menu property\'s full wording ("strap colour", not just "colour"). A clue property absent from every listing is placed after that product\'s three most widely shown properties.\n5. Share the questions across products according to how unsettled each one is (a product with a strong clue goes first), and give every product some questions even when its menu is empty: then use the plain names of the properties buyers of that product most commonly specify.'
CH_REPLY = 'A shopping assistant sent a shopper the message below, naming one or more properties of the products listed under PRODUCTS (in the order given), and the shopper replied. Read the reply and return ONE JSON object and nothing else:\n{"kind": "need" or "repeat" or "free" or "split" or "unclear" or "other", "item": the PRODUCTS number of the product the reply states a firm wish about (0 when it states none), "property": the property that wish is about (the shopper\'s own words when the reply names one, which may differ from the ones asked; else the asked property it answers), "op": "is" or "is not" or "at least" or "at most" or "one of" or "points", "value": the wanted value, limit, excluded value or list (list values joined by " | "), "need": "property: constraint" (the wish as one requirement line, rules below), "points": true when the wanted value is to be read from another listing\'s page, else false, "drop": [properties of earlier requirements the shopper now gives up], "budget": a new total spending limit the reply states, as a number, else null, "steps": [only when the reply itself demands something new be done before the order or the ending, such as asking their approval or showing them the cart: {"when": "order" or "end", "step": "what must be done, in at most 12 words", "messages": messages it takes}], else []}\nKinds:\n- "need": the reply insists on a value, a limit, an exclusion or a closed list for some property, however it is worded (folksy, roundabout, "will not budge on", "nothing but", "anything bar"). Decide which side the shopper turns away: accepting anything except X is op "is not" with value X; turning away everything that is not X wants X itself (op "is"), even when X sounds negative. Explicit limit words (at least, at most, no more than, under) decide the op even inside an odd sentence; without them an insisted figure is op "is".\n- "repeat": the reply only restates a requirement listed under KNOWN REQUIREMENTS instead of answering.\n- "free": no preference, it is left open, anything will do.\n- "split": the shopper complains the message bundled several subjects and asks for them one by one, or replies to just its opening part because of that.\n- "unclear": the reply shows the shopper lost the thread or wants the message repeated in plainer words, with no complaint about its number of subjects.\n- "other": none of these (small talk, a counter-question, or just a new budget).\nThe "need" line: name the property as the reply names it; copy the value word for word with its nouns and units ("at least 3 shelves", not "at least 3"), leaving out only the framing around it ("oval" for "settled on oval and nothing else"); write "not X" only when the reply rejects X itself. When the wanted value is to be taken from another listing (like, unlike or near what some shop\'s product with some properties has), the line holds the reply\'s own words from the comparison words to the end of that listing\'s description, never shortened or reworded, with "points": true and op "points"; the properties describing that other product are not wishes on this one.'
CH_SIDE = 'Below: a reply a shopper gave when asked about one product property, and one value taken from it. Decide the shopper\'s stance on it: "insist" when they want that value and turn away all others, "reject" when they turn that value away and accept the rest. Watch for double negatives: refusing everything except a value means insisting on it, even when the value sounds negative itself. Output ONE JSON object only: {"stance": "insist"} or {"stance": "reject"}.'
CH_SPLIT = 'A message to a shopper bundled several questions; sort the shopper\'s answer into one of three cases. "bundled": they object to the bundling as such (too many things at once, want them one by one, will only take the first). "confused": they lost the thread, want it repeated or put more simply, without objecting to how many things were asked. "answered": anything else (a value, a wish, no preference). Output ONE JSON object only: {"verdict": "bundled"}, {"verdict": "confused"} or {"verdict": "answered"}.'
CH_NEGATED = '^\\s*(?:(?:not|no)(?!\\s+(?:more|less|fewer|over|under|above|below)\\b)(?!\\s+\\w+er\\s+than\\b)|never|anything but|anything except|anything other than|any but|everything but|all but|other than|except|excluding|apart from)\\b'


def _ch_plain(text: Any) -> str:
    """A name or a reply in plain lower-case words (camel case and marks split), for whole-word lookups."""
    return ' '.join(re.findall('[a-z0-9]+', _words(str(text or ''))))


def _ch_set(text: Any) -> frozenset[str]:
    return _spelled(_ch_plain(text))


def _ch_same(a: Any, b: Any) -> bool:
    """Two property names that are one property in another spelling (colours / colour, lidColour / lid color)."""
    sa = _ch_set(a)
    return bool(sa) and sa == _ch_set(b)


def _ch_subject(a: Any, b: Any) -> bool:
    """Two names a shopper hears as one subject: the same property, or a name of two or more words holding all of the
    other's words. A one-word name stays apart from the longer names holding it ("type" beside "mounting type")."""
    if _ch_same(a, b):
        return True
    sa, sb = (_ch_set(a), _ch_set(b))
    return min(len(sa), len(sb)) >= 2 and (sa <= sb or sb <= sa)


def _ch_overlap(a: Any, b: Any) -> float:
    sa, sb = (_ch_set(a), _ch_set(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

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

def _conditional(need: str) -> bool:
    """#111 E4: whether a requirement's value holds a waiver or restriction clause and says nothing about a page, a pointer
    or what a shop sells (such a rule is read from the cart, never chased as a pointer)."""
    value = need.partition(':')[2] if ':' in need else need
    return bool(re.search(COND_CLAUSE, value, re.I)) and not re.search(POINTER_TALK, value, re.I)

def _sentences(text: str) -> list[str]:
    """The request cut at sentence ends and semicolons (a decimal point is no end)."""
    return [s.strip() for s in re.split('(?<=[.!?;])\\s+|\\n+', str(text or '')) if s.strip()]

def _values_in(values: Any, text: str) -> bool:
    """USE_R16_LINK_PAGE_UNSHOWN: whether a text holds every value word as a whole word or in another form of it (plural
    -s folded the way _fold folds it)."""
    held = {_fold(w) for w in re.findall('[a-z0-9]+', str(text).lower())}
    return all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) or (_fold(v) and _fold(v) in held) for v in values if v))

def _word_roots(text: str) -> list[str]:
    """USE_R16_SCOPE_ROOTS: a text's words folded the way _roots folds them, in order with repeats, punctuation left out (a
    word written "pattern;" is the word pattern)."""
    return [r for w in re.findall('[A-Za-z0-9]+', str(text or '')) for r in _roots(w)]

def _clauses(sentence: str) -> list[str]:
    """USE_R16_SCOPE_ROOTS: a sentence cut at dashes standing between words, commas, colons and semicolons."""
    return [c.strip() for c in re.split('\\s[\u2014\u2013-]+\\s|[,;:\u2014\u2013]', str(sentence or '')) if c.strip()]

def _root(word: str) -> str:
    """One word folded the way _roots folds it ('' for a short or filler word)."""
    found = _roots(word)
    return next(iter(found)) if len(found) == 1 else ''

def _relation_part(sentence: str, prop: str) -> str:
    """#111 E3: the part of a request sentence from its main verb on: after the property's first mention, past an auxiliary
    when one follows it (else past leading function words)."""
    props = _roots(prop)
    tokens = list(re.finditer("[A-Za-z0-9']+", sentence))
    start = 0
    for index, token in enumerate(tokens):
        if props and _root(token.group(0)) in props:
            end = index
            while end + 1 < len(tokens) and _root(tokens[end + 1].group(0)) in props:
                end += 1
            start = tokens[end].end()
            break
    rest = sentence[start:]
    aux = re.search(AUX_WORDS, rest, re.I)
    part = rest[aux.end():] if aux else rest
    return re.sub(LEAD_FILLER, '', part.strip(' ,;:.'), flags=re.I).strip(' ,;:.')

def _name_pattern(name: str) -> str:
    """An item name as a pattern: its words in order, any spacing or hyphen between, an optional plural ending."""
    words = [re.escape(w) for w in re.findall('[A-Za-z0-9]+', name)]
    return '[\\s-]+'.join(words) + '(?:e?s)?\\b' if words else ''

def _count_before_name(request: str, name: str, others: list[str] | None=None) -> int | None:
    """#111 E5: the one number (QTY_LOW..QTY_HIGH) the request writes right before the item's full name, optionally with a
    times sign; None when there is none or more than one."""
    pattern = _name_pattern(name)
    if not pattern:
        return None
    if USE_T_E5_NAME_GUARD:
        # every number written before the name counts (a written 1 makes a 2 elsewhere ambiguous), and a place where a longer
        # requested item's name begins is that item's count, not this one's ('2 sofa covers' is no count of 'sofa')
        size = len(re.findall('[A-Za-z0-9]+', name))
        longer = [_name_pattern(o) for o in others or [] if o and len(re.findall('[A-Za-z0-9]+', o)) > size]
        seen: set[int] = set()
        for hit in re.finditer('(?<![\\w.,])(\\d{1,2})\\s*(?:[x×*]\\s*)?(?=' + pattern + ')', request, re.I):
            if any((p and re.match(p, request[hit.end():], re.I) for p in longer)):
                continue
            seen.add(int(hit.group(1)))
        only = seen.pop() if len(seen) == 1 else None
        return only if only is not None and QTY_LOW <= only <= QTY_HIGH else None
    found = {int(hit.group(1)) for hit in re.finditer('(?<![\\w.,])(\\d{1,2})\\s*(?:[x×*]\\s*)?' + pattern, request, re.I)}
    found = {n for n in found if QTY_LOW <= n <= QTY_HIGH}
    return found.pop() if len(found) == 1 else None

def _times_counts(request: str) -> list[tuple[int, str]]:
    """#111 E6: counts of one or two digits written in multiplication form ("3 x", "3x ", "3 ×") with the first words after them."""
    out = []
    for hit in re.finditer('(?<![\\w.,])(\\d{1,2})\\s*[x×*]\\s+(?=[A-Za-z])', request, re.I):
        tail = re.split('[.,;:!?()\\n]', request[hit.end():], maxsplit=1)[0]
        words = ' '.join(tail.split()[:TIMES_WORDS])
        if words and int(hit.group(1)) >= 1:
            out.append((int(hit.group(1)), words))
    return out

_FIGURE = '(?<![\\w.,])(\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(\\d+))?(?![\\d])'

def _figures(text: str) -> list[tuple[float, int, int]]:
    """Every plain number in a text: (value, start, end)."""
    out = []
    for hit in re.finditer(_FIGURE, text):
        whole = hit.group(1).replace(',', '') + ('.' + hit.group(2) if hit.group(2) else '')
        out.append((float(whole), hit.start(), hit.end()))
    return out

def _leaves(node: Any, depth: int=0) -> list[str]:
    """#111: every text and number a record holds, nested blocks as "key value"."""
    if depth > 6:
        return []
    if isinstance(node, dict):
        out = []
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                out += _leaves(value, depth + 1)
            elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
                out.append(f'{key} {value}' if depth else str(value))
        return out
    if isinstance(node, list):
        return [s for v in node for s in _leaves(v, depth + 1)]
    return [str(node)] if isinstance(node, (str, int, float)) and not isinstance(node, bool) else []

def _text_codes(text: str) -> list[str]:
    """#111 E24: the code-like words a listing's text holds, in any case, spelled as _codes spells them."""
    return [re.sub('[\\s-]+', ' ', hit.group(0)).strip().lower() for hit in re.finditer('\\b' + FIT_CODE + '\\b', str(text or ''), re.I)]

def _code_match(said: str, code: str) -> bool:
    """#111 E24/E28: whether a code a listing names covers a required code; with E28 also a required series covering it."""
    return _code_hit(said, code) or (USE_E28_CODE_EITHER_WAY and _code_hit(code, said))

def _covers_code(pattern: str, code: str) -> bool:
    """#111 E28: the king's coverage test (`pattern` covers `code`), with E28 applied in both directions."""
    return _code_hit(pattern, code) or (USE_E28_CODE_EITHER_WAY and _code_hit(code, pattern))

def _code_prefix(code: str) -> str:
    """#111 E26/E42: a code's family: its leading letters."""
    found = re.match('[a-z]+', re.sub('[\\s-]+', '', code.lower()))
    return found.group(0) if found else ''

def _fold(word: str) -> str:
    word = re.sub('[^a-z0-9%]+', '', word.lower())
    word = word[:-1] if len(word) > 3 and word.endswith('s') and not word.endswith('ss') else word
    return re.sub('our$', 'or', word)

def _key_words(text: str) -> list[str]:
    """#111: a key's or property's words in order, a bracketed part left out, each folded."""
    words = [_fold(w) for w in _words(re.sub('\\([^)]*\\)|\\[[^\\]]*\\]', ' ', str(text))).split()]
    return [w for w in words if w]

def _key_named(key: str, prop: str) -> bool:
    """#111 E39/E43/E49: whether an attribute key is named for a requirement's property: the same words, or the same last word."""
    k, p = (_key_words(key), _key_words(prop))
    return bool(k and p) and (set(k) == set(p) or k[-1] == p[-1])

def _lit_words(text: str) -> set[str]:
    """#111 E49/E50: a value's words for the literal word check; with E50 list punctuation is no part of a word."""
    low = _words(str(text))
    if USE_E50_LITERAL_LIST_SEPARATORS:
        low = re.sub(LIST_MARKS, ' ', low)
    out = set()
    for word in low.split():
        bare = re.sub('^[^\\w%]+|[^\\w%]+$', '', word) if USE_E50_LITERAL_LIST_SEPARATORS else word
        bare = bare[:-1] if len(bare) > 3 and bare.endswith('s') and not bare.endswith('ss') else bare
        if bare and bare not in FILLER_WORDS:
            out.add(bare)
    return out

def _unit_norm(unit: str) -> str:
    low = re.sub('[^a-z%°µ"\']+', '', unit.lower())
    return low[:-1] if len(low) > 2 and low.endswith('s') and not low.endswith('ss') else low

def _same_unit(a: str, b: str) -> bool:
    """#111: two unit spellings for one unit, loosely: equal, or one the other's start (in / inch / inches)."""
    x, y = (_unit_norm(a), _unit_norm(b))
    if not x or not y:
        return False
    short, long = sorted((x, y), key=len)
    return x == y or (len(short) >= 2 and long.startswith(short) and len(long) - len(short) <= 4)

def _figure(whole: str, part: str | None) -> float:
    return float(whole.replace(',', '') + ('.' + part if part else ''))

def _bare_number(value: Any) -> float | None:
    """#111 E38: a value that is a number with nothing else."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(?:\\.\\d+)?\\s*', value):
        return float(value)
    return None

def _unit_figures(text: Any, unit: str, bare: bool=True) -> tuple[list[float], bool]:
    """#111: the figures a value writes in `unit` (both ends of a range) and whether it writes a range; with `bare`, a value
    that is only a number counts as written in that unit."""
    text = str(text)
    figs: list[float] = []
    ranged = False
    for hit in re.finditer(FIG_RANGE, text):
        if _same_unit(hit.group(3), unit):
            figs += [float(hit.group(1)), float(hit.group(2))]
            ranged = True
    rest = re.sub(FIG_RANGE, ' ', text)
    for hit in re.finditer(FIG_UNIT, rest):
        if _same_unit(hit.group(3), unit):
            figs.append(_figure(hit.group(1), hit.group(2)))
    if not figs and bare:
        number = _bare_number(text)
        if number is not None:
            figs = [number]
    return (figs, ranged)

def _need_limit(value: str) -> dict[str, Any] | None:
    """#111 E38-E49: a requirement value holding a figure with a unit: {kind: min/max/range/exact, lo, hi, strict, num, unit, decimal}."""
    text = str(value or '')
    between = re.search('\\bbetween\\s+' + _NUM + '\\s*(?:[A-Za-z%°µ"\']{1,12}\\s*)?(?:and|to|-|–)\\s*' + _NUM + '\\s*-?\\s*' + _UNIT, text, re.I)
    if between and USE_G10_MEASURED_LIMITS and _bound_word(between.group(5) or ''):
        between = None
    if between:
        lo, hi = (_figure(between.group(1), between.group(2)), _figure(between.group(3), between.group(4)))
        return {'kind': 'range', 'lo': min(lo, hi), 'hi': max(lo, hi), 'strict': False, 'num': lo, 'unit': between.group(5), 'decimal': bool(between.group(2) or between.group(4))}
    span = _unit_hit(FIG_RANGE, text, 3)
    hit = _unit_hit(FIG_UNIT, text, 3)
    if span and (not hit or span.start() <= hit.start()) and not re.search(LIMIT_MIN + '|' + LIMIT_MAX + '|' + LIMIT_MIN_STRICT + '|' + LIMIT_MAX_STRICT, text[:span.start()], re.I):
        lo, hi = (float(span.group(1)), float(span.group(2)))
        return {'kind': 'range', 'lo': min(lo, hi), 'hi': max(lo, hi), 'strict': False, 'num': lo, 'unit': span.group(3), 'decimal': '.' in span.group(1) + span.group(2)}
    if not hit:
        return None
    num = _figure(hit.group(1), hit.group(2))
    before, after = (text[:hit.start()], text[hit.end():])
    kind, strict = ('exact', False)
    if re.search(LIMIT_MIN, before, re.I) or re.search(LIMIT_MIN_AFTER, after, re.I):
        kind = 'min'
    elif re.search(LIMIT_MAX, before, re.I) or re.search(LIMIT_MAX_AFTER, after, re.I):
        kind = 'max'
    elif re.search(LIMIT_MIN_STRICT, before, re.I):
        kind, strict = ('min', True)
    elif re.search(LIMIT_MAX_STRICT, before, re.I):
        kind, strict = ('max', True)
    return {'kind': kind, 'lo': num, 'hi': num, 'strict': strict, 'num': num, 'unit': hit.group(3), 'decimal': hit.group(2) is not None}

def _meets_limit(limit: dict[str, Any], figs: list[float], ranged: bool) -> bool:
    """#111 E49: whether figures written in the limit's unit meet it read literally (a range must lie wholly inside; one
    element of a list is enough)."""
    if not figs:
        return False
    eps = 1e-09
    kind = limit['kind']
    if kind == 'min':
        ok = lambda f: f > limit['lo'] + eps if limit['strict'] else f >= limit['lo'] - eps
    elif kind == 'max':
        ok = lambda f: f < limit['hi'] - eps if limit['strict'] else f <= limit['hi'] + eps
    elif kind == 'range':
        ok = lambda f: limit['lo'] - eps <= f <= limit['hi'] + eps
    else:
        return any((abs(f - limit['num']) < eps for f in figs))
    return all((ok(f) for f in figs)) if ranged else any((ok(f) for f in figs))

def _breaks_limit(limit: dict[str, Any], figs: list[float]) -> bool:
    """#111 E45: whether any figure breaks an at-least / at-most limit."""
    if limit['kind'] not in ('min', 'max', 'range'):
        return False
    return any((not _meets_limit(limit, [f], False) for f in figs))

def _flip_spacing(value: str) -> str:
    """#111 E36: a value with its number-unit spacing flipped: "16 GB" as "16GB", "16GB" as "16 GB"."""
    if re.search('\\d\\s+[A-Za-z]{1,5}\\b', value):
        return re.sub('(\\d)\\s+([A-Za-z]{1,5})\\b', '\\1\\2', value)
    return re.sub('(\\d)([A-Za-z]{1,5})\\b', '\\1 \\2', value)

def _plain_text(text: str) -> str:
    """#111 E52: a text without spacing, punctuation or case, to find one sentence inside another."""
    return re.sub('[^a-z0-9]+', '', str(text).lower())

class ChidoriAsk:
    """The question engine behind USE_CHIDORI_ASK: Cairn's own code for the design of Chidori's asking (a teammate's
    code-driven agent, first in the arena's ask-engine test), working on Cairn's items, ledger and message accounting.

    - planner (ch_plan): ONE model call over the request, the subjects it says to check, its wishes and per item the
      known requirements, the properties already asked and a property menu built from the opened listings' keys
      (listings carrying each, example values; keys holding the same values merged; a key every listing gives one value,
      or every listing its own value, weighs half). Rules: story pointers first, the property that sets the kind of
      product apart before generic colour/size/pattern/material, then the most carried with varying values; settled,
      junk and identifiers skipped. Code keeps the pointed-at properties right after the item's first pick and the
      CH_TOPS most carried among the first questions, then pads each item's queue from its menu.
    - queues (ch_build): one per item; the items are taken in turn, those whose need was told last.
    - widths (ch_width): the first list is short (CH_FIRST); a list answered without complaint lets the next grow by
      one up to CH_CALM; a need told from past a list's first label opens CH_FULL; a reply that lost a long list halves
      the widths; a reply wanting one subject per message ends lists for the episode (single questions follow).
    - order after a reply (ch_list): a told need settles its label and every label before it, the labels after it lead
      the next list; a reply with no need settles only the first label and the list's later labels lead the next one in
      the order they were named, so a label named second gets the first place of the next message.
    - mixed messages (ch_mixed): when more items are still unheard than messages are free, one message names a few
      labels of each of a share of them.
    - reply reading (ch_read): one model call (kind need/repeat/free/split/unclear/other, item, property, op, value,
      the requirement line, drops, budget, new process steps); an exclusion gets a second reading by the stronger
      model; a split or lost reply to a list is told apart by a small check; a need the reply files under another
      property the listings carry is filed under that property. Told needs, drops and budgets go through take_heard,
      so they reach item.told and the ledger exactly as the king's reader's do."""

    def ch_reset(self) -> None:
        self.ch_planned: list[tuple[int, str]] | None = None  # the planner's order with the hint rule applied, once an episode
        self.ch_hinted: dict[int, list[str]] = {}  # per item: properties a pointer of the request names
        self.ch_junk: set[str] = set()  # menu properties the planner calls junk (plain spelling)
        self.ch_menus: dict[int, list[dict[str, Any]]] = {}
        self.ch_queues: dict[int, list[str]] = {}  # per item: the labels still to name, next first
        self.ch_order: list[int] = []  # the items' turn order
        self.ch_asked: dict[int, list[str]] = {}  # per item: labels a reply settled
        self.ch_named: dict[int, int] = {}  # per item: labels named so far
        self.ch_heard: set[int] = set()  # items whose need the shopper told
        self.ch_last = 0  # the item the last message was about
        self.ch_said: dict[int, list[str]] = {}  # per item: the plain words of replies that told a need
        self.ch_follow: list[tuple[int, str]] = []  # single questions to ask again, narrowed
        self.ch_narrowed: set[tuple[int, str]] = set()
        self.ch_late = False  # the late question: the ending's kept messages are free
        self.ch_steps_added = 0

    # ---------------- bookkeeping ----------------
    def ch_free(self) -> int:
        return self.msg_limit - self.msg_used if self.ch_late else self.msgs_free()

    def ch_records(self, number: int) -> list[dict[str, Any]]:
        """The item's opened listing records with a fact block, in the order seen (on a one-item task, every record when
        none carries the item's label)."""
        item = self.items[number - 1]
        cat_key = self.category_key()
        records = [self.full[i] for i in self.seen_order if isinstance(self.full.get(i), dict) and _fact_block(self.full[i])]
        own = [r for r in records if cat_key and self.owns(item, str(r.get(cat_key) or ''))]
        return own if own or len(self.items) > 1 else records

    def ch_menu(self, number: int) -> list[dict[str, Any]]:
        """The properties the item's opened listings carry, heaviest first: keys whose values agree on 80% of the listings
        holding both are one property (the plainer, more carried name speaks for it); a key every listing gives one value,
        or (on 4 or more) every listing its own value, weighs half; identifiers, option lists and long texts are left out."""
        per: list[dict[str, str]] = []
        for record in self.ch_records(number):
            facts: dict[str, str] = {}
            for key, value in (_fact_block(record) or {}).items():
                name = _ch_plain(key)
                if name and name not in facts:
                    facts[name] = str(value).strip().lower()[:60]
            per.append(facts)
        counts: dict[str, int] = {}
        for facts in per:
            for key in facts:
                counts[key] = counts.get(key, 0) + 1
        keys = sorted(counts, key=lambda k: (-counts[k], len(k), k))
        alias: dict[str, str] = {}
        for i, a in enumerate(keys):
            if a in alias:
                continue
            for b in keys[i + 1:]:
                if b in alias:
                    continue
                both = [f for f in per if a in f and b in f]
                if len(both) >= 2 and sum((1 for f in both if f[a] == f[b])) >= 0.8 * len(both):
                    alias[b] = a
        menu: list[dict[str, Any]] = []
        for key in keys:
            if key in alias or (re.search(NOT_PROPERTY, key) and not re.search('\\w[ _]models?\\b', key)):
                continue
            names = [key] + [b for b, a in alias.items() if a == key]
            values: list[str] = []
            for facts in per:
                value = next((facts[n] for n in names if n in facts), None)
                if value and value not in values:
                    values.append(value)
            if not values:
                continue
            if sum((len(v) for v in values)) / len(values) > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values):
                continue
            carriers = sum((1 for f in per if any((n in f for n in names))))
            weight = carriers * (0.5 if len(values) == 1 and carriers >= 3 else 1.0)
            if carriers >= 4 and len(values) >= carriers:
                weight *= 0.5
            menu.append({'property': key, 'listings': carriers, 'of': len(per), 'values': values[:3], 'same_as': names[1:4], 'all': values, 'w': weight})
        menu.sort(key=lambda e: (-e['w'], e['property']))
        return menu[:CH_MENU]

    def ch_apart(self, number: int, label: str, value: str) -> bool:
        """Whether a menu property whose name holds a requirement's (or is held by it) is another property: the
        requirement's value words are not all among the values the item's listings give it."""
        if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
            return False
        words = set(re.findall('[a-z0-9]{3,}', value.lower()))
        entry = next((e for e in self.ch_menus.get(number) or [] if _ch_same(e['property'], label)), None)
        if not words or not entry:
            return False
        return not words <= set(re.findall('[a-z0-9]{3,}', ' '.join(entry['all']).lower()))

    def ch_told_before(self, number: int, label: str) -> bool:
        """A label of two or more words a reply that told a need already used whole."""
        name = _ch_plain(label)
        return len(name.split()) >= 2 and any((f' {name} ' in line for line in self.ch_said.get(number, [])))

    def ch_settled(self, number: int, label: str) -> bool:
        """A label not worth a question: asked (the same subject), told, said not to matter, called junk, fixed by a
        requirement of the request (any spelling, a name holding the other's unless the listings show it is another
        property) or decided by a fit rule."""
        item = self.items[number - 1]
        if not _ch_set(label):
            return True
        if any((_ch_subject(label, a) for a in self.ch_asked.get(number, []))):
            return True
        if self.told_about(number, label) or self.ch_told_before(number, label):
            return True
        if any((_ch_same(label, f) for f in item.free)) or _ch_plain(label) in self.ch_junk:
            return True
        mine = _ch_set(label)
        for must in item.musts + item.dropped:
            prop, _, value = must.partition(':')
            have = _ch_set(prop)
            if not have:
                continue
            if have == mine:
                return True
            if (_names(have, mine) or _names(mine, have)) and (not self.ch_apart(number, label, value)):
                return True
        if any((part == number for part, _ in self.bound)) and (_ch_overlap(label, 'compatibility') >= 0.5 or _ch_same(label, 'compatible with')):
            return True
        return False

    def ch_item(self, said: Any) -> int:
        number = _count(said)
        if number is not None:
            return min(max(1, number), len(self.items))
        text = str(said or '')
        scores = [_ch_overlap(text, item.name) for item in self.items]
        return scores.index(max(scores)) + 1 if scores and max(scores) > 0 else 1

    def ch_label(self, number: int, said: Any) -> str:
        """A planner's property as a label: its first name only (no second property, no brackets), in the menu's
        spelling when the menu holds it."""
        label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(said or ''))[0].strip().rstrip('?.').strip()
        if not re.search('[a-z0-9]', label.lower()) or len(label.split()) > 5:
            return ''
        known = next((e['property'] for e in self.ch_menus.get(number) or [] if _ch_same(e['property'], label)), None)
        return known or ' '.join(_words(label).split())

    # ---------------- planner ----------------
    def ch_plan(self) -> None:
        """One planner call for the episode; the code then keeps each item's pointed-at properties right after its first
        pick (in the order the pointer gives them, a longer name before one it holds)."""
        self.adopt_labels()
        count = len(self.items)
        self.ch_menus = {n: self.ch_menu(n) for n in range(1, count + 1)}
        left = max(1, self.ch_free())
        total = min(12 * count, max(sum((self.ch_width(k) for k in range(left))), 6 + 4 * min(left, 2)) + 2 * count)
        blocks = []
        for n, item in enumerate(self.items, 1):
            live = [m for m in item.musts + item.told if m not in item.dropped]
            menu = [e for e in self.ch_menus[n] if not self.ch_settled(n, e['property'])]
            blocks.append({'item': n, 'name': item.name, 'store label': item.shelf or '', 'known requirements': live, 'already asked': list(self.ch_asked.get(n, [])),
                           'menu': [dict({'property': e['property'], 'listings': e['listings'], 'of': e['of'], 'examples': e['values']}, **({'other names': e['same_as']} if e['same_as'] else {})) for e in menu]})
        subjects = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        wishes = [str(x) for x in _items(self.brief.get('wishes')) if str(x).strip()]
        prompt = CH_PLAN.replace('COUNT', str(total)) + '\n\nREQUEST:\n' + self.request + '\n\nSUBJECTS THE REQUEST SAYS TO CHECK WITH THE SHOPPER: ' + _dump(subjects) + '\nWISHES THE REQUEST VOICES: ' + _dump(wishes) + '\n\nITEMS:\n' + _dump(blocks)
        got = self.ask_json(prompt, [HEAVY, LIGHT], CH_PLAN_TOKENS) or {}
        ordered: list[tuple[int, str]] = []
        marked: dict[int, list[tuple[str, str]]] = {n: [] for n in range(1, count + 1)}
        for pick in _items(got.get('questions')):
            if not isinstance(pick, dict):
                continue
            number = self.ch_item(pick.get('item'))
            label = self.ch_label(number, pick.get('property') or pick.get('label'))
            if not label or any((n == number and _ch_same(label, p) for n, p in ordered)):
                continue
            ordered.append((number, label))
            if str(pick.get('hint') or '').strip():
                marked[number].append((label, str(pick['hint'])))
        self.ch_junk = {_ch_plain(j) for j in _items(got.get('junk')) if isinstance(j, str) and j.strip()}
        phrases: dict[int, list[str]] = {n: [h for _, h in marked[n]] for n in marked}
        for text in subjects + wishes:
            targets = [n for n, item in enumerate(self.items, 1) if _ch_set(item.name) & _ch_set(text)] or list(range(1, count + 1))
            for n in targets:
                phrases[n].append(text)
        hinted: dict[int, list[str]] = {n: [] for n in marked}
        for n in hinted:
            menu = self.ch_menus[n]
            names = list(dict.fromkeys([e['property'] for e in menu] + [p for m, p in ordered if m == n]))
            for label, _ in marked[n]:
                if any((_ch_same(label, e['property']) for e in menu)) and not any((_ch_subject(label, h) for h in hinted[n])):
                    hinted[n].append(label)
            for phrase in phrases[n]:
                words = _ch_set(phrase)
                low = _ch_plain(phrase)
                found = [p for p in names if _ch_set(p) and _ch_set(p) <= words]
                found.sort(key=lambda p: (min((low.find(w) for w in _ch_plain(p).split() if low.find(w) >= 0), default=999), -len(_ch_set(p))))
                for p in found:
                    if not any((_ch_subject(p, h) for h in hinted[n])):
                        hinted[n].append(p)
        merged: list[tuple[int, str]] = []
        for n, p in ordered + [(n, h) for n in hinted for h in hinted[n]]:
            if any((m == n and _ch_subject(p, q) for m, q in merged)):
                continue
            merged.append((n, p))
            if sum((1 for m, _ in merged if m == n)) == 1:
                for h in hinted[n]:
                    if not any((m == n and _ch_subject(h, q) for m, q in merged)):
                        merged.append((n, h))
        self.ch_planned = merged
        self.ch_hinted = hinted
        self.trace.append({'kind': 'ch_plan', 'count': total, 'asked': _clip(got.get('questions'), 80), 'junk': sorted(self.ch_junk)[:10], 'hinted': {n: v[:4] for n, v in hinted.items() if v}})

    def ch_build(self) -> None:
        """Each item's queue: its planned labels (unsettled, one per subject), its CH_TOPS most carried properties placed
        among its first questions (second and fourth on one item, second and third on several; never ahead of the
        pointed-at properties that follow its first pick), then its menu; names lumping several features go last. The
        items take turns in the planner's order, those whose need is already told last, then those asked least."""
        count = len(self.items)
        queues: dict[int, list[str]] = {n: [] for n in range(1, count + 1)}
        first: list[int] = []
        for n, p in self.ch_planned or []:
            if self.ch_settled(n, p) or any((_ch_subject(p, q) for q in queues[n])):
                continue
            queues[n].append(p)
            if n not in first:
                first.append(n)
        for n, q in queues.items():
            menu = [e for e in self.ch_menus.get(n) or [] if e['listings'] >= 2 or e['of'] <= 2]
            tops = [e['property'] for e in menu if e['listings'] >= CH_TOP_MIN and (not self.ch_settled(n, e['property'])) and (not re.search(CATCH_ALL, e['property']))][:CH_TOPS]
            lead = 1
            while lead < len(q) and any((_ch_same(q[lead], h) for h in self.ch_hinted.get(n, []))):
                lead += 1
            for k, top in enumerate(tops):
                target = max(1 + 2 * k if count == 1 else 1 + k, lead + k)
                at = next((i for i, p in enumerate(q) if _ch_subject(p, top)), None)
                if at is not None and at <= target:
                    continue
                q[:] = [p for p in q if not _ch_subject(p, top)]
                q.insert(min(target, len(q)), top)
            for e in menu:
                if not self.ch_settled(n, e['property']) and (not any((_ch_subject(e['property'], p) for p in q))):
                    q.append(e['property'])
            queues[n] = [p for p in q if not re.search(CATCH_ALL, _words(p))] + [p for p in q if re.search(CATCH_ALL, _words(p))]
            if n not in first:
                first.append(n)
        told = {n for n, item in enumerate(self.items, 1) if item.told}
        self.ch_heard |= told
        self.ch_order = sorted(first, key=lambda n: (n in told, len(self.ch_asked.get(n, [])), first.index(n)))
        self.ch_queues = queues
        self.trace.append({'kind': 'ch_queues', 'order': self.ch_order, 'queues': {n: q[:8] for n, q in queues.items()}})

    def ch_prepare(self) -> None:
        if self.ch_planned is None:
            self.ch_plan()
            self.ch_build()
            return
        for n in range(1, len(self.items) + 1):  # listings opened since the plan add their keys at the queues' tails
            self.ch_menus[n] = self.ch_menu(n)
            q = self.ch_queues.setdefault(n, [])
            for e in self.ch_menus[n]:
                if (e['listings'] >= 2 or e['of'] <= 2) and (not self.ch_settled(n, e['property'])) and (not any((_ch_subject(e['property'], p) for p in q))):
                    q.append(e['property'])
            if n not in self.ch_order:
                self.ch_order.append(n)

    # ---------------- list widths ----------------
    def ch_width(self, ahead: int=0) -> int:
        """Labels the next message names (`ahead`: messages before it). Shoppers differ by world: some answer a list
        about the need it names (race 18: first place told 98-99%, second 61-68%, third 39-48%, later 24-30%), some
        want one subject a message. Only the replies tell: so the first list is short (3 when it is the only message),
        a list answered without complaint lets the next one name one more, up to CH_CALM, and only a need told from
        past a list's first label opens CH_FULL (Chidori: doubled after any list answered, 8/6 after one read). A long
        list the shopper lost halves the widths after it; a reply wanting one subject a message ends lists."""
        if not self.listing:
            return 1
        one = len(self.items) == 1
        k = self.list_asks + ahead
        if k == 0:
            width = (3 if self.ch_free() - ahead <= 1 else CH_FIRST) if one else CH_FIRST_MULTI
        elif self.lists_read:
            width = CH_FULL if one else CH_FULL_MULTI
        elif self.list_taken:
            width = min(CH_CALM if one else CH_CALM_MULTI, self.list_taken + 1)
        else:
            width = CH_FIRST if one else CH_FIRST_MULTI
        return max(1, min(width, self.list_limit) if self.list_limit else width)

    def ch_list_answered(self, width: int, place: int | None) -> None:
        """A list of `width` labels answered without complaint; `place`: where in it the told need stood."""
        self.list_taken = max(self.list_taken, width)
        if place is not None and place >= 1:
            self.lists_read = True

    def ch_place(self, labels: list[str], said: str, line: str) -> int | None:
        """Index of the label a told need is about: the same property or one sharing half its words, else the label whose
        listing values the need's value names (answered_label); None when none."""
        said = said or line.split(':', 1)[0]
        scores = [2.0 if _ch_same(said, p) else _ch_overlap(said, p) for p in labels]
        if scores and max(scores) >= 0.5:
            return scores.index(max(scores))
        hit = self.answered_label(labels, [line]) if line else -1
        return hit if hit >= 0 else None

    # ---------------- messages ----------------
    def ch_message(self, told: list[str], floor: int) -> list[tuple[int, str, str]]:
        """Send the next message (a narrowed single question first, else the next mixed message or list), read its reply
        and settle its labels. Returns the sent batch, or [] when nothing went out."""
        while self.ch_follow:
            number, label = self.ch_follow.pop(0)
            if self.ch_settled(number, label):
                continue
            samples = self.samples(number, label)
            text = f'One thing only: which {label} would you like for the {self.item_said(number)}' + (f', for example {samples[0]} or {samples[1]}?' if len(samples) == 2 else '?')
            if not (self.message_args(text) and _says(self.message_args(text), text)):
                text = self.question(number, label)
            return self.ch_list(number, [label], text, told, floor)
        for _ in range(CH_TRIES):
            for n in self.ch_order:
                self.ch_queues[n] = [p for p in self.ch_queues.get(n, []) if not self.ch_settled(n, p)]
            order = [n for n in self.ch_order if self.ch_queues.get(n)]
            if not order:
                return []
            pool = [n for n in order if n not in self.ch_heard] or order
            turn = self.ch_order
            start = turn.index(self.ch_last) + 1 if self.ch_last in turn else 0
            number = next((turn[(start + s) % len(turn)] for s in range(len(turn)) if turn[(start + s) % len(turn)] in pool))
            width = self.ch_width()
            free = self.ch_free()
            unheard = [n for n in pool if n not in self.ch_heard]
            if self.listing and 1 <= free < len(unheard) and len(unheard) >= 2:
                got = self.ch_mixed(number, unheard, width, free, told, floor)
                if got is not None:
                    return got
            labels = self.ch_queues[number][:width]
            fitted, text = self.fit_ask(number, labels, False)
            if fitted:
                return self.ch_list(number, fitted, text, told, floor)
            self.ch_asked.setdefault(number, []).append(labels[0])
            self.trace.append({'kind': 'ask_too_long', 'item': number, 'label': labels[0]})
        return []

    def ch_list(self, number: int, labels: list[str], text: str, told: list[str], floor: int) -> list[tuple[int, str, str]]:
        """One message about one item naming `labels` (one or several), and what its reply settles."""
        batch = self.send([(number, labels[0], text)], floor)
        if not batch:
            return []
        self.list_asks += 1
        self.ch_named[number] = self.ch_named.get(number, 0) + len(labels)
        self.ch_last = number
        reading = self.ch_read(text, batch[0][2], [(number, labels)])
        kind = reading['kind']
        asked = self.ch_asked.setdefault(number, [])
        queue = self.ch_queues.setdefault(number, [])
        if kind == 'split' and len(labels) >= 2:
            # one subject a message wanted: lists end, and nothing of this one was heard, so its labels stay in order
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': number, 'labels': labels})
            self.ch_take(reading, [(number, labels)], told, batch[0][2], False)
            return batch
        if kind == 'unclear' and len(labels) >= CH_LOST_LONG:
            self.list_limit = max(1, len(labels) // 2)
            self.trace.append({'kind': 'list_shortened', 'item': number, 'cap': self.list_limit})
        if kind in ('need', 'repeat') and reading['item'] == number:
            place = self.ch_place(labels, reading['property'], reading['line'])
            self.ch_list_answered(len(labels), place)
            used = labels[:place + 1] if place is not None else labels[:1]
            asked.extend(used)
            self.ch_queues[number] = [p for p in queue if p not in used]
            self.ch_heard.add(number)
        elif kind == 'split':
            # a single question turned away as too much at once: lists end; asked again narrowed (one product of a name
            # joining several, else with example values) when the task allows messages to spare, else set aside
            if self.listing:
                self.listing = False
                self.trace.append({'kind': 'lists_off', 'item': number, 'labels': labels})
            key = (number, _ch_plain(labels[0]))
            if key not in self.ch_narrowed and self.msg_limit >= CH_NARROW_MIN:
                self.ch_narrowed.add(key)
                if number not in self.one_product and self.name_parts(number):
                    self.one_product.add(number)
                self.ch_follow.insert(0, (number, labels[0]))
            else:
                asked.append(labels[0])
                self.ch_queues[number] = [p for p in queue if p != labels[0]]
        else:
            if kind == 'free':
                self.ch_list_answered(len(labels), None)
            # no need told: the first label is settled (a need there is answered) and the list's later labels lead the
            # next list in the order they were named (run 20244 t-6b06: a need named second, then first, was told)
            asked.append(labels[0])
            self.ch_queues[number] = [p for p in queue if p != labels[0]]
        self.ch_take(reading, [(number, labels)], told, batch[0][2], True)
        self.trace.append({'kind': 'ch_ask', 'item': number, 'labels': labels, 'read': kind, 'need': reading['line'][:120], 'next': self.ch_queues.get(number, [])[:4]})
        return batch

    def ch_mixed_text(self, parts: list[tuple[int, list[str]]]) -> tuple[list[tuple[int, list[str]]], str] | None:
        """The mixed question the message tool takes whole, cutting every part's labels to the same count and then the
        last part; None when fewer than two parts fit."""
        while len(parts) >= 2:
            for size in range(max((len(labels) for _, labels in parts)), 0, -1):
                cut = [(n, labels[:size]) for n, labels in parts]
                pieces = [f"{', '.join(labels[:-1])} or {labels[-1]}" if len(labels) >= 2 else labels[0] for _, labels in cut]
                text = ', and '.join((f'what {piece} for the {self.item_said(n)}' for piece, (n, _) in zip(pieces, cut)))
                text = 'W' + text[1:].replace(' for the ', ' would you like for the ', 1) + '?'
                args = self.message_args(text)
                if args and _says(args, text):
                    return (cut, text)
            parts = parts[:-1]
        return None

    def ch_mixed(self, number: int, unheard: list[int], width: int, free: int, told: list[str], floor: int) -> list[tuple[int, str, str]] | None:
        """One message about several items when more items are still unheard than messages are free: a share of them
        (the least asked first), the list's width shared out evenly (the first named one more). The reply is about one
        of them at most: its need settles that item's labels up to it; a reply with none settles the first label named.
        None when no mixed question fits the message tool."""
        size = -(-len(unheard) // free)

        def named(j: int) -> int:
            return self.ch_named.get(j, 0) + len(self.ch_asked.get(j, []))
        rest = sorted((j for j in unheard if j != number), key=lambda j: (named(j), unheard.index(j)))
        group = ([number] + rest)[:size]
        place = {j: g for g, j in enumerate(group)}
        group.sort(key=lambda j: (named(j), place[j]))
        shares = [width // len(group) + (1 if g < width % len(group) else 0) for g in range(len(group))]
        parts = [(j, self.ch_queues[j][:max(1, shares[g])]) for g, j in enumerate(group)]
        fitted = self.ch_mixed_text([(j, labels) for j, labels in parts if labels])
        if not fitted:
            return None
        parts, text = fitted
        batch = self.send([(parts[0][0], parts[0][1][0], text)], floor)
        if not batch:
            return []
        self.list_asks += 1
        for j, labels in parts:
            self.ch_named[j] = self.ch_named.get(j, 0) + len(labels)
        self.ch_last = parts[-1][0]
        reading = self.ch_read(text, batch[0][2], parts)
        kind, hit = (reading['kind'], reading['item'])
        if kind == 'split' and not hit:
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': parts[0][0], 'labels': [p for _, labels in parts for p in labels]})
            self.ch_take(reading, parts, told, batch[0][2], False)
            return batch
        labels = next((labels for j, labels in parts if j == hit), None)
        if kind in ('need', 'repeat') and labels:
            k = self.ch_place(labels, reading['property'], reading['line'])
            flat = [(j, p) for j, part in parts for p in part]
            self.ch_list_answered(len(flat), flat.index((hit, labels[k])) if k is not None else None)
            used = labels[:k + 1] if k is not None else labels[:1]
            self.ch_asked.setdefault(hit, []).extend(used)
            self.ch_queues[hit] = [p for p in self.ch_queues.get(hit, []) if p not in used]
            self.ch_heard.add(hit)
        else:
            lead, part = parts[0]
            self.ch_asked.setdefault(lead, []).append(part[0])
            self.ch_queues[lead] = [p for p in self.ch_queues.get(lead, []) if p != part[0]]
        self.ch_take(reading, parts, told, batch[0][2], True)
        self.trace.append({'kind': 'mixed_ask', 'parts': parts, 'read': kind, 'item': hit, 'need': reading['line'][:120]})
        return batch

    # ---------------- reading a reply ----------------
    def ch_line(self, prop: str, op: str, value: str, line: str) -> str:
        """The told need as Cairn's requirement line "property: constraint": the reader's own line when it has both
        parts, else built from the property, op and value."""
        head, _, tail = line.partition(':')
        if head.strip() and tail.strip():
            return f'{head.strip()}: {tail.strip()}'
        if not prop or not value:
            return ''
        if op == 'one of':
            value = ' or '.join((v.strip() for v in value.split('|') if v.strip()))
        lead = {'is not': 'not ', 'at least': 'at least ', 'at most': 'at most '}.get(op, '')
        return f'{prop}: {lead}{value}'

    def ch_named_prop(self, number: int, reply: str, asked: list[str], value: str) -> str | None:
        """The one listing key a reply names word for word when it names none of the asked labels (a need filed under
        another property the listings carry), else None; a longer key holding a shorter one is one mention."""
        text = ' ' + _ch_plain(reply) + ' '
        if any((a and _ch_plain(a) and ' ' + _ch_plain(a) + ' ' in text for a in asked)):
            return None
        keys: set[str] = set()
        for record in self.ch_records(number):
            for key in (_fact_block(record) or {}):
                name = _ch_plain(key)
                if len(name) >= 4 and (not _ch_set(name) & _ch_set(value)) and (not any((_ch_same(name, a) for a in asked if a))):
                    keys.add(name)
        hits = [k for k in keys if ' ' + k + ' ' in text]
        hits = [k for k in hits if not any((k != o and ' ' + k + ' ' in ' ' + o + ' ' for o in hits))]
        return hits[0] if len(hits) == 1 else None

    def ch_read(self, text: str, reply: str, parts: list[tuple[int, list[str]]]) -> dict[str, Any]:
        """One reading of a reply to a message naming `parts` (item, labels in the order named)."""
        out: dict[str, Any] = {'kind': 'other', 'item': 0, 'property': '', 'line': '', 'points': False, 'drop': [], 'budget': None, 'steps': [], 'quiet': False}
        named = sum((len(labels) for _, labels in parts))
        if self.quiet_reply([(parts[0][0] if len(parts) == 1 else 0, text, reply)]):
            self.trace.append({'kind': 'quiet_reply', 'a': reply[:80]})
            out.update(kind='free', quiet=True)
            return out
        self.hear_reads += 1
        listing = '\n'.join((f"{k}. {self.items[n - 1].name}: " + ' | '.join(labels) for k, (n, labels) in enumerate(parts, 1)))
        known = '\n'.join((f'{k}. ' + ('; '.join((m for m in self.items[n - 1].musts + self.items[n - 1].told if m not in self.items[n - 1].dropped)) or 'none') for k, (n, _) in enumerate(parts, 1)))
        prompt = CH_REPLY + '\n\nPRODUCTS AND THE PROPERTIES THE MESSAGE NAMED (in its order):\n' + listing + '\n\nKNOWN REQUIREMENTS:\n' + known + '\n\nMESSAGE: ' + text + '\nREPLY: ' + reply
        got = self.ask_json(prompt, [LIGHT, HEAVY], CH_READ_TOKENS) or {}
        kind = str(got.get('kind') or '').strip().lower()
        kind = kind if kind in CH_KINDS else 'other'
        k = _count(got.get('item'))
        number = parts[k - 1][0] if k is not None and 1 <= k <= len(parts) else 0
        prop = str(got.get('property') or '').strip()
        op = str(got.get('op') or '').strip().lower()
        value = str(got.get('value') or '').strip()
        points = got.get('points') is True or op == 'points'
        line = self.ch_line(prop, op, value, str(got.get('need') or '').strip())
        if kind == 'repeat' and line and not any((_ch_same(line.split(':', 1)[0], m.split(':', 1)[0]) for n, _ in parts for m in self.items[n - 1].musts + self.items[n - 1].told)):
            kind = 'need'  # a "repeat" of nothing known is a need
        if kind in ('need', 'repeat') and not number:
            number = next((n for n, labels in parts if self.ch_place(labels, prop, line) is not None), 0)
            if not number:
                number = next((n for n, _ in parts if _ch_plain(self.items[n - 1].name) and ' ' + _ch_plain(self.items[n - 1].name) + ' ' in ' ' + _ch_plain(reply) + ' '), parts[0][0])
        if kind == 'need' and not line:
            kind = 'other'
        if kind == 'need' and not points:
            head, _, tail = line.partition(':')
            if op == 'is not' or re.search(CH_NEGATED, tail, re.I):
                side = self.ask_json(CH_SIDE + f'\n\nSHOPPER\'S REPLY: {reply}\nPROPERTY: {prop or head.strip()}\nVALUE: {value or re.sub(CH_NEGATED, "", tail, flags=re.I).strip()}', [HEAVY, LIGHT], 400) or {}
                if str(side.get('stance') or '').strip().lower() == 'insist':
                    wanted = value if value and not re.search(CH_NEGATED, value, re.I) else re.sub(CH_NEGATED, '', tail, flags=re.I).strip()
                    if wanted:
                        line = f'{head.strip()}: {wanted}'
                        self.trace.append({'kind': 'exclusion_turned', 'need': line[:120]})
        if kind == 'need':
            asked = [p for n, labels in parts if n == number for p in labels]
            head, _, tail = line.partition(':')
            other = self.ch_named_prop(number, reply, asked + [prop, head], value or tail)
            if other:
                line = f'{other}: {tail.strip()}'
                prop = other
                self.trace.append({'kind': 'need_refiled', 'need': line[:120]})
        if kind in ('split', 'unclear') and named >= 2:
            check = self.ask_json(CH_SPLIT + f'\n\nMESSAGE: {text}\nREPLY: {reply}', [LIGHT, HEAVY], 300) or {}
            kind = 'split' if str(check.get('verdict') or '').strip().lower() == 'bundled' else 'unclear'
        budget = got.get('budget')
        drops = []
        for entry in _items(got.get('drop')):
            said = entry.get('property') if isinstance(entry, dict) else entry
            if isinstance(said, str) and said.strip():
                drops.append(said.strip())
        out.update(kind=kind, item=number if kind in ('need', 'repeat') else 0, property=prop, line=line if kind == 'need' else '', points=points and kind == 'need', drop=drops,
                   budget=budget if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 else None, steps=[s for s in _items(got.get('steps')) if isinstance(s, dict)])
        return out

    def ch_take(self, reading: dict[str, Any], parts: list[tuple[int, list[str]]], told: list[str], reply: str, needs: bool) -> None:
        """A reading into Cairn's books through take_heard (told needs to item.told and the ledger, pointers to refs,
        drops, a not-minded single label, the budget); new process steps to self.process with the messages they keep."""
        labels = [p for _, part in parts for p in part]
        heard: dict[str, Any] = {}
        if needs and reading['kind'] == 'need' and reading['item'] and reading['line']:
            heard['needs'] = [{'item': reading['item'], 'need': reading['line'], 'points': bool(reading['points'])}]
        if reading['drop']:
            heard['drop'] = [{'item': reading['item'] or parts[0][0], 'property': p} for p in reading['drop']]
        if needs and reading['kind'] == 'free' and not reading.get('quiet') and len(labels) == 1:
            heard['free'] = [{'item': parts[0][0], 'property': labels[0]}]
        if reading['budget'] is not None:
            heard['budget'] = reading['budget']
        self.asked_now = set(labels)
        try:
            self.take_heard(heard, told)
        finally:
            self.asked_now = set()
        if needs and reading['kind'] in ('need', 'repeat') and reading['item']:
            self.ch_said.setdefault(reading['item'], []).append(' ' + _ch_plain(reply) + ' ')
        added = []
        for step in reading['steps']:
            goal = str(step.get('step') or step.get('goal') or '').strip()
            if not goal or self.ch_steps_added >= CH_STEPS_MAX:
                continue
            said = str(step.get('when') or step.get('moment') or '').lower()
            when = 'end' if ('end' in said or 'decline' in said) else 'choose' if 'choose' in said else 'order'
            if any((p['when'] == when and _ch_overlap(goal, p['step']) >= 0.6 for p in self.process)):
                continue
            self.process.append({'step': goal, 'when': when, 'messages': max(0, _count(step.get('messages')) or 0)})
            self.ch_steps_added += 1
            added.append(goal)
        if added:
            if self.roles.get('ask'):
                talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
                self.reserve = min(max(talk), self.msg_limit) if talk else 0
            told.append('the shopper now also asks: ' + '; '.join(added))
            self.trace.append({'kind': 'reply_steps', 'steps': added, 'reserve': self.reserve})

    # ---------------- the two entry points ----------------
    def ch_interview(self) -> str:
        """interview() with this engine: every free message is spent (no early stop), then the king's follow-ups."""
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.widen()
        self.ch_prepare()
        had = [len(item.told) for item in self.items]
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.clock() < QUESTION_BY):
            batch = self.ch_message(told, 6)
            if not batch:
                break
            sent += batch
        found_items = [n for n, (item, size) in enumerate(zip(self.items, had), 1) if len(item.told) > size]
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

    def ch_late_question(self) -> str:
        """late_question() with this engine: one more message before a checkout, with the messages the ending keeps
        counted as free (several items still unheard share it as a mixed message)."""
        if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit:
            return ''
        if any((p['when'] == 'order' and p['messages'] > 0 for p in self.process)):
            return ''
        if self.left() < LATE_STEPS or self.clock() > LATE_BY or not self.judges():
            return ''
        self.late_asked = True
        had = [len(item.told) for item in self.items]
        told: list[str] = []
        budget = self.budget
        self.ch_late = True
        try:
            self.ch_prepare()
            batch = self.ch_message(told, LATE_STEPS - 2)
        finally:
            self.ch_late = False
        if not batch:
            self.trace.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
            return ''
        if self.budget != budget and not any((t.startswith('budget now') for t in told)):
            told.append(f'budget now {self.budget}')
        _, question, answer = batch[0]
        self.trace.append({'kind': 'late_ask', 'q': question, 'a': answer[:200], 'told': told})
        if not told:
            return ''
        self.version += 1
        summary = f'Cairn asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
        found = [n for n, (item, size) in enumerate(zip(self.items, had), 1) if len(item.told) > size]
        searched = self.value_search(found) if found else ''
        return summary + ('\n' + searched if searched else '')

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Cairn(ChidoriAsk):

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
        self.list_taken = 0
        self.lists_read = False
        self.list_limit = 0
        self.heard_lost = False
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
        self.item_caps: dict[int, float] = {}
        self.cap_rules: dict[int, float] = {}  # USE_R16_CAP_FULLEST_NAME: the caps read_item_caps placed by the fullest name
        self.refused: dict[str, str] = {}  # USE_R16_REFUSED_LINES: line -> the cart check's problem that refused it
        self.tie_listed: set[str] = set()  # USE_R16_SOLD_TIES: sold-out lines whose label was listed at their price
        self.effort_refused = False
        self.cut_noted: set[tuple[str, str]] = set()
        self.fit_codes_dead = False
        self.fit_scan_pages = 0
        self.fit_rows: dict[int, list[str]] = {}
        self.swept: dict[int, set[str]] = {}
        self.chain_page_ids: list[str] = []
        self.trail_opened = -1
        self.link_queries: set[str] = set()
        self.late_drop = False
        self.holds_fired: set[str] = set()
        self.cheaper_round = 0
        self.decline_checks: set[str] = set()
        self.code_memo: dict[str, tuple[Any, list[str]]] = {}
        self.own_rows: dict[int, list[str]] = {}  # #97 item 19: rows of each item's own value searches, in order
        self.own_searches: list[tuple[int, str]] = []  # #97 items 19/20: (item, call mark) of each opening value search with rows
        self.front_paged = False  # #97 item 20: up-front paging ran
        self.dear_checked: set[int] = set()  # #97 item 21: items whose carted line had the dear-line check
        self.series_swept: set[tuple[int, tuple[str, ...]]] = set()  # #97 item 22: (device item, wanted series) swept
        self.chain_findings: list[str] = []  # #97 item 8: findings on chain pages (rejected or deferring links; filled by items 40/41)
        self.trail_early = False  # #97 item 34: the running chain follow is the early one
        self.link_memo: dict[str, dict[str, Any]] = {}  # #97 item 35: link search (lower case) -> its rows and next page
        self.link_found: str = ''  # #97 items 35/37: the opened page find_link last recognised as the link
        self.deferred: set[tuple[str, str]] = set()  # #97 item 41: (requirement, page) already deferred once
        self.hold_causes: set[tuple[str, str]] = set()  # #97 item 45: (cart line, requirement) a checkout hold already named
        self.hold_new_causes: list[tuple[str, str]] = []  # #97 item 45: causes of the hold being built
        self.cheaper_named: list[tuple[str, str]] = []  # #97 items 49/52: (cart line, cheaper listing) pairs the cheaper hold named
        self.better_by_cart: dict[str, list[str]] = {}  # #97 item 52: CHECK's better listings per exact cart it rejected
        self.left_at_call = 0  # #97 item 61: steps left when the call being run arrived
        self.search_reopened = False  # #97 item 59: search reopened after an idle turn
        self.ever_held: set[str] = set()  # #97 item 73: every listing a cart call ever held
        self.last_args: dict[str, dict[str, Any]] = {}  # #97 item 66: the last arguments sent to each tool
        self._sets_memo: tuple[Any, list[tuple[float, dict[str, int]]]] | None = None  # #97 item 54: whole sets by state
        self.repaired: list[tuple[str, str]] = []  # #97 item 74: (misfit part line, its replacement) of the last repair plan
        self.nearest_read: bool | None = None  # #110 item 1 / #111 E52: what the light reading found (None: not read yet)
        self.nearest_sentence = ''  # #110 item 1 / #111 E52: the request's sentence the reading quoted
        self.near_misses: list[str] = []  # #110 item 1: what the heavy nearest pick says its lines miss
        self.near_made = False  # #110 item 1: Cairn carted a nearest pick
        self.near_rescues = 0  # #110 item 1: nearest carts made
        self.set_memo: tuple[Any, Any] | None = None  # #110 item 6: the set check by state
        self.set_held: set[str] = set()  # #110 item 6: cart states a set hold named
        self.proven_mark: Any = None  # #110 item 7: the state the last proven-order look saw
        self.proven_tries = 0  # #110 item 7
        self.proven_swaps = 0  # #110 item 8
        self.settle_held = False  # #110 item 11
        self.picked_lines = False  # #104 item 2: the strong-model pick ran
        self.asked_now: set[str] = set()  # #101 item 9: the labels the question being read asked about
        self.ch_reset()  # USE_CHIDORI_ASK: the engine's state

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

    def clock_bound(self) -> bool:
        """#97 items 21-23, 29-31: whether the clock, not the step limit, ends this episode: the steps left beyond the tail at
        STEP_SECONDS each would outlast the model time left (race 16: true on its 78-109-step tasks, false on its 35-step ones)."""
        return (self.left() - self.tail()) * STEP_SECONDS > self.model_room()

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
        low = USE_E15_LATE_LOW_EFFORT and bool(tools) and model.startswith('anthropic/') and (not self.effort_refused) and self.clock() > EFFORT_AFTER
        if low:
            body['reasoning'] = {'effort': 'low'}
        result = _llm_within(self.model_room()).post_verbose('/inference/chat/completions', body)
        if low:
            error = result.error or {}
            refused = error.get('status') == 400 or (isinstance(result.data, dict) and isinstance(result.data.get('error'), dict) and _count(result.data['error'].get('code')) == 400)
            if refused:
                self.effort_refused = True
                self.trace.append({'kind': 'effort_refused', 'model': model})
                return self.call_model(model, messages, max_tokens, tools)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.cost += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            if USE_G1_CUT_TURN_TO_SPARE and tools and not message.get('tool_calls') and self.cut_at_cap(data['choices'][0]):
                self.trace.append({'kind': 'llm_cut', 'model': model, 'chars': len(str(message.get('content') or ''))})
                return (None, 0)
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

    @staticmethod
    def cut_at_cap(choice: Any) -> bool:
        """#97 item 1: whether the provider says this reply stopped at the token cap (hidden reasoning counts against it)."""
        if not isinstance(choice, dict):
            return False
        return any((str(choice.get(k) or '').strip().lower() in ('length', 'max_tokens', 'max_output_tokens') for k in ('finish_reason', 'native_finish_reason')))

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
        self.last_args[name] = dict(args) if isinstance(args, dict) else {}
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

    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        brief_text = BRIEF_G4 if USE_G4_BRIEF_RELATION_WORDING else BRIEF
        if USE_H103_1_BRIEF_VALUE_CLAUSES:  # #103 item 1: value clauses
            brief_text = brief_text.replace('do not swap it for a neighbouring property.', 'do not swap it for a neighbouring property.' + H103_CLAUSES, 1)
        prompt = brief_text + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.request
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
            if USE_E5_QTY_BEFORE_NAME and count == 1 and entry.get('name'):
                written = _count_before_name(self.request, str(entry['name']), [str(e.get('name') or '') for e in _items(brief.get('items')) if isinstance(e, dict) and e is not entry])
                if written:
                    self.trace.append({'kind': 'qty_from_name', 'item': str(entry['name']), 'qty': written})
                    count = written
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
            if USE_E4_CONDITIONS_NOT_LOOKUPS and text and _conditional(text):
                self.trace.append({'kind': 'condition_not_lookup', 'need': text[:200]})
                continue
            if USE_R104_5_RELATION_NOT_POINTER and text and self.relates_items(number, text):
                self.trace.append({'kind': 'relation_not_lookup', 'need': text[:200]})
                continue
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        if USE_E3_KEEP_RELATION_WORDING:
            self.keep_relation_wording()
        self.unmix()
        if USE_G9_NEED_ON_NAMED_ITEMS:
            self.narrow_shared_needs()
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            self.budget = budget
        if USE_E6_QTY_TIMES_FORM:
            self.times_quantities()
        if USE_E7_ITEM_PRICE_CAPS:
            self.read_item_caps()
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

    def relates_items(self, number: int, need: str) -> bool:
        """#104 item 5: whether a lookup only relates its item to another requested item: its value names every name word
        of another item and says nothing of a page, listing, shop or seller (E4's pointer talk)."""
        value = need.partition(':')[2] if ':' in need else need
        if len(self.items) < 2 or re.search(POINTER_TALK, value, re.I):
            return False
        have = _roots(value)
        return any((n != number and _roots(item.name) and _roots(item.name) <= have for n, item in enumerate(self.items, 1)))

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

    def names_item(self, sentence: str, number: int) -> bool:
        """#97 item 9: whether a request sentence names the item by its whole name, not only inside a longer requested item's
        name that holds it (a name written more often than such longer names still counts)."""
        item = self.items[number - 1]
        pattern = _name_pattern(item.name)
        if not pattern:
            return False
        spans = [hit.span() for hit in re.finditer('(?<![A-Za-z0-9])' + pattern, sentence, re.I)]
        mine = _roots(item.name)
        longer: list[tuple[int, int]] = []
        for k, other in enumerate(self.items, 1):
            theirs = _roots(other.name)
            if k != number and mine and mine < theirs and _name_pattern(other.name):
                longer += [hit.span() for hit in re.finditer('(?<![A-Za-z0-9])' + _name_pattern(other.name), sentence, re.I)]
        return any((not any((a <= s and e <= b for a, b in longer)) for s, e in spans))

    def root_named(self, words: list[str], holders: list[int]) -> set[int]:
        """USE_R16_SCOPE_ROOTS: which of `holders` a run of folded words (`_word_roots`) names: every root of the item's name,
        or of its store label, stands in it; not when each of those roots stands there once and inside the name or label of
        a wider requested item that stands there whole too (the narrower name is then only part of the wider one)."""
        held = set(words)
        names = {n: [r for r in (_roots(it.name), _roots(it.shelf or '')) if r] for n, it in enumerate(self.items, 1)}
        out: set[int] = set()
        for number in holders:
            for mine in names.get(number, []):
                if not mine <= held:
                    continue
                wider = any((mine < theirs <= held for k, sets in names.items() if k != number for theirs in sets))
                if wider and all((words.count(r) == 1 for r in mine)):
                    continue
                out.add(number)
                break
        return out

    @_soft(None)
    def narrow_by_roots(self) -> None:
        """USE_R16_SCOPE_ROOTS: #97 item 9 read by word roots. A requirement (not a pointer) the brief gave two or more items
        stays only on the items its stating sentences name, and leaves the other holders sharing a name word with a kept one.
        A stating sentence holds every figure of the value and names its property, or (a value of words only) holds every
        word of the value. It names the holders `root_named` finds in it; when it names several, the clauses holding the value
        decide when one of them names a holder. A stating sentence naming no holder but sharing a name word with two or more
        (an opening "a cream camera" before "1 cctv camera in cream") is set aside; any other one naming none keeps all."""
        pointers = {need for _, need in self.refs} | {str(x) for x in _items(self.brief.get('lookups'))}
        sentences = _sentences(self.request)
        groups: dict[str, list[int]] = {}
        for number, item in enumerate(self.items, 1):
            for need in item.musts:
                prop, colon, value = need.partition(':')
                if colon and prop.strip() and value.strip() and need not in pointers and number not in groups.get(_words(need), []):
                    groups.setdefault(_words(need), []).append(number)
        for key, holders in groups.items():
            if len(holders) < 2:
                continue
            need = next((m for m in self.items[holders[0] - 1].musts if _words(m) == key))
            prop, _, value = need.partition(':')
            figures = [f for f, _, _ in _figures(value)]
            prop_roots = _roots(prop)
            wanted = {_fold(w) for w in re.findall('[A-Za-z]+', value)} - set(FILLER_WORDS) - {''} if not figures else set()
            if not prop_roots or not (figures or wanted):
                continue

            def holds(text: str) -> bool:
                said = {_fold(w) for w in re.findall('[A-Za-z0-9]+', text)}
                if wanted and not wanted <= said:
                    return False
                return all((any((abs(f - g) < 1e-09 for g, _, _ in _figures(text))) for f in figures))
            stating: list[tuple[set[int], int]] = []
            for sentence in sentences:
                words = _word_roots(sentence)
                if not holds(sentence) or (figures and not prop_roots & set(words)):
                    continue
                named = self.root_named(words, holders)
                if len(named) > 1:
                    inner: set[int] = set()
                    for clause in _clauses(sentence):
                        if holds(clause):
                            inner |= self.root_named(_word_roots(clause), holders)
                    named = inner or named
                shared = sum((1 for n in holders if _roots(self.items[n - 1].name) & set(words)))
                stating.append((named, shared))
            if not any((named for named, _ in stating)):
                continue
            kept: set[int] = set()
            for named, shared in stating:
                if named:
                    kept |= named
                elif shared < 2:
                    kept = set()
                    break
            if not kept or kept >= set(holders):
                continue
            near = set().union(*(_roots(self.items[n - 1].name) for n in kept))
            gone = [n for n in holders if n not in kept and _roots(self.items[n - 1].name) & near]
            for number in gone:
                item = self.items[number - 1]
                item.musts = [m for m in item.musts if _words(m) != key]
            if gone:
                self.trace.append({'kind': 'need_narrowed', 'need': need[:120], 'kept': sorted(kept), 'removed': gone, 'roots': True})

    @_soft(None)
    def narrow_shared_needs(self) -> None:
        """#97 item 9: a requirement (not a pointer) the brief gave two or more items stays only on the items the request's
        stating sentences name, when each such sentence names one of them: it leaves the other holders whose names share a word
        with a kept one (a "phone" keeps no requirement written for the "phone case"). A stating sentence names the property and
        holds every figure of the value (a value without figures: every word of it). Told, dropped and pointer records stay."""
        if len(self.items) < 2:
            return
        if USE_R16_SCOPE_ROOTS:
            self.narrow_by_roots()
            return
        pointers = {need for _, need in self.refs} | {str(x) for x in _items(self.brief.get('lookups'))}
        sentences = _sentences(self.request)
        groups: dict[str, list[int]] = {}
        for number, item in enumerate(self.items, 1):
            for need in item.musts:
                prop, colon, value = need.partition(':')
                if colon and prop.strip() and value.strip() and need not in pointers and number not in groups.get(_words(need), []):
                    groups.setdefault(_words(need), []).append(number)
        for key, holders in groups.items():
            if len(holders) < 2:
                continue
            need = next((m for m in self.items[holders[0] - 1].musts if _words(m) == key))
            prop, _, value = need.partition(':')
            figures = [f for f, _, _ in _figures(value)]
            prop_roots = _roots(prop)
            wanted = {_fold(w) for w in re.findall('[A-Za-z]+', value)} - set(FILLER_WORDS) - {''} if not figures else set()
            if not prop_roots or not (figures or wanted):
                continue
            stating = []
            for sentence in sentences:
                said = {_fold(w) for w in re.findall('[A-Za-z0-9]+', sentence)}
                if not prop_roots & _roots(sentence) or not wanted <= said:
                    continue
                if all((any((abs(f - g) < 1e-09 for g, _, _ in _figures(sentence))) for f in figures)):
                    stating.append(sentence)
            if not stating:
                continue
            kept: set[int] = set()
            for sentence in stating:
                named = [n for n in holders if self.names_item(sentence, n)]
                if not named:
                    kept = set()
                    break
                kept.update(named)
            if not kept or kept >= set(holders):
                continue
            near = set().union(*(_roots(self.items[n - 1].name) for n in kept))
            gone = [n for n in holders if n not in kept and _roots(self.items[n - 1].name) & near]
            for number in gone:
                item = self.items[number - 1]
                item.musts = [m for m in item.musts if _words(m) != key]
            if gone:
                self.trace.append({'kind': 'need_narrowed', 'need': need[:120], 'kept': sorted(kept), 'removed': gone})

    @_soft(None)
    def keep_relation_wording(self) -> None:
        """#111 E3: a lookup the brief wrote as sameness takes the relation its own request sentence states, when that sentence
        claims no sameness and the part from its main verb on shares at least three words with the lookup."""
        sentences = _sentences(self.request)
        for at, (number, need) in enumerate(list(self.refs)):
            prop, _, value = need.partition(':')
            if not prop.strip() or not value.strip() or not re.search(SAMENESS, value, re.I):
                continue
            wanted = _roots(value)
            best = max(((len(_roots(s) & wanted), -i, s) for i, s in enumerate(sentences)), default=None)
            if not best or not best[0] or re.search(SAMENESS, best[2], re.I):
                continue
            part = _relation_part(best[2], prop)
            shared = _roots(part) & _roots(need)
            if len(shared) < 3:
                continue
            words = re.findall("[A-Za-z0-9']+", part)
            first = next((i for i, w in enumerate(words) if _root(w) in shared), 0)
            if not any((len(w) >= 2 and not re.fullmatch('(?:the|an?|of|ones?|that|which)', w, re.I) for w in words[:first])):
                continue
            new = prop.strip() + ': ' + part
            if new == need:
                continue
            if 1 <= number <= len(self.items):
                item = self.items[number - 1]
                item.musts = [new if m == need else m for m in item.musts]
            self.refs[at] = (number, new)
            self.brief['lookups'] = [new if str(x) == need else x for x in _items(self.brief.get('lookups'))]
            self.trace.append({'kind': 'relation_kept', 'item': number, 'from': need[:200], 'to': new[:200]})

    @_soft(None)
    def times_quantities(self) -> None:
        """#111 E6: a count written in multiplication form names the item most of whose name roots its next words hold (a
        unique best); an item named by exactly one such count takes it as its quantity."""
        found: dict[int, set[int]] = {}
        for count, words in _times_counts(self.request):
            have = _roots(words)
            scores = sorted(((len(_roots(item.name) & have) / len(_roots(item.name)), n) for n, item in enumerate(self.items, 1) if _roots(item.name)), reverse=True)
            if not scores or scores[0][0] <= 0.5 or (len(scores) > 1 and scores[1][0] == scores[0][0]):
                continue
            found.setdefault(scores[0][1], set()).add(count)
        for number, counts in sorted(found.items()):
            item = self.items[number - 1]
            if len(counts) == 1 and next(iter(counts)) != item.qty:
                self.trace.append({'kind': 'qty_times', 'item': number, 'from': item.qty, 'to': next(iter(counts))})
                item.qty = next(iter(counts))

    def money_marks(self) -> tuple[str, set[str], bool]:
        """#111 E7: how the request writes an amount in the budget's unit: the sign written before the budget figure, the
        unit word after it (when it is the brief's budget unit), and whether any sign before a number may stand for money
        (no budget figure to learn from)."""
        unit = str(self.brief.get('budget_unit') or '')
        unit_roots = _roots(unit)
        sign = ''
        word: set[str] = set()
        located = False
        if isinstance(self.budget, (int, float)):
            for value, start, end in _figures(self.request):
                if abs(value - float(self.budget)) > 1e-09:
                    continue
                located = True
                before = self.request[max(0, start - 2):start].strip()
                if before and re.fullmatch(MONEY_SIGN, before[-1]):
                    sign = sign or before[-1]
                after = re.match('\\s*([A-Za-z]{1,12})\\b', self.request[end:])
                if after and unit_roots and _roots(after.group(1)) & unit_roots:
                    word = unit_roots
        if not word and unit_roots and not sign:
            word = unit_roots
        return (sign, word, not located and not sign and not word)

    def money_in(self, text: str) -> list[float]:
        """#111 E7: the amounts in the budget's unit a text writes as limits (a floor such as "at least" is no limit)."""
        sign, word, loose = self.money_marks()
        out = []
        for value, start, end in _figures(text):
            before = text[:start].rstrip()
            mark = before[-1:]
            after = re.match('\\s*-?\\s*([A-Za-z]{1,12})\\b', text[end:])
            hit = bool(sign) and mark == sign or (loose and bool(mark) and bool(re.fullmatch(MONEY_SIGN, mark))) or (bool(word) and bool(after) and bool(_roots(after.group(1)) & word))
            if not hit:
                continue
            lead = before[:-1].rstrip() if mark and re.fullmatch(MONEY_SIGN, mark) else before
            if re.search(FLOOR_BEFORE, lead, re.I):
                continue
            out.append(value)
        return out

    @_soft(None)
    def read_item_caps(self) -> None:
        """#111 E7: per-unit price caps the request writes for one item: a sentence holding exactly one amount in the budget's
        unit (besides the total budget figure) that names exactly one requested item; the smallest per item."""
        self.item_caps = {}
        self.cap_rules = {}
        if len(self.items) == 1 and self.budget is None:
            return
        total = float(self.budget) if isinstance(self.budget, (int, float)) else None
        for sentence in _sentences(self.request):
            amounts = [a for a in self.money_in(sentence) if total is None or abs(a - total) > 1e-09]
            if len(amounts) != 1 or amounts[0] <= 0:
                continue
            have = _roots(sentence)
            named = [n for n, item in enumerate(self.items, 1) if any((roots and len(roots & have) * 2 > len(roots) for roots in (_roots(item.name), _roots(item.shelf or ''))))]
            if len(named) > 1 and USE_R16_CAP_FULLEST_NAME:
                named = self.fullest_named(named, set(_word_roots(sentence)))
            if len(named) != 1:
                continue
            self.item_caps[named[0]] = min(self.item_caps.get(named[0], amounts[0]), amounts[0])
            if USE_R16_CAP_FULLEST_NAME:
                self.cap_rules[named[0]] = amounts[0]
        if self.item_caps:
            self.trace.append({'kind': 'item_caps', 'caps': {str(k): v for k, v in self.item_caps.items()}})

    def fullest_named(self, named: list[int], words: set[str]) -> list[int]:
        """USE_R16_CAP_FULLEST_NAME: of several items a cap sentence names, the one whose whole name or store label stands in
        it with the most words (a store label "slipcovers and furniture covers" over the item "furniture cover"); all of them
        when no name stands whole or two tie."""
        sizes = {n: max((len(r) for r in (_roots(self.items[n - 1].name), _roots(self.items[n - 1].shelf or '')) if r and r <= words), default=0) for n in named}
        top = max(sizes.values(), default=0)
        best = [n for n in named if sizes[n] == top]
        return best if top and len(best) == 1 else named

    def cap_rule_note(self, rule: str) -> str:
        """USE_R16_CAP_FULLEST_NAME: the ledger's note after a request rule that restates one item's own price cap (the rule
        holds that cap's figure): whose limit it is, so the loop does not put it on a sibling item whose name shares words."""
        if not USE_R16_CAP_FULLEST_NAME or not self.cap_rules:
            return ''
        figures = [f for f, _, _ in _figures(rule)]
        owners = [n for n, cap in self.cap_rules.items() if any((abs(f - cap) < 1e-09 for f in figures))]
        if len(owners) != 1 or len(self.items) < 2:
            return ''
        others = [str(n) for n in range(1, len(self.items) + 1) if n != owners[0]]
        return f" (Cairn reads this as the price limit of item {owners[0]} ({self.items[owners[0] - 1].name}) only; item{'s' if len(others) > 1 else ''} {', '.join(others)} {'have' if len(others) > 1 else 'has'} no such limit)"

    def item_cap(self, number: int, request_only: bool=False) -> float | None:
        """#111 E7: the most one unit of the item may cost: the request's own cap for it, and a per-item ("each") budget."""
        if not USE_E7_ITEM_PRICE_CAPS:
            return None
        caps = [self.item_caps[number]] if number in self.item_caps else []
        if not request_only and self.brief.get('budget_scope') == 'each' and isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool):
            caps.append(float(self.budget))
        return min(caps) if caps else None

    def over_cap(self, number: int, ident: str) -> bool:
        """#111 E7: whether a listing's price breaks the item's cap (a price over CAP_SANITY times the cap reads as a unit mix-up)."""
        cap = self.item_cap(number)
        price = self.price_of(ident)
        return cap is not None and price is not None and cap + 1e-09 < price <= CAP_SANITY * cap

    def described(self, number: int, need: str) -> bool:
        if need in [m for _, m in self.refs] or need in [str(x) for x in _items(self.brief.get('lookups'))]:
            return False
        prop, _, value = need.partition(':')
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
            return False
        if USE_E8_FIGURE_STAYS_FIRM and re.search('\\d', value):
            return False
        if USE_P110_13_SHORT_VALUE_FIRM and len(re.findall('[a-z]{3,}', value.lower())) < 4:
            return False  # #110 item 13: a short value is a value, not the shopper's own description
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
            if USE_E10_POINTER_GIVES_ONLY:
                needs = [m + (f' ({self.pointer_gives(m)})' if m in self.ref_values else '') for m in live if m not in told]
            else:
                needs = [m + (f' (the page it points at gives: {self.ref_values[m]})' if m in self.ref_values else '') for m in live if m not in told]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            if item.shelf:
                head += f' (store label: {item.shelf})'
            head += f', quantity {item.qty}'
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            cap = self.item_cap(number, request_only=True)
            if cap is not None:
                text += f"; price: each unit may cost at most {cap:g} {self.brief.get('budget_unit') or ''} (the request's own limit for this item)".replace('  ', ' ')
            if told:
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
            withdrawn = [m for m in dict.fromkeys(item.dropped) if m not in live] if USE_G5_LEDGER_DROPPED else []
            if withdrawn:
                text += '; withdrawn by the shopper after the request (no longer required, whatever the request text says): ' + '; '.join(withdrawn)
            parts.append(text)
        nested = self.nested_labels_line() if USE_G26_NESTED_LABEL_LEDGER and many else ''
        if nested:
            parts.append(nested)
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
            parts.append(f'rule from the request: {rule}' + self.cap_rule_note(str(rule)))
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

    def pointer_gives(self, need: str) -> str:
        """#111 E10: what a resolved pointer requirement means: the property its page gives, and nothing else of the clause."""
        prop = need.partition(':')[0].strip() if ':' in need else ''
        return f"the page it points at gives {prop or 'this property'}: {self.ref_values.get(need, '')}; the other details in this requirement only pick out that page and are not requirements on this item"

    def short_items(self, floor: int | None=None) -> list[int]:
        """#111 E12/E13: the items with no cart line while nothing has been checked out (and, with `floor`, more than that many
        steps are left): every item while the cart is empty; with lines, the items no held line's label belongs to; [] when
        the cart's lines cannot be told apart by label."""
        if self.buys or (floor is not None and self.left() <= floor):
            return []
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not ids:
            return list(range(1, len(self.items) + 1))
        cat_key = self.category_key()
        if not cat_key:
            return []
        held = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids]
        return [n for n, item in enumerate(self.items, 1) if not any((self.owns(item, label) for label in held))]

    def fresh_line(self, key: str, line: str) -> str:
        """A repeated status line: shown when it changes, and again after STATUS_REFRESH statuses (trim shortens older results)."""
        marks = self.__dict__.setdefault('status_marks', {})
        count = self.__dict__.get('status_count', 0)
        shown, at = marks.get(key, ('', -STATUS_REFRESH))
        if line == shown and count - at < STATUS_REFRESH:
            return ''
        marks[key] = (line, count)
        return line

    def status(self) -> str:
        self.status_count = self.__dict__.get('status_count', 0) + 1
        left = self.left()
        text = f'[Cairn] step {self.step} of {self.max_steps} used, {left} left.'
        short = self.short_items(self.tail() + 3) if USE_E12_SHORT_ITEM_KEEPS_TOOLS and left <= len(self.items) + 5 else []
        if left <= self.tail() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif short:
            named = ', '.join((f'item {n} ({self.items[n - 1].name})' if len(self.items) > 1 else self.items[n - 1].name for n in short))
            text += f' Few steps left and no cart line yet for {named}: open the likeliest unopened listing for it, cart the one that fits, then check out.'
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
        if plan and USE_T_FIT_PLAN_REFRESH:
            plan = self.fresh_line('fit_plan', plan)
        if plan:
            text += '\n' + plan
        if USE_G54_WHOLE_SETS and self.bound:
            note = self.fresh_line('sets', self.sets_note())
            if note:
                text += note
        if USE_P110_6_SET_CHECK and self.brief.get('pick') == 'cheapest' and self.buys == 0:
            checked = self.set_line()  # #110 item 6
            checked = self.fresh_line('set_check', checked) if checked else ''  # run 19870: the same line went out 34 times
            if checked:
                text += '\n' + checked
        conditions = self.condition_notes() if USE_E11_CONDITION_NOTES else []
        if conditions:
            text += self.fresh_line('conditions', '\n[conditions between items, read from the cart] ' + ' | '.join(conditions))
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        text += '\n[ledger]\n' + self.ledger()
        return text

    @_soft(list)
    def condition_notes(self) -> list[str]:
        """#111 E11: for each live requirement with a waiver or restriction clause about another requested item, what that
        item's first cart line says about the condition: the rule binds, does not bind now, or waits for that item's line."""
        notes: list[str] = []
        cat_key = self.category_key()
        cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        for number, need in self.needs_now():
            if len(notes) >= CONDITION_NOTES:
                break
            prop, _, value = need.partition(':')
            found = re.search(COND_CLAUSE, value, re.I)
            if not prop.strip() or not found or re.search(POINTER_TALK, value, re.I):
                continue
            rule = value[:found.start()].strip(' ,;:.')
            condition = value[found.end():].strip(' ,;:.')
            waiver = bool(re.match(COND_WAIVER, found.group(0), re.I))
            said = _roots(condition)
            scores = sorted(((len(said & (_roots(it.name) | _roots(it.shelf or ''))), n) for n, it in enumerate(self.items, 1) if n != number), reverse=True)
            if not scores or not scores[0][0] or (len(scores) > 1 and scores[1][0] == scores[0][0]):
                continue
            other = scores[0][1]
            thing = self.items[other - 1]
            head = f"item {number} '{need.strip()}'"
            if cart and not cat_key:
                continue
            lines = [i for i in cart if self.owns(thing, str((self.rows.get(i) or {}).get(cat_key) or ''))] if cart else []
            if not lines:
                notes.append(f'{head}: decided once item {other} ({thing.name}) is chosen')
                continue
            line = lines[0]
            verb = re.search(COND_VERB, condition, re.I)
            subject, allowed = (condition[:verb.start()], condition[verb.end():]) if verb else ('', condition)
            negated = bool(re.match('\\s*not\\b', allowed, re.I))
            allowed = re.sub('^\\s*not\\b', '', allowed, flags=re.I)
            values = [re.sub('^(?:(?:the|an?|one|of)\\s+)+', '', v.strip(), flags=re.I).strip() for v in re.split('\\s*(?:,|/|\\bor\\b|\\beither\\b|\\bnor\\b)\\s*', allowed, flags=re.I)]
            values = [v for v in values if re.search('[A-Za-z0-9]', v)]
            block = _fact_block(self.full.get(line) or {}) or {}
            if not values or not block:
                notes.append(f"{head}: check {line} (the {thing.name} in the cart) yourself for whether {condition}")
                continue
            subject_roots = _roots(re.sub("'s\\b", '', subject)) - _roots(thing.name) - _roots(thing.shelf or '')
            keys = [k for k in block if subject_roots and _roots(_words(k)) & subject_roots]
            if not keys:
                keys = [k for k, v in block.items() if any((_spelled(val) <= _spelled(str(v)) for val in values))]
            held = [k for k in keys if any((_spelled(val) and _spelled(val) <= _spelled(str(block[k])) for val in values))]
            holds = bool(held) != negated
            shown = ', '.join((f'{k}={str(block[k])[:40]}' for k in (held or keys)[:2])) or 'no attribute for it'
            state = 'met' if holds else 'not met'
            if holds != waiver:
                text = f'{head}: on {line} ({shown}) the condition is {state}, so the rule binds: item {number} must have {prop.strip()}: {rule}'
                if waiver:
                    named = [w for w in re.findall('[A-Za-z0-9]+', re.sub("'s\\b", '', subject)) if _root(w) in subject_roots]
                    whose = f"whose {' '.join(named)} is" if named else 'with'
                    text += f" (a {thing.name} listing {whose} {'not ' if negated else ''}{' or '.join(values)} would lift it)"
            else:
                text = f'{head}: on {line} ({shown}) the condition is {state}, so this rule does not bind now'
            notes.append(text)
        return notes

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
        matched = [ids for label, ids in groups.items() if any((_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)]
        if USE_G25_LOOSE_LABEL_LANES and matched:
            matched = self.close_groups([(label, ids) for label, ids in groups.items() if any((ids is m for m in matched))])
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.items))]

    def label_closeness(self, label: str, number: int) -> float:
        """#97 item 25: how closely a store label names its item: 3 for the item's shelf or an adopted label, 2 for its name,
        else the share of word roots it has in common with the item's name and shelf (the king's owner_of score)."""
        item = self.items[number - 1]
        words = _words(label)
        if self.adopted.get(label) == number or words == _words(item.shelf or ''):
            return 3.0
        if words == _words(item.name):
            return 2.0
        mine = _roots(item.shelf or '') | _roots(item.name)
        return len(_roots(label) & mine) / max(1, len(_roots(label) | mine))

    def close_groups(self, groups: list[tuple[str, list[str]]]) -> list[list[str]]:
        """#97 item 25: the label groups that keep their widen lanes: a label scoring below LOOSE_SHARE of its item's best label
        is left out when that best label scores at least LOOSE_FLOOR; labels no item owns stay."""
        scored: dict[str, tuple[int, float]] = {}
        best: dict[int, float] = {}
        for label, _ in groups:
            owner = self.owner_of(label) if label else 0
            if owner:
                scored[label] = (owner, self.label_closeness(label, owner))
                best[owner] = max(best.get(owner, 0.0), scored[label][1])
        kept: list[list[str]] = []
        left: list[str] = []
        for label, ids in groups:
            owner, score = scored.get(label, (0, 0.0))
            if owner and best[owner] >= LOOSE_FLOOR and score < LOOSE_SHARE * best[owner]:
                left.append(label)
            else:
                kept.append(ids)
        if left:
            self.trace.append({'kind': 'loose_lanes', 'left_out': left[:4]})
        return kept

    def single_target(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        if USE_P110_2_ROW_PRICE and not self.price_key:
            return self.own_price(ident)
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str):
            figure = re.fullmatch('\\s*[^\\d\\s]{0,3}\\s?(\\d[\\d,]*(?:\\.\\d+)?)\\s*[A-Za-z]{0,6}\\.?\\s*', value)
            value = float(figure.group(1).replace(',', '')) if figure else None
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def own_price(self, ident: str) -> float | None:
        """#110 item 2: a listing's price with no price key learned: its row's (else its record's) own field whose name says
        price or cost, a field named only that first."""
        for record in (self.rows.get(ident), self.full.get(ident)):
            keys = [k for k in (record or {}) if re.search(ROW_PRICE_KEY, str(k), re.I)]
            keys.sort(key=lambda k: not re.fullmatch('(?:' + ROW_PRICE_KEY + ')', _snake(k)))
            for key in keys:
                price = _price_value(record[key])
                if price is not None:
                    return price
        return None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        if isinstance(value, str) and re.search(STOCK_OUT, value.strip(), re.I):
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
        if (USE_E20_NEXT_ARG_FROM_SCHEMA or USE_G66_NEXT_ARG_STRICT) and not shape['next_arg'] and shape['next'] and observation.get(shape['next']) is not None:
            guess = self.next_arg_strict(name, observation[shape['next']]) if USE_G66_NEXT_ARG_STRICT else self.next_arg_guess(name, observation[shape['next']])
            if guess:
                shape['next_arg'] = guess
                self.trace.append({'kind': 'next_arg_guess', 'tool': name, 'arg': guess})
        self.shapes[name] = shape
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def next_arg_guess(self, name: str, token: Any) -> str | None:
        """#111 E20: the one optional, non-enumerated parameter of `name` whose type takes the reply's next-page token (an
        integer token may also go to a number parameter); None unless exactly one."""
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []

        def takes(spec: dict[str, Any]) -> bool:
            kinds = _kinds(spec)
            if isinstance(token, bool):
                return 'boolean' in kinds
            if isinstance(token, int):
                return 'integer' in kinds or 'number' in kinds
            if isinstance(token, float):
                return 'number' in kinds
            return isinstance(token, str) and 'string' in kinds
        found = [k for k, v in props.items() if k not in required and isinstance(v, dict) and 'enum' not in v and takes(v)]
        return found[0] if len(found) == 1 else None

    @_soft(None)
    def next_arg_strict(self, name: str, token: Any) -> str | None:
        """#97 item 66 (refines E20): the paging parameter from the schema: optional, no fixed choices, taking the token's
        type, length and range, and not the field that carried the call's words (a parameter the last call to this tool
        sent a non-empty text, or one named as a words field); an integer token prefers integer parameters over number
        ones (a price limit is not taken). None unless exactly one is left."""
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        sent = self.last_args.get(name) or {}

        def fits(spec: dict[str, Any]) -> bool:
            kinds = _kinds(spec)
            if isinstance(token, bool):
                return 'boolean' in kinds
            if isinstance(token, (int, float)):
                if not ('number' in kinds or ('integer' in kinds and isinstance(token, int))):
                    return False
                low, high = (spec.get('minimum'), spec.get('maximum'))
                return not (isinstance(low, (int, float)) and token < low) and not (isinstance(high, (int, float)) and token > high)
            if not isinstance(token, str) or 'string' not in kinds:
                return False
            return not (isinstance(spec.get('maxLength'), int) and len(token) > spec['maxLength']) and not (isinstance(spec.get('minLength'), int) and len(token) < spec['minLength'])
        found = []
        for key, spec in props.items():
            if key in required or not isinstance(spec, dict) or 'enum' in spec or not fits(spec):
                continue
            if isinstance(sent.get(key), str) and sent[key].strip() or re.search(QUERY_WORDS, _snake(key), re.I):
                continue
            found.append(key)
        if isinstance(token, int) and not isinstance(token, bool) and len(found) > 1:
            whole = [k for k in found if 'integer' in _kinds(props[k])]
            found = whole or found
        return found[0] if len(found) == 1 else None

    def page_arg(self, name: str) -> str | None:
        """#111 E17: the search's page argument: the one the base learned, else the integer next-page argument the reply-shape
        reading named."""
        key = self.page_args.get(name)
        if key or not USE_E17_PAGE_ARG_FROM_SHAPE:
            return key
        shape = self.shapes.get(name) or {}
        arg = shape.get('next_arg')
        spec = ((self.schemas.get(name) or {}).get('properties') or {}).get(arg or '')
        return arg if isinstance(spec, dict) and 'integer' in _kinds(spec) else None

    def page_low(self, name: str, key: str) -> int:
        """The first value of a page argument: the schema's minimum; with E17 also its default, or 0 for an argument named
        as a row offset; else 1."""
        spec = ((self.schemas.get(name) or {}).get('properties') or {}).get(key) or {}
        if isinstance(spec.get('minimum'), int) and not isinstance(spec.get('minimum'), bool):
            return spec['minimum']
        if USE_E17_PAGE_ARG_FROM_SHAPE:
            if isinstance(spec.get('default'), int) and not isinstance(spec.get('default'), bool) and spec['default'] in (0, 1):
                return spec['default']
            if re.search(OFFSET_ARG, _snake(key), re.I):
                return 0
        return 1

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
        if not plan and USE_E27_CHAIN_PAGING and pick not in ('cheapest', 'most', 'least'):
            chained = self.chain_paging(name, args, observation)
            return chained or (self.big_next_page(name, args, observation) if USE_G29_BIG_NEXT_PAGE else '')
        if not plan:
            return self.big_next_page(name, args, observation) if USE_G29_BIG_NEXT_PAGE and pick not in ('cheapest', 'most', 'least') else ''
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
        room = ONE_PAGE_ROOM if USE_G28_ONE_PAGE_ROOM and pages == 1 else 12
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < room:
            return f"\n[Cairn] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
        texts, value = self.page_on(name, args, param, value, pages)
        rest = f"\n[Cairn] {count} listings matched and Cairn has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Cairn read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    @_soft('')
    def big_next_page(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """#97 item 29: on clock-bound tasks, for pick rules the king's auto-paging leaves alone: the next page of a model's
        first-page search that matched more than it showed, read in the same turn (one step) while BIG_ROOM steps beyond the
        tail remain and its rows hold an item with no cart line yet."""
        if not self.clock_bound() or self.left() - self.tail() < BIG_ROOM or self.finished:
            return ''
        short = set(self.short_items())
        cat_key = self.category_key()
        if not short or not any((set(self.line_items(i, cat_key)) & short for i, _ in self.records_in(observation))):
            return ''
        plan = self.paging(name, observation)
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        texts, _ = self.page_on(name, args, param, value, 1)
        if not texts:
            return ''
        self.trace.append({'kind': 'big_next_page', 'tool': name, 'count': count})
        return f'\n[Cairn read the next page of this search itself (one step; {count} listings matched, {shown} were shown)]\n' + '\n'.join(texts)

    def page_on(self, name: str, args: dict[str, Any], param: str, value: Any, pages: int) -> tuple[list[str], Any]:
        """The king's auto-paging loop: up to `pages` further pages of a search; what each showed and the next page's value."""
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
        return (texts, value)

    def pending_refs(self) -> list[str]:
        live = {need for _, need in self.needs_now()}
        return [need for _, need in self.refs if need in live and need not in self.ref_values]

    def shop_values(self) -> dict[str, int]:
        """#111 E26/E27/E29: short name-like row field values (who sells or makes a listing could be any of them), with how
        many rows carry each."""
        not_names = set(self.not_maker_keys()) | {self.price_key, self.stock_key}
        counts: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if key in not_names or not isinstance(value, str):
                    continue
                text = value.strip()
                if 2 < len(text) <= 40 and text.lower() not in NO_NAME and text[:1].isupper() and not re.search('\\d', text) and len(text.split()) <= 4:
                    counts[text] = counts.get(text, 0) + 1
        return counts

    def chain_shops(self) -> list[str]:
        """#111 E27: the shops the open pointer requirements name, then those the long texts of opened listings name (row
        field values written there with the same spelling); at most CHAIN_SHOPS."""
        pending = self.pending_refs()
        if not pending:
            return []
        prose = [t for ident in self.opened for t in _leaves(self.full.get(ident) or self.rows.get(ident) or {}) if len(t.split()) >= CHAIN_PROSE_WORDS]
        counts = self.shop_values()
        ranked = sorted(counts, key=lambda v: (-counts[v], v))
        found: list[str] = []
        for source in (pending, prose):
            for value in ranked:
                if len(found) >= CHAIN_SHOPS:
                    return found
                if value not in found and any((re.search('(?<![A-Za-z0-9])' + re.escape(value) + '(?![A-Za-z0-9])', text) for text in source)):
                    found.append(value)
        return found

    @_soft('')
    def chain_paging(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """#111 E27: while a pointer requirement is open, a search naming one of its chain's shops is paged by Cairn, when at
        most CHAIN_PAGES pages are needed and the auto-page budget and steps allow; nothing is said otherwise."""
        if not self.pending_refs():
            return ''
        said = ' '.join((v for v in args.values() if isinstance(v, str))).lower()
        shops = [s for s in self.chain_shops() if re.search('(?<![a-z0-9])' + re.escape(s.lower()) + '(?![a-z0-9])', said)]
        if not shops:
            return ''
        plan = self.paging(name, observation)
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        pages = -(-(count - shown) // shown)
        if pages > CHAIN_PAGES or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
            return ''
        texts, value = self.page_on(name, args, param, value, pages)
        self.trace.append({'kind': 'chain_pages', 'shops': shops, 'read': len(texts)})
        return '\n[Cairn read the next pages of this search itself, one step a page: it names a shop a requirement points at]\n' + '\n'.join(texts) if texts else ''

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

    @_soft(None)
    def named_referrals(self, ids: list[str], keep: int) -> None:
        """#104 item 5: a cart line whose own page says to take a value from another listing named in words ("for the
        battery type, refer to the Velo X listing") gets that listing searched for and its best row opened, once per line,
        when steps and time allow; the record joins what the checks see."""
        plan = self.value_template()
        if not plan:
            return
        tool, base, field = plan
        done = self.__dict__.setdefault('referrals_done', set())
        title = self.title_key() or 'title'
        for ident in ids:
            if ident in done or ident not in self.full:
                continue
            record = self.full.get(ident) or {}
            text = ' '.join((str(v) for k, v in record.items() if isinstance(v, str) and k != title and len(v) >= 20))
            found = [m.group(1).strip(' ,') for m in re.finditer(REFERRAL, text, re.I)]
            found = [f for f in found if 2 <= len(f.split()) <= 7 and not any((w.lower() in ('this', 'that', 'our', 'other') for w in f.split()[:1]))]
            if not found:
                continue
            done.add(ident)
            if self.left() <= keep or self.finished or self.rushed():
                return
            query = ' '.join(found[0].split()[:6])
            faults: list[str] = []
            args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
            if faults:
                continue
            reply = self.act(tool, args)
            if reply is None or self.finished:
                return
            seen = reply.get('observation')
            self.observe(tool, seen)
            words = {w for w in _roots(query) if len(w) >= 3}
            rows = [(i, len(words & _roots(str((r or {}).get(title) or '') + ' ' + ' '.join((str(v) for v in (r or {}).values() if isinstance(v, str)))))) for i, r in self.records_in(seen)]
            rows = [(i, n) for i, n in rows if n >= max(2, len(words) // 2) and i != ident]
            self.trace.append({'kind': 'named_referral', 'line': ident, 'query': query, 'rows': len(rows)})
            if rows and self.left() > keep:
                best = max(rows, key=lambda r: r[1])[0]
                self.fill_facts([best], keep)

    def show_links(self, ids: list[str], keep: int) -> None:
        """Run the part code a cart line sends the reader to through the search, once per code, and open
        the page it finds when steps allow; the result feeds the ledger, not the model. Multi-line carts only."""
        plan = self.value_template()
        if not plan or (not (USE_E47_LINKS_ONE_LINE_CARTS and ids) and len(self.items) < 2 and len(ids) < 2):
            return
        tool, base, field = plan
        for ident in ids:
            marks = self.fit_index()['marks']
            code = (marks.get(ident) or {}).get('more')
            if USE_P110_12_SEEN_OWNER_LINK and code and code not in self.link_tried:
                owners = [i for i, mark in marks.items() if mark.get('own') == code]
                if owners and not any((i in self.full for i in owners)):
                    if self.left() <= keep or self.finished or self.rushed():
                        return
                    self.link_tried.add(code)  # #110 item 12: the referred part was seen as a row only: open it
                    self.fill_facts(owners[:1], keep)
                    self.trace.append({'kind': 'show_link_seen', 'line': ident, 'code': code})
                    continue
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

    def plausible_below(self, line: str, label: str, under: float) -> list[str]:
        """#110 item 3: seen rows of `label` priced below `under` that can be bought and whose own row text holds every
        required value word of the label's item, cheapest first; [] when that item has no such words."""
        cat_key = self.category_key()
        owner = self.owner_of(label) if label else 0
        tokens = {w for value, _ in (self.value_words(owner) if owner else []) for w in _value_tokens(value)}
        if not tokens or not cat_key:
            return []
        low = label.strip().lower()
        out = []
        for ident in self.seen_order:
            price = self.price_of(ident)
            if ident == line or price is None or price >= under or str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower() != low:
                continue
            if self.buyable(ident) and tokens <= _text_words(_dump(self.rows.get(ident) or {})):
                out.append(ident)
        return sorted(out, key=lambda i: (self.price_of(i) or 0.0, i))

    def descent_below(self, line: str, price: float, label: str, plan: tuple[str, dict[str, Any], str, str]) -> None:
        """#110 item 3: list the line's shelf under a price cap in rounds. The first cap sits at the cheapest plausible row
        already seen, else just under the line's price; a listing of at most DESCENT_WHOLE rows is read to its end, a longer
        one DESCENT_PAGES pages (stopping at the first page holding a cheaper plausible row); that row's price less one is the
        next cap, for at most DESCENT_ROUNDS rounds while the steps allow."""
        tool, base, shelf, cap = plan
        under = price
        known = self.plausible_below(line, label, under)
        if known:
            under = (self.price_of(known[0]) or price) + 1
        tried: set[int] = set()
        for _ in range(DESCENT_ROUNDS):
            limit = int(under) - 1 if float(under).is_integer() else int(under)
            if limit < 1 or limit in tried or self.left() <= self.tail() + (WALK_ROOM if tried else 7) or self.finished or self.rushed():
                return
            tried.add(limit)
            faults: list[str] = []
            call = _shape({**base, shelf: label, cap: limit}, self.schemas.get(tool), 'args', faults)
            if faults:
                return
            before = self.plausible_below(line, label, limit + 1)
            best = self.price_of(before[0]) if before else None
            pages = 0
            while True:
                reply = self.act(tool, call)
                if reply is None:
                    return
                seen = reply.get('observation')
                self.observe(tool, seen)
                pages += 1
                rows = self.records_in(seen)
                page = self.paging(tool, seen)
                found = self.plausible_below(line, label, limit + 1)
                cheaper = bool(found) and (best is None or (self.price_of(found[0]) or 0.0) < best)
                whole = bool(page) and page[0] <= DESCENT_WHOLE
                if not rows or (cheaper and not whole) or not page or page[3] is None or (not whole and pages >= DESCENT_PAGES) or self.price_sorted(seen):
                    break
                if self.left() <= self.tail() + 7 or self.finished or self.rushed():
                    break
                faults = []
                call = _shape({**call, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                if faults:
                    break
            found = self.plausible_below(line, label, limit + 1)
            self.trace.append({'kind': 'descent', 'line': line, 'limit': limit, 'pages': pages, 'plausible': found[:3]})
            if not found or (best is not None and (self.price_of(found[0]) or 0.0) >= best):
                return
            under = self.price_of(found[0]) or 0.0

    def list_below(self, cart: list[str]) -> None:
        plan = self.cap_template()
        cat_key = self.category_key()
        if not plan or not cat_key:
            return
        if USE_P110_3_CAP_DESCENT:
            for line in cart:
                price = self.price_of(line)
                label = str((self.rows.get(line) or {}).get(cat_key) or '')
                if price is None or price < 2 or not label or (line, price) in self.capped:
                    continue
                self.capped.add((line, price))
                self.descent_below(line, price, label, plan)
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

    def cheaper_rows(self, extra: bool=False) -> list[str]:
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.cart_mark() + f'#{self.version}'
        cat_key = self.category_key()
        if (mark in self.cheaper_marks and not extra) or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        if not extra:
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
            if USE_E45_LIMIT_BREAKERS_LAST:
                owner = self.line_owner(line)
                breakers = {i for i in pool if self.breaks_limit(i, owner)}
                pool.sort(key=lambda i: i in breakers)
                lowest = [i for i in pool if i in self.value_rows and i not in breakers][:1]
            else:
                breakers = set()
                lowest = [i for i in pool if i in self.value_rows][:1]
            if USE_P110_3_CAP_DESCENT:  # #110 item 3: rows whose own text holds every required value word, cheapest first
                plausible = [i for i in self.plausible_below(line, str((self.rows.get(line) or {}).get(cat_key) or ''), price) if i in pool and i not in breakers]
                lowest = plausible + [i for i in lowest if i not in plausible]
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
            if USE_E45_LIMIT_BREAKERS_LAST:
                owner = self.line_owner(line)
                breakers = {i for i in pool if self.breaks_limit(i, owner)}
                pool.sort(key=lambda i: i in breakers)
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
        late = USE_E37_LATE_DROP_HUNT and self.late_drop and self.brief.get('pick') == 'cheapest'
        if self.brief.get('pick') != 'cheapest' or (self.hunts >= HUNT_ROUNDS and not (late and self.hunts < HUNT_ROUNDS + 1)) or self.hunt_mark == mark:
            return ''
        plan = self.value_template()
        if not plan or not self.price_key or self.clock() > HUNT_BY + (HUNT_GRACE if late else 0.0):
            return ''
        tool, base, field = plan
        room = min(HUNT_STEPS, self.left() - self.tail() - HUNT_KEEP - 2 * len(self.items))
        if room < 3 and late:
            room = min(LATE_HUNT_ROOM, self.left() - self.tail() - 2 - 2 * len(self.items))
            if room >= 1:
                self.trace.append({'kind': 'late_drop_hunt', 'room': room, 'left': self.left()})
        if room < 3 and not (late and room >= 1):
            return ''
        if late:
            self.late_drop = False
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
        """Up to `room` searches for one item: all its required values together, then split by further words."""
        tried = self.hunt_tried.setdefault(number, set())
        noun, _ = self.hunt_words(number)
        values = [value for value, _ in self.value_words(number)]
        sets: list[list[str]] = []
        if USE_G27_HUNT_CHOICE_SETS:
            values, sets = self.hunt_sets(number, values)
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
        single = self.singular_noun(number, noun) if USE_S101_6_SINGULAR_NOUN and stem is None else ''
        bare = {noun, single} - {''}

        def wider(rungs: list[str], at: int) -> None:
            """#101 item 5: after the rung with rows, the wider rungs of the same ladder (each still holding a value word; the
            bare noun stays the fallback), within the room."""
            for query in rungs[at + 1:]:
                if spent >= room or refused or self.finished or query in bare or query.lower() in tried:
                    continue
                self.trace.append({'kind': 'hunt_wider', 'item': number, 'query': query[:80]})
                run(query)
        if stem is None and USE_G27_HUNT_CHOICE_SETS:
            stem = ''
            for group in sets:
                forms = [' '.join(group + [noun]).strip()] + ([' '.join(group + [single]).strip()] if single else []) + [' '.join(group)] + ([' '.join(group[:-1] + [noun]).strip()] if len(group) >= 2 else [])
                rungs = list(dict.fromkeys((f for f in forms if f)))
                for at, query in enumerate(rungs):
                    issued = False
                    if run(query):
                        stem = stem or query
                        if USE_S101_5_WIDEST_RUNGS:
                            wider(rungs, at)
                        break
                    if not issued:
                        return (spent, gained)
            if not stem and single:  # #101 item 6: the store's singular before the bare plural
                issued = False
                if run(single):
                    stem = single
                elif not issued:
                    return (spent, gained)
            if not stem and noun:
                issued = False
                if run(noun):
                    stem = noun
                elif not issued:
                    return (spent, gained)
            self.hunt_stem[mark] = stem
            self.trace.append({'kind': 'hunt_sets', 'item': number, 'sets': [' / '.join(g)[:60] for g in sets], 'stem': stem[:60]})
        if stem is None:
            tries = ([' '.join((values + [noun])).strip()] if values else []) + ([' '.join((values + [single])).strip()] if values and single else []) + ([' '.join(values)] if values else []) + ([' '.join((values[:-1] + [noun])).strip()] if len(values) >= 2 else []) + ([single] if single else []) + ([noun] if noun else [])
            stem = ''
            rungs = list(dict.fromkeys(tries))
            for at, query in enumerate(rungs):
                issued = False
                if run(query):
                    stem = query
                    if USE_S101_5_WIDEST_RUNGS:
                        wider(rungs, at)
                    break
                if not issued:
                    return (spent, gained)
                if query == noun:
                    self.hunt_no_noun.add(number)
                    noun = ''
            self.hunt_stem[mark] = stem
        if USE_E36_FLIPPED_SPACING_HUNT and values and spent < room and not refused and not self.finished:
            flipped = [_flip_spacing(v) for v in values]
            query = ' '.join(flipped + [noun]).strip()
            if flipped != values and query.lower() not in tried:
                self.trace.append({'kind': 'flipped_hunt', 'item': number, 'query': query[:80]})
                run(query)
        while spent < room and idle < HUNT_IDLE and not refused and not self.finished and self.clock() <= HUNT_BY + HUNT_GRACE:
            _, words = self.hunt_words(number)
            words = [w for w in words if f'{w} {stem}'.strip().lower() not in tried and w not in stem.lower().split()]
            if not words:
                break
            run(f'{words[0]} {stem}'.strip())
        return (spent, gained)

    def singular_noun(self, number: int, noun: str) -> str:
        """#101 item 6: the store's own singular of the hunt's category noun: the title word of the item's rows, shorter than
        the noun by at most three letters and its start or a form of its stem, found in at least SINGULAR_ROWS titles (the most
        often); '' when none. No suffix table: the word comes from the store's titles."""
        title = self.title_key()
        cat_key = self.category_key()
        if not noun or not title or not cat_key:
            return ''
        item = self.items[number - 1]
        counts: dict[str, int] = {}
        for row in self.rows.values():
            if not self.owns(item, str(row.get(cat_key) or '')):
                continue
            for word in set(re.findall('[a-z]{3,}', str(row.get(title) or '').lower())):
                if word != noun and len(noun) - 3 <= len(word) < len(noun) and (noun.startswith(word) or _same_stem(word, noun)):
                    counts[word] = counts.get(word, 0) + 1
        best = max(sorted(counts), key=lambda w: counts[w], default='')
        return best if best and counts[best] >= SINGULAR_ROWS else ''

    def hunt_sets(self, number: int, values: list[str]) -> tuple[list[str], list[list[str]]]:
        """#97 item 27: the cheapest hunt's value sets for an item: the base set (its plain values without codes); for the first
        set only, the base with a value of words ending in a number of three or more digits as that number alone; and for the
        first closed list of choices (no pointer, exclusion or limit) up to CHOICE_SETS choices, each with the base (codes and
        choices longer than HUNT_VALUE_CHARS left out). A value of words and a spaced number is no code here: it keeps its place
        and gets the number-only set. Number-unit spellings are E36's flipped search."""
        pointers = {need for _, need in self.refs}
        listed, choices = ('', [])
        for owner, need in self.needs_now():
            prop, colon, value = need.partition(':')
            if owner != number or not colon or need in pointers or _conditional(need) or re.search(NEGATION, value, re.I):
                continue
            if re.search(OPEN_NOT, re.sub('\\b(?:or|either)\\b', ' ', value, flags=re.I), re.I) or (_need_limit(value) or {}).get('kind') not in (None, 'exact'):
                continue
            parts = [c.strip(' .') for c in re.split('\\s*(?:,|/|;|\\bor\\b|\\beither\\b)\\s*', value, flags=re.I)]
            parts = [c for c in parts if re.search('[A-Za-z0-9]', c)]
            if len(parts) >= 2:
                listed, choices = (value.strip(), [c for c in parts if not _codes(c) and len(c) <= HUNT_VALUE_CHARS][:CHOICE_SETS])
                break
        spaced = '[A-Za-z][A-Za-z -]*?\\s(\\d{3,})'  # words, a space, a number of three or more digits ("LGA 1700")
        base = [v for v in values if (not _codes(v) or re.fullmatch(spaced, v.strip())) and v.strip() != listed]
        sets = [base] if base else []
        for at, value in enumerate(base):
            bare = re.fullmatch(spaced, value.strip())
            if bare:
                sets.append(base[:at] + [bare.group(1)] + base[at + 1:])
                break
        sets += [base + [choice] for choice in choices]
        return (base, [s for s in sets if s])

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
            if USE_E7_ITEM_PRICE_CAPS:
                pool = [i for i in pool if not self.over_cap(number, i)]
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

    @_soft('')
    def top_rows(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """#111 E18: a brand-new search sent past its first rows (before its page argument was known) never shows them: Cairn
        runs the same search from the first row itself (one step, no model call) and shows those rows too."""
        if not isinstance(observation, dict) or not self.records_in(observation) or self.left() <= self.tail() + 4 or self.finished:
            return ''
        props = (self.schemas.get(name) or {}).get('properties') or {}
        numeric = [k for k, v in args.items() if isinstance(v, int) and not isinstance(v, bool) and k in props and v > self.page_low(name, k) and not re.search(CAP_PRICED, _snake(k), re.I)]
        if not numeric:
            return ''
        if name not in self.shapes:
            self.read_shape(name, observation)
        key = self.page_arg(name)
        if not key or key not in numeric:
            return ''
        low = self.page_low(name, key)
        faults: list[str] = []
        call = _shape({**args, key: low}, self.schemas.get(name), 'args', faults)
        if faults or self.call_mark(name, call) in self.read_calls:
            return ''
        reply = self.act(name, call)
        if reply is None:
            return ''
        seen = reply.get('observation')
        self.observe(name, seen)
        self.paged_obs.append(seen)
        shown_text = self.show(seen)
        self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
        self.trace.append({'kind': 'top_rows', 'tool': name, 'sent': args[key], 'rows': len(self.records_in(seen))})
        return f'\n[Cairn] This new search started at {key}={args[key]}, past its first rows, so Cairn also read its first rows ({key}={low}, one step):\n' + shown_text

    def seen_offsets(self, name: str, args: dict[str, Any], key: str) -> tuple[int | None, bool, bool]:
        """#111 E19: of the replies to this same search (its arguments apart from `key`): the highest next offset the store gave,
        whether a page ended the list, and whether the argument counts rows (next minus sent equals the rows shown)."""
        shape = self.shapes.get(name) or {}
        if not shape.get('next') or not shape.get('rows'):
            return (None, False, False)
        low = self.page_low(name, key)
        want = self.call_mark(name, {k: v for k, v in args.items() if k != key})
        best: int | None = None
        ended = counts_rows = False
        for mark, (tool, sent) in list(self.read_args.items()):
            if tool != name or self.call_mark(name, {k: v for k, v in sent.items() if k != key}) != want:
                continue
            seen = self.read_calls.get(mark)
            if not isinstance(seen, dict) or shape['next'] not in seen:
                continue
            value = seen.get(shape['next'])
            if value is None:
                ended = True
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            start = sent.get(key) if isinstance(sent.get(key), int) and not isinstance(sent.get(key), bool) else low
            rows = seen.get(shape['rows'])
            if isinstance(rows, list) and len(rows) >= 2 and value - start == len(rows):
                counts_rows = True
            best = value if best is None else max(best, value)
        return (best, ended, counts_rows)

    def skip_seen_offset(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """#111 E19: a repeated search whose row offset lies inside rows this search already showed is sent at the next offset
        the store gave for it."""
        key = self.page_arg(name)
        sent = args.get(key) if key else None
        if not key or not isinstance(sent, int) or isinstance(sent, bool) or sent <= 0:
            return (args, '')
        best, ended, counts_rows = self.seen_offsets(name, args, key)
        if best is None or ended or not counts_rows or sent >= best:
            return (args, '')
        top = (((self.schemas.get(name) or {}).get('properties') or {}).get(key) or {}).get('maximum')
        if isinstance(top, (int, float)) and not isinstance(top, bool) and best > top:
            return (args, '')
        faults: list[str] = []
        call = _shape({**args, key: best}, self.schemas.get(name), 'args', faults)
        if faults:
            return (args, '')
        self.trace.append({'kind': 'offset_skip', 'tool': name, 'sent': sent, 'to': best})
        return (call, f'\n[Cairn] {key} is a row offset (the first row to show), not a page number: this search showed its rows before {best} already, so Cairn sent {key}={best} (you sent {sent}).')

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
        key = self.page_arg(name) if USE_E17_PAGE_ARG_FROM_SHAPE else self.page_args.get(name)
        if not key or not fresh or not isinstance(args.get(key), int):
            return args
        if USE_E17_PAGE_ARG_FROM_SHAPE:
            low = self.page_low(name, key)
            return args if args[key] == low or isinstance(args[key], bool) else {**args, key: low}
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
        chained = self.chain_paging(name, call, seen) if USE_E27_CHAIN_PAGING and self.brief.get('pick') not in ('cheapest', 'most', 'least') else ''
        return f'\n[Cairn] Nothing has all those words; the same search with {short!r} shows:\n' + self.show(seen) + chained

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
        stop_at = self.step + self.left() - self.tail() - self.msgs_free() - OPEN_LATER * len(self.items) - HUNT_KEEP // 2
        for number, item in enumerate(self.items, 1):
            wanted = [m.split(':', 1)[1].strip() for m in item.musts if ':' in m and m not in [need for _, need in self.refs]]
            wanted = [w for w in wanted if w and len(w.split()) <= 3 and not re.search(OPEN_NOT, w, re.I)]
            noun = item.name if len(item.name.split()) <= 3 else ' '.join(item.name.split()[-2:])
            if USE_E9_HEAD_OF_LONG_NAME and len(item.name.split()) > 3:
                head = re.split(JOIN_WORDS, item.name.strip(), maxsplit=1, flags=re.I)[0].split()
                noun = ' '.join(head) if 0 < len(head) <= 3 else ' '.join(head[-2:]) if head else noun
            tries = list(dict.fromkeys(([' '.join(wanted[:2] + [noun])] if wanted else []) + [noun]))
            singles = list(dict.fromkeys((f'{w} {noun}'.strip() for w in wanted[:2]))) if USE_G19_SINGLE_VALUE_OPENERS and len(wanted) >= 2 else []
            remember = USE_G19_SINGLE_VALUE_OPENERS or USE_G20_FRONT_PAGING or USE_G23_BIG_FRONT_PAGING or USE_G24_OWN_LANE_RANK
            at = 0
            while at < len(tries):
                query = tries[at]
                at += 1
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
                    if remember and wanted and query != noun:
                        self.remember_own(number, tool, args, seen)
                    if singles:
                        tries = tries[:at] + [q for q in tries[at:] if q not in singles or q == noun]
                    self.searched = (self.searched + [dict(args)])[-8:]
                    shown.append(f'{query}: ' + self.show(seen))
                    shown += self.more_rows_for_keys(tool, args, seen, stop_at, number)
                    if self.rows_of_item(number) >= OPEN_EACH:
                        break
                else:
                    shown.append(f"{query}: {'the store refused the call: ' + self.show(seen) if _failed(seen) else 'no rows'}")
                    if singles and at == 1 and wanted and not _failed(seen) and not (isinstance(seen, dict) and seen.get('error')):
                        tries = tries[:at] + [q for q in singles if q not in tries] + tries[at:]
                        self.trace.append({'kind': 'single_value_openers', 'item': number, 'queries': singles})
        if (USE_G20_FRONT_PAGING and self.brief.get('pick') == 'cheapest') or (USE_G23_BIG_FRONT_PAGING and self.clock_bound()):
            shown += self.front_pages(tool, field, stop_at)
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
        if USE_G22_DEVICE_SERIES_SWEEP:
            swept = self.series_sweep()
            if swept:
                shown.append(swept)
        self.env_gap = ENV_GAP
        self.trace.append({'kind': 'first_moves', 'rows': len(self.rows), 'opened': len(self.full)})
        shown += ['read in full: ' + self.page_text(ident, record) for ident, record in self.full.items()]
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

    def remember_own(self, number: int, tool: str, args: dict[str, Any], seen: Any) -> None:
        """#97 item 19: the rows of one of the item's own value searches, kept in order, and the search for item 20's paging."""
        mine = self.own_rows.setdefault(number, [])
        mine += [i for i, _ in self.records_in(seen) if i not in mine]
        mark = self.call_mark(tool, args)
        if (number, mark) not in self.own_searches:
            self.own_searches.append((number, mark))

    @_soft(list)
    def front_pages(self, tool: str, field: str, stop_at: int) -> list[str]:
        """#97 items 20/23: once an episode, after the opening searches: of the items' own value searches whose reply has a next
        page, matches at most FRONT_SPAN pages of rows, is not already in price order (cheapest picks) and whose first page
        mostly carries that item's own label, the largest FRONT_SEARCHES (on clock-bound tasks one per item when more) are paged
        FRONT_PAGES further while FRONT_ROOM steps remain before the first-moves stop line. A page already read costs nothing; the
        rows join the item's own rows; each page read is shown. Item 20 alone runs on cheapest picks, item 23 on any pick when
        the clock binds."""
        pick = self.brief.get('pick')
        big = USE_G23_BIG_FRONT_PAGING and self.clock_bound()
        cat_key = self.category_key()
        if self.front_paged or not self.own_searches or not cat_key or not (big or (USE_G20_FRONT_PAGING and pick == 'cheapest')):
            return []
        found = []
        for number, mark in self.own_searches:
            seen = self.read_calls.get(mark)
            sent = self.read_args.get(mark)
            if seen is None or not sent or sent[0] != tool:
                continue
            plan = self.paging(tool, seen)
            if not plan or plan[3] is None:
                continue
            count, shown, _, _ = plan
            if not shown or count <= shown or count > shown * FRONT_SPAN or (pick == 'cheapest' and self.price_sorted(seen)):
                continue
            rows = [i for i, _ in self.records_in(seen)]
            item = self.items[number - 1]
            mine = sum((1 for i in rows if self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))))
            if mine * 2 > len(rows):
                found.append((count, number, mark, sent[1], seen))
        if not found:
            return []
        self.front_paged = True
        take = max(FRONT_SEARCHES, len(self.items)) if big else FRONT_SEARCHES
        found.sort(key=lambda f: -f[0])
        texts: list[str] = []
        fresh = 0
        for _, number, mark, call, seen in found[:take]:
            walked = {mark}
            mine = self.own_rows.setdefault(number, [])
            for _ in range(FRONT_PAGES):
                if stop_at - self.step < FRONT_ROOM or self.finished:
                    break
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
                else:
                    reply = self.act(tool, call)
                    if reply is None:
                        break
                    seen = reply.get('observation')
                    self.observe(tool, seen)
                    fresh += 1
                    texts.append(f'{call.get(field)} ({page[2]}={_dump(page[3])}): ' + self.show(seen))
                mine += [i for i, _ in self.records_in(seen) if i not in mine]
        self.trace.append({'kind': 'front_pages', 'searches': [(f[1], f[0]) for f in found[:take]], 'read': fresh})
        return texts

    @_soft('')
    def series_sweep(self) -> str:
        """#97 item 22: on clock-bound tasks with a part bound to a device by a series system the parts carry: when one or two
        read parts meet every requirement as written (E49's proof, with items 13-17), and no device row names their series yet,
        Cairn pages the device's earlier searches (oldest first; pages already read cost nothing, SERIES_PAGES fresh ones at
        most) until a device row names one, then reads up to two such rows; once per device and series set."""
        cat_key = self.category_key()
        plan = self.value_template()
        if not self.bound or not cat_key or not plan or not self.clock_bound() or self.left() - self.tail() < BIG_ROOM or self.finished:
            return ''
        tool = plan[0]
        notes = []
        for part, main in self.bound:
            index = self.fit_index()
            system = self.link_system(part, main, index)
            if not system or system[0] != part:
                continue
            proven = [i for _, i, _ in self.proven(part) if self.buyable(i)]
            if not 1 <= len(proven) <= 2:
                continue
            wanted = sorted({f for i in proven for f in (index['marks'].get(i) or {}).get('fam') or []} & set(system[2]))
            key = (main, tuple(wanted))
            if not wanted or key in self.series_swept:
                continue

            def naming() -> list[str]:
                marks = self.fit_index()['marks']
                return [i for i in self.item_ids(main) if set((marks.get(i) or {}).get('named') or []) & set(wanted) and self.buyable(i)]
            if naming():
                continue
            device = self.items[main - 1]
            words = _roots(device.name) | {r for need in self.item_needs(main) for r in _roots(need.partition(':')[2])}
            walks = [(mark, args) for mark, (name, args) in list(self.read_args.items()) if name == tool and _roots(' '.join((v for v in args.values() if isinstance(v, str)))) & words and any((main in self.line_items(i, cat_key) for i, _ in self.records_in(self.read_calls.get(mark))))]
            if not walks:
                continue
            self.series_swept.add(key)
            fresh = 0
            for mark, args in walks:
                seen = self.read_calls.get(mark)
                walked, call = ({mark}, args)
                while not naming() and fresh < SERIES_PAGES and self.left() - self.tail() >= BIG_ROOM and not self.finished and not self.rushed():
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
                    if reply is None:
                        break
                    seen = reply.get('observation')
                    self.observe(tool, seen)
                    fresh += 1
                if naming() or fresh >= SERIES_PAGES or self.left() - self.tail() < BIG_ROOM or self.finished:
                    break
            named = naming()
            read = self.fill_facts([i for i in named if i not in self.full][:2], self.tail() + SERIES_KEEP) if named else []
            self.trace.append({'kind': 'series_sweep', 'device': main, 'series': wanted, 'pages': fresh, 'named': named[:4]})
            head = f"[Cairn paged the {device.name} searches for a row naming series {', '.join(wanted)}, the series of the read {self.items[part - 1].name} {', '.join(proven)} that meets its requirements ({fresh} page(s), one step each)] "
            notes.append(head + (f"{device.name} rows naming it: {', '.join(named[:4])}" if named else f'no {device.name} row seen names it') + ('\n' + '\n'.join(read) if read else ''))
        return '\n'.join(notes)

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
        wanted = {w for _, need in self.needs_now() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())}
        each = WIDEN_EACH if len(self.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in self.item_groups():
            opened = sum((1 for i in ids if i in self.full))
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if USE_G24_OWN_LANE_RANK:
                pool = self.rank_lane(ids, pool)
            else:
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

    def need_words(self, number: int) -> set[str]:
        """#97 items 24/30: the words (three or more letters or digits, joined capitalised words split) of an item's live
        requirement values."""
        return {w for owner, need in self.needs_now() if owner == number for w in re.findall('[a-z0-9]{3,}', _words(need.split(':', 1)[-1]))}

    def rank_lane(self, ids: list[str], pool: list[str]) -> list[str]:
        """#97 item 24: a widen lane's unread rows: first the rows of the own value searches of the items owning the lane's
        label, then the most matches with those items' requirement words (every item's for a label no item owns), then on
        cheapest picks the lower price."""
        cat_key = self.category_key()
        label = str((self.rows.get(ids[0]) or {}).get(cat_key) or '') if cat_key and ids else ''
        owners = [n for n, item in enumerate(self.items, 1) if label and self.owns(item, label)]
        words: set[str] = set()
        for number in owners or range(1, len(self.items) + 1):
            words |= self.need_words(number)
        own = {i for n in owners for i in self.own_rows.get(n, [])}
        cheapest = self.brief.get('pick') == 'cheapest'

        def rank(ident: str) -> tuple[bool, int, float]:
            price = self.price_of(ident)
            said = set(re.findall('[a-z0-9]{3,}', _words(_dump(self.rows.get(ident) or {}))))
            return (ident not in own, -len(words & said), (price if price is not None else float('inf')) if cheapest else 0.0)
        return sorted(pool, key=rank)

    def ready_to_ask(self) -> bool:
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
        if USE_CHIDORI_ASK:
            return self.ch_interview()
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
        if USE_CHIDORI_ASK:
            return self.ch_late_question()
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
                if USE_E7_ITEM_PRICE_CAPS:
                    rows = [i for i in rows if not self.over_cap(number, i)]
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
        if not crowded and not (USE_ASK_LOST and any((_count(x) == 1 for x in _items(heard.get('lost'))))):
            flat = [(number, label) for number, labels in parts for label in labels]
            self.list_taken = max(self.list_taken, len(flat))
            for number, labels in parts:
                new = self.items[number - 1].told[had[number - 1]:]
                if new and flat.index((number, labels[max(0, self.answered_label(labels, list(new)))])) >= 1:
                    self.lists_read = True
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
        self.asked_now = set(labels)  # #101 item 9: what this question asked about
        self.take_heard(heard, told)
        self.asked_now = set()
        new = [need for item, size in zip(self.items, had) for need in item.told[size:]]
        for n, (item, size) in enumerate(zip(self.items, had), 1):
            if len(item.told) > size:
                self.lane_found[n] = self.lane_found.get(n, 0) + 1
        crowded = any((_count(x) == 1 for x in _items(heard.get('crowded'))))
        self.heard_lost = USE_ASK_LOST and any((_count(x) == 1 for x in _items(heard.get('lost'))))
        if self.heard_lost and len(labels) >= 6:
            self.list_limit = max(1, len(labels) // 2)
            self.trace.append({'kind': 'list_shortened', 'item': number, 'cap': self.list_limit})
        self.settle(number, labels, shown, new, crowded)

    def list_cap(self) -> int:
        return LIST_SIZE if len(self.items) == 1 else LIST_MIXED

    def list_now(self) -> int:
        """Labels the next list names: few in the episode's first questions, since nothing shows yet how this shopper takes a list."""
        if USE_ASK_WIDTHS:
            one = len(self.items) == 1
            full = min(self.list_cap(), ASK_FULL if one else ASK_FULL_MIXED)
            if self.list_asks == 0:
                width = (3 if self.msgs_free() <= 1 else ASK_FIRST) if one else ASK_FIRST_MIXED
            elif self.lists_read:
                width = full
            elif self.list_taken:
                width = 2 * self.list_taken
            else:
                width = 4 if one else 3
            width = min(width, full)
            return max(1, min(width, self.list_limit) if self.list_limit else width)
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
        if not crowded and not self.heard_lost:
            self.list_taken = max(self.list_taken, len(labels))
            if hit >= 1:
                self.lists_read = True
        ahead = labels[hit + 1:] if hit >= 0 else labels[:0:-1]
        used.update((_words(label) for label in labels[:max(hit, 0) + 1]))
        rest = [label for label in self.lanes.get(number) or [] if label not in labels]
        self.lane_misses[number] = 0 if hit >= 0 else self.lane_misses.get(number, 0) + 1
        if USE_ASK_FRESH_FIRST and hit < 0:
            fresh = [label for label in rest if _words(label) not in used][:2]
            self.lanes[number] = fresh + ahead + [label for label in rest if label not in fresh]
        elif self.lane_misses[number] and self.lane_misses[number] % 2 == 0:
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
        prompt = (PLAN_ASK if USE_ASK_PLAN_DECOY else PLAN).replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
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
            if USE_S101_7_REPLY_NOTES:
                self.reply_notes(answer)
            sent.append((number, text, words))
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    @_soft(None)
    def reply_notes(self, answer: Any) -> None:
        """#101 item 7: a reply to one question holding more than one passage: the first is the answer, the further ones are
        the world talking, so they go to the world-note reader like a note on any other call (the meaning is queued for the
        next summary or result)."""
        passages: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, str) and len(node.split()) >= 2:
                passages.append(node)
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(answer, 0)
        if len(passages) < 2:
            return
        lines = [line for line in self.loose_lines(answer) if line != passages[0]]
        if not lines:
            return
        meaning = self.take_note(lines)
        self.trace.append({'kind': 'reply_notes', 'lines': len(lines), 'meaning': meaning[:200]})
        if meaning:
            self.queued.append(f"(seen in a shopper's reply) {meaning}")

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
            if entry.get('points') is True and (number, need) not in self.refs and not (USE_E4_CONDITIONS_NOT_LOOKUPS and _conditional(need)):
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
                if USE_S101_9_DONT_CARE_DROP or USE_S101_10_DONT_CARE_POINTER:
                    self.dont_care(number, str(entry['property']), told)
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.budget = self.unit_budget(heard['budget'], True) if USE_H103_2_BUDGET_UNIT else heard['budget']

    def dont_care(self, number: int, prop: str, told: list[str]) -> None:
        """#101 items 9/10: a property the reply says the shopper does not mind about releases the live requirement on it
        when the question being read did not ask about it (9), or when that requirement points at another listing whose
        value is still unread (10)."""
        item = self.items[number - 1]
        live = [m for m in item.musts + item.told if m not in item.dropped]
        hit = [m for m in live if _words(m.split(':')[0]) == _words(prop.split(':')[0])] or self.closest_property(live, prop)
        if not hit:
            return
        need = hit[0]
        asked = any((_spelled(prop) & _spelled(label) for label in self.asked_now))
        pointer = need in self.pending_refs()
        if (USE_S101_9_DONT_CARE_DROP and not asked) or (USE_S101_10_DONT_CARE_POINTER and pointer):
            item.dropped.append(need)
            told.append(f'dropped {need}')
            self.trace.append({'kind': 'dont_care_drop', 'need': need[:120], 'asked': asked, 'pointer': pointer})

    def unit_budget(self, budget: Any, reply: bool) -> Any:
        """#103 item 2: a new budget read in the unit of the budget in force: E33's factor-of-100 rescale (for a reply too),
        and a figure with a fractional part while the budget and every priced row are whole numbers (the store counts in
        its smallest unit) read x100 when that brings it within BUDGET_JUMP of the budget in force and it was not within
        BUDGET_NEAR before."""
        if reply:
            budget = self.rescaled_budget(budget)
        held = self.budget
        if not isinstance(budget, float) or budget.is_integer() or budget <= 0 or not isinstance(held, (int, float)) or isinstance(held, bool) or held <= 0 or not float(held).is_integer():
            return budget
        prices = [p for p in (self.price_of(i) for i in self.rows) if p is not None]
        if len(prices) < BUDGET_PRICES or any((not float(p).is_integer() for p in prices)):
            return budget
        scaled = round(budget * 100, 6)
        if 1 / BUDGET_NEAR <= budget / held <= BUDGET_NEAR or not 1 / BUDGET_JUMP <= scaled / held <= BUDGET_JUMP:
            return budget
        scaled = int(scaled) if float(scaled).is_integer() else scaled
        self.trace.append({'kind': 'budget_unit', 'said': budget, 'read': scaled})
        return scaled

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        if self.quiet_reply(sent):
            self.trace.append({'kind': 'quiet_reply', 'a': sent[0][2][:80]})
            return {}
        self.hear_reads += 1
        prompt = (HEAR_ASK if USE_ASK_LOST else HEAR) + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
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
        if USE_E33_BUDGET_TIMES_100:
            budget = self.rescaled_budget(budget)
        if USE_H103_2_BUDGET_UNIT:
            budget = self.unit_budget(budget, False)
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
            if not hit and USE_E32_DROP_BY_PROPERTY_WORDS:
                hit = self.closest_property(live, target)
            item.dropped.extend(hit or [target])
            changed.append(f'dropped {target}')
        if USE_E37_LATE_DROP_HUNT and drops and self.brief.get('pick') == 'cheapest':
            self.late_drop = True
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

    def rescaled_budget(self, budget: Any) -> Any:
        """#111 E33: a note's new budget more than BUDGET_JUMP times off the one in force, that a factor of 100 brings within
        BUDGET_NEAR of it, read in the budget's own unit."""
        held = self.budget
        if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0 or not isinstance(held, (int, float)) or isinstance(held, bool) or held <= 0:
            return budget
        if 1 / BUDGET_JUMP <= budget / held <= BUDGET_JUMP:
            return budget
        for scaled in (budget / 100, budget * 100):
            if 1 / BUDGET_NEAR <= scaled / held <= BUDGET_NEAR:
                scaled = int(scaled) if float(scaled).is_integer() else round(scaled, 6)
                self.trace.append({'kind': 'budget_rescaled', 'said': budget, 'read': scaled})
                return scaled
        return budget

    @staticmethod
    def closest_property(live: list[str], target: str) -> list[str]:
        """#111 E32: the live requirement whose property-name word stems overlap the dropped one's most (one holds the other,
        or at least half of them shared)."""
        want = _spelled(target.split(':')[0])
        best: tuple[tuple[bool, int], str] | None = None
        for need in live:
            have = _spelled(need.split(':')[0])
            shared = len(want & have)
            if not shared:
                continue
            inside = want <= have or have <= want
            if not inside and shared * 2 < max(len(want), len(have)):
                continue
            if best is None or (inside, shared) > best[0]:
                best = ((inside, shared), need)
        return [best[1]] if best else []

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
        pointers = {m for _, m in self.refs}
        limits = {need: _measured_limit(need.split(':', 1)[1]) if USE_G12_BARE_FIGURE_NOTE and ':' in need and need not in pointers else None for _, need in needs[:8]}
        for ident in ids[:12]:
            block = _fact_block(self.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:80]}' + _bare_note(v, limits.get(need)) + (self.split_note(need, v) if USE_H103_4_SPLIT_NOTE else '') for k, v in block.items() if _names(_stem(prop), _stem(k))]
                tag = f'(item {number}) ' if many else ''
                if hits:
                    parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]))
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        if not lines:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)

    def split_note(self, need: str, held: Any) -> str:
        """#103 item 4: for an attribute value holding a plain multi-word wanted value only with other words between its words
        (no list part holds them as one run), a note that it is another value; '' otherwise."""
        value = need.partition(':')[2].strip() if ':' in need else ''
        if need in {m for _, m in self.refs} or not _plain_words_value(value) or _split_run(value, held) is not False:
            return ''
        return f' [holds the words of "{value}" only apart: another value]'

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
        return self.cart_lines.get(ident, 1)

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def cut_record(self, record: Any, clip: int, limit: int) -> str:
        """A record as a judge prompt shows it, cut to `limit`; #111 E31: when it does not fit, its plain fields come first
        and its nested blocks after, so the cut falls inside the nested facts and the prose stays whole."""
        text = _dump(_clip(record, clip))
        if not USE_E31_CUT_KEEPS_PROSE or len(text) <= limit or not isinstance(record, dict):
            return text[:limit]
        plain = {k: v for k, v in record.items() if not isinstance(v, (dict, list))}
        nested = {k: v for k, v in record.items() if isinstance(v, (dict, list))}
        return _dump(_clip({**plain, **nested}, clip))[:limit]

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return self.cut_record(record, 400, 3000) if record else '(no record seen)'

    def others(self, skip: list[str], limit: int=30) -> str:
        lines = []
        order = list(reversed(self.seen_order))
        if self.brief.get('pick') == 'cheapest':
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float('inf')))
        for ident in order:
            if ident in skip or self.sold_out(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(self.cut_record(record, 160, 900))
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
            if USE_G69_CODE_EXEMPT_SERIES:
                named = set(index['marks'].get(devices[0], {}).get('codes') or [])
                exempt = [l for l in parts if named and any((_covers_code(p, c) for p in self.fit_set(l, index) for c in named)) and not any((_code_hit(b, c) for b in index['marks'].get(l, {}).get('bars') or [] for c in named))]
                if exempt:
                    self.trace.append({'kind': 'series_exempt', 'parts': exempt})
                linked = self.link_faults(part, main, devices[0], [l for l in parts if l not in exempt], index)
            else:
                linked = self.link_faults(part, main, devices[0], parts, index)
            if linked:
                faults += linked
                continue
            codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
            if not codes:
                continue
            barred = lambda i: set(index['marks'].get(i, {}).get('bars') or [])
            covers = lambda i: any((_covers_code(p, c) for p in self.fit_set(i, index) for c in codes)) and not any((_code_hit(p, c) for p in barred(i) for c in codes))
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
            plain = {n: self.plain_values(n) for n in parts + [main]} if USE_H103_6_FIT_PLAN_VALUES else {}
            holding: dict[str, dict[int, list[str]]] = {}  # #103 item 6: code -> part item -> parts holding every plain value

            def counted(p: int, i: str, code: str) -> None:
                tally.setdefault(code, {}).setdefault(p, 0)
                tally[code][p] += 1
                if plain.get(p) and self.holds_values(i, p, plain[p]):
                    held = holding.setdefault(code, {}).setdefault(p, [])
                    if i not in held:
                        held.append(i)
            for p in parts:
                for i in self.item_ids(p):
                    if self.sold_out(i) or (USE_H103_7_FIT_PLAN_CAPS and self.over_cap(p, i)):
                        continue
                    for code in self.fit_set(i, index):
                        counted(p, i, code)
            for p in parts:
                system = self.link_system(p, main, index)
                for i in self.item_ids(p) if system and system[0] == p else []:
                    if USE_H103_7_FIT_PLAN_CAPS and self.over_cap(p, i):
                        continue
                    for code in set((index['marks'].get(i) or {}).get('fam') or []) & system[2] if not self.sold_out(i) else ():
                        counted(p, i, code)
            if USE_E2_FIT_TIE_BY_CODE:
                ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values()), kv[0]))
            else:
                ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))
            valued = [p for p in parts if plain.get(p)]
            if valued:  # #103 item 6: series whose parts hold every plain required value of more bound items first
                ranked.sort(key=lambda kv: -sum((1 for p in valued if holding.get(kv[0], {}).get(p))))
            ranked = ranked[:6]
            devices = {code: [i for i in self.item_ids(main) if code in ((index['marks'].get(i) or {}).get('named') or []) or any((_code_hit(code, c) for c in index['marks'].get(i, {}).get('codes') or []))] for code, _ in ranked}
            if plain.get(main):
                devices = {code: sorted(ids, key=lambda i: not self.holds_values(i, main, plain[main])) for code, ids in devices.items()}
            devices = {code: ids[:2] for code, ids in devices.items()}
            text = '; '.join((f'{code}: ' + ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') + (self.holding_text(holding.get(code, {}), valued) if valued else '') for code, per in ranked))
            if text:
                lines.append(f"codes the {self.items[main - 1].name} candidates' parts name -> {text}")
            if valued:
                switch = self.device_switch(main, valued, ranked, devices, holding)
                if switch:
                    lines.append(switch)
            if USE_H103_7_FIT_PLAN_CAPS:  # #103 item 7: the plan names each part's price ceiling
                for p in parts:
                    cap = self.item_cap(p, request_only=True)
                    if cap is not None:
                        lines.append(f"item {p} ({self.items[p - 1].name}) may cost at most {cap:g} a unit (the request's own limit): a part over it counts for no code above")
            sets = self.cheapest_sets(main, parts, index) if self.brief.get('pick') == 'cheapest' else ''
            if sets:
                lines.append(sets)
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as, or when the device's text names the series of the part's code; try one search of a code or series with the item's words, and if it finds nothing list the device's category and read the codes on its rows] " + ' | '.join(lines)

    def plain_values(self, number: int) -> list[tuple[str, str]]:
        """#103 item 6: an item's plain word requirements (no pointer, code, limit, exclusion, bound word or shopper's words
        description) as (property, value)."""
        out = []
        for need, prop, value in self.plain_needs(number):
            if self.described(number, need) or _codes(value) or _need_limit(value) or re.search(NEGATION, value, re.I):
                continue
            if re.search(OPEN_NOT, re.sub('\\s+or\\s+', ' ', value), re.I) or not _lit_words(value):
                continue
            out.append((prop, value))
        return out

    def holds_values(self, ident: str, number: int, plain: list[tuple[str, str]]) -> bool:
        """#103 item 6: whether a listing's own values (row and page values, not keys) hold the words of one choice of every
        plain required value; for a value no seen listing of the item writes (a bare "yes"), the property's words instead."""
        def values(i: str) -> set[str]:
            row = self.rows.get(i) or {}
            texts = [str(v) for v in row.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
            texts += [str(v) for v in (_fact_block(self.full.get(i) or {}) or {}).values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
            return _lit_words(' '.join(texts))
        have = values(ident)
        for prop, value in plain:
            options = [_lit_words(o) for o in re.split('\\s+or\\s+', value) if _lit_words(o)]
            if any((_words_hold(o, have) for o in options)):
                continue
            written = any((_words_hold(o, values(i)) for i in self.item_ids(number) for o in options))
            if written or not _words_hold(_lit_words(prop), have):
                return False
        return True

    @staticmethod
    def holding_text(per: dict[int, list[str]], valued: list[int]) -> str:
        """#103 item 6: the parts of one code holding every plain required value, per bound item."""
        named = [f"{', '.join(per[p][:3])}" for p in valued if per.get(p)]
        return (' [parts holding every required value: ' + '; '.join(named) + ']') if named else ' [no part holding every required value seen]'

    def device_switch(self, main: int, valued: list[int], ranked: list[tuple[str, dict[int, int]]], devices: dict[str, list[str]], holding: dict[str, dict[int, list[str]]]) -> str:
        """#103 item 6: when the cart's device takes only codes with no part holding every plain required value of some bound
        item while another code has such parts for all of them: a note to switch the device."""
        cart = [i for i in self.cart_ids() if i in self.item_ids(main)]
        if len(cart) != 1:
            return ''
        device = cart[0]
        mine = [code for code, ids in devices.items() if device in ids]
        good = [code for code, _ in ranked if all((holding.get(code, {}).get(p) for p in valued))]
        if not mine or not good or any((code in good for code in mine)):
            return ''
        seen = [i for code in good for i in devices.get(code, []) if i != device][:2]
        return f"the cart's {self.items[main - 1].name} {device} takes {', '.join(mine[:3])}, and no part of those holds every required value; codes whose parts do: {', '.join(good[:3])}" + (f" (devices seen: {', '.join(seen)})" if seen else '') + ': switch the device'

    @_soft('')
    def cheapest_sets(self, main: int, parts: list[int], index: dict[str, Any]) -> str:
        """The lowest totals of one device of the item with one part of each bound item its codes fit, among buyable rows
        seen, requirements not checked: the pick counts the whole order, not each line on its own."""
        options = []
        systems = {part: self.link_system(part, main, index) for part in parts}
        shelves = {part: sorted(((self.price_of(i), i, self.fit_set(i, index), index['marks'].get(i, {}).get('bars') or []) for i in self.item_ids(part) if self.price_of(i) is not None and self.buyable(i) and not (USE_E7_ITEM_PRICE_CAPS and self.over_cap(part, i))), key=lambda o: (o[0], o[1])) for part in parts}
        for device in self.item_ids(main):
            codes = index['marks'].get(device, {}).get('codes') or []
            price = self.price_of(device)
            if not (codes or (index['marks'].get(device, {}).get('named'))) or price is None or not self.buyable(device):
                continue
            if USE_E7_ITEM_PRICE_CAPS and self.over_cap(main, device):
                continue
            total = price * self.items[main - 1].qty
            picks = [device]
            for part in parts:
                system = systems[part]
                fitting = next((i for _, i, fits, bars in shelves[part] if (any((_covers_code(p, c) for p in fits for c in codes)) and not any((_code_hit(p, c) for p in bars for c in codes))) or (system and system[0] == part and self.link_holds(i, device, system[2], index) is True)), None)
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
            if USE_H103_3_STORE_WORDS:
                if not words or not words <= self.store_values(prop):
                    return False  # #103 item 3: a word no read listing shows for the property (a limit, an exclusion, a description)
            elif not words or words & {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'}:
                return False
            if not any((_names(_spelled(prop), _spelled(k)) and words <= split(str(v)) for k, v in block.items())):
                return False
        return True

    def store_values(self, prop: str) -> set[str]:
        """#103 item 3: the words of the values the listings read so far show under keys named for a property."""
        out: set[str] = set()
        for record in self.full.values():
            for key, value in (_fact_block(record) or {}).items():
                if _names(_spelled(prop), _spelled(key)) and not isinstance(value, (dict, list)):
                    out |= set(re.sub('[,/;|]', ' ', _words(str(value))).split())
        return out

    def fill_short_cart(self) -> bool:
        """Before a forced order: add, for each item with no line, the cheapest opened listing that states its requirements.
        False when an item stays uncovered, so the episode ends by refusing rather than with a short order."""
        cart = self.roles.get('cart') or ''
        bound = {n for pair in self.bound for n in pair}
        for number in self.missing_items():
            item = self.items[number - 1]
            fits = [i for i in self.item_ids(number) if i in self.full and i not in self.cart_lines and self.buyable(i) and self.price_of(i) is not None and self.states_needs(i, number)]
            if USE_E7_ITEM_PRICE_CAPS:
                fits = [i for i in fits if not self.over_cap(number, i)]
            if USE_R16_REFUSED_LINES:
                fits = [i for i in fits if i not in self.refused]
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

    def own_label(self, number: int) -> str:
        """#97 item 26: the item's own store label: a label seen on rows whose word roots equal the item's name or shelf and
        which the item owns (the one most rows carry); '' when none."""
        cat_key = self.category_key()
        if not cat_key or not 1 <= number <= len(self.items):
            return ''
        item = self.items[number - 1]
        targets = [r for r in (_roots(item.name), _roots(item.shelf or '')) if r]
        counts: dict[str, int] = {}
        other: set[str] = set()
        for row in self.rows.values():
            label = str(row.get(cat_key) or '').strip()
            if not label or label in other:
                continue
            if label in counts:
                counts[label] += 1
            elif _roots(label) in targets and self.owns(item, label):
                counts[label] = 1
            else:
                other.add(label)
        return max(counts, key=lambda l: (counts[l], l)) if counts else ''

    @_soft('')
    def nested_labels_line(self) -> str:
        """#97 item 26: a ledger line when one item's own store label sits wholly inside another's (a kind and a wider kind) and
        the brief did not already give each named item that label: the label, not the title, decides the item a listing serves."""
        if len(self.items) < 2 or not self.category_key():
            return ''
        own = {n: self.own_label(n) for n in range(1, len(self.items) + 1)}
        named: set[int] = set()
        for a, first in own.items():
            for b, second in own.items():
                if a != b and first and second and _roots(first) < _roots(second):
                    named |= {a, b}
        if len(named) < 2 or all((_words(self.items[n - 1].shelf or '') == _words(own[n]) for n in named)):
            return ''
        return 'store labels: ' + '; '.join((f"item {n} ({self.items[n - 1].name}) is sold under '{own[n]}'" for n in sorted(named))) + ". The store label, not the title, decides which item a listing serves: take each item's line from its own label."

    @_soft('')
    def no_line_note(self, number: int, ids: list[str], label: dict[str, str]) -> str:
        """#97 item 26: for an item with no cart line: its own label, and the cart lines carrying another label that shares a
        word root with it (with the item owning that label when only one does), which cannot serve it."""
        own = self.own_label(number)
        if not own:
            return ''
        text = f" (its own store label is '{own}'"
        others = []
        for ident in ids:
            theirs = label.get(ident) or ''
            if theirs and _words(theirs) != _words(own) and _roots(theirs) & _roots(own):
                owners = [n for n, item in enumerate(self.items, 1) if self.owns(item, theirs)]
                others.append(f"{ident} carries '{theirs}'" + (f' (the label of item {owners[0]})' if len(owners) == 1 else ''))
        if others:
            text += '; ' + ', '.join(others) + ', so it cannot serve this item'
        return text + f"; take its line from '{own}')"

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
                faults.append(f'item {n} ({item.name}) has no line in the cart' + (self.no_line_note(n, ids, label) if USE_G26_NESTED_LABEL_LEDGER else ''))
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.held_units(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
                fuller = self.fuller_label(n, label[i]) or self.named_label(n, label[i])
                if fuller:
                    faults.append(f'{i} is from {label[i]}, but the store has the label {fuller}, which names item {n} ({item.name}) in full: take the line from there')
                cap = self.item_cap(n, request_only=True)
                price = self.price_of(i)
                each = self.brief.get('budget_scope') == 'each' and isinstance(self.budget, (int, float)) and price is not None and self.budget < price <= 10 * self.budget
                if owner[i] == [n] and cap is not None and price is not None and cap + 1e-09 < price <= CAP_SANITY * cap and not each:
                    faults.append(f"{i} costs {price:g} a unit, over the request's own limit of {cap:g} for item {n} ({item.name})")
                if USE_S101_8_WORD_CONTRADICTION and owner[i] == [n]:
                    faults += self.word_faults(i, n)
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

    @_soft(list)
    def word_faults(self, ident: str, number: int) -> list[str]:
        """#101 item 8: a cart line whose own attribute, under a key named for a plain word requirement's property (the same
        words, or those plus the item's own name), shares no word or word form with any choice of the wanted value. Limits,
        ranges, exclusions, codes, pointers and the shopper's own descriptions are left out; a missing key is no fault."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        out: list[str] = []
        for need, prop, value in self.plain_needs(number) if block else []:
            if self.described(number, need) or _codes(value) or _need_limit(value) or re.search(NEGATION, value, re.I) or re.search(OPEN_NOT, re.sub('\\s+or\\s+', ' ', value), re.I):
                continue
            options = [_lit_words(o) for o in re.split('\\s+or\\s+', value) if _lit_words(o)]
            mine = set(_key_words(prop))
            keys = [k for k in block if not isinstance(block[k], (dict, list)) and (set(_key_words(k)) == mine or k in self.fuller_keys(number, prop, [k]))]
            held = [(k, _lit_words(str(block[k]))) for k in keys]
            held = [(k, h) for k, h in held if h]
            if not options or not held:
                continue
            shares = lambda want, have: any((w in have or any((_same_stem(w, x) for x in have)) for w in want))
            if all((not any((shares(o, h) or shares(h, o) for o in options)) for _, h in held)):
                out.append(f'{ident} states {held[0][0]}={str(block[held[0][0]])[:60]}, not the required {need}')
        return out

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

    def note_refused(self, ids: list[str], problems: list[str]) -> None:
        """USE_R16_REFUSED_LINES: the lines a refusing cart check names: a problem that opens on one item ("Item 2 ...")
        refuses that item's one line in the judged cart unless it names another item too; one that opens otherwise refuses
        the one judged line it names. A
        problem naming another line only as context ("the chosen sofa (p-...)") refuses nothing else. The check also writes
        lines it finds fine into its problems ("Item 2: ... is chenille, so this line is fine"): a problem that speaks only
        of word requirements (no figure) which the line meets as its attributes write them refuses nothing (`check_praises`)."""
        for text in problems:
            lead = re.match('\\W*item\\s+(\\d+)\\b', text, re.I)
            if lead:
                number = int(lead.group(1))
                others = {int(n) for n in re.findall('\\bitems?\\s+(\\d+)', text, re.I)} - {number}
                lines = [i for i in ids if self.line_items(i) == [number]] if not others else []  # another item named: not about this line alone
            else:
                lines = [i for i in ids if re.search('(?<![A-Za-z0-9])' + re.escape(i) + '(?![A-Za-z0-9])', text)]
            if len(lines) == 1 and lines[0] not in self.refused and not self.check_praises(lines[0], text):
                self.refused[lines[0]] = text[:200]
                self.trace.append({'kind': 'line_refused', 'line': lines[0], 'why': text[:120]})

    def check_praises(self, ident: str, text: str) -> bool:
        """USE_R16_REFUSED_LINES: whether a cart check's problem about a line speaks of its item's requirements (by property
        or by every word of the value) only where they hold no figure and the line meets them as its attributes write them;
        False when it names none of them or names one with a figure (a measure, a code) or one the line does not meet."""
        number = self.line_owner(ident)
        said = set(_word_roots(text))
        named = []
        for need in self.item_needs(number) if number else []:
            prop, _, value = need.partition(':')
            words = _roots(value)
            if _roots(prop) & said or (words and words <= said):
                named.append(need)
        if not named:
            return False
        return all((not re.search('\\d', need.partition(':')[2]) and self.literal_state(ident, number, need)[0] == 'met' for need in named))

    def check_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.version}' + (f'~{self.cheaper_round}' if USE_E44_CHEAPER_MORE_ROUNDS and self.cheaper_round else '')

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or not self.judges(limit) or (not ids):
            return ''
        self.checked.add(key)
        nearest = USE_P110_1_CHECK_NEAREST and self.nearest_read is True
        check = _check_text()
        if nearest:  # #110 item 1 (prompt): the nearest rule, when the reading found one in the request
            check = check.replace('\nReturn ONE JSON', CHECK_NEAREST + '\nReturn ONE JSON', 1)
        prompt = check + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1)))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.sold_out(i) else '') + self.record_of(i) for i in ids))
        weak = self.weak_lines() if USE_G44_WEAK_LINES and not lead and (USE_G46_WEAK_CART_HEAVY or USE_G46_CHECK_WEAK_SECTION) else {}
        if weak and USE_G46_CHECK_WEAK_SECTION:
            prompt += G46_SECTION_HEAD + '\n'.join((f'{line}: ' + '; '.join((n for n, _, _ in needs)) for line, needs in weak.items()))
        links = self.code_links(ids) if USE_G47_CHECK_CODE_LINKS and self.bound else []
        if links:
            prompt += G47_SECTION_HEAD + '\n'.join(links)
        if weak:
            self.trace.append({'kind': 'check_weak', 'lines': list(weak)})
        if self.brief.get('pick') in ('cheapest', 'most', 'least') or nearest:
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
        judges = self.judges(limit)
        got = None
        if len(judges) > 1 and not lead and not self.audited and self.brief.get('pick') not in ('cheapest', 'most', 'least') and not (weak and USE_G46_WEAK_CART_HEAVY):
            got = self.ask_json(prompt, [LIGHT], 3000)
            if got:
                judges = [HEAVY] if got.get('ok') is False and HEAVY in self.judges(limit) else []
        if judges and not self.no_credit:
            got = self.ask_json(prompt, judges, 3000) or got
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        if USE_R16_REFUSED_LINES and got and got.get('ok') is True and any((i in self.refused for i in ids)):
            for i in ids:
                self.refused.pop(i, None)  # a later check passed a cart holding the line
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in ids]
        if USE_R16_REFUSED_LINES:
            self.note_refused(ids, [str(p) for p in _items(got.get('problems'))])
        problems = [str(p) for p in _items(got.get('problems'))][:5]
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        if USE_P110_10_REVIEW_SET_ASIDE and better and not lead and not self.cart_gaps() and all((self.listing_gaps(b) for b in better)):
            self.trace.append({'kind': 'review_set_aside', 'better': better})  # #110 item 10
            self.verdicts[key] = True
            return ''
        if USE_E46_BETTER_WORD_CONTRADICTION and better:
            kept = [b for b in better if not self.contradicts_words(b)]
            if len(kept) < len(better):
                self.trace.append({'kind': 'better_dropped', 'ids': [b for b in better if b not in kept]})
            better = kept
        if USE_G52_AGREED_SWAP and not lead:
            self.better_by_cart['|'.join(sorted(ids))] = list(better)
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def refuse_trail(self) -> str:
        """Once, before a decline with room left: follow what a requirement points at; what that showed, '' when nothing."""
        live = {need for _, need in self.needs_now()}
        if self.refuse_trailed or self.left() < TRAIL_ROOM or self.clock() > HUNT_BY or not any((need in live and need not in self.ref_values for _, need in self.refs)):
            return ''
        self.refuse_trailed = True
        text, _ = self.trail(HUNT_BY)
        return text + 'Search for those values now and buy what fits; end without a purchase only if nothing does.' if text else ''

    def early_trail(self) -> str:
        """Follow the chains of requirements that point at another listing, once, while there is room to act on what they say."""
        live = {need for _, need in self.needs_now()}
        if self.left() < TRAIL_ROOM or self.clock() > HUNT_BY or not any((need in live and need not in self.ref_values for _, need in self.refs)):
            return ''
        kept = (self.trails, self.trail_blocks)
        self.trails, self.trail_room = 0, (EARLY_ROUNDS if USE_G34_CHAIN_ROUNDS else EARLY_TRAILS)
        self.trail_early = USE_G34_CHAIN_ROUNDS
        try:
            text, _ = self.trail(HUNT_BY)
        finally:
            (self.trails, self.trail_blocks), self.trail_room = kept, None
            self.trail_early = False
        self.trace.append({'kind': 'early_trail', 'left': self.left(), 'values': list(self.ref_values.values())[:3]})
        return text

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
        early = USE_G34_CHAIN_ROUNDS and self.trail_early  # #97 item 34: the early follow's own round and page limits
        pages_open = 0
        while pending and self.trails < (self.trail_room or EARLY_ROUNDS if early else min(6 if self.left() > 15 else 4, self.trail_room or 6)) and (self.cost < SPEND_HARD) and (stop is None or self.clock() <= stop):
            self.trails += 1
            self.trail_opened = len(self.opened)
            rows_before = len(self.seen_order)
            flagged = False  # #97 items 40/41: a link rejected or a page found deferring in this round
            records = []
            for ident in reversed(self.seen_order):
                record = self.full.get(ident) or self.rows.get(ident) if self.page_read(ident) else None
                if record and len(records) < 45:
                    records.append('OPENED ' + self.cut_record(record, 300, 1400))
            for ident in reversed(self.seen_order):
                if len(records) >= 60:
                    break
                if not self.page_read(ident) or not (self.full.get(ident) or self.rows.get(ident)):
                    wide = USE_R16_LINK_PAGE_UNSHOWN and ident in self.full and self.of_shop(' | '.join(pending), ident)  # attributes read, of a shop a chain names: shown uncut like an opened page
                    records.append('ROW ' + (self.cut_record(self.full[ident], 300, 1400) if wide else self.cut_record(self.full[ident], 120, 700) if ident in self.full else self.cut_record(self.rows.get(ident), 80, 300)))
            prompt = _trail_text() + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            if USE_G8_TRAIL_REQUEST:
                prompt += G8_REQUEST_HEAD + self.request[:TRAIL_REQUEST_CHARS]
                if self.chain_findings:
                    prompt += G8_FINDINGS_HEAD + '\n'.join(self.chain_findings[-TRAIL_FINDINGS:])
            prompt += (G8_RECORDS_HEAD if USE_G8_TRAIL_REQUEST else '\n\nRECORDS (newest first):\n') + '\n'.join(records)
            got = self.ask_json(prompt, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in _items(got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need and USE_G39_CHAIN_NEED_OVERLAP:
                    need = self.reworded_need(str(chain.get('need') or ''), pending)
                if not need:
                    continue
                links = [l for l in _items(chain.get('links')) if isinstance(l, dict)]
                if USE_G38_TITLE_CODE_LINK:
                    links = [self.soft_call(self.coded_link, l, l) for l in links]
                ids = [str(l.get('id') or '').strip() for l in links]
                if USE_G40_LINK_REJECT:
                    links, ids, rejected = self.soft_call(self.reject_links, (links, ids, False), need, links, ids)
                    flagged = flagged or rejected
                if chain.get('done') is True and links and all((self.page_read(i) for i in ids)) and str(chain.get('value') or '').strip():
                    if USE_G41_DEFER_PAGE and self.defers(need, ids[-1], str(chain.get('value') or '')):
                        flagged = True
                        continue
                    self.ref_values[need] = str(chain['value']).strip()
                    self.version += 1
                    if USE_E35_CHAIN_OFFER_VIEWS:
                        self.chain_page_ids += [i for i in ids if i in self.rows and i not in self.chain_page_ids]
                    continue
                if links and all((self.page_read(i) for i in ids)) and (not str(links[-1].get('find') or '').strip()):
                    self.ref_values[need] = str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}'
                    self.version += 1
                    if USE_E35_CHAIN_OFFER_VIEWS:
                        self.chain_page_ids += [i for i in ids if i in self.rows and i not in self.chain_page_ids]
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
            cap = LINK_CAP_MEMO if USE_G35_LINK_MEMORY else LINK_CAP_WIDE if USE_E29_WIDER_LINK_SEARCH else LINK_CAP
            if not reach and gaps and tool and target and (self.left() > LINK_STEPS) and (self.link_searches < cap) and USE_G35_LINK_MEMORY:
                found_at: set[int] = set()
                for number, gap in enumerate(gaps[:LINK_GAPS]):  # #97 item 35: up to two missing links a round
                    if self.left() <= LINK_STEPS or self.link_searches >= cap or self.finished:
                        break
                    self.link_found = ''
                    reach += [i for i in self.find_link(*gap) if i not in reach]
                    if self.link_found:
                        seen.append(f'{self.link_found}: an opened page of the shop this chain names that shows the values its description gives; read the value from it')
                        found_at.add(number)
                lost = [l for number, l in enumerate(lost) if number not in found_at]
            elif not reach and gaps and tool and target and (self.left() > 12) and (self.link_searches < (LINK_CAP_WIDE if USE_E29_WIDER_LINK_SEARCH else LINK_CAP)):
                reach = self.find_link(*gaps[0])
                if self.trace and self.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
            if early:
                batch = reach[:max(0, min(ROUND_PAGES, EARLY_PAGES - pages_open))]
                reserve = 3 * len(self.items) + self.tail() + 2
            else:
                batch = reach[:max(0, min(3, 6 - len(seen)))]
                reserve = len(batch) + self.tail() + 1 if USE_E30_CHAIN_OPEN_RESERVE else len(reach) + 4
            if not reach and USE_G34_CHAIN_ROUNDS and flagged and pending and not self.finished and (len(self.seen_order) > rows_before or USE_G8_TRAIL_REQUEST):
                self.trace.append({'kind': 'trail_again', 'round': self.trails})
                continue
            if not reach or not tool or (not target) or (self.left() < reserve) or self.finished:
                break
            opened_now: list[str] = []
            for ident in batch:
                if early and self.left() <= 3 * len(self.items) + self.tail() + 1:
                    break
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
                if faults or self.rushed():
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                pages_open += 1
                self.observe(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.show(reply.get('observation'))[:1500])
                opened_now.append(ident)
            if USE_R16_LINK_KIND_FACTS and opened_now and not self.finished:
                seen += self.own_kind_facts(opened_now, tool)
        text = ''
        if seen:
            text += '[Cairn opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values and USE_E10_POINTER_GIVES_ONLY:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {self.pointer_gives(k)}' for k in self.ref_values)) + '\n'
        elif self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (text, hold)

    @_soft([])
    def own_kind_facts(self, ids: list[str], page_tool: str) -> list[str]:
        """USE_R16_LINK_KIND_FACTS: chain pages just shown whose store label is a requested item's own get their attributes
        read with the attribute view (when that is another tool than the page view), while R16_KIND_ROOM steps beyond the
        tail remain: the listing a chain points at can itself be the cheapest fit (race 16: a pointed-at video camera met
        every requirement of the item, but with its attributes never read the pick check left it out)."""
        facts = self.facts_tool()
        cat_key = self.category_key()
        if not facts or facts == page_tool or not cat_key or self.left() <= self.tail() + R16_KIND_ROOM:
            return []
        kind = lambda i: str((self.rows.get(i) or {}).get(cat_key) or '')
        own = [i for i in ids if i in self.rows and i not in self.full and kind(i) and any((self.owns(item, kind(i)) for item in self.items)) and not self.sold_out(i) and self.buyable(i)]
        if not own:
            return []
        read = self.fill_facts(own[:2], self.tail() + R16_KIND_ROOM)
        if read:
            self.trace.append({'kind': 'link_kind_facts', 'ids': own[:2]})
        return ["(attributes of a pointed-at listing of a requested item's own kind, read so it can be judged as a purchase too) " + r for r in read]

    def not_maker_keys(self) -> tuple[Any, ...]:
        """The row keys that cannot hold the name of who sells or makes a listing: its id, its label and its name. Any other
        text field can, and the value must still appear word for word in the description being matched."""
        return (self.id_key, self.category_key(), self.title_key())

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        not_names = self.not_maker_keys()
        low = about.lower()
        out = []
        for ident in ids:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and k not in not_names and v.strip().lower() not in NO_NAME]
            if ident not in self.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

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
        if USE_G36_LINK_WORDS:
            return self.find_link_words(about, words)
        not_names = self.not_maker_keys()
        known: list[str] = []
        named = {w for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and k not in not_names and v.strip().lower() not in NO_NAME for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        for ident in self.opened_shop(about):
            if USE_G37_ATTR_LINK:
                if self.attrs_hold(ident, values) and not (USE_P110_14_KEYED_PAIRS and self.pairs_hold(ident, about) is False):
                    self.link_found = ident
                    self.trace.append({'kind': 'link_known', 'id': ident})
                    return []
                continue
            stack, leaves = ([self.full.get(ident) or self.rows.get(ident) or {}], [])
            while stack:
                node = stack.pop()
                if isinstance(node, (dict, list)):
                    stack += list(node.values()) if isinstance(node, dict) else node
                else:
                    leaves.append(str(node))
            text = ' '.join(leaves).lower()
            if values and all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values)):
                self.link_found = ident
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
        if USE_E29_WIDER_LINK_SEARCH and plan:
            wider = self.wider_link(about, words, sorted(values), memo)
            if wider:
                return wider
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

    # ---- #97 items 34-43: pointer chains
    def soft_call(self, method: Any, fallback: Any, *args: Any) -> Any:
        """#97 items 38/40: a helper whose failure leaves the chain as the judge gave it (the trace says so)."""
        try:
            return method(*args)
        except Exception as exc:
            self.trace.append({'kind': 'soft_crash', 'where': getattr(method, '__name__', '?'), 'error': f'{type(exc).__name__}: {exc}'[:300]})
            return fallback

    LINK_STOP = ('the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of')  # the king's find_link stop words

    def link_pages_apart(self) -> bool:
        """USE_R16_LINK_PAGE_UNSHOWN: whether the store shows a listing's page with a tool of its own, apart from the
        attribute view (the chain counts only with that page shown)."""
        page, opener = (self.roles.get('offer'), self.roles.get('open'))
        return USE_R16_LINK_PAGE_UNSHOWN and bool(page) and page != opener and bool(self.single_target(page)) and page in self.schemas

    def attrs_hold(self, ident: str, values: Any) -> bool:
        """#97 item 37: whether a listing's attribute block (not its title or prose) holds every value word as a whole word;
        False without a value or a block."""
        block = _fact_block(self.full.get(ident) or self.rows.get(ident) or {}) or {}
        text = ' '.join((str(v) for v in block.values())).lower()
        values = [v for v in values if v]
        if self.link_pages_apart():  # USE_R16_LINK_PAGE_UNSHOWN: word forms count
            return bool(values) and bool(text) and _values_in(values, text)
        return bool(values) and bool(text) and all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values))

    def pairs_hold(self, ident: str, text: str) -> bool | None:
        """#110 item 14: whether an opened page's attributes hold each "property value" pair a link description gives: a
        segment of the description (split at commas, brackets, "and", "with", "or") that starts with one of the page's
        attribute keys must find its value under that key, every number when it holds numbers (another unit is no
        contradiction), else every word. None when no segment starts with a key of the page or the page's shop is not
        named in the description."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        if not block or not text or not self.of_shop(text, ident):
            return None
        norm = lambda v: ' '.join(re.findall('[a-z0-9]+(?:\\.[0-9]+)?', _words(str(v))))
        fold = lambda w: w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') else w
        keys = sorted({norm(k): norm(v) for k, v in block.items() if norm(k) and not isinstance(v, (dict, list))}.items(), key=lambda kv: -len(kv[0]))
        label, body = text.split(':', 1) if ':' in text.split('(')[0] else ('', text)
        own = norm(label)
        verdict = None
        for segment in re.split('[,;()]|\\band\\b|\\bwith\\b|\\bor\\b', body.lower()):
            words = norm(segment).split()
            while words and len(words[0]) <= 3 and not any((k.split()[0] == words[0] for k, _ in keys)):
                words = words[1:]
            seg = ' '.join(words)
            for key, value in keys:
                if not (seg == key or seg.startswith(key + ' ')):
                    continue
                want = seg[len(key):].split()
                if not want or len(want) > 4 or (own and (key in own or own in key)):
                    break
                numbers = [w for w in want if re.fullmatch('[0-9]+(?:\\.[0-9]+)?', w)]
                if numbers:
                    held = set(re.findall('[0-9]+(?:\\.[0-9]+)?', value))
                    good = all((n in held for n in numbers))
                    units = {fold(w) for w in want if w not in numbers} - {fold(w) for w in key.split()}
                    if not good and units - {fold(w) for w in re.findall('[a-z]+', value)}:
                        break
                else:
                    good = {fold(w) for w in want} <= {fold(w) for w in value.split()}
                verdict = good if verdict is None else verdict and good
                break
        if verdict is False:
            self.trace.append({'kind': 'link_pair_contradicted', 'id': ident})
        return verdict

    def name_values(self, ident: str) -> list[str]:
        """#97 items 36/40/42: a row's short name-like texts (who sells or makes it could be any of them), as the king's
        shop_rows reads them."""
        not_names = self.not_maker_keys()
        row = self.rows.get(ident) or {}
        return [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and k not in not_names and v.strip().lower() not in NO_NAME]

    def of_shop(self, about: str, ident: str) -> bool:
        """#97 item 36: whether a row's name-like text appears as a whole word in a link's description (opened or not)."""
        low = about.lower()
        return any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in self.name_values(ident)))

    def attr_name_words(self) -> set[str]:
        """#97 items 36/40: the words of the attribute names seen in read records."""
        return {w for record in self.full.values() for k in (_fact_block(record) or {}) for w in re.findall('[a-z0-9]+', _words(k))}

    def next_token(self, tool: str, observation: Any) -> tuple[str, Any] | None:
        """#97 item 35: the paging parameter and value of a search reply, when its shape names both."""
        shape = self.read_shape(tool, observation) if isinstance(observation, dict) else None
        if not shape or not shape.get('next') or not shape.get('next_arg') or not isinstance(observation, dict):
            return None
        value = observation.get(shape['next'])
        if value is None or isinstance(value, bool) or not isinstance(value, (str, int)) or value == '':
            return None
        return (shape['next_arg'], value)

    def link_search(self, query: str, about: str, candidate: Any) -> list[str]:
        """#97 item 35: one link search, remembered by its words (case ignored) with its rows and next page. A repeat is
        answered from memory without a step when an unopened candidate is left among its rows, it has no next page or none of
        its rows is of the named shop; otherwise its next page is read (one step, counted). A fresh search whose rows are all
        of the shop with no candidate among them is paged while new rows come."""
        plan = self.value_template()
        if not plan or not query.strip():
            return []
        tool, base, field = plan
        key = ' '.join(query.lower().split())
        memo = self.link_memo.get(key)
        cap = LINK_CAP_MEMO if USE_G35_LINK_MEMORY else LINK_CAP_WIDE if USE_E29_WIDER_LINK_SEARCH else LINK_CAP
        room = lambda: self.link_searches < cap and self.left() > LINK_STEPS and not self.finished and not self.rushed()
        if memo is None:
            if not room():
                return []
            faults: list[str] = []
            args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
            if faults:
                return []
            self.link_searches += 1
            reply = self.act(tool, args)
            if reply is None:
                return []
            seen = reply.get('observation')
            self.observe(tool, seen)
            memo = {'rows': [i for i, _ in self.records_in(seen)], 'next': self.next_token(tool, seen), 'args': args}
            self.link_memo[key] = memo
            self.trace.append({'kind': 'link_search', 'words': query[:80], 'rows': len(memo['rows'])})
            pages = AUTO_PAGES
            while USE_G35_LINK_MEMORY and pages > 0 and memo['rows'] and memo['next'] and room() and all((self.of_shop(about, i) for i in memo['rows'])) and not any((candidate(i) for i in memo['rows'])):
                pages -= 1
                if not self.link_page(tool, memo):
                    break
        elif USE_G35_LINK_MEMORY and memo['next'] and room() and not any((candidate(i) for i in memo['rows'])) and any((self.of_shop(about, i) for i in memo['rows'])):
            self.link_page(tool, memo)
        return list(memo['rows'])

    def link_page(self, tool: str, memo: dict[str, Any]) -> bool:
        """#97 item 35: the next page of a remembered link search (one step, counted as a link search); whether new rows came."""
        param, value = memo['next']
        faults: list[str] = []
        args = _shape({**memo['args'], param: value}, self.schemas.get(tool), 'args', faults)
        if faults:
            memo['next'] = None
            return False
        self.link_searches += 1
        reply = self.act(tool, args)
        if reply is None:
            memo['next'] = None
            return False
        seen = reply.get('observation')
        self.observe(tool, seen)
        new = [i for i, _ in self.records_in(seen) if i not in memo['rows']]
        memo['rows'] += new
        memo['next'] = self.next_token(tool, seen) if new else None
        self.trace.append({'kind': 'link_page', 'new': len(new)})
        return bool(new)

    @_soft(list)
    def find_link_words(self, about: str, words: str) -> list[str]:
        """#97 items 36/37/43 (with 35's memory): the link's search words are the chain judge's find words without the
        attribute names seen in read records (the first word, the shop, kept). An opened page of the shop whose attribute
        block holds every value is the link already (37). Searches: all the words (a candidate ends the walk); with three or
        more words the shop with each single value word; the shop alone only when no value search found a candidate. A
        candidate is an unopened row of the shop the description names whose read attributes do not lack a value; ranked by
        the value searches finding it, then the smaller result, then shared description words (none: last). With no search
        candidate, the unopened shop rows seen anywhere sharing a description word, newest first. At most LINK_CANDIDATES."""
        attr_words = self.attr_name_words()
        unshown = self.link_pages_apart()
        said = [w for w in re.findall('[a-z0-9]+', (words or '').lower()) if w not in self.LINK_STOP]
        query_words = said[:1] + [w for w in said[1:] if w not in attr_words]
        shop = query_words[0] if query_words else ''
        named = {w for ident in self.rows if self.of_shop(about, ident) for n in self.name_values(ident) for w in re.findall('[a-z0-9]{3,}', n.lower())}
        values = [w for w in dict.fromkeys(query_words[1:]) if (len(w) >= LINK_WORD_CHARS or re.search('\\d', w)) and w not in named and w != shop]
        if not values:
            values = [w for w in dict.fromkeys(re.findall('[a-z0-9]+', about.lower())) if (len(w) >= LINK_WORD_CHARS or w.isdigit()) and w not in named and w not in self.LINK_STOP and w not in attr_words]
        unshown = self.link_pages_apart()
        for ident in self.opened_shop(about):
            if USE_P110_14_KEYED_PAIRS and self.pairs_hold(ident, about) is False:
                continue  # #110 item 14: a "property value" pair of the description is contradicted under that key
            if self.attrs_hold(ident, values) if USE_G37_ATTR_LINK else bool(values) and all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', ' '.join(_leaves(self.full.get(ident) or self.rows.get(ident) or {})).lower()) for v in values)):
                if unshown and not self.page_read(ident):
                    self.trace.append({'kind': 'link_page_unshown', 'id': ident})
                    return [ident]  # USE_R16_LINK_PAGE_UNSHOWN: the link by its attributes, its page still to show
                self.link_found = ident
                self.trace.append({'kind': 'link_known', 'id': ident})
                return []
        described = {w for w in re.findall('[a-z0-9]+', about.lower()) if (len(w) >= LINK_WORD_CHARS or w.isdigit()) and w not in self.LINK_STOP} - named

        def lacks(ident: str) -> bool:
            block = _fact_block(self.full.get(ident) or {}) or {}
            if not block:
                return False
            text = ' '.join((str(v) for v in block.values())).lower()
            if unshown:  # USE_R16_LINK_PAGE_UNSHOWN: word forms count ("1 piece" for "1 pieces")
                return not _values_in(values, text)
            return any((not re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values))

        def candidate(ident: str) -> bool:
            if unshown:  # USE_R16_LINK_PAGE_UNSHOWN: a row read through the attribute view only still has its page to show
                return not self.page_read(ident) and self.of_shop(about, ident) and not lacks(ident)
            return ident not in self.opened and self.of_shop(about, ident) and not lacks(ident)

        def overlap(ident: str) -> int:
            return len(described & set(re.findall('[a-z0-9]+', _dump(self.rows.get(ident) or {}).lower())))
        hits: dict[str, int] = {}
        size: dict[str, int] = {}

        def run(query: str, value_search: bool) -> list[str]:
            rows = self.link_search(query, about, candidate)
            found = [i for i in rows if candidate(i)]
            for ident in found:
                hits[ident] = hits.get(ident, 0) + (1 if value_search else 0)
                size[ident] = min(size.get(ident, len(rows)), len(rows))
            return found
        if query_words and self.value_template():
            if not run(' '.join(query_words), False) and len(query_words) >= 2 and shop:
                by_value: list[str] = []
                for value in values if len(query_words) >= 3 else []:
                    by_value += run(f'{shop} {value}', True)
                if not by_value:
                    run(shop, False)
        if hits:
            ranked = sorted(hits, key=lambda i: (overlap(i) == 0, -hits[i], size[i], -overlap(i), i))
            self.trace.append({'kind': 'link_candidates', 'ids': ranked[:LINK_CANDIDATES]})
            return ranked[:LINK_CANDIDATES]
        pool = [i for i in reversed(self.seen_order) if (not self.page_read(i) if unshown else i not in self.opened) and self.of_shop(about, i) and overlap(i) > 0]
        pool.sort(key=lambda i: (lacks(i), -overlap(i)))
        return pool[:LINK_CANDIDATES]

    @staticmethod
    def reworded_need(name: str, pending: list[str]) -> str | None:
        """#97 item 39: the pending requirement a chain names in other words: the highest share of shared words, when it
        reaches NEED_SHARE and leads the next one by more than NEED_LEAD."""
        said = set(re.findall('[a-z0-9]+', name.lower()))
        if not said or not pending:
            return None
        scores = sorted(((len(said & set(re.findall('[a-z0-9]+', p.lower()))) / max(1, len(said | set(re.findall('[a-z0-9]+', p.lower())))), p) for p in pending), key=lambda o: -o[0])
        if scores[0][0] >= NEED_SHARE and (len(scores) == 1 or scores[0][0] - scores[1][0] > NEED_LEAD):
            return scores[0][1]
        return None

    def coded_link(self, link: dict[str, Any]) -> dict[str, Any]:
        """#97 item 38: a link the judge names by an id that is not known: the one known id of LINK_ID_CHARS or more inside
        the name; else, for a name of CODE_LINK_CHARS with a digit, the one seen row whose title holds it as a whole word,
        else the one whose record holds it. Several matches leave the name as given."""
        name = str(link.get('id') or '').strip()
        if not name or name in self.rows:
            return link
        inside = [i for i in self.rows if len(i) >= LINK_ID_CHARS and i in name]
        if len(inside) == 1:
            self.trace.append({'kind': 'link_id', 'said': name[:60], 'id': inside[0]})
            return {**link, 'id': inside[0]}
        if not CODE_LINK_CHARS[0] <= len(name) <= CODE_LINK_CHARS[1] or not re.search('\\d', name):
            return link
        pattern = '(?<![a-z0-9])' + re.escape(name.lower()) + '(?![a-z0-9])'
        title_key = self.title_key()
        hits = [i for i, row in self.rows.items() if title_key and re.search(pattern, str(row.get(title_key) or '').lower())]
        if not hits:
            hits = [i for i in self.rows if re.search(pattern, _dump(self.full.get(i) or self.rows.get(i) or {}).lower())]
        if len(hits) != 1:
            return link
        self.trace.append({'kind': 'link_id', 'said': name[:60], 'id': hits[0]})
        return {**link, 'id': hits[0]}

    def add_finding(self, text: str) -> None:
        """#97 items 40/41: a chain finding for the judge (#8), kept for the episode."""
        if text not in self.chain_findings:
            self.chain_findings.append(text)
            del self.chain_findings[:-FINDINGS_KEPT]

    def reject_links(self, need: str, links: list[dict[str, Any]], ids: list[str]) -> tuple[list[dict[str, Any]], list[str], bool]:
        """#97 item 40: the first read link whose page lacks a value the judge gave for it (only words its own description
        source holds: the requirement for the first link, the previous page for later ones; no shop-name words, no short
        words unless they start with a digit, no unit right after a number) is rejected: a finding is kept and the link
        becomes an empty one searched by the judge's find words, else the shop the description names and those values. A
        link read only by its attributes is rejected only for a missing word that is an attribute name of read records."""
        names = None
        for at, (link, ident) in enumerate(zip(links, ids)):
            if not ident or ident not in self.rows:
                break
            opened = self.page_read(ident)
            if not opened and ident not in self.full:
                continue
            given = link.get('values')
            text = ' '.join((str(v) for v in given)) if isinstance(given, list) else str(given or '')
            tokens = re.findall('[a-z0-9]+', text.lower())
            if not tokens:
                continue
            source = need if at == 0 else _dump(self.full.get(ids[at - 1]) or self.rows.get(ids[at - 1]) or {})
            source_words = set(re.findall('[a-z0-9]+', source.lower()))
            shop_words = {w for n in self.name_values(ident) for w in re.findall('[a-z0-9]+', n.lower())}
            kept = []
            for k, word in enumerate(tokens):
                if word not in source_words or word in shop_words or (len(word) < LINK_WORD_CHARS and not word[:1].isdigit()):
                    continue
                if k and re.fullmatch('\\d+(?:\\.\\d+)?', tokens[k - 1]) and not word[:1].isdigit():
                    continue
                kept.append(word)
            record = self.full.get(ident) or self.rows.get(ident) or {}
            block = _fact_block(record) or {}
            if not block:
                continue
            title_key = self.title_key()
            page = _compact(' '.join((f'{k} {v}' for k, v in block.items())) + ' ' + str((self.rows.get(ident) or {}).get(title_key) or '' if title_key else ''))
            missing = [w for w in dict.fromkeys(kept) if w not in page]
            if not missing:
                continue
            if not opened:
                names = self.attr_name_words() if names is None else names
                if not any((w in names for w in missing)):
                    continue
            about = str(link.get('about') or '') or 'the listing it points at'
            self.add_finding(f"{ident} is not the listing '{about[:120]}' that '{need[:120]}' points at: its attributes and title lack {', '.join(missing[:FINDING_WORDS])}")
            find = str(link.get('find') or '').strip()
            if not find:
                shops = [n for n in self.shop_values() if re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', (about + ' ' + need).lower())]
                find = ' '.join(([max(shops, key=len)] if shops else []) + list(dict.fromkeys(kept)))
            self.trace.append({'kind': 'link_rejected', 'id': ident, 'missing': missing[:FINDING_WORDS]})
            return (links[:at] + [{'id': '', 'about': about, 'find': find}], ids[:at] + [''], True)
        return (links, ids, False)

    @_soft(False)
    def defers(self, need: str, page: str, value: str) -> bool:
        """#97 item 41: whether a chain's last page passes the property on: a long text field (over LONG_TEXT characters, not
        the title) names the requirement's property in its words but does not hold the value (letters and digits only). Once
        per requirement and page: when the judge names that page again, its value is taken."""
        if (need, page) in self.deferred or ':' not in need:
            return False
        prop = set(_key_words(need.partition(':')[0]))
        want = _compact(value)
        title_key = self.title_key()
        title = str((self.rows.get(page) or {}).get(title_key) or '') if title_key else ''
        texts: list[str] = []
        stack: list[Any] = [self.full.get(page) or self.rows.get(page) or {}]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                stack += list(node.values())
            elif isinstance(node, list):
                stack += node
            elif isinstance(node, str) and len(node) > LONG_TEXT and node != title:
                texts.append(node)
        for text in texts:
            words = set(_key_words(text))
            if prop and prop <= words and want and want not in _compact(text):
                self.deferred.add((need, page))
                self.add_finding(f'{page} passes {need.partition(":")[0].strip()} on to another listing: its text names the property without the value {value[:60]!r}; follow the listing it refers to')
                self.trace.append({'kind': 'page_defers', 'need': need[:80], 'page': page})
                return True
        return False

    def chain_names(self) -> set[str]:
        """#97 item 42: the shop names a live pointer chain reaches: name-like row texts written in the open pointer
        requirements, widened up to CHAIN_WIDEN times by those written in the long texts of viewed pages of those shops."""
        pending = self.pending_refs()
        if not pending:
            return set()
        counts = self.shop_values()
        said = ' '.join(pending)
        found = {v for v in counts if re.search('(?<![A-Za-z0-9])' + re.escape(v) + '(?![A-Za-z0-9])', said)}
        for _ in range(CHAIN_WIDEN):
            prose = [t for ident in self.opened if self.page_read(ident) and set(self.name_values(ident)) & found for t in _leaves(self.full.get(ident) or self.rows.get(ident) or {}) if len(t.split()) >= CHAIN_PROSE_WORDS]
            more = {v for v in counts if v not in found and any((re.search('(?<![A-Za-z0-9])' + re.escape(v) + '(?![A-Za-z0-9])', t) for t in prose))}
            if not more:
                break
            found |= more
        return found

    @_soft(None)
    def chain_page_view(self, name: str, args: dict[str, Any], allowed: set[str]) -> tuple[str, dict[str, Any], str] | None:
        """#97 item 42: an attribute read of one row is sent as its page view when the store has a separate page-view tool,
        the row is known, its page not viewed and it is not gone, its store label is no requested item's, a pointer
        requirement is still open and the row's shop is one its chain reaches."""
        page = self.roles.get('offer')
        many = None
        if USE_T_PAGE_TOOL_ANY and (not page or page == name):
            page = next((t for t in sorted(self.page_tools()) if t != name and t in allowed), None)
        if not page or page == name or page not in allowed or name != self.roles.get('open'):
            return None
        target, there = (self.single_target(name), self.single_target(page))
        if USE_T_PAGE_TOOL_ANY and not there and page == self.roles.get('compare'):
            many = self.compare_target()
        ident = str(args.get(target) or '').strip() if target else ''
        if not (there or many) or not ident or ident not in self.rows or ident in self.gone or page in self.views.get(ident, set()):
            return None
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
        if label and any((self.owns(item, label) for item in self.items)):
            return None
        names = self.chain_names()
        if not names or not set(self.name_values(ident)) & names:
            return None
        faults: list[str] = []
        sent = _shape({there: ident} if there else {many: [ident]}, self.schemas.get(page), 'args', faults)
        if faults:
            return None
        self.trace.append({'kind': 'chain_page_view', 'id': ident})
        note = f'\n[Cairn] Shown with {page}, the page view, instead of {name}: {ident} is a listing of a shop a requirement points at, and only its page says where it points and counts for the order. Call {name} again if you still need its attributes.'
        return (page, sent, note)

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if USE_E35_CHAIN_OFFER_VIEWS:
            stale += [i for i in self.chain_page_ids if i not in stale and i not in self.gone and self.offer_unseen(i)]
        if stale and self.left() < len(stale) + 5:
            never = [i for i in stale if self.offer_unseen(i)] if USE_E34_OFFER_VIEW_UNSEEN_LINES else []
            if not never or self.left() < len(never) + self.tail() + 1:
                return ''
            self.trace.append({'kind': 'offer_unseen', 'ids': never, 'left': self.left()})
            stale = never
        if not stale:
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

    @_soft('')
    def late_page_show(self) -> str:
        """#97 item 3: at a checkout with one step beyond the ending reserve (too few for the pre-checkout reads), after an
        earlier checkout, when exactly one cart line's offer page was never shown: Cairn shows it first (one step), since an
        ordered line counts only after its page was shown; the checkout then goes ahead."""
        if self.buys < 1 or self.left() != self.tail() + 1 or self.left() > 2 or self.rushed() or self.finished:
            return ''
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ''
        unseen = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and i not in self.gone and self.offer_unseen(i)]
        if len(unseen) != 1:
            return ''
        faults: list[str] = []
        args = _shape({target: unseen[0]}, self.schemas.get(tool), 'args', faults)
        reply = self.act(tool, args) if not faults else None
        if reply is None:
            return ''
        self.offer_seen[unseen[0]] = self.step
        self.observe(tool, reply.get('observation'))
        self.trace.append({'kind': 'late_page_show', 'id': unseen[0]})
        if self.offer_unseen(unseen[0]):
            return ''
        return f"\n[Cairn] Cairn showed {unseen[0]}'s page first (one step): an ordered line counts only after its page was shown."

    @_soft('')
    def ignored_cap_repeat(self, name: str, args: dict[str, Any]) -> str:
        """#97 item 32: a search with one price-cap argument, in a mode (its fixed-choice arguments) already seen to ignore that
        cap, that an earlier search of the tool matches but for the cap value: answered with the earlier rows, no step used."""
        props = (self.schemas.get(name) or {}).get('properties') or {}
        limits = self.cap_keys(name)
        caps = [k for k, v in args.items() if k in limits and isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(caps) != 1:
            return ''
        cap = caps[0]
        mode = tuple(sorted(((k, v) for k, v in args.items() if isinstance(props.get(k), dict) and isinstance(props[k].get('enum'), list) and isinstance(v, (str, int, float)))))
        if self.cap_modes(name).get((cap, mode)) is not False:
            return ''
        want = self.call_mark(name, {k: v for k, v in args.items() if k != cap})
        for mark, (tool, sent) in reversed(list(self.read_args.items())):
            if tool != name or cap not in sent or sent.get(cap) == args[cap] or self.call_mark(name, {k: v for k, v in sent.items() if k != cap}) != want:
                continue
            seen = self.read_calls.get(mark)
            if not self.records_in(seen):
                continue
            self.trace.append({'kind': 'ignored_cap_repeat', 'cap': cap, 'sent': args[cap], 'earlier': sent.get(cap)})
            return f'Not run (no step used): in this search mode the store ignores {cap} (an earlier search in it showed rows priced above its {cap}), so this call would return the same rows as the one with {cap}={_dump(sent.get(cap))}. Those rows:\n' + self.show(seen)[:3000]
        return ''

    @_soft('')
    def dear_line_page(self, before: dict[str, int]) -> str:
        """#97 item 21: on cheapest picks while the clock does not bind: the first line this cart call added with a known price,
        once per item. The newest search showing it with an unread next page: when at least DEAR_SHARE of its DEAR_ROWS or more
        buyable priced rows cost less, Cairn reads that next page (one step) and the attributes of up to DEAR_READS cheaper
        unread rows of the line's label on it, cheapest first."""
        cat_key = self.category_key()
        if self.brief.get('pick') != 'cheapest' or not cat_key or not self.price_key or self.clock_bound() or self.left() - self.tail() < DEAR_ROOM:
            return ''
        added = [i for i, n in self.cart_lines.items() if n > before.get(i, 0) and self.price_of(i) is not None]
        line = next((i for i in added if self.line_owner(i) and self.line_owner(i) not in self.dear_checked), None)
        if not line:
            return ''
        self.dear_checked.add(self.line_owner(line))
        price = self.price_of(line) or 0.0
        label = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
        searches = self.code_tools()
        for mark, (name, args) in reversed(list(self.read_args.items())):
            seen = self.read_calls.get(mark)
            ids = [i for i, _ in self.records_in(seen)] if name in searches else []
            if line not in ids:
                continue
            page = self.paging(name, seen)
            faults: list[str] = []
            call = _shape({**args, page[2]: page[3]}, self.schemas.get(name), 'args', faults) if page and page[3] is not None else None
            if call is None or faults or self.call_mark(name, call) in self.read_calls:
                continue
            priced = [i for i in ids if self.buyable(i) and self.price_of(i) is not None]
            cheaper = [i for i in priced if (self.price_of(i) or 0.0) < price]
            if len(priced) < DEAR_ROWS or len(cheaper) < DEAR_SHARE * len(priced):
                return ''
            reply = self.act(name, call)
            if reply is None:
                return ''
            after = reply.get('observation')
            self.observe(name, after)
            shown_text = self.show(after)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            picks = [i for i, _ in self.records_in(after) if i not in self.full and self.buyable(i) and self.price_of(i) is not None and (self.price_of(i) or 0.0) < price and str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == label]
            picks = sorted(picks, key=lambda i: (self.price_of(i) or 0.0, i))[:DEAR_READS]
            read = self.fill_facts(picks, self.tail() + DEAR_KEEP) if picks else []
            proof = self.evidence([self.full[i] for i in picks if i in self.full])
            self.trace.append({'kind': 'dear_line', 'line': line, 'cheaper': len(cheaper), 'of': len(priced), 'read': picks})
            text = f'\n[Cairn] {line} costs {price:g}, more than {len(cheaper)} of the {len(priced)} buyable rows of its search page, and the store lists rows in no price order: Cairn read the next page ({page[2]}={_dump(page[3])}, one step)' + (' and the attributes of its cheaper rows of the same label' if read else '') + ':\n' + shown_text
            return text + ('\n' + '\n'.join(read) if read else '') + ('\n' + proof if proof else '')
        return ''

    @_soft('')
    def big_row_reads(self, observations: list[Any]) -> str:
        """#97 item 30: on clock-bound tasks, after a model search: the attributes of up to DEAR_READS unread, buyable rows of
        the reply and its auto-read pages that belong to an item with no cart line, most requirement words first (then price on
        cheapest picks), read while more than BIG_ROOM steps beyond the tail remain."""
        cat_key = self.category_key()
        if not cat_key or not self.clock_bound() or self.left() - self.tail() < BIG_ROOM or self.finished:
            return ''
        short = set(self.short_items())
        if not short:
            return ''
        cheapest = self.brief.get('pick') == 'cheapest'
        ranked = []
        for ident in dict.fromkeys((i for seen in observations for i, _ in self.records_in(seen))):
            owners = [n for n in self.line_items(ident, cat_key) if n in short]
            if ident in self.full or not owners or not self.buyable(ident):
                continue
            said = set(re.findall('[a-z0-9]{3,}', _words(_dump(self.rows.get(ident) or {}))))
            price = self.price_of(ident)
            ranked.append((-len(self.need_words(owners[0]) & said), (price if price is not None else float('inf')) if cheapest else 0.0, ident))
        picks = [i for _, _, i in sorted(ranked)][:DEAR_READS]
        read = self.fill_facts(picks, self.tail() + BIG_ROOM) if picks else []
        if not read:
            return ''
        self.trace.append({'kind': 'big_row_reads', 'ids': picks})
        return "\n[Cairn read these rows' attributes itself (one step each): they belong to an item with no cart line yet]\n" + '\n'.join(read)

    @_soft('')
    def short_cheap_reads(self, name: str, observations: list[Any]) -> str:
        """#97 item 31: on cheapest picks while the clock does not bind: when Cairn's auto-paging read this search to its last
        page (or its page shape is unknown), the attributes of its DEAR_READS cheapest unread, priced, buyable rows that belong
        to an item with no cart line, read while more than SHORT_READ_KEEP steps beyond the tail remain."""
        cat_key = self.category_key()
        if self.brief.get('pick') != 'cheapest' or not self.price_key or not cat_key or self.clock_bound() or self.left() - self.tail() < SHORT_READ_ROOM or self.finished:
            return ''
        last = self.paged_obs[-1] if self.paged_obs else None
        page = self.paging(name, last) if last is not None else None
        if page and page[3] is not None:
            return ''
        short = set(self.short_items())
        pool = [i for i in dict.fromkeys((i for seen in observations for i, _ in self.records_in(seen))) if i not in self.full and self.buyable(i) and self.price_of(i) is not None and set(self.line_items(i, cat_key)) & short]
        picks = sorted(pool, key=lambda i: (self.price_of(i) or 0.0, i))[:DEAR_READS]
        read = self.fill_facts(picks, self.tail() + SHORT_READ_KEEP) if picks else []
        if not read:
            return ''
        proof = self.evidence([self.full[i] for i in picks if i in self.full])
        self.trace.append({'kind': 'short_cheap_reads', 'ids': picks})
        return "\n[Cairn read the cheapest rows of this search's pages itself (one step each): the pick is the cheapest and these rows were not opened yet]\n" + '\n'.join(read) + ('\n' + proof if proof else '')

    @_soft(None)
    def read_sold_peers(self, lost: list[str]) -> None:
        """#97 item 33: on cheapest picks, for each sold-out cart line: its DEAR_READS cheapest unread, buyable rows of the same
        label, not in the cart, costing no more than it (the replacement is most likely among them), read before replacements
        are chosen, while more than PEER_KEEP steps beyond the tail remain."""
        cat_key = self.category_key()
        if self.brief.get('pick') != 'cheapest' or not cat_key:
            return
        cart = set(self.cart_ids())
        taken: list[str] = []
        for line in lost:
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not label:
                continue
            peers = [i for i in self.seen_order if i != line and i not in self.full and i not in cart and self.buyable(i) and self.price_of(i) is not None and (self.price_of(i) or 0.0) <= price and str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == label]
            peers.sort(key=lambda i: (self.price_of(i) or 0.0, i))
            taken += [i for i in peers[:DEAR_READS] if i not in taken]
        if taken:
            read = self.fill_facts(taken, self.tail() + PEER_KEEP)
            self.trace.append({'kind': 'sold_peers', 'lost': lost, 'read': taken[:len(read)]})

    @_soft(None)
    def list_ties(self, lost: list[str]) -> None:
        """USE_R16_SOLD_TIES: on a lowest-price pick, once per sold-out cart line, the line's store label listed with the
        store's capped listing tool at the line's own price (equal prices kept), while R16_TIE_ROOM steps beyond the tail
        remain: the store keeps same-price twins of a line it sells out, and a search under the line's price leaves them
        out. The sold-out peer reads (#97 item 33) then open the cheapest of the rows it shows."""
        if self.brief.get('pick') != 'cheapest':
            return
        plan = self.cap_template()
        cat_key = self.category_key()
        if not plan or not cat_key:
            return
        tool, base, shelf, cap = plan
        for line in lost:
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            if price is None or price <= 0 or not label or line in self.tie_listed:
                continue
            if self.finished or self.rushed() or self.left() <= self.tail() + R16_TIE_ROOM:
                return
            self.tie_listed.add(line)
            faults: list[str] = []
            ceiling = int(price) if float(price).is_integer() else price
            args = _shape(dict(base, **{shelf: label, cap: ceiling}), self.schemas.get(tool), 'args', faults)
            if faults or (self.call_mark(tool, args) in self.read_calls):
                continue
            reply = self.act(tool, args)
            if reply is None:
                return
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.trace.append({'kind': 'tie_listing', 'gone': line, 'at': price, 'found': len(self.records_in(seen))})

    def dearer_swap(self, best: str, lost: list[str]) -> bool:
        """#97 item 33: whether the best replacement costs more than the sold line (or a price is unknown), so one more search
        under its price is worth a step."""
        price = self.price_of(best)
        sold = [p for p in (self.price_of(i) for i in lost) if p is not None]
        return price is None or not sold or price > max(sold) + 1e-09

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
        if USE_R16_SOLD_TIES:
            self.list_ties(lost)
        if USE_G33_SOLD_PEERS:
            self.read_sold_peers(lost)
        picks = self.replacements(lost)
        text = f"[Cairn] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get('pick') == 'cheapest' and (not USE_G33_SOLD_PEERS or self.dearer_swap(picks[0], lost)):
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
            records.append(self.cut_record(self.full.get(ident) or self.rows.get(ident), 200, 1200))
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = _swap_text() + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
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
            records.append('OPENED ' + self.cut_record(record, 200, 1200) if record else 'ROW ' + self.cut_record(self.rows.get(ident), 120, 500))
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = _audit_text() + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
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
        if USE_G55_BASKET_PLAN:
            basket = self.basket_plan(self.confirmed)
            if not basket:
                return ''
            now = self.cart_ids()
            lines = basket
            drop = [i for i in now if i not in lines]
            calls = self.cart_calls(lines, drop)
        else:
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
        if USE_G2_COMPLETE_REREAD:
            changed = self.reread_offers()
            lost = [i for i in lines if not self.buyable(i)]
            if lost or self.finished or self.left() < 1:
                self.trace.append({'kind': 'completed_reread', 'lost': lost})
                return 'Not run (Cairn acted instead): the independent check confirmed ' + self.confirmed + ', so Cairn set the cart (' + '; '.join(done) + '), but re-reading its lines before the checkout showed ' + (', '.join(lost) + ' can no longer be bought: replace that line, then check out.' if lost else 'no step left for the checkout.') + ('\n' + changed if changed else '')
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

    @_soft(False)
    def cart_short(self) -> bool:
        """#97 item 59: the cart holds fewer lines than items, or an item has no line."""
        held = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        return len(held) < len(self.items) or bool(self.short_items())

    @_soft('')
    def reopen_search(self) -> str:
        """#97 item 59: once an episode, when the model answers without a call while nothing was bought, the search tool
        exists, more than the tail and three steps are left and the cart is short: when the search is not offered now, it is
        reopened (the middle narrowing of the tool list skipped while the cart is short) and the nudge says so."""
        search = self.roles.get('search')
        if self.search_reopened or self.buys or not search or self.left() <= self.tail() + 3 or not self.cart_short():
            return ''
        if search in {t['function']['name'] for t in self.offered()}:
            return ''
        self.search_reopened = True
        self.trace.append({'kind': 'search_reopened', 'left': self.left()})
        return '[Cairn] Search and listing views are open again: find, read and cart a listing for each item that has no cart line yet, then check out in the steps left.\n'

    def offered(self) -> list[dict[str, Any]]:
        left = self.left()
        ask = self.roles.get('ask')
        keep: set[str] | None = None
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        if left <= self.tail():
            keep = set(ends)
        elif left <= self.tail() + 3:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view')}
            if USE_E13_LAST_STEPS_OPEN_SEARCH and left >= 3 and self.short_items():
                keep |= {self.facts_tool() or self.roles.get('open')} | ({self.roles.get('search')} if left >= 4 else set())
        elif left <= len(self.items) + 5 and self.buys == 0:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
            if USE_E12_SHORT_ITEM_KEEPS_TOOLS and self.short_items(self.tail() + 3):
                keep = None
            elif USE_G59_REOPEN_SEARCH and self.search_reopened and self.cart_short():
                keep = None
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

    def required_codes(self) -> list[str]:
        """#111 E21: the fit codes the live requirements state (the base's code reader; pointer requirements left out)."""
        pointers = {need for _, need in self.refs}
        out: list[str] = []
        for _, need in self.needs_now():
            if need in pointers:
                continue
            for code in _codes(need.partition(':')[2] if ':' in need else need):
                if code not in out:
                    out.append(code)
        return out

    def strip_codes(self, text: str, codes: list[str]) -> tuple[str, list[str]]:
        """#111 E21: `text` without the required codes it names, their bare numbers and the word introducing them; the codes taken."""
        spans = []
        for hit in re.finditer('\\b' + FIT_CODE + '\\b', text, re.I):
            said = re.sub('[\\s-]+', ' ', hit.group(0)).strip().lower()
            if any((_code_hit(code, said) or _code_hit(said, code) for code in codes)):
                spans.append(hit)
        if not spans:
            return (text, [])
        out = text
        for hit in reversed(spans):
            before = out[:hit.start()]
            intro = re.search(FIT_INTRO, before, re.I)
            out = (before[:intro.start()] if intro else before) + ' ' + out[hit.end():]
        for hit in spans:
            for digits in re.findall('\\d+', hit.group(0)):
                out = re.sub('(?<![\\w-])' + digits + '(?![\\w-])', ' ', out)
        return (' '.join(out.split()), [hit.group(0) for hit in spans])

    def code_tools(self) -> set[str]:
        plan = self.value_template()
        return {t for t in (self.roles.get('search'), plan[0] if plan else None) if t}

    def drop_codes(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str, bool]:
        """#111 E21: an actor search after a fit-code search found nothing goes without the codes; (args, note, nothing left)."""
        codes = self.required_codes()
        props = (self.schemas.get(name) or {}).get('properties') or {}
        if not codes:
            return (args, '', False)
        out = dict(args)
        taken: list[str] = []
        for key, value in args.items():
            if not isinstance(value, str) or 'enum' in (props.get(key) or {}):
                continue
            text, gone = self.strip_codes(value, codes)
            if not gone:
                continue
            taken += gone
            if not re.search('[A-Za-z0-9]', text):
                self.trace.append({'kind': 'code_search_refused', 'sent': value[:80]})
                return (args, f"Not run (no step used): this store's search does not find fit codes (an earlier search for one showed no rows), and nothing is left of {value!r} without {', '.join(gone)}. Search the item's kind together with its other required values instead; a part fits when its own text names the code.", True)
            out[key] = text
        if not taken:
            return (args, '', False)
        self.trace.append({'kind': 'code_dropped', 'codes': taken, 'args': _clip(out, 80)})
        where = 'the rows naming it are marked below the search result' if USE_E24_FIT_SCAN else 'read the codes on the rows and pages'
        return (out, f"\n[Cairn] This store's search does not find fit codes (an earlier search for one showed no rows), so Cairn sent this search without {', '.join(taken)}: {_dump(out)}. A part fits when its own text names the code: {where}.", False)

    def holds_required_code(self, args: dict[str, Any]) -> bool:
        codes = self.required_codes()
        return bool(codes) and any((self.strip_codes(v, codes)[1] for v in args.values() if isinstance(v, str)))

    def wanted_units(self, ident: str) -> int:
        """#111 E22: the quantity of the item a listing belongs to (by its label; the only item on one-item tasks), else 1."""
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
        number = self.owner_of(label) if label else 0
        if not number and len(self.items) == 1:
            number = 1
        return self.items[number - 1].qty if 1 <= number <= len(self.items) else 1

    def zero_count_add(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """#111 E22: a cart call that adds a listing not in the cart with a count of 0 is sent with its item's quantity."""
        props = (self.schemas.get(name) or {}).get('properties') or {}
        if any((isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list) and re.search(REMOVE_MODE, v, re.I) for k, v in args.items())):
            return (args, '')
        held = set(self.cart_lines) | set(self.cart_ids())

        def fix(line: dict[str, Any], spec: dict[str, Any]) -> tuple[dict[str, Any], str]:
            zero = [k for k, v in line.items() if isinstance(spec.get(k), dict) and 'integer' in _kinds(spec[k]) and not isinstance(v, (str, bool)) and _count(v) == 0]
            ids = [v.strip() for v in line.values() if isinstance(v, str) and v.strip() in self.rows]
            if len(zero) != 1 or len(ids) != 1 or ids[0] in held:
                return (line, '')
            want = self.wanted_units(ids[0])
            top = spec[zero[0]].get('maximum')
            if isinstance(top, (int, float)) and not isinstance(top, bool):
                want = min(want, int(top))
            return ({**line, zero[0]: want}, f'{zero[0]}={want} for {ids[0]}') if want > 0 else (line, '')
        out, said = fix(args, props)
        notes = [said] if said else []
        for key, value in args.items():
            inner = ((props.get(key) or {}).get('items') or {}).get('properties') if isinstance(props.get(key), dict) else None
            if isinstance(value, list) and isinstance(inner, dict):
                lines = []
                for line in value:
                    fixed, told = fix(line, inner) if isinstance(line, dict) else (line, '')
                    lines.append(fixed)
                    notes += [told] if told else []
                out = {**out, key: lines}
        if not notes:
            return (args, '')
        self.trace.append({'kind': 'zero_count_add', 'fixed': notes})
        return (out, '\n[Cairn] A count of 0 adds nothing: the count is the number of units to add, so Cairn sent ' + '; '.join(notes) + '.')

    def budget_broken(self, lines: dict[str, int]) -> bool:
        """#111 E23: whether the cart's lines already break the budget (unknown prices: no)."""
        if not isinstance(self.budget, (int, float)) or isinstance(self.budget, bool):
            return False
        prices = {i: self.price_of(i) for i in lines}
        if not prices or any((p is None for p in prices.values())):
            return False
        if self.brief.get('budget_scope') == 'each':
            return any((p > self.budget + 1e-09 for p in prices.values()))
        return sum((prices[i] * (lines[i] or 1) for i in lines)) > self.budget + 1e-09

    def keeps_lines(self, name: str, args: dict[str, Any], later: list[dict[str, Any]], start: dict[str, int] | None=None) -> str:
        """#111 E23: with LATE_GUARD_STEPS or fewer steps left, the refusal of a cart call that, with the turn's later cart
        calls, would leave fewer lines than the cart held when the turn began (`start`), unless the lines taken out are sold
        out or the cart breaks the budget; '' when the call may run."""
        start = dict(self.cart_lines) if start is None else start
        if self.left() > LATE_GUARD_STEPS or not start:
            return ''
        original = self.cart_lines
        saved = (self.cart_touched, self.cart_at)
        held_before = set(self.ever_held)
        self.cart_lines = dict(original)
        try:
            self.note_cart(name, args)
            for call in later:
                fn = call.get('function') or {}
                if fn.get('name') != name:
                    continue
                raw = fn.get('arguments')
                more = raw if isinstance(raw, dict) else _json_obj(raw) if isinstance(raw, str) and raw.strip() else {}
                faults: list[str] = []
                more = _shape(more, self.schemas.get(name), 'args', faults) if isinstance(more, dict) else None
                if more is None or faults:
                    continue
                more, wrong = self.resolve_ids(name, more)
                if not wrong:
                    self.note_cart(name, self.whole_removal(name, more))
            after = dict(self.cart_lines)
        finally:
            self.cart_lines = original
            self.cart_touched, self.cart_at = saved
            self.ever_held = held_before
        if len(after) >= len(start):
            return ''
        taken = [i for i in start if i not in after and not self.sold_out(i)]
        if not taken or self.budget_broken(start):
            return ''
        self.trace.append({'kind': 'late_lines_kept', 'taken': taken, 'left': self.left()})
        return f"Not run (no step used): only {self.left()} steps are left, and this would take {', '.join(taken)} out of the cart, leaving fewer lines than it holds. Keep {', '.join(start)} in the cart and check out now."

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
            self.ever_held.update(self.cart_lines)
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
                self.ever_held.add(ident)
            else:
                self.cart_lines.pop(ident, None)

    def show(self, observation: Any) -> str:
        if USE_R104_3_COMPACT_ROWS and len(self.records_in(observation)) >= 2:
            observation = _drop_empty(observation)  # #104 item 3: rows without their empty fields
            if USE_T_R104_3_CUTS:
                title = self.title_key() or 'title'
                for _, row in self.records_in(observation):
                    for key, value in list(row.items()):
                        limit = ROW_TITLE_CHARS if key == title else ROW_TEXT_CHARS
                        if isinstance(value, str) and len(value) > limit + 3:
                            row[key] = value[:limit].rstrip() + '...'
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

    def line_items(self, ident: str, cat_key: Any=False) -> list[int]:
        """#111: the items a listing belongs to by its store label; the only item on one-item tasks. Loops pass the
        category key they read once (`cat_key`)."""
        if len(self.items) == 1:
            return [1]
        cat_key = self.category_key() if cat_key is False else cat_key
        label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
        return [n for n, item in enumerate(self.items, 1) if label and self.owns(item, label)]

    def line_owner(self, ident: str) -> int:
        owners = self.line_items(ident)
        return owners[0] if len(owners) == 1 else 0

    def item_needs(self, number: int) -> list[str]:
        """#111: the item's live requirements, conditional rules (E4) left out: those are read from the cart."""
        return [need for owner, need in self.needs_now() if owner == number and not _conditional(need)]

    def item_codes(self, number: int) -> list[str]:
        """#111 E24: the fit codes the item's live requirements state (the king's code reader; pointer and conditional
        requirements left out)."""
        pointers = {need for _, need in self.refs}
        out: list[str] = []
        for need in self.item_needs(number):
            if need in pointers:
                continue
            for code in _codes(need.partition(':')[2] if ':' in need else need):
                if code not in out:
                    out.append(code)
        return out

    def codes_of(self, ident: str) -> list[str]:
        """#111 E24: the code-like words of a listing's text (kept until its record grows or is opened)."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        mark = (ident in self.full, len(record), len(_dump(record)))
        held = self.code_memo.get(ident)
        if held is None or held[0] != mark:
            held = (mark, _text_codes(' '.join(_leaves(record))))
            self.code_memo[ident] = held
        return held[1]

    def names_codes(self, ident: str, codes: list[str]) -> bool:
        """#111 E24/E28: whether a listing's own text names every code (series wildcards either way with E28)."""
        said = self.codes_of(ident)
        return bool(codes) and all((any((_code_match(s, code) for s in said)) for code in codes))

    def fitting_rows(self, number: int) -> list[str]:
        """#111 E42: the buyable rows seen of the item whose text names every required code, cheapest first."""
        codes = self.item_codes(number)
        cat_key = self.category_key()
        found = [i for i in self.seen_order if number in self.line_items(i, cat_key) and self.buyable(i) and self.names_codes(i, codes)] if codes else []
        return sorted(found, key=lambda i: (self.price_of(i) is None, self.price_of(i) or 0.0, i))

    def mark_fits(self, numbers: list[int], ids: list[str]) -> None:
        cat_key = self.category_key()
        for number in numbers:
            codes = self.item_codes(number)
            marked = self.fit_rows.setdefault(number, [])
            marked += [i for i in ids if i not in marked and number in self.line_items(i, cat_key) and self.buyable(i) and self.names_codes(i, codes)]

    def fit_verdict(self, ident: str, number: int) -> tuple[int, str]:
        """#111 E25: a fitting row against the item's other requirements as written (code requirements, conditional rules
        and unresolved pointers left out): 0 opened and meets them, 1 not opened, 2 opened with a required property missing
        from its page, 3 opened with another value for a property."""
        if ident not in self.full:
            return (1, 'not opened')
        missing: list[str] = []
        other: list[str] = []
        for need in self.item_needs(number):
            prop, _, value = need.partition(':')
            if _codes(value):
                continue
            state, shown = self.literal_state(ident, number, need)
            if state == 'differs':
                other.append(f'{shown or prop.strip()} (wanted {value.strip()})')
            elif state == 'missing':
                missing.append(prop.strip())
        if other:
            return (3, 'opened, states ' + '; '.join(other[:2]))
        if missing:
            return (2, 'opened, its page does not state ' + ', '.join(missing[:3]))
        return (0, 'opened, meets the other requirements as written')

    def live_fits(self, number: int) -> int:
        """#111 E24: the item's marked fitting rows still worth carting: buyable, and unopened or opened and meeting the rest."""
        return sum((1 for i in self.fit_rows.get(number, []) if self.buyable(i) and (i not in self.full or self.fit_verdict(i, number)[0] == 0)))

    def scan_pages(self, tool: str, args: dict[str, Any], seen: Any, numbers: list[int], ids: list[str], enough: Any) -> int:
        """#111 E24/E26: further pages of one search while it has one, up to FIT_SCAN_PAGES (FIT_SCAN_TOTAL an episode), with
        FIT_SCAN_ROOM steps beyond the tail kept and the clock not rushed, until `enough()`; rows land in `ids`. Steps taken."""
        pages = 0
        walked = {self.call_mark(tool, args)}
        for _ in range(4 * FIT_SCAN_PAGES):
            if pages >= FIT_SCAN_PAGES or self.fit_scan_pages >= FIT_SCAN_TOTAL or self.left() <= self.tail() + FIT_SCAN_ROOM or self.rushed() or self.finished or enough():
                break
            page = self.paging(tool, seen)
            if not page or page[3] is None:
                break
            faults: list[str] = []
            call = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
            mark = self.call_mark(tool, call) if not faults else ''
            if faults or mark in walked:
                break
            walked.add(mark)
            if mark in self.read_calls:
                seen = self.read_calls[mark]
            else:
                reply = self.act(tool, call)
                if reply is None:
                    break
                seen = reply.get('observation')
                self.observe(tool, seen)
                pages += 1
                self.fit_scan_pages += 1
            new = [i for i, _ in self.records_in(seen) if i not in ids]
            ids += new
            self.mark_fits(numbers, new)
        return pages

    def open_fits(self, picks: list[str]) -> int:
        """#111 E25/E26: open fitting rows (the king's reader), keeping FIT_SCAN_ROOM steps beyond the tail; how many opened."""
        if not picks:
            return 0
        self.fill_facts(picks, self.tail() + FIT_SCAN_ROOM)
        return sum((1 for i in picks if i in self.full))

    def cheapest_unopened(self, number: int, ids: list[str]) -> list[str]:
        fresh = [i for i in ids if i not in self.full and self.buyable(i) and not (USE_E7_ITEM_PRICE_CAPS and self.over_cap(number, i))]
        return sorted(fresh, key=lambda i: (self.price_of(i) is None, self.price_of(i) or 0.0, i))[:FIT_OPEN_EACH]

    def sweep_shops(self, number: int) -> list[str]:
        """#111 E26: shops to sweep for an item: short name-like field values of its rows whose text carries a code of the
        required code's family, most frequent first, none swept before for it."""
        prefixes = {_code_prefix(c) for c in self.item_codes(number)} - {''}
        if not prefixes:
            return []
        not_names = set(self.not_maker_keys()) | {self.price_key, self.stock_key}
        cat_key = self.category_key()
        counts: dict[str, int] = {}
        for ident in self.seen_order:
            if number not in self.line_items(ident, cat_key) or not any((_code_prefix(s) in prefixes for s in self.codes_of(ident))):
                continue
            for key, value in (self.rows.get(ident) or {}).items():
                text = value.strip() if isinstance(value, str) and key not in not_names else ''
                if 2 < len(text) <= 40 and text[:1].isupper() and not re.search('\\d', text) and len(text.split()) <= 4 and text.lower() not in NO_NAME:
                    counts[text] = counts.get(text, 0) + 1
        tried = self.swept.get(number, set())
        return sorted((s for s in counts if s not in tried), key=lambda s: (-counts[s], s))[:SWEEP_SHOPS]

    def shop_sweep(self, number: int, name: str, args: dict[str, Any]) -> tuple[list[str], int, int]:
        """#111 E26: for an item with no live fitting row, search each sweep shop with the kind words of the actor's query
        (fit codes and words holding a digit left out), paged like the scan, until a new unopened fitting row shows; then open
        the cheapest new ones. (queries, pages, opened)."""
        plan = self.value_template()
        if not plan:
            return ([], 0, 0)
        tool, base, field = plan
        said = args.get(field) if name == tool and isinstance(args.get(field), str) else max((v for v in args.values() if isinstance(v, str)), key=len, default='')
        text, _ = self.strip_codes(said, self.item_codes(number))
        kind = ' '.join([w for w in text.split() if not re.search('\\d', w)][:SWEEP_WORDS])
        if not kind:
            return ([], 0, 0)
        before = set(self.fit_rows.get(number, []))
        queries: list[str] = []
        pages = 0
        new = lambda: [i for i in self.fit_rows.get(number, []) if i not in before and i not in self.full and self.buyable(i)]
        for shop in self.sweep_shops(number):
            if new() or self.left() <= self.tail() + FIT_SCAN_ROOM or self.clock() >= FIT_SCAN_BY or self.rushed() or self.finished or self.fit_scan_pages >= FIT_SCAN_TOTAL:
                break
            self.swept.setdefault(number, set()).add(shop)
            query = f'{shop} {kind}'
            faults: list[str] = []
            call = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.act(tool, call)
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            queries.append(query)
            self.fit_scan_pages += 1
            ids = [i for i, _ in self.records_in(seen)]
            self.mark_fits([number], ids)
            pages += self.scan_pages(tool, call, seen, [number], ids, lambda: bool(new()))
        opened = self.open_fits(self.cheapest_unopened(number, new())) if new() else 0
        self.trace.append({'kind': 'shop_sweep', 'item': number, 'queries': queries, 'pages': pages, 'opened': opened})
        return (queries, pages, opened)

    @_soft('')
    def fit_scan(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """#111 E24-E26: after an actor search, on tasks whose requirements state fit codes: mark the rows naming every code of
        their item, read up to FIT_SCAN_PAGES more pages until each such item has FIT_SCAN_ENOUGH live fitting rows (E24),
        open the cheapest unopened fitting rows (E25), sweep shops of the code's family for an item still without one (E26),
        and append per item the fitting rows with a one-line verdict each."""
        numbers = [n for n in range(1, len(self.items) + 1) if self.item_codes(n)]
        if not numbers:
            return ''
        ids = list(dict.fromkeys((i for seen in [observation] + list(self.paged_obs) for i, _ in self.records_in(seen))))
        cat_key = self.category_key()
        active = [n for n in numbers if any((n in self.line_items(i, cat_key) for i in ids))]
        if not active:
            return ''
        self.mark_fits(active, ids)
        last = self.paged_obs[-1] if self.paged_obs else observation
        pages = self.scan_pages(name, args, last, active, ids, lambda: all((self.live_fits(n) >= FIT_SCAN_ENOUGH for n in active)))
        opened = 0
        if USE_E25_FIT_SCAN_OPEN and self.left() > self.tail() + FIT_OPEN_ROOM and self.clock() < FIT_SCAN_BY and not self.rushed() and not self.finished:
            picks: list[str] = []
            for number in active:
                picks += [i for i in self.cheapest_unopened(number, self.fit_rows.get(number, [])) if i not in picks]
            opened = self.open_fits(picks[:FIT_OPEN_SCAN])
        swept: list[str] = []
        if USE_E26_FIT_SHOP_SWEEP:
            for number in active:
                if not self.live_fits(number) and self.clock() < FIT_SCAN_BY and not self.finished:
                    queries, more, found = self.shop_sweep(number, name, args)
                    swept += queries
                    pages += more
                    opened += found
        many = len(self.items) > 1
        lines = []
        for number in active:
            codes = ', '.join(self.item_codes(number))
            head = f'item {number} ({self.items[number - 1].name})' if many else self.items[number - 1].name
            kinds = sum((1 for i in ids if number in self.line_items(i, cat_key)))
            fits = [i for i in self.fit_rows.get(number, []) if self.buyable(i)]
            price = lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf')
            fits.sort(key=lambda i: ((self.fit_verdict(i, number)[0] if USE_E25_FIT_SCAN_OPEN else 0), price(i), i))
            if fits:
                shown = []
                for ident in fits[:FIT_SCAN_SHOWN]:
                    bit = ident + (f' at {self.price_of(ident):g}' if self.price_of(ident) is not None else '')
                    shown.append(bit + (': ' + self.fit_verdict(ident, number)[1] if USE_E25_FIT_SCAN_OPEN else ''))
                lines.append(f'{head}: rows naming {codes} -> ' + '; '.join(shown) + f' ({kinds} rows of that kind in this search)')
            else:
                lines.append(f"{head}: none of the {kinds} rows of that kind read here names {codes}, and this store's search does not find codes, so search the kind with another required value")
        text = '\n[Cairn fit scan: a part fits only when its own text names the code' + (f'; Cairn read {pages} more page(s), one step each' if pages else '') + '] ' + ' | '.join(lines)
        if opened:
            text += f'\n[Cairn] Cairn opened {opened} fitting row(s) (one step each): cart a row that meets the requirements instead of opening it again.'
        if swept:
            text += '\n[Cairn] Cairn also searched shops that make parts of this code family: ' + '; '.join((repr(q) for q in swept)) + '.'
        self.trace.append({'kind': 'fit_scan', 'items': active, 'pages': pages, 'opened': opened, 'swept': swept, 'fits': {str(n): len(self.fit_rows.get(n, [])) for n in active}})
        return text

    @_soft(list)
    def wider_link(self, about: str, words: str, values: list[str], memo: str) -> list[str]:
        """#111 E29: for the shop a chain's description names most often, also search the shop with each of its LINK_VALUES
        strongest described values and the shop alone, collecting its rows across queries until one shows every value; the
        unopened rows of that shop showing the most described values, best two."""
        plan = self.value_template()
        counts = self.shop_values()
        said = f'{about} {words}'.lower()
        named = [v for v in counts if re.search('(?<![a-z0-9])' + re.escape(v.lower()) + '(?![a-z0-9])', said)]
        if not plan or not named:
            return []
        shop = max(named, key=lambda v: (counts[v], len(v), v))
        own = set(re.findall('[a-z0-9]+', shop.lower()))
        wanted = [v for v in values if v not in own]
        not_names = set(self.not_maker_keys())

        def of_shop(ids: Any) -> list[str]:
            return [i for i in dict.fromkeys(ids) if any((isinstance(v, str) and v.strip().lower() == shop.lower() for k, v in (self.rows.get(i) or {}).items() if k not in not_names))]

        def shows(ident: str) -> int:
            text = ' '.join(_leaves(self.full.get(ident) or self.rows.get(ident) or {})).lower()
            return sum((1 for v in wanted if re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text)))
        collected = of_shop(self.link_seen.get(memo, []))
        tool, base, field = plan
        strongest = sorted(wanted, key=lambda v: (not re.search('\\d', v), -len(v), v))[:LINK_VALUES]
        for query in [f'{shop} {v}' for v in strongest] + [shop]:
            if any((shows(i) == len(wanted) for i in collected)) or self.link_searches >= LINK_CAP_WIDE or self.left() <= 12 or self.finished or self.rushed():
                break
            if query.lower() in self.link_queries:
                continue
            self.link_queries.add(query.lower())
            faults: list[str] = []
            args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            self.link_searches += 1
            reply = self.act(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            collected += [i for i in of_shop([i for i, _ in self.records_in(seen)]) if i not in collected]
            self.trace.append({'kind': 'link_search_wide', 'words': query[:80], 'rows': len(collected)})
        pool = [i for i in of_shop(collected + list(reversed(self.seen_order))) if i not in self.opened]
        ranked = sorted(pool, key=lambda i: -shows(i))
        return [i for i in ranked if shows(i) > 0 or not wanted][:2]

    def offer_unseen(self, ident: str) -> bool:
        """#111 E34/E35: whether a listing was never shown by the offer view."""
        tool = self.offer_tool()
        return ident not in self.offer_seen and (not tool or tool not in self.views.get(ident, set()))

    @_soft(None)
    def offer_views_before_buy(self) -> None:
        """#111 E35: before the forced buy, the offer view of every cart line and resolved chain page it never showed, while
        more than the tail of steps is left and the clock is not rushed."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        ids = [i for i in dict.fromkeys(self.cart_ids() + self.chain_page_ids) if i not in self.gone and self.offer_unseen(i)]
        viewed = []
        for ident in ids:
            if self.left() <= self.tail() or self.rushed() or self.finished:
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            self.offer_seen[ident] = self.step
            self.observe(tool, reply.get('observation'))
            viewed.append(ident)
        if viewed:
            self.trace.append({'kind': 'offer_views_before_buy', 'ids': viewed})

    @_soft(False)
    def breaks_limit(self, ident: str, number: int) -> bool:
        """#111 E45: whether a row's own text gives a figure in a requirement's unit that breaks that requirement's at-least or
        at-most limit (requirements of item `number`, the item of the cart line the row is cheaper than)."""
        if not number:
            return False
        pointers = {need for _, need in self.refs}
        text = ' '.join(_leaves(self.rows.get(ident) or {}))
        for need in self.item_needs(number):
            limit = _need_limit(need.partition(':')[2]) if need not in pointers else None
            if limit and limit['kind'] in ('min', 'max', 'range') and _breaks_limit(limit, _unit_figures(text, limit['unit'], bare=False)[0]):
                return True
        return False

    def literal_state(self, ident: str, number: int, need: str) -> tuple[str, str]:
        """#111 E49/E50: one requirement against a listing's own attributes, read as written: 'met', 'missing' (no key named
        for the property), 'differs' (such a key says something else) or 'unknown' (a negation, an unresolved pointer, one
        worded as a difference); with the attribute shown. Figures compare literally in the same unit; words all appear in
        the value of a key named for the property; a code requirement is met when the listing's text names the code."""
        prop, colon, value = need.partition(':')
        if not colon or not prop.strip() or not value.strip():
            return ('unknown', '')
        if need in {m for _, m in self.refs}:
            got = self.ref_values.get(need, '')
            if not got or got.startswith('as stated on') or re.search(DIFFERENCE, value, re.I):
                return ('unknown', '')
            if USE_G16_POINTER_BOUND_VALUE and re.search(OPEN_NOT, got, re.I):
                return ('unknown', '')
            value = got
        codes = _codes(value)
        if codes:
            hit = self.names_codes(ident, codes)
            barred = self.barred_codes(ident, codes) if USE_G15_BARRED_CODES and hit else []
            if barred:
                return ('differs', 'its fit clause excludes ' + ', '.join(barred))
            return ('met' if hit else 'differs', ('its text names ' if hit else 'its text does not name ') + ', '.join(codes))
        limit = _need_limit(value)
        if re.search(NEGATION, value, re.I) and (limit is None or limit['kind'] == 'exact'):
            return ('unknown', '')
        block = _fact_block(self.full.get(ident) or {}) or {}
        keys = [k for k in block if _key_named(k, prop)]
        if not keys and not limit and USE_G13_WORD_IN_TITLE:
            return self.word_elsewhere(ident, value, block)
        if not keys:
            return ('missing', '')
        if not limit and (USE_G14_WORD_FORMS or USE_G17_KEY_PROOF):
            return self.word_by_keys(ident, number, prop, value, block, keys)
        if limit:
            shown = ''
            for key in keys:
                figs, ranged = _unit_figures(block[key], limit['unit'])
                if not figs:
                    continue
                shown = shown or f'{key}={block[key]}'
                if _meets_limit(limit, figs, ranged):
                    return ('met', f'{key}={block[key]}')
            return ('differs', shown or f'{keys[0]}={block[keys[0]]}')
        options = [o for o in re.split('\\s+or\\s+', value.strip()) if o.strip()] or [value]
        for key in keys:
            have = _lit_words(block[key])
            if any((_lit_words(o) and _lit_words(o) <= have for o in options)):
                return ('met', f'{key}={block[key]}')
        return ('differs', f'{keys[0]}={block[keys[0]]}')

    def barred_codes(self, ident: str, codes: list[str]) -> list[str]:
        """#97 item 15: the required codes the listing's fit clause excludes (the king's fit index: "fits ... except X")."""
        bars = (self.fit_index()['marks'].get(ident) or {}).get('bars') or []
        return [c for c in codes if any((_code_hit(b, c) or _code_hit(c, b) for b in bars))]

    def word_elsewhere(self, ident: str, value: str, block: dict[str, Any]) -> tuple[str, str]:
        """#97 items 13/14: a word requirement no key names: stated when one choice's words (forms of one stem with G14) all
        sit in the title or all in the joined attribute values, or when the initials of three or more of its words in a row
        are one word there; never from the description alone."""
        title_key = self.title_key()
        title = str((self.rows.get(ident) or {}).get(title_key) or '') if title_key else ''
        joined = ' '.join((str(v) for v in block.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)))
        options = [o for o in re.split('\\s+or\\s+', value.strip()) if o.strip()] or [value]
        for where, text in (('title', title), ('attributes', joined)):
            if not text:
                continue
            have = _lit_words(text)
            tokens = set(re.findall('[a-z0-9]+', text.lower()))
            for option in options:
                want = _lit_words(option)
                if want and _words_hold(want, have) and not (USE_G70_SPLIT_WORDS and self.words_apart(option, [title] if where == 'title' else [str(v) for v in block.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)])):
                    return ('met', f'{where}: {text[:80]}')
                plain = [w for w in re.findall('[a-z0-9]+', option.lower()) if w not in FILLER_WORDS]
                if len(plain) >= 3 and _initials(plain) & tokens:
                    return ('met', f'{where} (initials): {text[:80]}')
        return ('missing', '')

    def fuller_keys(self, number: int, prop: str, keys: list[str]) -> list[str]:
        """#97 item 17: keys that add words of the item's own name or shelf to the property's words (a bag's "bag shape")."""
        if not 1 <= number <= len(self.items):
            return []
        item = self.items[number - 1]
        own = set(_key_words(item.name)) | set(_key_words(item.shelf or ''))
        mine = set(_key_words(prop))
        out = []
        for key in keys:
            words = set(_key_words(key))
            if mine and mine < words and words - mine <= own:
                out.append(key)
        return out

    def word_by_keys(self, ident: str, number: int, prop: str, value: str, block: dict[str, Any], keys: list[str]) -> tuple[str, str]:
        """#97 items 14/17: a word requirement against the keys named for its property. With G17 the longest key adding the
        item's own name decides over the bare key, one list part must hold every word of one choice (refining E50's list
        separators, so only with E50), and a key in the property's plural form listing variants establishes nothing when the
        title names another of them and not the wanted one. With G14 a form of one stem counts as the same word."""
        options = [o for o in re.split('\\s+or\\s+', value.strip()) if o.strip()] or [value]
        decide = keys
        remark = ''
        if USE_G17_KEY_PROOF:
            fuller = self.fuller_keys(number, prop, keys)
            if fuller:
                decide = [max(fuller, key=lambda k: (len(_key_words(k)), k))]
                bare = [k for k in keys if set(_key_words(k)) == set(_key_words(prop))]
                if bare and not any((_lit_words(o) and _words_hold(_lit_words(o), _lit_words(str(block[bare[0]]))) for o in options)):
                    remark = f' (decides over {bare[0]}={str(block[bare[0]])[:60]})'
        title_key = self.title_key()
        title = _lit_words(str((self.rows.get(ident) or {}).get(title_key) or '')) if title_key else set()
        prop_last = (_words(prop).split() or [''])[-1]
        for key in decide:
            raw = str(block[key])
            parts = [p for p in re.split(LIST_SPLIT, raw) if p.strip()] if USE_G17_KEY_PROOF and USE_E50_LITERAL_LIST_SEPARATORS else [raw]
            key_last = (_words(key).split() or [''])[-1]
            plural = USE_G17_KEY_PROOF and len(key_last) > len(prop_last) and _fold(key_last) == _fold(prop_last) and len(parts) >= 2
            for option in options:
                want = _lit_words(option)
                if not want or not any((_words_hold(want, _lit_words(p)) for p in parts)):
                    continue
                if USE_G70_SPLIT_WORDS and self.words_apart(option, [raw]):
                    continue
                if plural and title and not _words_hold(want, title) and any((_lit_words(p) and _lit_words(p) <= title and not _words_hold(want, _lit_words(p)) for p in parts)):
                    continue
                return ('met', f'{key}={raw}' + remark)
        return ('differs', f'{decide[0]}={block[decide[0]]}' + remark)

    def within_money(self, number: int, ident: str) -> bool:
        """#111 E40/E49: a priced listing whose units fit the budget and the item's cap."""
        price = self.price_of(ident)
        if price is None or (USE_E7_ITEM_PRICE_CAPS and self.over_cap(number, ident)):
            return False
        if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool):
            cost = price if self.brief.get('budget_scope') == 'each' else price * self.items[number - 1].qty
            return cost <= self.budget + 1e-09
        return True

    def literal_pool(self, number: int) -> list[str]:
        """#111 E49/E51: the opened listings the literal check reads for an item: by store label, or on single-item tasks
        (E51) every opened listing."""
        if len(self.items) == 1 and USE_E51_SINGLE_ITEM_ALL_OPENED:
            return [i for i in self.seen_order if i in self.full]
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if i in self.full and cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def proven(self, number: int, skip: Any=()) -> list[tuple[float, str, list[str]]]:
        """#111 E49: the item's opened, buyable listings within the budget and its cap whose own attributes meet every live
        requirement as written, cheapest first, each with the attributes that meet them."""
        needs = self.item_needs(number)
        out = []
        for ident in self.literal_pool(number):
            if ident in skip or not self.buyable(ident) or not self.within_money(number, ident):
                continue
            shown = []
            for need in needs:
                state, attr = self.literal_state(ident, number, need)
                if state != 'met':
                    break
                shown.append(attr)
            else:
                out.append((self.price_of(ident) or 0.0, ident, shown))
        return sorted(out, key=lambda o: (o[0], o[1]))

    def literal_picks(self) -> dict[int, list[tuple[float, str, list[str]]]]:
        """#111 E49/E53: proven listings for every item; {} when an item has no live requirement or none proven."""
        picks = {}
        for number in range(1, len(self.items) + 1):
            found = self.proven(number) if self.item_needs(number) else []
            if not found:
                return {}
            picks[number] = found
        return picks

    @_soft('')
    def checkout_holds(self) -> str:
        """#111 E38-E43: the first new checkout hold that fires (each at most once an episode, with more than HOLD_ROOM steps
        beyond the tail); no step is used."""
        if self.left() <= self.tail() + HOLD_ROOM:
            return ''
        lines = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not lines:
            return ''
        for key, on, check in (('bare', USE_E38_BARE_NUMBER_HOLD, self.bare_hold), ('variant', USE_E39_VARIANT_LIST_HOLD, self.variant_hold), ('weak', USE_E40_WEAK_EVIDENCE_HOLD, self.weak_hold), ('code', USE_E42_FIT_CODE_HOLD, self.code_hold), ('alike', USE_E43_LOOK_ALIKE_KEY_HOLD, self.lookalike_hold)):
            if not on or key in self.holds_fired:
                continue
            self.hold_new_causes = []
            text = check(lines)
            if text:
                self.holds_fired.add(key)
                if USE_G45_WEAK_LINE_HOLD:
                    self.hold_causes.update(self.hold_new_causes)
                self.trace.append({'kind': 'hold_' + key, 'text': text[:300]})
                return text
        return ''

    def plain_needs(self, number: int) -> list[tuple[str, str, str]]:
        """#111 E38-E43: the item's live requirements that are no pointer: (requirement, property, value)."""
        pointers = {need for _, need in self.refs}
        out = []
        for need in self.item_needs(number):
            prop, _, value = need.partition(':')
            if need not in pointers and prop.strip() and value.strip():
                out.append((need.strip(), prop.strip(), value.strip()))
        return out

    @_soft('')
    def bare_hold(self, lines: list[str]) -> str:
        """#111 E38: a line's attribute named for a requirement's property holds a bare number while the requirement gives a
        figure with a unit, and the two differ ten-fold or more."""
        for line in lines:
            number = self.line_owner(line)
            block = _fact_block(self.full.get(line) or {}) or {}
            for need, prop, value in self.plain_needs(number) if number and block else []:
                limit = _need_limit(value)
                if not limit or limit['num'] <= 0 or (USE_G45_WEAK_LINE_HOLD and (line, need) in self.hold_causes):
                    continue
                for key, held in block.items():
                    bare = _bare_number(held)
                    if bare is None or bare <= 0 or not _names(_spelled(prop), _spelled(re.sub('\\([^)]*\\)', ' ', str(key)))):
                        continue
                    if max(bare / limit['num'], limit['num'] / bare) >= 10:
                        self.hold_new_causes = [(line, need)]
                        return f"{line} gives {key}={held}, a bare number with no unit, while '{need}' asks for a figure in {limit['unit']}; read as written the two differ ten-fold or more, so the line does not show that it meets that requirement. Put in a listing that states the {prop} with its unit and meets it; keep {line} only if no such listing exists."
        return ''

    @_soft('')
    def variant_hold(self, lines: list[str]) -> str:
        """#111 E39: a line's attribute for a requirement's property lists two or more sizes in one unit, the wanted one among
        them, while every figure in that unit in its title is another one: the listing is the variant its title names."""
        title_key = self.title_key()
        for line in lines:
            number = self.line_owner(line)
            block = _fact_block(self.full.get(line) or {}) or {}
            title = str((self.rows.get(line) or {}).get(title_key) or '') if title_key else ''
            for need, prop, value in self.plain_needs(number) if number and block and title else []:
                limit = _need_limit(value)
                if not limit:
                    continue
                want, unit = (limit['num'], limit['unit'])
                for key, held in block.items():
                    if not _key_named(key, prop):
                        continue
                    figs, _ = _unit_figures(held, unit, bare=False)
                    named, _ = _unit_figures(title, unit, bare=False)
                    if len(figs) >= 2 and any((abs(f - want) < 1e-09 for f in figs)) and named and all((abs(f - want) >= 1e-09 for f in named)):
                        sizes = ', '.join(sorted({f'{f:g}' for f in named}))
                        return f"{line} lists {key}={held}, but its title ({title[:100]}) names {sizes} {unit}: the listing is that variant, not the {want:g} {unit} one '{need}' asks for. Put in the listing of the wanted variant; keep {line} only if no such listing exists."
        return ''

    @_soft('')
    def weak_hold(self, lines: list[str]) -> str:
        """#111 E40/E41: a bounded or decimal figure requirement that no key of the line sharing a word root with its property
        states: name proven listings of the same label and unread rows of it, and (E41) keys stating a figure in the
        requirement's unit under another name."""
        cat_key = self.category_key()
        found = []
        for line in lines:
            if len(found) >= WEAK_LINES:
                break
            number = self.line_owner(line)
            block = _fact_block(self.full.get(line) or {}) or {}
            for need, prop, value in self.plain_needs(number) if number and block else []:
                limit = _need_limit(value)
                if not limit or re.search(POINTER_TALK, value, re.I) or not (limit['kind'] != 'exact' or limit['decimal']) or (USE_G45_WEAK_LINE_HOLD and (line, need) in self.hold_causes):
                    continue
                roots = _roots(prop)
                strict = self.unit_under_keys(number, line, roots, limit['unit'])
                proof = [k for k in block if _roots(re.sub('\\([^)]*\\)', ' ', str(k))) & roots and re.search('\\d', str(block[k])) and (not strict or _unit_figures(block[k], limit['unit'])[0])]
                if not proof:
                    found.append((line, number, need, prop, limit, block))
                    break
        parts = []
        alike_any = False
        for line, number, need, prop, limit, block in found:
            label = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower() if cat_key else ''
            same = lambda i: str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == label if label else len(self.items) == 1
            proven = [(p, i) for p, i, _ in self.proven(number, skip={line}) if same(i)][:WEAK_SHOWN]
            unread: list[str] = []
            if self.left() > self.tail() + 4 and (not proven or self.brief.get('pick') == 'cheapest'):
                ceiling = min((p for p, _ in proven), default=None)
                title_key = self.title_key()
                mine = _lit_words(str((self.rows.get(line) or {}).get(title_key) or '')) if title_key else set()
                pool = [i for i in self.seen_order if i != line and i not in self.full and same(i) and self.buyable(i) and self.within_money(number, i) and (ceiling is None or (self.price_of(i) or 0.0) < ceiling)]
                pool.sort(key=lambda i: (-len(mine & _lit_words(str((self.rows.get(i) or {}).get(title_key) or ''))) if title_key else 0, self.price_of(i) or 0.0, i))
                unread = pool[:WEAK_SHOWN]
            if not proven and not unread:
                continue
            self.hold_new_causes.append((line, need))
            text = f'{line} does not state its {prop} under a key of its own (requirement: {need}), and the order is judged on that attribute.'
            alike = [k for k in block if not _roots(re.sub('\\([^)]*\\)', ' ', str(k))) & _roots(prop) and _unit_figures(block[k], limit['unit'], bare=False)[0]][:SAME_MEASURE_KEYS] if USE_E41_WEAK_SAME_MEASURE_KEY else []
            if alike:
                alike_any = True
                text += ' Its ' + ', '.join((f'{k}={block[k]}' for k in alike)) + f" give{'s' if len(alike) == 1 else ''} a figure in {limit['unit']}: if that is the same measure under another name, {line} already meets the requirement and stays."
            if proven:
                text += ' Read listings of the same kind that state it and meet every requirement as written: ' + ', '.join((f'{i} at {p:g}' for p, i in proven)) + '.'
            if unread:
                text += ' Unread rows of the same kind that may state it: ' + ', '.join((f'{i} at {self.price_of(i):g}' for i in unread)) + '.'
            parts.append(text)
        if not parts:
            return ''
        close = ' Swap such a line for a listing that states the property under its own key and meets it, keeping the pick rule; keep the line only if no such listing exists' + (' or its other key is that same measure.' if alike_any else '.')
        return ' '.join(parts) + close

    def unit_under_keys(self, number: int, line: str, roots: set[str], unit: str) -> bool:
        """#111 E40: whether other opened listings of the item write `unit` under keys sharing a root with the property."""
        cat_key = self.category_key()
        for ident in self.full:
            if ident == line or number not in self.line_items(ident, cat_key):
                continue
            block = _fact_block(self.full.get(ident) or {}) or {}
            if any((_roots(re.sub('\\([^)]*\\)', ' ', str(k))) & roots and _unit_figures(v, unit, bare=False)[0] for k, v in block.items())):
                return True
        return False

    @_soft('')
    def code_hold(self, lines: list[str]) -> str:
        """#111 E42: a line of an item with required fit codes whose record does not name every one of them (E28 matching)."""
        for line in lines:
            number = self.line_owner(line)
            codes = self.item_codes(number) if number else []
            if not codes or self.names_codes(line, codes):
                continue
            prefixes = {_code_prefix(c) for c in codes}
            family = [s for s in dict.fromkeys(self.codes_of(line)) if _code_prefix(s) in prefixes]
            if line not in self.full and not family:
                continue
            good = [i for i in self.fitting_rows(number) if i != line][:FIT_HOLD_SHOWN]
            text = f"{line} does not fit: its page does not name {', '.join(codes)}" + (f" (it names {', '.join(family[:3])})" if family else '') + '. '
            if good:
                text += 'Seen rows that name it, cheapest first: ' + ', '.join((i + (f' at {self.price_of(i):g}' if self.price_of(i) is not None else '') for i in good)) + '. Put one of them that meets the other requirements in the cart instead.'
            else:
                text += "No row seen names it: this store's search does not find codes, so search the item's kind with another required value and cart a part whose own text names the code."
            return text
        return ''

    @_soft('')
    def lookalike_hold(self, lines: list[str]) -> str:
        """#111 E43: a line with no attribute named for a figure requirement's property but another attribute in the same
        unit, while other opened listings of the item state the property under a key named for it."""
        for line in lines:
            number = self.line_owner(line)
            block = _fact_block(self.full.get(line) or {}) or {}
            for need, prop, value in self.plain_needs(number) if number and block else []:
                limit = _need_limit(value)
                if not limit or (re.search(NEGATION, value, re.I) and limit['kind'] == 'exact') or any((_key_named(k, prop) for k in block)) or (USE_G45_WEAK_LINE_HOLD and (line, need) in self.hold_causes):
                    continue
                alike = [k for k in block if _unit_figures(block[k], limit['unit'], bare=False)[0]]
                if not alike:
                    continue
                others = []
                cat_key = self.category_key()
                for ident in self.full:
                    theirs = _fact_block(self.full.get(ident) or {}) or {}
                    named = [k for k in theirs if _key_named(k, prop)]
                    if ident == line or not named or number not in self.line_items(ident, cat_key) or not self.buyable(ident):
                        continue
                    cover = sum((1 for _, other, _ in self.plain_needs(number) if any((_key_named(k, other) for k in theirs))))
                    price = self.price_of(ident)
                    others.append((-cover, price if price is not None else float('inf'), ident, named[0], theirs[named[0]]))
                if not others:
                    continue
                others.sort(key=lambda o: (o[0], o[1], o[2]))
                self.hold_new_causes = [(line, need)]
                shown = ', '.join((f'{i} ({k}={v}' + (f', {p:g})' if p != float('inf') else ')') for _, p, i, k, v in others[:WEAK_SHOWN]))
                return f'{line} has no attribute named for {prop} (requirement: {need}); its {alike[0]}={block[alike[0]]} is a neighbouring measure in the same unit, not that property. Opened listings of the same item that state {prop} under its own key: {shown}. Put one that meets the requirement in the cart instead, keeping the pick rule; keep {line} only if none of them does.'
        return ''

    @_soft(('', []))
    def cheaper_rounds(self) -> tuple[str, list[str]]:
        """#111 E44: after a passing review whose cheaper read opened something, up to CHEAPER_ROUNDS more rounds: open the next
        unread cheaper rows of the cart lines' labels and review again (each round a new review)."""
        verdict, more = ('', [])
        for _ in range(CHEAPER_ROUNDS):
            if self.left() <= self.tail() + CHEAPER_ROOM or self.finished or self.rushed() or not self.below_cart():
                break
            got = self.cheaper_rows(extra=True)
            self.trace.append({'kind': 'cheaper_round', 'round': self.cheaper_round + 1, 'opened': len(got)})
            if not got:
                break
            self.cheaper_round += 1
            more += got
            verdict = self.check_cart()
            if verdict:
                break
        return (verdict, more)

    @_soft(False)
    def contradicts_words(self, ident: str) -> bool:
        """#111 E46: whether an opened listing's keys named for one of its item's word requirements (no digit, no pointer, no
        negation) all state values sharing no word with it."""
        number = self.line_owner(ident)
        block = _fact_block(self.full.get(ident) or {}) or {}
        for need, prop, value in self.plain_needs(number) if number and block else []:
            if re.search('\\d', value) or re.search(NEGATION, value, re.I) or not _lit_words(value):
                continue
            want = _lit_words(value)
            said = [_lit_words(block[k]) for k in block if _key_named(k, prop)]
            if said and all((have and not want & have for have in said)):
                return True
        return False

    @_soft('')
    def decline_chain_round(self) -> str:
        """#111 E48: at a decline, one more chain round (the king's trail, one model call) when a pointer requirement is still
        open and listings were opened since the last round; the king's chain counters are left as they were."""
        if not self.pending_refs() or len(self.opened) <= max(0, self.trail_opened) or self.left() < 3 or self.model_room() < DECLINE_TRAIL_SECONDS:
            return ''
        kept = (self.trails, self.trail_blocks, self.trail_room)
        before = dict(self.ref_values)
        self.trails, self.trail_room = (0, 1)
        try:
            text, _ = self.trail()
        finally:
            self.trails, self.trail_blocks, self.trail_room = kept
        self.trace.append({'kind': 'decline_trail', 'resolved': [k for k in self.ref_values if k not in before][:3]})
        return text.split('The order counts only after')[0]

    @_soft('')
    def literal_hold(self) -> str:
        """#111 E49: once, at a decline with no review leads: when every item has opened listings meeting every requirement as
        their attributes write it, name the cheapest of them and hold the decline."""
        if 'e49' in self.decline_checks or self.leads or self.left() <= self.tail() + HOLD_ROOM:
            return ''
        if USE_G72_LITERAL_HOLD_TOTAL and (self.left() <= SET_ROOM or self.bound or 'sets' in self.decline_checks):
            return ''
        picks = self.literal_picks()
        if not picks:
            return ''
        if USE_G72_LITERAL_HOLD_TOTAL and isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) and self.brief.get('budget_scope') != 'each':
            used: set[str] = set()
            total = 0.0
            for number, found in picks.items():
                pick = next(((p, i) for p, i, _ in found if i not in used), None)
                if not pick:
                    return ''
                used.add(pick[1])
                total += pick[0] * self.items[number - 1].qty
            if total > self.budget + 1e-09:
                self.trace.append({'kind': 'literal_hold_over_budget', 'total': total})
                return ''
        self.decline_checks.add('e49')
        many = len(self.items) > 1
        parts = []
        for number, found in picks.items():
            shown = '; '.join((f"{i} at {p:g} ({', '.join((a for a in attrs if a)) or 'its attributes'})" for p, i, attrs in found[:LITERAL_SHOWN]))
            parts.append((f'item {number} ({self.items[number - 1].name}): ' if many else '') + shown)
        self.trace.append({'kind': 'literal_hold', 'picks': {str(n): [i for _, i, _ in f[:LITERAL_SHOWN]] for n, f in picks.items()}})
        return 'Listings already opened meet every requirement as their own attributes write it: ' + ' | '.join(parts) + '. Read every figure exactly as it is written, not as you think it should read. Put the best of them for each item in the cart at the requested quantity (keep the pick rule) and check out; end without a purchase only if each of them clearly breaks a requirement.'

    @_soft('')
    def nearest_hold(self) -> str:
        """#111 E52: once, at a decline with time left and a listing opened: one light call reads only the request for a rule
        to buy the nearest listing when nothing fits; when it quotes such a sentence found in the request, hold the decline."""
        if 'e52' in self.decline_checks or self.model_room() < NEAREST_SECONDS or not (self.full or self.opened) or self.no_credit:
            return ''
        self.decline_checks.add('e52')
        if USE_P110_1_NEAREST_ORDER and self.nearest_read is not None:  # #110 item 1: the request was read for this already
            if not self.nearest_read:
                return ''
            sentence = self.nearest_sentence
        else:
            got = self.ask_json(NEAREST_RULE + '\n\nREQUEST:\n' + self.request, [LIGHT], 800) or {}
            sentence = str(got.get('sentence') or '').strip().strip('"').strip()
            self.trace.append({'kind': 'nearest_rule', 'got': _clip(got, 200)})
            if not self.read_nearest(got):
                return ''
        return f'The request says: "{sentence}" So ending without a purchase is not what it asks for when nothing meets everything. For each item, put in the cart the listing you have read that misses the fewest requirements (being out of stock or over the budget counts as a miss; stay within any slack the request allows), at the requested quantity, then check out and say in your text which requirement it misses.'

    @_soft('')
    def proven_order(self) -> str:
        """#111 E53: once, at a decline with LATE_ORDER_STEPS or fewer steps left: when every item has a literally proven
        listing of its own label in stock and within its cap, distinct and within a total budget, Cairn sets the cart to
        exactly those lines, runs the pre-order steps and checks out."""
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if 'e53' in self.decline_checks or not cart or not buy or self.left() < 1 or self.finished:
            return ''
        self.decline_checks.add('e53')
        picks = self.literal_picks()
        if not picks:
            return ''
        cat_key = self.category_key()
        lines: dict[str, int] = {}
        for number, found in picks.items():
            item = self.items[number - 1]
            labelled = bool(cat_key) and any((self.owns(item, str(r.get(cat_key) or '')) for r in self.rows.values()))
            choice = next((i for _, i, _ in found if i not in lines and self.buyable(i) and not (USE_E7_ITEM_PRICE_CAPS and self.over_cap(number, i)) and (not labelled or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')))), None)
            if not choice:
                return ''
            lines[choice] = item.qty
        if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) and self.brief.get('budget_scope') != 'each':
            if sum(((self.price_of(i) or 0.0) * q for i, q in lines.items())) > self.budget + 1e-09:
                return ''
        drop = [i for i in self.cart_ids() if i not in lines]
        calls = self.cart_calls(lines, drop)
        if calls is None or len(calls) + 1 > self.left():
            return ''
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            reply = self.act(cart, args) if not faults else None
            if reply is None or self.finished:
                self.trace.append({'kind': 'proven_order', 'lines': lines, 'stopped': _dump(args)[:200]})
                return 'Not run (Cairn acted instead): listings already opened meet every requirement as written, so Cairn began to set the cart to ' + ', '.join(lines) + ' (' + ('; '.join(done) or 'no call ran') + '), but a cart call did not go through. Cart now: ' + self.cart_text() if done else ''
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
            done.append(_dump(args))
        head = 'Not run (Cairn acted instead): listings already opened meet every requirement as written, so Cairn set the cart to ' + ', '.join((f'{i} x{q}' for i, q in lines.items())) + (' (' + '; '.join(done) + ')' if done else ' (it held them already)')
        go, said = self.gate('order')
        if not go:
            return head + '. ' + said
        faults = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
        if faults or self.left() < 1 or self.finished:
            return head + '. Check out now.'
        reply = self.act(buy, args)
        self.trace.append({'kind': 'proven_order', 'lines': lines, 'reply': _clip(reply, 200)})
        if reply is None:
            return head + '. The checkout call failed; check out now.'
        self.buys += 1
        if not reply.get('finished'):
            self.refused_at = self.step
        self.observe(buy, reply.get('observation'))
        return head + (' and ran the steps the request asks for before an order: ' + said if said else '') + ' and checked out. Store reply: ' + self.show(reply.get('observation'))

    # ---- #97 items 44-54: holds before the checkout and the cart check
    def label_of(self, ident: str) -> str:
        cat_key = self.category_key()
        return str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ''

    def need_list(self, number: int) -> list[tuple[str, str, str]] | None:
        """#97 item 16: the item's live requirements as (requirement, property, value), without those only the shopper's
        words state; a pointer requirement carries the value its chain read. None when a pointer value is missing, still a
        reference to a page, holds a bound or condition word, or the pointer asks for a different value."""
        pointers = {need for _, need in self.refs}
        out = []
        for need in self.item_needs(number):
            prop, colon, value = need.partition(':')
            if not colon or not prop.strip() or not value.strip() or self.described(number, need):
                continue
            if need in pointers:
                got = self.ref_values.get(need, '')
                if not got or got.startswith('as stated on') or re.search(OPEN_NOT, got, re.I) or re.search(DIFFERENCE, value, re.I) or re.search(NEGATION, value, re.I):
                    return None
                value = got
            out.append((need.strip(), prop.strip(), value.strip()))
        return out

    def codes_stated(self, ident: str, codes: list[str]) -> bool:
        """#97 item 15: every code is named by the listing (its text or fit clause) and none is barred by its fit clause."""
        return bool(codes) and self.names_codes(ident, codes) and not self.barred_codes(ident, codes)

    def words_apart(self, option: str, texts: list[Any]) -> bool:
        """#97 item 70: a plain multi-word value whose words no text holds as one run while one holds them with other
        words between: not stated."""
        if not USE_G70_SPLIT_WORDS or not _plain_words_value(option):
            return False
        runs = [_split_run(option, t) for t in texts]
        return False in runs and True not in runs

    def word_said(self, ident: str, number: int, prop: str, value: str, block: dict[str, Any], strict: bool) -> bool:
        """#97 item 13: a word requirement stated as the grader reads it. A key named for the property decides: in strict mode
        one value word (or a form of it) must sit in such a key, in lenient mode any such key counts. With no such key: a code
        named in the fit clause (#15), a value word or form in the joined attribute values or the title, or the initials of
        three or more words of a choice as one token there. The description is not used; a value with no words is stated."""
        options = [o for o in re.split('\\s+or\\s+', value.strip()) if o.strip()] or [value]
        if not any((_lit_words(o) for o in options)):
            return True
        keys = [k for k in block if _key_named(k, prop)]
        if keys:
            if not strict:
                return True
            for key in keys:
                have = _lit_words(block[key])
                for option in options:
                    if any((w in have or any((_same_stem(w, h) for h in have)) for w in _lit_words(option))) and not self.words_apart(option, [block[key]]):
                        return True
            return False
        codes = _codes(value)
        if codes and self.codes_stated(ident, codes):
            return True
        title_key = self.title_key()
        title = str((self.rows.get(ident) or {}).get(title_key) or '') if title_key else ''
        values = [str(v) for v in block.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
        for text, parts in ((title, [title]), (' '.join(values), values)):
            if not text:
                continue
            have = _lit_words(text)
            tokens = set(re.findall('[a-z0-9]+', text.lower()))
            for option in options:
                if any((w in have or any((_same_stem(w, h) for h in have)) for w in _lit_words(option))) and not self.words_apart(option, parts):
                    return True
                plain = [w for w in re.findall('[a-z0-9]+', option.lower()) if w not in FILLER_WORDS]
                if len(plain) >= 3 and _initials(plain) & tokens:
                    return True
        return False

    @staticmethod
    def stem_keys(block: dict[str, Any], prop: str) -> list[str]:
        """#97 items 11/12: the keys sharing a word stem with the property whose value holds a digit."""
        roots = _roots(prop)
        return [k for k in block if _roots(re.sub('\\([^)]*\\)', ' ', str(k))) & roots and re.search('\\d', str(block[k]))]

    def limit_stated(self, block: dict[str, Any], prop: str, limit: dict[str, Any]) -> bool:
        """#97 item 11: a measured limit is stated when a key sharing a word stem with the property holds a digit and either
        the requirement has no unit, the value has no letters, or the unit follows a digit (space or hyphen allowed)."""
        for key in self.stem_keys(block, prop):
            held = str(block[key])
            if not limit.get('unit') or not re.search('[A-Za-z]', held) or _unit_figures(held, limit['unit'], bare=False)[0]:
                return True
        return False

    def misread(self, block: dict[str, Any], prop: str, limit: dict[str, Any]) -> list[str]:
        """#97 item 12: the keys of a line misread on a measured limit: every key for the property holding a digit holds a
        bare figure BARE_OFF times or more off the limit ([] when one does not)."""
        keys = self.stem_keys(block, prop)
        return keys if keys and all((_bare_note(block[k], limit) for k in keys)) else []

    def could_state(self, line: str, number: int, prop: str, value: str, label: str) -> bool:
        """#97 item 44 guard: some listing of the label could state the word: fewer than WEAK_GUARD_READ others read, or a
        read one states it in strict mode, or a value word sits in the title of a seen listing of that label."""
        others = [i for i in self.seen_order if i != line and self.label_of(i) == label]
        read = [i for i in others if _fact_block(self.full.get(i) or {})]
        if len(read) < WEAK_GUARD_READ:
            return True
        if any((self.word_said(i, number, prop, value, _fact_block(self.full.get(i) or {}) or {}, True) for i in read)):
            return True
        words = _lit_words(re.sub('\\s+or\\s+', ' ', value))
        title_key = self.title_key()
        return bool(title_key) and any((words & _lit_words(str((self.rows.get(i) or {}).get(title_key) or '')) for i in others))

    @_soft(dict)
    def weak_lines(self) -> dict[str, list[tuple[str, str, str]]]:
        """#97 item 44: held cart lines with read attributes whose label belongs to one item, with the requirements they do
        not state the way the order is judged: (requirement, property, 'key' | 'bare' | 'word'). A measured limit is weak
        when no key named for it states it (#11) or every such key holds a far-off bare figure (#12); a plain word
        requirement (choices allowed) when it is not stated in lenient mode (#13) and some listing of the label could state
        it. Pointer, code and shopper's-words requirements are left out."""
        if not USE_G44_WEAK_LINES or not self.category_key():
            return {}
        pointers = {need for _, need in self.refs}
        out: dict[str, list[tuple[str, str, str]]] = {}
        for line in [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]:
            block = _fact_block(self.full.get(line) or {}) or {}
            owners = self.line_items(line)
            if not block or len(owners) != 1:
                continue
            number = owners[0]
            weak = []
            for need in self.item_needs(number):
                prop, colon, value = need.partition(':')
                if not colon or not prop.strip() or not value.strip() or need in pointers or _codes(value) or self.described(number, need):
                    continue
                prop, value = (prop.strip(), value.strip())
                limit = _measured_limit(value)
                if limit:
                    if not self.limit_stated(block, prop, limit):
                        weak.append((need, prop, 'key'))
                    elif self.misread(block, prop, limit):
                        weak.append((need, prop, 'bare'))
                    continue
                if re.search(OPEN_NOT, re.sub('\\s+or\\s+', ' ', value), re.I) or re.search(NEGATION, value, re.I):
                    continue
                if not self.word_said(line, number, prop, value, block, False) and self.could_state(line, number, prop, value, self.label_of(line)):
                    weak.append((need, prop, 'word'))
            if weak:
                out[line] = weak
        return out

    def states_weak(self, ident: str, number: int, need: str, prop: str, how: str) -> bool:
        """#97 item 45: whether a read listing states one weak requirement (the way the line did not)."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        if not block:
            return False
        value = need.partition(':')[2].strip()
        if how == 'word':
            return self.word_said(ident, number, prop, value, block, False)
        limit = _measured_limit(value)
        return bool(limit) and self.limit_stated(block, prop, limit) and not self.misread(block, prop, limit)

    def money_ok(self, number: int, ident: str) -> bool:
        """#97 items 45/54: inside a per-item budget and the item's cap; an unpriced listing is kept."""
        price = self.price_of(ident)
        if price is None:
            return True
        if USE_E7_ITEM_PRICE_CAPS and self.over_cap(number, ident):
            return False
        return not (self.brief.get('budget_scope') == 'each' and isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) and price > self.budget + 1e-09)

    @_soft('')
    def weak_line_hold(self, read: list[str]) -> str:
        """#97 item 45: once an episode (counted when it fires), with more than HOLD_ROOM steps beyond the tail: for up to
        WEAK_LINES weak lines (#44), name read listings of the same label (not in the cart, buyable, inside a per-item
        budget) that state every weak requirement of the line, cheapest first; else, with more than WEAK_UNREAD_ROOM steps
        beyond the tail, unread rows of that label (rows mentioning a weak property first, then shared title words, then
        price). Causes another hold already named are left out."""
        if 'g45' in self.holds_fired or self.left() <= self.tail() + HOLD_ROOM:
            return ''
        weak = self.weak_lines()
        cart = set(self.cart_ids())
        title_key = self.title_key()
        parts: list[str] = []
        causes: list[tuple[str, str]] = []
        lines = 0
        for line, needs in weak.items():
            needs = [n for n in needs if (line, n[0]) not in self.hold_causes]
            if not needs or lines >= WEAK_LINES:
                continue
            lines += 1
            number = self.line_owner(line)
            label = self.label_of(line)
            pool = [i for i in self.seen_order if i != line and i not in cart and label and self.label_of(i) == label and self.buyable(i) and self.money_ok(number, i)]
            named = sorted([i for i in pool if i in self.full and all((self.states_weak(i, number, n, p, h) for n, p, h in needs))], key=lambda i: (self.price_of(i) is None, self.price_of(i) or 0.0, i))[:WEAK_SHOWN]
            unread: list[str] = []
            if not named and self.left() > self.tail() + WEAK_UNREAD_ROOM:
                mine = _lit_words(str((self.rows.get(line) or {}).get(title_key) or '')) if title_key else set()
                props = [set(_key_words(p)) for _, p, _ in needs]
                rest = [i for i in pool if i not in self.full]
                rest.sort(key=lambda i: (-any((p and p <= set(_key_words(_dump(self.rows.get(i) or {}))) for p in props)), -len(mine & _lit_words(str((self.rows.get(i) or {}).get(title_key) or ''))) if title_key else 0, self.price_of(i) is None, self.price_of(i) or 0.0, i))
                unread = rest[:WEAK_SHOWN]
            if not named and not unread:
                continue
            block = _fact_block(self.full.get(line) or {}) or {}
            said = []
            for need, prop, how in needs:
                if how == 'key':
                    said.append(f'no key named for {prop} states its figure (requirement: {need})')
                elif how == 'bare':
                    limit = _measured_limit(need.partition(':')[2]) or {}
                    said.append(f"its {prop} is only a bare figure far off the requirement ({', '.join((f'{k}={block[k]}' for k in self.misread(block, prop, limit)))}, read in {limit.get('unit', 'its unit')}; requirement: {need})")
                else:
                    said.append(f"no attribute value or title states {need.partition(':')[2].strip()!r} (requirement: {need}; a description alone does not count)")
            text = f'{line} does not show every requirement the way the order is judged: ' + '; '.join(said) + '.'
            price = lambda i: f' at {self.price_of(i):g}' if self.price_of(i) is not None else ''
            if named:
                text += ' Read listings of the same kind that state it: ' + ', '.join((i + price(i) for i in named)) + '.'
            else:
                text += ' Unread rows of the same kind that may state it: ' + ', '.join((i + price(i) for i in unread)) + '.'
            parts.append(text)
            causes += [(line, n) for n, _, _ in needs]
        if not parts:
            return ''
        self.holds_fired.add('g45')
        self.hold_causes.update(causes)
        self.trace.append({'kind': 'hold_g45', 'lines': [c[0] for c in causes][:4]})
        head = "Cairn read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
        return head + ' '.join(parts) + ' Swap such a line for a listing that both shows and meets every requirement, keeping the pick rule, then check out; keep the line only when no such listing exists.'

    @_soft(list)
    def code_links(self, ids: list[str]) -> list[str]:
        """#97 item 47: the fit links the cart's own records establish by code: for each fit-bound pair with exactly one device
        line, each part line whose series link holds, or (no series system) whose fit clause names a model code of the device
        that it does not bar; and each line of one item whose fit clause names every code of a code requirement of it."""
        index = self.fit_index()
        marks = index['marks']
        out: list[str] = []
        for part, main in self.bound:
            devices = [i for i in ids if main in self.line_items(i)]
            if len(devices) != 1:
                continue
            device = devices[0]
            system = self.link_system(part, main, index)
            codes = list(dict.fromkeys((marks.get(device) or {}).get('codes') or []))
            for line in [i for i in ids if part in self.line_items(i) and i != device]:
                if system:
                    carried, namer = (line, device) if system[0] == part else (device, line)
                    if self.link_holds(carried, namer, system[2], index) is True:
                        series = sorted(set((marks.get(carried) or {}).get('fam') or []) & set((marks.get(namer) or {}).get('named') or []) & system[2])
                        out.append(f"{line} works with {device}: series {', '.join(series)}")
                elif codes:
                    bars = (marks.get(line) or {}).get('bars') or []
                    hit = [c for c in codes if any((_covers_code(p, c) for p in self.fit_set(line, index))) and not any((_code_hit(b, c) for b in bars))]
                    if hit:
                        out.append(f"{line} works with {device}: its fit clause names {', '.join(hit)}")
        for line in ids:
            owners = self.line_items(line)
            if len(owners) != 1:
                continue
            bars = (marks.get(line) or {}).get('bars') or []
            fits = self.fit_set(line, index)
            for need in self.item_needs(owners[0]):
                codes = _codes(need.partition(':')[2])
                if codes and all((any((_covers_code(p, c) for p in fits)) for c in codes)) and not any((_code_hit(b, c) for b in bars for c in codes)):
                    out.append(f"{line} meets '{need}': its fit clause names {', '.join(codes)}")
        return out

    def keeps_links(self, number: int, new: str, old: str) -> bool:
        """#97 item 48: whether `new`, taking cart line `old`'s place for item `number`, keeps every fit link: for each
        fit-bound pair holding the item a cart line of the other item exists, and each pairing holds (a series system: the
        king's link_holds says yes; none: the device has model codes, the part's fit set names one, the part bars none)."""
        cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and i != old]
        index = self.fit_index()
        marks = index['marks']
        for part, main in self.bound:
            if number not in (part, main):
                continue
            other = main if number == part else part
            partners = [i for i in cart if other in self.line_items(i)]
            if not partners:
                return False
            system = self.link_system(part, main, index)
            for partner in partners:
                device, piece = (partner, new) if number == part else (new, partner)
                if system:
                    carried, namer = (piece, device) if system[0] == part else (device, piece)
                    if self.link_holds(carried, namer, system[2], index) is not True:
                        return False
                    continue
                codes = (marks.get(device) or {}).get('codes') or []
                bars = (marks.get(piece) or {}).get('bars') or []
                if not codes or not any((_covers_code(p, c) for p in self.fit_set(piece, index) for c in codes)) or any((_code_hit(b, c) for b in bars for c in codes)):
                    return False
        return True

    def established(self, ident: str, number: int, needs: list[tuple[str, str, str]]) -> list[str] | None:
        """#97 item 17: the keys by which a read listing establishes every requirement of the list (#16): each value has
        words and no bound or condition word (choices allowed) and no measured limit; among the keys named for the property
        the king's word_by_keys (fuller key, list parts, plural variants, #70) must find it; a code requirement by its fit
        clause (#15). None when one fails or the listing has no attributes."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        if not block:
            return None
        shown = []
        for need, prop, value in needs:
            codes = _codes(value)
            if codes:
                if not self.codes_stated(ident, codes):
                    return None
                shown.append(f"names {', '.join(codes)}")
                continue
            if not _lit_words(value) or re.search(OPEN_NOT, re.sub('\\s+or\\s+', ' ', value), re.I) or _measured_limit(value):
                return None
            keys = [k for k in block if _key_named(k, prop)]
            if not keys:
                return None
            state, attr = self.word_by_keys(ident, number, prop, value, block, keys)
            if state != 'met':
                return None
            shown.append(attr)
        return shown

    @_soft(list)
    def cheaper_pairs(self) -> list[tuple[str, str, list[str]]]:
        """#97 item 49: on a lowest-price pick, for each held cart line of one item with a requirement list (#16) and a known
        price: the cheapest read listing of exactly its label, not in the cart, buyable, cheaper by any amount, inside the
        budget and cap, that establishes every requirement (#17) and keeps every fit link (#48)."""
        if self.brief.get('pick') != 'cheapest' or not self.category_key():
            return []
        cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        bound = {n for pair in self.bound for n in pair}
        out = []
        for line in cart:
            price = self.price_of(line)
            owners = self.line_items(line)
            label = self.label_of(line)
            if price is None or len(owners) != 1 or not label:
                continue
            number = owners[0]
            if number in bound and not USE_G48_SWAP_KEEPS_LINKS:
                continue
            needs = self.need_list(number)
            if not needs:  # no list (#16), or no requirement a listing could establish
                continue
            pool = sorted((i for i in self.full if i not in cart and self.label_of(i) == label and self.buyable(i) and self.price_of(i) is not None and (self.price_of(i) or 0.0) < price - 1e-09 and self.within_money(number, i)), key=lambda i: (self.price_of(i) or 0.0, i))
            for ident in pool:
                keys = self.established(ident, number, needs)
                if keys is None or (number in bound and not self.keeps_links(number, ident, line)):
                    continue
                out.append((line, ident, keys))
                break
        return out

    def cheaper_text(self, pairs: list[tuple[str, str, list[str]]]) -> str:
        """#97 items 49-51: the cheaper hold's text."""
        bound = {n for pair in self.bound for n in pair}
        parts = []
        links: list[str] = []
        for old, new, keys in pairs:
            parts.append(f'{new} at {self.price_of(new):g} in place of {old} at {self.price_of(old):g}' + (f" ({', '.join((k for k in keys if k))})" if any(keys) else ''))
            if USE_G51_CHEAPER_CODE_LINKS and self.line_owner(old) in bound:
                swapped = [new if i == old else i for i in self.cart_ids()]
                links += [l for l in self.code_links(swapped) if new in l and l not in links][:CODE_LINKS_SHOWN]
        text = 'The pick rule is the lowest price, and read listings that establish every requirement under the keys named for them cost less than cart lines: ' + '; '.join(parts) + '.'
        if links:
            text += ' Fit is judged by codes, so a description that lists other uses does not cancel these links: ' + '; '.join(links[:CODE_LINKS_SHOWN]) + '.'
        return text + ' Swap each with the cart tool (take the cart line out, add the cheaper listing at the requested quantity), then check out; keep a line only if the named record clearly fails a requirement.'

    @_soft('')
    def cheaper_hold(self, read: list[str], late: bool=False) -> str:
        """#97 items 49/50: once an episode (shared by both): name the cheaper read listings (#49) instead of checking out:
        with more than HOLD_ROOM steps beyond the tail after the checkout's reads, or (late, #50) with at most that many left
        when the steps cover the tail, one more and two per swap; Cairn remembers the pairs it named."""
        if 'g49' in self.holds_fired:
            return ''
        if late:
            if self.left() > self.tail() + HOLD_ROOM:
                return ''
        elif self.left() <= self.tail() + HOLD_ROOM:
            return ''
        pairs = self.cheaper_pairs()
        if not pairs or (late and self.left() < self.tail() + 1 + 2 * len(pairs)):
            return ''
        self.holds_fired.add('g49')
        self.cheaper_named += [(o, n) for o, n, _ in pairs if (o, n) not in self.cheaper_named]
        self.trace.append({'kind': 'hold_cheaper', 'pairs': [(o, n) for o, n, _ in pairs], 'late': late})
        head = "Cairn read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read and not late else ''
        return head + self.cheaper_text(pairs)

    @_soft('')
    def agreed_swap(self) -> str:
        """#97 items 52/53: once an episode, on a lowest-price pick: when a pair the cheaper hold named is still found and CHECK,
        having rejected this exact cart, named the cheaper listing as better (52), or the line's item is fit-bound and the
        cheaper listing keeps every fit link (53), Cairn swaps the line itself at its quantity, when the steps cover the cart
        calls, two more and the tail."""
        cart_tool = self.roles.get('cart')
        if 'g52' in self.holds_fired or not cart_tool or self.brief.get('pick') != 'cheapest' or not self.cheaper_named:
            return ''
        pairs = {(o, n) for o, n, _ in self.cheaper_pairs()}
        better = self.better_by_cart.get('|'.join(sorted(self.cart_ids())), [])
        bound = {n for pair in self.bound for n in pair}
        for old, new in self.cheaper_named:
            if (old, new) not in pairs:
                continue
            number = self.line_owner(old)
            if new in better:
                reason = f'its own check and the careful review of this cart both named {new} as the better pick'
            elif USE_G53_BOUND_SWAP and USE_G48_SWAP_KEEPS_LINKS and number in bound and self.keeps_links(number, new, old):
                links = [l for l in self.code_links([new if i == old else i for i in self.cart_ids()]) if new in l]
                reason = f'{new} keeps every fit link with the cart' + (' (' + '; '.join(links[:CODE_LINKS_SHOWN]) + ')' if links else '')
            else:
                continue
            units = self.held_units(old) or self.cart_lines.get(old) or (self.items[number - 1].qty if number else 1)
            kept = {i: self.cart_lines[i] for i in self.cart_ids() if i in self.cart_lines and i != old}  # a whole-cart tool sets every line
            calls = self.cart_calls({**kept, new: units}, [old])
            if not calls or len(calls) + 2 + self.tail() > self.left():
                continue
            self.holds_fired.add('g52')
            done = []
            for args in calls:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(cart_tool), 'args', faults)
                reply = self.act(cart_tool, args) if not faults else None
                if reply is None:
                    break
                self.note_cart(cart_tool, args)
                self.observe(cart_tool, reply.get('observation'))
                done.append(_dump(args) + ' -> ' + self.show(reply.get('observation'))[:200])
                if self.finished:
                    break
            if old not in self.cart_lines:
                self.dropped.add(old)
            self.dropped.discard(new)
            self.built = [new if i == old else i for i in self.built] if old in self.built else self.built + ([new] if new in self.cart_lines and new not in self.built else [])
            self.trace.append({'kind': 'agreed_swap', 'old': old, 'new': new, 'calls': len(done)})
            return f'Not run yet (no step used for the checkout). The pick rule is the lowest price, {new} costs {self.price_of(new):g} against {old} at {self.price_of(old):g} and establishes every requirement, and {reason}, so Cairn swapped the line itself: ' + ('; '.join(done) or 'no call went through') + '. Check out now.'
        return ''

    @_soft('')
    def late_holds(self, read: list[str]) -> str:
        """#97 items 49, 45, 52/53 in the guide's order, after the checkout's reads and before the cart check."""
        if USE_G49_CHEAPER_HOLD:
            text = self.cheaper_hold(read)
            if text:
                return 'Not run yet (no step used). ' + text
        if USE_G45_WEAK_LINE_HOLD:
            text = self.weak_line_hold(read)
            if text:
                return 'Not run yet (no step used). ' + text
        if USE_G52_AGREED_SWAP:
            return self.agreed_swap()
        return ''

    def set_parts(self) -> tuple[int, list[int]] | None:
        """#97 item 54: (device item, its part items) when every fit-bound pair shares one device item."""
        mains = {m for _, m in self.bound}
        if len(mains) != 1:
            return None
        main = mains.pop()
        return (main, list(dict.fromkeys((p for p, m in self.bound if m == main))))

    def figure_as_written(self, block: dict[str, Any], prop: str, limit: dict[str, Any]) -> bool:
        """#97 item 71: an at-least or exact measured requirement met as written: the first figure in the requirement's unit
        (a bare figure counts in it) under some key named for the property reaches the floor, or is the same figure."""
        if limit['kind'] not in ('min', 'exact'):
            return False
        for key in [k for k in block if _key_named(k, prop)]:
            figs, _ = _unit_figures(block[key], limit['unit'])
            if not figs:
                continue
            first = figs[0]
            if limit['kind'] == 'min' and (first > limit['lo'] + 1e-09 if limit['strict'] else first >= limit['lo'] - 1e-09):
                return True
            if limit['kind'] == 'exact' and abs(first - limit['num']) < 1e-09:
                return True
        return False

    def meets_item(self, ident: str, number: int) -> bool:
        """#97 items 18/71: a read, buyable, priced listing meets its item when the item has a non-empty requirement list
        (#16), no requirement only the shopper's words state, and every requirement is a word requirement (choices allowed)
        established under a key named for it (#17) or, with no such key, stated in strict mode (#13); with #71 an at-least or
        exact measured requirement may be met as written; a code requirement by the fit clause (#15)."""
        if ident not in self.full or not self.buyable(ident) or self.price_of(ident) is None:
            return False
        if any((self.described(number, need) for need in self.item_needs(number))):
            return False
        needs = self.need_list(number)
        block = _fact_block(self.full.get(ident) or {}) or {}
        if not needs or not block:
            return False
        for need, prop, value in needs:
            codes = _codes(value)
            if codes:
                if not self.codes_stated(ident, codes):
                    return False
                continue
            limit = _measured_limit(value)
            if limit or re.search(OPEN_NOT, re.sub('\\s+or\\s+', ' ', value), re.I):
                if limit and USE_G71_SET_FIGURES and self.figure_as_written(block, prop, limit):
                    continue
                return False
            keys = [k for k in block if _key_named(k, prop)]
            if keys:
                if self.word_by_keys(ident, number, prop, value, block, keys)[0] != 'met':
                    return False
            elif not self.word_said(ident, number, prop, value, block, True):
                return False
        return True

    def fits_device(self, part: int, main: int, piece: str, device: str, index: dict[str, Any]) -> bool:
        """#97 item 54: a part listing fits a device listing by its series link or by a device model code its fit clause
        names and does not bar."""
        system = self.link_system(part, main, index)
        if system:
            carried, namer = (piece, device) if system[0] == part else (device, piece)
            if self.link_holds(carried, namer, system[2], index) is True:
                return True
        marks = index['marks']
        codes = (marks.get(device) or {}).get('codes') or []
        bars = (marks.get(piece) or {}).get('bars') or []
        return bool(codes) and any((_covers_code(p, c) for p in self.fit_set(piece, index) for c in codes)) and not any((_code_hit(b, c) for b in bars for c in codes))

    @_soft(list)
    def whole_sets(self) -> list[tuple[float, dict[str, int]]]:
        """#97 item 54: on tasks whose fit-bound pairs share one device item, whole sets of read listings meeting their item
        (#18): for each meeting device, the cheapest meeting part of each part item that fits it plus the cheapest meeting
        listing of every other item; sets over a total budget left out; cheapest total first."""
        shape = self.set_parts() if USE_G54_WHOLE_SETS and self.bound else None
        if not shape:
            return []
        state = (len(self.seen_order), len(self.full), self.version, len(self.gone), len(self.unavailable), str(self.budget), sum((len(v) for v in self.views.values())))
        cached = getattr(self, '_sets_memo', None)
        if cached and cached[0] == state:
            return cached[1]
        main, parts = shape
        index = self.fit_index()
        meeting = {n: sorted((i for i in self.item_ids(n) if self.meets_item(i, n) and self.money_ok(n, i)), key=lambda i: (self.price_of(i) or 0.0, i)) for n in range(1, len(self.items) + 1)}
        sets = []
        for device in meeting[main]:
            lines = {device: self.items[main - 1].qty}
            for number in range(1, len(self.items) + 1):
                if number == main:
                    continue
                pick = next((i for i in meeting[number] if i not in lines and (number not in parts or self.fits_device(number, main, i, device, index))), None)
                if not pick:
                    lines = {}
                    break
                lines[pick] = self.items[number - 1].qty
            if not lines:
                continue
            total = sum(((self.price_of(i) or 0.0) * q for i, q in lines.items()))
            if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) and self.brief.get('budget_scope') != 'each' and total > self.budget + 1e-09:
                continue
            sets.append((total, lines))
        sets.sort(key=lambda s: (s[0], sorted(s[1])))
        self._sets_memo = (state, sets)
        return sets

    def is_set(self, lines: dict[str, int]) -> bool:
        held = {i: self.held_units(i) for i in self.cart_ids() if (self.held_units(i) or 0) > 0}
        return held == lines

    def sets_text(self, sets: list[tuple[float, dict[str, int]]]) -> str:
        return '; '.join((' + '.join((f'{i} x{q}' for i, q in lines.items())) + f' = {total:g}' for total, lines in sets[:SET_SHOWN]))

    @_soft('')
    def sets_note(self) -> str:
        """#97 item 54: the status line naming whole sets while the cart is none of them."""
        sets = self.whole_sets()
        if not sets or any((self.is_set(lines) for _, lines in sets)):
            return ''
        return '\n[whole sets read: each line meets its item as its attributes write it and the parts fit the device] ' + self.sets_text(sets) + '. Switch the cart to one of them, then check out.'

    @_soft('')
    def set_hold(self) -> str:
        """#97 item 54: once, at a decline with more than SET_ROOM steps left: name the whole sets read and ask for one."""
        if 'sets' in self.decline_checks or self.left() <= SET_ROOM:
            return ''
        sets = self.whole_sets()
        if not sets:
            return ''
        self.decline_checks.add('sets')
        if USE_G72_LITERAL_HOLD_TOTAL:
            self.decline_checks.add('e49')
        self.trace.append({'kind': 'set_hold', 'sets': [list(l) for _, l in sets[:SET_SHOWN]]})
        return 'Whole sets of listings already read meet every requirement as their attributes write it, and their parts fit the device: ' + self.sets_text(sets) + '. Put one of them in the cart at the requested quantities (the cheapest when the pick is the lowest price) and check out; end without a purchase only if a record clearly fails a requirement.'

    @_soft(False)
    def set_cart(self, lines: dict[str, int]) -> bool:
        """#97 items 54/62/73/74: Cairn sets the cart to exactly `lines` with its own cart calls, when they plus the checkout fit
        the steps left; whether every call went through."""
        cart_tool = self.roles.get('cart')
        if not cart_tool:
            return False
        drop = [i for i in self.cart_ids() if i not in lines]
        calls = self.cart_calls(lines, drop)
        if calls is None or len(calls) + 1 > self.left():
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart_tool), 'args', faults)
            reply = self.act(cart_tool, args) if not faults else None
            if reply is None or self.finished:
                return False
            self.note_cart(cart_tool, args)
            self.observe(cart_tool, reply.get('observation'))
            counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None]
            for value in args.values():
                if counts and isinstance(value, str) and value in self.rows:
                    (self.dropped.add if counts[0] <= 0 or value in drop else self.dropped.discard)(value)
        self.built = [i for i in lines if not self.sold_out(i)]
        return True

    # ---- #97 items 55-74: endings
    @_soft(None)
    def basket_plan(self, lead: str) -> dict[str, int] | None:
        """#97 item 55: one line per item from the review leads (the given lead first): the first unsold lead of each item,
        skipping a lead whose kind is already planned; cart lines not planned, not sold out and of a kind not planned are
        kept, the others dropped. None unless the plan and the kept kinds cover every item and the lead is planned."""
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ident

        def owner(ident: str) -> int:
            for key, ids in self.lead_items.items():
                found = re.search('\\d+', str(key))
                if ident in ids and found and 1 <= int(found.group()) <= len(self.items):
                    return int(found.group())
            return self.line_owner(ident) or (1 if len(self.items) == 1 else 0)
        plan: dict[str, int] = {}
        planned: set[int] = set()
        for ident in [lead] + [i for i in self.leads if i != lead]:
            number = owner(ident)
            if self.sold_out(ident) or not number or number in planned or kind(ident) in {kind(p) for p in plan}:
                continue
            plan[ident] = self.items[number - 1].qty
            planned.add(number)
        kinds = {kind(p) for p in plan}
        keep = {i: self.cart_lines.get(i, 1) for i in self.cart_ids() if i not in plan and not self.sold_out(i) and kind(i) not in kinds}
        if lead not in plan or len(plan) + len({kind(i) for i in keep}) < len(self.items):
            return None
        return {**keep, **plan}

    @_soft(list)
    def own_checkout(self) -> list[str]:
        """#97 items 57/73: Cairn's own checkout: never-shown pages shown first with two or more steps left (E35's view), no
        order with a sold-out or over-budget line, then the checkout through the king's gate (two or more steps left), one
        more try after a refusal; the store's replies."""
        buy = self.roles.get('buy')
        faults: list[str] = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults) if buy else None
        if not buy or faults or args is None:
            return []
        if self.left() >= 2:
            self.offer_views_before_buy()
        blocked = [f for f in self.line_faults() if f.endswith(' is sold out') or ' over the ' in f]
        if blocked:
            self.trace.append({'kind': 'own_checkout_blocked', 'faults': blocked[:3]})
            return []
        replies: list[str] = []
        for _ in range(2):
            if self.finished or self.left() < 1 or any((self.sold_out(i) for i in self.cart_ids())):
                break
            if self.left() >= 2:
                go, said = self.gate('order')
                if said:
                    replies.append(said)
                if not go or self.finished or self.left() < 1:
                    break
            reply = self.act(buy, args)
            self.trace.append({'kind': 'own_checkout', 'reply': _clip(reply, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
            self.observe(buy, reply.get('observation'))
            replies.append(self.show(reply.get('observation')))
            if self.finished:
                break
        return replies

    @_soft('')
    def late_finish(self) -> str:
        """#97 item 57: the first decline with LATE_ORDER_STEPS or fewer steps left: when the unsold lines built cover every
        item, Cairn puts back those missing from the cart (at the owning item's quantity) and checks out itself; on a
        multi-item task with a non-empty short cart, the king's short-cart fill first. '' when it did nothing."""
        cart_tool, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if 'g57' in self.decline_checks or self.late_held or not cart_tool or not buy or self.left() < 1:
            return ''
        self.decline_checks.add('g57')
        lines = [i for i in self.built if not self.sold_out(i)]
        if USE_R16_REFUSED_LINES:
            lines = self.late_lines(lines)
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and not fresh)]
        put: list[str] = []
        if len(lines) >= len(self.items):
            if missing:
                kept = {i: self.cart_lines[i] for i in self.cart_ids() if i in self.cart_lines and i not in missing}  # a whole-cart tool sets every line
                calls = self.cart_calls({**kept, **{i: (self.items[self.line_owner(i) - 1].qty if self.line_owner(i) else 1) for i in missing}}, [])
                if not calls or len(calls) + 1 > self.left():
                    return ''
                for args in calls:
                    faults: list[str] = []
                    args = _shape(args, self.schemas.get(cart_tool), 'args', faults)
                    reply = self.act(cart_tool, args) if not faults else None
                    if reply is None or self.finished:
                        return ''
                    self.note_cart(cart_tool, args)
                    self.observe(cart_tool, reply.get('observation'))
                    self.dropped -= set(missing)
                put = missing
        elif len(self.items) >= 2 and self.cart_ids():
            if not self.fill_short_cart() or self.missing_items() or len([i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]) < len(self.items):
                return ''
        else:
            return ''
        replies = self.own_checkout()
        self.trace.append({'kind': 'late_finish', 'put': put, 'replies': len(replies)})
        if not replies and not put:
            return ''
        return 'Not run (Cairn acted instead): few steps are left and the cart holds, or held, a line for every item; ending without a purchase earns nothing then, so Cairn ' + (f"put {', '.join(put)} back in the cart and " if put else '') + ('checked out. Store reply: ' + ' | then: '.join(replies) if replies else 'could not check out: fix the line the store refuses, then check out.')

    def late_lines(self, lines: list[str]) -> list[str]:
        """USE_R16_REFUSED_LINES: the built lines the late finish may put back: not refused by the cart check, and not of an
        item that already holds another line in the cart."""
        held = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        owned = {self.line_owner(i) for i in held} - {0}
        out = [i for i in lines if i not in self.refused and (i in held or self.line_owner(i) not in owned)]
        if len(out) < len(lines):
            self.trace.append({'kind': 'late_lines_kept_out', 'lines': [i for i in lines if i not in out]})
        return out

    @_soft('')
    def late_short_review(self) -> str:
        """#97 item 58: a decline with two to LATE_ORDER_STEPS steps left, no confirmed lead, fewer than two review blocks used
        and fewer unsold built lines than items: both review blocks marked used; with no leads and no review yet, the king's
        review (clock at or under LATE_REVIEW_BY); the first unsold lead's planned basket (#55) checked under the higher spend
        limit when the cart calls fit; a pass confirms the lead and complete_cart sets the cart and orders."""
        if not 2 <= self.left() <= LATE_ORDER_STEPS or self.confirmed or self.refuse_blocks >= 2:
            return ''
        if len([i for i in self.built if not self.sold_out(i)]) >= len(self.items):
            return ''
        self.refuse_blocks = 2
        if not self.leads and not self.audited and self.clock() <= LATE_REVIEW_BY and self.judges():
            self.audited = True
            named = self.audit_fits(self.audit_call())
            if named:
                self.lead_items = named
                self.leads = [i for ids in named.values() for i in ids]
        lead = next((i for i in self.leads if not self.sold_out(i)), None)
        basket = self.basket_plan(lead) if lead else None
        if not basket or self.clock() > LATE_REVIEW_BY:
            return ''
        calls = self.cart_calls(basket, [i for i in self.cart_ids() if i not in basket])
        if calls is None or len(calls) > self.left() - 1:
            return ''
        self.check_cart(list(basket), SPEND_HARD)
        self.trace.append({'kind': 'late_review', 'lead': lead, 'basket': list(basket), 'ok': self.verdicts.get(self.check_key(list(basket)))})
        if not self.verdicts.get(self.check_key(list(basket))):
            return ''
        self.confirmed = lead
        return self.complete_cart()

    @_soft(False)
    def forced_basket(self) -> bool:
        """#97 item 62: at the forced ending on an empty cart (no pointer requirement, no fit-bound item): the cheapest read,
        buyable, priced listing establishing each item (#17), inside a total budget, carted by Cairn."""
        if not self.roles.get('cart') or not self.roles.get('buy') or self.refs or self.bound:
            return False
        lines: dict[str, int] = {}
        for number in range(1, len(self.items) + 1):
            needs = self.need_list(number)
            pool = sorted((i for i in self.item_ids(number) or (list(self.full) if len(self.items) == 1 else []) if i in self.full and i not in lines and self.buyable(i) and self.price_of(i) is not None), key=lambda i: (self.price_of(i) or 0.0, i)) if needs is not None else []
            pick = next((i for i in pool if self.established(i, number, needs or []) is not None), None)
            if not pick:
                return False
            lines[pick] = self.items[number - 1].qty
        if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) and self.brief.get('budget_scope') != 'each' and sum(((self.price_of(i) or 0.0) * q for i, q in lines.items())) > self.budget + 1e-09:
            return False
        done = self.set_cart(lines)
        self.trace.append({'kind': 'forced_basket', 'lines': lines, 'done': done})
        return done

    def fit_state(self, part: int, main: int, piece: str, device: str, index: dict[str, Any]) -> bool | None:
        """#97 items 73/74: True when the rows show the part works with the device (series link or a model code its fit clause
        names), False when they show it does not, None when they say nothing."""
        if self.fits_device(part, main, piece, device, index):
            return True
        system = self.link_system(part, main, index)
        if system:
            carried, namer = (piece, device) if system[0] == part else (device, piece)
            if self.link_holds(carried, namer, system[2], index) is False:
                return False
        codes = (index['marks'].get(device) or {}).get('codes') or []
        fits = self.fit_set(piece, index)
        bars = (index['marks'].get(piece) or {}).get('bars') or []
        if codes and (fits and not any((_covers_code(p, c) for p in fits for c in codes)) or any((_code_hit(b, c) for b in bars for c in codes))):
            return False
        return None

    def judged(self, ident: str, number: int) -> tuple[int, int]:
        """#97 item 73: (requirements met, requirements code can judge) of a read listing (E49's literal reading)."""
        met = total = 0
        for need in self.item_needs(number):
            state, _ = self.literal_state(ident, number, need)
            if state != 'unknown':
                total += 1
                met += state == 'met'
        return (met, total)

    @_soft(None)
    def best_basket(self, repair: bool=False) -> dict[str, int] | None:
        """#97 items 73/74: the best order among the listings read. Each item keeps its current buyable line (with `repair`,
        a part line the rows show misfits the device line is replaced when a read listing works with it). An item without a
        line takes the read, buyable, priced listing of its label, not gone or used, not shown to misfit the device line,
        meeting one requirement code can judge (or with none to judge), ranked: works with the chosen device and parts, once
        held, a review lead, more requirements met, lower price; the first keeping the order inside the budget. Device items
        first. None when an item has no candidate."""
        index = self.fit_index()
        cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and not self.sold_out(i)]
        mains = sorted({m for _, m in self.bound})
        order = mains + [n for n in range(1, len(self.items) + 1) if n not in mains]
        chosen: dict[int, str] = {}
        for number in order:
            mine = [i for i in cart if self.line_items(i) == [number]]
            if mine:
                chosen[number] = mine[0]
        popped: dict[int, str] = {}
        self.repaired = []
        if repair:
            for part, main in self.bound:
                piece, device = (chosen.get(part), chosen.get(main))
                if piece and device and self.fit_state(part, main, piece, device, index) is False:
                    popped[part] = chosen.pop(part)
        lines = {i: self.items[n - 1].qty for n, i in chosen.items()}
        for number in order:
            if number in chosen:
                continue
            spent = sum(((self.price_of(i) or 0.0) * q for i, q in lines.items()))
            ranked = []
            for ident in self.item_ids(number):
                if ident not in self.full or ident in lines or ident in self.gone or not self.buyable(ident) or self.price_of(ident) is None:
                    continue
                works = True
                for part, main in self.bound:
                    other = main if number == part else part if number == main else 0
                    partner = chosen.get(other) if other else None
                    if not partner:
                        continue
                    piece, device = (ident, partner) if number == part else (partner, ident)
                    state = self.fit_state(part, main, piece, device, index)
                    if state is False:
                        works = None
                        break
                    works = works and state is True
                if works is None or (repair and number in {p for p, _ in self.bound} and works is not True):
                    continue
                met, total = self.judged(ident, number)
                if total and not met:
                    continue
                refused = USE_R16_REFUSED_LINES and ident in self.refused  # refused by the cart check: last
                ranked.append(((refused, not works, ident not in self.ever_held, ident not in self.leads, -met, self.price_of(ident) or 0.0, ident), ident))
            pick = None
            for _, ident in sorted(ranked):
                cost = (self.price_of(ident) or 0.0) * self.items[number - 1].qty
                if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool):
                    if self.brief.get('budget_scope') == 'each' and (self.price_of(ident) or 0.0) > self.budget + 1e-09:
                        continue
                    if self.brief.get('budget_scope') != 'each' and spent + cost > self.budget + 1e-09:
                        continue
                pick = ident
                break
            if not pick and number in popped:  # #97 item 74: no listing shown to work with the device: the line stays
                pick = popped.pop(number)
            elif not pick:
                return None
            elif number in popped:
                self.repaired.append((popped[number], pick))
            chosen[number] = pick
            lines[pick] = self.items[number - 1].qty
        return lines

    @_soft('')
    def clock_order(self) -> str:
        """#97 item 73: late on the clock (LATE_ORDER_ROOM model seconds or less), on a request for two or more items whose
        cart holds or once held lines, with cart and buy tools and store labels: the decline becomes the best order read,
        set and checked out by Cairn."""
        if not USE_G73_CLOCK_BEST_ORDER or 'g73' in self.decline_checks or len(self.items) < 2 or self.model_room() > LATE_ORDER_ROOM:
            return ''
        if not self.roles.get('cart') or not self.roles.get('buy') or not self.category_key() or not (self.cart_ids() or self.ever_held):
            return ''
        self.decline_checks.add('g73')
        lines = self.best_basket()
        if not lines or not self.set_cart(lines):
            return ''
        replies = self.own_checkout()
        self.trace.append({'kind': 'clock_order', 'lines': lines, 'replies': len(replies)})
        return 'Not run (Cairn acted instead): time is nearly up, and running out of time does not show that nothing fits, so Cairn set the cart to the best order among the listings read (' + ', '.join((f'{i} x{q}' for i, q in lines.items())) + ') and checked out. Cart now: ' + self.cart_text() + ('. Store reply: ' + ' | then: '.join(replies) if replies else '. The checkout did not run: fix the cart and check out.')

    @_soft(False)
    def forced_set(self) -> bool:
        """#97 item 54 (forced finish): with two or more steps left, when the cart is not held, holds a fit fault or misses an
        item line and is not already a whole set, Cairn sets the cart to the cheapest whole set read."""
        sets = self.whole_sets()
        if not sets or self.left() < 2 or any((self.is_set(lines) for _, lines in sets)):
            return False
        held = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if held and not self.fit_faults() and not self.missing_items() and len(held) >= len(self.items):
            return False
        done = self.set_cart(sets[0][1])
        self.trace.append({'kind': 'forced_set', 'lines': sets[0][1], 'done': done})
        return done

    @_soft(False)
    def fit_repair(self) -> bool:
        """#97 item 74: at the forced ending of a fit task with a whole cart, three or more steps left and fit faults: a part
        line the rows show misfits the device line is swapped for the best read listing shown to work with it (#73's
        ranking); none: the line stays. Never adds a line for an item that has one."""
        if not self.bound or self.left() < 3 or not self.fit_faults():
            return False
        planned = self.best_basket(repair=True)
        if not planned or not self.repaired:
            return False
        lines = {i: q for i, q in self.cart_lines.items() if i in self.cart_ids() and i not in {old for old, _ in self.repaired}}
        lines.update({new: planned[new] for _, new in self.repaired})
        done = self.set_cart(lines)
        self.trace.append({'kind': 'fit_repair', 'lines': lines, 'done': done})
        return done

    # ---- stage 3: #110 items 1, 6-11 (nearest order, set check, proven sets, holds), #104 items 2/4
    def read_nearest(self, got: dict[str, Any]) -> bool:
        """#110 item 1 / #111 E52: keep what the light nearest-rule reading found: a rule only when it quotes a sentence of
        NEAREST_SENTENCE or more characters that stands in the request."""
        sentence = str(got.get('sentence') or '').strip().strip('"').strip()
        said = got.get('nearest') is True or str(got.get('nearest')).strip().lower() in ('true', 'yes')
        valid = said and len(sentence) >= NEAREST_SENTENCE and _plain_text(sentence) in _plain_text(self.request)
        self.nearest_read = bool(valid)
        self.nearest_sentence = sentence if valid else ''
        return self.nearest_read

    def nearest_wanted(self, fetch: bool) -> bool:
        """#110 item 1: whether the request asks for the nearest listing when nothing meets everything, by E52's light reading
        of the request (made once, here when `fetch` and nothing was read yet and the time allows); no word list."""
        if self.nearest_read is None and fetch and not self.no_credit and self.calls < ALL_CALLS and self.model_room() >= NEAREST_SECONDS:
            got = self.ask_json(NEAREST_RULE + '\n\nREQUEST:\n' + self.request, [LIGHT], 800) or {}
            self.trace.append({'kind': 'nearest_rule', 'got': _clip(got, 200)})
            self.read_nearest(got)
        return self.nearest_read is True

    def row_price(self, ident: str) -> float | None:
        """#110 items 1/2: the learned price key's value, else the listing's own price field."""
        price = self.price_of(ident)
        return price if price is not None else self.own_price(ident)

    def within_budget(self, lines: dict[str, int]) -> bool:
        """#110 item 1: every price known and the order inside the budget (each unit inside a per-item budget)."""
        if not isinstance(self.budget, (int, float)) or isinstance(self.budget, bool):
            return True
        prices = {i: self.row_price(i) for i in lines}
        if any((p is None for p in prices.values())):
            return False
        if self.brief.get('budget_scope') == 'each':
            return all(((prices[i] or 0.0) <= self.budget + 1e-09 for i in lines))
        return sum(((prices[i] or 0.0) * n for i, n in lines.items())) <= self.budget + 1e-09

    @_soft(None)
    def nearest_set(self) -> dict[str, int] | None:
        """#110 item 1: one heavy pick of the nearest listing per item among the listings read of its label (opened first,
        cheapest first, NEAREST_RECORDS in all); None when it names none or the order breaks the budget."""
        pool: list[tuple[int, str]] = []
        share = max(NEAREST_EACH, NEAREST_RECORDS // max(1, len(self.items)))
        for number in range(1, len(self.items) + 1):
            ids = [i for i in self.item_ids(number) if not self.sold_out(i) and i not in self.gone and self.buyable(i)]
            ids.sort(key=lambda i: (USE_R16_REFUSED_LINES and i in self.refused, i not in self.full, self.row_price(i) if self.row_price(i) is not None else float('inf'), i))
            pool += [(number, i) for i in ids[:share]]
        if not pool:
            return None
        shown = '\n'.join((f'item {n} - {i}: ' + self.record_of(i)[:NEAREST_ROW_CHARS] for n, i in pool[:NEAREST_RECORDS]))
        items = '\n'.join((f'item {n}: {it.name}, quantity {it.qty}' for n, it in enumerate(self.items, 1)))
        got = self.ask_json(NEAREST_PICK + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + items + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger() + '\n\nLISTINGS (by item):\n' + shown, [HEAVY, LIGHT], 1500) or {}
        known = {i for _, i in pool[:NEAREST_RECORDS]}
        pick: dict[int, str] = {}
        misses: list[str] = []
        for entry in _items(got.get('lines')):
            if not isinstance(entry, dict):
                continue
            number = _count(entry.get('item'))
            ident = str(entry.get('id') or '').strip()
            if not number or not 1 <= number <= len(self.items) or ident not in known or number in pick or ident in pick.values():
                continue
            pick[number] = ident
            misses += [f'{ident}: {m}' for m in (str(x).strip() for x in _items(entry.get('misses'))) if m]
        self.trace.append({'kind': 'nearest_pick', 'lines': dict(pick), 'misses': misses[:6]})
        if not pick:
            return None
        lines = {ident: self.items[n - 1].qty for n, ident in pick.items()}
        if not self.within_budget(lines):
            return None
        self.near_misses = misses[:4]
        return lines

    @_soft(False)
    def near_cart(self) -> bool:
        """#110 item 1: on a nearest-rule request about to end without a purchase: True when the cart already holds a line
        for every item; else (NEAREST_RESCUES times at most) Cairn carts the nearest listing per item, by the heavy pick when
        the time allows, else by #97 item 73's best order among the listings read."""
        if not self.roles.get('cart') or not self.items:
            return False
        held = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and not self.sold_out(i) and i not in self.gone]
        if held and not self.missing_items() and len(held) >= len(self.items):
            return True
        if self.near_rescues >= NEAREST_RESCUES or self.left() < 2:
            return False
        self.near_rescues += 1
        lines = self.nearest_set() if not self.no_credit and self.model_room() >= NEAREST_PICK_SECONDS else None
        if not lines:
            lines = self.best_basket()
            if lines and not self.within_budget(lines):
                lines = None
        if not lines:
            return False
        done = self.set_cart(lines)
        self.near_made = self.near_made or done
        self.trace.append({'kind': 'near_cart', 'lines': lines, 'done': done})
        return done

    def free_text_args(self, args: dict[str, Any], text: str, always: bool) -> dict[str, Any]:
        """#110 items 1/7: the checkout's fixed arguments with `text` in its first free text field (a required one always,
        an optional one named for free text when `always`), cut to the field's length."""
        buy = self.roles.get('buy') or ''
        schema = self.schemas.get(buy) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        free = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and k not in args and (k in required or (always and re.search(TEXT_FIELD, _snake(k), re.I)))]
        if free and text:
            room = (props[free[0]] or {}).get('maxLength')
            args = {**args, free[0]: _cut_words(text, room) if isinstance(room, int) and room > 0 else text[:300]}
        return args

    def near_args(self) -> dict[str, Any]:
        """#110 item 1: the checkout of a nearest order: the fixed arguments, a free text field saying that no listing meets
        every requirement (with what the heavy pick says the order misses)."""
        text = 'No listing meets every requirement, so this order holds the nearest listings' + ('. Missed: ' + '; '.join(self.near_misses) if self.near_misses else '.')
        return self.free_text_args(dict(self.brief.get('buy_args') or {}), text, bool(self.near_misses))

    # #110 items 6-10: the set check (lenient) and the proven set (strict), both on E49's literal reading
    def judged_needs(self, number: int) -> list[str]:
        """#110 items 6-10: the item's live requirements these checks read (conditional rules and requirements only the
        shopper's words state left out)."""
        return [need for need in self.item_needs(number) if not self.described(number, need)]

    def listing_gaps(self, ident: str, number: int | None=None) -> list[str]:
        """#110 items 6/8/10: the requirements of its item an opened listing does not meet as its own attributes write them
        (another value, or no key and no word for the property); [] for a listing not opened or of no single item."""
        number = number if number is not None else self.line_owner(ident)
        if ident not in self.full or not number:
            return []
        return [need for need in self.judged_needs(number) if self.literal_state(ident, number, need)[0] in ('differs', 'missing')]

    def cart_gaps(self) -> list[str]:
        """#110 items 6/8/10: the held cart lines' gaps, as "<line> does not state / contradicts <requirement>"."""
        out = []
        for ident in self.cart_ids():
            number = self.line_owner(ident)
            if (self.held_units(ident) or 0) <= 0 or ident not in self.full or not number:
                continue
            for need in self.judged_needs(number):
                state = self.literal_state(ident, number, need)[0]
                if state in ('differs', 'missing'):
                    out.append(f"{ident} {'contradicts' if state == 'differs' else 'does not state'} {need}")
        return out

    @staticmethod
    def combos(heads: list[list[Any]], cap: int) -> list[tuple[Any, ...]]:
        """#110 items 6-8: the first `cap` combinations of one entry per list (all of them when fewer)."""
        out: list[tuple[Any, ...]] = [()]
        for head in heads:
            out = [c + (e,) for c in out for e in head][:cap]
        return out

    def set_pool(self, number: int) -> list[tuple[int, float, str, list[str]]]:
        """#110 item 6: the item's opened, buyable listings inside its money that contradict none of its requirements as
        written: (requirements not stated, price, id, their properties), fewest unstated first, then cheapest."""
        needs = self.judged_needs(number)
        pool = []
        for ident in self.item_ids(number):
            if ident not in self.full or ident in self.gone or not self.buyable(ident) or not self.within_money(number, ident):
                continue
            states = [(need, self.literal_state(ident, number, need)[0]) for need in needs]
            if any((state == 'differs' for _, state in states)):
                continue
            unstated = [need.split(':', 1)[0].strip() for need, state in states if state != 'met']
            pool.append((len(unstated), self.price_of(ident) or 0.0, ident, unstated))
        return sorted(pool)[:SET_ITEMS * 3]

    @_soft(None)
    def set_check(self) -> tuple[int, float, dict[str, int], str] | None:
        """#110 item 6: the opened set (one line per item, each bound part not shown to misfit its device) with the fewest
        unstated requirements, then the lowest total inside the budget: (unstated, total, lines, text); None when an item
        has no such listing. Kept by state."""
        state = (len(self.seen_order), len(self.full), self.version, len(self.gone), len(self.unavailable), str(self.budget), len(self.ref_values), sum((len(v) for v in self.views.values())))
        if self.set_memo and self.set_memo[0] == state:
            return self.set_memo[1]
        self.set_memo = (state, None)
        pools = {}
        for number in range(1, len(self.items) + 1):
            pools[number] = self.set_pool(number)
            if not pools[number]:
                return None
        index = self.fit_index() if self.bound else None
        mains = sorted({m for _, m in self.bound})
        parts = {p: m for p, m in self.bound}
        best: tuple[tuple[int, float], dict[int, tuple[int, float, str, list[str]]]] | None = None
        for combo in self.combos([pools[m][:SET_ITEMS] for m in mains], SET_COMBOS):
            chosen = dict(zip(mains, combo))
            picks: dict[int, tuple[int, float, str, list[str]]] = {}
            for number in range(1, len(self.items) + 1):
                used = {e[2] for e in picks.values()}
                if number in chosen:
                    entry = chosen[number] if chosen[number][2] not in used else None
                else:
                    main = parts.get(number)
                    pool = [e for e in pools[number] if e[2] not in used and (main is None or main not in chosen or self.fit_state(number, main, e[2], chosen[main][2], index) is not False)]
                    entry = pool[0] if pool else None
                if entry is None:
                    picks = {}
                    break
                picks[number] = entry
            if not picks:
                continue
            total = sum((e[1] * self.items[n - 1].qty for n, e in picks.items()))
            if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) and self.brief.get('budget_scope') != 'each' and total > self.budget + 1e-09:
                continue
            key = (sum((e[0] for e in picks.values())), total)
            if best is None or key < best[0]:
                best = (key, picks)
        if not best:
            return None
        (unstated, total), picks = best
        lines = {e[2]: self.items[n - 1].qty for n, e in picks.items()}
        text = ' + '.join((f'item {n}: {e[2]} at {e[1]:g}' + (f' x{self.items[n - 1].qty}' if self.items[n - 1].qty > 1 else '') for n, e in sorted(picks.items())))
        unsure = '; '.join((f"item {n} {', '.join(e[3])}" for n, e in sorted(picks.items()) if e[3]))
        text += f' = {total:g}' + (f'; not stated on their pages yet: {unsure}' if unsure else '; each requirement this check reads is stated')
        self.set_memo = (state, (unstated, total, lines, text))
        return self.set_memo[1]

    def set_line(self) -> str:
        """#110 item 6: the status line of the set check (cheapest picks, nothing ordered yet)."""
        best = self.set_check()
        if not best:
            return ''
        return '[set check, listings opened so far only (unopened ones may be cheaper): the lowest total whose attributes contradict no requirement' + (' (no part shown to misfit its device)' if self.bound else '') + '] ' + best[3]

    def cart_total(self) -> float | None:
        prices = [(self.price_of(i), self.held_units(i) or 0) for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        return None if not prices or any((p is None for p, _ in prices)) else sum(((p or 0.0) * u for p, u in prices))

    @_soft('')
    def set_check_hold(self) -> str:
        """#110 item 6: once per cart state, no step used: an opened set states every requirement this check reads and differs
        from the cart, while a cart line contradicts or does not state one, or, on a cheapest pick, the set costs less."""
        if self.left() <= self.tail() + len(self.items) + 2:
            return ''
        best = self.set_check()
        mark = self.cart_mark()
        if not best or best[0] != 0 or mark in self.set_held:
            return ''
        held = {i for i in self.cart_ids() if (self.held_units(i) or 0) > 0}
        if set(best[2]) == held:
            return ''
        total = self.cart_total()
        gaps = self.cart_gaps()
        dearer = self.brief.get('pick') == 'cheapest' and total is not None and total > best[1] + 1e-09
        if not gaps and not dearer:
            return ''
        self.set_held.add(mark)
        self.trace.append({'kind': 'set_check_hold', 'cart': total, 'set': best[1], 'gaps': gaps[:4]})
        why = ('; '.join(gaps[:4]) + '. ') if gaps else ''
        cost = f" and cost {best[1]:g} in all, less than the cart's {total:g}" if dearer else ''
        return f'Not run yet (no step used). {why}Of the listings opened so far, these state every requirement{cost}: {best[3]}. Put them in the cart instead, unless one of them fails something this check cannot read (a compatibility note, a condition, a value the page words differently); then check out.'

    def names_other_item(self, number: int, value: str) -> bool:
        """#110 item 9: whether a requirement's value names another requested item (all of its name's words)."""
        have = _roots(value)
        return any((n != number and _roots(item.name) and _roots(item.name) <= have for n, item in enumerate(self.items, 1)))

    def strict_ok(self, ident: str, number: int) -> bool:
        """#110 items 7/9: an opened, buyable listing inside its money proves its item strictly: the item has a live
        requirement and no conditional one; each requirement is met as its own attributes write it (E49's reading: a choice
        of an either-or value, a fit code by its fit clause, word forms; a pointer only by its read value), a requirement of
        a fit-bound item that names another requested item being left to the fit check; and the item name's words its store
        label does not cover appear in its title, attributes or label."""
        if ident not in self.full or ident in self.gone or not self.buyable(ident) or not self.within_money(number, ident):
            return False
        item = self.items[number - 1]
        needs = self.item_needs(number)
        if not needs or any((_conditional(need) for n, need in self.needs_now() if n == number)):
            return False
        bound = any((number in pair for pair in self.bound))
        for need in needs:
            if bound and self.names_other_item(number, need.partition(':')[2]):
                continue
            if self.literal_state(ident, number, need)[0] != 'met':
                return False
        label = self.label_of(ident)
        extra = (_roots(item.name) | _roots(item.shelf or '')) - _roots(label)
        title = self.title_key()
        said = _roots(label) | _roots(str((self.rows.get(ident) or {}).get(title) or '') if title else '') | _roots(' '.join((str(v) for v in (_fact_block(self.full.get(ident) or {}) or {}).values() if not isinstance(v, (dict, list)))))
        return not extra or extra <= said

    @_soft(None)
    def proven_set(self) -> tuple[float, dict[str, int], str] | None:
        """#110 items 7/8: the cheapest whole set of strictly proven listings (one per item, distinct; each bound part fitting
        its device by a code or series its rows name) inside the budget: (total, lines, text); None when an item has none."""
        strict = {}
        for number in range(1, len(self.items) + 1):
            strict[number] = sorted((i for i in self.item_ids(number) if self.strict_ok(i, number)), key=lambda i: (self.price_of(i) or 0.0, i))
            if not strict[number]:
                return None
        index = self.fit_index() if self.bound else None
        mains = sorted({m for _, m in self.bound})
        parts = {p: m for p, m in self.bound}
        best: tuple[float, dict[int, str]] | None = None
        for combo in self.combos([strict[m][:SET_ITEMS] for m in mains], SET_COMBOS):
            chosen = dict(zip(mains, combo))
            if len(set(chosen.values())) < len(chosen):
                continue
            picks = dict(chosen)
            for number in range(1, len(self.items) + 1):
                if number in picks:
                    continue
                main = parts.get(number)
                pick = next((i for i in strict[number] if i not in picks.values() and (main is None or main not in chosen or self.fits_device(number, main, i, chosen[main], index))), None)
                if not pick:
                    picks = {}
                    break
                picks[number] = pick
            if not picks:
                continue
            total = sum(((self.price_of(i) or 0.0) * self.items[n - 1].qty for n, i in picks.items()))
            if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) and self.brief.get('budget_scope') != 'each' and total > self.budget + 1e-09:
                continue
            if best is None or total < best[0]:
                best = (total, picks)
        if not best:
            return None
        total, picks = best
        lines = {i: self.items[n - 1].qty for n, i in picks.items()}
        text = ' + '.join((f'item {n}: {i} at {self.price_of(i) or 0.0:g}' + (f' x{self.items[n - 1].qty}' if self.items[n - 1].qty > 1 else '') for n, i in sorted(picks.items())))
        return (total, lines, text)

    @_soft(False)
    def check_rejected(self, lines: list[str]) -> bool:
        """Whether the final check judged exactly this set of listings and found it wrong (under any state version)."""
        prefix = '|'.join(sorted(lines)) + '#'
        return any((key.startswith(prefix) and ok is False for key, ok in self.verdicts.items()))

    def proven_try(self) -> bool:
        """#110 item 7: at the top of a model turn, on an "any" pick with nothing ordered, the interview done or impossible,
        no pointer value unread and room left (twice an episode at most, once per state): when a strictly proven set exists,
        Cairn sets the cart to it and checks out through the model's own checkout path, so every check before an order
        still runs. True when Cairn acted."""
        if self.proven_tries >= PROVEN_TRIES or self.finished or self.buys or self.brief.get('pick') in ('cheapest', 'most', 'least'):
            return False
        if not self.roles.get('cart') or not self.roles.get('buy') or not self.chat or (self.roles.get('ask') and not self.asked and self.msgs_free() > 0):
            return False
        if self.left() <= self.tail() + len(self.items) + 3 or self.model_room() < PROVEN_SECONDS or self.pending_refs():
            return False
        mark = (len(self.full), len(self.ref_values), self.version, len(self.notes), str(self.budget))
        if mark == self.proven_mark:
            return False
        self.proven_mark = mark
        best = self.proven_set()
        if not best:
            return False
        total, lines, text = best
        if USE_T_PROVEN_RESPECTS_CHECK and self.check_rejected(lines):
            self.trace.append({'kind': 'proven_order_skipped', 'lines': lines, 'why': 'the final check rejected this set'})
            return False
        self.proven_tries += 1
        held = {i for i in self.cart_ids() if (self.held_units(i) or 0) > 0}
        if set(lines) != held and not self.set_cart(lines):
            self.trace.append({'kind': 'proven_order', 'lines': lines, 'done': False})
            self.chat.append({'role': 'user', 'content': '[Cairn] Opened listings state every requirement (' + text + '); Cairn began to set the cart to them, but a cart call did not go through. Cart now: ' + self.cart_text() + '\n' + self.status()})
            return True
        if self.finished:
            return True
        buy = self.roles['buy']
        args = self.free_text_args(dict(self.brief.get('buy_args') or {}), 'Each line states every requirement on its page: ' + text, False)
        call = {'id': f'proven{self.proven_tries}', 'type': 'function', 'function': {'name': buy, 'arguments': _dump(args)}}
        self.chat.append({'role': 'assistant', 'content': f"[Cairn] Cairn's code found that these opened listings state every requirement as their own attributes write it: {text} = {total:g}. It set the cart to them and checks out.", 'tool_calls': [call]})
        self.trace.append({'kind': 'proven_order', 'lines': lines, 'total': total})
        replies = self.run_calls([call])
        for index, (cid, reply) in enumerate(replies):
            if index == len(replies) - 1:
                reply += '\n' + self.status()
            self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': reply})
        return True

    @_soft('')
    def proven_swap(self) -> str:
        """#110 item 8: at the model's checkout (PROVEN_SWAPS times at most, never on most/least picks, every pointer value
        read): when a strictly proven set differs from the cart and a cart line has a gap, Cairn sets the cart to that set;
        its text, '' when nothing changed. A cart that only costs more on a cheapest pick is #97 items 49/52/53's case (a
        cheaper line named, then swapped); with those off, this swap takes it too."""
        pick = self.brief.get('pick')
        if pick in ('most', 'least') or self.proven_swaps >= PROVEN_SWAPS or self.left() <= self.tail() + len(self.items) + 2 or self.pending_refs():
            return ''
        best = self.proven_set()
        if not best:
            return ''
        total, lines, text = best
        held = {i for i in self.cart_ids() if (self.held_units(i) or 0) > 0}
        if set(lines) == held:
            return ''
        cost = self.cart_total()
        dearer = pick == 'cheapest' and cost is not None and cost > total + 1e-09 and not (USE_G49_CHEAPER_HOLD and USE_G52_AGREED_SWAP)
        if not self.cart_gaps() and not dearer:
            return ''
        if USE_T_PROVEN_RESPECTS_CHECK and self.check_rejected(lines):
            return ''
        self.proven_swaps += 1
        if not self.set_cart(lines):
            return ''
        self.trace.append({'kind': 'proven_swap', 'from': sorted(held), 'to': lines})
        return f'{text} = {total:g}'

    @_soft('')
    def settle_hold(self) -> str:
        """#110 item 11: once, on a request the reading finds a nearest-listing rule in, with more than SETTLE_STEPS steps (or
        a quarter of the step limit) and SETTLE_SECONDS left: a checkout of a cart line whose own attribute states another
        value than a requirement is held, asking for a real search first."""
        if self.settle_held or self.left() <= max(SETTLE_STEPS, self.max_steps // 4) or self.model_room() <= SETTLE_SECONDS:
            return ''
        lines = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and i in self.full and self.line_owner(i) and any((self.literal_state(i, self.line_owner(i), need)[0] == 'differs' for need in self.judged_needs(self.line_owner(i))))]
        if not lines or not self.nearest_wanted(True):
            return ''
        self.settle_held = True
        self.trace.append({'kind': 'settle_hold', 'lines': lines, 'left': self.left()})
        return SETTLE_HOLD.format(lines=', '.join(lines[:3]), left=self.left())

    @_soft('')
    def needless_decline(self) -> str:
        """#104 item 4: once, at a decline: when the cart holds a built, unsold line for every requested item, ending without
        a purchase is held."""
        if 'r104' in self.decline_checks:
            return ''
        cart = set(self.cart_ids())
        lines = [i for i in self.built if i in cart and not self.sold_out(i) and i not in self.gone and (self.held_units(i) or 0) > 0]
        covered = {n for i in lines for n in self.line_items(i)}
        if not lines or len(covered) < len(self.items):
            return ''
        self.decline_checks.add('r104')
        self.trace.append({'kind': 'needless_decline_held', 'lines': lines})
        return 'The cart holds a line you built for every requested item (' + ', '.join(lines) + '), and ending without a purchase earns nothing while the store sells a fitting listing. Check out this cart (fix a line first if its record breaks a requirement); end without a purchase only if your search showed that some requested item has no listing that meets its requirements, and then say which in your text.'

    @_soft('')
    def line_pick(self) -> str:
        """#104 item 2: once an episode, at a checkout with time and spend room: one strong-model call picks, for each cart
        line, the listing the shopper should get among the line and up to PICK_ROWS read listings of its label (cheapest
        first; on an "any" pick only for a line with a gap, against listings with fewer). A differing pick holds the
        checkout when it is inside the item's money and the budget, and, unless the pick rule is cheapest, most or least,
        it meets more of the line's requirements as its attributes write them."""
        if self.picked_lines or self.no_credit or self.model_room() < PICK_SECONDS or HEAVY not in self.judges() or self.left() <= self.tail() + HOLD_ROOM:
            return ''
        cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        ranked = self.brief.get('pick') in ('cheapest', 'most', 'least')
        candidates: dict[str, list[str]] = {}
        for line in cart:
            label = self.label_of(line)
            number = self.line_owner(line)
            pool = [i for i in self.full if label and i not in cart and self.label_of(i) == label and self.buyable(i) and i not in self.gone]
            if not ranked and not USE_T_R104_2_EVERY_PICK:  # an "any" pick: only a line with a gap, against listings with fewer
                gaps = len(self.listing_gaps(line, number)) if number else 0
                pool = [i for i in pool if gaps and len(self.listing_gaps(i, number)) < gaps]
            pool.sort(key=lambda i: (self.price_of(i) is None, self.price_of(i) or 0.0, i))
            if pool:
                candidates[line] = [line] + pool[:PICK_ROWS]
        if not candidates:
            return ''
        self.picked_lines = True
        shown = '\n\n'.join((f'CART LINE {line}, candidates:\n' + '\n'.join((f'{i}: ' + self.record_of(i)[:900] for i in ids)) for line, ids in candidates.items()))
        got = self.ask_json(LINE_PICK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger() + '\n\nCART:\n' + self.cart_text() + '\n\n' + shown, [HEAVY], 1500) or {}
        swaps = []
        for entry in _items(got.get('lines')):
            if not isinstance(entry, dict):
                continue
            line, pick = (str(entry.get('line') or '').strip(), str(entry.get('pick') or '').strip())
            if line not in candidates or pick not in candidates[line] or pick == line:
                continue
            number = self.line_owner(line)
            if not number or not self.money_ok(number, pick):
                continue
            if not ranked and ((not self.listing_gaps(line, number)) if USE_T_R104_2_EVERY_PICK else len(self.listing_gaps(pick, number)) >= len(self.listing_gaps(line, number))):
                continue  # an "any" pick never swaps a line the code proves fits (every pick rule: the strong model judges the others)
            new = {i: (self.held_units(i) or 1) for i in cart if i != line}
            new[pick] = self.held_units(line) or self.items[number - 1].qty
            if not self.within_budget(new):
                continue
            swaps.append((line, pick, str(entry.get('why') or '').strip()[:200]))
        self.trace.append({'kind': 'line_pick', 'swaps': [(a, b) for a, b, _ in swaps]})
        if not swaps:
            return ''
        said = '; '.join((f'{pick} instead of {line}' + (f' ({why})' if why else '') for line, pick, why in swaps))
        return 'Not run yet (no step used). A second look at the listings read picks ' + said + '. Swap those lines (take the old line out, add the new one at the requested quantity), then check out; keep a line only if the named record clearly fails a requirement.'

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        turn_lines = dict(self.cart_lines)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            self.left_at_call = self.left()
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
            cuts: list[tuple[str, int, str]] = []
            args = _shape(args, self.schemas[name], 'args', faults, cuts)
            if faults:
                out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6])))
                continue
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            viewed = ''
            if USE_G42_CHAIN_PAGE_VIEW and name == self.roles.get('open'):
                swapped = self.chain_page_view(name, args, allowed)
                if swapped:
                    name, args, viewed = swapped
            counted = ''
            if name == self.roles.get('cart'):
                args = self.whole_removal(name, args)
                if USE_E22_ZERO_COUNT_ADD:
                    args, counted = self.zero_count_add(name, args)
                if USE_E23_LATE_KEEP_LINES:
                    kept = self.keeps_lines(name, args, calls[index + 1:], turn_lines)
                    if kept:
                        out.append((cid, kept + ' Cart now: ' + self.cart_text()))
                        continue
            trimmed = counted + viewed
            for where, limit, value in cuts:
                field = where.split('.', 1)[-1]
                if (name, field) not in self.cut_noted:
                    self.cut_noted.add((name, field))
                    trimmed += f'\n[Cairn] {field} takes at most {limit} characters, so the store read {value!r}. Keep later values of {field} within {limit} characters, the most telling words first.'
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
            elif USE_E19_SKIP_SEEN_OFFSETS and not fresh and name not in plain_jobs and not ending:
                args, skipped = self.skip_seen_offset(name, args)
                trimmed += skipped
            args = self.row_ids(name, args)
            if USE_E21_DROP_DEAD_FIT_CODES and self.fit_codes_dead and not ending and name in self.code_tools():
                args, dropped, empty = self.drop_codes(name, args)
                if empty:
                    out.append((cid, dropped))
                    continue
                trimmed += dropped
            if name == self.roles.get('search') and self.codes_dead and self.names_code(args):
                out.append((cid, 'Not run (no step used): this store\'s search found nothing for a model or part code. Search by words and values, list the category, and read the codes on the rows.'))
                continue
            mark = name + _dump(args)
            if not ending and name != self.roles.get('cart') and (mark in self.recent):
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            if USE_G32_IGNORED_CAP_REPEAT and not ending and name in self.code_tools():
                same = self.ignored_cap_repeat(name, args)
                if same:
                    out.append((cid, same))
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
                settle = self.settle_hold() if USE_P110_11_NEAR_SETTLE_HOLD and self.is_buy(name, args) else ''
                if settle:
                    out.append((cid, settle + '\nCart now: ' + self.cart_text()))
                    continue
                late = self.cheaper_hold([], late=True) if USE_G50_CHEAPER_HOLD_LATE and self.is_buy(name, args) else ''
                if late:
                    out.append((cid, 'Not run yet (no step used). ' + late + '\nCart now: ' + self.cart_text()))
                    continue
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
                        sets = self.whole_sets() if USE_G54_WHOLE_SETS else []
                        whole = (' Whole sets already read that meet every requirement and fit: ' + self.sets_text(sets) + '.') if sets else ''
                        out.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.' + whole + '\nCart now: ' + self.cart_text()))
                        continue
                    swapped = self.proven_swap() if USE_P110_8_PROVEN_SWAP else ''
                    if swapped:  # #110 item 8: Cairn put a strictly proven set in the cart; the checkout goes on with it
                        self.show_links(self.cart_ids(), self.tail() + 1)
                        swap_note = '[Cairn] A cart line did not state a requirement, or the cart cost more than needed; Cairn set the cart to opened listings whose own attributes state every requirement: ' + swapped
                        chain = swap_note + '\n' + chain
                        trimmed = '\n' + swap_note + trimmed
                    stopped = self.checkout_holds()
                    if stopped:
                        out.append((cid, 'Not run yet (no step used). ' + stopped + '\nCart now: ' + self.cart_text()))
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
                    if USE_T_R104_5_NAMED_REFERRAL:
                        self.named_referrals(lines, self.tail() + 2)
                    cheap = self.cheaper_rows()
                    read += cheap
                    held_late = self.late_holds(read) if USE_G49_CHEAPER_HOLD or USE_G45_WEAK_LINE_HOLD or USE_G52_AGREED_SWAP else ''
                    if held_late:
                        out.append((cid, held_late + '\nCart now: ' + self.cart_text()))
                        continue
                    held_set = self.set_check_hold() if USE_P110_6_SET_CHECK and not swapped else ''
                    if held_set:
                        out.append((cid, held_set + '\nCart now: ' + self.cart_text()))
                        continue
                    picked = self.line_pick() if USE_R104_2_HEAVY_PICK and not swapped else ''
                    if picked:
                        out.append((cid, picked + '\nCart now: ' + self.cart_text()))
                        continue
                    verdict = self.check_cart()
                    if USE_E44_CHEAPER_MORE_ROUNDS and cheap and not verdict:
                        verdict, more = self.cheaper_rounds()
                        read += more
                    if verdict:
                        seen = "Cairn read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        seen = (hunted + ' ' + self.below_cart() + '\n' if hunted else '') + seen
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Cairn re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.left() > 5:
                    sets = self.set_hold() if USE_G54_WHOLE_SETS else ''
                    if sets:
                        out.append((cid, 'Not run yet (no step used). ' + sets))
                        continue
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
                    chained = self.decline_chain_round() if USE_E48_DECLINE_CHAIN_ROUND and not self.leads and 'e49' not in self.decline_checks else ''
                    literal = self.literal_hold() if USE_E49_LITERAL_FIGURE_HOLD else ''
                    if literal:
                        out.append((cid, 'Not run yet (no step used). ' + chained + literal))
                        continue
                    basket = self.basket_plan(self.leads[0]) if USE_G56_BASKET_REVIEW and self.leads and self.left() > 4 and self.refuse_blocks == 1 and len(self.items) >= 2 else None
                    if basket:
                        self.refuse_blocks = 2
                        first = self.leads[0]
                        self.check_cart(list(basket), SPEND_HARD)
                        if self.verdicts.get(self.check_key(list(basket))) and first in self.leads:
                            self.confirmed = first
                            self.trace.append({'kind': 'refuse_blocked', 'confirmed': first, 'basket': list(basket)})
                            out.append((cid, 'Not run yet (no step used). An independent check passed the whole basket ' + ', '.join((f'{i} x{q}' for i, q in basket.items())) + ': it meets every requirement. Put exactly those lines in the cart at the requested quantities (remove anything else) and check out; end without a purchase only if a record clearly breaks a requirement.'))
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
                    needless = self.needless_decline() if USE_R104_4_NEEDLESS_DECLINE_HOLD else ''
                    if needless:
                        out.append((cid, 'Not run yet (no step used). ' + needless))
                        continue
                    near = self.nearest_hold() if USE_E52_NEAREST_RULE_HOLD else ''
                    if near:
                        out.append((cid, 'Not run yet (no step used). ' + near))
                        continue
            if self.is_refuse(name, args) and USE_G58_LATE_SHORT_REVIEW and 2 <= self.left() <= LATE_ORDER_STEPS:
                done = self.late_short_review()
                if done or self.finished:
                    out.append((cid, (done or 'Not run: the episode is over.') + (' [episode finished]' if self.finished and done else '')))
                    continue
            if self.is_refuse(name, args) and USE_G57_LATE_FINISH and self.left() <= LATE_ORDER_STEPS and not self.late_held:
                done = self.late_finish()
                if done or self.finished:
                    out.append((cid, (done or 'Not run: the episode is over.') + (' [episode finished]' if self.finished and done else '')))
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
            if self.is_refuse(name, args) and (not self.leads) and (not self.pushed) and ((self.left_at_call > self.max_steps // 3 and self.left() > self.tail() + len(self.items) + 2) if USE_G61_PUSH_STEPS_AT_CALL else self.left() > self.max_steps // 3):
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args) and USE_E53_LATE_PROVEN_ORDER and self.left() <= LATE_ORDER_STEPS:
                ordered = self.proven_order()
                if ordered:
                    out.append((cid, ordered + (' [episode finished]' if self.finished else '')))
                    continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                done = self.clock_order() if USE_G73_CLOCK_BEST_ORDER else ''
                if done or self.finished:
                    out.append((cid, (done or 'Not run: the episode is over.') + (' [episode finished]' if self.finished and done else '')))
                    continue
                if USE_P110_1_NEAREST_ORDER and self.roles.get('buy') and self.left() >= 2 and self.nearest_wanted(True) and self.near_cart():
                    if self.finished or self.left() < 1:
                        out.append((cid, 'Not run: the episode is over.'))
                        continue
                    faults = []
                    near = _shape(self.near_args(), self.schemas.get(self.roles['buy']), 'args', faults)
                    name, args = (self.roles['buy'], near if not faults else dict(self.brief.get('buy_args') or {}))
                    self.trace.append({'kind': 'nearest_order', 'cart': self.cart_ids()})
                    trimmed = '\n[Cairn] The request asks for the nearest listing when nothing meets every requirement, so instead of ending without a purchase Cairn ordered the nearest listings: ' + self.cart_text() + trimmed
                else:
                    args = self.quiet(name, args)
            if self.is_buy(name, args):
                late = self.late_question()
                if late:
                    out.append((cid, 'Not run yet. ' + late + '\nCart now: ' + self.cart_text()))
                    continue
            if USE_G3_LATE_PAGE_SHOW and self.is_buy(name, args):
                trimmed = self.late_page_show() + trimmed
                if self.finished:
                    out.append((cid, 'Not run: the episode is over.'))
                    continue
            if ending and self.left() >= 2:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Cairn] ' + said + trimmed
            if USE_P110_12_SEEN_OWNER_LINK and self.is_buy(name, args) and self.left() >= 3 and not self.finished:
                self.show_links(self.cart_ids(), 1)  # #110 item 12: references followed right before every checkout
                if self.finished or self.left() < 1:
                    out.append((cid, 'Not run: the episode is over.'))
                    continue
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
            pages_before = self.pages_read
            paged = self.more_pages(name, args, observation) if name not in plain and (not ending) else ''
            paged += self.last_page_first(name, args, observation) if fresh and (not ending) else ''
            paged += self.shorter_search(name, args, observation) if name not in plain and (not ending) else ''
            if USE_E18_TOP_ROWS_OF_MIDLIST and fresh and not ending and name not in plain and not paged:
                paged += self.top_rows(name, args, observation)
            if USE_E24_FIT_SCAN and not ending and name not in plain and name in self.code_tools() and not _failed(observation):
                paged += self.fit_scan(name, args, observation)
            if USE_G30_BIG_ROW_READS and not ending and name not in plain and not _failed(observation):
                paged += self.big_row_reads([observation] + list(self.paged_obs))
            if USE_G31_SHORT_CHEAP_READS and not ending and name not in plain and self.pages_read > pages_before:
                paged += self.short_cheap_reads(name, [observation] + list(self.paged_obs))
            if USE_G22_DEVICE_SERIES_SWEEP and not ending and name in readers:
                swept = self.series_sweep()
                paged += '\n' + swept if swept else ''
            if USE_E21_DROP_DEAD_FIT_CODES and not self.fit_codes_dead and not ending and not _failed(observation) and not (isinstance(observation, dict) and observation.get('error')) and not self.records_in(observation) and self.holds_required_code(args) and name in self.code_tools():
                self.fit_codes_dead = True
                self.trace.append({'kind': 'fit_codes_dead', 'args': _clip(args, 80)})
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
            dear = ''
            if name == self.roles.get('cart'):
                before_lines = dict(self.cart_lines)
                self.note_cart(name, args)
                if USE_G21_DEAR_LINE_PAGE and not self.finished:
                    dear = self.dear_line_page(before_lines)
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
            text = self.show(observation) + paged + trimmed + dear
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
        blind = self.cart_touched and (not fresh) and (self.left() >= 2)
        if USE_G63_NO_BLIND_CHECKOUT:  # #97 item 63: taken before items 54/62/73 change the cart
            blind = blind and not self.dropped and (self.buys == 0 or self.cart_at > self.refused_at)
        changed = False
        if USE_G54_WHOLE_SETS and self.bound and buy:
            changed = self.forced_set()
        if USE_G62_FORCED_BASKET and not changed and not holding and buy:
            changed = self.forced_basket()
        if USE_G73_CLOCK_BEST_ORDER and not changed and not holding and buy and len(self.items) >= 2 and self.ever_held and self.model_room() <= LATE_ORDER_ROOM and self.category_key():
            lines = self.best_basket()
            changed = bool(lines) and self.set_cart(lines)
        if changed:
            holding = bool(self.cart_lines)
        rescued = False
        short = lambda: not [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0] or bool(self.missing_items())
        if USE_P110_1_NEAREST_ORDER and not changed and buy and self.left() >= 2 and not self.finished and short() and self.nearest_wanted(True):
            rescued = self.near_cart()  # #110 item 1: the forced ending orders the nearest listings for an empty or short cart
            holding = holding or rescued
        if self.finished or self.left() <= 0:
            return ''
        whole = True
        if holding and len(self.items) >= 2:
            whole = rescued or self.fill_short_cart() or not self.can_refuse()
            if not whole and USE_G73_CLOCK_BEST_ORDER and buy and self.model_room() <= LATE_ORDER_ROOM and self.category_key() and not self.finished:
                lines = self.best_basket()
                whole = bool(lines) and self.set_cart(lines)
            if whole and len(self.cart_ids()) >= 2:
                self.fill_facts(self.cart_ids(), self.tail())
                if self.left() >= 3 and self.model_room() > 5:
                    self.show_links(self.cart_ids(), self.tail())
            if not whole:
                self.trace.append({'kind': 'short_cart_refused', 'lines': self.cart_ids()})
        elif USE_E47_LINKS_ONE_LINE_CARTS and holding and len(self.items) == 1 and self.left() >= 3 and self.model_room() > 5:
            self.show_links(self.cart_ids(), self.tail())
        if USE_G74_FORCED_FIT_REPAIR and holding and whole and self.bound and buy and not self.finished:
            self.fit_repair()
        if self.finished or self.left() <= 0:
            return ''
        if whole and buy and (holding or blind):
            faults: list[str] = []
            args = _shape(self.near_args() if USE_P110_1_NEAREST_ORDER and self.near_made else dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
            if not faults and USE_P110_12_SEEN_OWNER_LINK and holding and self.left() >= 3 and self.model_room() > 5:
                self.show_links(self.cart_ids(), self.tail())  # #110 item 12: before the forced checkout too
                if self.finished or self.left() <= 0:
                    return ''
            if not faults:
                if USE_E35_CHAIN_OFFER_VIEWS or (USE_G63_NO_BLIND_CHECKOUT and holding and self.left() >= 2):
                    self.offer_views_before_buy()
                    if self.finished or self.left() <= 0:
                        return ''
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
        moved = self.first_moves() if not self.finished else ''
        self.env_gap = ENV_GAP
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        if moved:
            intro += '\n\nCairn has searched for each item and read some of its listings (one step each):\n' + moved + '\nJudge these against the requirements and search further only for what is still missing.\n\n' + self.status()
        else:
            intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE_R104 if USE_R104_7_GUIDE_RULES else GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                self.chat.append({'role': 'user', 'content': '[Cairn] Cairn tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                ended = self.finish()
                rounds = 0
                while USE_G64_MORE_ROUNDS and rounds < 2 and ended.startswith('ending without a purchase was refused') and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                    rounds += 1  # #97 item 64: the store refused the ending without a purchase: up to two more working rounds
                    self.trace.append({'kind': 'more_round', 'round': rounds, 'left': self.left()})
                    self.chat.append({'role': 'user', 'content': '[Cairn] Cairn tried to end the episode and ' + ended + '\nApply what the store says, then finish.\n' + self.status()})
                    self.work(self.loop_calls + MORE_ROUND_TURNS, SPEND_HARD + MORE_ROUND_SPEND)
                    ended = self.finish() if rounds < 2 and not self.finished else ''
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
        quiet = False  # #110 item 15: the last model turn sent no tool call
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.clock() < LAST_TURN or (not self.cart_ids() and self.model_room() > EMPTY_ROOM)):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue
            if USE_P110_7_PROVEN_ORDER_ANY and self.proven_try():
                if self.finished:
                    break
                continue
            if USE_P110_1_NEAREST_ORDER and self.nearest_read is None and self.clock() >= WIND_DOWN and (not self.cart_ids() or self.missing_items()) and self.model_room() >= NEAREST_SECONDS + CALL_FLOOR:
                self.nearest_wanted(True)  # #110 item 1: read the request for a nearest rule while the forced ending can still use it
            self.trim()
            self.loop_calls += 1
            models = ([HEAVY, LIGHT] if HEAVY in self.judges() else [LIGHT, SPARE]) if USE_P110_15_IDLE_OTHER_MODEL and quiet else [LIGHT, SPARE]  # run 19870: the spare first broke the prompt cache and answered empty
            message = self.think(self.chat, models, ACTOR_TOKENS if USE_E14_ACTOR_TOKENS_8000 else BASE_ACTOR_TOKENS, self.offered())
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
            quiet = not calls
            if not calls:
                idle += 1
                reopened = self.reopen_search() if USE_G59_REOPEN_SEARCH else ''
                if reopened:
                    idle = 0
                self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + reopened + self.status()})
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
