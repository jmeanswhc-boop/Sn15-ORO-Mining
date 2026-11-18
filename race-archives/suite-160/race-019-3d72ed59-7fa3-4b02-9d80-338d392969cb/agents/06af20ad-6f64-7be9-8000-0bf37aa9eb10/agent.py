"""Tobi: buys for one shopper inside an ORO store world. Built on Cairn v2's public code (Cairn: Halo v3 line).

A light model walks the store with the world's own tools; code keeps the ledger, plans the questions to the shopper
and checks the cart before an order and the evidence before a decline. Tobi's own changes to that base:
- a one-item request's first question names one planned property and the second three not named yet; a basket's
  first question names one property of the first planned item, and with more items than messages the last message
  names every item still unasked, its first item with the most properties; lists stop after a reply that wants one
  question at a time;
- planned labels that a shopper cannot hold a product preference on (stock, delivery, warranty, seller, price, yes/no
  flags, names no listing carries and no story sentence hints) go to the back of the line;
- the review before an ending without a purchase reads the never-opened in-budget rows first, and a listing it names
  can still be opened, carted and ordered with three steps left;
- under a lowest-price rule, cheaper fitting rows that were seen but never opened are opened before the checkout.
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
TALK_TOOL = '\\b(?:messag\\w*|ask\\w*|questions?|contact\\w*|notes?|send\\w*|repl\\w*|clarif\\w*|talk\\w*|chat\\w*|tell\\w*|inquir\\w*|enquir\\w*|say|says|speak\\w*|ping\\w*|relay\\w*|write|writes)\\b'
TALK_TO = '\\b(?:customers?|shoppers?|buyers?|users?|clients?|requesters?|humans?|persons?|patrons?|principals?|owners?|employers?|guests?|callers?)\\b'
NOT_TALK = '\\b(?:declin\\w*|refus\\w*|abandon\\w*|finish\\w*|checkout|purchas\\w*|buy(?:s|ing)?|orders?|carts?|baskets?|search\\w*|find\\w*|lookup|brows\\w*|details?|compar\\w*|quantit\\w*|listings?|products?|items?|stock|price\\w*)\\b'
ENDS_TALK = '\\b(?:ends?|ending|final|terminat\\w*|closes?)\\b[^.]*\\b(?:episode|session|task|conversation|shopping|visit|errand)\\b|\\b(?:episode|session|task|conversation|shopping|visit|errand)\\b[^.]*\\b(?:ends?|ending|is over|terminat\\w*|closes?)\\b|\\bwithout (?:buying|paying|purchas\\w*|a purchase)\\b'
SAID_LIMIT = '\\b(\\d{1,2})\\s+(?:[a-z\']+\\s+){0,2}(?:messages?|notes?|questions?)\\b'
TRAIL_ROOM = 15
AUDIT_RECORDS = 30  # records one decline review reads
AUDIT_OPENED = 18  # of them, opened records at most while unopened in-budget rows wait
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
HUNT_GRACE = 25.0
HUNT_VALUE_CHARS = 30
ASK_WORDS = '\\b(?:one|each|single|first|time|many|much|several|list|once|separate\\w*|split|short\\w*|questions?|ask\\w*|turn|lot|too|all)\\b'
SWEEP_SAMPLE = 2  # pages of a search too long to read whole that are still read for a pick by price or size
SWEEP_STALE = 3  # fresh pages in a row with nothing below the price frontier that end an item's sweep
SWEEP_SPARE = 12  # steps kept beyond the finish before a search the model ran is paged in code
SWEEP_NARROW = 3  # narrowing words named to the model for a search that overflows the store's reach
SWEEP_WORDS = 6  # fitting-row title words kept per item, most common first
COVER_SHOWN = 4  # searches per item the coverage line names, latest first
SWEEP_SHARE = 0.5  # a word on more than this share of the fitting rows narrows a search too little
HUNT_ROUNDS = 2
HUNT_KEEP = 10
HUNT_FIT_WORDS = 4
HUNT_FIT_OPENS = 4
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
LIST_STEPS = (1, 3)  # labels in the episode's first and second question, before the full list size: one property
                    # first, which a shopper limiting each message to one property still answers, then three not
                    # named yet (a reply that wants one question at a time turns lists off for the rest)
UNASKABLE = '\\b(?:stock|availab\\w*|in store|deliver\\w*|shipping|dispatch\\w*|warrant\\w*|guarantee\\w*|seller\\w*|vendor\\w*|merchant\\w*|price\\w*|cost\\w*|discount\\w*|refund\\w*|returns?)\\b|^(?:is|has|can|does|are)\\s|^(?:options?|model|model (?:number|name)|part number|sku|id|code)$'
LIST_STEPS_MIXED = (1, 3)  # the same on several items: each message is about one item, the first item planned gets one
                          # property and the next item three, so a shopper who limits a message to one property still
                          # answers the first and the last allowed message names three
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
GUIDE = 'You are Tobi, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun); a store\'s search can read attribute values and shop names as well as titles and may want every word, so a few required values plus the noun can list nearly the whole fitting set, and a model or part code may find nothing (one try tells). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Tobi knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Tobi\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Tobi carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Tobi\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Tobi adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category. A listing under the requested label is that kind of product even when its title or description names something else, so judge it on the requirements, price and stock.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value; when the requirement asks for what goes with that listing\'s value (its partner, its counterpart) and the page names what its value pairs with, the requirement is that other value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Tobi reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Price and stock can change silently: Tobi re-reads the cart\'s lines and looks at the cart when you call the checkout, so do not spend a turn on either.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"search_words": {"tool": the tool that searches listings by free words (when one tool lists a category and another takes words, the one that takes words), "field": its parameter that takes the words, "args": the fixed arguments that make it search by words (for a tool with a mode argument, that mode), else {}}.\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "lost": [numbers of the questions whose answer states no preference and only says it did not follow or catch the question, or asks for it again in other or plainer words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write \"not X\" only when the answer rejects X itself.'
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
    excludes them. Which bare words are series is decided against the codes other records carry (Tobi.link_system)."""
    fam = {hit.upper() for hit in re.findall('\\b([A-Za-z]{1,5}\\d{1,3})-\\d{2,7}\\b', text)}
    rest = re.sub('\\b[A-Za-z]{1,5}\\d{1,3}-\\d{2,7}\\b', ' ', text)
    rest = re.sub('(?i)\\b(?:except|excluding|but not|not for|not compatible with|does not (?:fit|take))\\b[^.;]*', ' ', rest)
    return sorted(fam), sorted({hit.upper() for hit in re.findall('\\b[A-Za-z]{1,5}\\d{1,3}\\b', rest)})
PAIR_MARKS = (': ', ' = ', '=', ' -> ', ' - ')
FIELD_MARKS = (' | ', '\n', ' || ', ' ; ', '; ', '\t', ' // ', ' \u00b7 ', ' \u2022 ', '|', ';', ', ')
FIELD_NAME = "[A-Za-z][A-Za-z0-9_ ./&'()-]{0,38}"
LIST_BULLET = '^\\s*(?:[-*\u2022\u00b7]|\\d{1,3}[.)])\\s+'
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
    money = re.fullmatch('[$\u20ac\u00a3]?\\s?((?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?)', bare)
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

def _repeat_records(bare: str) -> list[dict[str, Any]] | None:
    """Records written one field a line with nothing between them but the field each record opens with (often behind a
    bullet, a number or a heading line): the text is cut before each line naming that first field again. The first field
    is the first line holding a value, so a heading such as "Found 3 listings:" is no field; a line before the first record
    or naming no field belongs to no record; the other lines keep their indent, less the shallowest."""
    lines = [line.rstrip() for line in bare.splitlines() if line.strip()]
    if len(lines) < 2 * RECORD_FIELDS:
        return None
    pairs = [_pair(line.strip(), ': ') for line in lines]
    names = [pair[0] if pair else '' for pair in pairs]
    lead = next((pair[0] for pair in pairs if pair and pair[1]), '')
    if not lead or names.count(lead) < 2:
        return None
    groups: list[list[str]] = []
    for line, name in zip(lines, names):
        if name == lead:
            groups.append([re.sub(LIST_BULLET, '', line.strip())])
        elif groups and name:
            groups[-1].append(line)
    rows = []
    for group in groups:
        rest = group[1:]
        cut = min((len(line) - len(line.lstrip()) for line in rest), default=0)
        rows.append(_record_text('\n'.join([group[0]] + [line[cut:] for line in rest])))
    good = [row for row in rows if row]
    return good if len(good) >= 2 and len(good) * 5 >= len(groups) * 4 else None

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
    split = _repeat_records(bare) if '\n' in bare else None
    if split:
        return split
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

def _text_rows(text: Any) -> Any:
    """A text field of a result object that holds the result's rows (a table, record blocks, one record a line): its rows
    when it reads as two or more listings, else the text as it was (a note or a description stays text)."""
    if not isinstance(text, str) or ('\n' not in text and ' | ' not in text):
        return text
    found = _records_text(text)
    if isinstance(found, list) and len(found) >= 2 and all((isinstance(r, dict) and _holds_id(r) for r in found)):
        return found
    return text

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
        out = {k: _restore_field(v, depth, carded, page) if not isinstance(v, str) or len(node) <= 2 else _text_rows(_block_text(v) if listing else v) for k, v in node.items()}
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
            if isinstance(got, dict):
                return got
        except ValueError:
            pass
    best: dict[str, Any] | None = None
    decoder = json.JSONDecoder()
    for brace in re.finditer('{', text):
        try:
            got, _ = decoder.raw_decode(text, brace.start())
        except ValueError:
            continue
        if isinstance(got, dict) and (best is None or len(got) > len(best)):
            best = got
    return best

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

LIMIT_SIDE = (('ge', 'no less than|not less than|no lower than|not lower than|no smaller than|no shorter than|no lighter than|no fewer than|not under|not below'),
              ('le', 'no more than|not more than|no bigger than|no larger than|no greater than|no heavier than|no longer than|no higher than|no wider than|not over|not above|not exceeding|at most|up to|maximum of|maximum|max|or less|or under|or below|or lower|or fewer|tops|under|below|less than|lower than|smaller than|shorter than|lighter than|fewer than|within'),
              ('ge', 'at least|minimum of|minimum|or more|or over|or above|or higher|over|above|more than|greater than|bigger than|larger than|longer than|heavier than|higher than'))
NOT_VALUE = '^(?:not|no|anything but|anything except|anything other than|nothing|never|except|other than|without|excluding|avoid)\\b'
NOT_ANYWHERE = '\\b(?:not|non|anything but|anything except|other than|without|excluding|except|avoid|rather than|instead of)\\b'
POINTED = '\\b(?:matching|same as|same|like the|unlike|different from|differs?|what the page|the one on|as on|points? to|page of|similar|close to|near|than the)\\b|\\bwithin\\b.*\\bof\\b'
COMPARED = '\\b(?:than|within|near|close to|similar|at least|at most|longer|shorter|bigger|smaller|heavier|lighter|cheaper|wider|narrower|higher|lower|more|less|fewer)\\b'


def _plain_text(text: Any) -> str:
    """Lower case with possessive 's and apostrophes folded ("women's" -> "women"), word separators as spaces."""
    low = re.sub("'s\\b", '', str(text).lower()).replace("'", '')
    return _words(low)


def _has_word(phrase: str, text: str) -> bool:
    """Whether the phrase occurs in the text as whole words ("men" is not in "women", "l" is not in "xl")."""
    return bool(phrase) and re.search('(?<![a-z0-9])' + re.escape(phrase) + '(?![a-z0-9])', text) is not None


def _figures(text: str) -> list[tuple[float, str]]:
    """(number, unit word after it) for each figure in a text: "3.7 GHz" -> (3.7, 'ghz'), "240gb" -> (240.0, 'gb'),
    "2 pieces" -> (2.0, 'pieces'); thousands commas folded, a decimal comma read as a point."""
    out = []
    for m in re.finditer('(?<![\\w.])(-?\\d+(?:[.,]\\d+)*)\\s*([a-z"\']{0,12})', str(text).lower()):
        raw = m.group(1)
        raw = raw.replace(',', '') if re.fullmatch('-?\\d{1,3}(?:,\\d{3})+', raw) else raw.replace(',', '.')
        try:
            out.append((float(raw), m.group(2).strip('"\'') or ''))
        except ValueError:
            continue
    return out


def _limit_of(value: str) -> tuple[str, float, str] | None:
    """(side, number, unit) of a limit need ("at most 3.7 ghz" -> ('le', 3.7, 'ghz'), "no less than 5 kg" -> ('ge',
    5.0, 'kg'), "4 kg or less" -> ('le', 4.0, 'kg')), else None. Negated wording is read before the plain words it
    holds, and a limit includes its edge. The side words are ordinary English comparison words."""
    low = ' ' + str(value).lower().strip() + ' '
    figures = _figures(low)
    if len(figures) != 1:
        return None
    for side, words in LIMIT_SIDE:
        if re.search('(?<![a-z])(?:' + words + ')(?![a-z])', low):
            number, unit = figures[0]
            return (side, number, unit)
    return None


def _unit_same(a: str, b: str) -> bool:
    """Two unit words name one unit only when they are spelled the same (plural -s folded) or both are empty; no unit
    is converted or equated with another (units by rule, never by table)."""
    fold = lambda u: u[:-1] if len(u) > 2 and u.endswith('s') else u
    return fold(a) == fold(b)


def _value_fit(want: str, have: Any) -> bool | None:
    """Whether one attribute value meets one need value: True when it states it, False when it clearly does not, None
    when the attribute cannot tell. A limit holds or breaks only on a figure in its own unit; a plain figure value
    ("150 kg") is equal or not only in its own unit; a range or any other value with several figures is not judged
    unless it appears word for word; a plain value is stated by its whole words and broken only when the attribute
    shares none of them; an exclusion at the value's start breaks when the excluded value is there, and any other
    negated wording is not judged."""
    text = _plain_text(have)
    wanted = _plain_text(want)
    if not wanted:
        return None
    if _has_word(wanted, text):
        return True
    limit = _limit_of(want)
    if limit:
        side, bound, unit = limit
        figures = [n for n, u in _figures(str(have)) if _unit_same(u, unit)]
        if not figures:
            return None
        ok = (lambda n: n <= bound) if side == 'le' else (lambda n: n >= bound)
        return True if any((ok(n) for n in figures)) else False
    negated = re.search(NOT_VALUE, wanted)
    if not negated and re.search(NOT_ANYWHERE, wanted):
        return None
    plain = re.sub(NOT_VALUE, '', wanted).strip()
    squeeze = lambda t: re.sub('[^a-z0-9]+', '', t)
    tokens = text.split()
    glued = set(tokens) | {a + b for a, b in zip(tokens, tokens[1:])} | {a + b + c for a, b, c in zip(tokens, tokens[1:], tokens[2:])}
    if len(squeeze(plain)) >= 5 and squeeze(plain) in glued:
        return False if negated else True
    figures = _figures(plain)
    if len(figures) > 1 or (figures and re.search(COMPARED, plain)):
        return None
    if len(figures) == 1:
        number, unit = figures[0]
        theirs = [n for n, u in _figures(str(have)) if _unit_same(u, unit)]
        if not theirs:
            return None
        stated = any((abs(n - number) < 1e-09 for n in theirs))
        return (not stated) if negated else stated
    joins = {'and', 'or', 'with', 'the', 'a', 'of'}
    words = {w for w in plain.split() if re.search('[a-z0-9]', w) and w not in joins}
    if not words:
        return None
    seen = {w for w in tokens if re.search('[a-z0-9]', w) and w not in joins}
    fold = lambda w: w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
    mine, theirs = {fold(w) for w in words}, {fold(w) for w in seen}
    whole = mine <= theirs
    if negated:
        return False if whole else (True if not (mine & theirs) else None)
    if whole:
        return True
    return False if not (mine & theirs) else None


def _need_fit(block: dict[str, Any] | None, need: str, item: str='') -> bool | None:
    """A listing's verdict on one need "property: constraint" from its attributes, by the base's reading rule. A key that
    adds the item's own name to the property (a bag's "bag shape") decides, even against the bare key; else the key
    that is the property itself decides; any of them stating the need is enough, and all of them must break it for
    False. A key for another property, even a related one ("strap colour", "pack size", "colour" for "rim colour"),
    decides nothing, but with no key for the property the value stated under any attribute counts. A pointed value, a
    need without a property, or no attribute for it gives None."""
    prop, _, value = str(need).partition(':')
    prop, value = prop.strip(), value.strip()
    if not block or not prop or not value or re.search(POINTED, value.lower()):
        return None
    mine, own = _spelled(prop), _spelled(item)
    named = [v for k, v in block.items() if mine < _spelled(k) and (_spelled(k) - mine) <= own]
    exact = [v for k, v in block.items() if _spelled(k) == mine]
    for group in (named, exact):
        if group:
            found = [_value_fit(value, v) for v in group]
            if any((f is True for f in found)):
                return True
            return False if all((f is False for f in found)) else None
    return True if any((_value_fit(value, v) is True for v in block.values())) else None


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

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Tobi:

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
        self.short_reviewed = False
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
        self.better: list[str] = []
        self.swapped = False
        self.fault_marks: set[str] = set()
        self.fault_held = 0
        self.stated_held = False
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
        self.hunt_no_noun: set[int] = set()
        self.bare_pages: dict[str, int] = {}
        self.code_tries: list[bool] = []
        self.codes_dead = False
        self.hear_reads = 0
        self.fit_hunted = False
        self.hunt_mark = ''
        self.sweeps: dict[int, dict[str, Any]] = {}
        self.page_args: dict[str, str] = {}
        self.page_queries: dict[str, set[str]] = {}

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
        self.roles = {k: tool for k, tool in ((k, self.tool_named(v)) for k, v in roles.items()) if tool}
        if not self.roles.get('ask'):
            guessed = self.talk_tool()
            if guessed:
                self.roles['ask'] = guessed
                self.trace.append({'kind': 'ask_role_from_tools', 'tool': guessed})
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
        if limit is None and self.roles.get('ask'):
            said = [int(n) for n in re.findall(SAID_LIMIT, self.request, re.I)]
            limit = min(said) if said else None
        if limit is not None and limit >= 0:
            self.msg_limit = limit
        elif self.roles.get('ask'):
            self.msg_limit = 3
        steps = _count(brief.get('steps'))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in _items(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                if self.store_check(str(entry['step'])):
                    self.trace.append({'kind': 'process_dropped', 'step': str(entry['step'])[:120]})
                    continue
                when = str(entry.get('when') or '').strip().lower()
                self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _count(entry.get('messages')) or 0)})
        if self.roles.get('ask'):
            talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def tool_named(self, value: Any) -> str | None:
        """The store tool a brief's role value names: the name itself, the name in another case, or the one tool name
        written inside a longer value (a list, an object, a name with a remark after it); None when it names none or
        several. A brief that wrote a role in any other shape lost that role, and a lost message role turns every
        question off for the episode."""
        if isinstance(value, str) and value in self.schemas:
            return value
        text = value if isinstance(value, str) else _dump(value) if isinstance(value, (list, dict)) and value else ''
        if not text.strip():
            return None
        same = [t for t in self.schemas if t.lower() == text.strip().lower()]
        if len(same) == 1:
            return same[0]
        hits = [t for t in self.schemas if re.search('(?<![A-Za-z0-9_])' + re.escape(t) + '(?![A-Za-z0-9_])', text)]
        return hits[0] if len(hits) == 1 else None

    def talk_tool(self) -> str | None:
        """The tool that messages the shopper, read from the tool list when the brief names none: the one tool whose
        name and description speak of messaging a customer, whose name names no search, page, cart, order or ending
        job, whose description does not say it ends the episode, and whose required inputs are free text only. When
        several qualify, the one whose own name speaks of messaging; None when that still leaves a doubt. Without a
        message role the interviewer never runs and the loop model writes its own messages, or none."""
        taken = set(self.roles.values())
        found = []
        for tool in self.tools:
            fn = tool['function']
            name = fn['name']
            if name in taken or name not in self.schemas:
                continue
            said = _words(name)
            text = said + ' ' + _words(str(fn.get('description') or ''))
            if re.search(NOT_TALK, said) or re.search(ENDS_TALK, text):
                continue
            if not (re.search(TALK_TOOL, text) and re.search(TALK_TO, text)):
                continue
            schema = self.schemas.get(name) or {}
            props = schema.get('properties') or {}
            free = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and 'const' not in v]
            if not free or any((k not in free for k in schema.get('required') or [])):
                continue
            found.append(name)
        named = [n for n in found if re.search(TALK_TOOL, _words(n))] if len(found) > 1 else found
        return named[0] if len(named) == 1 else None

    @staticmethod
    def store_check(step: str) -> bool:
        """Whether a process step is only a check the listings answer themselves (stock, availability, price, budget).
        A request's plea to make sure the item can be bought today is a store check, yet the brief sometimes reads it as a
        step with a message, which keeps a message from the questions and spends it on a yes/no about stock. A step that
        asks for the shopper's approval, a confirmation, a summary or a message to the shopper is kept."""
        text = _words(step)
        if re.search('\\b(?:approv\\w*|confirm\\w*|consent|permission|agree\\w*|summar\\w*|tell|inform|show|send|message|list|recap|ask|check with|ok|okay|go ahead|proceed|basket|cart|total|acceptable|fine|happy|before (?:choosing|ordering|buying))\\b', text):
            return False
        return bool(re.search('\\b(?:stock|availab\\w*|in store|sold out|price\\w*|cost\\w*|budget|afford\\w*|spend\\w*)\\b', text))

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
            needs = [m + (f' (the page it points at gives: {self.ref_values[m]})' if m in self.ref_values else '') for m in live if m not in told]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            if item.shelf:
                head += f' (store label: {item.shelf})'
            head += f', quantity {item.qty}'
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if told:
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
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
            when = {'choose': 'before choosing', 'order': 'at the order (Tobi does it when you check out)', 'end': 'at an ending without a purchase (Tobi does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.page_tools() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        left = self.left()
        text = f'[Tobi] step {self.step} of {self.max_steps} used, {left} left.'
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
        if self.buys == 0 and left > self.tail() + 1:
            text += self.coverage()
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
                if not values or re.search(UNASKABLE, name) or all((re.fullmatch('(?:true|false|yes|no|y|n|0|1)', v) for v in values)):
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

    def page_cap(self, name: str, param: str) -> int | None:
        """Pages one search can show: the page argument's own maximum when its schema states one, else None (a store
        whose schema states no last page is taken to show a whole search, page by page)."""
        spec = ((self.schemas.get(name) or {}).get('properties') or {}).get(param)
        spec = spec if isinstance(spec, dict) else {}
        top, low = (spec.get('maximum'), spec.get('minimum'))
        low = low if isinstance(low, int) and (not isinstance(low, bool)) else 1
        if isinstance(top, int) and (not isinstance(top, bool)) and top >= low:
            return top - low + 1
        return None

    def reach(self, name: str, param: str, count: int, shown: int, value: Any) -> int:
        """Rows the store can show of one search: pages to the page argument's stated maximum times the rows a page shows,
        or, when the argument counts rows (its next value is the first one plus a page of rows), up to that maximum plus a
        page; the whole count when the schema states no maximum."""
        cap = self.page_cap(name, param)
        if not cap:
            return count
        spec = ((self.schemas.get(name) or {}).get('properties') or {}).get(param) or {}
        low = spec.get('minimum') if isinstance(spec.get('minimum'), int) and (not isinstance(spec.get('minimum'), bool)) else 1
        if shown > 1 and isinstance(value, int) and (not isinstance(value, bool)) and value - low == shown:
            return cap - 1 + shown
        return shown * cap

    def window(self, name: str, observation: Any) -> tuple[int, int, int] | None:
        """(listings a search matched, rows one page shows, rows the store can show of it at all), or None when the
        result tells no count or no page size."""
        plan = self.paging(name, observation)
        if not plan or plan[1] <= 0:
            return None
        count, shown, param, value = plan
        return (count, shown, self.reach(name, param, count, shown, value))

    def frontier(self, number: int) -> float | None:
        """The lowest price among the item's buyable rows inside the budget whose own text breaks no requirement in force:
        the price a further search has to beat to matter for a lowest-price pick."""
        pool = self.item_ids(number) if self.category_key() else list(self.seen_order)
        cap = self.budget if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) else None
        prices = [p for i in pool for p in [self.price_of(i)] if p is not None and self.buyable(i) and (cap is None or p <= cap) and (not self.row_marks(i, number)[1])]
        return min(prices) if prices else None

    def search_item(self, observation: Any) -> int:
        """The requested item most rows of a search result belong to: the only item of a one-item request, else the most
        common owner of the rows' store labels; 0 when no row names one."""
        if len(self.items) == 1:
            return 1
        cat_key = self.category_key()
        owners = [n for _, row in self.records_in(observation) for n in [self.owner_of(str(row.get(cat_key) or ''))] if n] if cat_key else []
        return max(set(owners), key=owners.count) if owners else 0

    @_soft('')
    def coverage(self) -> str:
        """Per item, what the episode's searches covered: each search's match count, the rows read of it and whether the
        store can show it whole, and, for a lowest-price pick, the lowest price seen of a row breaking nothing in force.
        A search that overflows what the store shows of it hides most of its matches in no price order (in a recent race a
        fitting listing never showed in any search row on many failed runs of tasks other agents solved), so the model sees
        where a narrower search is due and which searches are done."""
        tool = self.roles.get('search')
        if not tool:
            return ''
        heads: dict[str, list[Any]] = {}
        for mark, (name, args) in list(self.read_args.items()):
            seen = self.read_calls.get(mark) if name == tool else None
            size, page = (self.window(tool, seen), self.paging(tool, seen)) if seen is not None else (None, None)
            if not size or not page:
                continue
            words = ' '.join((v for v in args.values() if isinstance(v, str))).strip()
            head = heads.setdefault(self.call_mark(tool, {k: v for k, v in args.items() if k != page[2]}), [words, size[0], 0, size[2], self.search_item(seen)])
            head[2] += len(self.records_in(seen))
        lines = []
        for number in range(1, len(self.items) + 1) if heads else []:
            said = []
            for words, count, read, reach, owner in [h for h in heads.values() if h[4] in (number, 0)][-COVER_SHOWN:]:
                if read >= min(count, reach):
                    state = 'all read' if count <= reach else f'all {reach} rows the store shows of it read, the rest unreachable: narrow it'
                else:
                    state = f'{read} read, the store shows at most {reach}: narrow it' if count > reach else f'{read} of {count} read'
                said.append(f'"{words}" {count} matched, {state}')
            low = self.frontier(number) if self.brief.get('pick') == 'cheapest' else None
            if said:
                lines.append(('item %d: ' % number if len(self.items) > 1 else '') + '; '.join(said) + (f'; lowest price seen of a row breaking nothing: {low:g}' if low is not None else ''))
        return ''.join(('\n[searches] ' + line for line in lines))

    @_soft('')
    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """For a pick by price or size, the rest of a search the model ran: read here, one step a page, when the store can
        show the search's whole match set within the spare steps. A set larger than the store shows of any one search is
        not paged at all: its pages sample it in no price order, so the lowest fit may sit where no page reaches; the
        model is told to narrow it instead, with the words fitting rows carry that would do so."""
        self.paged_obs = []
        pick = self.brief.get('pick')
        plan = self.paging(name, observation) if pick in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown or (pick == 'cheapest' and self.price_sorted(observation)):
            return ''
        size = self.window(name, observation)
        reach = size[2] if size else count
        rest = -(-(min(count, reach) - shown) // shown)
        spare = self.left() - self.tail() - SWEEP_SPARE
        whole = count <= reach and rest <= spare and self.pages_read + rest <= AUTO_PAGES
        pages = rest if whole else min(SWEEP_SAMPLE, rest) if spare >= SWEEP_SAMPLE and self.pages_read + SWEEP_SAMPLE <= AUTO_PAGES else 0
        hint = ''
        if not whole:
            number = self.search_item(observation)
            words = self.hunt_words(number)[1][:SWEEP_NARROW] if number else []
            where = f'the store shows at most {reach} rows of any one search' if count > reach else f'its {rest} further pages would take steps the order needs'
            hint = f'\n[Tobi] This search matched {count} listings in no price order, and {where}. The pick counts every fitting listing: narrow the search (a requirement\'s value or one more word) until its matches fit, then read every page of it.' + (f" Words fitting rows carry that would narrow it: {', '.join(words)}." if words else '')
        texts: list[str] = []
        while value is not None and len(texts) < pages and (not self.finished):
            faults: list[str] = []
            call = _shape({**args, param: value}, self.schemas.get(name), 'args', faults)
            reply = None if faults else self.act(name, call)
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
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts), 'matched': count, 'reach': reach})
        if not texts:
            return hint
        said = 'these are all the rows the store shows of it' if whole else f'a sample of its next {len(texts)} page(s)'
        return f'\n[Tobi read more of this search itself, one step a page: {said}]\n' + '\n'.join(texts) + hint

    def hunt_words(self, number: int) -> tuple[str, list[str]]:
        """The item's noun (a word of its store label no other item's label holds; a search reads labels too, so it keeps the
        rows to this item) and the words to narrow or turn a search by, most promising first.

        The words come from the titles of the item's rows that look like fits (they state a requirement in force and break
        none), since a fitting row's own wording is what a search must hold to list it; when fewer than three rows look like
        fits, every row breaking nothing counts. A word most such rows share narrows little and a word on one row finds
        little else, so the kept words sit between. Words of the request's own name for the item that its store label lacks
        (who it is for, what it is used for) follow them."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        title = self.title_key()
        pool = self.item_ids(number) if cat_key else list(self.seen_order)
        labels = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in pool] if cat_key else []
        others = {w for n, other in enumerate(self.items, 1) if n != number for w in re.findall('[a-z]{4,}', ((other.shelf or '') + ' ' + other.name).lower())}
        label = max(labels, key=labels.count).lower() if labels else (item.shelf or item.name).lower()
        nouns = [w for w in re.findall('[a-z]{4,}', label) if w not in others]
        noun = '' if number in self.hunt_no_noun else (max(nouns, key=len) if nouns else '')
        marks = {i: self.row_marks(i, number) for i in pool}
        fits = [i for i in pool if marks[i][0] and (not marks[i][1])]
        basis = fits if len(fits) >= 3 else [i for i in pool if not marks[i][1]]
        names = [str((self.rows.get(i) or {}).get(title) or '') for i in basis] if title else []
        own = _roots(item.shelf or '') | _roots(item.name)
        counts: dict[str, int] = {}
        for name in names:
            for word in {w.lower() for w in re.findall("[A-Za-z][A-Za-z'-]{2,}", name)}:
                counts[word] = counts.get(word, 0) + 1
        ceiling = max(2, int(len(names) * SWEEP_SHARE))
        words = [w for w, n in counts.items() if 2 <= n <= ceiling and w != noun and _roots(w) and (not _roots(w) & own)]
        words.sort(key=lambda w: (-counts[w], w))
        said = [w for w in dict.fromkeys(re.findall("[a-z][a-z'-]{2,}", item.name.lower())) if w != noun and _roots(w) and (not _roots(w) & _roots(label))]
        return (noun, words[:SWEEP_WORDS] + [w for w in said if w not in words])

    def hunt_item(self, number: int, tool: str, base: dict[str, Any], field: str, room: int) -> tuple[int, int]:
        """Up to `room` searches for one item, each read so that the store shows its whole match set where it can.

        A search shows a few rows a page and only so many pages, in no price order, while it may match hundreds of listings,
        so a broad search's cheapest fit may sit where no page reaches: in a recent race the passing listing never showed in
        any search row on most failed lowest-price runs of every agent looked at. A search whose matches fit what the store
        shows is read to its end; one that overflows it is read one page and then narrowed by a word fitting rows carry,
        a narrower search that still overflows becoming the one narrowed next. Once a set was read whole, the next search
        turns sideways (the told values under another fitting-row word instead of the noun), since a narrower one could
        only repeat it. The item's searching stops when SWEEP_STALE fresh pages in a row showed nothing priced below the
        cheapest row known to fit (the frontier). Returns (steps spent, new rows of the item)."""
        state = self.sweeps.setdefault(number, {'tried': set(), 'stale': 0, 'totals': {}, 'reach': {}, 'round': None})
        if state['round'] != self.hunt_mark:
            # a new round (first hunt, or a dropped requirement or a new budget since): the frontier moved, so the pages
            # that showed nothing below the old one say nothing about the new one
            state['round'], state['stale'] = (self.hunt_mark, 0)
        noun, words = self.hunt_words(number)
        values = [value for value, _ in self.value_words(number)]
        spent = gained = 0
        refused = False

        def ids() -> set[str]:
            return set(self.item_ids(number)) if self.category_key() else set(self.seen_order)

        def walk(query: str) -> tuple[bool, int | None]:
            """Run `query` (pages to its end when its matches fit the store's reach, else its first page): (whether any row
            came back, its match count when the result tells one)."""
            nonlocal spent, gained, refused
            state['tried'].add(query.lower())
            faults: list[str] = []
            args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
            if faults:
                refused = True
                return (False, None)
            found, total, walked = (False, None, set())
            while not self.finished and self.clock() <= HUNT_BY + HUNT_GRACE:
                mark = self.call_mark(tool, args)
                if mark in walked:
                    break
                walked.add(mark)
                if mark in self.read_calls:
                    seen = self.read_calls[mark]
                elif spent >= room:
                    break
                else:
                    low, before = (self.frontier(number), ids())
                    reply = self.act(tool, args)
                    spent += 1
                    if reply is None:
                        break
                    seen = reply.get('observation')
                    self.observe(tool, seen)
                    fresh = ids() - before
                    gained += len(fresh)
                    below = [i for i in fresh if self.price_of(i) is not None and (low is None or self.price_of(i) < low)]
                    state['stale'] = 0 if below else state['stale'] + 1
                rows = self.records_in(seen)
                found = found or bool(rows)
                if values and all((v.lower() in query.lower() for v in values)):
                    self.value_rows.update((i for i, _ in rows))
                size = self.window(tool, seen)
                page = self.paging(tool, seen)
                if not size or not page:
                    break
                total = size[0]
                state['reach'][query.lower()] = size[2]
                if page[3] is None or total > size[2] or state['stale'] >= SWEEP_STALE:
                    break
                faults = []
                args = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                if faults:
                    break
            if total is not None:
                state['totals'][query.lower()] = total
            return (found, total)

        def overflows(query: str) -> bool:
            total, reach = (state['totals'].get(query.lower()), state['reach'].get(query.lower()))
            return total is not None and reach is not None and total > reach
        stem = ''
        for query in dict.fromkeys((q for q in ([' '.join(values + [noun]).strip(), ' '.join(values)] if values else []) + [noun] if q)):
            if spent >= room or self.finished:
                return (spent, gained)
            if query.lower() in state['tried']:
                if state['totals'].get(query.lower()):
                    stem = query
                    break
                continue
            found, total = walk(query)
            if refused:
                return (spent, gained)
            if found:
                stem = query
                break
            if query == noun:
                self.hunt_no_noun.add(number)
                noun = ''
        while stem and spent < room and (not refused) and (not self.finished) and state['stale'] < SWEEP_STALE and self.clock() <= HUNT_BY + HUNT_GRACE:
            noun, words = self.hunt_words(number)
            if overflows(stem):
                options = [f'{w} {stem}' for w in words if w not in stem.lower().split()]
            else:
                core = ' '.join(values)
                options = [f'{w} {core}'.strip() for w in words if w not in core.lower().split() and w != noun]
            options = [q for q in options if q.lower() not in state['tried']]
            if not options:
                break
            found, total = walk(options[0])
            if found and overflows(stem) and overflows(options[0]):
                stem = options[0]
        return (spent, gained)

    def unread_pages(self, tool: str, wanted: dict[int, set[str]], room: int) -> int:
        """Up to `room` further pages of the episode's own searches that hold a required value of an item; the steps it took.

        Searches go whole-set first: one whose matches fit what the store can show comes before one that overflows it, and
        of those the one fewest pages from its end first, so each step goes to completing a set rather than sampling an
        endless one. A search whose rows all belong to another item is left. In a practice world accepted rows sat on unread
        pages of the episode's own searches in nearly every wrong decline, and in races 13-15 about 23% of passing carts
        were first seen on page 2 or later."""
        cat_key = self.category_key()
        heads: dict[str, dict[str, Any]] = {}
        for mark, (name, args) in list(self.read_args.items()):
            seen = self.read_calls.get(mark)
            page = self.paging(tool, seen) if name == tool else None
            if not page or page[1] <= 0:
                continue
            text = ' '.join((v for v in args.values() if isinstance(v, str))).lower()
            numbers = {n for n, words in wanted.items() if words & _text_words(text)}
            owners = {self.owner_of(str(r.get(cat_key) or '')) for _, r in self.records_in(seen)} if cat_key else set()
            if not numbers or (owners and 0 not in owners and (not owners & numbers)):
                continue
            count, shown, param, value = page
            family = self.call_mark(tool, {k: v for k, v in args.items() if k != param})
            head = heads.setdefault(family, {'pages': 0, 'count': count, 'shown': shown, 'param': param, 'args': None, 'seen': None, 'reach': None})
            head['reach'] = head['reach'] or self.reach(tool, param, count, shown, value)
            head['pages'] += 1
            if value is not None and self.call_mark(tool, {**args, param: value}) not in self.read_calls:
                head['args'], head['seen'] = (args, seen)

        def order(head: dict[str, Any]) -> tuple[bool, int]:
            return (head['count'] > head['reach'], -(-min(head['count'], head['reach']) // head['shown']) - head['pages'])
        spent = 0
        for head in sorted((h for h in heads.values() if h['args'] is not None), key=order):
            call, seen, walked, read = (head['args'], head['seen'], set(), head['pages'])
            while spent < room and (not self.finished) and read * head['shown'] < min(head['count'], head['reach']):
                page = self.paging(tool, seen)
                if not page or page[3] is None:
                    break
                faults: list[str] = []
                call = _shape({**call, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                now = self.call_mark(tool, call) if not faults else ''
                if faults or now in walked:
                    break
                walked.add(now)
                read += 1
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
        return f'\n[Tobi] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Tobi] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

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
            # a cheaper row whose own text already breaks a limit in force is not worth a step; the rest go cheapest
            # first, a row stating more of the requirements first at one price
            number = self.item_of(line)
            marks = {i: self.row_marks(i, number) if number else (0, 0) for i in pool}
            pool = [i for i in pool if not marks[i][1]]
            pool.sort(key=lambda i: (self.price_of(i) or 0.0, -marks[i][0]))
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

    def rows_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for row in self.rows.values() if not cat_key or self.owns(item, str(row.get(cat_key) or ''))))

    def hunt_room(self, early: bool=False) -> int:
        """Searches a lowest-price hunt may spend now. The finish, two steps per item and a reserve for opening and buying
        stay free; the reserve follows the task's length, since HUNT_KEEP was set on long tasks and on a task of under
        forty steps it leaves no room at any checkout, so the cheapest fitting row is never searched for. Early, right
        after the questions, a hunt spends at most a third of the steps left."""
        keep = min(HUNT_KEEP, max(4, self.max_steps // 6))
        room = min(HUNT_STEPS, self.left() - self.tail() - keep - 2 * len(self.items))
        return min(room, self.left() // 3) if early else room

    @_soft('')
    def hunt_cheaper(self, early: bool=False) -> str:
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
        room = self.hunt_room(early)
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
        if spent and not early:
            self.wider_held = True
        if not spent:
            return ''
        said = f'Tobi ran {spent} more searches for the lowest price, one step each, and they showed {fresh_total} more rows.'
        if early:
            pool = [i for n in range(1, len(self.items) + 1) for i in self.item_ids(n)] if self.category_key() else list(self.seen_order)
            rows = sorted({(self.price_of(i), i) for i in pool if i not in self.full and self.buyable(i) and self.price_of(i) is not None})
            cap = self.budget if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) else None
            rows = [(p, i) for p, i in rows if cap is None or p <= cap]
            if rows:
                said += ' Unopened buyable rows, cheapest first (open the cheapest that may fit before any dearer one): ' + ', '.join((f'{i} ({p:g})' for p, i in rows[:CHEAPER_SHOWN + 2]))
        return said

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
        room = self.hunt_room()
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
            marks = {i: self.row_marks(i, number) for i in pool}
            pool = [i for i in pool if not marks[i][1]]
            pool.sort(key=lambda i: (-marks[i][0], -len(words & _text_words(_dump(self.rows.get(i) or {}))), self.price_of(i) if self.price_of(i) is not None else float('inf')))
            picks += pool[:HUNT_FIT_OPENS]
        read: list[str] = []
        for start in range(0, len(picks), HUNT_FIT_OPENS):
            if self.clock() > HUNT_BY + HUNT_GRACE:
                break
            read += self.fill_facts(picks[start:start + HUNT_FIT_OPENS], keep)
        self.trace.append({'kind': 'hunt_fits', 'pages': paged, 'searches': spent, 'opened': len(read)})
        return read

    def row_marks(self, ident: str, number: int) -> tuple[int, int]:
        """(requirements a search row's own text states, requirements it breaks) for the item. A row shows only part of
        a listing, so a value it does not show breaks nothing; only a figure in the limit's own unit past the limit does
        (a row titled with 512 of some unit under a requirement of at most 256 of that unit). Fitting rows often state a
        limited property in their titles while the rows opened first in store order do not."""
        row = self.rows.get(ident) or {}
        text = ' '.join((str(v) for v in row.values() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))))
        stated = broken = 0
        pointed = {need for _, need in self.refs}
        for owner, need in self.needs_now():
            value = need.partition(':')[2].strip()
            if owner != number or not value or need in pointed or re.search(POINTED, value.lower()):
                continue
            limit = _limit_of(value)
            if limit and not limit[2]:
                continue
            verdict = _value_fit(value, text)
            if verdict is True:
                stated += 1
            elif verdict is False and limit:
                broken += 1
        return (stated, broken)

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
        return f'\n[Tobi] {key} is the page number of this search: the call above showed its last page. Page {low} of the same search:\n' + self.show(seen)

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
        return f'\n[Tobi] Nothing has all those words; the same search with {short!r} shows:\n' + self.show(seen)

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
        cap_price = self.budget if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) else None
        # the keys the planner shows come from the listings read here: the ones that can be bought inside the budget come
        # first, since an unwritten preference is graded on what is bought, not on listings priced out or sold out
        reachable = lambda i: (not self.buyable(i), cap_price is not None and (self.price_of(i) or 0.0) > cap_price)
        lanes = [sorted([i for i in ids if i not in self.full], key=reachable) for ids in ranked]
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
        hunted = self.hunt_cheaper(early=True) if self.brief.get('pick') == 'cheapest' else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "Tobi's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if chain:
            summary += '\n' + chain
        if searched:
            summary += '\n' + searched
        if hunted:
            summary += '\n' + hunted
        if self.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def late_question(self) -> str:
        """Before a checkout, ask the next planned question with a message the order itself does not need.

        The interview leaves out every message the request keeps for a process step, whether the step comes with the
        order or with an ending without a purchase. When Tobi is about to buy, the messages kept for the ending are
        free. Race 13 allowed two messages: the base asked one question in 28 of 180 episodes and left a hidden need
        undisclosed in 12 of them (no episode with an undisclosed need passed). Returns the summary for the loop model
        when the answer adds a requirement, else ''."""
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
        summary = f'Tobi asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
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
                home = self.home_label(number)
                rows.sort(key=lambda i: self.value_rank(i, number, fresh, home, cat_key))
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
        return "[Tobi searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def value_rank(self, ident: str, number: int, fresh: list[str], home: str, cat_key: str | None) -> tuple[Any, ...]:
        """Order of the rows a value search opens: the item's own category label first, then rows whose own text already
        holds every value an answer gave (a row's words are what the page will show), then, under a lowest-price rule,
        the cheaper first; the store's order breaks ties. Opening the first rows in the store's order, which follows no
        price, leaves the cheapest stated row unread under a lowest-price rule."""
        row = self.rows.get(ident) or {}
        text = ' '.join((str(v) for v in row.values() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))))
        holds = bool(fresh) and all((_value_fit(v, text) is True for v in fresh))
        price = self.price_of(ident) if self.brief.get('pick') == 'cheapest' else None
        own = bool(cat_key and home) and _words(str(row.get(cat_key) or '')) != home
        return (own, -holds, price if price is not None else float('inf'))

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
        # Tobi: every message but the last asks one item, in planned order, and the last asks every item still unasked.
        # Measured over one race's baskets, a need named in the first item's part of a several-item message was stated
        # 19 times in 20 and one in a later item's part 2 times in 8, while a need on an item no message named was never
        # found; so items share only the last message, where nothing later is left to spend.
        return [[n] for n in numbers[:free - 1]] + [numbers[free - 1:]]

    def ask_group(self, group: list[int], told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """One message naming a few planned properties of each item of the group, the same list rules applying: the reply is
        about the first tracked property named, so each item's best keys lead its part and the first item's first label is the
        only one a reply disclosing nothing settles."""
        early = self.list_now() < self.list_cap()
        share = max(1, self.list_now() // len(group)) if early else max(MIX_LABELS, self.list_cap() // len(group))
        lead = max(1, self.list_now() - (len(group) - 1)) if early else share
        parts: list[tuple[int, list[str]]] = []
        for number in group:
            left = [label for label in self.lanes.get(number) or [] if _words(label) not in self.lane_used.setdefault(number, set()) and (not self.told_about(number, label))]
            if left:
                # the first item's part is the one a reply mostly answers: it takes the list's labels but one per later item
                parts.append((number, left[:lead if not parts else (1 if early else share)]))
        share = max((len(labels) for _, labels in parts), default=1)
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
        if not answered and any((_count(x) == 1 for x in _items(heard.get('lost')))):
            for number, labels in parts:
                self.lane_used.setdefault(number, set()).update((_words(label) for label in labels))
            self.trace.append({'kind': 'mixed_lost', 'parts': parts})
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
        lost = not new and any((_count(x) == 1 for x in _items(heard.get('lost'))))
        self.settle(number, labels, shown, new, crowded, lost)

    def list_cap(self) -> int:
        return LIST_SIZE if len(self.items) == 1 else LIST_MIXED

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

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool, lost: bool=False) -> None:
        """Mark what an answer showed about the labels its question named.

        One label: it is asked; a reply asking for a narrower question gets the base's follow-up (one product of an
        item name joining several, the property's head word, or example values). A list: the shopper answers about
        the first property of a message it tracks, and its reply to a message naming none of its properties is the
        same as to a property it does not mind. So a disclosed need settles its label and every label before it, and
        the labels after it lead the item's next list; a reply that discloses nothing settles only the first label
        (a need there would have been answered) and the others come next in reverse order, so whatever the shopper
        does not mind no longer stands in front of them; after a second such reply in a row the labels not yet named lead, since the
        order would otherwise return to the first list's. A reply asking for one question at a time ends lists for
        the episode and the list's first label is asked alone.

        Tobi: a reply that only says it did not follow the question ("lost") retires every label the question named, and
        the item's next list starts on labels not named yet. Measured over two races, such a reply followed a question
        naming an unwritten preference 11 times in 156 and one naming none 145 times, so re-asking the list's later
        labels spends a message on properties the shopper holds no preference on."""
        used = self.lane_used.setdefault(number, set())
        if lost and not crowded and len(labels) > 1:
            used.update((_words(label) for label in labels))
            named = {_words(label) for label in labels}
            self.lanes[number] = [label for label in self.lanes.get(number) or [] if _words(label) not in named]
            self.lane_misses[number] = self.lane_misses.get(number, 0) + 1
            self.trace.append({'kind': 'list_lost', 'item': number, 'labels': labels})
            return
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
            lanes[number] = self.askable_order(lanes[number], own, hinted[number])
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def askable_order(self, lane: list[str], own: dict[str, float], hinted: set[str]) -> list[str]:
        """The lane with the labels a shopper holds no product preference on moved behind the rest, their order kept.

        Those are stock, delivery, warranty, seller and price words, yes/no flags, purchase-option lists and identifiers
        (UNASKABLE), and, once listings were read, labels naming no key of the item's listings that no sentence of the
        story points at. A first question spent on a raw key fragment, a yes/no flag or a stock word names no property a
        shopper holds a preference on, and with two messages that is half the episode's questions."""
        plain = lambda text: re.sub('(?<=[a-z]{3})s\\b', '', _words(text))
        keys = {plain(k) for k in own}

        def doubtful(label: str) -> bool:
            if re.search(UNASKABLE, _words(label)):
                return True
            if not keys or _words(label) in hinted or plain(label) in keys:
                return False
            return not any((_names(_stem(label), _stem(k)) or _names(_stem(k), _stem(label)) for k in own))
        sure = [x for x in lane if not doubtful(x)]
        return sure + [x for x in lane if x not in sure]

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
                prop = str(entry['property'])
                self.items[number - 1].free.append(prop)
                # the shopper said it does not mind this property: no later list names it again, whichever question it
                # answered (a reply to one property can free another)
                self.lane_used.setdefault(number, set()).add(_words(prop))
                if number in self.lanes:
                    self.lanes[number] = [label for label in self.lanes[number] if not _names(_spelled(label), _spelled(prop))]
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
                return (False, ('Before going on, Tobi did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2):
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Tobi: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
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
        return (True, 'Before going on, Tobi did the steps the request asks for:\n' + '\n'.join(said) if said else '')

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
            if drops or any((c.startswith('budget now') for c in changed)):
                # the fitting set changed, so the lowest price may sit where no hunt looked: one more round (a requirement
                # dropped at the first checkout widens the set, and the cheapest fitting row is then often a new one)
                self.hunts = min(self.hunts, HUNT_ROUNDS - 1)
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
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by Tobi's record of the cart calls (listing: units): " + _dump(self.cart_lines)

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

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else '(no record seen)'

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

    def fill_short_cart(self) -> bool:
        """Before a forced order: add, for each item with no line, the best opened listing whose attributes state every
        requirement in force (code_passing: limits and exclusions read, not only words), the cheapest first under a
        lowest-price rule. A one-item episode with an empty cart counts its item as missing. False when an item stays
        uncovered, so the episode ends by refusing rather than with a short order."""
        cart = self.roles.get('cart') or ''
        bound = {n for pair in self.bound for n in pair}
        empty = not [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        numbers = self.missing_items() or ([1] if len(self.items) == 1 and empty else [])
        for number in numbers:
            item = self.items[number - 1]
            fits = [i for i in self.code_passing(number, tuple(self.cart_lines), stated=True) if self.price_of(i) is not None]
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
        if not lead:
            self.better = better
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def item_of(self, ident: str) -> int:
        """The requested item a listing belongs to: the item whose label its row carries, else the only item, else 0."""
        if len(self.items) == 1:
            return 1
        for number in range(1, len(self.items) + 1):
            if ident in self.item_ids(number):
                return number
        return 0

    def line_verdicts(self, ident: str, number: int) -> list[tuple[str, bool | None]]:
        """(need, verdict) for each requirement of the item in force, read from the listing's own attributes (_need_fit).
        A requirement whose value another listing gives is read with the value its chain's last page gave (trail), as an
        exclusion when the requirement wants anything but, unlike or different from that value; unresolved, it stays
        unknown. Without it a chain can be followed to its end and a listing chosen before the chain still be bought."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        pointed = {need for _, need in self.refs}
        out = []
        for owner, need in self.needs_now():
            if owner != number:
                continue
            if need not in pointed:
                out.append((need, _need_fit(block, need, self.items[number - 1].name)))
                continue
            got = self.ref_values.get(need) or ''
            prop, _, value = need.partition(':')
            low = value.lower()
            negated = re.search('\\b(?:anything but|anything except|other than|unlike|different from|differs? from|not the same|except)\\b', low)
            equal = re.search('\\b(?:matching|match|same as|same|the one on|as on|as stated|what the page)\\b', low)
            compared = re.search('\\b(?:than|within|near|close to|similar|longer|shorter|bigger|smaller|heavier|lighter|wider|narrower|more|less)\\b', low)
            if not got or got.startswith('as stated on ') or compared or not (negated or equal):
                out.append((need, None))
            else:
                out.append((need, _need_fit(block, f"{prop}: {'not ' if negated else ''}{got}", self.items[number - 1].name)))
        return out

    def code_passing(self, number: int, skip: tuple[str, ...]=(), stated: bool=False) -> list[str]:
        """Opened listings of the item that can be bought inside the budget and whose attributes break no requirement in
        force (with `stated`, that state every one), the ones stating the most first, then by the pick rule (lowest
        price first unless the rule asks for the most or least of something, which the model judges)."""
        if not 1 <= number <= len(self.items):
            return []
        cap = self.budget if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) else None
        qty = 1 if self.brief.get('budget_scope') == 'each' else max(1, self.items[number - 1].qty or 1)
        pool = self.item_ids(number) if len(self.items) > 1 or self.category_key() else list(self.seen_order)
        ranked = []
        for ident in pool:
            if ident in skip or ident not in self.full or not self.buyable(ident):
                continue
            price = self.price_of(ident)
            if cap is not None and price is not None and price * qty > cap:
                continue
            marks = [v for _, v in self.line_verdicts(ident, number)]
            if any((v is False for v in marks)) or (stated and (not all((v is True for v in marks)))):
                continue
            cost = price if price is not None and self.brief.get('pick') not in ('most', 'least') else 0.0
            ranked.append((-sum((v is True for v in marks)), cost, ident))
        return [ident for _, _, ident in sorted(ranked)]

    def fault_hold(self) -> str:
        """Before the model's cart check: a cart line whose own attributes break a requirement in force (a value its
        property's attributes do not hold, a figure past a limit in the same unit, an excluded value present) holds the
        order once per cart, naming the attribute and the opened listings that break nothing. Measured over two races, the
        base's orders failed about a third of the time even with every unwritten preference heard, among them limits
        read wrong (a figure above an at-most limit bought while a fitting listing was open) and look-alike values."""
        if self.fault_held >= 2:
            return ''
        mark = self.cart_mark() + f'#{self.version}'
        if mark in self.fault_marks:
            return ''
        self.fault_marks.add(mark)
        faults = []
        for ident in [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]:
            number = self.item_of(ident)
            if not number or ident not in self.full:
                continue
            block = _fact_block(self.full.get(ident) or {}) or {}
            for need, verdict in self.line_verdicts(ident, number):
                if verdict is False:
                    prop = _spelled(need.partition(':')[0])
                    said = [f'{k}={v}' for k, v in block.items() if _names(prop, _spelled(k)) or _names(_spelled(k), prop)][:2]
                    faults.append((ident, number, need, said))
        if not faults:
            return ''
        self.fault_held += 1
        parts = []
        for ident, number, need, said in faults[:3]:
            others = self.code_passing(number, (ident,))[:3]
            parts.append(f"{ident} does not meet '{need}' (its page says {', '.join(said)})" + (f'; opened listings that break nothing: {", ".join(others)}' if others else ''))
        self.trace.append({'kind': 'fault_hold', 'faults': [(i, n) for i, _, n, _ in faults[:4]]})
        return 'Tobi read the cart against the requirements in force: ' + '; '.join(parts) + '. Replace that line, unless its page shows the requirement met under another spelling or name; then check out again.'

    def stated_hold(self) -> str:
        """Once, before an ending without a purchase: when opened listings state every requirement in force for each
        requested item (and fit the budget, and can be bought), name them; the ending goes through after that."""
        if self.stated_held or not self.needs_now():
            return ''
        picks = {n: self.code_passing(n, stated=True)[:2] for n in range(1, len(self.items) + 1)}
        if not all(picks.values()) or self.left() < len(self.items) + 1:
            return ''
        self.stated_held = True
        self.leads = [i for ids in picks.values() for i in ids]
        self.lead_items = {str(n): list(ids) for n, ids in picks.items()}
        self.trace.append({'kind': 'stated_hold', 'picks': picks})
        named = '; '.join((f'item {n}: {", ".join(ids)}' if len(self.items) > 1 else ', '.join(ids) for n, ids in picks.items()))
        return ('Tobi read the opened pages against the requirements in force, and these state every one of them, inside the budget and in stock: '
                + named + '. Put the best one per item in the cart (respecting the pick rule) and check out; end without a purchase only if its page clearly breaks a requirement.')

    def tight_swap(self) -> str:
        """With a pick rule and at most 4 steps left, put the review's better listing in the cart in code (one step) when
        its page was read, it can be bought, it fits the budget and beats the line on the rule, and a check of it alone
        passes; the order then goes out on the loop's next call. After a requirement dropped at checkout, the lowest-price
        search can find and open the fitting cheaper listing two steps before the end, and the order then goes out with
        the dearer line because no step is left for the loop to swap it."""
        rule = self.brief.get('pick')
        if self.swapped or rule != 'cheapest' or len(self.items) != 1 or self.left() > 4 or self.left() < 2:
            return ''
        lines = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if len(lines) != 1:
            return ''
        old = lines[0]
        cap = self.budget if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) else None
        qty = max(1, self.items[0].qty or 1)
        units = 1 if self.brief.get('budget_scope') == 'each' else qty
        for pick in self.better + [i for i in self.code_passing(1, (old,), stated=True) if i not in self.better]:
            price, was = self.price_of(pick), self.price_of(old)
            if pick == old or pick not in self.full or not self.buyable(pick) or price is None:
                continue
            if cap is not None and price * units > cap:
                continue
            if rule == 'cheapest' and was is not None and price >= was:
                continue
            self.check_cart([pick], SPEND_HARD)
            if not self.verdicts.get(self.check_key([pick])):
                continue
            calls = self.cart_calls({pick: qty}, [old])
            name = self.roles.get('cart')
            if not calls or not name or len(calls) + 1 > self.left():
                return ''
            self.swapped = True
            done = []
            for args in calls:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(name), 'args', faults)
                if faults:
                    return ''
                reply = self.act(name, args)
                if reply is None or self.finished:
                    return ''
                self.note_cart(name, args)
                self.observe(name, reply.get('observation'))
                done.append(_dump(args))
            self.trace.append({'kind': 'tight_swap', 'out': old, 'in': pick, 'calls': done})
            return (f'The review found {pick} better on the pick rule than {old}, and a check of {pick} alone passed; with '
                    + f'{self.left()} steps left Tobi swapped the cart line itself ({"; ".join(done)}). Check out now.')
        return ''

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
        self.trails, self.trail_room = 0, EARLY_TRAILS
        try:
            text, _ = self.trail(HUNT_BY)
        finally:
            (self.trails, self.trail_blocks), self.trail_room = kept, None
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
            text += '[Tobi opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
            text += self.resolved_fits()
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (text, hold)

    def resolved_fits(self) -> str:
        """After a chain gave its value: per item with such a requirement, the opened listings whose attributes now state
        every requirement in force, or, with none, those that break none of them, so the order goes to one of them and
        not to a listing chosen before the chain was read (a listing opened before a chain of several pages can
        otherwise be bought once the chain ends)."""
        lines = []
        for number in sorted({n for n, need in self.refs if need in self.ref_values}):
            sure = self.code_passing(number, stated=True)[:3]
            ids = sure or self.code_passing(number)[:3]
            if ids:
                head = f'item {number}: ' if len(self.items) > 1 else ''
                lines.append(head + ', '.join(ids) + (' state every requirement in force' if sure else ' break no requirement in force (some not shown)'))
        return ('[opened listings that fit the values the chain gave] ' + '; '.join(lines) + '\n') if lines else ''

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
        not_names = self.not_maker_keys()
        known: list[str] = []
        named = {w for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and k not in not_names and v.strip().lower() not in NO_NAME for w in re.findall('[a-z0-9]{3,}', v.lower())}
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
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]

        def described(ident: str) -> int:
            # a pointer names its listing by a few of its values ("its kind, with <property> <value> and ..."): the row
            # holding most of them as whole words is the likeliest target, so it is opened first and a chain of pages
            # spends fewer steps on the shop's other listings
            text = _plain_text(_dump(self.rows.get(ident) or {}))
            return sum((_has_word(v, text) for v in values))
        found = of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order))))
        return sorted(found, key=lambda i: -described(i))[:2]

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def read_line_pages(self) -> None:
        """Right before the order: each cart line whose own page was never read through the store's page view gets that
        page read now, one step each, while the steps left still hold the order (and its due message). An order of a
        listing whose page nobody opened rarely counts, and the hunts often cart the last pick with only a step or two
        left, too few for the re-read before checkout."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        unread = [i for i in self.cart_ids() if i not in self.gone and (not self.page_read(i))]
        done: list[str] = []
        for ident in unread:
            if self.left() - 1 < self.tail() or self.rushed() or self.finished:
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
            done.append(ident)
        if unread:
            self.trace.append({'kind': 'line_pages', 'unread': unread, 'read': done, 'left': self.left()})

    def reread_offers(self) -> str:
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
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ''
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        text = f"[Tobi] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
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
        number = self.item_of(lost[0]) if lost else 0
        if number:
            marks = {i: [v for _, v in self.line_verdicts(i, number)] for i in ranked if i in self.full}
            ranked = [i for i in ranked if not any((v is False for v in marks.get(i, [])))]
            ranked.sort(key=lambda i: -sum((v is True for v in marks.get(i, []))))
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits, 'kept': ranked[:4]})
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

    def audit_call(self, rows_first: bool=False) -> dict[str, Any] | None:
        """The review of the listings seen, for an ending without a purchase.

        Opened records fill at most AUDIT_OPENED of its AUDIT_RECORDS places, so the rows a search showed and nobody
        opened, inside the budget and buyable, cheapest first, keep the rest (fitting
        rows can sit in search results unopened while opened records fill the review). With rows_first (few steps
        left) those rows come first, since a row is all that one step to open can still turn into an order."""
        cap = self.budget if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) else None
        newest = [i for i in reversed(self.seen_order) if not self.sold_out(i)]
        opened = [i for i in newest if i in self.full]
        fresh = [i for i in newest if i not in self.full and self.buyable(i) and (cap is None or (self.price_of(i) or 0.0) <= cap)]
        fresh.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
        rows = [('ROW', i) for i in fresh]
        kept = [('OPENED', i) for i in opened]
        lead, rest = (rows, kept) if rows_first else (kept, rows)
        picked = lead[:AUDIT_OPENED] + rest[:AUDIT_RECORDS - AUDIT_OPENED]
        picked += [x for x in lead[AUDIT_OPENED:] + rest[AUDIT_RECORDS - AUDIT_OPENED:]][:max(0, AUDIT_RECORDS - len(picked))]
        records = []
        for mark, ident in picked[:AUDIT_RECORDS]:
            if mark == 'OPENED':
                records.append('OPENED ' + _dump(_clip(self.full.get(ident), 200))[:1200])
            else:
                records.append('ROW ' + _dump(_clip(self.rows.get(ident), 120))[:500])
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

    def short_review(self) -> str:
        """With 3 to 5 steps left the full pre-decline search no longer fits, but a row the store showed can still be
        opened, carted and ordered. The review reads the never-opened rows first and names at most one; the reply
        asks for that listing's page and the order, and the ending goes through when its page breaks a requirement.
        Without it, an episode can end without a purchase three steps before its limit, nothing carted, while fitting rows
        sat unopened in its own searches."""
        if self.short_reviewed or self.left() < 3 or not self.judges():
            return ''
        if len(self.items) > 1 and len(self.missing_items()) != 1:
            return ''
        self.short_reviewed = True
        got = self.audit_call(rows_first=True)
        named = self.audit_fits(got)
        if not named:
            return ''
        lead = next(iter(named.values()))[0]
        self.leads = [lead]
        self.refuse_blocks = 2
        self.trace.append({'kind': 'short_review', 'lead': lead})
        shown = 'its page is already read' if lead in self.full else 'open its page first (one step)'
        return (f'A review of the rows seen found {lead}, which appears to meet every requirement in force ({shown}). '
                + f'Put it in the cart at the requested quantity and check out within the {self.left()} steps left; end without a purchase only if its page clearly breaks a requirement.')

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
        broken = [i for i in lines if any((v is False for _, v in self.line_verdicts(i, self.item_of(i) or 1)))]
        if broken:
            self.trace.append({'kind': 'complete_refused', 'broken': broken})
            return ''
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
        return 'Not run (Tobi acted instead): the independent check confirmed ' + self.confirmed + ', so Tobi set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

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

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Tobi opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask') and self.message_args('x' * 200) is not None:
                out.append((cid, "Not run (no step used): Tobi's interviewer handles every message to the shopper."))
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
                trimmed = f"\n[Tobi] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
                trimmed += f"\n[Tobi] This is a new search, so Tobi asked for its first page ({_dump(asked_for)} was the call you sent)."
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
                    trimmed += f"\n[Tobi] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
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
                            out.append((cid, "Not run yet (no step used). Tobi set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()))
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
                    faulted = self.fault_hold()
                    if faulted:
                        out.append((cid, 'Not run yet (no step used). ' + faulted + '\nCart now: ' + self.cart_text()))
                        continue
                    verdict = self.check_cart()
                    if verdict:
                        swapped = self.tight_swap()
                        if swapped:
                            out.append((cid, 'Not run yet. ' + swapped + '\nCart now: ' + self.cart_text()))
                            continue
                        seen = "Tobi read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        seen = (hunted + ' ' + self.below_cart() + '\n' if hunted else '') + seen
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Tobi re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.left() <= 5:
                    review = self.stated_hold() or self.short_review()
                    if review:
                        out.append((cid, 'Not run yet (no step used). ' + review))
                        continue
                else:
                    pointed = self.refuse_trail()
                    if pointed:
                        out.append((cid, 'Not run yet (no step used). ' + pointed))
                        continue
                    self.hunt_fits()
                    stated = self.stated_hold()
                    if stated:
                        out.append((cid, 'Not run yet (no step used). ' + stated))
                        continue
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
                self.read_line_pages()
            if ending and self.left() >= 2:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Tobi] ' + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Tobi] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
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
            text = self.show(observation) + paged + trimmed
            if name not in plain and (not ending):
                text += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, args, observation)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                    text += '\n[Tobi] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
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
                    text += "\n[Tobi re-read the cart's lines after the refusal]\n" + seen
            self.note_sold = False
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if name == self.roles.get('cart') and self.bound and not self.finished:
                misfit = self.fit_faults()
                if misfit:
                    text += '\n[Tobi] These lines do not work together yet: ' + '; '.join(misfit)
            if self.queued:
                text += "\n[notes seen on Tobi's own calls] " + ' | '.join(self.queued)
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
        whole = True
        if not holding and buy and self.left() >= 2 and self.needs_now() and self.fill_short_cart():
            holding = bool(self.cart_lines)
            self.trace.append({'kind': 'empty_cart_filled', 'lines': dict(self.cart_lines)})
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
            intro += '\n\nTobi has searched for each item and read some of its listings (one step each):\n' + moved + '\nJudge these against the requirements and search further only for what is still missing.\n\n' + self.status()
        else:
            intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                self.chat.append({'role': 'user', 'content': '[Tobi] Tobi tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
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
    try:
        episode = Tobi(problem_data)
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
